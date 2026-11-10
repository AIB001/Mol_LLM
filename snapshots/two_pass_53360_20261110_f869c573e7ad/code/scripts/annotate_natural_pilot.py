#!/usr/bin/env python3
"""Generate free English molecular descriptions for the first 10,000 source rows.

Five source SMILES per request; no controlled text or style selections. All model
responses and billed repair attempts are retained. Accepted format does not mean
independently verified chemistry. STOP/SIGINT drains active requests. A 25-row
pilot and 10,000-row continuation share configuration and reuse accepted rows.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import csv
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import signal
import time

import aiohttp
from annotate_deepseek import FEATURE_NAMES, ROOT, get_balance, read_items, read_keys
from annotate_deepseek_v2 import AsyncWriter as BaseWriter, BatchStore as BaseStore, digest, file_hash
from download_chembl import utc_now
import natural_description_checks
from format_controlled_json import pretty_array

DEFAULT_OUTPUT = ROOT / "results/chembl_10000_natural_pilot"
SYSTEM_PATH = ROOT / "prompts/natural_description_system_v4.txt"
SCHEMA_VERSION = "natural_molecular_annotation/1"
GENERATION_MODE = "deepseek_free_natural_language_from_smiles"
VALIDATION_SCOPE = "format and automated textual screening only; chemistry not independently verified"
BATCH_SIZE, ROUND_SIZE, DATASET_SIZE = 5, 1000, 10000
REQUIRED_IDS = {"CHEMBL397577", "CHEMBL1956235"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_items(args):
    source = read_items(args.molecules, args.properties)
    seen = set()
    limit = args.limit or DATASET_SIZE
    opener = gzip.open if args.input.suffix == ".gz" else open
    with opener(args.input, "rt", encoding="utf-8") as handle:
        for index, (line, original) in enumerate(itertools.zip_longest(handle, source), 1):
            if index > limit:
                return
            if line is None or original is None:
                raise RuntimeError("Structural facts and molecule/property source lengths differ")
            row = json.loads(line)
            cid = row["chembl_id"]
            if cid != original["chembl_id"] or row["smiles"] != original["smiles"] or cid in seen:
                raise RuntimeError(f"Source identity/order/SMILES mismatch at row {index}")
            seen.add(cid)
            if digest(row["structural_facts"]) != row["facts_sha256"]:
                raise RuntimeError(f"Structural fact hash mismatch at row {index}")
            if row.get("graph_validation", {}).get("status") != "exact_graph_roundtrip":
                raise RuntimeError(f"Source graph validation absent at row {index}")
            yield {**original, **row, "source_index": index,
                   "property_vector": [original["properties"][name] for name in FEATURE_NAMES]}


def batches(items):
    iterator = iter(items)
    for bid in itertools.count(1):
        batch = list(itertools.islice(iterator, BATCH_SIZE))
        if not batch:
            return
        yield bid, (bid - 1) // (ROUND_SIZE // BATCH_SIZE) + 1, batch


def next_batch(iterator):
    return next(iterator, None)


def messages_for(items, config, errors=None):
    payload = {"molecules": [{"key": str(i), "chembl_id": item["chembl_id"], "smiles": item["smiles"]}
                             for i, item in enumerate(items, 1)]}
    if errors:
        payload["previous_format_errors"] = errors
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def validate_batch(value, items):
    if not isinstance(value, dict) or set(value) != {"annotations"} or not isinstance(value["annotations"], list):
        return {}, {str(i): ["invalid_batch_schema"] for i in range(1, len(items) + 1)}
    rows = value["annotations"]
    if any(not isinstance(row, dict) or set(row) != {"key", "description"}
           or not isinstance(row["key"], str) for row in rows):
        return {}, {str(i): ["invalid_annotation_fields"] for i in range(1, len(items) + 1)}
    keys = [row["key"] for row in rows]
    expected = {str(i) for i in range(1, len(items) + 1)}
    if len(keys) != len(items) or len(set(keys)) != len(items) or set(keys) != expected:
        return {}, {key: ["missing_duplicate_or_unexpected_key"] for key in expected}
    mapped = {row["key"]: row for row in rows}
    checks = {str(i): natural_description_checks.check_description(mapped[str(i)]["description"], item["structural_facts"])
              for i, item in enumerate(items, 1)}
    return {key: {**mapped[key], "checks": checks[key]} for key in mapped}, {
        key: result["format_errors"] for key, result in checks.items() if result["format_errors"]}


def base_record(item, bid, rid, config):
    return {"schema_version": SCHEMA_VERSION, "chembl_id": item["chembl_id"], "smiles": item["smiles"],
            "properties": item["properties"], "property_vector": item["property_vector"],
            "description": None, "description_sha256": None, "word_count": None, "word_count_in_target": None, "editorial_hints": [], "status": "pending",
            "issues": [], "review_reasons": [], "chemistry_verified": False,
            "chemical_review_status": "not_independently_reviewed", "validation_scope": VALIDATION_SCOPE,
            "generation_mode": GENERATION_MODE, "facts_sha256": item["facts_sha256"],
            "source_index": item["source_index"], "structural_facts_source": {
                "path": config["input_path"], "sha256": config["input_sha256"], "line": item["source_index"]},
            "graph_validation": item["graph_validation"], "batch_id": bid, "round_id": rid,
            "model": None, "model_requested": config["model"], "account": None,
            "prompt_sha256": config["prompt_sha256"], "run_fingerprint": config["run_fingerprint"],
            "chembl_release": item.get("chembl_release"), "representative_activity_id": item.get("representative_activity_id"),
            "rdkit_version": item["structural_facts"]["rdkit_version"]}


def row_accepted(row):
    return bool(row and row.get("status") == "accepted_format" and row.get("description"))


def check_saved(previous, items):
    if previous is None:
        return False
    records = previous.get("records", [])
    if len(records) != len(items):
        raise RuntimeError("Saved batch size differs from the source; use limits divisible by five")
    for row, item in zip(records, items):
        if any(row.get(key) != item.get(key) for key in ("chembl_id", "smiles", "facts_sha256", "source_index")):
            raise RuntimeError("Saved batch differs from source identity or evidence")
    return all(row_accepted(row) for row in records)


class BatchStore(BaseStore):
    def persist(self, result):
        if result.get("run_fingerprint") != self.fingerprint:
            raise RuntimeError("Refusing batch from a different run")
        atomic_json(self.directory / "batches" / f"batch_{result['batch_id']:06d}.json", result)
        with self.db:
            self._index(result)

    def export(self, round_id=None):
        basename = f"round_{round_id:04d}" if round_id is not None else "annotations"
        query, parameters = "SELECT payload FROM annotations", ()
        if round_id is not None:
            query += " WHERE round_id=?"
            parameters = (round_id,)
        query += " ORDER BY batch_id,CAST(json_extract(payload,'$.source_index') AS INTEGER)"
        fields = ["chembl_id", "smiles", *FEATURE_NAMES, "property_vector", "description", "word_count", "word_count_in_target", "editorial_hints", "status",
                  "issues", "review_reasons", "chemical_review_status", "chemistry_verified", "validation_scope",
                  "generation_mode", "source_index", "batch_id", "round_id", "model", "model_requested", "account",
                  "prompt_sha256", "facts_sha256", "description_sha256", "run_fingerprint", "schema_version",
                  "structural_facts_source", "response_id", "selected_pass_number", "selected_attempt_index",
                  "chembl_release", "rdkit_version"]
        cp, jp = self.directory / f"{basename}.csv", self.directory / f"{basename}.jsonl"
        ct, jt = cp.with_suffix(".csv.tmp"), jp.with_suffix(".jsonl.tmp")
        counts = Counter()
        with ct.open("w", newline="", encoding="utf-8", buffering=1024 * 1024) as cf, \
             jt.open("w", encoding="utf-8", buffering=1024 * 1024) as jf:
            writer = csv.DictWriter(cf, fieldnames=fields)
            writer.writeheader()
            for (payload,) in self.db.execute(query, parameters):
                row = json.loads(payload)
                counts[row["status"]] += 1
                jf.write(payload + "\n")
                flat = {key: row.get(key) for key in fields}
                flat.update(row["properties"])
                for key, value in flat.items():
                    if isinstance(value, (dict, list)):
                        flat[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                writer.writerow(flat)
        os.replace(ct, cp)
        os.replace(jt, jp)
        return dict(counts)


class AsyncWriter(BaseWriter):
    async def json(self, name, data):
        return await asyncio.get_running_loop().run_in_executor(self.pool, atomic_json, self.store.directory / name, data)

    async def pretty(self):
        return await asyncio.get_running_loop().run_in_executor(self.pool, pretty_array, self.store.directory / "annotations.jsonl", self.store.directory / "annotations.json")


class RequestClient:
    """Bounded account-aware transport with shared cooldown after rate limiting."""
    def __init__(self, session, keys, config, args, stop):
        self.session, self.keys, self.config, self.args, self.stop = session, keys, config, args, stop
        self.caps = args.account_caps[:len(keys)]
        if len(self.caps) != len(keys):
            raise ValueError("Provide an account cap for every configured account")
        self.semaphores = [asyncio.Semaphore(cap) for cap in self.caps]
        self.loads, self.cooldowns, self.rate_failures = [0] * len(keys), [0.0] * len(keys), [0] * len(keys)
        self.disabled, self.disabled_reasons = [False] * len(keys), [None] * len(keys)
        self.stop_reason = None

    async def pause(self, seconds):
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def choose_account(self, preferred):
        while not self.stop.is_set():
            active = [i for i in range(len(self.keys)) if not self.disabled[i]]
            if not active:
                self.stop_reason = "all_accounts_unavailable"
                self.stop.set()
                return None
            now = time.monotonic()
            ready = [i for i in active if self.cooldowns[i] <= now]
            if ready:
                return min(ready, key=lambda i: (self.loads[i] / self.caps[i], i != preferred))
            await self.pause(max(0.01, min(self.cooldowns[i] for i in active) - now))
        return None

    async def request(self, messages, batch_id, pass_number):
        payload = {"model": self.config["model"], "messages": messages, "response_format": {"type": "json_object"},
                   "thinking": {"type": self.config["thinking"]}, "max_tokens": self.config["max_tokens"], "stream": False}
        if self.config["thinking"] == "enabled":
            payload["reasoning_effort"] = "low"
        else:
            payload["temperature"] = self.config["temperature"]
        logs, error = [], "paused_before_request"
        for retry in range(self.args.http_retries):
            account = await self.choose_account((batch_id - 1 + retry) % len(self.keys))
            if account is None:
                break
            self.loads[account] += 1
            started = time.monotonic()
            event = {"pass_number": pass_number, "http_attempt": retry + 1, "account": f"account_{account + 1}",
                     "started_at_utc": utc_now(), "request_sha256": digest(payload), "usage": {}}
            try:
                async with self.semaphores[account]:
                    if self.stop.is_set():
                        return None, logs, "paused_before_request"
                    if self.disabled[account]:
                        continue
                    await self.pause(max(0, self.cooldowns[account] - time.monotonic()))
                    if self.stop.is_set():
                        return None, logs, "paused_before_request"
                    async with self.session.post("https://api.deepseek.com/chat/completions", json=payload,
                            headers={"Authorization": "Bearer " + self.keys[account]}) as response:
                        event["http_status"] = response.status
                        if response.status in {401, 402, 403}:
                            self.disabled[account], self.disabled_reasons[account] = True, f"http_{response.status}"
                            error = f"account_http_{response.status}"
                        elif response.status == 429 or response.status >= 500:
                            error = f"retryable_http_{response.status}"
                            self.rate_failures[account] += 1
                            delay = min(2 ** min(self.rate_failures[account], 5), 30) + random.random()
                            self.cooldowns[account] = max(self.cooldowns[account], time.monotonic() + delay)
                            event["account_cooldown_seconds"] = round(delay, 3)
                        elif response.status != 200:
                            error = f"configuration_http_{response.status}"
                            self.stop_reason = error
                            self.stop.set()
                        else:
                            try:
                                data = await response.json(content_type=None)
                            except (ValueError, UnicodeError):
                                data = None
                            if isinstance(data, dict):
                                raw_usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                                event.update(usage={k: v for k, v in raw_usage.items() if type(v) is int},
                                             model_returned=data.get("model"), response_id=data.get("id"),
                                             duration_seconds=round(time.monotonic() - started, 3), error=None)
                                self.rate_failures[account] = max(0, self.rate_failures[account] - 1)
                                logs.append(event)
                                return data, logs, None
                            error = "invalid_http_json"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error = f"transport_{type(exc).__name__}"
            finally:
                self.loads[account] -= 1
            event.update(error=error, duration_seconds=round(time.monotonic() - started, 3))
            logs.append(event)
            if all(self.disabled):
                self.stop_reason = "all_accounts_unavailable"
                self.stop.set()
            if self.stop.is_set():
                break
            if retry + 1 < self.args.http_retries and not self.disabled[account]:
                await self.pause(min(2 ** retry, 30) + random.random())
        return None, logs, error if not self.stop_reason else self.stop_reason


async def annotate_batch(bid, rid, items, previous, client, writer, config, args):
    if check_saved(previous, items):
        return previous
    old = (previous or {}).get("records", [])
    records = [old[i] if i < len(old) and row_accepted(old[i]) else base_record(item, bid, rid, config)
               for i, item in enumerate(items)]
    attempts, revision = list((previous or {}).get("attempt_log", [])), (previous or {}).get("revision", 0)
    errors = {}

    async def checkpoint(status):
        nonlocal revision
        revision += 1
        usage, accounts = Counter(), {}
        for event in attempts:
            usage.update(event.get("usage", {}))
            accounts.setdefault(event["account"], Counter()).update(event.get("usage", {}))
        result = {"schema_version": SCHEMA_VERSION, "generation_mode": GENERATION_MODE,
                  "batch_id": bid, "round_id": rid, "status": "completed" if all(map(row_accepted, records)) else status,
                  "revision": revision, "run_fingerprint": config["run_fingerprint"], "records": records,
                  "chembl_ids": [item["chembl_id"] for item in items], "attempt_log": attempts,
                  "usage": dict(usage), "usage_by_account": {key: dict(value) for key, value in accounts.items()},
                  "model_requested": config["model"], "updated_at_utc": utc_now()}
        await writer.call("persist", result)
        return result

    for _ in range(args.repair_passes):
        if client.stop.is_set():
            return await checkpoint("paused")
        pending_positions = [i for i, row in enumerate(records) if not row_accepted(row)]
        pending_items = [items[i] for i in pending_positions]
        repair_errors = {str(local): records[position]["issues"]
                         for local, position in enumerate(pending_positions, 1) if records[position]["issues"]}
        pass_number = max((event.get("pass_number", 0) for event in attempts), default=0) + 1
        response, logs, error = await client.request(messages_for(pending_items, config, repair_errors), bid, pass_number)
        for event in logs:
            event["requested_keys"] = [str(i) for i in range(1, len(pending_items) + 1)]
            event["requested_ids"] = [item["chembl_id"] for item in pending_items]
        attempts.extend(logs)
        if error:
            for row in records:
                if not row_accepted(row):
                    row["issues"] = [error]
            return await checkpoint("paused" if client.stop.is_set() else "request_failed")
        if not logs:
            raise RuntimeError("Successful response has no request provenance")
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        try:
            mapped, errors = validate_batch(json.loads(content or ""), pending_items)
        except (ValueError, TypeError):
            mapped, errors = {}, {str(i): ["invalid_or_empty_json"] for i in range(1, len(pending_items) + 1)}
        if choice.get("finish_reason") != "stop":
            for i in range(1, len(pending_items) + 1):
                errors.setdefault(str(i), []).append("incomplete_generation")
        attempts[-1].update(response=response, response_content=content, validation_errors=errors)
        for i, position in enumerate(pending_positions, 1):
            row = records[position]
            if str(i) in errors:
                row["issues"] = errors[str(i)]
                continue
            result = mapped[str(i)]
            checks, text = result["checks"], result["description"]
            reasons = checks.get("review_reasons", [])
            row.update(description=text, description_sha256=hashlib.sha256(text.encode()).hexdigest(),
                       word_count=checks["word_count"], word_count_in_target=checks["word_count_in_target"],
                       editorial_hints=checks.get("editorial_hints", []), status="accepted_format", issues=[], review_reasons=reasons,
                       chemical_review_status="needs_review" if reasons else "not_independently_reviewed",
                       chemistry_verified=False, automated_checks=checks, model=response.get("model"),
                       account=attempts[-1]["account"], response_id=response.get("id"),
                       selected_pass_number=pass_number, selected_attempt_index=len(attempts) - 1)
        result = await checkpoint("validation_failed")
        if result["status"] == "completed":
            return result
    return await checkpoint("validation_failed")


def make_config(args):
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    config = {"schema_version": SCHEMA_VERSION, "generation_mode": GENERATION_MODE,
              "input_path": str(args.input.resolve()), "input_sha256": file_hash(args.input),
              "molecules_sha256": file_hash(args.molecules), "properties_sha256": file_hash(args.properties),
              "prompt_path": str(args.prompt.resolve()), "system_prompt": prompt,
              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "runner_code_sha256": file_hash(Path(__file__)),
              "validation_code_sha256": file_hash(Path(natural_description_checks.__file__)),
              "shared_validation_code_sha256": file_hash(ROOT / "scripts/annotation_checks_v2.py"),
              "shared_runner_code_sha256": file_hash(ROOT / "scripts/annotate_deepseek.py"),
              "shared_store_code_sha256": file_hash(ROOT / "scripts/annotate_deepseek_v2.py"),
              "format_code_sha256": file_hash(ROOT / "scripts/format_controlled_json.py"),
              "model": args.model, "thinking": args.thinking, "reasoning_effort": "low" if args.thinking == "enabled" else None,
              "temperature": args.temperature, "max_tokens": args.max_tokens, "batch_size": BATCH_SIZE,
              "round_size": ROUND_SIZE, "dataset_size": DATASET_SIZE, "feature_names": FEATURE_NAMES,
              "validation_scope": VALIDATION_SCOPE}
    config["run_fingerprint"] = digest(config)
    return config


async def run_async(args, config, items, keys, store, existing):
    writer, stop = AsyncWriter(store), asyncio.Event()
    tasks, registered, exported = set(), [], set()
    finished_rounds, returned = Counter(), 0
    target = args.limit or DATASET_SIZE
    capacity = min(args.max_pending_batches, sum(args.account_caps[:len(keys)]))
    iterator, started, stop_reason = iter(batches(items)), time.monotonic(), None

    def request_stop(reason):
        nonlocal stop_reason
        if not stop.is_set():
            stop_reason = reason
            stop.set()
            print(f"Stopping dispatch: {reason}; draining requests and saving results", flush=True)

    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop, signal.Signals(signum).name)
            registered.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    async def stop_watcher():
        while not stop.is_set():
            if args.stop_file.exists():
                request_stop("stop_file")
                return
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(stop_watcher())
    await writer.json("runtime.json", {"pid": os.getpid(), "started_at_utc": utc_now(), "target_molecules": target,
                        "account_caps": args.account_caps[:len(keys)], "max_pending_batches": capacity,
                        "http_retries": args.http_retries, "repair_passes": args.repair_passes,
                        "stop_file": str(args.stop_file), "sqlite_synchronous": "NORMAL"})
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30, sock_read=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=capacity, limit_per_host=capacity, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = RequestClient(session, keys, config, args, stop)

        async def report(final=False):
            result = await writer.call("progress")
            result.update(target_molecules=target, updated_at_utc=utc_now(), generation_mode=GENERATION_MODE,
                          elapsed_seconds=round(time.monotonic() - started, 1), in_flight_batches=len(tasks),
                          paused=stop.is_set(), stop_reason=stop_reason or client.stop_reason,
                          account_disabled=client.disabled, account_disabled_reasons=client.disabled_reasons,
                          account_caps=client.caps, validation_scope=VALIDATION_SCOPE, chemistry_verified=False)
            result["all_descriptions_present"] = result["status_counts"].get("accepted_format", 0) == target
            await writer.json("summary.json" if final else "progress.json", result)
            print(f"Saved={result['stored_molecules']}/{target}; statuses={result['status_counts']}; active={len(tasks)}", flush=True)
            return result

        async def completed(result):
            nonlocal returned
            returned += 1
            rid = result["round_id"]
            finished_rounds[rid] += 1
            if returned % 20 == 0:
                await report()
            expected = min(ROUND_SIZE // BATCH_SIZE, (target - (rid - 1) * ROUND_SIZE) // BATCH_SIZE)
            if finished_rounds[rid] == expected and rid not in exported:
                await writer.call("export", rid)
                await writer.call("export")
                exported.add(rid)
                print(f"Exported round {rid} and cumulative CSV/JSONL", flush=True)

        exhausted = False
        try:
            while tasks or (not exhausted and not stop.is_set()):
                while len(tasks) < capacity and not exhausted and not stop.is_set():
                    entry = await asyncio.to_thread(next_batch, iterator)
                    if entry is None:
                        exhausted = True
                        break
                    bid, rid, batch = entry
                    old = await writer.call("load", bid) if bid in existing else None
                    if check_saved(old, batch):
                        await completed(old)
                    else:
                        tasks.add(asyncio.create_task(annotate_batch(bid, rid, batch, old, client, writer, config, args)))
                    ready = {task for task in tasks if task.done()}
                    tasks.difference_update(ready)
                    for task in ready:
                        await completed(task.result())
                if tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = set(pending)
                    for task in done:
                        await completed(task.result())
        except BaseException:
            request_stop("runner_error")
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            for signum in registered:
                loop.remove_signal_handler(signum)
            for rid in sorted(finished_rounds):
                if rid not in exported:
                    await writer.call("export", rid)
            await writer.call("export")
            await writer.pretty()
            summary = await report(final=True)
            await writer.close()
    return summary


def account_caps(value):
    try:
        result = tuple(int(part) for part in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated account concurrency caps") from None
    if not result or any(cap < 1 or cap > 500 for cap in result):
        raise argparse.ArgumentTypeError("Each cap must be 1..500")
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_structural_facts_v2.jsonl.gz")
    parser.add_argument("--molecules", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--prompt", type=Path, default=SYSTEM_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=DATASET_SIZE, help="0 means 10000; otherwise a multiple of five up to 10000")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--thinking", choices=["enabled", "disabled"], default="enabled")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--account-caps", type=account_caps, default=(8, 8))
    parser.add_argument("--max-pending-batches", type=int, default=128)
    parser.add_argument("--repair-passes", type=int, default=3)
    parser.add_argument("--http-retries", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=300)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args(argv)
    if args.limit < 0 or args.limit > DATASET_SIZE or args.limit % 5:
        parser.error("limit must be 0 or a multiple of five up to 10000")
    if min(args.max_pending_batches, args.http_retries, args.request_timeout, args.max_tokens) < 1 or not 1 <= args.repair_passes <= 3:
        parser.error("Positive transport limits and 1..3 repair passes are required")
    if not 0 <= args.temperature <= 2:
        parser.error("temperature must be 0..2")
    if args.output_dir.resolve() == (ROOT / "results/chembl_100000_controlled").resolve():
        parser.error("Use the separate natural pilot output directory")
    args.stop_file = args.stop_file or args.output_dir / "STOP"
    return args


def main(argv=None):
    args = parse_args(argv)
    config = make_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Prompt, code, input or model settings changed; use a new output directory")
    elif any((args.output_dir / "batches").glob("batch_*.json")) or (args.output_dir / "annotations.sqlite").exists():
        raise RuntimeError("Existing result files lack a matching configuration")
    else:
        atomic_json(config_path, config)
    with args.molecules.open(newline="", encoding="utf-8") as handle:
        selected = list(itertools.islice(csv.DictReader(handle), args.limit or DATASET_SIZE))
    if len(selected) != (args.limit or DATASET_SIZE) or not REQUIRED_IDS <= {row["chembl_id"] for row in selected}:
        raise RuntimeError("Selected source range is incomplete or lacks the two required example molecules")
    items = source_items(args)
    if args.dry_run:
        target = args.output_dir / "dry_run_requests.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for bid, rid, batch in batches(items):
                handle.write(json.dumps({"batch_id": bid, "round_id": rid, "status": "not_submitted",
                                         "messages": messages_for(batch, config)}, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
        pretty_array(target)
        print(f"Dry run: {len(selected)} molecules; no key access or network; {target}", flush=True)
        return
    if args.stop_file.exists():
        if args.resume:
            args.stop_file.unlink()
        else:
            print(f"STOP exists: {args.stop_file}; use --resume to continue", flush=True)
            return
    keys = read_keys()
    if len(args.account_caps) < len(keys):
        raise RuntimeError("Provide one account cap for each configured key")
    if not args.skip_balance_check:
        balances = {f"account_{i}": get_balance(key) for i, key in enumerate(keys, 1)}
        if not (args.output_dir / "balance_before.json").exists():
            atomic_json(args.output_dir / "balance_before.json", balances)
        atomic_json(args.output_dir / "balance_latest.json", balances)
    store = BatchStore(args.output_dir, config["run_fingerprint"])
    existing = store.recover()
    return asyncio.run(run_async(args, config, items, keys, store, existing))


if __name__ == "__main__":
    main()
