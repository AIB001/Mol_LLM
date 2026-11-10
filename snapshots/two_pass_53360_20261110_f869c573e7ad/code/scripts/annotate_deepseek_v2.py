#!/usr/bin/env python3
"""Bounded asynchronous v2 annotation with atomic batch journals and resumable exports.

No request or credential access occurs during --dry-run. STOP (or SIGINT/SIGTERM)
stops dispatch and retries, drains existing HTTP requests, then exports all saved rows.
SQLite uses synchronous=NORMAL: atomic JSON batches are the authoritative recovery
journal, so a stale SQLite WAL after a crash is rebuilt without repeating good rows.
Atomic renames protect process-crash consistency; no per-batch fsync is performed.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import signal
import sqlite3
import time
from typing import Any

import aiohttp
from rdkit import Chem, RDLogger, rdBase

from annotate_deepseek import FEATURE_NAMES, ROOT, get_balance, property_context, read_items, read_keys
from annotation_checks_v2 import validate_annotation_v2
from download_chembl import utc_now
from structural_facts_v2 import compact_facts_v2, structural_facts_v2

SYSTEM_PATH = ROOT / "prompts/molecular_annotation_v2_system.txt"
DEFAULT_OUTPUT = ROOT / "results/chembl_100000_deepseek_v4_pro_v2"
LEGACY_OUTPUT = ROOT / "results/chembl_100000_deepseek_v4_pro"
VALID_STATUSES = {"ok", "needs_review"}
VALIDATION_SCOPE = ("automated schema and numerical-property checks only; "
                    "model-reported chemistry, not independently verified")
SCHEMA_VERSION = "molecular-annotation-v2-async-1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Replace a complete UTF-8 JSON file; the previous file survives a failed write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def style_plan(chembl_id: str, smiles: str) -> dict:
    choices = {
        "opening_focus": ["core_scaffold", "linker_connectivity", "substitution_pattern",
                          "heteroatom_environments", "ring_topology"],
        "traversal": ["core_to_periphery", "periphery_to_core", "along_main_linker", "functional_regions"],
        "sentence_cadence": ["compact_varied", "mixed_length", "connected_narrative"],
        "closing_focus": ["stereochemistry_and_charge", "qualitative_polarity", "rigidity_and_flexibility"],
    }
    entropy = bytes.fromhex(digest(["style-plan-v2", chembl_id, smiles]))
    return {name: values[entropy[i] % len(values)] for i, (name, values) in enumerate(choices.items())}


def prepare_items(items: list[dict], readable_facts: bool = False) -> list[dict]:
    result = []
    for original in items:
        item = dict(original)
        mol = Chem.MolFromSmiles(item["smiles"])
        if mol is None:
            raise ValueError(f"Invalid input SMILES for {item['chembl_id']}")
        item["facts"] = structural_facts_v2(mol, original_smiles=item["smiles"])
        item["facts_sha256"] = digest(item["facts"])
        if readable_facts:
            from readable_facts_v3 import readable_facts_v3
            item["readable_structural_facts"] = readable_facts_v3(item["facts"])
            item["readable_facts_sha256"] = digest(item["readable_structural_facts"])
        item["context"] = property_context(item["properties"])
        item["style_plan"] = style_plan(item["chembl_id"], item["smiles"])
        result.append(item)
    return result


def messages_for(items: list[dict], system: str, repair_errors: dict | None = None) -> list[dict]:
    payload = {"molecules": [{"chembl_id": item["chembl_id"], "smiles": item["smiles"],
                              "verified_structural_facts": compact_facts_v2(item["facts"]),
                              "qualitative_properties": item["context"],
                              "style_plan": item["style_plan"]} for item in items]}
    for output, item in zip(payload["molecules"], items):
        if "readable_structural_facts" in item:
            output["readable_structural_facts"] = item["readable_structural_facts"]
    if repair_errors:
        payload["repair_validation_failures"] = repair_errors
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def validate_batch(result: Any, items: list[dict]) -> tuple[dict, dict]:
    """Map by identity, never response order; reject ambiguous batch identity globally."""
    ids = [item["chembl_id"] for item in items]
    if not isinstance(result, dict) or set(result) != {"annotations"} or not isinstance(result["annotations"], list):
        return {}, {cid: ["invalid_batch_schema"] for cid in ids}
    rows = result["annotations"]
    if any(not isinstance(row, dict) or not isinstance(row.get("chembl_id"), str) for row in rows):
        return {}, {cid: ["invalid_annotation_object"] for cid in ids}
    returned = [row["chembl_id"] for row in rows]
    if len(returned) != len(ids) or len(set(returned)) != len(returned) or set(returned) != set(ids):
        return {}, {cid: ["missing_duplicate_or_unexpected_identifier"] for cid in ids}
    mapped = {row["chembl_id"]: row for row in rows}
    errors = {item["chembl_id"]: validate_annotation_v2(mapped[item["chembl_id"]], item["chembl_id"], item["facts"])
              for item in items}
    return mapped, {cid: issues for cid, issues in errors.items() if issues}


def record_is_accepted(row: dict | None) -> bool:
    return bool(row and row.get("status") in VALID_STATUSES and row.get("description"))


def base_record(item: dict, batch_id: int, round_id: int, config: dict) -> dict:
    return {"chembl_id": item["chembl_id"], "smiles": item["smiles"],
            "properties": item["properties"], "property_vector": [item["properties"][name] for name in FEATURE_NAMES],
            "structural_facts": item["facts"], "facts_sha256": item["facts_sha256"],
            "readable_structural_facts": item.get("readable_structural_facts"),
            "readable_facts_sha256": item.get("readable_facts_sha256"),
            "readable_fact_code_sha256": config.get("readable_fact_code_sha256"),
            "style_plan": item["style_plan"], "description": None, "status": "pending", "issues": [],
            "batch_id": batch_id, "round_id": round_id, "model": config["model"], "model_requested": config["model"],
            "account": None, "prompt_sha256": config["prompt_sha256"],
            "fact_code_sha256": config["fact_code_sha256"], "run_fingerprint": config["run_fingerprint"],
            "chembl_release": item["chembl_release"], "rdkit_version": config["rdkit_version"],
            "representative_activity_id": item["representative_activity_id"], "validation_scope": VALIDATION_SCOPE}


class BatchStore:
    """Only the single writer executor uses this object after construction."""

    def __init__(self, directory: Path, fingerprint: str):
        self.directory = directory
        self.fingerprint = fingerprint
        self.db = sqlite3.connect(directory / "annotations.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS annotations(
                chembl_id TEXT PRIMARY KEY,batch_id INTEGER,round_id INTEGER,status TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS annotations_by_round ON annotations(round_id,batch_id);
            CREATE TABLE IF NOT EXISTS batches(
                batch_id INTEGER PRIMARY KEY,round_id INTEGER,status TEXT,revision INTEGER,usage TEXT,
                run_fingerprint TEXT,updated_at_utc TEXT);
        """)

    def _index(self, result: dict) -> None:
        if result.get("run_fingerprint") != self.fingerprint:
            raise RuntimeError("Batch fingerprint mismatch; choose a fresh output directory")
        self.db.execute("INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?,?,?)",
                        (result["batch_id"], result["round_id"], result["status"], result["revision"],
                         json.dumps(result.get("usage", {})), self.fingerprint, result["updated_at_utc"]))
        for row in result["records"]:
            self.db.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?,?,?)",
                            (row["chembl_id"], row["batch_id"], row["round_id"], row["status"],
                             json.dumps(row, ensure_ascii=False, separators=(",", ":"))))

    def recover(self) -> set[int]:
        """Reindex every atomic file, including revisions newer than an existing DB row."""
        seen = set()
        with self.db:
            self.db.execute("DELETE FROM annotations")
            self.db.execute("DELETE FROM batches")
            for path in sorted((self.directory / "batches").glob("batch_*.json")):
                result = json.loads(path.read_text(encoding="utf-8"))
                self._index(result)
                seen.add(result["batch_id"])
        return seen

    def load(self, batch_id: int) -> dict | None:
        path = self.directory / "batches" / f"batch_{batch_id:06d}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def persist(self, result: dict) -> None:
        if result.get("run_fingerprint") != self.fingerprint:
            raise RuntimeError("Refusing batch from a different run fingerprint")
        # File is authoritative and must be published before the derived SQLite index.
        atomic_json(self.directory / "batches" / f"batch_{result['batch_id']:06d}.json", result)
        with self.db:
            self._index(result)

    def export(self, round_id: int | None = None) -> dict:
        basename = f"round_{round_id:04d}" if round_id is not None else "annotations"
        sql = "SELECT payload FROM annotations"
        params = ()
        if round_id is not None:
            sql += " WHERE round_id=?"
            params = (round_id,)
        sql += " ORDER BY batch_id,chembl_id"
        fields = ["chembl_id", "smiles", *FEATURE_NAMES, "property_vector", "description", "status", "issues",
                  "batch_id", "round_id", "model", "model_requested", "account", "prompt_sha256", "facts_sha256",
                  "fact_code_sha256", "readable_facts_sha256", "readable_fact_code_sha256", "run_fingerprint", "chembl_release", "rdkit_version", "validation_scope"]
        csv_target = self.directory / f"{basename}.csv"
        json_target = self.directory / f"{basename}.jsonl"
        csv_tmp = csv_target.with_suffix(".csv.tmp")
        json_tmp = json_target.with_suffix(".jsonl.tmp")
        counts = Counter()
        with csv_tmp.open("w", newline="", encoding="utf-8") as cf, json_tmp.open("w", encoding="utf-8") as jf:
            writer = csv.DictWriter(cf, fieldnames=fields)
            writer.writeheader()
            for (payload,) in self.db.execute(sql, params):
                row = json.loads(payload)
                counts[row["status"]] += 1
                # Full graph evidence remains in canonical batch JSON. Exports retain its hash.
                exported = {key: value for key, value in row.items()
                            if key not in {"structural_facts", "readable_structural_facts"}}
                jf.write(json.dumps(exported, ensure_ascii=False, separators=(",", ":")) + "\n")
                flat = {key: row.get(key) for key in fields}
                flat.update(row["properties"])
                for key in ("property_vector", "issues"):
                    flat[key] = json.dumps(flat[key], ensure_ascii=False)
                writer.writerow(flat)
        os.replace(csv_tmp, csv_target)
        os.replace(json_tmp, json_target)
        return dict(counts)

    def progress(self) -> dict:
        statuses = dict(self.db.execute("SELECT status,COUNT(*) FROM annotations GROUP BY status"))
        usage = Counter()
        for (raw,) in self.db.execute("SELECT usage FROM batches"):
            usage.update(json.loads(raw))
        return {"status_counts": statuses, "stored_molecules": sum(statuses.values()), "usage": dict(usage),
                "finished_batches": self.db.execute("SELECT COUNT(*) FROM batches WHERE status='completed'").fetchone()[0],
                "saved_batches": self.db.execute("SELECT COUNT(*) FROM batches").fetchone()[0]}

    def close(self) -> None:
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.db.close()


class AsyncWriter:
    def __init__(self, store: BatchStore):
        self.store = store
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="annotation-writer")

    async def call(self, name: str, *args):
        return await asyncio.get_running_loop().run_in_executor(self.pool, getattr(self.store, name), *args)

    async def json(self, name: str, data: dict):
        return await asyncio.get_running_loop().run_in_executor(self.pool, atomic_json, self.store.directory / name, data)

    async def close(self):
        await self.call("close")
        self.pool.shutdown(wait=True)


class RequestClient:
    def __init__(self, session, keys, config, args, stop):
        self.session, self.keys, self.config, self.args, self.stop = session, keys, config, args, stop
        self.semaphores = [asyncio.Semaphore(args.max_inflight) for _ in keys]
        self.loads = [0 for _ in keys]
        self.disabled = [False for _ in keys]
        self.disabled_reasons = [None for _ in keys]
        self.stop_reason = None

    async def request(self, messages: list[dict], batch_id: int, pass_number: int) -> tuple[dict | None, list[dict], str | None]:
        payload = {"model": self.config["model"], "messages": messages,
                   "response_format": {"type": "json_object"}, "thinking": {"type": self.config["thinking"]},
                   "max_tokens": self.config["max_tokens"], "stream": False}
        if self.config["thinking"] == "enabled":
            payload["reasoning_effort"] = "low"
        else:
            payload["temperature"] = self.config["temperature"]
        logs = []
        error = "request_not_started"
        for retry in range(self.args.http_retries):
            if self.stop.is_set():
                return None, logs, "paused_before_request"
            active = [i for i, disabled in enumerate(self.disabled) if not disabled]
            if not active:
                self.stop_reason = "all_accounts_unavailable"
                self.stop.set()
                return None, logs, self.stop_reason
            preferred = (batch_id - 1 + retry) % len(self.keys)
            account = min(active, key=lambda i: (self.loads[i], i != preferred))
            semaphore = self.semaphores[account]
            self.loads[account] += 1
            started = time.monotonic()
            event = {"pass_number": pass_number, "http_attempt": retry + 1, "account": f"account_{account + 1}",
                     "started_at_utc": utc_now(), "request_sha256": digest(payload), "requested_ids": [], "usage": {}}
            try:
                async with semaphore:
                    if self.stop.is_set():
                        return None, logs, "paused_before_request"
                    if self.disabled[account]:
                        continue
                    # Headers and exception messages are never written to journals or logs.
                    async with self.session.post("https://api.deepseek.com/chat/completions", json=payload,
                                                 headers={"Authorization": "Bearer " + self.keys[account]}) as response:
                        event["http_status"] = response.status
                        if response.status in {401, 402, 403}:
                            self.disabled[account] = True
                            self.disabled_reasons[account] = f"http_{response.status}"
                            error = f"account_http_{response.status}"
                        elif response.status == 429 or response.status >= 500:
                            error = f"retryable_http_{response.status}"
                        elif response.status != 200:
                            error = f"configuration_http_{response.status}"
                            self.stop_reason = error
                            self.stop.set()
                        else:
                            try:
                                data = await response.json(content_type=None)
                            except (ValueError, UnicodeError):
                                data = None
                                error = "invalid_http_json"
                            if isinstance(data, dict):
                                usage = data.get("usage")
                                event["usage"] = {name: value for name, value in (usage if isinstance(usage, dict) else {}).items()
                                                  if isinstance(value, int) and not isinstance(value, bool)}
                                event["model_returned"] = data.get("model")
                                event["response_id"] = data.get("id")
                                event["duration_seconds"] = round(time.monotonic() - started, 3)
                                event["error"] = None
                                logs.append(event)
                                return data, logs, None
                            error = "invalid_http_json"
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                error = f"transport_{type(exc).__name__}"
            finally:
                self.loads[account] -= 1
            event["duration_seconds"] = round(time.monotonic() - started, 3)
            event["error"] = error
            logs.append(event)
            if self.stop.is_set():
                break
            if retry + 1 < self.args.http_retries and not self.disabled[account]:
                try:
                    await asyncio.wait_for(self.stop.wait(), timeout=min(2 ** retry, 30) + random.random())
                except asyncio.TimeoutError:
                    pass
        return None, logs, error


async def annotate_batch(batch_id: int, round_id: int, items: list[dict], previous: dict | None,
                         client: RequestClient, writer: AsyncWriter, config: dict, args) -> dict:
    old_records = {row["chembl_id"]: row for row in (previous or {}).get("records", [])}
    records = {}
    for item in items:
        old = old_records.get(item["chembl_id"])
        if old and (old["smiles"] != item["smiles"] or old.get("facts_sha256") != item["facts_sha256"]):
            raise RuntimeError("Saved molecule evidence differs from current input")
        records[item["chembl_id"]] = old if record_is_accepted(old) else base_record(item, batch_id, round_id, config)
    attempt_log = list((previous or {}).get("attempt_log", []))
    revision = (previous or {}).get("revision", 0)
    errors = {}
    passes_this_run = 0

    async def checkpoint(status=None):
        nonlocal revision
        revision += 1
        usage = Counter()
        by_account = {}
        for attempt in attempt_log:
            usage.update(attempt.get("usage", {}))
            by_account.setdefault(attempt["account"], Counter()).update(attempt.get("usage", {}))
        accepted = all(record_is_accepted(records[item["chembl_id"]]) for item in items)
        result = {"batch_id": batch_id, "round_id": round_id, "run_fingerprint": config["run_fingerprint"],
                  "revision": revision, "status": "completed" if accepted else (status or "partial"),
                  "chembl_ids": [item["chembl_id"] for item in items], "records": list(records.values()),
                  "usage": dict(usage), "usage_by_account": {key: dict(value) for key, value in by_account.items()},
                  "attempt_log": attempt_log, "attempts": len(attempt_log), "passes_this_run": passes_this_run,
                  "updated_at_utc": utc_now(), "model_requested": config["model"]}
        await writer.call("persist", result)
        return result

    for repair_pass in range(args.repair_passes):
        pending = [item for item in items if not record_is_accepted(records[item["chembl_id"]])]
        if not pending:
            return await checkpoint()
        if client.stop.is_set():
            break
        messages = messages_for(pending, config["system_prompt"], errors or None)
        pass_number = max([entry.get("pass_number", 0) for entry in attempt_log], default=0) + 1
        response, request_logs, request_error = await client.request(messages, batch_id, pass_number)
        passes_this_run += 1
        for entry in request_logs:
            entry["requested_ids"] = [item["chembl_id"] for item in pending]
        attempt_log.extend(request_logs)
        if request_error:
            for item in pending:
                records[item["chembl_id"]].update(status="request_failed", issues=[request_error])
            return await checkpoint("paused" if client.stop.is_set() else "request_failed")
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else {}
        finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        generated = None
        try:
            generated = json.loads(content or "")
            mapped, errors = validate_batch(generated, pending)
        except (ValueError, TypeError, KeyError):
            mapped, errors = {}, {item["chembl_id"]: ["invalid_or_empty_json"] for item in pending}
        if finish_reason != "stop":
            for item in pending:
                errors.setdefault(item["chembl_id"], []).append("incomplete_generation")
        attempt_log[-1].update(finish_reason=finish_reason, response_content=content,
                               validation_errors=errors)
        for item in pending:
            cid = item["chembl_id"]
            row = records[cid]
            row.update(model=response.get("model") or config["model"], account=attempt_log[-1]["account"])
            if cid in errors:
                row.update(description=None, status="validation_failed", issues=errors[cid])
            else:
                annotation = mapped[cid]
                row.update(description=annotation["description"],
                           status=annotation["status"],
                           model_reported_status=annotation["status"], issues=annotation["issues"])
        result = await checkpoint()
        if result["status"] == "completed":
            return result
    return await checkpoint("paused" if client.stop.is_set() else "validation_failed")


def batch_sequence(items, batch_size=5, round_size=10000):
    iterator = iter(items)
    for batch_id in itertools.count(1):
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch_id, (batch_id - 1) * batch_size // round_size + 1, batch


def make_config(args) -> dict:
    config = {"schema_version": SCHEMA_VERSION, "model": args.model, "batch_size": args.batch_size,
              "round_size": args.round_size, "temperature": args.temperature, "thinking": args.thinking,
              "max_tokens": args.max_tokens, "prompt_sha256": file_hash(args.prompt),
              "input_sha256": file_hash(args.input), "properties_sha256": file_hash(args.properties),
              "rdkit_version": rdBase.rdkitVersion, "feature_names": FEATURE_NAMES,
              "fact_code_sha256": file_hash(ROOT / "scripts/structural_facts_v2.py"),
              "validation_code_sha256": file_hash(ROOT / "scripts/annotation_checks_v2.py"),
              "feature_code_sha256": file_hash(ROOT / "scripts/molecular_features.py"),
              "shared_runner_sha256": file_hash(ROOT / "scripts/annotate_deepseek.py"),
              "runner_code_sha256": file_hash(Path(__file__)), "system_prompt": args.prompt.read_text(encoding="utf-8"),
              "validation_scope": VALIDATION_SCOPE, "readable_facts": args.readable_facts,
              "readable_fact_code_sha256": (file_hash(ROOT / "scripts/readable_facts_v3.py")
                                             if args.readable_facts else None)}
    config["run_fingerprint"] = digest(config)
    return config


def ensure_config(directory: Path, config: dict):
    path = directory / "run_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise RuntimeError("Input, prompt, code, model or generation settings changed; use a new output directory")
    elif (directory / "annotations.sqlite").exists() or any((directory / "batches").glob("batch_*.json")):
        raise RuntimeError("Existing outputs have no v2 run config; choose a fresh output directory")
    else:
        atomic_json(path, config)


async def run_async(args, config, items, keys, store, existing):
    writer = AsyncWriter(store)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    stop_reason = None

    def request_stop(reason):
        nonlocal stop_reason
        if not stop.is_set():
            stop_reason = reason
            stop.set()
            print(f"Pausing: {reason}; draining in-flight requests and saving results", flush=True)

    registered = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, request_stop, signal.Signals(signum).name)
            registered.append(signum)
        except (NotImplementedError, RuntimeError):
            pass

    async def watch_stopfile():
        while not stop.is_set():
            if args.stop_file.exists():
                request_stop("stop_file")
                break
            await asyncio.sleep(0.5)

    watcher = asyncio.create_task(watch_stopfile())
    max_tasks = min(args.max_inflight * len(keys), max(1, (len(items) + 4) // 5))
    round_expected = Counter(round_id for _, round_id, _ in batch_sequence(items, args.batch_size, args.round_size))
    round_finished = Counter()
    exported_rounds = set()
    counts = Counter()
    tasks = set()
    iterator = iter(batch_sequence(items, args.batch_size, args.round_size))
    exhausted = False
    started = time.monotonic()
    returned = 0
    runtime = {"transport": "aiohttp", "processes": 1, "writer_threads": 1, "accounts": len(keys),
               "max_inflight_per_account": args.max_inflight, "max_inflight_total": max_tasks,
               "selected_molecules": len(items), "batch_size": args.batch_size,
               "repair_passes": args.repair_passes, "http_retries": args.http_retries,
               "started_at_utc": utc_now(), "pid": os.getpid(), "stop_file": str(args.stop_file),
               "sqlite_synchronous": "NORMAL", "recovery_source": "atomic batch JSON files"}
    await writer.json("runtime.json", runtime)
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30, sock_read=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=max_tasks, limit_per_host=max_tasks, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = RequestClient(session, keys, config, args, stop)

        async def report_progress(final=False):
            progress = await writer.call("progress")
            progress.update(updated_at_utc=utc_now(), target_molecules=len(items),
                            elapsed_seconds=round(time.monotonic() - started, 1), in_flight_batches=len(tasks),
                            returned_batches_this_run=returned, result_counts_this_run=dict(counts),
                            account_disabled=client.disabled, account_disabled_reasons=client.disabled_reasons,
                            paused=stop.is_set(), stop_reason=stop_reason or client.stop_reason)
            await writer.json("progress.json", progress)
            if final:
                progress.update(all_rows_stored=progress["stored_molecules"] >= len(items),
                                all_descriptions_present=(progress["stored_molecules"] >= len(items)
                                    and not any(n for status, n in progress["status_counts"].items() if status not in VALID_STATUSES)),
                                chemical_verification=VALIDATION_SCOPE)
                await writer.json("summary.json", progress)
            return progress

        async def maybe_export_round(round_id):
            if round_finished[round_id] == round_expected[round_id] and round_id not in exported_rounds:
                await writer.call("export", round_id)
                await writer.call("export")
                exported_rounds.add(round_id)
                print(f"Round {round_id}: exported {round_finished[round_id]} batches", flush=True)

        async def consume(done):
            nonlocal returned
            for task in done:
                result = task.result()
                returned += 1
                counts[result["status"]] += 1
                round_finished[result["round_id"]] += 1
                await maybe_export_round(result["round_id"])
            if done and (returned % 20 < len(done) or not tasks):
                progress = await report_progress()
                print(f"Returned={returned}; in_flight={len(tasks)}; statuses={progress['status_counts']}", flush=True)

        print(f"Async accounts={len(keys)}; total in-flight<={max_tasks}; per account<={args.max_inflight}; "
              f"molecules={len(items)}; batch_size=5", flush=True)
        try:
            while tasks or (not exhausted and not stop.is_set()):
                while not exhausted and not stop.is_set() and len(tasks) < max_tasks:
                    try:
                        batch_id, round_id, raw_items = next(iterator)
                    except StopIteration:
                        exhausted = True
                        break
                    previous = await writer.call("load", batch_id) if batch_id in existing else None
                    previous_rows = {row["chembl_id"]: row for row in (previous or {}).get("records", [])}
                    if all(record_is_accepted(previous_rows.get(item["chembl_id"])) for item in raw_items):
                        for item in raw_items:
                            if previous_rows[item["chembl_id"]]["smiles"] != item["smiles"]:
                                raise RuntimeError("Saved identity/SMILES mismatch")
                        round_finished[round_id] += 1
                        await maybe_export_round(round_id)
                        continue
                    prepared = await asyncio.to_thread(prepare_items, raw_items, config.get("readable_facts", False))
                    if stop.is_set():
                        break
                    task = asyncio.create_task(annotate_batch(batch_id, round_id, prepared, previous, client,
                                                              writer, config, args))
                    tasks.add(task)
                    # Consume finished work while filling; dispatch never waits for a whole round.
                    ready = {pending for pending in tasks if pending.done()}
                    if ready:
                        tasks.difference_update(ready)
                        await consume(ready)
                if tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = set(pending)
                    await consume(done)
                elif exhausted or stop.is_set():
                    break
        except BaseException:
            request_stop("runner_error")
            if tasks:
                completed = await asyncio.gather(*tasks, return_exceptions=True)
                for result in completed:
                    if isinstance(result, dict):
                        round_finished[result["round_id"]] += 1
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            for signum in registered:
                loop.remove_signal_handler(signum)
            for round_id in sorted(round_finished):
                if round_id not in exported_rounds:
                    await writer.call("export", round_id)
            await writer.call("export")
            summary = await report_progress(final=True)
            await writer.close()
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", type=Path, default=SYSTEM_PATH)
    parser.add_argument("--readable-facts", action="store_true",
                        help="Add deterministic readable v3 facts alongside the compact graph; fingerprint both")
    parser.add_argument("--max-inflight", "--max-inflight-per-account", dest="max_inflight", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--round-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=100, help="Molecules from the beginning; 0 selects all")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--thinking", choices=["disabled", "enabled"], default="disabled")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--repair-passes", type=int, default=3, help="Maximum generation/repair passes per invalid row per run")
    parser.add_argument("--http-retries", type=int, default=5)
    parser.add_argument("--request-timeout", type=int, default=240)
    parser.add_argument("--stop-file", type=Path, default=None, help="Default: OUTPUT_DIR/STOP")
    parser.add_argument("--resume", action="store_true", help="Remove the STOP file and resume; accepted rows are always reused")
    parser.add_argument("--dry-run", action="store_true", help="Write request preview without reading keys or making API calls")
    parser.add_argument("--skip-balance-check", action="store_true", help="Skip read-only starting account balance requests")
    args = parser.parse_args(argv)
    if args.batch_size != 5 or args.round_size < 5 or args.round_size % 5:
        parser.error("Use batch size 5 and a positive round size divisible by 5")
    if not 1 <= args.max_inflight <= 500:
        parser.error("Per-account max-inflight must be in [1, 500]")
    if args.limit < 0 or min(args.max_tokens, args.repair_passes, args.http_retries, args.request_timeout) < 1:
        parser.error("Sizes must be positive; --limit 0 selects all")
    if not 0 <= args.temperature <= 2:
        parser.error("Temperature must be between 0 and 2")
    if args.output_dir.resolve() == LEGACY_OUTPUT.resolve() or LEGACY_OUTPUT.resolve() in args.output_dir.resolve().parents:
        parser.error("The v1 result directory is protected; use a separate v2 output directory")
    args.stop_file = args.stop_file or args.output_dir / "STOP"
    return args


def main(argv=None):
    args = parse_args(argv)
    RDLogger.DisableLog("rdApp.warning")
    config = make_config(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ensure_config(args.output_dir, config)
    selected = read_items(args.input, args.properties)
    items = list(itertools.islice(selected, args.limit)) if args.limit else list(selected)
    if not items:
        raise RuntimeError("No molecules selected")
    ids = [item["chembl_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate input identifiers")
    if args.dry_run:
        target = args.output_dir / "dry_run_requests.jsonl"
        temporary = target.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for batch_id, round_id, raw in batch_sequence(items, args.batch_size, args.round_size):
                prepared = prepare_items(raw, args.readable_facts)
                handle.write(json.dumps({"status": "not_submitted", "batch_id": batch_id, "round_id": round_id,
                                         "run_fingerprint": config["run_fingerprint"],
                                         "messages": messages_for(prepared, config["system_prompt"])}, ensure_ascii=False) + "\n")
        os.replace(temporary, target)
        print(f"Dry run: {len(items)} molecules; no API calls or key access; {target}", flush=True)
        return
    if args.stop_file.exists():
        if args.resume:
            args.stop_file.unlink()
        else:
            print(f"Paused: {args.stop_file} exists. Use --resume to clear it and continue.", flush=True)
            return
    # Credentials are read only after all offline validation and the dry-run branch.
    keys = read_keys()
    if not args.skip_balance_check:
        balances = {f"account_{index + 1}": get_balance(key) for index, key in enumerate(keys)}
        if not (args.output_dir / "balance_before.json").exists():
            atomic_json(args.output_dir / "balance_before.json", balances)
        atomic_json(args.output_dir / "balance_latest.json", balances)
    store = BatchStore(args.output_dir, config["run_fingerprint"])
    existing = store.recover()
    asyncio.run(run_async(args, config, items, keys, store, existing))


if __name__ == "__main__":
    main()
