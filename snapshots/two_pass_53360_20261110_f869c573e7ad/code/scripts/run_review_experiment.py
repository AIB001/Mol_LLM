#!/usr/bin/env python3
"""Fixed random-500 paired experiment: first annotation, then immediate review.

Every completed five-molecule first-pass batch immediately enters the second
pass. Both stages retain actual API responses and immutable first-pass texts.
STOP drains active calls; rerunning resumes missing stages without deleting STOP.
No production code, existing trial, prompt, or package is modified.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import signal
import time
from types import SimpleNamespace

import aiohttp
import annotate_natural_grounded as grounded
import review_annotation_module as review

core = grounded.runner
ROOT = core.ROOT
TARGET, POPULATION, SEED = 500, 100000, 20261108
SCHEMA = "review_annotation_experiment/1"
ALLOWED_ROOT = ROOT / "results/prompt_trials"
DEFAULT_OUTPUT = ALLOWED_ROOT / "minimal_review_500"
ORIGINAL_RECORD = core.base_record


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_structural_facts_v2.jsonl.gz")
    parser.add_argument("--molecules", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--first-prompt", type=Path, default=ROOT / "prompts/experiments/natural_local_regions_system.txt")
    parser.add_argument("--review-prompt", type=Path, default=ROOT / "prompts/experiments/natural_minimal_review_system.txt")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=TARGET)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--model", default="deepseek-v4-pro", choices=["deepseek-v4-pro"])
    parser.add_argument("--thinking", default="enabled", choices=["enabled"])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--account-caps", type=core.account_caps, default=(50, 50))
    parser.add_argument("--max-pending-batches", type=int, default=100)
    parser.add_argument("--repair-passes", type=int, default=3)
    parser.add_argument("--review-passes", type=int, default=2)
    parser.add_argument("--http-retries", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args(argv)
    if args.limit != TARGET or args.seed != SEED:
        parser.error(f"This experiment is fixed at {TARGET} molecules with seed {SEED}")
    if args.max_tokens != 32768 or args.temperature != 0.6:
        parser.error("Keep the paired experiment model settings: enabled/low, max_tokens32768, temperature0.6")
    if not 1 <= args.repair_passes <= 3 or not 1 <= args.review_passes <= 3 or min(args.max_pending_batches, args.http_retries, args.request_timeout) < 1:
        parser.error("Positive transport limits and 1..3 validation passes are required")
    args.stop_file, args.prompt = args.output_dir / "STOP", args.first_prompt
    return args


def guard_output(directory):
    directory, root = Path(directory).resolve(), ALLOWED_ROOT.resolve()
    if directory == root or not directory.is_relative_to(root):
        raise ValueError("Experiment must use its own directory below results/prompt_trials")
    for ancestor in [directory, *directory.parents]:
        path = ancestor / "run_config.json"
        if path.exists() and (ancestor != directory or json.loads(path.read_text()).get("schema_version") != SCHEMA):
            raise ValueError("Refusing another experiment or an enclosing run directory")
        if ancestor == root:
            break
    if directory.exists() and not (directory / "run_config.json").exists():
        if any(path.name != ".review_experiment.lock" for path in directory.iterdir()):
            raise ValueError("Nonempty output has no matching experiment configuration")


@contextmanager
def output_lock(directory):
    guard_output(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".review_experiment.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process owns this experiment output") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def base_record(item, bid, rid, config):
    row = ORIGINAL_RECORD(item, bid, rid, config)
    row.update(experiment_index=item["experiment_index"], experiment_selection_sha256=config["selection_sha256"],
               experiment_role=config["experiment_role"])
    return row


@contextmanager
def patched_runner():
    with grounded.patched_runner():
        record, size = core.base_record, core.DATASET_SIZE
        core.base_record, core.DATASET_SIZE = base_record, TARGET
        try:
            yield
        finally:
            core.base_record, core.DATASET_SIZE = record, size


def selected_indices():
    return sorted(random.Random(SEED).sample(range(1, POPULATION + 1), TARGET))


def load_selected(args):
    indices = selected_indices()
    wanted, items = set(indices), {}
    opener = gzip.open if args.input.suffix == ".gz" else open
    original = core.read_items(args.molecules, args.properties)
    count = 0
    with opener(args.input, "rt", encoding="utf-8") as handle:
        for index, (line, source) in enumerate(itertools.zip_longest(handle, original), 1):
            count = index
            if line is None or source is None or index > POPULATION:
                raise ValueError("Original fact/molecule/property source count does not match the 100000 population")
            if index not in wanted:
                continue
            row = json.loads(line)
            if any(row[key] != source[key] for key in ("chembl_id", "smiles")):
                raise ValueError(f"Source identity mismatch at row {index}")
            if core.digest(row["structural_facts"]) != row["facts_sha256"] or row.get("graph_validation", {}).get("status") != "exact_graph_roundtrip":
                raise ValueError(f"Invalid source facts or graph proof at row {index}")
            properties = source["properties"]
            vector = [properties[name] for name in core.FEATURE_NAMES]
            if ("properties" in row and row["properties"] != properties) or ("property_vector" in row and row["property_vector"] != vector):
                raise ValueError(f"Source facts and original 24-property table disagree at row {index}")
            context = review.natural_structure_context(row["structural_facts"])
            items[index] = {**source, **row, "source_index": index, "properties": properties,
                            "property_vector": vector, "structural_context_sha256": core.digest(context)}
    if count != POPULATION or set(items) != wanted or len({item["chembl_id"] for item in items.values()}) != TARGET:
        raise ValueError("Selected original population is incomplete or contains duplicate IDs")
    return [{**items[index], "experiment_index": position} for position, index in enumerate(indices, 1)]


def make_configs(args, items):
    first = grounded.make_config(args)
    first.pop("run_fingerprint")
    selected = {"schema_version": "review_experiment_selection/1", "seed": SEED, "population": POPULATION,
                "count": TARGET, "selection_method": "Python random.Random(seed).sample without replacement; sorted original source indices",
                "input_path": first["input_path"], "input_sha256": first["input_sha256"],
                "molecules_path": str(args.molecules.resolve()), "molecules_sha256": first["molecules_sha256"],
                "properties_path": str(args.properties.resolve()), "properties_sha256": first["properties_sha256"],
                "records": [{key: item[key] for key in ("experiment_index", "source_index", "chembl_id", "smiles",
                    "facts_sha256", "properties", "property_vector", "structural_context_sha256")} for item in items]}
    shared = {"selection_sha256": core.digest(selected), "selection_digest_definition": "canonical SHA256 of the entire selection.json object",
              "experiment_runner_code_sha256": core.file_hash(Path(__file__)),
              "review_module_code_sha256": core.file_hash(Path(review.__file__)),
              "molecules_path": str(args.molecules.resolve()), "properties_path": str(args.properties.resolve()),
              "first_validation_passes_per_invocation": args.repair_passes, "review_validation_passes_per_invocation": args.review_passes,
              "record_order": "experiment_index; source_index is the original 100000-row line"}
    first.update(shared, dataset_size=TARGET, experiment_role="first_pass")
    first["run_fingerprint"] = core.digest(first)
    second = {key: value for key, value in first.items() if key != "run_fingerprint"}
    prompt = args.review_prompt.read_text(encoding="utf-8").strip()
    second.update(schema_version=review.SCHEMA_VERSION, experiment_role="second_pass", generation_mode=review.GENERATION_MODE,
                  prompt_path=str(args.review_prompt.resolve()), system_prompt=prompt,
                  prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  first_pass_run_fingerprint=first["run_fingerprint"],
                  review_validator_code_sha256=shared["review_module_code_sha256"],
                  validation_scope="Explicit format/content restrictions only; no word-count or style repair; chemical screens are flags only",
                  model_input_fields=["molecules[key,chembl_id,smiles,structural_context,properties]", "annotations[key,description]"])
    second["run_fingerprint"] = core.digest(second)
    top = {"schema_version": SCHEMA, "dataset_size": TARGET, "source_population": POPULATION, "selection_seed": SEED,
           **shared, **{key: first[key] for key in ("input_path", "input_sha256", "molecules_sha256", "properties_sha256",
                                                   "model", "thinking", "reasoning_effort", "max_tokens", "temperature", "feature_names")},
           "first_prompt_path": str(args.first_prompt.resolve()), "first_prompt_file_sha256": core.file_hash(args.first_prompt),
           "review_prompt_path": str(args.review_prompt.resolve()), "review_prompt_file_sha256": core.file_hash(args.review_prompt),
           "first_pass_run_fingerprint": first["run_fingerprint"], "second_pass_run_fingerprint": second["run_fingerprint"],
           "control_group": "The exact first-pass output from the same molecule; no separate paid control generation",
           "review_input_policy": "Every review/repair request contains the original accepted first-pass text, never a failed review",
           "chemistry_verified": False}
    top["run_fingerprint"] = core.digest(top)
    return top, first, second, selected


def preserve_config(path, config):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"Configuration changed; use a new experiment directory: {path}")
    else:
        core.atomic_json(path, config)


class ExperimentStore(core.BatchStore):
    def __init__(self, directory, config, items):
        directory.mkdir(parents=True, exist_ok=True)
        self.config, self.items = config, items
        super().__init__(directory, config["run_fingerprint"])

    def validate(self, batch):
        bid = batch.get("batch_id")
        if type(bid) is not int or not 1 <= bid <= TARGET // 5 or batch.get("round_id") != 1 or batch.get("run_fingerprint") != self.fingerprint:
            raise ValueError("Stage journal batch/fingerprint is outside this experiment")
        expected = self.items[(bid - 1) * 5:bid * 5]
        rows = batch.get("records", [])
        if len(rows) != len(expected) or len(rows) != 5:
            raise ValueError("Stage journal does not contain the original five selected records")
        for row, item in zip(rows, expected):
            if any(row.get(key) != item[key] for key in ("chembl_id", "smiles", "source_index", "experiment_index",
                                                        "facts_sha256", "properties", "property_vector")):
                raise ValueError("Stage journal selected identity/property mismatch")
            if row.get("run_fingerprint") != self.fingerprint or row.get("batch_id") != bid or row.get("round_id") != 1:
                raise ValueError("Stage record fingerprint/batch mismatch")
            if core.row_accepted(row) and review.text_hash(row["description"]) != row.get("description_sha256"):
                raise ValueError("Saved accepted description hash mismatch")

    def _index(self, batch):
        self.validate(batch)
        return super()._index(batch)

    def persist(self, batch):
        self.validate(batch)
        return super().persist(batch)

    def recover(self):
        for path in (self.directory / "batches").glob("batch_*.json"):
            value = json.loads(path.read_text(encoding="utf-8"))
            if path.name != f"batch_{value['batch_id']:06d}.json":
                raise ValueError("Stage journal filename mismatch")
            self.validate(value)
        return super().recover()

    def records(self):
        return [json.loads(raw) for (raw,) in self.db.execute("SELECT payload FROM annotations ORDER BY batch_id,CAST(json_extract(payload,'$.experiment_index') AS INTEGER)")]

    def export(self, round_id=None):
        counts = super().export(round_id)
        name = "annotations" if round_id is None else f"round_{round_id:04d}"
        cp = self.directory / f"{name}.csv"
        with cp.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, rows = list(reader.fieldnames), list(reader)
        source = {row["chembl_id"]: row for row in self.records()}
        extra = ["experiment_index", "experiment_role", "experiment_selection_sha256"]
        if self.config["experiment_role"] == "second_pass":
            extra += ["first_description", "first_description_sha256", "first_batch_records_sha256", "text_changed", "text_diff"]
        temporary = cp.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + extra)
            writer.writeheader()
            for row in rows:
                for key in extra:
                    value = source[row["chembl_id"]].get(key)
                    row[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value
                writer.writerow(row)
        os.replace(temporary, cp)
        core.pretty_array(self.directory / f"{name}.jsonl", self.directory / f"{name}.json")
        return counts


def request_payload(messages, config):
    return {"model": config["model"], "messages": messages, "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"}, "max_tokens": config["max_tokens"], "stream": False, "reasoning_effort": "low"}


class RecordingClient(core.RequestClient):
    async def request(self, messages, batch_id, pass_number):
        payload = request_payload(messages, self.config)
        response, logs, error = await super().request(messages, batch_id, pass_number)
        for event in logs:
            if event["request_sha256"] != core.digest(payload):
                raise RuntimeError("Recorded transport payload differs from the transmitted request hash")
            event["request_payload"] = payload
            event["request_stage"] = "first_pass" if messages[0]["content"] == self.config["system_prompt"] else "second_pass"
        return response, logs, error


def export_pairs(directory, selection, first_rows, second_rows):
    first, second = ({row["chembl_id"]: row for row in rows} for rows in (first_rows, second_rows))
    rows, counts = [], Counter()
    for source in selection["records"]:
        cid = source["chembl_id"]
        a, b = first.get(cid), second.get(cid)
        accepted_a, accepted_b = core.row_accepted(a), core.row_accepted(b)
        if accepted_b and not accepted_a:
            raise ValueError("Reviewed output exists without its accepted first pass")
        status = "complete_pair" if accepted_a and accepted_b else "awaiting_second_pass" if accepted_a else "first_pass_not_accepted"
        counts[status] += 1
        changed = a["description"] != b["description"] if status == "complete_pair" else None
        if changed is not None:
            counts["changed" if changed else "unchanged"] += 1
        rows.append({**source, "schema_version": "review_experiment_pair/1", "pair_status": status,
                     "first_pass": a, "second_pass": b, "first_description": a.get("description") if a else None,
                     "second_description": b.get("description") if b else None,
                     "first_description_sha256": a.get("description_sha256") if a else None,
                     "second_description_sha256": b.get("description_sha256") if b else None,
                     "text_changed": changed, "text_diff": b.get("text_diff") if accepted_b else None, "chemistry_verified": False})
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "annotations.jsonl"
    temporary = directory / "annotations.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, target)
    core.atomic_json(directory / "annotations.json", rows)
    fields = ["experiment_index", "source_index", "chembl_id", "smiles", *core.FEATURE_NAMES, "property_vector", "facts_sha256",
              "pair_status", "first_description", "second_description", "first_description_sha256", "second_description_sha256",
              "text_changed", "text_diff", "chemistry_verified"]
    temporary = directory / "annotations.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            flat.update(row["properties"])
            for key, value in flat.items():
                if isinstance(value, (dict, list)):
                    flat[key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            writer.writerow(flat)
    os.replace(temporary, directory / "annotations.csv")
    return {"target_count": TARGET, "paired_rows": len(rows), "first_pass_accepted": sum(core.row_accepted(row) for row in first_rows),
            "second_pass_accepted": sum(core.row_accepted(row) for row in second_rows), "pair_status_counts": dict(counts),
            "complete": counts["complete_pair"] == TARGET, "chemistry_verified": False}


async def process_pair(bid, rid, items, old_first, old_second, client, first_writer, second_writer, first_config, second_config, args):
    first = old_first if core.check_saved(old_first, items) else await core.annotate_batch(
        bid, rid, items, old_first, client, first_writer, first_config, args)
    if first.get("status") != "completed":
        return first, None
    # Even a completed review is checked against the exact immutable first pass.
    if old_second is not None and review.check_saved_review(old_second, items, first, second_config):
        return first, old_second
    if client.stop.is_set():
        return first, old_second
    second = await review.review_batch(bid, rid, items, first, old_second, client, second_writer, second_config, args)
    return first, second


async def run_async(args, top, first_config, second_config, selection, items, keys, balances, first_store, second_store):
    first_writer, second_writer = core.AsyncWriter(first_store), core.AsyncWriter(second_store)
    stop, tasks, registered = asyncio.Event(), set(), []
    capacity = min(args.max_pending_batches, sum(args.account_caps[:len(keys)]), TARGET // 5)
    started, completed, reason = time.monotonic(), 0, None
    loop = asyncio.get_running_loop()
    def stop_now(value):
        nonlocal reason
        if not stop.is_set():
            reason = value
            stop.set()
            print(f"Stopping both stages: {value}; draining active calls", flush=True)
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_now, signal.Signals(signum).name)
        registered.append(signum)
    async def watch():
        while not stop.is_set():
            if args.stop_file.exists():
                stop_now("stop_file")
                return
            await asyncio.sleep(0.5)
    watcher = asyncio.create_task(watch())
    core.atomic_json(args.output_dir / "runtime.json", {"pid": os.getpid(), "started_at_utc": core.utc_now(), "target_molecules": TARGET,
                     "account_caps": args.account_caps[:len(keys)], "max_pending_batches": capacity, "stop_file": str(args.stop_file),
                     "run_fingerprint": top["run_fingerprint"], "pipeline": "each successful first batch immediately enters review"})
    timeout = aiohttp.ClientTimeout(total=args.request_timeout, connect=30, sock_read=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=capacity, limit_per_host=capacity, ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        client = RecordingClient(session, keys, first_config, args, stop)
        for index in range(len(keys)):
            if balances.get(f"account_{index + 1}", {}).get("is_available") is False:
                client.disabled[index], client.disabled_reasons[index] = True, "balance_explicitly_unavailable"
        async def report(final=False):
            await first_writer.call("export")
            await second_writer.call("export")
            a, b = await first_writer.call("records"), await second_writer.call("records")
            value = await asyncio.to_thread(export_pairs, args.output_dir / "paired", selection, a, b)
            ap, bp = await first_writer.call("progress"), await second_writer.call("progress")
            usage = Counter(ap["usage"])
            usage.update(bp["usage"])
            value.update(first_pass=ap, second_pass=bp, usage=dict(usage), usage_by_stage={"first_pass": ap["usage"], "second_pass": bp["usage"]},
                         run_fingerprint=top["run_fingerprint"], elapsed_seconds=round(time.monotonic() - started, 2),
                         active_batches=len(tasks), paused=stop.is_set(), stop_reason=reason or client.stop_reason,
                         account_disabled=client.disabled, account_disabled_reasons=client.disabled_reasons, updated_at_utc=core.utc_now())
            core.atomic_json(args.output_dir / ("summary.json" if final else "progress.json"), value)
            print(f"First={value['first_pass_accepted']}/{TARGET}; reviewed={value['second_pass_accepted']}/{TARGET}; pairs={value['pair_status_counts']}; active={len(tasks)}", flush=True)
            return value
        iterator, exhausted = iter(core.batches(items)), False
        try:
            while tasks or (not exhausted and not stop.is_set()):
                while len(tasks) < capacity and not exhausted and not stop.is_set():
                    entry = next(iterator, None)
                    if entry is None:
                        exhausted = True
                        break
                    bid, rid, batch = entry
                    a, b = await first_writer.call("load", bid), await second_writer.call("load", bid)
                    tasks.add(asyncio.create_task(process_pair(bid, rid, batch, a, b, client, first_writer, second_writer, first_config, second_config, args)))
                if tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    tasks = set(pending)
                    for task in done:
                        task.result()
                        completed += 1
                    if completed // 20 != (completed - len(done)) // 20:
                        await report()
        except BaseException:
            stop_now("runner_error")
            await asyncio.gather(*tasks, return_exceptions=True)
            tasks.clear()
            raise
        finally:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            for signum in registered:
                loop.remove_signal_handler(signum)
            try:
                summary = await report(final=True)
            finally:
                await first_writer.close()
                await second_writer.close()
    return summary


def main(argv=None):
    args = parse_args(argv)
    with output_lock(args.output_dir), patched_runner():
        paths = [args.input, args.molecules, args.properties, args.first_prompt, args.review_prompt]
        before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
        items = load_selected(args)
        top, first_config, second_config, selection = make_configs(args, items)
        if before != [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]:
            raise ValueError("A frozen source or prompt changed during preparation")
        preserve_config(args.output_dir / "run_config.json", top)
        preserve_config(args.output_dir / "first_pass/run_config.json", first_config)
        preserve_config(args.output_dir / "second_pass/run_config.json", second_config)
        preserve_config(args.output_dir / "selection.json", selection)
        previews = [{"batch_id": bid, "requested_ids": [item["chembl_id"] for item in batch],
                     "request_payload": request_payload(grounded.messages_for(batch, first_config), first_config)}
                    for bid, _, batch in core.batches(items)]
        for preview in previews:
            preview["request_sha256"] = core.digest(preview["request_payload"])
        core.atomic_json(args.output_dir / "first_pass/requests_preview.json", previews)
        if args.dry_run:
            if not (args.output_dir / "paired/annotations.jsonl").exists():
                export_pairs(args.output_dir / "paired", selection, [], [])
            print(f"Dry run: {TARGET} uniformly sampled molecules, {TARGET // 5} first batches, no keys/API; fingerprint={top['run_fingerprint']}", flush=True)
            return top
        if args.stop_file.exists():
            print(f"STOP preserved; experiment not started: {args.stop_file}", flush=True)
            return None
        keys = core.read_keys()
        if len(args.account_caps) < len(keys):
            raise ValueError("Provide a concurrency cap for each configured account")
        balances = {} if args.skip_balance_check else {f"account_{i}": core.get_balance(key) for i, key in enumerate(keys, 1)}
        if balances:
            if not (args.output_dir / "balance_before.json").exists():
                core.atomic_json(args.output_dir / "balance_before.json", balances)
            core.atomic_json(args.output_dir / "balance_latest.json", balances)
        first_store = ExperimentStore(args.output_dir / "first_pass", first_config, items)
        second_store = ExperimentStore(args.output_dir / "second_pass", second_config, items)
        first_store.recover()
        existing_second = second_store.recover()
        for bid in sorted(existing_second):
            first_batch = first_store.load(bid)
            if first_batch is None or first_batch.get("status") != "completed":
                raise ValueError("Saved second-pass batch lacks a completed first-pass baseline")
            review.check_saved_review(second_store.load(bid), items[(bid - 1) * 5:bid * 5], first_batch, second_config)
        return asyncio.run(run_async(args, top, first_config, second_config, selection, items, keys, balances, first_store, second_store))


if __name__ == "__main__":
    main()
