#!/usr/bin/env python3
"""Stream the 100,000-molecule two-pass run, reusing the audited random 500.

Original seed journals retain their bytes, identifiers and fingerprints in a
separate namespace. The remaining 99,500 source-ordered molecules form 19,900
new five-molecule batches. Each successful first batch immediately enters review.
Only bounded in-flight batches retain structural facts; exports are per-round,
with cumulative source-ordered exports at graceful stop/completion.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import fcntl
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import aiohttp
import annotate_natural_grounded as grounded
import review_annotation_module as review
import run_review_experiment as experiment

core = grounded.runner
ROOT = core.ROOT
TARGET, SEED_COUNT, BATCH_SIZE, ROUND_SIZE = 100000, 500, 5, 1000
NEW_COUNT = TARGET - SEED_COUNT
SCHEMA = "two_pass_full_run/1"
SEED_SCHEMA = "two_pass_seed_import/1"
DEFAULT_OUTPUT = ROOT / "results/chembl_100000_two_pass"
DEFAULT_SEED = ROOT / "results/prompt_trials/minimal_review_500"
CORE_RECORD = core.base_record


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unsigned(value):
    return {key: item for key, item in value.items() if key != "run_fingerprint"}


def same_path(left, right):
    try:
        return Path(left).samefile(right)
    except OSError:
        return Path(left).resolve() == Path(right).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_structural_facts_v2.jsonl.gz")
    parser.add_argument("--molecules", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--first-prompt", type=Path, default=ROOT / "prompts/experiments/natural_local_regions_system.txt")
    parser.add_argument("--review-prompt", type=Path, default=ROOT / "prompts/experiments/natural_minimal_review_system.txt")
    parser.add_argument("--seed-dir", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=TARGET)
    parser.add_argument("--account-caps", type=core.account_caps, default=(400, 400))
    parser.add_argument("--max-pending-batches", type=int, default=800)
    parser.add_argument("--http-retries", type=int, default=6)
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--repair-passes", type=int, default=3)
    parser.add_argument("--review-passes", type=int, default=2)
    parser.add_argument("--dry-run", "--prepare-only", dest="prepare_only", action="store_true")
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args(argv)
    if args.limit != TARGET or min(args.max_pending_batches, args.http_retries, args.request_timeout) < 1:
        parser.error("The full population is fixed at 100000; transport limits must be positive")
    if not 1 <= args.repair_passes <= 3 or not 1 <= args.review_passes <= 3:
        parser.error("Validation passes must be 1..3")
    args.prompt, args.stop_file = args.first_prompt, args.output_dir / "STOP"
    args.model, args.thinking, args.max_tokens, args.temperature = "deepseek-v4-pro", "enabled", 32768, 0.6
    return args


def guard_output(directory, seed_directory):
    directory, results = Path(directory).resolve(), (ROOT / "results").resolve()
    if directory == results or not directory.is_relative_to(results) or same_path(directory, seed_directory):
        raise ValueError("Use a separate full-run directory below results")
    for ancestor in [directory, *directory.parents]:
        path = ancestor / "run_config.json"
        if path.exists() and (ancestor != directory or read_json(path).get("schema_version") != SCHEMA):
            raise ValueError("Refusing an existing incompatible or enclosing result directory")
        if ancestor == results:
            break
    if not (directory / "run_config.json").exists():
        for phase in ("first_pass", "second_pass"):
            if list((directory / phase / "batches").glob("*.json")) or (directory / phase / "annotations.sqlite").exists():
                raise ValueError("Existing generation data lacks a full-run configuration")


@contextmanager
def output_lock(directory, seed_directory):
    guard_output(directory, seed_directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".two_pass_worker.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another full two-pass worker holds this output lock") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def build_seed_manifest(directory):
    """Validate complete trial evidence and pin original bytes; no writes/network."""
    directory = Path(directory)
    top = read_json(directory / "run_config.json")
    audit = read_json(directory / "analysis/summary.json")
    summary = read_json(directory / "summary.json")
    if (core.digest(unsigned(top)) != top.get("run_fingerprint") or top.get("dataset_size") != SEED_COUNT
            or audit.get("valid_source_and_provenance") is not True or audit.get("completed_pair_count") != SEED_COUNT
            or audit.get("error_count") != 0 or summary.get("complete") is not True
            or summary.get("first_pass_accepted") != SEED_COUNT or summary.get("second_pass_accepted") != SEED_COUNT
            or summary.get("active_batches") != 0):
        raise ValueError("The original 500 seed must be complete and have a successful source/provenance audit")
    files, records, usage = [], {}, {}
    relatives = ["run_config.json", "selection.json", "summary.json", "audit_selection.json", "analysis/summary.json",
                 "first_pass/run_config.json", "second_pass/run_config.json"]
    relatives.extend(str(path.relative_to(directory)) for path in sorted((directory/"analysis").glob("chemical_review*.json")))
    for phase in ("first_pass", "second_pass"):
        config = read_json(directory / phase / "run_config.json")
        if core.digest(unsigned(config)) != config.get("run_fingerprint"):
            raise ValueError("Seed phase configuration fingerprint mismatch")
        paths = sorted((directory / phase / "batches").glob("batch_*.json"))
        if len(paths) != SEED_COUNT // 5:
            raise ValueError("Seed journal count mismatch")
        totals = Counter()
        seen = set()
        for bid, path in enumerate(paths, 1):
            batch = read_json(path)
            rows = batch.get("records", [])
            if path.name != f"batch_{bid:06d}.json" or batch.get("batch_id") != bid or batch.get("status") != "completed" or len(rows) != 5 or batch.get("run_fingerprint") != config["run_fingerprint"]:
                raise ValueError("Seed batch identity/status mismatch")
            relative = str(path.relative_to(directory))
            relatives.append(relative)
            for position, row in enumerate(rows):
                if not core.row_accepted(row) or review.text_hash(row["description"]) != row.get("description_sha256") or row["chembl_id"] in seen:
                    raise ValueError("Seed accepted-record hash/identity mismatch")
                seen.add(row["chembl_id"])
                entry = records.setdefault(row["source_index"], {"source_index": row["source_index"], "chembl_id": row["chembl_id"]})
                if entry["chembl_id"] != row["chembl_id"]:
                    raise ValueError("Seed phases disagree on source identity")
                entry[phase] = {"relative_path": "lineage/seed500/" + relative, "record_index": position,
                                "record_sha256": core.digest(row), "run_fingerprint": config["run_fingerprint"]}
            totals.update(batch.get("usage", {}))
        if len(seen) != SEED_COUNT or Counter(summary["usage_by_stage"][phase]) != totals or Counter(audit[phase]["usage"]) != totals:
            raise ValueError("Seed source/provenance usage or population mismatch")
        usage[phase] = dict(totals)
    if len(records) != SEED_COUNT or any(not all(phase in row for phase in ("first_pass", "second_pass")) for row in records.values()):
        raise ValueError("Seed has incomplete pairs")
    for relative in relatives:
        path = directory / relative
        files.append({"source_path": str(path.resolve()), "relative_path": "lineage/seed500/" + relative,
                      "sha256": core.file_hash(path), "size": path.stat().st_size})
    manifest = {"schema_version": SEED_SCHEMA, "seed_directory": str(directory.resolve()), "seed_count": SEED_COUNT,
            "original_run_fingerprint": top["run_fingerprint"], "files": files,
            "records": [records[index] for index in sorted(records)], "seed_usage_by_stage": usage,
            "policy": "Copy original journals unchanged; reuse paired texts without new calls; seed batch namespace remains separate"}
    verify_seed_audit_bindings(manifest, audit)
    return manifest


def verify_seed_audit_bindings(manifest, audit):
    """Bind imported source bytes to the prior successful audit, not only its flag."""
    if (audit.get("valid_source_and_provenance") is not True or audit.get("completed_pair_count") != SEED_COUNT
            or audit.get("error_count") != 0):
        raise ValueError("Seed audit is not a successful complete paired audit")
    audited = audit.get("input_file_sha256")
    if not isinstance(audited, dict):
        raise ValueError("Seed audit does not bind its input file bytes")
    required = {"run_config.json", "selection.json", "summary.json", "audit_selection.json",
                "first_pass/run_config.json", "second_pass/run_config.json"}
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        if relative.parts[:2] != ("lineage", "seed500") or ".." in relative.parts:
            raise ValueError("Unsafe seed archive path in audit binding")
        suffix = str(Path(*relative.parts[2:]))
        is_journal = suffix.startswith(("first_pass/batches/", "second_pass/batches/"))
        if suffix not in required and not is_journal:
            continue  # The audit report itself and later semantic reviews are separately pinned by the manifest.
        expected = audited.get(entry["source_path"])
        if expected is None:
            matches = [checksum for path, checksum in audited.items() if same_path(path, entry["source_path"])]
            expected = matches[0] if len(matches) == 1 else None
        if expected != entry["sha256"]:
            raise ValueError("Imported seed file does not match the prior successful audit: " + suffix)


def copy_seed(directory, manifest):
    for entry in manifest["files"]:
        relative = Path(entry["relative_path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("lineage", "seed500"):
            raise ValueError("Unsafe seed archive path")
        target = directory / relative
        if target.exists():
            if core.file_hash(target) != entry["sha256"]:
                raise ValueError("Existing archived seed bytes changed")
            continue
        source = Path(entry["source_path"])
        if core.file_hash(source) != entry["sha256"]:
            raise ValueError("Original seed changed during import")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        if core.file_hash(temporary) != entry["sha256"]:
            raise ValueError("Copied seed checksum mismatch")
        os.replace(temporary, target)


def seed_rows(directory, manifest):
    result = {"first_pass": {}, "second_pass": {}}
    cache = {}
    for entry in manifest["records"]:
        for phase in result:
            pointer = entry[phase]
            relative = pointer["relative_path"]
            if relative not in cache:
                cache[relative] = read_json(directory / relative)["records"]
            raw = cache[relative][pointer["record_index"]]
            if core.digest(raw) != pointer["record_sha256"] or raw["source_index"] != entry["source_index"] or raw["chembl_id"] != entry["chembl_id"]:
                raise ValueError("Seed record pointer no longer matches original journal")
            result[phase][entry["source_index"]] = {**raw, "origin": {"kind": "seed500", "journal_path": relative,
                "batch_id": raw["batch_id"], "record_index": pointer["record_index"],
                "original_record_sha256": pointer["record_sha256"], "run_fingerprint": raw["run_fingerprint"]}}
    return result


def source_hashes(args):
    return {"input_path": str(args.input.resolve()), "input_sha256": core.file_hash(args.input),
            "molecules_path": str(args.molecules.resolve()), "molecules_sha256": core.file_hash(args.molecules),
            "properties_path": str(args.properties.resolve()), "properties_sha256": core.file_hash(args.properties)}


def prepare_index(args, manifest, hashes, seeds):
    """Stream facts once into a small identity/property index, never retaining graphs."""
    target, header = args.output_dir / "source_index.jsonl", args.output_dir / "source_index_manifest.json"
    expected = {**hashes, "population": TARGET, "seed_manifest_sha256": core.digest(manifest),
                "context_code_sha256": core.file_hash(Path(review.natural_structure_context.__code__.co_filename))}
    if target.exists() and header.exists():
        saved = read_json(header)
        if any(saved.get(key) != value for key, value in expected.items()) or saved.get("source_index_sha256") != core.file_hash(target):
            raise ValueError("Prepared source index or its inputs changed")
        return saved["source_index_sha256"]
    seed_set = {row["source_index"] for row in manifest["records"]}
    temporary = target.with_name(target.name + ".tmp")
    opener = gzip.open if args.input.suffix == ".gz" else open
    new_index, count, ids = 0, 0, set()
    with opener(args.input, "rt", encoding="utf-8") as handle, temporary.open("w", encoding="utf-8", buffering=1024*1024) as out:
        for index, (line, original) in enumerate(itertools.zip_longest(handle, core.read_items(args.molecules, args.properties)), 1):
            count = index
            if line is None or original is None or index > TARGET:
                raise ValueError("Source input lengths differ from exactly 100000")
            raw = json.loads(line)
            if raw["chembl_id"] != original["chembl_id"] or raw["smiles"] != original["smiles"] or raw["chembl_id"] in ids:
                raise ValueError("Original source identity/order mismatch")
            ids.add(raw["chembl_id"])
            if core.digest(raw["structural_facts"]) != raw["facts_sha256"] or raw.get("graph_validation", {}).get("status") != "exact_graph_roundtrip":
                raise ValueError("Original fact digest/graph proof mismatch")
            seeded = index in seed_set
            if not seeded:
                new_index += 1
            row = {**original, "source_index": index, "facts_sha256": raw["facts_sha256"],
                   "property_vector": [original["properties"][key] for key in core.FEATURE_NAMES],
                   "structural_context_sha256": core.digest(review.natural_structure_context(raw["structural_facts"])),
                   "new_index": None if seeded else new_index, "new_batch_id": None if seeded else (new_index - 1)//5 + 1}
            if seeded:
                for phase in seeds:
                    prior = seeds[phase][index]
                    if any(prior.get(key) != row[key] for key in ("chembl_id", "smiles", "source_index", "facts_sha256", "properties", "property_vector")):
                        raise ValueError("Seed differs from the full source index")
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 10000 == 0:
                print(f"Prepared source index {index}/{TARGET}; structural graphs discarded after each row", flush=True)
    if count != TARGET or new_index != NEW_COUNT:
        raise ValueError("Source or seed exclusion population mismatch")
    os.replace(temporary, target)
    checksum = core.file_hash(target)
    core.atomic_json(header, {**expected, "source_index_sha256": checksum})
    return checksum


def load_index(directory):
    with (directory / "source_index.jsonl").open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != TARGET or [row["source_index"] for row in rows] != list(range(1, TARGET+1)):
        raise ValueError("Full source index is incomplete or unordered")
    return rows


def make_configs(args, manifest, index_hash):
    first = grounded.make_config(args)
    first.pop("run_fingerprint")
    source_seed = read_json(args.output_dir / "lineage/seed500/first_pass/run_config.json")
    seed_top = read_json(args.output_dir / "lineage/seed500/run_config.json")
    for key, path in [("review_module_code_sha256", Path(review.__file__)),
                      ("experiment_runner_code_sha256", Path(experiment.__file__))]:
        if core.file_hash(path) != seed_top.get(key):
            raise ValueError("Current review/pipeline code differs from the frozen seed: " + key)
    for key in [key for key in first if key.endswith("code_sha256")] + ["input_sha256", "molecules_sha256", "properties_sha256", "model", "thinking", "max_tokens", "temperature", "prompt_sha256"]:
        if first[key] != source_seed.get(key):
            raise ValueError("Full run changes a frozen seed model, prompt, source or dependency: " + key)
    shared = {"full_runner_code_sha256": core.file_hash(Path(__file__)), "review_module_code_sha256": core.file_hash(Path(review.__file__)),
              "experiment_runner_code_sha256": core.file_hash(Path(experiment.__file__)),
              "seed_manifest_sha256": core.digest(manifest), "source_index_sha256": index_hash,
              "selection_sha256": index_hash, "molecules_path": str(args.molecules.resolve()), "properties_path": str(args.properties.resolve()),
              "new_molecule_count": NEW_COUNT, "seed_molecule_count": SEED_COUNT,
              "first_validation_passes_per_invocation": args.repair_passes, "review_validation_passes_per_invocation": args.review_passes,
              "batch_policy": "New molecules in source order excluding seed500; 5/new batch. Original seed batches remain unchanged in lineage/seed500",
              "record_order": "Final exports source_index1..100000; new experiment_index is dense1..99500; seed experiment_index is unchanged"}
    first.update(shared, dataset_size=TARGET, experiment_role="first_pass", round_size=ROUND_SIZE)
    first["run_fingerprint"] = core.digest(first)
    second = unsigned(first)
    prompt = args.review_prompt.read_text(encoding="utf-8").strip()
    if hashlib.sha256(prompt.encode()).hexdigest() != read_json(args.output_dir / "lineage/seed500/second_pass/run_config.json")["prompt_sha256"]:
        raise ValueError("The review prompt differs from the original 500 experiment")
    second.update(schema_version=review.SCHEMA_VERSION, experiment_role="second_pass", generation_mode=review.GENERATION_MODE,
                  first_pass_run_fingerprint=first["run_fingerprint"], system_prompt=prompt,
                  prompt_path=str(args.review_prompt.resolve()), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                  review_validator_code_sha256=shared["review_module_code_sha256"],
                  validation_scope="Explicit format/content restrictions only; no word-count or style repair; chemical flags do not trigger rewriting",
                  model_input_fields=["molecules[key,chembl_id,smiles,structural_context,properties]", "annotations[key,description]"])
    second["run_fingerprint"] = core.digest(second)
    top = {"schema_version": SCHEMA, "dataset_size": TARGET, "target_count": TARGET, "source_population": TARGET,
           "batch_size": 5, "round_size": ROUND_SIZE, **shared,
           **{key: first[key] for key in ("input_path", "input_sha256", "molecules_sha256", "properties_sha256", "feature_names", "model", "thinking", "reasoning_effort", "max_tokens", "temperature")},
           "seed_directory": str(args.seed_dir.resolve()), "seed_manifest_path": "seed_manifest.json",
           "first_prompt_path": str(args.first_prompt.resolve()), "review_prompt_path": str(args.review_prompt.resolve()),
           "first_prompt_file_sha256": core.file_hash(args.first_prompt), "review_prompt_file_sha256": core.file_hash(args.review_prompt),
           "first_pass_run_fingerprint": first["run_fingerprint"], "second_pass_run_fingerprint": second["run_fingerprint"],
           "chemistry_verified": False}
    top["run_fingerprint"] = core.digest(top)
    return top, first, second


def validate_existing_configuration(directory, *, verify_seed_files=True, verify_sources=True):
    """Read-only API for supervisors/auditors; never imports seeds or reads keys."""
    directory = Path(directory)
    top = read_json(directory / "run_config.json")
    if top.get("schema_version") != SCHEMA or top.get("dataset_size") != TARGET or core.digest(unsigned(top)) != top.get("run_fingerprint"):
        raise ValueError("Invalid full configuration fingerprint/scope")
    manifest = read_json(directory / "seed_manifest.json")
    if core.digest(manifest) != top["seed_manifest_sha256"] or core.file_hash(directory / "source_index.jsonl") != top["source_index_sha256"]:
        raise ValueError("Full source/seed index changed")
    if verify_seed_files:
        for entry in manifest["files"]:
            relative = Path(entry["relative_path"])
            if relative.is_absolute() or ".." in relative.parts or relative.parts[:2] != ("lineage", "seed500"):
                raise ValueError("Unsafe seed manifest path")
            if core.file_hash(directory / relative) != entry["sha256"]:
                raise ValueError("Archived seed bytes changed")
    verify_seed_audit_bindings(manifest, read_json(directory / "lineage/seed500/analysis/summary.json"))
    if verify_sources:
        for path_key, hash_key in [("input_path", "input_sha256"), ("molecules_path", "molecules_sha256"), ("properties_path", "properties_sha256"),
                                   ("first_prompt_path", "first_prompt_file_sha256"), ("review_prompt_path", "review_prompt_file_sha256")]:
            if core.file_hash(Path(top[path_key])) != top[hash_key]:
                raise ValueError("Frozen input/prompt file changed: " + path_key)
    args = SimpleNamespace(input=Path(top["input_path"]), molecules=Path(top["molecules_path"]), properties=Path(top["properties_path"]),
                           prompt=Path(top["first_prompt_path"]), first_prompt=Path(top["first_prompt_path"]), review_prompt=Path(top["review_prompt_path"]),
                           seed_dir=Path(top["seed_directory"]), output_dir=directory, model=top["model"], thinking=top["thinking"],
                           max_tokens=top["max_tokens"], temperature=top["temperature"], repair_passes=top["first_validation_passes_per_invocation"],
                           review_passes=top["review_validation_passes_per_invocation"])
    with patched_runner():
        expected, first, second = make_configs(args, manifest, top["source_index_sha256"])
    if expected != top or read_json(directory / "first_pass/run_config.json") != first or read_json(directory / "second_pass/run_config.json") != second:
        raise ValueError("Current code or stage configuration differs from the frozen full run")
    return top


def base_record(item, bid, rid, config):
    row = CORE_RECORD(item, bid, rid, config)
    row.update(experiment_index=item["new_index"], experiment_role=config["experiment_role"], experiment_selection_sha256=config["selection_sha256"])
    return row


@contextmanager
def patched_runner():
    with grounded.patched_runner():
        old_record, old_size = core.base_record, core.DATASET_SIZE
        core.base_record, core.DATASET_SIZE = base_record, TARGET
        try:
            yield
        finally:
            core.base_record, core.DATASET_SIZE = old_record, old_size


class FullStore(core.BatchStore):
    """New-journal cache; seed journals stay outside this namespace."""
    def __init__(self, directory, config, new_items):
        directory.mkdir(parents=True, exist_ok=True)
        self.config, self.items, self.stats = config, new_items, None
        super().__init__(directory, config["run_fingerprint"])
        self.db.executescript("""CREATE TABLE IF NOT EXISTS journal_cache(batch_id INTEGER PRIMARY KEY,size INTEGER,mtime_ns INTEGER,sha256 TEXT,first_hash TEXT);
            CREATE TABLE IF NOT EXISTS full_metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE INDEX IF NOT EXISTS annotations_source_order ON annotations(CAST(json_extract(payload,'$.source_index') AS INTEGER));""")
        old = self.db.execute("SELECT value FROM full_metadata WHERE key='fingerprint'").fetchone()
        if old and old[0] != self.fingerprint:
            raise ValueError("SQLite cache belongs to a different full run")
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO full_metadata VALUES('fingerprint',?)", (self.fingerprint,))

    def validate(self, batch):
        bid = batch.get("batch_id")
        if type(bid) is not int or not 1 <= bid <= NEW_COUNT//5 or batch.get("round_id") != (bid-1)//200+1 or batch.get("run_fingerprint") != self.fingerprint:
            raise ValueError("New journal identity/namespace/fingerprint mismatch")
        rows, expected = batch.get("records", []), self.items[(bid-1)*5:bid*5]
        if len(rows) != 5 or [row.get("chembl_id") for row in rows] != batch.get("chembl_ids"):
            raise ValueError("New journal population/order mismatch")
        for row, item in zip(rows, expected):
            if any(row.get(key) != item[key] for key in ("chembl_id", "smiles", "source_index", "facts_sha256", "properties", "property_vector")) or row.get("experiment_index") != item["new_index"]:
                raise ValueError("Journal differs from full source identity/properties")
            if row.get("batch_id") != bid or row.get("round_id") != batch["round_id"] or row.get("run_fingerprint") != self.fingerprint:
                raise ValueError("New record batch/fingerprint mismatch")
            if core.row_accepted(row) and review.text_hash(row["description"]) != row.get("description_sha256"):
                raise ValueError("Accepted new description hash mismatch")

    def index_journal(self, batch, path, checksum):
        self.validate(batch)
        old = self.db.execute("SELECT status,usage FROM batches WHERE batch_id=?", (batch["batch_id"],)).fetchone()
        old_counts = Counter(dict(self.db.execute("SELECT status,COUNT(*) FROM annotations WHERE round_id=? AND batch_id=? GROUP BY status", (batch["round_id"], batch["batch_id"])))) if old else Counter()
        super()._index(batch)
        stat = path.stat()
        self.db.execute("INSERT OR REPLACE INTO journal_cache VALUES(?,?,?,?,?)", (batch["batch_id"],stat.st_size,stat.st_mtime_ns,checksum,batch.get("first_batch_records_sha256")))
        if self.stats is not None:
            self.stats["status_counts"].subtract(old_counts)
            self.stats["status_counts"].update(row["status"] for row in batch["records"])
            self.stats["usage"].subtract(json.loads(old[1]) if old else {})
            self.stats["usage"].update(batch.get("usage", {}))
            self.stats["saved_batches"] += 0 if old else 1
            self.stats["finished_batches"] += int(batch["status"] == "completed") - int(bool(old) and old[0] == "completed")

    def persist(self, batch):
        self.validate(batch)
        path = self.directory / "batches" / f"batch_{batch['batch_id']:06d}.json"
        core.atomic_json(path, batch)
        with self.db:
            self.index_journal(batch, path, core.file_hash(path))

    def recover(self):
        cached = {row[0]: row[1:] for row in self.db.execute("SELECT batch_id,size,mtime_ns,sha256 FROM journal_cache")}
        seen = set()
        for path in sorted((self.directory / "batches").glob("batch_*.json")):
            bid = int(path.stem.split("_")[-1])
            if path.name != f"batch_{bid:06d}.json" or not 1 <= bid <= NEW_COUNT//5:
                raise ValueError("Unexpected new-journal filename")
            seen.add(bid)
            stat, prior = path.stat(), cached.get(bid)
            if prior and (stat.st_size, stat.st_mtime_ns) == prior[:2]:
                continue
            batch = read_json(path)
            if batch.get("batch_id") != bid:
                raise ValueError("Journal filename disagrees with its contents")
            with self.db:
                self.index_journal(batch, path, core.file_hash(path))
        if set(cached) - seen:
            raise ValueError("A previously indexed journal is missing")
        counts = Counter(dict(self.db.execute("SELECT status,COUNT(*) FROM annotations GROUP BY status")))
        usage = Counter()
        for (raw,) in self.db.execute("SELECT usage FROM batches"):
            usage.update(json.loads(raw))
        self.stats = {"status_counts": counts, "usage": usage, "saved_batches": len(seen),
                      "finished_batches": self.db.execute("SELECT COUNT(*) FROM batches WHERE status='completed'").fetchone()[0]}
        return {row[0] for row in self.db.execute("SELECT batch_id FROM batches WHERE status='completed'")}

    def load(self, bid):
        state = self.db.execute("SELECT round_id,status FROM batches WHERE batch_id=?", (bid,)).fetchone()
        if state is None:
            return None
        if state[1] != "completed":
            return read_json(self.directory / "batches" / f"batch_{bid:06d}.json")
        rows = [json.loads(raw) for (raw,) in self.db.execute("SELECT payload FROM annotations WHERE round_id=? AND batch_id=? ORDER BY CAST(json_extract(payload,'$.source_index') AS INTEGER)", (state[0],bid))]
        first_hash = self.db.execute("SELECT first_hash FROM journal_cache WHERE batch_id=?", (bid,)).fetchone()[0]
        return {"batch_id":bid,"round_id":state[0],"status":"completed","records":rows,"run_fingerprint":self.fingerprint,"first_batch_records_sha256":first_hash}

    def progress(self):
        return {"status_counts": {key:value for key,value in self.stats["status_counts"].items() if value},
                "stored_molecules": sum(self.stats["status_counts"].values()), "usage":dict(self.stats["usage"]),
                "saved_batches":self.stats["saved_batches"],"finished_batches":self.stats["finished_batches"]}

    def rows(self, round_id=None):
        query, params = "SELECT payload FROM annotations", ()
        if round_id is not None:
            query += " WHERE round_id=?"
            params = (round_id,)
        query += " ORDER BY CAST(json_extract(payload,'$.source_index') AS INTEGER)"
        for (raw,) in self.db.execute(query, params):
            row = json.loads(raw)
            bid = row["batch_id"]
            position = (row["experiment_index"]-1)%5
            yield {**row, "origin":{"kind":"new","journal_path":f"{self.config['experiment_role']}/batches/batch_{bid:06d}.json",
                "batch_id":bid,"record_index":position,"original_record_sha256":core.digest(row),"run_fingerprint":row["run_fingerprint"]}}

    def export(self, round_id=None, seeds=None):
        rows = self.rows(round_id)
        if round_id is None:
            import heapq
            rows = heapq.merge(rows, (seeds[index] for index in sorted(seeds or {})), key=lambda row:row["source_index"])
        return write_records(self.directory, "annotations" if round_id is None else f"round_{round_id:04d}", rows, stage_fields())


def stage_fields():
    return ["chembl_id","smiles",*core.FEATURE_NAMES,"property_vector","description","status","source_index","experiment_index",
            "experiment_role","facts_sha256","description_sha256","word_count","word_count_in_target","review_reasons","chemical_review_status",
            "chemistry_verified","generation_mode","batch_id","round_id","model","account","response_id","selected_attempt_index","selected_pass_number",
            "prompt_sha256","run_fingerprint","origin","first_description","first_description_sha256","first_batch_records_sha256","text_changed","text_diff"]


def write_records(directory, name, records, fields):
    directory.mkdir(parents=True, exist_ok=True)
    jp, cp = directory/f"{name}.jsonl", directory/f"{name}.csv"
    jt, ct = jp.with_suffix(".jsonl.tmp"), cp.with_suffix(".csv.tmp")
    counts = Counter()
    with jt.open("w",encoding="utf-8",buffering=1024*1024) as jf, ct.open("w",newline="",encoding="utf-8",buffering=1024*1024) as cf:
        writer=csv.DictWriter(cf,fieldnames=fields);writer.writeheader()
        for row in records:
            counts[row.get("status",row.get("pair_status","unknown"))]+=1
            jf.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
            flat={key:row.get(key) for key in fields};flat.update(row.get("properties",{}))
            for key,value in flat.items():
                if isinstance(value,(dict,list)):flat[key]=json.dumps(value,ensure_ascii=False,separators=(",",":"))
            writer.writerow(flat)
    os.replace(jt,jp);os.replace(ct,cp)
    core.pretty_array(jp,directory/f"{name}.json")
    return dict(counts)


def jsonl_rows(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:yield json.loads(line)


def export_pairs(directory, metadata, name="annotations"):
    first=iter(jsonl_rows(directory/"first_pass"/f"{name}.jsonl"));second=iter(jsonl_rows(directory/"second_pass"/f"{name}.jsonl"))
    a,b=next(first,None),next(second,None)
    def rows():
        nonlocal a,b
        for source in metadata:
            index=source["source_index"]
            left=a if a is not None and a["source_index"]==index else None
            right=b if b is not None and b["source_index"]==index else None
            if left is not None:a=next(first,None)
            if right is not None:b=next(second,None)
            la,ra=core.row_accepted(left),core.row_accepted(right)
            if ra and not la:raise ValueError("Reviewed row lacks accepted first-pass provenance")
            state="complete_pair" if la and ra else "awaiting_second_pass" if la else "first_pass_not_accepted"
            yield {"schema_version":"two_pass_full_pair/1",**source,"pair_status":state,"first_pass":left,"second_pass":right,
                "first_description":left.get("description") if left else None,"second_description":right.get("description") if right else None,
                "first_description_sha256":left.get("description_sha256") if left else None,"second_description_sha256":right.get("description_sha256") if right else None,
                "text_changed":left["description"]!=right["description"] if la and ra else None,"text_diff":right.get("text_diff") if ra else None,"chemistry_verified":False}
        if a is not None or b is not None:raise ValueError("Stage export contains rows outside the paired source index")
    fields=["source_index","chembl_id","smiles",*core.FEATURE_NAMES,"property_vector","facts_sha256","pair_status","first_description","second_description",
            "first_description_sha256","second_description_sha256","text_changed","text_diff","chemistry_verified"]
    return write_records(directory/"paired",name,rows(),fields)


def pending_batches(args, metadata, completed):
    """Skip seed/completed rows before JSON decoding; retain at most one new batch."""
    opener=gzip.open if args.input.suffix==".gz" else open
    batch=[]
    index=0
    with opener(args.input,"rt",encoding="utf-8") as handle:
        for index,line in enumerate(handle,1):
            if index>TARGET:raise ValueError("Source gained rows after preparation")
            item=metadata[index-1];bid=item["new_batch_id"]
            if bid is None or bid in completed:continue
            raw=json.loads(line)
            if any(raw[key]!=item[key] for key in ("chembl_id","smiles","facts_sha256")) or core.digest(raw["structural_facts"])!=item["facts_sha256"]:
                raise ValueError("Streaming source differs from its frozen full index")
            batch.append({**item,**raw})
            if len(batch)==5:
                yield bid,(bid-1)//200+1,batch
                batch=[]
    if batch or index != TARGET:raise ValueError("Incomplete five-molecule batch or truncated full source")


def phase_summary(new, seed_usage):
    counts=Counter(new["status_counts"]);counts["accepted_format"]+=SEED_COUNT
    usage=Counter(new["usage"]);usage.update(seed_usage)
    return {**new,"status_counts":dict(counts),"stored_molecules":new["stored_molecules"]+SEED_COUNT,
            "saved_batches":new["saved_batches"]+SEED_COUNT//5,"finished_batches":new["finished_batches"]+SEED_COUNT//5,
            "usage":dict(usage),"new_usage":new["usage"],"seed_usage":seed_usage}


async def run_async(args, top, first_config, second_config, manifest, metadata, seeds, keys, balances, stores, completed):
    writers={name:core.AsyncWriter(store) for name,store in stores.items()}
    stop,tasks,registered=asyncio.Event(),set(),[]
    capacity=min(args.max_pending_batches,sum(args.account_caps[:len(keys)]))
    started,reason,returned=time.monotonic(),None,0
    completed_pairs=set(completed)
    rounds=Counter((bid-1)//200+1 for bid in completed_pairs)
    exported=set()
    loop=asyncio.get_running_loop()
    def request_stop(value):
        nonlocal reason
        if not stop.is_set():reason=value;stop.set();print(f"Stop dispatch: {value}; drain both stages",flush=True)
    for signum in (signal.SIGINT,signal.SIGTERM):
        loop.add_signal_handler(signum,request_stop,signal.Signals(signum).name);registered.append(signum)
    async def watch():
        while not stop.is_set():
            if args.stop_file.exists():request_stop("stop_file");return
            await asyncio.sleep(.5)
    watcher=asyncio.create_task(watch())
    core.atomic_json(args.output_dir/"runtime.json",{"pid":os.getpid(),"started_at_utc":core.utc_now(),"target_molecules":TARGET,
        "account_caps":args.account_caps[:len(keys)],"max_pending_batches":capacity,"run_fingerprint":top["run_fingerprint"],"stop_file":str(args.stop_file)})
    timeout=aiohttp.ClientTimeout(total=args.request_timeout,connect=30,sock_read=args.request_timeout)
    connector=aiohttp.TCPConnector(limit=capacity,limit_per_host=capacity,ttl_dns_cache=300)
    async with aiohttp.ClientSession(timeout=timeout,connector=connector) as session:
        client=experiment.RecordingClient(session,keys,first_config,args,stop)
        for index in range(len(keys)):
            if balances.get(f"account_{index+1}",{}).get("is_available") is False:
                client.disabled[index],client.disabled_reasons[index]=True,"balance_explicitly_unavailable"
        async def report(final=False):
            phases={name:phase_summary(await writer.call("progress"),manifest["seed_usage_by_stage"][name]) for name,writer in writers.items()}
            paired=phases["second_pass"]["status_counts"].get("accepted_format",0)
            usage=Counter(phases["first_pass"]["usage"]);usage.update(phases["second_pass"]["usage"])
            stored=phases["first_pass"]["stored_molecules"]
            value={"schema_version":"two_pass_full_progress/1","target_count":TARGET,"target_molecules":TARGET,"complete":paired==TARGET,
                "first_pass_accepted":phases["first_pass"]["status_counts"].get("accepted_format",0),
                "second_pass_accepted":phases["second_pass"]["status_counts"].get("accepted_format",0),
                "completed_pair_count":paired,"stored_molecules":stored,
                "status_counts":{"accepted_format":paired,**({"pending":stored-paired} if stored>paired else {})},
                "saved_batches":phases["first_pass"]["saved_batches"],"finished_batches":SEED_COUNT//5+len(completed_pairs),
                "first_pass":phases["first_pass"],"second_pass":phases["second_pass"],"usage":dict(usage),
                "usage_by_stage":{name:phase["usage"] for name,phase in phases.items()},
                "seed_usage_by_stage":manifest["seed_usage_by_stage"],"new_usage_by_stage":{name:phase["new_usage"] for name,phase in phases.items()},
                "run_fingerprint":top["run_fingerprint"],"active_batches":len(tasks),"in_flight_batches":len(tasks),
                "all_descriptions_present":paired==TARGET,"paused":stop.is_set(),"stop_reason":reason or client.stop_reason,
                "account_disabled":client.disabled,"account_disabled_reasons":client.disabled_reasons,
                "elapsed_seconds":round(time.monotonic()-started,2),"updated_at_utc":core.utc_now(),"chemistry_verified":False}
            core.atomic_json(args.output_dir/("summary.json" if final else "progress.json"),value)
            print(f"First={value['first_pass_accepted']}; reviewed={value['second_pass_accepted']}; complete batches={value['finished_batches']}/20000; active={len(tasks)}",flush=True)
            return value
        async def one(entry):
            bid,rid,batch=entry
            left=await writers["first_pass"].call("load",bid);right=await writers["second_pass"].call("load",bid)
            first,second=await experiment.process_pair(bid,rid,batch,left,right,client,writers["first_pass"],writers["second_pass"],first_config,second_config,args)
            return bid,rid,first,second
        iterator,exhausted=iter(pending_batches(args,metadata,completed_pairs)),False
        try:
            while tasks or (not exhausted and not stop.is_set()):
                while len(tasks)<capacity and not exhausted and not stop.is_set():
                    entry=await asyncio.to_thread(next,iterator,None)
                    if entry is None:exhausted=True;break
                    tasks.add(asyncio.create_task(one(entry)))
                if tasks:
                    done,pending=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED);tasks=set(pending)
                    for task in done:
                        bid,rid,first,second=task.result();returned+=1
                        if second and second.get("status")=="completed":completed_pairs.add(bid)
                        rounds[rid]+=1
                        expected=min(200,NEW_COUNT//5-(rid-1)*200)
                        if rounds[rid]>=expected and rid not in exported:
                            for writer in writers.values():await writer.call("export",rid)
                            subset=[row for row in metadata if row["new_batch_id"] is not None and (row["new_batch_id"]-1)//200+1==rid]
                            await asyncio.to_thread(export_pairs,args.output_dir,subset,f"round_{rid:04d}")
                            exported.add(rid)
                    if returned//20!=(returned-len(done))//20:await report()
        except BaseException:
            request_stop("runner_error")
            finished=await asyncio.gather(*tasks,return_exceptions=True)
            for result in finished:
                if isinstance(result,tuple) and result[3] and result[3].get("status")=="completed":completed_pairs.add(result[0])
            tasks.clear();raise
        finally:
            watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
            for signum in registered:loop.remove_signal_handler(signum)
            try:
                for phase,writer in writers.items():await writer.call("export",None,seeds[phase])
                await asyncio.to_thread(export_pairs,args.output_dir,metadata)
                summary=await report(final=True)
            finally:
                for writer in writers.values():await writer.close()
    return summary


def prepared_summary(top,manifest):
    usage=Counter()
    for value in manifest["seed_usage_by_stage"].values():usage.update(value)
    return {"schema_version":"two_pass_full_progress/1","target_count":TARGET,"target_molecules":TARGET,"complete":False,
        "first_pass_accepted":SEED_COUNT,"second_pass_accepted":SEED_COUNT,"completed_pair_count":SEED_COUNT,
        "stored_molecules":SEED_COUNT,"status_counts":{"accepted_format":SEED_COUNT},"saved_batches":SEED_COUNT//5,"finished_batches":SEED_COUNT//5,
        "active_batches":0,"in_flight_batches":0,"all_descriptions_present":False,"paused":True,"stop_reason":"seed_prepared",
        "seed_usage_by_stage":manifest["seed_usage_by_stage"],"new_usage_by_stage":{"first_pass":{},"second_pass":{}},"usage":dict(usage),
        "run_fingerprint":top["run_fingerprint"],"updated_at_utc":core.utc_now(),"chemistry_verified":False}


def main(argv=None):
    args=parse_args(argv)
    with output_lock(args.output_dir,args.seed_dir),patched_runner():
        manifest_path=args.output_dir/"seed_manifest.json"
        manifest=read_json(manifest_path) if manifest_path.exists() else build_seed_manifest(args.seed_dir)
        if (args.output_dir/"run_config.json").exists():
            old_config=read_json(args.output_dir/"run_config.json")
            if core.digest(unsigned(old_config)) != old_config.get("run_fingerprint") or core.digest(manifest) != old_config.get("seed_manifest_sha256"):
                raise ValueError("Existing full configuration or seed manifest changed")
        copy_seed(args.output_dir,manifest)
        verify_seed_audit_bindings(manifest, read_json(args.output_dir/"lineage/seed500/analysis/summary.json"))
        if not manifest_path.exists():core.atomic_json(manifest_path,manifest)
        seeds=seed_rows(args.output_dir,manifest)
        hashes=source_hashes(args)
        index_hash=prepare_index(args,manifest,hashes,seeds)
        top,first_config,second_config=make_configs(args,manifest,index_hash)
        experiment.preserve_config(args.output_dir/"run_config.json",top)
        experiment.preserve_config(args.output_dir/"first_pass/run_config.json",first_config)
        experiment.preserve_config(args.output_dir/"second_pass/run_config.json",second_config)
        metadata=load_index(args.output_dir)
        if args.prepare_only:
            if not (args.output_dir/"runtime.json").exists() and not (args.output_dir/"summary.json").exists():
                for phase in seeds:write_records(args.output_dir/phase,"annotations",(seeds[phase][index] for index in sorted(seeds[phase])),stage_fields())
                export_pairs(args.output_dir,metadata)
                core.atomic_json(args.output_dir/"summary.json",prepared_summary(top,manifest))
            print(f"Prepared: {TARGET} sources, {SEED_COUNT} seed pairs reused, {NEW_COUNT} new molecules; no keys/API; fingerprint={top['run_fingerprint']}",flush=True)
            return top
        if args.stop_file.exists():print(f"STOP preserved: {args.stop_file}",flush=True);return None
        new_items=[row for row in metadata if row["new_batch_id"] is not None]
        stores={phase:FullStore(args.output_dir/phase,config,new_items) for phase,config in [("first_pass",first_config),("second_pass",second_config)]}
        complete_first=stores["first_pass"].recover();complete_second=stores["second_pass"].recover()
        for (bid,) in stores["second_pass"].db.execute("SELECT batch_id FROM batches"):
            first=stores["first_pass"].load(bid)
            if bid not in complete_first or first is None:raise ValueError("Existing review has no completed original first batch")
            review.check_saved_review(stores["second_pass"].load(bid),new_items[(bid-1)*5:bid*5],first,second_config)
        keys=core.read_keys()
        if len(args.account_caps)<len(keys):raise ValueError("Provide a concurrency cap for every configured account")
        balances={} if args.skip_balance_check else {f"account_{i}":core.get_balance(key) for i,key in enumerate(keys,1)}
        if balances:
            if not (args.output_dir/"balance_before.json").exists():core.atomic_json(args.output_dir/"balance_before.json",balances)
            core.atomic_json(args.output_dir/"balance_latest.json",balances)
        return asyncio.run(run_async(args,top,first_config,second_config,manifest,metadata,seeds,keys,balances,stores,complete_first&complete_second))


if __name__=="__main__":main()
