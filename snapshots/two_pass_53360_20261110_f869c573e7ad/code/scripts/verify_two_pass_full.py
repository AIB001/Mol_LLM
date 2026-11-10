#!/usr/bin/env python3
"""Read-only, bounded-memory audit of the complete two-pass source population.

Seed500 retains its original batch namespace and fingerprints. New journals use
the complementary source indices. An ephemeral SQLite index bounds memory; it
is neither a production database nor a release artifact. No model reasoning is
read as evidence or reproduced in the report. No network calls are made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import sqlite3
import tempfile

from analyze_review_experiment import Checks, accepted, finite_number, verify_payload
from audit_natural_provenance import FEATURE_NAMES, digest, file_hash, integer_usage, parse_annotations
from natural_structure_context import natural_structure_context
from screen_local_region_trial import text_hash

SCHEMA = "two_pass_full_integrity_audit/1"
TARGET = 100000
SEED_COUNT = 500
CODE_FILES = {"runner_code_sha256":"annotate_natural_pilot.py","validation_code_sha256":"natural_description_checks.py",
    "shared_validation_code_sha256":"annotation_checks_v2.py","shared_runner_code_sha256":"annotate_deepseek.py",
    "shared_store_code_sha256":"annotate_deepseek_v2.py","format_code_sha256":"format_controlled_json.py",
    "grounding_wrapper_code_sha256":"annotate_natural_grounded.py","structure_context_code_sha256":"natural_structure_context.py",
    "readable_fact_code_sha256":"readable_facts_v3.py","experiment_runner_code_sha256":"run_review_experiment.py",
    "review_module_code_sha256":"review_annotation_module.py","review_validator_code_sha256":"review_annotation_module.py",
    "full_runner_code_sha256":"run_two_pass_full.py"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def json_array(path):
    """Incremental JSON-array reader, holding at most one record plus a chunk."""
    decoder, buffer, pos, eof = json.JSONDecoder(), "", 0, False
    with Path(path).open(encoding="utf-8") as handle:
        def more():
            nonlocal buffer, pos, eof
            chunk = handle.read(65536)
            buffer, pos, eof = buffer[pos:] + chunk, 0, not chunk
        def skip():
            nonlocal pos
            while True:
                while pos < len(buffer) and buffer[pos].isspace(): pos += 1
                if pos < len(buffer) or eof: return
                more()
        more(); skip()
        if not buffer or buffer[pos] != "[": raise ValueError("Expected JSON array")
        pos += 1; first = True
        while True:
            skip()
            if pos < len(buffer) and buffer[pos] == "]":
                pos += 1; skip()
                if pos != len(buffer) or not eof: raise ValueError("Trailing JSON array content")
                return
            if not first:
                if pos >= len(buffer) or buffer[pos] != ",": raise ValueError("Missing array separator")
                pos += 1; skip()
            while True:
                try:
                    row, end = decoder.raw_decode(buffer, pos)
                    pos = end; break
                except json.JSONDecodeError:
                    if eof: raise
                    more()
            yield row
            first = False


def relative_path(root, value):
    if not isinstance(value, str): raise ValueError("Invalid lineage path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe lineage path")
    result = Path(root) / p
    if result.is_symlink() or any(x.is_symlink() for x in result.parents if x != Path(root).parent):
        raise ValueError("Lineage path traverses a symlink")
    return result


def fingerprint(config, checks, label):
    checks.require(config.get("run_fingerprint") == digest({k: v for k, v in config.items() if k != "run_fingerprint"}),
                   f"{label}: configuration fingerprint mismatch")


def code_bindings(config, checks, hashes, label):
    for key, name in CODE_FILES.items():
        if key in config:
            path=Path(__file__).parent/name
            sha=file_hash(path); hashes[str(path)]=sha
            checks.require(config[key] == sha, f"{label}: frozen dependency changed: {name}")


def identity(row, source, checks, label, experiment_index=None):
    checks.require(type(row.get("source_index")) is int, f"{label}: source index is not an integer")
    for key in ("chembl_id", "smiles", "source_index", "facts_sha256"):
        checks.require(row.get(key) == source.get(key), f"{label}: {key} mismatch")
    if experiment_index is not None:
        checks.require(type(row.get("experiment_index")) is int and row.get("experiment_index") == experiment_index, f"{label}: experiment index mismatch")
    props, vector = row.get("properties"), row.get("property_vector")
    checks.require(isinstance(props, dict) and set(props) == set(FEATURE_NAMES)
                   and all(finite_number(v) for v in props.values()) and props == source["properties"], f"{label}: 24-properties mismatch")
    checks.require(isinstance(vector, list) and len(vector) == 24 and all(finite_number(v) for v in vector)
                   and vector == [source["properties"][k] for k in FEATURE_NAMES], f"{label}: 24-vector mismatch")


def inventory(directory):
    directory=Path(directory)
    paths=[directory/name for name in ("run_config.json","seed_manifest.json","source_index.jsonl","source_index_manifest.json","summary.json")]
    for phase in ("first_pass","second_pass","paired"):
        paths.extend(directory/phase/name for name in ("run_config.json","annotations.jsonl","annotations.json","annotations.csv","summary.json"))
        paths.extend((directory/phase/"batches").glob("batch_*.json"))
    paths.extend((directory/"lineage/seed500").rglob("*.json"))
    # Release/documentation snapshots may contain files with the same basename;
    # they are not mutable generation inputs or journals in this namespace.
    return {str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in paths if path.is_file()}


def seed_lineage(directory, config, checks, hashes):
    manifest_path = directory / "seed_manifest.json"
    manifest = read_json(manifest_path)
    hashes[str(manifest_path)] = file_hash(manifest_path)
    checks.require(manifest.get("schema_version") == "two_pass_seed_import/1", "Unexpected seed manifest schema")
    checks.require(config.get("seed_manifest_sha256") == digest(manifest), "Seed manifest canonical digest mismatch")
    listed = set()
    for entry in manifest.get("files", []):
        path = relative_path(directory, entry["relative_path"])
        checks.require(entry["relative_path"] not in listed, "Duplicate seed manifest file")
        listed.add(entry["relative_path"])
        sha = file_hash(path); hashes[str(path)] = sha
        checks.require(sha == entry.get("sha256") and path.stat().st_size == entry.get("size", entry.get("size_bytes")),
                       f"Seed archive bytes differ: {entry['relative_path']}")
        original = Path(entry["source_path"])
        checks.require(original.is_file() and file_hash(original) == sha, f"Original seed evidence differs: {entry['relative_path']}")
    root = directory / "lineage/seed500"
    required = {f"lineage/seed500/{name}" for name in ("run_config.json", "selection.json", "first_pass/run_config.json", "second_pass/run_config.json", "analysis/summary.json")}
    checks.require(required <= listed, "Seed manifest omits configuration/selection/passed audit")
    prior = read_json(root / "analysis/summary.json")
    checks.require(prior.get("valid_source_and_provenance") is True and prior.get("error_count") == 0
                   and prior.get("completed_pair_count") == SEED_COUNT, "Seed experiment did not pass its independent 500-pair audit")
    prior_inputs=prior.get("input_file_sha256",{})
    if prior_inputs:
        for entry in manifest.get("files",[]):
            original=entry["source_path"]
            archived=entry["relative_path"].removeprefix("lineage/seed500/")
            required_prior = archived in {"run_config.json","selection.json","summary.json","audit_selection.json","first_pass/run_config.json","second_pass/run_config.json"} or "/batches/" in archived
            # Assistant claim reviews were written after the automatic audit.
            # Their original/archive hashes are still checked above, but the
            # earlier audit cannot legitimately contain their later hashes.
            if required_prior or original in prior_inputs:
                checks.require(prior_inputs.get(original) == entry["sha256"], f"Seed bytes differ from previously audited evidence: {entry['relative_path']}")
    selection, seed_config = read_json(root / "selection.json"), read_json(root / "run_config.json")
    fingerprint(seed_config, checks, "seed top")
    checks.require(seed_config.get("selection_sha256") == digest(selection), "Archived seed selection digest mismatch")
    rows, seed_map = selection.get("records", []), {}
    checks.require(len(rows) == SEED_COUNT and len(manifest.get("records", [])) == SEED_COUNT, "Seed selection/manifest must contain 500 rows")
    pointers = {r["source_index"]: r for r in manifest.get("records", [])}
    checks.require(len(pointers) == len(rows), "Duplicate seed source indices")
    for position, row in enumerate(rows, 1):
        index = row["source_index"]
        checks.require(type(index) is int and index not in seed_map and type(row.get("experiment_index")) is int and row.get("experiment_index") == position, "Seed selection indices are invalid")
        pointer = pointers.get(index, {})
        checks.require(pointer.get("chembl_id") == row["chembl_id"], "Seed pointer identity mismatch")
        for phase in ("first_pass", "second_pass"):
            ref = pointer.get(phase, {})
            expected = f"lineage/seed500/{phase}/batches/batch_{(position-1)//5+1:06d}.json"
            checks.require(ref.get("relative_path") == expected and ref.get("record_index") == (position-1)%5 and expected in listed,
                           f"Seed {index}/{phase}: original batch/record mapping mismatch")
        seed_map[index] = {"source": row, "pointer": pointer}
    return seed_map, seed_config


def load_sources(directory, config, seed, db, checks, hashes, target):
    paths = {"input_sha256": Path(config["input_path"]), "molecules_sha256": Path(config["molecules_path"]),
             "properties_sha256": Path(config["properties_path"])}
    for key, path in paths.items():
        sha = file_hash(path); hashes[str(path)] = sha
        checks.require(config.get(key) == sha, f"Source file digest mismatch: {key}")
    index_path = directory / "source_index.jsonl"
    hashes[str(index_path)] = file_hash(index_path)
    checks.require(config.get("source_index_sha256") == hashes[str(index_path)], "Source-index file digest mismatch")
    opener = gzip.open if paths["input_sha256"].suffix == ".gz" else open
    count = new_count = 0
    with paths["molecules_sha256"].open(newline="", encoding="utf-8") as mh, paths["properties_sha256"].open(newline="", encoding="utf-8") as ph, opener(paths["input_sha256"], "rt", encoding="utf-8") as fh:
        streams = itertools.zip_longest(csv.DictReader(mh), csv.DictReader(ph), fh, jsonl(index_path))
        for count, values in enumerate(streams, 1):
            molecule, prop_row, fact_line, indexed = values
            if not checks.require(all(v is not None for v in values), f"Source streams unequal length at {count}"): continue
            facts = json.loads(fact_line)
            props = {k: float(prop_row[k]) for k in FEATURE_NAMES}
            source = {"source_index": count, "chembl_id": molecule["chembl_id"], "smiles": molecule["smiles"],
                      "properties": props, "property_vector": [props[k] for k in FEATURE_NAMES], "facts_sha256": facts["facts_sha256"]}
            checks.require(prop_row["chembl_id"] == source["chembl_id"] and facts.get("chembl_id") == source["chembl_id"] and facts.get("smiles") == source["smiles"], f"Source alignment mismatch at {count}")
            checks.require(digest(facts["structural_facts"]) == facts["facts_sha256"], f"Source facts digest mismatch at {count}")
            checks.require(facts.get("graph_validation", {}).get("status") == "exact_graph_roundtrip", f"Source graph proof missing at {count}")
            context = natural_structure_context(facts["structural_facts"])
            checks.require(digest(facts["structural_facts"]) == facts["facts_sha256"], f"Context construction mutated facts at {count}")
            source["structural_context_sha256"] = digest(context)
            identity(indexed, source, checks, f"Source index {count}")
            checks.require(indexed.get("structural_context_sha256") == source["structural_context_sha256"], f"Source index {count}: context digest mismatch")
            if count in seed:
                identity(seed[count]["source"], source, checks, f"Seed source {count}")
                checks.require(indexed.get("new_index") is None and indexed.get("new_batch_id") is None, f"Seed {count} assigned to new generation")
            else:
                new_count += 1
                checks.require(type(indexed.get("new_index")) is int and type(indexed.get("new_batch_id")) is int and indexed.get("new_index") == new_count and indexed.get("new_batch_id") == (new_count-1)//5+1, f"New-source dense mapping mismatch at {count}")
            try:
                db.execute("INSERT INTO sources VALUES(?,?,?,?,?)", (count, source["chembl_id"], json.dumps(source), json.dumps(context), None if count in seed else new_count))
            except sqlite3.IntegrityError:
                checks.require(False, f"Duplicate source ID/index at {count}")
            if count % 1000 == 0: db.commit()
    db.commit()
    checks.require(count == target and new_count == target-len(seed), "Original source population does not equal target")
    return count


def fetch_source(db, index):
    row = db.execute("SELECT source,context,new_index FROM sources WHERE idx=?", (index,)).fetchone()
    if row is None: raise ValueError("Journal refers to an unknown source index")
    return json.loads(row[0]), json.loads(row[1]), row[2]


def unusable_response_kind(event):
    """Only explicit failed receipt/parsing with no usable response evidence."""
    if (event.get("response") is not None or event.get("usage") != {}
            or any(event.get(key) is not None for key in ("response_id","model_returned","response_content"))):
        return None
    error=event.get("error")
    if isinstance(error,str) and error.startswith("transport_"):
        return "transport"
    if error == "invalid_http_json" and event.get("http_status") == 200:
        return "invalid_http_json"
    return None


def verify_batch(directory, path, phase, namespace, config, expected, seed, db, checks, response_ids, hashes):
    batch = read_json(path); bid = batch.get("batch_id"); records = batch.get("records", [])
    label = f"{namespace}/{phase}/{bid}"
    hashes[str(path)] = file_hash(path)
    ids = [s["chembl_id"] for s in expected]
    checks.require([r.get("chembl_id") for r in records] == ids == batch.get("chembl_ids"), f"{label}: batch identity/order mismatch")
    checks.require(batch.get("run_fingerprint") == config["run_fingerprint"], f"{label}: original batch fingerprint mismatch")
    round_id = 1 if namespace == "seed500" else (bid-1)//200+1
    checks.require(batch.get("round_id") == round_id and path.name == f"batch_{bid:06d}.json", f"{label}: batch filename/round mismatch")
    sources, contexts, firsts, first_hashes = {}, {}, {}, {}
    for source in expected:
        actual, context, _ = fetch_source(db, source["source_index"])
        sources[source["chembl_id"]] = actual; contexts[source["chembl_id"]] = context
        if phase == "second_pass":
            first = db.execute("SELECT row_json,batch_hash FROM records WHERE phase='first_pass' AND idx=?", (source["source_index"],)).fetchone()
            if checks.require(first is not None, f"{label}: review has no original first batch"):
                firsts[source["chembl_id"]] = json.loads(first[0]); first_hashes[source["chembl_id"]] = first[1]
    attempts = batch.get("attempt_log", [])
    if phase == "second_pass" and "first_batch_records_sha256" in batch:
        checks.require(set(first_hashes.values()) == {batch["first_batch_records_sha256"]}, f"{label}: batch-level original first digest mismatch")
    usage, account_usage, parsed, unknown, invalid_json = Counter(), defaultdict(Counter), {}, [], []
    for ai, event in enumerate(attempts):
        event_label = f"{label}/attempt{ai}"; requested = event.get("requested_ids", [])
        valid_request = isinstance(requested, list) and bool(requested) and len(set(requested)) == len(requested) and requested == [cid for cid in ids if cid in requested]
        checks.require(valid_request, f"{event_label}: invalid requested subset/order")
        checks.require(event.get("requested_keys") == [str(i) for i in range(1,len(requested)+1)], f"{event_label}: key remapping mismatch")
        if valid_request:
            verify_payload(event.get("request_payload"), event, config, requested, sources, contexts, firsts if phase == "second_pass" else None, checks, event_label)
        checks.require(event.get("request_stage") == phase, f"{event_label}: request stage mismatch")
        counts = event.get("usage", {})
        checks.require(isinstance(counts, dict) and all(type(v) is int and v >= 0 for v in counts.values()), f"{event_label}: malformed usage")
        if isinstance(counts, dict) and all(type(v) is int for v in counts.values()):
            usage.update(counts); account_usage[event.get("account")].update(counts)
        checks.require(finite_number(event.get("duration_seconds")) and event["duration_seconds"] >= 0, f"{event_label}: invalid duration")
        response = event.get("response")
        if not isinstance(response, dict):
            failed_kind=unusable_response_kind(event)
            checks.require(event.get("http_status") != 200 or failed_kind is not None, f"{event_label}: successful body missing without explicit receipt/parsing failure")
            checks.require(not counts, f"{event_label}: usage exists without retained raw response")
            if failed_kind == "transport": unknown.append(event_label)
            if failed_kind == "invalid_http_json": invalid_json.append(event_label)
            continue
        checks.require(event.get("http_status") == 200 and event.get("error") is None, f"{event_label}: raw response lacks success evidence")
        rid = response.get("id")
        checks.require(isinstance(rid, str) and rid and rid not in response_ids and rid == event.get("response_id"), f"{event_label}: response ID duplicated/mismatched")
        response_ids.add(rid)
        checks.require(response.get("model") == event.get("model_returned") == config.get("model"), f"{event_label}: model mismatch")
        checks.require(integer_usage(response.get("usage")) == counts and all(type(counts.get(k)) is int for k in ("prompt_tokens", "completion_tokens", "total_tokens")) and counts.get("total_tokens") == counts.get("prompt_tokens",0)+counts.get("completion_tokens",0), f"{event_label}: raw usage mismatch")
        choices = response.get("choices", [])
        content = choices[0].get("message", {}).get("content") if choices else None
        checks.require(content == event.get("response_content"), f"{event_label}: raw final content mismatch")
        try:
            if phase == "second_pass":
                annotations = json.loads(content)["annotations"]
                if [r.get("key") for r in annotations] != [str(i) for i in range(1,len(requested)+1)] or any(set(r) != {"key","description"} or not isinstance(r["description"],str) for r in annotations):
                    raise ValueError("Invalid strict review response")
            parsed[ai] = parse_annotations(response, len(requested))
        except (ValueError, TypeError, KeyError, AttributeError): pass
    checks.require(Counter(batch.get("usage", {})) == usage, f"{label}: batch usage mismatch")
    checks.require({k:Counter(v) for k,v in batch.get("usage_by_account", {}).items()} == dict(account_usage), f"{label}: account usage mismatch")
    batch_records_hash = digest(records)
    accepted_count = 0
    for pos, (row, source) in enumerate(zip(records, expected)):
        cid, index = source["chembl_id"], source["source_index"]
        row_label = f"{label}/{cid}"
        identity(row, sources[cid], checks, row_label, source["experiment_index"])
        checks.require(row.get("batch_id") == bid and row.get("round_id") == round_id, f"{row_label}: row batch mismatch")
        checks.require(row.get("run_fingerprint") == config["run_fingerprint"] and row.get("prompt_sha256") == config["prompt_sha256"], f"{row_label}: row config/prompt mismatch")
        checks.require(row.get("chemistry_verified") is False, f"{row_label}: unsubstantiated chemistry verification")
        if namespace == "seed500":
            checks.require(seed[index]["pointer"][phase].get("record_sha256") == digest(row), f"{row_label}: imported seed record digest mismatch")
        if accepted(row):
            accepted_count += 1; ai = row.get("selected_attempt_index")
            if checks.require(type(ai) is int and ai in parsed, f"{row_label}: selected response invalid"):
                event = attempts[ai]; requested = event["requested_ids"]
                if checks.require(cid in requested, f"{row_label}: selected response never requested ID"):
                    key = str(requested.index(cid)+1)
                    checks.require(parsed[ai].get(key) == row["description"], f"{row_label}: final text differs from actual response")
                    checks.require(not event.get("validation_errors", {}).get(key), f"{row_label}: selected text failed validation")
                for rk, ek in (("response_id","response_id"),("model","model_returned"),("account","account"),("selected_pass_number","pass_number")):
                    checks.require(row.get(rk) == event.get(ek), f"{row_label}: selected provenance {rk} mismatch")
            checks.require(text_hash(row["description"]) == row.get("description_sha256"), f"{row_label}: description SHA mismatch")
            checks.require("\n" not in row["description"] and "\r" not in row["description"], f"{row_label}: description is not one paragraph")
            if phase == "second_pass":
                first = firsts.get(cid, {})
                checks.require(accepted(first), f"{row_label}: review lacks accepted first text")
                checks.require(row.get("first_description") == first.get("description") and row.get("first_description_sha256") == first.get("description_sha256"), f"{row_label}: first text changed")
                checks.require(row.get("first_batch_records_sha256") == first_hashes.get(cid), f"{row_label}: first batch digest mismatch")
                checks.require(type(row.get("text_changed")) is bool and row["text_changed"] == (row["description"] != first.get("description")), f"{row_label}: text change flag mismatch")
        origin = {"kind": namespace, "journal_path": path.relative_to(directory).as_posix(), "batch_id": bid,
                  "record_index": pos, "original_record_sha256": digest(row), "run_fingerprint": config["run_fingerprint"]}
        try:
            db.execute("INSERT INTO records VALUES(?,?,?,?,?,?)", (phase,index,json.dumps(row),digest({**row,"origin":origin}),batch_records_hash,int(accepted(row))))
        except sqlite3.IntegrityError: checks.require(False, f"{row_label}: duplicate record/source")
    checks.require(batch.get("status") != "completed" or accepted_count == len(expected), f"{label}: false completed status")
    return usage, len(attempts), unknown, dict(account_usage), invalid_json


def audit_phases(directory, config, seed, db, checks, hashes, target):
    all_totals, response_ids, result = Counter(), set(), {}
    for phase in ("first_pass", "second_pass"):
        totals, attempts, unknown, invalid_json, origin_usage = Counter(), 0, [], [], {}
        for namespace, folder in (("seed500", directory/"lineage/seed500"/phase), ("new", directory/phase)):
            cfg_path = folder/"run_config.json"; phase_config = read_json(cfg_path)
            hashes[str(cfg_path)] = file_hash(cfg_path); fingerprint(phase_config, checks, f"{namespace}/{phase}")
            code_bindings(phase_config,checks,hashes,f"{namespace}/{phase}")
            checks.require(phase_config.get("feature_names") == FEATURE_NAMES, f"{namespace}/{phase}: feature order mismatch")
            checks.require(text_hash(phase_config.get("system_prompt", "")) == phase_config.get("prompt_sha256"), f"{namespace}/{phase}: prompt digest mismatch")
            if namespace == "new":
                checks.require(config.get(phase+"_run_fingerprint") == phase_config["run_fingerprint"], f"{phase}: top phase fingerprint mismatch")
                seed_cfg = read_json(directory/"lineage/seed500"/phase/"run_config.json")
                for key in ("system_prompt","prompt_sha256","model","thinking","reasoning_effort","max_tokens","temperature"):
                    checks.require(phase_config.get(key) == seed_cfg.get(key), f"{phase}: seed/new generation setting differs: {key}")
            prompt_path = Path(phase_config["prompt_path"])
            hashes[str(prompt_path)] = file_hash(prompt_path)
            checks.require(prompt_path.read_text(encoding="utf-8").strip() == phase_config["system_prompt"], f"{phase}: prompt file differs")
            subtotal = Counter(); seen_bids = set()
            for path in sorted((folder/"batches").glob("batch_*.json")):
                bid = int(path.stem.split("_")[-1])
                maximum = math.ceil(SEED_COUNT/5) if namespace == "seed500" else math.ceil((target-len(seed))/5)
                if not checks.require(1 <= bid <= maximum and bid not in seen_bids, f"{namespace}/{phase}: invalid batch ID"): continue
                seen_bids.add(bid)
                if namespace == "seed500":
                    expected = [value["source"] for _, value in sorted(seed.items(), key=lambda kv: kv[1]["source"]["experiment_index"])][(bid-1)*5:bid*5]
                else:
                    expected = []
                    for source_json, new_index in db.execute("SELECT source,new_index FROM sources WHERE new_index BETWEEN ? AND ? ORDER BY new_index", ((bid-1)*5+1,bid*5)):
                        expected.append({**json.loads(source_json), "experiment_index":new_index})
                usage, calls, warnings, _, failed_json = verify_batch(directory,path,phase,namespace,phase_config,expected,seed,db,checks,response_ids,hashes)
                subtotal.update(usage); attempts += calls; unknown.extend(warnings);invalid_json.extend(failed_json)
                if bid % 100 == 0: db.commit()
            if namespace == "seed500": checks.require(len(seen_bids) == math.ceil(SEED_COUNT/5), f"{phase}: seed journals missing")
            totals.update(subtotal); origin_usage[namespace] = dict(subtotal)
        db.commit(); all_totals.update(totals)
        stored, accepted_rows = db.execute("SELECT count(*),coalesce(sum(accepted),0) FROM records WHERE phase=?", (phase,)).fetchone()
        result[phase] = {"stored_rows":stored,"accepted_rows":accepted_rows,"http_attempt_count":attempts,"usage":dict(totals),"usage_by_origin":origin_usage,"unknown_transport_billing_attempts":unknown,"unknown_invalid_http_json_billing_attempts":invalid_json}
    return result, dict(all_totals)


def verify_export(directory, name, db, checks, hashes, target):
    folder = directory/name
    paths = [folder/"annotations.jsonl",folder/"annotations.csv",folder/"annotations.json"]
    if not all(path.exists() for path in paths):
        checks.require(False, f"{name}: missing JSONL/CSV/JSON export"); return 0, 0, [], {}
    exports = [{"path":str(path),"sha256":file_hash(path)} for path in paths]
    count = complete = previous_index = 0
    stats, reasons, word_counts = Counter(), Counter(), Counter()
    with paths[1].open(newline="",encoding="utf-8") as handle:
        reader = csv.DictReader(handle); fields = reader.fieldnames or []
        checks.require({"source_index","chembl_id","smiles",*FEATURE_NAMES} <= set(fields), f"{name}: CSV source columns missing")
        for count, (row, flat, pretty) in enumerate(itertools.zip_longest(jsonl(paths[0]), reader, json_array(paths[2])), 1):
            if not checks.require(row is not None and flat is not None and pretty is not None, f"{name}: export lengths disagree"): continue
            checks.require(row == pretty, f"{name}/{count}: pretty JSON differs")
            index = row.get("source_index")
            checks.require(type(index) is int and index > previous_index, f"{name}: export source order is not unique/increasing")
            previous_index = index
            if name == "paired": checks.require(index == count, "Paired source sequence has a gap")
            source, _, _ = fetch_source(db,index)
            identity(row,source,checks,f"{name}/{count}")
            values = {**row, **row.get("properties",{})}
            for key in fields:
                value = values.get(key)
                rendered = json.dumps(value,ensure_ascii=False,separators=(",",":")) if isinstance(value,(dict,list)) else "" if value is None else str(value)
                checks.require(flat[key] == rendered, f"{name}/{count}: CSV {key} differs")
            if name in ("first_pass","second_pass"):
                journal = db.execute("SELECT row_sha,accepted FROM records WHERE phase=? AND idx=?", (name,index)).fetchone()
                checks.require(journal is not None and journal[0] == digest(row), f"{name}/{count}: complete journal-to-export record mismatch")
                complete += bool(journal and journal[1])
                if accepted(row):
                    words=len(row["description"].split());word_counts[words]+=1
                    stats["word_count_in_100_to_150"]+=100 <= words <= 150
                    for reason in set(row.get("review_reasons",[])): reasons[reason]+=1
                    stats["rows_with_automated_review_reasons"]+=bool(row.get("review_reasons"))
            else:
                ok = True
                for phase in ("first_pass","second_pass"):
                    journal = db.execute("SELECT row_sha,accepted FROM records WHERE phase=? AND idx=?",(phase,index)).fetchone()
                    nested = row.get(phase)
                    if journal:
                        checks.require(isinstance(nested,dict) and digest(nested) == journal[0], f"paired/{count}: nested {phase} differs")
                    else: checks.require(nested is None, f"paired/{count}: fabricated missing stage")
                    ok = ok and bool(journal and journal[1])
                    prefix = "first" if phase == "first_pass" else "second"
                    checks.require(row.get(prefix+"_description") == (nested or {}).get("description") and row.get(prefix+"_description_sha256") == (nested or {}).get("description_sha256"), f"paired/{count}: flattened {prefix} text mismatch")
                checks.require((row.get("pair_status") == "complete_pair") is ok, f"paired/{count}: pair status mismatch")
                if ok:
                    complete += 1
                    checks.require(type(row.get("text_changed")) is bool and row["text_changed"] == (row["first_description"] != row["second_description"]), f"paired/{count}: exact change flag mismatch")
                    stats["changed" if row["first_description"] != row["second_description"] else "unchanged"]+=1
    if name == "paired": checks.require(count == target, "Paired export does not retain the complete fixed denominator")
    else: checks.require(count == db.execute("SELECT count(*) FROM records WHERE phase=?",(name,)).fetchone()[0], f"{name}: journal/export row count differs")
    quality={**dict(stats),"automated_review_reason_counts":dict(reasons),"word_count_min":min(word_counts) if word_counts else None,
             "word_count_max":max(word_counts) if word_counts else None,
             "word_count_mean":sum(k*v for k,v in word_counts.items())/sum(word_counts.values()) if word_counts else None,
             "interpretation":"Format/editorial statistics and screening flags only; not chemical error counts or accuracy."}
    return count, complete, exports, quality


def audit(directory, expected_count=TARGET, allow_incomplete=False, scratch_dir=None):
    directory = Path(directory).resolve(); checks = Checks(); before = inventory(directory)
    config_path = directory/"run_config.json"; config = read_json(config_path)
    hashes = {str(config_path):file_hash(config_path)}
    fingerprint(config,checks,"full top")
    code_bindings(config,checks,hashes,"full top")
    checks.require(config.get("schema_version") == "two_pass_full_run/1" and config.get("dataset_size") == expected_count, "Full-run schema/target mismatch")
    seed, _ = seed_lineage(directory,config,checks,hashes)
    with tempfile.TemporaryDirectory(prefix="two-pass-audit-",dir=scratch_dir) as temporary:
        db = sqlite3.connect(str(Path(temporary)/"audit.sqlite"))
        db.executescript("PRAGMA cache_size=-8192; PRAGMA journal_mode=OFF; CREATE TABLE sources(idx INTEGER PRIMARY KEY,cid TEXT UNIQUE,source TEXT,context TEXT,new_index INTEGER UNIQUE); CREATE TABLE records(phase TEXT,idx INTEGER,row_json TEXT,row_sha TEXT,batch_hash TEXT,accepted INTEGER,PRIMARY KEY(phase,idx));")
        source_count = load_sources(directory,config,seed,db,checks,hashes,expected_count)
        phases, usage = audit_phases(directory,config,seed,db,checks,hashes,expected_count)
        exports, complete, quality = [], 0, {}
        for name in ("first_pass","second_pass","paired"):
            _, completed, files, statistics = verify_export(directory,name,db,checks,hashes,expected_count)
            exports.extend(files)
            quality[name]=statistics
            if name == "paired": complete = completed
        db.close()
    summary = read_json(directory/"summary.json")
    hashes[str(directory/"summary.json")] = file_hash(directory/"summary.json")
    checks.require(summary.get("active_batches",0) == 0, "Writer still has active request batches")
    for field, value in (("first_pass_accepted",phases["first_pass"]["accepted_rows"]),("second_pass_accepted",phases["second_pass"]["accepted_rows"])):
        checks.require(summary.get(field) == value, f"Summary {field} mismatch")
    checks.require(Counter(summary.get("usage",{})) == Counter(usage), "Summary total token usage mismatch")
    if "completed_pair_count" in summary:
        checks.require(summary["completed_pair_count"] == complete, "Summary completed-pair count mismatch")
    if "run_fingerprint" in summary:
        checks.require(summary["run_fingerprint"] == config["run_fingerprint"], "Summary run fingerprint mismatch")
    for phase in ("first_pass","second_pass"):
        if phase in summary:
            checks.require(Counter(summary[phase].get("usage",{})) == Counter(phases[phase]["usage"]), f"Summary {phase} usage mismatch")
        for summary_field, origin in (("seed_usage_by_stage","seed500"),("new_usage_by_stage","new")):
            if summary_field in summary:
                checks.require(Counter(summary[summary_field].get(phase,{})) == Counter(phases[phase]["usage_by_origin"][origin]), f"Summary {phase}/{origin} usage mismatch")
    if not allow_incomplete:
        checks.require(complete == expected_count and all(phases[p]["accepted_rows"] == expected_count for p in phases), "Not every source has two accepted descriptions")
    checks.require(inventory(directory) == before, "Run files changed during audit")
    for path, sha in hashes.items():
        checks.require(file_hash(path) == sha, "Audited input changed during verification")
    for entry in exports:
        checks.require(file_hash(entry["path"]) == entry["sha256"], "Export changed during verification")
    valid = checks.count == 0
    return {"schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),"valid":valid,
            "valid_source_and_provenance":valid,"publish_ready":valid and expected_count == TARGET and complete == TARGET,
            "run_fingerprint":config["run_fingerprint"],"target_count":expected_count,"source_count":source_count,
            "complete_pairs":complete,"first_pass_accepted":phases["first_pass"]["accepted_rows"],"second_pass_accepted":phases["second_pass"]["accepted_rows"],
            "missing_pairs":expected_count-complete,"seed_pairs":len(seed),"new_target_pairs":expected_count-len(seed),
            "error_count":checks.count,"errors_first_200":checks.errors,"phases":phases,"reported_usage":usage,"editorial_and_format_statistics":quality,
            "input_hashes":hashes,"data_exports":exports,"chemistry_verified":False,
            "limitations":["Source/request/response/export consistency is not independent chemical accuracy verification.",
                "Model self-review and exact text changes do not prove correction.",
                "Transport failures without a complete response may have unknown provider billing; only received raw usage is summed.",
                "Explicit invalid_http_json attempts retain no usable response body or usage; their cost and discarded body contents are unknown, and no accepted record may select them.",
                "Request-task duration includes queuing and transport; it is not isolated model computation time.",
                "Original journals retain private reasoning, but reasoning is neither reviewed as evidence nor included in this report."]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--expected-count",type=int,default=TARGET)
    parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--scratch-dir",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv)
    report=audit(args.run_dir,args.expected_count,args.allow_incomplete,args.scratch_dir)
    path=args.report or args.run_dir/"two_pass_integrity_audit.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");tmp.replace(path)
    print(json.dumps({key:report[key] for key in ("valid","publish_ready","complete_pairs","missing_pairs","error_count")}))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
