#!/usr/bin/env python3
"""Audit a two-pass description experiment and export paired editorial comparisons.

This is source/request/response validation, not chemical verification. Text edit
metrics are descriptive only. The uniform chemical-review sample must be pinned
before any API attempts; failures remain in its fixed denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import csv
from difflib import SequenceMatcher
import gzip
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import unicodedata

from audit_natural_provenance import FEATURE_NAMES, digest, file_hash, parse_annotations, integer_usage
from screen_local_region_trial import sentences, text_hash

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/prompt_trials/minimal_review_500"
SCHEMA = "paired_review_experiment_analysis/1"
SAMPLE_SEED = 20261109


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Checks:
    def __init__(self):
        self.count, self.errors = 0, []

    def require(self, condition, message):
        if not condition:
            self.count += 1
            if len(self.errors) < 200:
                self.errors.append(message)
        return bool(condition)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def accepted(row):
    return bool(row and row.get("status") == "accepted_format" and isinstance(row.get("description"), str)
                and row["description"].strip())


def unique_index(rows, label, checks):
    result = {}
    for row in rows:
        cid = row.get("chembl_id")
        if checks.require(isinstance(cid, str) and cid and cid not in result, f"{label}: missing/duplicate ID {cid}"):
            result[cid] = row
    return result


def selection_rows(directory, expected_count, checks):
    config, selection = read_json(directory / "run_config.json"), read_json(directory / "selection.json")
    checks.require(config.get("run_fingerprint") == digest({k: v for k, v in config.items() if k != "run_fingerprint"}),
                   "Top configuration fingerprint mismatch")
    checks.require(config.get("schema_version") == "review_annotation_experiment/1", "Unexpected experiment schema")
    checks.require(config.get("dataset_size") == expected_count, "Experiment target differs from fixed denominator")
    checks.require(selection.get("schema_version") == "review_experiment_selection/1", "Unexpected selection schema")
    checks.require(config.get("selection_sha256") == digest(selection), "Selection canonical digest mismatch")
    rows = selection.get("records", [])
    checks.require(len(rows) == expected_count, "Selection does not contain the fixed number of molecules")
    unique_index(rows, "selection", checks)
    checks.require([row.get("experiment_index") for row in rows] == list(range(1, expected_count + 1)),
                   "Selection experiment indices are not complete/ordered")
    indices = [row.get("source_index") for row in rows]
    checks.require(all(type(i) is int and 1 <= i <= config.get("source_population", 100000) for i in indices)
                   and len(set(indices)) == expected_count, "Duplicate/invalid original source indices")
    checks.require(indices == sorted(random.Random(config["selection_seed"]).sample(
        range(1, config["source_population"] + 1), expected_count)), "Selection differs from the declared uniform sampling seed")
    return config, selection, rows


def prepare_sample(directory, expected_count=500, sample_count=50, seed=SAMPLE_SEED):
    """Create an immutable pre-generation sample; never inspect generated text."""
    directory = Path(directory)
    checks = Checks()
    config, selection, rows = selection_rows(directory, expected_count, checks)
    if checks.count:
        raise ValueError("; ".join(checks.errors))
    if seed == config.get("selection_seed", selection.get("seed")):
        raise ValueError("Review sample seed must be independent of the generation-selection seed")
    if not 0 < sample_count <= expected_count:
        raise ValueError("Invalid fixed review sample size")
    indices = sorted(random.Random(seed).sample(range(1, expected_count + 1), sample_count))
    chosen = [{key: rows[index - 1][key] for key in
               ("experiment_index", "source_index", "chembl_id", "smiles", "facts_sha256")} for index in indices]
    path = directory / "audit_selection.json"
    stable = {"schema_version": "paired_review_sampling_plan/1", "seed": seed,
        "fixed_experiment_denominator": expected_count, "random_pair_count": sample_count,
        "selection_canonical_sha256": digest(selection), "selection_file_sha256": file_hash(directory / "selection.json"),
        "selected_pairs": chosen,
        "diagnostic_rule": {"max_count": 10, "exclude_random_sample": True,
            "minimum_word_insert_delete_fraction": 0.20,
            "ranking": "descending word_insert_delete_fraction, then descending changed_word_count, then experiment_index"},
        "failed_pairs_are_not_replaced": True, "chemistry_verified": False}
    if path.exists():
        prior = read_json(path)
        if any(prior.get(key) != value for key, value in stable.items()):
            raise ValueError("Existing preregistered sample differs; it will not be overwritten")
        return prior
    for phase in ("first_pass", "second_pass"):
        for batch_path in (directory / phase / "batches").glob("batch_*.json"):
            if read_json(batch_path).get("attempt_log"):
                raise ValueError("Cannot retrospectively create a pre-generation review sample after API attempts")
    plan = {**stable, "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration_fingerprint_at_sampling": config["run_fingerprint"],
            "prepared_before_any_recorded_api_attempts": True,
            "sampling_code_sha256": file_hash(Path(__file__)),
            "scope": "Uniform paired review sample and separate edit-enriched diagnostic plan; neither is model self-verification."}
    write_json(path, plan)
    return plan


def edit_metrics(first, second):
    """Exact equality is primary; alignment metrics are not accuracy judgments."""
    if not isinstance(first, str) or not isinstance(second, str):
        return None
    a, b = first.split(), second.split()
    opcodes = SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    deleted = sum(j-i for tag, i, j, _, _ in opcodes if tag in ("delete", "replace"))
    inserted = sum(j-i for tag, _, _, i, j in opcodes if tag in ("insert", "replace"))
    old_sentences, new_sentences = sentences(first), sentences(second)
    sentence_edits = [{"operation": tag, "first_sentences": old_sentences[i:j],
                       "second_sentences": new_sentences[k:l]} for tag, i, j, k, l in
                      SequenceMatcher(None, old_sentences, new_sentences, autojunk=False).get_opcodes() if tag != "equal"]
    no_punctuation = lambda text: "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return {"text_changed": first != second, "exactly_unchanged": first == second,
        "first_word_count": len(a), "second_word_count": len(b),
        "first_character_count": len(first), "second_character_count": len(second),
        "deleted_word_count": deleted, "inserted_word_count": inserted,
        "changed_word_count": deleted + inserted,
        "word_insert_delete_fraction": (deleted + inserted) / max(1, len(a) + len(b)),
        "character_alignment_difference": 1 - SequenceMatcher(None, first, second, autojunk=False).ratio(),
        "whitespace_only_change": first != second and "".join(first.split()) == "".join(second.split()),
        "punctuation_only_change": first != second and no_punctuation(first) == no_punctuation(second),
        "sentence_diff": sentence_edits,
        "interpretation": "Editorial alignment only; no evidence that a changed or unchanged claim is chemically correct."}


def bind_final_configuration(directory, expected_count=500):
    """Preserve the original prelaunch plan while binding its final frozen run."""
    directory = Path(directory)
    plan_path = directory / "audit_selection.json"
    plan = read_json(plan_path)
    prepare_sample(directory, expected_count, plan["random_pair_count"], plan["seed"])
    config = read_json(directory / "run_config.json")
    stable = {"schema_version": "paired_review_sample_config_binding/1",
        "audit_selection_sha256": file_hash(plan_path),
        "selection_canonical_sha256": plan["selection_canonical_sha256"],
        "selection_file_sha256": plan["selection_file_sha256"],
        "configuration_fingerprint_at_sampling": plan.get("configuration_fingerprint_at_sampling"),
        "final_run_fingerprint": config["run_fingerprint"],
        "final_run_config_file_sha256": file_hash(directory / "run_config.json"),
        "sampling_seed_indices_and_created_at_preserved": True}
    path = directory / "audit_selection_final_config_binding.json"
    if path.exists():
        value = read_json(path)
        if any(value.get(key) != item for key, item in stable.items()):
            raise ValueError("Existing final sample/config binding differs; it will not be overwritten")
        return value
    value = {**stable, "bound_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "Attach the final frozen configuration to the already committed selection; no re-sampling and no retrospective sample selection."}
    write_json(path, value)
    return value


def numeric_identity(row, source, checks, label):
    for key in ("chembl_id", "smiles", "source_index", "experiment_index", "facts_sha256"):
        checks.require(row.get(key) == source.get(key), f"{label}: source {key} mismatch")
    props, vector = row.get("properties", {}), row.get("property_vector")
    checks.require(isinstance(props, dict) and set(props) == set(FEATURE_NAMES)
                   and all(finite_number(v) for v in props.values()) and props == source["properties"],
                   f"{label}: exact finite 24-property mismatch")
    checks.require(isinstance(vector, list) and len(vector) == 24 and all(finite_number(v) for v in vector)
                   and vector == [source["properties"][k] for k in FEATURE_NAMES], f"{label}: 24-vector mismatch")


def verify_sources(config, selected, checks):
    import natural_structure_context
    paths = {"input_sha256": Path(config["input_path"]), "molecules_sha256": Path(config["molecules_path"]),
             "properties_sha256": Path(config["properties_path"])}
    hashes = {key: file_hash(path) for key, path in paths.items()}
    for key, value in hashes.items():
        checks.require(config.get(key) == value, f"Original source file hash mismatch: {key}")
    wanted = {row["source_index"]: row for row in selected}
    actual, contexts = {}, {}
    with paths["molecules_sha256"].open(newline="") as mh, paths["properties_sha256"].open(newline="") as ph:
        for index, (molecule, props) in enumerate(itertools.zip_longest(csv.DictReader(mh), csv.DictReader(ph)), 1):
            if index > max(wanted):
                break
            if index not in wanted:
                continue
            if not checks.require(molecule is not None and props is not None and molecule["chembl_id"] == props["chembl_id"],
                                  f"Source CSV alignment mismatch at {index}"):
                continue
            row = wanted[index]
            actual[index] = {**row, "chembl_id": molecule["chembl_id"], "smiles": molecule["smiles"],
                             "properties": {key: float(props[key]) for key in FEATURE_NAMES}}
            numeric_identity(row, actual[index], checks, f"Selection {index}")
    opener = gzip.open if paths["input_sha256"].suffix == ".gz" else open
    with opener(paths["input_sha256"], "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index > max(wanted):
                break
            if index not in wanted:
                continue
            value, row = json.loads(line), wanted[index]
            checks.require(value.get("facts_sha256") == digest(value["structural_facts"]) == row["facts_sha256"],
                           f"Selection {index}: actual source facts digest mismatch")
            checks.require(value.get("chembl_id") == row["chembl_id"] and value.get("smiles") == row["smiles"],
                           f"Selection {index}: actual source fact identity mismatch")
            context = natural_structure_context.natural_structure_context(value["structural_facts"])
            checks.require(row.get("structural_context_sha256") == digest(context), f"Selection {index}: context digest mismatch")
            contexts[row["chembl_id"]] = context
    checks.require(len(actual) == len(selected) == len(contexts), "Sources do not cover all selected molecules")
    return contexts, {str(path): hashes[key] for key, path in paths.items()}


def csv_cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return "" if value is None else str(value)


def verify_exports(folder, checks, required=True):
    path = folder / "annotations.jsonl"
    if not path.exists():
        checks.require(not required, f"Missing export {path}")
        return []
    rows = read_jsonl(path)
    pretty, flat = folder / "annotations.json", folder / "annotations.csv"
    if checks.require(pretty.exists(), f"Missing pretty export {pretty}"):
        checks.require(read_json(pretty) == rows, f"Pretty JSON differs from JSONL: {folder.name}")
    if checks.require(flat.exists(), f"Missing CSV export {flat}"):
        with flat.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields, csv_rows = list(reader.fieldnames or []), list(reader)
        checks.require({"chembl_id", "smiles", "experiment_index", "source_index", *FEATURE_NAMES} <= set(fields),
                       f"CSV lacks identity/property columns: {folder.name}")
        checks.require(len(csv_rows) == len(rows), f"CSV/JSONL row count mismatch: {folder.name}")
        for row, exported in zip(rows, csv_rows):
            values = {**row, **row.get("properties", {})}
            for key in fields:
                checks.require(exported[key] == csv_cell(values.get(key)), f"{folder.name}/{row.get('chembl_id')}: CSV {key} mismatch")
    return rows


def verify_payload(payload, event, config, requested, source_by_id, contexts, first_by_id, checks, label):
    if not checks.require(isinstance(payload, dict), f"{label}: actual request payload absent"):
        return
    allowed_payload = {"model","messages","response_format","thinking","max_tokens","stream",
                       "reasoning_effort" if config.get("thinking") == "enabled" else "temperature"}
    checks.require(set(payload) == allowed_payload, f"{label}: unexpected/missing request parameters")
    checks.require(digest(payload) == event.get("request_sha256"), f"{label}: actual request digest mismatch")
    for key, expected in (("model", config.get("model")), ("max_tokens", config.get("max_tokens")),
                          ("response_format", {"type": "json_object"}), ("stream", False),
                          ("thinking", {"type": config.get("thinking")})):
        checks.require(payload.get(key) == expected, f"{label}: request {key} mismatch")
    if config.get("thinking") == "enabled":
        checks.require(payload.get("reasoning_effort") == "low", f"{label}: reasoning effort mismatch")
        checks.require("temperature" not in payload, f"{label}: unexpected temperature with enabled thinking")
    else:
        checks.require(payload.get("temperature") == config.get("temperature"), f"{label}: temperature mismatch")
    messages = payload.get("messages", [])
    if not checks.require(len(messages) == 2 and messages[0] == {"role": "system", "content": config.get("system_prompt")}
                          and messages[1].get("role") == "user", f"{label}: transmitted prompt/message mismatch"):
        return
    try:
        body = json.loads(messages[1]["content"])
        molecules = body["molecules"]
        checks.require(len(molecules) == len(requested), f"{label}: requested molecule count mismatch")
        for index, (cid, molecule) in enumerate(zip(requested, molecules), 1):
            source = source_by_id[cid]
            expected = {"key": str(index), "chembl_id": cid, "smiles": source["smiles"], "structural_context": contexts[cid]}
            if first_by_id is not None:
                expected["properties"] = source["properties"]
            checks.require(molecule == expected, f"{label}/{cid}: actual molecule/context/properties input differs")
            if "properties" in molecule:
                checks.require(all(finite_number(v) for v in molecule["properties"].values()), f"{label}/{cid}: request has boolean/nonfinite properties")
        allowed = {"molecules", "previous_format_errors"}
        if first_by_id is not None:
            allowed.add("annotations")
            expected_annotations = []
            for index, cid in enumerate(requested, 1):
                first = first_by_id.get(cid)
                checks.require(accepted(first), f"{label}/{cid}: review requested without accepted first-pass text")
                expected_annotations.append({"key": str(index), "description": first.get("description") if first else None})
            checks.require(body.get("annotations") == expected_annotations, f"{label}: review input is not the exact original first-pass descriptions")
        checks.require(set(body) <= allowed, f"{label}: unexpected extra user input fields")
    except (KeyError, TypeError, ValueError) as exc:
        checks.require(False, f"{label}: malformed actual user payload ({type(exc).__name__})")


def verify_phase(directory, name, selected, contexts, checks, response_ids, first_result=None):
    folder = directory / name
    config = read_json(folder / "run_config.json")
    checks.require(config.get("run_fingerprint") == digest({k: v for k, v in config.items() if k != "run_fingerprint"}), f"{name}: configuration fingerprint mismatch")
    checks.require(config.get("dataset_size") == len(selected) and config.get("feature_names") == FEATURE_NAMES, f"{name}: dataset/feature contract mismatch")
    checks.require(text_hash(config.get("system_prompt", "")) == config.get("prompt_sha256"), f"{name}: transmitted prompt hash mismatch")
    prompt_path = Path(config["prompt_path"])
    checks.require(prompt_path.read_text().strip() == config.get("system_prompt"), f"{name}: prompt file differs from transmitted prompt")
    rows = verify_exports(folder, checks, required=False)
    by_id = unique_index(rows, name, checks)
    source_by_id = {row["chembl_id"]: row for row in selected}
    checks.require(set(by_id) <= set(source_by_id), f"{name}: exported unselected molecule")
    checks.require([row.get("experiment_index") for row in rows] == sorted(row.get("experiment_index", 0) for row in rows), f"{name}: export order mismatch")
    totals, accounts, statuses, http, batches_by_id = Counter(), defaultdict(Counter), Counter(), Counter(), {}
    journal_rows, attempts_count, raw_count, unknown_cost = {}, 0, 0, []
    durations, starts, ends = [], [], []
    first_by_id = first_result["rows_by_id"] if first_result else None
    for path in sorted((folder / "batches").glob("batch_*.json")):
        batch = read_json(path)
        bid, records, attempts = batch["batch_id"], batch.get("records", []), batch.get("attempt_log", [])
        checks.require(type(bid) is int and 1 <= bid <= math.ceil(len(selected)/5) and bid not in batches_by_id,
                       f"{name}: unexpected/duplicate batch ID {bid}")
        batches_by_id[bid] = batch
        expected_ids = [row["chembl_id"] for row in selected[(bid-1)*5:bid*5]]
        ids = [row.get("chembl_id") for row in records]
        checks.require(batch.get("run_fingerprint") == config["run_fingerprint"], f"{name}/{bid}: batch fingerprint mismatch")
        checks.require(batch.get("chembl_ids") == ids and ids == expected_ids, f"{name}/{bid}: batch selection/order mismatch")
        checks.require(path.name == f"batch_{bid:06d}.json" and batch.get("round_id") == 1, f"{name}/{bid}: filename/round mismatch")
        local, local_accounts, parsed = Counter(), defaultdict(Counter), {}
        for ai, event in enumerate(attempts):
            label = f"{name}/{bid}/attempt{ai}"
            attempts_count += 1
            requested = event.get("requested_ids", [])
            checks.require(isinstance(requested, list) and requested and len(set(requested)) == len(requested)
                           and requested == [cid for cid in ids if cid in requested], f"{label}: invalid requested subset/order")
            checks.require(event.get("requested_keys") == [str(i) for i in range(1, len(requested)+1)], f"{label}: repair key mapping mismatch")
            verify_payload(event.get("request_payload"), event, config, requested, source_by_id, contexts, first_by_id, checks, label)
            checks.require(event.get("request_stage") == name, f"{label}: request stage mismatch")
            usage = event.get("usage", {})
            valid_usage = isinstance(usage, dict) and all(type(v) is int and v >= 0 for v in usage.values())
            checks.require(valid_usage, f"{label}: invalid token usage")
            if valid_usage:
                local.update(usage); local_accounts[event.get("account")].update(usage)
            http[str(event.get("http_status"))] += 1
            duration = event.get("duration_seconds")
            if finite_number(duration) and duration >= 0:
                durations.append(duration)
                try:
                    start = datetime.fromisoformat(event["started_at_utc"])
                    starts.append(start); ends.append(start + timedelta(seconds=duration))
                except (ValueError, KeyError, TypeError):
                    checks.require(False, f"{label}: invalid attempt timestamp")
            else:
                checks.require(False, f"{label}: invalid attempt duration")
            response = event.get("response")
            if not isinstance(response, dict):
                incomplete_transport = (str(event.get("error", "")).startswith("transport_") and not usage
                    and all(event.get(k) is None for k in ("response_id", "model_returned", "response_content")))
                checks.require(event.get("http_status") != 200 or incomplete_transport,
                               f"{label}: HTTP200 response body missing without explicit transport failure")
                if incomplete_transport:
                    unknown_cost.append(label)
                checks.require(not usage, f"{label}: usage exists without retained response")
                continue
            raw_count += 1
            checks.require(event.get("http_status") == 200 and event.get("error") is None, f"{label}: response lacks successful HTTP evidence")
            rid = response.get("id")
            checks.require(isinstance(rid, str) and rid and rid not in response_ids and rid == event.get("response_id"), f"{label}: missing/duplicate/mismatched response ID")
            response_ids.add(rid)
            checks.require(response.get("model") == event.get("model_returned") == config.get("model"), f"{label}: returned model mismatch")
            checks.require(integer_usage(response.get("usage")) == usage, f"{label}: raw response/log usage mismatch")
            checks.require(all(type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
                           and usage.get("total_tokens") == usage.get("prompt_tokens", 0)+usage.get("completion_tokens", 0), f"{label}: incomplete token totals")
            choices = response.get("choices", [])
            content = choices[0].get("message", {}).get("content") if choices else None
            checks.require(content == event.get("response_content"), f"{label}: retained final content mismatch")
            try:
                if first_result:
                    annotations = json.loads(content)["annotations"]
                    if ([r.get("key") for r in annotations] != [str(i) for i in range(1,len(requested)+1)]
                            or any(set(r) != {"key","description"} or not isinstance(r["key"],str)
                                   or not isinstance(r["description"],str) for r in annotations)):
                        raise ValueError("Review response violates strict ordered two-string schema")
                parsed[ai] = parse_annotations(response, len(requested))
            except (ValueError, TypeError, KeyError, AttributeError):
                pass  # Truncated/invalid responses remain in token totals, never selected.
        checks.require(Counter(batch.get("usage", {})) == local, f"{name}/{bid}: batch token sum mismatch")
        checks.require({k: Counter(v) for k,v in batch.get("usage_by_account", {}).items()} == dict(local_accounts), f"{name}/{bid}: account token sums mismatch")
        totals.update(local)
        for account, value in local_accounts.items():
            accounts[account].update(value)
        for row in records:
            cid = row.get("chembl_id"); label = f"{name}/{bid}/{cid}"
            checks.require(cid not in journal_rows, f"{label}: duplicate journal row")
            journal_rows[cid] = row
            if cid not in source_by_id:
                continue
            numeric_identity(row, source_by_id[cid], checks, label)
            checks.require(row.get("batch_id") == bid and row.get("round_id") == 1, f"{label}: record batch/round mismatch")
            checks.require(row.get("run_fingerprint") == config["run_fingerprint"] and row.get("prompt_sha256") == config["prompt_sha256"], f"{label}: row run/prompt mismatch")
            checks.require(row.get("chemistry_verified") is False, f"{label}: model self-review promoted to chemistry verification")
            statuses[row.get("status", "missing_status")] += 1
            if not accepted(row):
                continue
            ai = row.get("selected_attempt_index")
            if not checks.require(type(ai) is int and ai in parsed, f"{label}: selected attempt not a complete response"):
                continue
            event = attempts[ai]; requested = event["requested_ids"]
            if not checks.require(cid in requested, f"{label}: selected response did not request molecule"):
                continue
            key = str(requested.index(cid)+1)
            checks.require(parsed[ai].get(key) == row["description"], f"{label}: final text differs from raw selected response")
            checks.require(text_hash(row["description"]) == row.get("description_sha256"), f"{label}: description hash mismatch")
            for row_key, event_key in (("response_id","response_id"),("model","model_returned"),("account","account"),("selected_pass_number","pass_number")):
                checks.require(row.get(row_key) == event.get(event_key), f"{label}: selected {row_key} mismatch")
            checks.require(not event.get("validation_errors", {}).get(key), f"{label}: selected text failed recorded validation")
            if first_result:
                first = first_by_id.get(cid)
                checks.require(accepted(first), f"{label}: accepted review lacks accepted first pass")
                if first:
                    checks.require(row.get("first_description") == first.get("description"), f"{label}: saved first description is not exact original text")
                    checks.require(row.get("first_description_sha256") == first.get("description_sha256"), f"{label}: first-description pointer mismatch")
                    first_batch = first_result["batches_by_id"].get(first.get("batch_id"))
                    checks.require(first_batch is not None and row.get("first_batch_records_sha256") == digest(first_batch["records"]), f"{label}: first batch records digest mismatch")
                    checks.require(type(row.get("text_changed")) is bool and row["text_changed"] == (first["description"] != row["description"]), f"{label}: exact text_changed mismatch")
        checks.require(batch.get("status") != "completed" or all(accepted(row) for row in records), f"{name}/{bid}: completed batch contains unaccepted row")
    checks.require(journal_rows == by_id, f"{name}: full journal records differ from JSONL export")
    summary_path = folder / "summary.json"
    top_summary_path = directory / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else read_json(top_summary_path).get(name,{}) if top_summary_path.exists() else {}
    if summary:
        checks.require(Counter(summary.get("usage", {})) == totals and summary.get("status_counts") == dict(statuses)
                       and summary.get("stored_molecules") == len(journal_rows), f"{name}: summary counts/usage mismatch")
        checks.require(summary.get("in_flight_batches", 0) == 0, f"{name}: writer has in-flight batches")
    return {"config": config, "rows_by_id": by_id, "batches_by_id": batches_by_id,
        "summary": {"stored_rows": len(by_id), "accepted_rows": sum(accepted(row) for row in by_id.values()),
            "status_counts": dict(statuses), "http_attempt_count": attempts_count, "saved_response_count": raw_count,
            "http_status_counts": dict(http), "usage": dict(totals), "usage_by_account": {k:dict(v) for k,v in accounts.items()},
            "unknown_transport_billing_attempts": unknown_cost,
            "sum_request_duration_seconds": sum(durations),
            "request_duration_definition": "Request-task elapsed time, including semaphore queue wait, transport and response receipt; not isolated model reasoning or HTTP service time.",
            "first_request_started_at_utc": min(starts).isoformat() if starts else None,
            "observed_request_wall_span_seconds": (max(ends)-min(starts)).total_seconds() if starts else None,
            "runner_elapsed_seconds": summary.get("elapsed_seconds")}}


def build_comparisons(selected, first_by_id, second_by_id):
    result = []
    for source in selected:
        cid = source["chembl_id"]
        first, second = first_by_id.get(cid), second_by_id.get(cid)
        first_ok, second_ok = accepted(first), accepted(second)
        metrics = edit_metrics(first["description"], second["description"]) if first_ok and second_ok else None
        result.append({**{k:source[k] for k in ("experiment_index","source_index","chembl_id","smiles","facts_sha256","properties","property_vector")},
            "first_status": first.get("status") if first else "not_stored", "second_status": second.get("status") if second else "not_stored",
            "paired_completed": first_ok and second_ok,
            "pair_outcome": "changed" if metrics and metrics["text_changed"] else "unchanged" if metrics else "first_pass_unavailable" if not first_ok else "second_pass_unavailable",
            "first_description": first.get("description") if first else None,
            "second_description": second.get("description") if second else None,
            "first_description_sha256": first.get("description_sha256") if first else None,
            "second_description_sha256": second.get("description_sha256") if second else None,
            "edit_metrics": metrics, "chemical_assessment": "not_independently_reviewed", "chemistry_verified": False})
    return result


def review_payload(comparison, group):
    return {**comparison, "review_selection_group": group, "review_status": "not_reviewed",
        "claim_followups": [], "review_scope": "Check selected key chemical claims against the source graph in both versions; not complete atom coverage or human expert certification."}


def diagnostic_selection(comparisons, plan):
    excluded = {row["chembl_id"] for row in plan["selected_pairs"]}
    rule = plan["diagnostic_rule"]
    eligible = [row for row in comparisons if row["paired_completed"] and row["pair_outcome"] == "changed"
                and row["chembl_id"] not in excluded and row["edit_metrics"]["word_insert_delete_fraction"] >= rule["minimum_word_insert_delete_fraction"]]
    eligible.sort(key=lambda row: (-row["edit_metrics"]["word_insert_delete_fraction"], -row["edit_metrics"]["changed_word_count"], row["experiment_index"]))
    return eligible[:rule["max_count"]], len(eligible)


def analyze(directory, expected_count=500, output_dir=None):
    directory = Path(directory)
    output_dir = Path(output_dir) if output_dir else directory / "analysis"
    checks = Checks()
    runtime_path = directory / "runtime.json"
    if runtime_path.exists():
        pid = read_json(runtime_path).get("pid")
        if type(pid) is int and pid > 0:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f"Experiment writer PID {pid} is still alive; wait before final analysis")
    config, selection, selected = selection_rows(directory, expected_count, checks)
    plan_path = directory / "audit_selection.json"
    if not plan_path.exists():
        raise ValueError("No pre-generation audit_selection.json; do not create one retrospectively")
    plan = read_json(plan_path)
    prepare_sample(directory,expected_count,plan["random_pair_count"],plan["seed"])
    binding_path = directory / "audit_selection_final_config_binding.json"
    if not binding_path.exists():
        raise ValueError("Missing explicit final sample/config binding")
    binding = read_json(binding_path)
    checks.require(binding.get("audit_selection_sha256")==file_hash(plan_path)
                   and binding.get("final_run_fingerprint")==config["run_fingerprint"]
                   and binding.get("final_run_config_file_sha256")==file_hash(directory/"run_config.json"),
                   "Final sample/config binding does not match this experiment")
    checks.require(plan.get("prepared_before_any_recorded_api_attempts") is True and plan.get("selection_canonical_sha256") == digest(selection)
                   and plan.get("selection_file_sha256") == file_hash(directory/"selection.json"), "Pre-registered review sample does not bind this selection")
    expected_indices = sorted(random.Random(plan["seed"]).sample(range(1,expected_count+1), plan["random_pair_count"]))
    checks.require([r["experiment_index"] for r in plan["selected_pairs"]] == expected_indices, "Random review sample differs from its predeclared seed")
    input_paths = [directory/"run_config.json",directory/"selection.json",plan_path,binding_path]
    if (directory/"summary.json").exists(): input_paths.append(directory/"summary.json")
    for phase in ("first_pass","second_pass","paired"):
        input_paths.extend(path for path in (directory/phase).glob("*.json*") if path.is_file())
        input_paths.extend((directory/phase/"batches").glob("batch_*.json"))
        csv_path = directory/phase/"annotations.csv"
        if csv_path.exists(): input_paths.append(csv_path)
    journal_inventory = {phase:sorted(str(p) for p in (directory/phase/"batches").glob("batch_*.json")) for phase in ("first_pass","second_pass")}
    external_paths = [Path(config[key]) for key in ("input_path","molecules_path","properties_path")]
    external_paths.extend(Path(read_json(directory/phase/"run_config.json")["prompt_path"]) for phase in ("first_pass","second_pass"))
    signatures = {str(path):(path.stat().st_size,path.stat().st_mtime_ns) for path in input_paths+external_paths}
    contexts, source_hashes = verify_sources(config,selected,checks)
    response_ids=set()
    first=verify_phase(directory,"first_pass",selected,contexts,checks,response_ids)
    second=verify_phase(directory,"second_pass",selected,contexts,checks,response_ids,first)
    for name,result in (("first_pass",first),("second_pass",second)):
        checks.require(config.get(name+"_run_fingerprint")==result["config"]["run_fingerprint"],f"Top/{name} fingerprint mismatch")
        for key in ("input_sha256","molecules_sha256","properties_sha256"):
            checks.require(config.get(key)==result["config"].get(key),f"Top/{name} source hash mismatch: {key}")
    paired=verify_exports(directory/"paired",checks)
    paired_by_id=unique_index(paired,"paired export",checks)
    checks.require(len(paired)==expected_count and [r.get("chembl_id") for r in paired]==[r["chembl_id"] for r in selected],"Paired export must retain all selected identities in order, including failures")
    for source in selected:
        cid=source["chembl_id"]; pair=paired_by_id.get(cid)
        if pair is None: continue
        numeric_identity(pair,source,checks,f"Paired/{cid}")
        a,b=first["rows_by_id"].get(cid),second["rows_by_id"].get(cid)
        checks.require(pair.get("first_pass")==a and pair.get("second_pass")==b,f"Paired/{cid}: full phase records differ")
        expected_change=(a["description"]!=b["description"]) if accepted(a) and accepted(b) else None
        checks.require(pair.get("text_changed")==expected_change and (expected_change is None or type(pair.get("text_changed")) is bool),f"Paired/{cid}: exact changed state mismatch")
        checks.require(pair.get("first_description")== (a.get("description") if a else None)
                       and pair.get("second_description")== (b.get("description") if b else None),f"Paired/{cid}: flattened exact descriptions differ")
        for prefix, phase_row in (("first",a),("second",b)):
            checks.require(pair.get(prefix+"_description_sha256")==(phase_row.get("description_sha256") if phase_row else None),f"Paired/{cid}: flattened description hash mismatch")
        expected_status="complete_pair" if expected_change is not None else "awaiting_second_pass" if accepted(a) else "first_pass_not_accepted"
        checks.require(pair.get("pair_status")==expected_status,f"Paired/{cid}: pair status mismatch")
    comparisons=build_comparisons(selected,first["rows_by_id"],second["rows_by_id"])
    outcomes=Counter(r["pair_outcome"] for r in comparisons)
    combined_usage=Counter(first["summary"]["usage"])+Counter(second["summary"]["usage"])
    overall=read_json(directory/"summary.json") if (directory/"summary.json").exists() else {}
    if overall:
        checks.require(overall.get("target_count")==expected_count and overall.get("paired_rows")==expected_count,
                       "Overall summary does not retain the fixed denominator")
        checks.require(Counter(overall.get("usage",{}))==combined_usage,"Overall token total differs from both journals")
        checks.require(overall.get("first_pass_accepted")==first["summary"]["accepted_rows"]
                       and overall.get("second_pass_accepted")==second["summary"]["accepted_rows"],"Overall accepted counts mismatch")
        checks.require(overall.get("active_batches")==0,"Experiment still has active batches")
    else:
        checks.require(False,"Final experiment summary missing")
    for phase in (first,second):
        earliest=phase["summary"]["first_request_started_at_utc"]
        checks.require(earliest is None or datetime.fromisoformat(plan["created_at_utc"])<=datetime.fromisoformat(earliest),
                       "Review sample was not committed before the first recorded API attempt")
    known={r["chembl_id"]:r for r in comparisons}
    random_rows=[review_payload(known[r["chembl_id"]],"uniform_random_50") for r in plan["selected_pairs"]]
    diagnostic_rows,eligible=diagnostic_selection(comparisons,plan)
    for path,sig in signatures.items():
        p=Path(path);checks.require(p.exists() and (p.stat().st_size,p.stat().st_mtime_ns)==sig,f"Experiment file changed during analysis: {path}")
    checks.require(journal_inventory=={phase:sorted(str(p) for p in (directory/phase/"batches").glob("batch_*.json")) for phase in ("first_pass","second_pass")},"Journal inventory changed during analysis")
    summary={"schema_version":SCHEMA,"created_at_utc":datetime.now(timezone.utc).isoformat(),
        "fixed_experiment_denominator":expected_count,"valid_source_and_provenance":checks.count==0,
        "error_count":checks.count,"errors_first_200":checks.errors,
        "completed_pair_count":outcomes["changed"]+outcomes["unchanged"],"pair_outcome_counts":dict(outcomes),
        "first_pass":first["summary"],"second_pass":second["summary"],"reported_usage_both_passes":dict(combined_usage),
        "runner_total_elapsed_seconds":overall.get("elapsed_seconds"),
        "random_review_count":len(random_rows),"large_edit_diagnostic_count":len(diagnostic_rows),"large_edit_eligible_count":eligible,
        "chemistry_verified":False,"reasoning_content_reviewed_or_reproduced":False,
        "source_file_hashes":source_hashes,"selection_sha256":digest(selection),"audit_selection_sha256":file_hash(plan_path),
        "input_file_sha256":{str(path):file_hash(path) for path in input_paths},
        "analysis_code_sha256":file_hash(Path(__file__)),
        "limitations":["Changed/unchanged and edit distance are editorial behavior, not chemical accuracy or proof of necessary editing.",
            "The second model's self-review is not independent chemical verification.",
            "Reported tokens include retained complete responses, including truncated/invalid attempts; transport-body failures may have unknown unreported billing.",
            "Request-task durations include semaphore queue wait and transport, overlap under concurrency, and are neither isolated model reasoning time nor wall-clock elapsed time when summed.",
            "Random50 keeps failures without replacement. Large-edit diagnostics are selected separately and cannot be pooled into a population error estimate.",
            "One sampled generation/review per molecule does not establish a causal accuracy improvement."]}
    output_dir.mkdir(parents=True,exist_ok=True)
    write_json(output_dir/"summary.json",summary)
    write_json(output_dir/"comparison.json",{"schema_version":SCHEMA,"summary":summary,"comparisons":comparisons})
    fields=["experiment_index","source_index","chembl_id","smiles",*FEATURE_NAMES,"property_vector","first_status","second_status","pair_outcome","first_description_sha256","second_description_sha256","first_description","second_description","edit_metrics"]
    with (output_dir/"comparison.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        for row in comparisons:
            flat={**row,**row["properties"]};writer.writerow({key:csv_cell(flat.get(key)) for key in fields})
    with (output_dir/"comparison.txt").open("w",encoding="utf-8") as handle:
        handle.write(f"Fixed paired experiment: {expected_count} molecules\nSource/provenance valid: {summary['valid_source_and_provenance']}\nText changes are not accuracy scores. Chemical review is pending.\n\n")
        for row in comparisons:
            handle.write(f"[{row['experiment_index']:03d}] {row['chembl_id']} | source {row['source_index']} | {row['pair_outcome']}\nSMILES: {row['smiles']}\nProperties: {json.dumps(row['properties'],ensure_ascii=False)}\nFIRST ({row['first_status']}):\n{row['first_description'] or '[unavailable]'}\nSECOND ({row['second_status']}):\n{row['second_description'] or '[unavailable]'}\n")
            if row["edit_metrics"]:
                handle.write("Sentence changes: "+json.dumps(row["edit_metrics"]["sentence_diff"],ensure_ascii=False)+"\n")
            handle.write("\n")
    review_schema={"schema_version":"paired_chemical_review_candidates/1","selection_only":True,"chemistry_verified":False,
        "claim_followup_assessments":["corrected","persisted","introduced","omitted","uncertain","unnecessary_edit"],
        "required_claim_fields":["category","first_quote","second_quote","assessment","source_graph_evidence","reason"],
        "scope":"Assistant-assisted selected-claim review. Use exact quotes and both source-bound text hashes; no full-molecule certification. Unnecessary edit requires a substantive judgment, not edit distance alone."}
    write_json(output_dir/"random_50_review_candidates.json",{**review_schema,"fixed_random_denominator":len(random_rows),"selected_pairs":random_rows})
    write_json(output_dir/"large_edit_10_diagnostics.json",{**review_schema,"eligible_count":eligible,"selected_count":len(diagnostic_rows),"rule":plan["diagnostic_rule"],"selected_pairs":[review_payload(r,"large_edit_diagnostic") for r in diagnostic_rows]})
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir",type=Path,default=DEFAULT)
    parser.add_argument("--expected-count",type=int,default=500)
    parser.add_argument("--prepare-sample-only",action="store_true")
    parser.add_argument("--bind-final-config-only",action="store_true")
    parser.add_argument("--sample-count",type=int,default=50)
    parser.add_argument("--sample-seed",type=int,default=SAMPLE_SEED)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    if args.bind_final_config_only:
        value=bind_final_configuration(args.experiment_dir,args.expected_count)
        print(json.dumps({k:value[k] for k in ("schema_version","audit_selection_sha256","final_run_fingerprint")}))
        return 0
    if args.prepare_sample_only:
        value=prepare_sample(args.experiment_dir,args.expected_count,args.sample_count,args.sample_seed)
        print(json.dumps({k:value[k] for k in ("schema_version","random_pair_count","seed","prepared_before_any_recorded_api_attempts")}))
        return 0
    report=analyze(args.experiment_dir,args.expected_count,args.output_dir)
    print(json.dumps({k:report[k] for k in ("valid_source_and_provenance","error_count","fixed_experiment_denominator","completed_pair_count","pair_outcome_counts")}))
    return 0 if report["valid_source_and_provenance"] else 1


if __name__=="__main__":
    raise SystemExit(main())
