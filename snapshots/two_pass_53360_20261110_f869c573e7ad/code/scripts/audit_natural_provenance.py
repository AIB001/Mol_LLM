#!/usr/bin/env python3
"""Audit natural-description API journals after the generator has fully drained.

No keys, network requests, generator imports or production mutations are used.
Rows may select different successful attempts within one partially retried batch.
This proves saved-journal consistency, not chemical correctness or server billing.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODE = "deepseek_free_natural_language_from_smiles"
GROUNDED_MODE = "deepseek_free_natural_language_from_smiles_and_verified_context"
FEATURE_NAMES = ["mol_wt", "exact_mol_wt", "crippen_logp", "tpsa", "hbd", "hba", "rotatable_bonds",
    "ring_count", "aromatic_rings", "aliphatic_rings", "saturated_rings", "heterocycles",
    "aromatic_heterocycles", "fraction_csp3", "heavy_atom_count", "heteroatom_count", "formal_charge",
    "positive_charge_centers", "negative_charge_centers", "molar_refractivity", "amide_bonds",
    "spiro_atoms", "bridgehead_atoms", "chiral_centers"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def integer_usage(value):
    return {key: number for key, number in value.items() if type(number) is int} if isinstance(value, dict) else {}


def parse_annotations(response, count):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Invalid API choices envelope")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Selected response did not finish normally")
    content = choice.get("message", {}).get("content")
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"annotations"} or not isinstance(value["annotations"], list):
        raise ValueError("Invalid annotations response schema")
    rows = value["annotations"]
    if len(rows) != count or any(not isinstance(row, dict) or set(row) != {"key", "description"}
                                 or not isinstance(row["key"], str) for row in rows):
        raise ValueError("Invalid response entry fields/count")
    keys = [row["key"] for row in rows]
    if len(set(keys)) != count or set(keys) != {str(index) for index in range(1, count + 1)}:
        raise ValueError("Missing/duplicate/unexpected response key")
    # Other rows may fail their individual checks in a partially accepted response.
    return {row["key"]: row["description"] for row in rows}


def request_digest(config, records, previous_errors=None, contexts=None):
    content = {"molecules": [{"key": str(index), "chembl_id": row["chembl_id"], "smiles": row["smiles"]}
                              for index, row in enumerate(records, 1)]}
    if config.get("generation_mode") == GROUNDED_MODE:
        if contexts is None:
            raise ValueError("Grounded request verification requires rebuilt source context")
        for row in content["molecules"]:
            row["structural_context"] = contexts[row["chembl_id"]]
    if previous_errors:
        content["previous_format_errors"] = previous_errors
    payload = {"model": config["model"], "messages": [
        {"role": "system", "content": config["system_prompt"]},
        {"role": "user", "content": json.dumps(content, ensure_ascii=False, separators=(",", ":"))}],
        "response_format": {"type": "json_object"}, "thinking": {"type": config["thinking"]},
        "max_tokens": config["max_tokens"], "stream": False}
    if config["thinking"] == "enabled":
        payload["reasoning_effort"] = "low"
    else:
        payload["temperature"] = config["temperature"]
    return digest(payload)


def audit(directory, expected_molecules=10000, expected_model="deepseek-v4-pro", *,
          molecules=None, properties=None, allow_incomplete=False):
    directory = Path(directory)
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    errors, warnings, error_count = [], [], 0

    def require(condition, message):
        nonlocal error_count
        if not condition:
            error_count += 1
            if len(errors) < 100:
                errors.append(message)

    fingerprint, prompt_hash = config.get("run_fingerprint"), config.get("prompt_sha256")
    require(fingerprint == digest({key: value for key, value in config.items() if key != "run_fingerprint"}),
            "Configuration fingerprint does not match its contents")
    generation_mode = config.get("generation_mode")
    require(generation_mode in {MODE, GROUNDED_MODE}, "Generation mode mismatch")
    context_builder = None
    context_verification = {"present": generation_mode == GROUNDED_MODE}
    if generation_mode == GROUNDED_MODE:
        import natural_structure_context as context_module
        context_builder = context_module.natural_structure_context
        require(config.get("structure_context_schema_version") == context_module.SCHEMA_VERSION,
                "Structure-context schema mismatch")
        for name, path in {
            "structure_context_code_sha256": Path(context_module.__file__),
            "readable_fact_code_sha256": ROOT / "scripts/readable_facts_v3.py",
            "grounding_wrapper_code_sha256": ROOT / "scripts/annotate_natural_grounded.py",
        }.items():
            current_hash = file_hash(path)
            context_verification[name] = current_hash
            require(config.get(name) == current_hash, f"Grounding code hash mismatch: {name}")
        require(config.get("model_input_fields") == ["key", "chembl_id", "smiles", "structural_context"],
                "Grounded model input field contract mismatch")
    require(config.get("model") == expected_model, "Requested model mismatch")
    require(config.get("batch_size") == 5 and config.get("round_size") == 1000,
            "Batch/round geometry differs from the natural pilot")
    require(config.get("dataset_size") == 10000 and 0 < expected_molecules <= 10000,
            "Natural pilot dataset/expected size mismatch")
    require(config.get("feature_names") == FEATURE_NAMES, "Feature order differs from the 24-property contract")
    require(hashlib.sha256(config.get("system_prompt", "").encode()).hexdigest() == prompt_hash,
            "Transmitted system-prompt hash mismatch")
    prompt_path = Path(config["prompt_path"])
    require(prompt_path.read_text(encoding="utf-8").strip() == config.get("system_prompt"),
            "Prompt file differs from the exact transmitted prompt")
    sources = {"input_sha256": Path(config["input_path"]),
               "molecules_sha256": Path(molecules) if molecules else ROOT / "data/processed/chembl_100000_molecules.csv",
               "properties_sha256": Path(properties) if properties else ROOT / "data/processed/chembl_100000_properties.csv"}
    source_hashes = {}
    for name, path in sources.items():
        source_hashes[name] = file_hash(path)
        require(source_hashes[name] == config.get(name), f"Source file hash mismatch: {name}")

    source_rows, contexts = {}, {}
    opener = gzip.open if sources["input_sha256"].suffix == ".gz" else open
    with opener(sources["input_sha256"], "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            if index > expected_molecules:
                break
            row = json.loads(line)
            require(row.get("facts_sha256") == digest(row["structural_facts"]), f"Source row {index}: facts digest mismatch")
            source_rows[index] = {key: row[key] for key in ("chembl_id", "smiles", "facts_sha256")}
            if context_builder is not None:
                contexts[row["chembl_id"]] = context_builder(row["structural_facts"])
    require(len(source_rows) == expected_molecules, "Insufficient source rows")

    usage, by_account = Counter(), defaultdict(Counter)
    statuses, row_statuses, http_statuses, paid_by_account, selected_counts = (Counter() for _ in range(5))
    batch_ids, ids, indices, response_ids = set(), set(), set(), set()
    expected_batches = math.ceil(expected_molecules / 5)
    http_attempts = api_responses = paid_calls = accepted_rows = 0
    for path in sorted((directory / "batches").glob("batch_*.json")):
        prefix = path.name
        try:
            batch = json.loads(path.read_text(encoding="utf-8"))
            bid = batch["batch_id"]
            require(type(bid) is int and 1 <= bid <= expected_batches, f"{prefix}: unexpected batch ID")
            require(path.name == f"batch_{bid:06d}.json" and bid not in batch_ids, f"{prefix}: filename/duplicate ID error")
            batch_ids.add(bid)
            rid, count = (bid - 1) // 200 + 1, min(5, expected_molecules - (bid - 1) * 5)
            require(batch.get("round_id") == rid, f"{prefix}: incorrect round ID")
            statuses[batch.get("status", "missing")] += 1
            require(batch.get("run_fingerprint") == fingerprint and batch.get("generation_mode") == generation_mode,
                    f"{prefix}: fingerprint/generation mode mismatch")
            require(batch.get("model_requested") == expected_model, f"{prefix}: requested model mismatch")
            rows, attempts = batch["records"], batch.get("attempt_log", [])
            require(len(rows) == count, f"{prefix}: incorrect record count")
            row_ids = [row.get("chembl_id") for row in rows]
            require(batch.get("chembl_ids") == row_ids, f"{prefix}: recorded identifier order mismatch")
            local_usage, local_accounts, issues_by_id = Counter(), defaultdict(Counter), {}
            parsed = {}
            for position, attempt in enumerate(attempts):
                label = f"{prefix}: attempt {position}"
                account, code = attempt.get("account"), attempt.get("http_status", "transport")
                require(isinstance(account, str) and account.startswith("account_")
                        and account[8:].isdigit() and int(account[8:]) > 0,
                        f"{label}: invalid account label")
                http_attempts += 1
                http_statuses[f"{account}:{code}"] += 1
                au = attempt.get("usage", {})
                require(isinstance(au, dict) and all(type(number) is int and number >= 0 for number in au.values()),
                        f"{label}: invalid usage")
                local_usage.update(au)
                local_accounts[account].update(au)
                requested_ids = attempt.get("requested_ids", [])
                require(isinstance(requested_ids, list) and requested_ids
                        and len(set(requested_ids)) == len(requested_ids)
                        and requested_ids == [cid for cid in row_ids if cid in requested_ids],
                        f"{label}: request key-to-source-ID mapping mismatch")
                requested_rows = [{row["chembl_id"]: row for row in rows}[cid] for cid in requested_ids]
                requested_count = len(requested_ids)
                require(attempt.get("requested_keys") == [str(index) for index in range(1, requested_count + 1)],
                        f"{label}: request-key order mismatch")
                pass_number = attempt.get("pass_number")
                require(type(pass_number) is int and pass_number >= 1, f"{label}: invalid pass number")
                prior_errors = {str(index): issues_by_id[cid] for index, cid in enumerate(requested_ids, 1)
                                if issues_by_id.get(cid)}
                # A resumed annotate_batch starts with no previous errors. Within
                # an invocation, the prior successful pass supplies the errors.
                valid_request_hashes = {request_digest(config, requested_rows, contexts=contexts),
                    request_digest(config, requested_rows, prior_errors, contexts=contexts)}
                require(attempt.get("request_sha256") in valid_request_hashes, f"{label}: reconstructed request hash mismatch")
                response = attempt.get("response")
                require(code != 200 or isinstance(response, dict),
                        f"{label}: HTTP 200 has no retained raw response/usage evidence")
                if isinstance(response, dict):
                    api_responses += 1
                    require(code == 200 and attempt.get("error") is None, f"{label}: raw response lacks successful HTTP evidence")
                    require(response.get("model") == expected_model and attempt.get("model_returned") == expected_model,
                            f"{label}: actual returned model mismatch")
                    response_id = response.get("id")
                    require(isinstance(response_id, str) and response_id and response_id not in response_ids,
                            f"{label}: missing or duplicated raw response ID")
                    response_ids.add(response_id)
                    require(attempt.get("response_id") == response_id, f"{label}: raw/log response ID mismatch")
                    require(au == integer_usage(response.get("usage")), f"{label}: raw/log usage mismatch")
                    require(all(type(au.get(key)) is int and au[key] >= 0 for key in
                                ("prompt_tokens", "completion_tokens", "total_tokens")), f"{label}: token accounting is incomplete")
                    require(au.get("total_tokens") == au.get("prompt_tokens", 0) + au.get("completion_tokens", 0),
                            f"{label}: token total differs from prompt plus completion")
                    choices = response.get("choices", [])
                    content = choices[0].get("message", {}).get("content") if choices and isinstance(choices[0], dict) else None
                    require(attempt.get("response_content") == content, f"{label}: raw/log response-content mismatch")
                    validation_errors = attempt.get("validation_errors", {})
                    for local_index, cid in enumerate(requested_ids, 1):
                        if validation_errors.get(str(local_index)):
                            issues_by_id[cid] = validation_errors[str(local_index)]
                        else:
                            issues_by_id.pop(cid, None)
                    try:
                        parsed[position] = parse_annotations(response, requested_count)
                    except (ValueError, TypeError, KeyError, AttributeError):
                        # Paid malformed/truncated attempts remain in accounting,
                        # but cannot be selected as the source of an accepted row.
                        pass
                if au.get("total_tokens", 0) > 0:
                    paid_calls += 1
                    paid_by_account[account] += 1
                    require(isinstance(response, dict), f"{label}: billed usage lacks raw response")
            require(batch.get("usage") == dict(local_usage), f"{prefix}: batch usage sum mismatch")
            require(batch.get("usage_by_account") == {key: dict(value) for key, value in local_accounts.items()},
                    f"{prefix}: batch account usage sum mismatch")
            usage.update(local_usage)
            for account, values in local_accounts.items():
                by_account[account].update(values)

            for offset, row in enumerate(rows, 1):
                label, index = f"{prefix}: row {offset}", (bid - 1) * 5 + offset
                cid = row.get("chembl_id")
                require(isinstance(cid, str) and cid not in ids, f"{label}: missing/duplicate identifier")
                ids.add(cid)
                require(row.get("source_index") == index and index not in indices, f"{label}: source order/index mismatch")
                indices.add(index)
                require(all(row.get(key) == value for key, value in source_rows.get(index, {}).items())
                        and index in source_rows, f"{label}: identity/SMILES/facts differ from source row")
                require(row.get("batch_id") == bid and row.get("round_id") == rid, f"{label}: batch/round mismatch")
                require(row.get("run_fingerprint") == fingerprint and row.get("prompt_sha256") == prompt_hash,
                        f"{label}: configuration/prompt hash mismatch")
                require(row.get("generation_mode") == generation_mode and row.get("model_requested") == expected_model,
                        f"{label}: generation mode/requested model mismatch")
                evidence = row.get("structural_facts_source", {})
                require(evidence == {"path": config["input_path"], "sha256": config["input_sha256"], "line": index},
                        f"{label}: source evidence pointer mismatch")
                props, vector = row.get("properties", {}), row.get("property_vector")
                require(set(props) == set(FEATURE_NAMES) and vector == [props.get(name) for name in FEATURE_NAMES],
                        f"{label}: internal 24-property/vector contract mismatch")
                require(all(type(value) in (int, float) and math.isfinite(value) for value in props.values()),
                        f"{label}: nonfinite/nonnumeric property")
                require(row.get("chemistry_verified") is False, f"{label}: unsupported chemistry-verified claim")
                row_statuses[row.get("status", "missing")] += 1
                if row.get("status") != "accepted_format":
                    require(batch.get("status") != "completed", f"{label}: completed batch contains unaccepted row")
                    continue
                accepted_rows += 1
                selected = row.get("selected_attempt_index")
                require(type(selected) is int and 0 <= selected < len(attempts), f"{label}: invalid selected attempt index")
                if type(selected) is not int or not 0 <= selected < len(attempts):
                    continue
                attempt = attempts[selected]
                selected_counts[str(selected)] += 1
                require(row.get("selected_pass_number") == attempt.get("pass_number"), f"{label}: selected pass mismatch")
                require(row.get("response_id") == attempt.get("response_id"), f"{label}: selected response ID mismatch")
                require(row.get("account") == attempt.get("account"), f"{label}: selected account mismatch")
                require(row.get("model") == expected_model and row.get("model") == attempt.get("model_returned"),
                        f"{label}: selected actual model mismatch")
                selected_ids = attempt.get("requested_ids", [])
                require(cid in selected_ids, f"{label}: molecule absent from selected request")
                key = str(selected_ids.index(cid) + 1) if cid in selected_ids else "missing"
                require(selected in parsed and key in parsed[selected], f"{label}: selected attempt is not a complete keyed response")
                require(not attempt.get("validation_errors", {}).get(key), f"{label}: selected key had validation errors")
                description = row.get("description")
                require(isinstance(description, str) and description and description == parsed.get(selected, {}).get(key),
                        f"{label}: saved description differs from selected raw API text")
                if isinstance(description, str):
                    require(hashlib.sha256(description.encode()).hexdigest() == row.get("description_sha256"),
                            f"{label}: description hash mismatch")

        except (ValueError, KeyError, TypeError, AttributeError, IndexError) as exc:
            require(False, f"{prefix}: unreadable/inconsistent journal: {type(exc).__name__}: {exc}")

    complete = (batch_ids == set(range(1, expected_batches + 1)) and len(ids) == expected_molecules
                and indices == set(range(1, expected_molecules + 1)) and accepted_rows == expected_molecules
                and statuses.get("completed", 0) == expected_batches)
    if not allow_incomplete:
        require(complete, "Corpus is incomplete")
    elif not complete:
        warnings.append("Incomplete snapshot allowed; not a final corpus validation")
    summary_path = directory / "summary.json"
    summary_matches = False
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_matches = (summary.get("usage") == dict(usage) and summary.get("stored_molecules") == len(ids)
            and summary.get("finished_batches") == statuses.get("completed", 0)
            and summary.get("status_counts") == dict(row_statuses)
            and summary.get("target_molecules") == expected_molecules)
        require(summary.get("chemistry_verified") is False, "Summary incorrectly asserts verified chemistry")
        if not allow_incomplete:
            require(summary.get("in_flight_batches") == 0 and summary.get("all_descriptions_present") is True,
                    "Final summary is not drained/complete")
        if not summary_matches and allow_incomplete:
            warnings.append("Summary differs from this snapshot; rerun after draining")
        else:
            require(summary_matches, "Summary counts/usage differ from authoritative journals")
    else:
        require(allow_incomplete, "Final summary is absent")
    return {"schema_version": "natural_provenance_audit/1", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "directory": str(directory.resolve()), "expected_molecules": expected_molecules,
        "expected_batches": expected_batches, "expected_model": expected_model,
        "generation_mode": generation_mode, "run_fingerprint": fingerprint, "prompt_sha256": prompt_hash,
        "structure_context_verification": context_verification,
        "source_file_hashes": source_hashes, "batch_status_counts": dict(statuses), "record_status_counts": dict(row_statuses),
        "accepted_molecules": accepted_rows, "http_attempts": http_attempts, "http_status_counts": dict(http_statuses),
        "successful_api_responses": api_responses, "paid_calls_with_reported_tokens": paid_calls,
        "paid_calls_by_account": dict(paid_by_account), "usage": dict(usage),
        "usage_by_account": {key: dict(value) for key, value in by_account.items()},
        "selected_attempt_index_counts": dict(selected_counts), "summary_matches_journals": summary_matches,
        "complete": complete, "valid": error_count == 0, "error_count": error_count, "errors_first_100": errors,
        "allow_incomplete": allow_incomplete, "warnings": warnings, "chemistry_verified": False,
        "validation_scope": "Saved raw API provenance, selected-attempt mapping, source identity/hash, internal property/vector contract and reported token accounting; not chemical correctness",
        "separate_required_audit": "audit_natural_descriptions.py checks CSV/JSONL and source properties row alignment plus description screening",
        "request_hash_boundary": "Reconstructed request from source identities/SMILES and, when configured, deterministic source-fact context; prior pass errors or empty resume errors are accepted because invocation boundaries are not independently journaled",
        "billing_boundary": "Reported tokens in saved responses, not an independent provider invoice or unseen/lost responses"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/chembl_10000_natural_language")
    parser.add_argument("--expected-molecules", type=int, default=10000)
    parser.add_argument("--expected-model", default="deepseek-v4-pro")
    parser.add_argument("--molecules", type=Path)
    parser.add_argument("--properties", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.expected_molecules <= 10000:
        parser.error("expected-molecules must be between 1 and 10000")
    report = audit(args.output_dir, args.expected_molecules, args.expected_model,
                   molecules=args.molecules, properties=args.properties, allow_incomplete=args.allow_incomplete)
    target = args.report or args.output_dir / "natural_provenance_audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    print(json.dumps({key: report[key] for key in ("valid", "complete", "accepted_molecules", "successful_api_responses", "error_count")}, ensure_ascii=False))
    print(f"Report: {target}")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
