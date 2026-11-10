"""Independent minimal-edit second pass; never replaces its first-pass baseline.

Only explicit output-format/content restrictions can trigger repair. Word count,
style and structural screening flags never request a rewrite. Returned strings
are preserved exactly, including punctuation, Unicode and surrounding spaces.
"""
from __future__ import annotations

from collections import Counter
import difflib
import hashlib
import json
import re

import annotate_natural_pilot as core
import natural_description_checks as screens
from natural_structure_context import natural_structure_context

SCHEMA_VERSION = "minimal_annotation_review/1"
GENERATION_MODE = "deepseek_minimal_review_of_first_pass_annotation"


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_first(items, records):
    if len(items) != len(records) or not items:
        raise ValueError("Review input has a missing first-pass record")
    for item, row in zip(items, records):
        if not core.row_accepted(row) or any(row.get(key) != item.get(key) for key in
                ("chembl_id", "smiles", "source_index", "facts_sha256", "properties", "property_vector")):
            raise ValueError("Review input identity/properties do not match the accepted first pass")
        if text_hash(row["description"]) != row.get("description_sha256"):
            raise ValueError("First-pass description hash mismatch")


def review_messages(items, first_records, config, errors=None):
    verify_first(items, first_records)
    molecules, annotations = [], []
    for index, (item, original) in enumerate(zip(items, first_records), 1):
        context = natural_structure_context(item["structural_facts"])
        if core.digest(item["structural_facts"]) != item["facts_sha256"]:
            raise ValueError("Review source structural facts hash mismatch")
        molecules.append({"key": str(index), "chembl_id": item["chembl_id"], "smiles": item["smiles"],
                          "structural_context": context, "properties": item["properties"]})
        annotations.append({"key": str(index), "description": original["description"]})
    value = {"molecules": molecules, "annotations": annotations}
    if errors:
        value["previous_format_errors"] = errors
    return [{"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))}]


def check_review_description(text, facts):
    errors, flags = [], []
    if not isinstance(text, str) or not text.strip():
        errors.append("missing_description")
        return {"format_errors": errors, "review_reasons": [], "word_count": 0,
                "word_count_in_target": False, "chemistry_verified": False}
    if "\n" in text or "\r" in text:
        errors.append("description_not_single_paragraph")
    if "```" in text or re.match(r"\s*(?:#{1,6}\s|[-*]\s|\d+[.)]\s)", text):
        errors.append("markdown_or_list_in_description")
    if screens._IDENTIFIER.search(text):
        errors.append("identifier_in_description")
    if screens._CONTROLLED_LABEL.search(text):
        errors.append("implementation_notation_in_description")
    errors.extend(screens._property_number_errors(text))
    if screens._ACTIVITY_ADMET.search(text):
        errors.append("activity_or_admet_claim_or_discussion")
    flags.extend(screens._structural_review(text, facts))
    count = len(text.split())
    return {"format_errors": sorted(set(errors)), "review_reasons": sorted(set(flags)),
            "word_count": count, "word_count_in_target": 100 <= count <= 150,
            "chemistry_verified": False,
            "validation_scope": "Explicit format/content restrictions only; word count, style and chemistry flags never trigger repair"}


def validate_review_response(value, items):
    expected = [str(i) for i in range(1, len(items) + 1)]
    def failure(reason):
        return {}, {key: [reason] for key in expected}
    if not isinstance(value, dict) or set(value) != {"annotations"} or not isinstance(value["annotations"], list):
        return failure("invalid_review_json_schema")
    rows = value["annotations"]
    if len(rows) != len(items) or any(not isinstance(row, dict) or set(row) != {"key", "description"}
            or not isinstance(row["key"], str) or not isinstance(row["description"], str) for row in rows):
        return failure("invalid_review_annotation_fields")
    keys = [row["key"] for row in rows]
    if keys != expected:
        return failure("missing_duplicate_unexpected_or_out_of_order_key")
    mapped, errors = {}, {}
    for key, row, item in zip(expected, rows, items):
        checked = check_review_description(row["description"], item["structural_facts"])
        mapped[key] = {**row, "checks": checked}
        if checked["format_errors"]:
            errors[key] = checked["format_errors"]
    return mapped, errors


def text_diff(original, reviewed):
    return {"changed": reviewed != original,
            "character_similarity": difflib.SequenceMatcher(None, original, reviewed, autojunk=False).ratio(),
            "unified_diff": "\n".join(difflib.unified_diff([original], [reviewed], fromfile="first_pass", tofile="second_pass", lineterm=""))}


def check_saved_review(previous, items, first_batch, config):
    if previous is None:
        return False
    first_rows = first_batch["records"]
    verify_first(items, first_rows)
    expected_hash = core.digest(first_rows)
    if previous.get("run_fingerprint") != config["run_fingerprint"] or previous.get("first_batch_records_sha256") != expected_hash:
        raise ValueError("Saved review belongs to another configuration or first-pass batch")
    rows = previous.get("records", [])
    if len(rows) != len(items):
        raise ValueError("Saved review record count mismatch")
    for row, original, item in zip(rows, first_rows, items):
        if any(row.get(key) != item.get(key) for key in ("chembl_id", "smiles", "source_index", "facts_sha256", "properties", "property_vector")):
            raise ValueError("Saved review identity/property mismatch")
        if row.get("first_description") != original["description"] or row.get("first_description_sha256") != original["description_sha256"]:
            raise ValueError("Saved review baseline text changed")
        if row.get("first_batch_records_sha256") != expected_hash:
            raise ValueError("Saved review record batch binding mismatch")
        if core.row_accepted(row):
            if row.get("description_sha256") != text_hash(row["description"]):
                raise ValueError("Saved accepted review description hash mismatch")
            if row.get("text_changed") is not (row["description"] != original["description"]):
                raise ValueError("Saved accepted review exact-change flag mismatch")
            if row.get("text_diff") != text_diff(original["description"], row["description"]):
                raise ValueError("Saved accepted review diff mismatch")
    return all(core.row_accepted(row) for row in rows)


async def review_batch(bid, rid, items, first_batch, previous, client, writer, config, args):
    first_rows = first_batch["records"]
    verify_first(items, first_rows)
    if check_saved_review(previous, items, first_batch, config):
        return previous
    first_hash = core.digest(first_rows)
    old = (previous or {}).get("records", [])
    records = []
    for i, (item, original) in enumerate(zip(items, first_rows)):
        if i < len(old) and core.row_accepted(old[i]):
            records.append(old[i])
            continue
        row = core.base_record(item, bid, rid, config)
        row.update(schema_version=SCHEMA_VERSION, generation_mode=GENERATION_MODE,
                   validation_scope="Output format/content restrictions only; model review is not independent chemistry verification",
                   first_description=original["description"], first_description_sha256=original["description_sha256"],
                   first_pass_run_fingerprint=original["run_fingerprint"], first_response_id=original.get("response_id"),
                   first_batch_records_sha256=first_hash, text_changed=None, text_diff=None)
        records.append(row)
    attempts, revision = list((previous or {}).get("attempt_log", [])), (previous or {}).get("revision", 0)

    async def save(status):
        nonlocal revision
        revision += 1
        usage, accounts = Counter(), {}
        for event in attempts:
            usage.update(event.get("usage", {}))
            accounts.setdefault(event["account"], Counter()).update(event.get("usage", {}))
        result = {"schema_version": SCHEMA_VERSION, "generation_mode": GENERATION_MODE,
                  "batch_id": bid, "round_id": rid, "status": "completed" if all(map(core.row_accepted, records)) else status,
                  "revision": revision, "run_fingerprint": config["run_fingerprint"], "records": records,
                  "chembl_ids": [item["chembl_id"] for item in items], "attempt_log": attempts,
                  "usage": dict(usage), "usage_by_account": {key: dict(value) for key, value in accounts.items()},
                  "first_batch_records_sha256": first_hash, "model_requested": config["model"], "updated_at_utc": core.utc_now()}
        await writer.call("persist", result)
        return result

    for _ in range(args.review_passes):
        if client.stop.is_set():
            return await save("paused")
        pending = [i for i, row in enumerate(records) if not core.row_accepted(row)]
        subset, baselines = [items[i] for i in pending], [first_rows[i] for i in pending]
        errors = {str(local): records[i]["issues"] for local, i in enumerate(pending, 1) if records[i]["issues"]}
        pass_number = max((event.get("pass_number", 0) for event in attempts), default=0) + 1
        messages = review_messages(subset, baselines, config, errors)
        response, logs, error = await client.request(messages, bid, pass_number)
        for event in logs:
            event.update(requested_ids=[item["chembl_id"] for item in subset],
                         requested_keys=[str(i) for i in range(1, len(subset) + 1)],
                         first_description_sha256_by_id={row["chembl_id"]: row["description_sha256"] for row in baselines})
        attempts.extend(logs)
        if error:
            for i in pending:
                records[i]["issues"] = [error]
            return await save("paused" if client.stop.is_set() else "request_failed")
        if not logs:
            raise RuntimeError("Review response lacks request provenance")
        choices = response.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        try:
            mapped, errors = validate_review_response(json.loads(content or ""), subset)
        except (ValueError, TypeError):
            mapped, errors = {}, {str(i): ["invalid_or_empty_json"] for i in range(1, len(subset) + 1)}
        if choice.get("finish_reason") != "stop":
            for i in range(1, len(subset) + 1):
                errors.setdefault(str(i), []).append("incomplete_generation")
        attempts[-1].update(response=response, response_content=content, validation_errors=errors)
        for local, position in enumerate(pending, 1):
            row = records[position]
            if str(local) in errors:
                row["issues"] = errors[str(local)]
                continue
            result = mapped[str(local)]
            text, checked = result["description"], result["checks"]
            reasons = checked["review_reasons"]
            row.update(description=text, description_sha256=text_hash(text), status="accepted_format", issues=[],
                       word_count=checked["word_count"], word_count_in_target=checked["word_count_in_target"], editorial_hints=[],
                       review_reasons=reasons, chemical_review_status="needs_review" if reasons else "not_independently_reviewed",
                       automated_checks=checked, chemistry_verified=False, model=response.get("model"),
                       account=attempts[-1]["account"], response_id=response.get("id"), selected_pass_number=pass_number,
                       selected_attempt_index=len(attempts) - 1, text_changed=text != first_rows[position]["description"],
                       text_diff=text_diff(first_rows[position]["description"], text))
        result = await save("validation_failed")
        if result["status"] == "completed":
            return result
    return await save("validation_failed")
