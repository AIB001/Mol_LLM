#!/usr/bin/env python3
"""Extract review candidates for a fixed paired prompt trial; never grade chemistry.

No model calls, response repairs, frozen-checker changes, or source mutations.
Missing/failed trial rows remain in the selected denominator. This output uses
``candidates``, not ``records``, and is not a reviewed-delivery overlay.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "results/chembl_100000_natural_v6/checkpoints/review_50000"
SCHEMA_VERSION = "local_region_trial_screen/1"
_LOCAL = re.compile(
    r"\b(?:hydrophob(?:ic(?:ally)?|icity)|hydrophil(?:ic(?:ally)?|icity)|"
    r"non[ -]?polar|polar(?:ity)?|lipophil(?:ic(?:ally)?|icity))\b", re.I)
_REGION = re.compile(r"\b(?:cores?|regions?|segments?|moiet(?:y|ies)|clusters?|"
                     r"frameworks?|scaffolds?|domains?|patches?)\b", re.I)
_GROUP = re.compile(r"\b(?:hydroxyls?|alcohols?|phenols?|amides?|carbonyls?|"
    r"carboxylates?|carboxylic\s+acids?|amines?|ethers?|esters?|ureas?|"
    r"carbamates?|sulfonamides?|sulphonamides?|sulfones?|sulphones?|"
    r"sulfonyls?|phosphates?|charged\s+groups?|aromatic\s+(?:rings?|systems?)|"
    r"phenyl\s+(?:rings?|groups?)|hydrocarbon(?:[ -]rich)?|alkyl\s+(?:chains?|groups?)|"
    r"nitro\s+groups?|nitriles?|cyano\s+groups?)\b", re.I)
_BEHAVIOR = re.compile(r"\b(?:solub(?:le|ility)|insolub(?:le|ility)|dissolv(?:e|es|ing)|"
    r"dissolution|permeab(?:le|ility)|bioavailab(?:le|ility)|absorption|"
    r"distribution\s+coefficient|partition\s+coefficient|log\s*[pd]|pka)\b", re.I)
_GLOBAL = re.compile(r"\b(?:molecule|compound|overall|whole[ -]molecule|"
                     r"molecule[ -]wide|globally)\b", re.I)
_CAUSAL = re.compile(r"\b(?:increas\w*|decreas\w*|improv\w*|enhanc\w*|"
                     r"promot\w*|confer\w*|reduc\w*|favor\w*|favour\w*)\b", re.I)
_ROLE = re.compile(r"\b(?:donors?|acceptors?|hydrogen[ -]bond(?:ing|ed)?)\b", re.I)
_NEGATION = re.compile(r"\b(?:no|not|without|cannot|neither|avoids?|"
                       r"does\s+not|do\s+not|unspecified|unknown)\b", re.I)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sentences(text):
    """Keep decimal/locant punctuation intact; sentence boundaries are heuristic."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text or "") if part.strip()]


def normalized(text):
    return " ".join(re.findall(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", text.lower()))


def extract(text):
    extracted = []
    for index, sentence in enumerate(sentences(text), 1):
        local = sorted({match.group().lower() for match in _LOCAL.finditer(sentence)})
        regions = sorted({match.group().lower() for match in _REGION.finditer(sentence)})
        groups = sorted({match.group().lower() for match in _GROUP.finditer(sentence)})
        behavior = sorted({match.group().lower() for match in _BEHAVIOR.finditer(sentence)})
        flags = []
        if local:
            flags.append("local_character_language_for_scope_and_structure_review")
            if not regions:
                flags.append("local_character_without_explicit_region_noun")
            if not groups:
                flags.append("local_character_without_dictionary_group_anchor")
            if _GLOBAL.search(sentence):
                flags.append("global_subject_and_character_language_check_scope")
        if behavior:
            flags.append("physicochemical_behavior_language_check_for_global_inference")
        if _CAUSAL.search(sentence) and (local or behavior):
            flags.append("causal_property_language_check_scope")
        if local and any(word.startswith("core") for word in regions):
            flags.append("core_centrality_requires_source_graph_review")
        if local and any(word.startswith("cluster") for word in regions):
            flags.append("cluster_locality_requires_source_graph_review")
        if _ROLE.search(sentence):
            flags.append("hydrogen_bond_role_requires_atom_and_clause_specific_review")
        if flags:
            extracted.append({"sentence_index": index, "text": sentence,
                "local_character_terms": local, "region_words": regions,
                "group_word_anchors": groups, "behavior_terms": behavior,
                "contains_negation_or_unknown_language": bool(_NEGATION.search(sentence)),
                "candidate_reasons": flags, "assertion_status": "not_determined",
                "chemical_assessment": "not_reviewed"})
    return extracted


def read_rows(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    value = json.loads(path.read_text())
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("annotations", "records"):
            if isinstance(value.get(key), list):
                return value[key]
    raise ValueError("Input must be JSONL, a JSON list, or an annotations/records array")


def index_unique(rows, name):
    indexed = {}
    for row in rows:
        cid = row.get("chembl_id")
        if not isinstance(cid, str) or not cid or cid in indexed:
            raise ValueError(f"{name} has a missing or duplicate chembl_id: {cid}")
        indexed[cid] = row
    return indexed


def checked_description(row, name, required=False):
    text = row.get("description")
    if text is None or text == "":
        if required:
            raise ValueError(f"{name} has no required description")
        return ""
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{name} has an invalid description")
    if row.get("description_sha256") != text_hash(text):
        raise ValueError(f"{name} description hash does not match its text")
    return text


def language_summary(texts, denominator):
    lengths = [len(text.split()) for text in texts]
    term_rows, flags, repeated_sentences, repeated_phrases, openings = (Counter() for _ in range(5))
    local_rows = 0
    for text in texts:
        found = extract(text)
        local_sentences = [item["text"] for item in found if item["local_character_terms"]]
        if local_sentences:
            local_rows += 1
        term_rows.update({term for item in found for term in item["local_character_terms"]})
        flags.update({flag for item in found for flag in item["candidate_reasons"]})
        repeated_sentences.update({normalized(sentence) for sentence in sentences(text)
                                   if len(normalized(sentence).split()) >= 8})
        ngrams = set()
        for sentence in local_sentences:
            words = normalized(sentence).split()
            ngrams.update(" ".join(words[i:i+6]) for i in range(max(0, len(words)-5)))
        repeated_phrases.update(ngrams)
        openings.update([" ".join(normalized(text).split()[:8])])
    def top(counter, minimum, limit=15):
        return [{"phrase": phrase, "distinct_rows": count} for phrase, count in
                sorted(counter.items(), key=lambda item: (-item[1], item[0])) if count >= minimum][:limit]
    return {"fixed_selected_denominator": denominator, "available_text_rows": len(texts),
        "word_count": {"min": min(lengths) if lengths else None,
                       "max": max(lengths) if lengths else None,
                       "mean": sum(lengths)/len(lengths) if lengths else None,
                       "within_100_150": sum(100 <= n <= 150 for n in lengths),
                       "within_95_160": sum(95 <= n <= 160 for n in lengths)},
        "rows_with_local_character_terms": local_rows,
        "literal_term_row_counts": dict(term_rows), "candidate_reason_row_counts": dict(flags),
        "repeated_sentences": top(repeated_sentences, 2),
        "repeated_six_word_local_phrases": top(repeated_phrases, 3),
        "repeated_eight_word_openings": top(openings, 2),
        "counts_are_language_features_not_error_counts": True}


def screen(input_path, selection_path, baseline_review_path, expected_count=100):
    paths = {"trial_input": Path(input_path), "selection": Path(selection_path),
             "baseline_review": Path(baseline_review_path), "screen_code": Path(__file__)}
    initial_hashes = {key: file_hash(path) for key, path in paths.items()}
    selection = json.loads(paths["selection"].read_text())
    selected = selection.get("selected_molecules")
    if not isinstance(selected, list) or len(selected) != expected_count:
        raise ValueError("Selection must retain the exact expected fixed denominator")
    selected_by_id = index_unique(selected, "selection")
    baseline = json.loads(paths["baseline_review"].read_text())
    baseline_by_id = index_unique(baseline.get("records", []), "baseline review")
    if set(baseline_by_id) != set(selected_by_id):
        raise ValueError("Baseline review IDs differ from the fixed selection")
    declared_selection_hash = baseline.get("selection", {}).get("selection_sha256")
    if declared_selection_hash and declared_selection_hash != initial_hashes["selection"]:
        raise ValueError("Baseline review pins a different selection file")
    trial_by_id = index_unique(read_rows(paths["trial_input"]), "trial input")
    if set(trial_by_id) - set(selected_by_id):
        raise ValueError("Trial input includes IDs outside the fixed selection")
    candidates, baseline_texts, completed_texts = [], [], []
    status_counts = Counter()
    for position, selected_row in enumerate(selected, 1):
        cid = selected_row["chembl_id"]
        old = baseline_by_id[cid]
        old_text = checked_description(selected_row, f"selection {cid}", required=True)
        for key in ("smiles", "source_index", "facts_sha256", "description_sha256", "description"):
            if old.get(key) != selected_row.get(key):
                raise ValueError(f"Baseline review/selection mismatch for {cid}: {key}")
        baseline_texts.append(old_text)
        row = trial_by_id.get(cid)
        trial_text, status, trial_index = "", "not_stored", None
        if row is not None:
            for key in ("smiles", "facts_sha256"):
                if row.get(key) != selected_row.get(key):
                    raise ValueError(f"Trial/selection mismatch for {cid}: {key}")
            source_index = row.get("original_source_index", row.get("source_index"))
            if source_index != selected_row["source_index"]:
                raise ValueError(f"Trial/selection mismatch for {cid}: original source index")
            trial_index = row.get("trial_selection_index", row.get("trial_index"))
            if trial_index is not None and trial_index != position:
                raise ValueError(f"Trial/selection mismatch for {cid}: trial index")
            if (row.get("baseline_description_sha256") is not None
                    and row["baseline_description_sha256"] != selected_row["description_sha256"]):
                raise ValueError(f"Trial/selection mismatch for {cid}: baseline description hash")
            if (row.get("trial_selection_sha256") is not None
                    and row["trial_selection_sha256"] != initial_hashes["selection"]):
                raise ValueError(f"Trial/selection mismatch for {cid}: fixed selection hash")
            trial_text = checked_description(row, f"trial {cid}")
            status = row.get("status", "missing_status")
        completed = status == "accepted_format" and bool(trial_text)
        if status == "accepted_format" and not trial_text:
            raise ValueError(f"Accepted trial row has no description: {cid}")
        status_counts[status] += 1
        if completed:
            completed_texts.append(trial_text)
        candidates.append({"selection_position": position, "chembl_id": cid,
            "source_index": selected_row["source_index"], "smiles": selected_row["smiles"],
            "facts_sha256": selected_row["facts_sha256"], "trial_selection_index": trial_index,
            "baseline_description_sha256": selected_row["description_sha256"],
            "trial_description_sha256": text_hash(trial_text) if trial_text else None,
            "trial_status": status, "trial_completed": completed,
            "baseline_word_count": len(old_text.split()),
            "trial_word_count": len(trial_text.split()) if trial_text else None,
            "trial_description": trial_text or None,
            "baseline_local_candidates": extract(old_text),
            "trial_local_candidates": extract(trial_text),
            "baseline_known_unsupported_claims": old.get("unsupported_claims", []),
            "baseline_uncertain_claims": old.get("uncertain_claims", []),
            "baseline_issue_followup_status": "not_reviewed" if old.get("unsupported_claims") else "not_applicable",
            "local_claims_review_status": "not_reviewed", "chemistry_verified": False})
    if initial_hashes != {key: file_hash(path) for key, path in paths.items()}:
        raise ValueError("Input or screen code changed during extraction")
    completed_count = len(completed_texts)
    return {"schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "screening_only": True, "candidate_identity_checks_passed": True,
        "chemistry_verified": False, "confirmed_chemical_error_count": None,
        "reasoning_content_read_or_reproduced": False,
        "fixed_selected_count": expected_count, "stored_trial_count": len(trial_by_id),
        "completed_trial_count": completed_count,
        "missing_or_not_accepted_count": expected_count - completed_count,
        "all_selected_trial_descriptions_completed": completed_count == expected_count,
        "trial_status_counts": dict(status_counts),
        "baseline_rows_with_known_unsupported_claims": sum(bool(row["baseline_known_unsupported_claims"]) for row in candidates),
        "inputs": {key: {"path": str(path.resolve()), "sha256": initial_hashes[key]} for key, path in paths.items()},
        "baseline_language": language_summary(baseline_texts, expected_count),
        "completed_trial_language": language_summary(completed_texts, expected_count),
        "review_rubric": {
            "local_region_claims": {"required_fields": ["claim", "assessment", "source_evidence", "reason"],
                "assessment_values": ["supported", "unsupported", "uncertain"],
                "rule": "Anchor qualitative local character to a specific source subgraph; check group identity, locality/cluster and core centrality. Allowed local language is not itself a defect."},
            "baseline_issue_followup": {"required_fields": ["baseline_claim", "status", "trial_claim", "evidence"],
                "status_values": ["corrected", "persists", "omitted", "uncertain"],
                "rule": "Corrected requires an explicit accurate replacement; a removed assertion is omitted, not proof of corrected understanding."},
            "global_inference": "Review solubility/lipophilicity/permeability and solution-state extrapolation separately. Keyword matches and negated discussions are candidates, never automatic chemistry failures.",
            "structural_information": "Record any important lost structural information separately from errors; unmentioned atoms are not automatically errors.",
            "hydrogen_bond_roles": "Resolve the clause's actual atom and its attached H, charge, aromatic and acylation environment; co-occurring donor/acceptor words do not assign a role."},
        "limitations": [
            "Regex extraction is incomplete and may include negated, allowed or unrelated wording; candidate counts are not confirmed errors.",
            "No source-graph polarity calculation or automatic donor/acceptor adjudication is performed.",
            "The baseline issue list reflects the earlier assistant-assisted review, not human expert certification or proof that unflagged text is correct.",
            "This single paired run has no repeated generations or randomized concurrent control; sampling, changed batch neighbors and backend/time effects prevent causal attribution.",
            "New local-character coverage is not a success target for every molecule and does not establish improved chemical accuracy.",
            "Only final descriptions and metadata are read; this file must not be passed as a completed chemical-review overlay."],
        "candidates": candidates}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=CHECKPOINT / "samples/random_100_selection.json")
    parser.add_argument("--baseline-review", type=Path, default=CHECKPOINT / "random_100_chemical_review.json")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() in {p.resolve() for p in (args.input, args.selection, args.baseline_review)}:
        raise ValueError("Output must not replace an input")
    report = screen(args.input, args.selection, args.baseline_review, args.expected_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ("schema_version", "fixed_selected_count", "stored_trial_count",
        "completed_trial_count", "missing_or_not_accepted_count", "baseline_rows_with_known_unsupported_claims", "chemistry_verified")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
