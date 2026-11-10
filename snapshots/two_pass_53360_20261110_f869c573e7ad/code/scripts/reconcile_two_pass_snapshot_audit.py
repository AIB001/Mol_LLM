#!/usr/bin/env python3
"""Recheck an explicitly failed-receipt classification against a pinned audit.

The complete prior audit and all original evidence bytes remain immutable.
Only unselected HTTP200 invalid_http_json attempts with no response/usage may
be reclassified. Every original input/export SHA and journal inventory is
checked again; any unrelated error or change blocks reconciliation.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import verify_two_pass_full as verifier
from audit_natural_provenance import file_hash

ERROR = re.compile(r"new/(first_pass|second_pass)/(\d+)/attempt(\d+): successful body missing without transport failure\Z")


def require(value, message):
    if not value: raise ValueError(message)


def reconcile(run_dir, base_report, expected_base_sha, prior_verifier, expected_prior_sha, expected_events=52):
    run_dir, base_report, prior_verifier = Path(run_dir).resolve(), Path(base_report).resolve(), Path(prior_verifier).resolve()
    require(file_hash(base_report) == expected_base_sha, "Base audit hash differs from the pinned failed report")
    require(file_hash(prior_verifier) == expected_prior_sha, "Prior verifier archive hash mismatch")
    before = verifier.inventory(run_dir)
    base = verifier.read_json(base_report)
    require(base.get("schema_version") == verifier.SCHEMA and base.get("valid") is False
            and base.get("valid_source_and_provenance") is False and base.get("publish_ready") is False,
            "Base must be the original failed, incomplete full-source audit")
    require(type(expected_events) is int and expected_events > 0 and base.get("error_count") == expected_events
            and len(base.get("errors_first_200", [])) == expected_events,
            "Every original error must be visible and exactly match the expected classification count")
    target=verifier.TARGET
    require(base.get("target_count") == base.get("source_count") == target
            and 0 < base.get("complete_pairs",0) < target
            and base.get("second_pass_accepted") == base["complete_pairs"]
            and base["complete_pairs"] <= base.get("first_pass_accepted",0) <= target
            and base.get("missing_pairs") == target-base["complete_pairs"]
            and base.get("chemistry_verified") is False,
            "Base source/count/scope contract is invalid")
    config=verifier.read_json(run_dir/"run_config.json")
    require(config.get("run_fingerprint") == base.get("run_fingerprint"), "Run fingerprint changed")
    hashes={str(Path(path).resolve()):sha for path,sha in base.get("input_hashes",{}).items()}
    exports=base.get("data_exports",[])
    required_exports={str(run_dir/phase/("annotations."+suffix)) for phase in ("first_pass","second_pass","paired") for suffix in ("jsonl","json","csv")}
    require({str(Path(e["path"]).resolve()) for e in exports} == required_exports, "Base does not bind all nine exports")
    for entry in exports: hashes[str(Path(entry["path"]).resolve())]=entry["sha256"]
    require(str(run_dir/"run_config.json") in hashes and str(run_dir/"source_index.jsonl") in hashes
            and str(run_dir/"seed_manifest.json") in hashes and str(run_dir/"summary.json") in hashes,
            "Base omits primary generation inputs")
    journals={str(p.resolve()) for phase in ("first_pass","second_pass") for root in (run_dir,run_dir/"lineage/seed500")
              for p in (root/phase/"batches").glob("batch_*.json")}
    expected_journals={p for p in hashes if Path(p).name.startswith("batch_") and Path(p).parent.name == "batches"}
    require(journals == expected_journals, "Journal set changed since original full audit")
    evidence=[];seen=set()
    for error in base["errors_first_200"]:
        match=ERROR.fullmatch(error)
        require(match is not None, "An unrelated original audit error cannot be reconciled")
        phase,bid,ai=match.group(1),int(match.group(2)),int(match.group(3))
        key=(phase,bid,ai);require(key not in seen, "Duplicate original error entry");seen.add(key)
        path=run_dir/phase/"batches"/f"batch_{bid:06d}.json"
        require(str(path) in hashes and file_hash(path) == hashes[str(path)], "Failed-attempt journal changed")
        batch=verifier.read_json(path)
        attempts=batch.get("attempt_log",[])
        require(0 <= ai < len(attempts), "Failed-attempt index missing")
        event=attempts[ai]
        require(verifier.unusable_response_kind(event) == "invalid_http_json", "Event is not the narrow empty invalid_http_json failure")
        selected=[r.get("chembl_id") for r in batch.get("records",[]) if r.get("selected_attempt_index") == ai]
        require(not selected, "A record selects the unusable response; cannot reclassify as harmless receipt failure")
        evidence.append({"phase":phase,"batch_id":bid,"attempt_index":ai,"journal_path":str(path.relative_to(run_dir)),
            "journal_sha256":hashes[str(path)],"error":"invalid_http_json","http_status":200,
            "raw_response_retained":False,"usage_retained":False,"selected_by_records":[],
            "billing_status":"unknown","discarded_response_body_contents":"unknown"})
    for path,sha in hashes.items():
        require(Path(path).is_file() and file_hash(path) == sha, "Original source/config/journal/export bytes changed: "+Path(path).name)
    require(verifier.inventory(run_dir) == before, "Generation files changed during incremental recheck")
    require(file_hash(base_report) == expected_base_sha and file_hash(prior_verifier) == expected_prior_sha,
            "Pinned audit or archived verifier changed during recheck")
    corrected=copy.deepcopy(base)
    corrected.update(created_at_utc=datetime.now(timezone.utc).isoformat(),valid=True,valid_source_and_provenance=True,
                     publish_ready=False,error_count=0,errors_first_200=[])
    for phase in ("first_pass","second_pass"):
        labels=[f"new/{e['phase']}/{e['batch_id']}/attempt{e['attempt_index']}" for e in evidence if e["phase"] == phase]
        prior=corrected["phases"][phase].get("unknown_invalid_http_json_billing_attempts",[])
        corrected["phases"][phase]["unknown_invalid_http_json_billing_attempts"]=sorted(set(prior+labels))
    corrected["correction_evidence"]={"schema_version":"two_pass_invalid_http_json_reconciliation/1",
        "base_audit_path":str(base_report),"base_audit_sha256":expected_base_sha,"base_error_count":base["error_count"],
        "base_errors_preserved":base["errors_first_200"],"prior_verifier_path":str(prior_verifier),"prior_verifier_sha256":expected_prior_sha,
        "corrected_verifier_sha256":file_hash(Path(verifier.__file__)),"reconciliation_code_sha256":file_hash(Path(__file__)),
        "input_and_export_files_rehashed":len(hashes),"journal_inventory_count":len(journals),"generation_files_unchanged":True,
        "reclassified_attempt_count":len(evidence),"attempts":evidence,
        "rule":"Explicit HTTP200 invalid_http_json; no response, ID, returned model, final content or usage; no record selects the attempt. All other original audit findings must be absent and every evidence hash must match.",
        "interpretation":"Failed response parsing/shape validation, not a successful response. Raw unusable bodies were not retained by the frozen client; their contents and provider billing are unknown."}
    corrected.setdefault("limitations",[]).append(f"{len(evidence)} historical invalid_http_json attempts had no usable response or usage; no output selected them. This correction does not reconstruct discarded bodies or certify billing/chemical accuracy.")
    return corrected


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--base-report",type=Path,required=True)
    parser.add_argument("--expected-base-sha256",required=True)
    parser.add_argument("--prior-verifier",type=Path,required=True)
    parser.add_argument("--expected-prior-sha256",required=True)
    parser.add_argument("--expected-events",type=int,default=52)
    parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(argv)
    require(args.base_report.resolve() != args.report.resolve(), "Original failed audit must remain unchanged")
    require(not args.report.exists(), "Corrected audit output must be new")
    result=reconcile(args.run_dir,args.base_report,args.expected_base_sha256,args.prior_verifier,args.expected_prior_sha256,args.expected_events)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    temp=args.report.with_suffix(args.report.suffix+".tmp")
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8");temp.replace(args.report)
    print(json.dumps({key:result[key] for key in ("valid","publish_ready","complete_pairs","missing_pairs","error_count")}))


if __name__=="__main__":main()
