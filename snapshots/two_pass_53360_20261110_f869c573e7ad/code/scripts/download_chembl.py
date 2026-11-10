#!/usr/bin/env python3
"""Download a reproducible activity-backed ChEMBL candidate pool (no LLM calls)."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3
import threading
import time

import requests
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.ebi.ac.uk/chembl/api/data"
FILTERS = {
    "standard_type__in": "IC50,Ki,Kd,EC50", "standard_relation": "=",
    "standard_units": "nM", "standard_flag": 1, "pchembl_value__isnull": "false",
    "potential_duplicate": 0, "data_validity_comment__isnull": "true",
}
FIELDS = (
    "activity_id,molecule_chembl_id,parent_molecule_chembl_id,canonical_smiles,"
    "standard_type,standard_relation,standard_units,standard_value,pchembl_value,"
    "standard_flag,potential_duplicate,data_validity_comment,assay_chembl_id,"
    "assay_type,document_chembl_id,document_journal,document_year,target_chembl_id,"
    "target_pref_name,target_organism,src_id"
)
ALLOWED_ATOMS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
HTTP_LOCAL = threading.local()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def get_json(url, params=None):
    if not hasattr(HTTP_LOCAL, "session"):
        HTTP_LOCAL.session = requests.Session()
    for attempt in range(7):
        try:
            query = dict(params or {})
            query["_request"] = f"{time.time_ns()}"
            response = HTTP_LOCAL.session.get(url, params=query, timeout=(10, 90 if params else 20))
            response.raise_for_status()
            return response.json(), response.url
        except (requests.RequestException, ValueError) as exc:
            print(f"HTTP retry {attempt + 1}: {url.rsplit('/', 1)[-1]} offset={(params or {}).get('offset')} {type(exc).__name__}", flush=True)
            if attempt == 6:
                raise RuntimeError(f"ChEMBL request failed: {url}: {type(exc).__name__}") from exc
            time.sleep(min(2 ** attempt, 30))


def check_activity(a):
    if a.get("standard_type") not in {"IC50", "Ki", "Kd", "EC50"}:
        return "endpoint"
    if a.get("standard_relation") != "=" or a.get("standard_units") != "nM":
        return "relation_or_units"
    if a.get("standard_flag") != 1 or a.get("potential_duplicate") != 0:
        return "standard_or_duplicate_flag"
    if a.get("data_validity_comment") is not None:
        return "validity_flag"
    try:
        value, pchembl = float(a["standard_value"]), float(a["pchembl_value"])
        if not (math.isfinite(value) and math.isfinite(pchembl) and value > 0):
            return "invalid_value"
        if abs(9 - math.log10(value) - pchembl) > 0.025:
            return "inconsistent_pchembl"
    except (KeyError, TypeError, ValueError):
        return "missing_value"
    if not all(a.get(k) for k in ("activity_id", "molecule_chembl_id", "canonical_smiles",
                                  "assay_chembl_id", "document_chembl_id", "target_chembl_id")):
        return "missing_provenance_or_structure"
    return None


def check_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "rdkit_parse"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "disconnected_structure"
    atoms = list(mol.GetAtoms())
    if not any(a.GetAtomicNum() == 6 for a in atoms):
        return None, "no_carbon"
    if any(a.GetAtomicNum() not in ALLOWED_ATOMS for a in atoms):
        return None, "excluded_element"
    if any(a.GetNumRadicalElectrons() for a in atoms):
        return None, "radical"
    if not 8 <= mol.GetNumHeavyAtoms() <= 80:
        return None, "heavy_atom_range"
    if not 100 <= Descriptors.MolWt(mol) <= 1000:
        return None, "molecular_weight_range"
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    return (canonical, scaffold), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-size", type=int, default=110000,
                        help="Required capacity after the per-scaffold cap")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--scaffold-cap", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pages", type=int, default=1500)
    args = parser.parse_args()
    if min(args.pool_size, args.scaffold_cap, args.workers, args.max_pages) < 1:
        parser.error("All size arguments must be positive")
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    status, _ = get_json(f"{API}/status.json")
    plan_path = raw / "download_plan.json"
    config = {"filters": FILTERS, "fields": FIELDS, "seed": args.seed,
              "page_size": 1000, "scaffold_cap": args.scaffold_cap,
              "rdkit_version": rdBase.rdkitVersion,
              "structure_filters": {"mol_wt": [100, 1000], "heavy_atoms": [8, 80],
                 "single_component": True, "carbon_required": True, "no_radicals": True,
                 "allowed_atomic_numbers": sorted(ALLOWED_ATOMS)}}
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["config"] != config or plan["source_status"]["chembl_db_version"] != status["chembl_db_version"]:
            raise RuntimeError("Existing download belongs to a different config/release; use a new data directory")
    else:
        first, _ = get_json(f"{API}/activity.json", {**FILTERS, "limit": 1, "only": FIELDS})
        total = first["page_meta"]["total_count"]
        offsets = list(range(0, total, 1000))
        random.Random(args.seed).shuffle(offsets)
        plan = {"created_at_utc": utc_now(), "source_status": status, "config": config,
                "eligible_activity_total": total, "offsets": offsets}
        save_json(plan_path, plan)
    pages = raw / status["chembl_db_version"].lower() / "activity_pages"
    pages.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(raw / "candidates.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS molecules (
        chembl_id TEXT PRIMARY KEY, smiles TEXT NOT NULL, rdkit_smiles TEXT UNIQUE NOT NULL,
        scaffold TEXT NOT NULL, parent_chembl_id TEXT, first_activity_id INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS rejected_molecules (chembl_id TEXT PRIMARY KEY, reason TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS activities (
        activity_id INTEGER PRIMARY KEY, chembl_id TEXT NOT NULL, payload TEXT NOT NULL);
    CREATE INDEX IF NOT EXISTS activity_molecule ON activities(chembl_id);
    CREATE TABLE IF NOT EXISTS pages (offset INTEGER PRIMARY KEY, records INTEGER, rejections TEXT);
    """)
    known = {r[0]: (r[1], r[2]) for r in db.execute("SELECT chembl_id, smiles, rdkit_smiles FROM molecules")}
    structures = {v[1] for v in known.values()}
    rejected = {r[0] for r in db.execute("SELECT chembl_id FROM rejected_molecules")}
    scaffold_counts = Counter(r[0] for r in db.execute("SELECT scaffold FROM molecules WHERE scaffold != ''"))
    acyclic = db.execute("SELECT COUNT(*) FROM molecules WHERE scaffold = ''").fetchone()[0]
    capacity = acyclic + sum(min(args.scaffold_cap, n) for n in scaffold_counts.values())
    done = {r[0] for r in db.execute("SELECT offset FROM pages")}
    pending = [o for o in plan["offsets"][:args.max_pages] if o not in done]

    def fetch(offset):
        path = pages / f"{offset:09d}.json.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)
        else:
            payload, url = get_json(f"{API}/activity.json", {**FILTERS, "limit": 1000, "offset": offset, "only": FIELDS})
            if payload["page_meta"]["total_count"] != plan["eligible_activity_total"]:
                raise RuntimeError("Source activity count changed during download")
            envelope = {"url": url, "retrieved_at_utc": utc_now(), "payload": payload}
            tmp = Path(str(path) + ".tmp")
            with gzip.open(tmp, "wt", encoding="utf-8") as handle:
                json.dump(envelope, handle, ensure_ascii=False, separators=(",", ":"))
            tmp.replace(path)
        payload = envelope["payload"]
        if payload["page_meta"]["offset"] != offset:
            raise RuntimeError("API pagination mismatch")
        expected = min(1000, plan["eligible_activity_total"] - offset)
        if len(payload["activities"]) != expected:
            raise RuntimeError("Incomplete API page")
        return offset, payload["activities"]

    print(json.dumps({"release": status["chembl_db_version"], "eligible_activities": plan["eligible_activity_total"],
                      "existing_candidates": len(known), "capacity": capacity}), flush=True)
    # A fixed batch size and ordered consumption make selection independent of thread timing.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(pending), 8):
            if capacity >= args.pool_size:
                break
            batch = list(pool.map(fetch, pending[start:start + 8]))
            for offset, records in batch:
                reasons = Counter()
                for a in records:
                    reason = check_activity(a)
                    if reason:
                        reasons[reason] += 1
                        continue
                    cid, smiles = a["molecule_chembl_id"], a["canonical_smiles"]
                    if cid in rejected:
                        reasons["previously_rejected_molecule"] += 1
                        continue
                    if cid not in known:
                        result, reason = check_molecule(smiles)
                        if result and result[0] in structures:
                            reason = "duplicate_isomeric_structure"
                        if reason:
                            db.execute("INSERT INTO rejected_molecules VALUES (?,?)", (cid, reason))
                            rejected.add(cid)
                            reasons[reason] += 1
                            continue
                        canonical, scaffold = result
                        db.execute("INSERT INTO molecules VALUES (?,?,?,?,?,?)", (cid, smiles, canonical,
                                   scaffold, a.get("parent_molecule_chembl_id"), a["activity_id"]))
                        known[cid] = (smiles, canonical)
                        structures.add(canonical)
                        if scaffold:
                            scaffold_counts[scaffold] += 1
                            if scaffold_counts[scaffold] <= args.scaffold_cap:
                                capacity += 1
                        else:
                            capacity += 1
                    elif known[cid][0] != smiles:
                        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
                        if canonical != known[cid][1]:
                            raise RuntimeError(f"Inconsistent structures for {cid}")
                    db.execute("INSERT OR IGNORE INTO activities VALUES (?,?,?)",
                               (a["activity_id"], cid, json.dumps(a, ensure_ascii=False, separators=(",", ":"))))
                db.execute("INSERT INTO pages VALUES (?,?,?)", (offset, len(records), json.dumps(reasons)))
                done.add(offset)
            db.commit()
            progress = {"updated_at_utc": utc_now(), "processed_pages": len(done), "candidates": len(known),
                        "capacity_with_scaffold_cap": capacity, "pool_size_requested": args.pool_size,
                        "cached_qualifying_activities": db.execute("SELECT COUNT(*) FROM activities").fetchone()[0]}
            save_json(raw / "download_progress.json", progress)
            print(json.dumps(progress), flush=True)
    if capacity < args.pool_size:
        raise RuntimeError(f"Insufficient pool: {capacity}; increase --max-pages and resume")
    final_status, _ = get_json(f"{API}/status.json")
    if final_status["chembl_db_version"] != status["chembl_db_version"]:
        raise RuntimeError("ChEMBL release changed during download")
    save_json(raw / "source_status_end.json", final_status)
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.close()
    print("Candidate download complete. Run scripts/prepare_dataset.py next.", flush=True)


if __name__ == "__main__":
    main()
