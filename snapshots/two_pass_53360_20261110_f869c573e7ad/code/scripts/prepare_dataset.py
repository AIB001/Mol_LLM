#!/usr/bin/env python3
"""Select unique scaffold-capped molecules, compute vectors and retain activity evidence."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
from rdkit import Chem, RDLogger

from download_chembl import ROOT, check_activity, save_json, utc_now
from molecular_features import FEATURE_NAMES, calculate_features, schema


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def descriptor_row(smiles):
    return calculate_features(Chem.MolFromSmiles(smiles))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100000)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    RDLogger.DisableLog("rdApp.warning")
    raw, out = ROOT / "data/raw", ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads((raw / "download_plan.json").read_text())
    if not (raw / "source_status_end.json").exists():
        raise RuntimeError("Download has not completed its source-release check")
    if schema()["rdkit_version"] != plan["config"]["rdkit_version"]:
        raise RuntimeError("RDKit version changed since source filtering")
    original = sqlite3.connect(f"file:{raw / 'candidates.sqlite'}?mode=ro", uri=True)
    cache_dir = Path(tempfile.gettempdir()) / "mol_llm_chembl37"
    cache_dir.mkdir(exist_ok=True)
    db = sqlite3.connect(cache_dir / "candidates_for_export.sqlite")
    original.backup(db)
    original.close()
    db.execute("PRAGMA cache_size=-131072")
    db.row_factory = sqlite3.Row
    candidates = [dict(r) for r in db.execute("SELECT * FROM molecules")]
    seed, cap = plan["config"]["seed"], plan["config"]["scaffold_cap"]
    candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['chembl_id']}".encode()).digest())
    selected, scaffold_counts = [], Counter()
    for row in candidates:
        scaffold = row["scaffold"]
        if scaffold and scaffold_counts[scaffold] >= cap:
            continue
        selected.append(row)
        if scaffold:
            scaffold_counts[scaffold] += 1
        if len(selected) == args.count:
            break
    if len(selected) != args.count:
        raise RuntimeError(f"Only {len(selected)} eligible structures; download more pages first")

    base_fields = ["chembl_id", "smiles", "rdkit_canonical_smiles", "parent_chembl_id",
        "murcko_scaffold_smiles", "source", "chembl_release", "qualified_activity_count",
        "qualified_assay_count", "qualified_target_count", "activity_types", "representative_activity_id",
        "representative_assay_chembl_id", "representative_document_chembl_id",
        "representative_target_chembl_id", "representative_standard_type",
        "representative_standard_value", "representative_standard_units", "representative_pchembl_value"]
    evidence_fields = ["molecule_chembl_id", "activity_id", "assay_chembl_id", "assay_type",
        "confidence_score", "assay_test_type", "assay_organism", "assay_description",
        "document_chembl_id", "document_journal", "document_year", "doi", "pubmed_id", "doc_type", "target_type", "target_chembl_id",
        "target_pref_name", "target_organism", "standard_type", "standard_relation",
        "standard_value", "standard_units", "pchembl_value", "standard_flag",
        "potential_duplicate", "data_validity_comment", "src_id"]
    prefix = f"chembl_{args.count}"
    paths = {"molecules": out / f"{prefix}_molecules.csv", "properties": out / f"{prefix}_properties.csv",
             "evidence": out / f"{prefix}_activities.csv.gz", "vectors": out / f"{prefix}_vectors.npz"}
    matrix = np.empty((args.count, len(FEATURE_NAMES)), dtype=np.float64)
    endpoint_counts, organism_counts, assay_types, source_counts = Counter(), Counter(), Counter(), Counter()
    activity_count = 0
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as executor, \
         open(paths["molecules"], "w", newline="", encoding="utf-8") as mf, \
         open(paths["properties"], "w", newline="", encoding="utf-8") as pf, \
         gzip.open(paths["evidence"], "wt", newline="", encoding="utf-8") as ef:
        mw, pw, ew = (csv.DictWriter(mf, fieldnames=base_fields),
                      csv.DictWriter(pf, fieldnames=["chembl_id"] + FEATURE_NAMES),
                      csv.DictWriter(ef, fieldnames=evidence_fields, extrasaction="ignore"))
        for writer in (mw, pw, ew):
            writer.writeheader()
        feature_rows = executor.map(descriptor_row, (r["smiles"] for r in selected), chunksize=64)
        for i, row in enumerate(selected):
            cid = row["chembl_id"]
            activities = [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM activities WHERE chembl_id = ? ORDER BY activity_id", (cid,))]
            if not activities or any(check_activity(a) for a in activities):
                raise RuntimeError(f"Missing/invalid activity evidence: {cid}")
            # Lowest ID is deterministic, not a claim that this is the best or most potent measurement.
            a = activities[0]
            mw.writerow({"chembl_id": cid, "smiles": row["smiles"],
                "rdkit_canonical_smiles": row["rdkit_smiles"], "parent_chembl_id": row["parent_chembl_id"],
                "murcko_scaffold_smiles": row["scaffold"], "source": plan["config"].get("source", "ChEMBL_REST_API"),
                "chembl_release": plan["source_status"]["chembl_db_version"],
                "qualified_activity_count": len(activities),
                "qualified_assay_count": len({x["assay_chembl_id"] for x in activities}),
                "qualified_target_count": len({x["target_chembl_id"] for x in activities}),
                "activity_types": "|".join(sorted({x["standard_type"] for x in activities})),
                "representative_activity_id": a["activity_id"],
                "representative_assay_chembl_id": a["assay_chembl_id"],
                "representative_document_chembl_id": a["document_chembl_id"],
                "representative_target_chembl_id": a["target_chembl_id"],
                "representative_standard_type": a["standard_type"],
                "representative_standard_value": a["standard_value"],
                "representative_standard_units": a["standard_units"],
                "representative_pchembl_value": a["pchembl_value"]})
            features = next(feature_rows)
            matrix[i] = [features[k] for k in FEATURE_NAMES]
            pw.writerow({"chembl_id": cid, **features})
            for a in activities:
                ew.writerow(a)
                endpoint_counts[a["standard_type"]] += 1
                organism_counts[a.get("target_organism") or "unknown"] += 1
                assay_types[a.get("assay_type") or "unknown"] += 1
                source_counts[a.get("src_id")] += 1
                activity_count += 1
            if (i + 1) % 10000 == 0:
                print(f"Prepared {i + 1:,}/{args.count:,} molecules", flush=True)
    np.savez_compressed(paths["vectors"], chembl_ids=np.array([r["chembl_id"] for r in selected]),
                        feature_names=np.array(FEATURE_NAMES), vectors=matrix)
    save_json(out / "property_schema.json", schema())
    rejections = Counter()
    for reason, count in db.execute("SELECT reason, COUNT(*) FROM rejected_molecules GROUP BY reason"):
        rejections[reason] = count
    manifest = {"created_at_utc": utc_now(), "source_status": plan["source_status"],
        "molecule_count": args.count, "candidate_count": len(candidates),
        "population_molecule_count": plan.get("activity_backed_molecule_population"),
        "source_archive_url": plan.get("source_url"), "source_archive_sha256": plan.get("source_sha256"),
        "source_table_counts": plan.get("source_table_counts"),
        "qualified_activity_count": activity_count, "downloaded_page_count": db.execute("SELECT COUNT(*) FROM pages").fetchone()[0],
        "qualified_activity_type_counts": endpoint_counts, "qualified_assay_type_counts": assay_types,
        "qualified_source_id_counts": source_counts, "qualified_target_organism_counts": organism_counts,
        "distinct_nonempty_murcko_scaffolds": len(scaffold_counts),
        "acyclic_molecules": sum(not r["scaffold"] for r in selected),
        "largest_nonempty_scaffold_group": max(scaffold_counts.values(), default=0),
        "rejected_candidate_molecules": rejections, "selection_config": plan["config"],
        "feature_schema": schema(), "files": {k: {"path": str(v.relative_to(ROOT)),
            "bytes": v.stat().st_size, "sha256": digest(v)} for k, v in paths.items()},
        "limitations": [
            "Seeded molecule-level selection from the complete release, followed by structural filtering and scaffold capping; not a representative sample of all chemistry.",
            "Evidence includes all records passing the stated activity filters for selected IDs, not every possible experimental endpoint in ChEMBL.",
            "No assay-to-target confidence-score threshold was imposed; assay types and target organisms are preserved.",
            "Different endpoints/targets/assay conditions must not be pooled as one experimental property label.",
            "Acyclic molecules have empty Murcko scaffolds and are not subject to a shared acyclic scaffold cap.",
            "Stereochemistry is preserved; no tautomer/protonation normalization or biological parent aggregation.",
            "Calculated RDKit descriptors are not experimental measurements; no LLM annotations have been generated.",
        ],
        "attribution": "ChEMBL data from https://www.ebi.ac.uk/chembl, release ChEMBL_37; CC BY-SA 3.0.",
        "references": ["https://www.ebi.ac.uk/chembl/api/data/docs",
            "https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions",
            "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/README"]}
    save_json(out / "dataset_manifest.json", manifest)
    print(json.dumps({"molecules": args.count, "activities": activity_count,
                      "scaffolds": len(scaffold_counts), "vector_shape": list(matrix.shape)}), flush=True)
    db.close()


if __name__ == "__main__":
    main()
