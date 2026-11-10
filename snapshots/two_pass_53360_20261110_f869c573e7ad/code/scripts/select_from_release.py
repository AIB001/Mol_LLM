#!/usr/bin/env python3
"""Build a compact local ChEMBL database, then sample molecules with quantitative evidence."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile

from rdkit import RDLogger, rdBase
from download_chembl import ROOT, FILTERS, ALLOWED_ATOMS, check_molecule, save_json, utc_now
from download_release import NAME, SHA256, URL

TABLES = {
    "activities": [("activity_id", "INTEGER PRIMARY KEY"), ("molregno", "INTEGER"),
        ("assay_id", "INTEGER"), ("doc_id", "INTEGER"), ("standard_type", "TEXT"),
        ("standard_value", "TEXT"), ("pchembl_value", "TEXT"), ("src_id", "INTEGER")],
    "molecule_dictionary": [("molregno", "INTEGER PRIMARY KEY"), ("chembl_id", "TEXT"),
        ("molecule_type", "TEXT"), ("structure_type", "TEXT")],
    "compound_structures": [("molregno", "INTEGER PRIMARY KEY"), ("canonical_smiles", "TEXT")],
    "molecule_hierarchy": [("molregno", "INTEGER PRIMARY KEY"), ("parent_molregno", "INTEGER")],
    "assays": [("assay_id", "INTEGER PRIMARY KEY"), ("chembl_id", "TEXT"), ("assay_type", "TEXT"),
        ("tid", "INTEGER"), ("confidence_score", "INTEGER"), ("assay_test_type", "TEXT"),
        ("assay_organism", "TEXT"), ("description", "TEXT")],
    "docs": [("doc_id", "INTEGER PRIMARY KEY"), ("chembl_id", "TEXT"), ("journal", "TEXT"),
        ("year", "INTEGER"), ("doi", "TEXT"), ("pubmed_id", "TEXT"), ("doc_type", "TEXT")],
    "target_dictionary": [("tid", "INTEGER PRIMARY KEY"), ("chembl_id", "TEXT"),
        ("pref_name", "TEXT"), ("organism", "TEXT"), ("target_type", "TEXT")],
}
ESCAPES = {"b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\"}


def unescape(value):
    if value == b"\\N":
        return None
    text = value.decode("utf-8")
    if "\\" not in text:
        return text
    def replace(match):
        token = match[1]
        if token[0] in "01234567":
            return chr(int(token, 8))
        if token.startswith("x") and len(token) > 1:
            return chr(int(token[1:], 16))
        return ESCAPES.get(token, token)
    return re.sub(r"\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)", replace, text)


def import_table(db, executable, dump, table):
    if db.execute("SELECT 1 FROM imports WHERE name=?", (table,)).fetchone():
        return
    columns = TABLES[table]
    db.execute(f'DROP TABLE IF EXISTS "{table}"')
    db.execute(f'CREATE TABLE "{table}" (' + ",".join(f'"{n}" {t}' for n, t in columns) + ")")
    statement = f'INSERT INTO "{table}" VALUES (' + ",".join("?" for _ in columns) + ")"
    env = os.environ.copy()
    # A local extracted Ubuntu client avoids requiring a running PostgreSQL server.
    local_lib = Path(executable).parents[4] / "lib/x86_64-linux-gnu"
    if local_lib.exists():
        env["LD_LIBRARY_PATH"] = str(local_lib) + ":" + env.get("LD_LIBRARY_PATH", "")
    logfile = ROOT / "data/raw/release" / f"restore_{table}.log"
    count = accepted = 0
    with open(logfile, "wb") as err:
        process = subprocess.Popen([executable, "--data-only", "--no-owner", "--no-privileges", "-t", table, "-f", "-", str(dump)],
                                   stdout=subprocess.PIPE, stderr=err, env=env)
        indices, source_names, batch = None, None, []
        for line in process.stdout:
            if line.startswith(b"COPY "):
                source_names = line.decode().split("(", 1)[1].split(")", 1)[0].replace('"', '').split(", ")
                indices = {name: i for i, name in enumerate(source_names)}
                missing = {n for n, _ in columns} - set(indices)
                if missing:
                    process.terminate()
                    raise RuntimeError(f"Missing source columns: {table}: {missing}")
                continue
            if indices is None:
                continue
            if line == b"\\.\n":
                break
            values = line.rstrip(b"\n").split(b"\t")
            if len(values) != len(source_names):
                process.terminate()
                raise RuntimeError(f"COPY row width mismatch in {table}")
            count += 1
            if table == "activities":
                if (values[indices["standard_type"]] not in {b"IC50", b"Ki", b"Kd", b"EC50"}
                    or values[indices["standard_relation"]] != b"="
                    or values[indices["standard_units"]] != b"nM"
                    or values[indices["standard_flag"]] != b"1"
                    or values[indices["potential_duplicate"]] != b"0"
                    or values[indices["data_validity_comment"]] != b"\\N"
                    or values[indices["pchembl_value"]] == b"\\N"):
                    continue
                value, pchembl = float(values[indices["standard_value"]]), float(values[indices["pchembl_value"]])
                if not (math.isfinite(value) and math.isfinite(pchembl) and value > 0
                        and abs(9 - math.log10(value) - pchembl) <= 0.025):
                    continue
            batch.append(tuple(unescape(values[indices[n]]) for n, _ in columns))
            accepted += 1
            if len(batch) >= 20000:
                db.executemany(statement, batch)
                db.commit()
                batch.clear()
                if accepted % 500000 == 0:
                    print(f"{table}: retained {accepted:,}; scanned {count:,}", flush=True)
        # Drain trailer before wait; otherwise a full stdout pipe could block pg_restore.
        process.stdout.read()
        code = process.wait()
        if code or source_names is None:
            raise RuntimeError(f"pg_restore failed for {table}; inspect {logfile}")
    if batch:
        db.executemany(statement, batch)
    db.execute("INSERT INTO imports VALUES (?,?,?)", (table, count, accepted))
    db.commit()
    print(f"Imported {table}: scanned={count:,}, retained={accepted:,}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg-restore", default=shutil.which("pg_restore") or "/tmp/mol_llm_tools/extracted/usr/lib/postgresql/14/bin/pg_restore")
    parser.add_argument("--pool-size", type=int, default=110000)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--scaffold-cap", type=int, default=50)
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    raw = ROOT / "data/raw"
    archive = raw / "release" / NAME
    dump = raw / "release/chembl_37_postgresql.dmp"
    if not dump.exists():
        with open(archive, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != SHA256:
                raise RuntimeError("Archive SHA-256 mismatch")
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.name.endswith("/chembl_37_postgresql.dmp") and member.isfile():
                    temp = Path(str(dump) + ".tmp")
                    with tar.extractfile(member) as src, open(temp, "wb") as dest:
                        shutil.copyfileobj(src, dest, 1024 * 1024)
                    if temp.stat().st_size != member.size:
                        raise RuntimeError("Truncated PostgreSQL dump")
                    temp.replace(dump)
                    break
        if not dump.exists():
            raise RuntimeError("PostgreSQL dump absent from official archive")
        print(f"Extracted {dump.name}", flush=True)
    source = sqlite3.connect(raw / "release/source_subset.sqlite")
    source.execute("PRAGMA journal_mode=WAL")
    source.execute("PRAGMA synchronous=NORMAL")
    source.execute("PRAGMA cache_size=-131072")
    source.execute("CREATE TABLE IF NOT EXISTS imports(name TEXT PRIMARY KEY, scanned INTEGER, retained INTEGER)")
    for table in TABLES:
        import_table(source, args.pg_restore, dump, table)
    source.execute("CREATE INDEX IF NOT EXISTS activities_molecule ON activities(molregno)")
    source.commit()
    source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    # WSL Windows mounts have costly random reads; keep the immutable working copy
    # on the native filesystem while retaining the authoritative subset in data/raw.
    cache_dir = Path(tempfile.gettempdir()) / "mol_llm_chembl37"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / "source_subset.sqlite"
    cached = sqlite3.connect(cache_path)
    source.backup(cached)
    source.close()
    source = cached
    source.execute("PRAGMA cache_size=-131072")
    source.row_factory = sqlite3.Row
    population_sql = """SELECT m.molregno,m.chembl_id,s.canonical_smiles,
        p.chembl_id AS parent_chembl_id FROM molecule_dictionary m
        JOIN compound_structures s ON s.molregno=m.molregno
        LEFT JOIN molecule_hierarchy h ON h.molregno=m.molregno
        LEFT JOIN molecule_dictionary p ON p.molregno=h.parent_molregno
        WHERE m.molecule_type='Small molecule' AND m.structure_type='MOL'
        AND s.canonical_smiles IS NOT NULL AND EXISTS (
          SELECT 1 FROM activities a JOIN assays ass ON a.assay_id=ass.assay_id
          JOIN docs d ON a.doc_id=d.doc_id JOIN target_dictionary t ON ass.tid=t.tid
          WHERE a.molregno=m.molregno)"""
    (raw / "release/population.sql").write_text(population_sql, encoding="utf-8")
    population = [dict(r) for r in source.execute(population_sql)]
    population.sort(key=lambda r: hashlib.sha256(f"{args.seed}:{r['chembl_id']}".encode()).digest())
    print(f"Activity-backed small-molecule population: {len(population):,}", flush=True)
    candidates = sqlite3.connect(raw / "candidates.sqlite")
    candidates.execute("PRAGMA journal_mode=WAL")
    candidates.execute("PRAGMA synchronous=NORMAL")
    candidates.execute("PRAGMA cache_size=-131072")
    candidates.executescript("""
        CREATE TABLE IF NOT EXISTS molecules (chembl_id TEXT PRIMARY KEY, smiles TEXT NOT NULL,
          rdkit_smiles TEXT UNIQUE NOT NULL, scaffold TEXT NOT NULL, parent_chembl_id TEXT, first_activity_id INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS rejected_molecules(chembl_id TEXT PRIMARY KEY, reason TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS activities(activity_id INTEGER PRIMARY KEY,chembl_id TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS activity_molecule ON activities(chembl_id);
        CREATE TABLE IF NOT EXISTS pages(offset INTEGER PRIMARY KEY,records INTEGER,rejections TEXT);
    """)
    selection_config = {"seed": args.seed, "pool_size": args.pool_size, "scaffold_cap": args.scaffold_cap,
                        "source_sha256": SHA256, "rdkit_version": rdBase.rdkitVersion}
    config_path = raw / "release/selection_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != selection_config:
        raise RuntimeError("Selection settings changed; use a fresh candidate database")
    save_json(config_path, selection_config)
    done_ids = {r[0] for r in candidates.execute("SELECT chembl_id FROM molecules")}
    rejected_ids = {r[0] for r in candidates.execute("SELECT chembl_id FROM rejected_molecules")}
    structures = {r[0] for r in candidates.execute("SELECT rdkit_smiles FROM molecules")}
    scaffold_counts = Counter(r[0] for r in candidates.execute("SELECT scaffold FROM molecules WHERE scaffold != ''"))
    selected = len(done_ids)
    capacity = candidates.execute("SELECT COUNT(*) FROM molecules WHERE scaffold = ''").fetchone()[0]
    capacity += sum(min(args.scaffold_cap,n) for n in scaffold_counts.values())
    evidence_sql = """SELECT a.*,ass.chembl_id AS assay_chembl_id,ass.assay_type,ass.confidence_score,
        ass.assay_test_type,ass.assay_organism,ass.description AS assay_description,
        d.chembl_id AS document_chembl_id,d.journal,d.year,d.doi,d.pubmed_id,d.doc_type,
        t.chembl_id AS target_chembl_id,t.pref_name AS target_pref_name,t.organism AS target_organism,t.target_type
        FROM activities a JOIN assays ass ON a.assay_id=ass.assay_id
        JOIN docs d ON a.doc_id=d.doc_id JOIN target_dictionary t ON ass.tid=t.tid
        WHERE a.molregno=? ORDER BY a.activity_id"""
    for row in population:
        if capacity >= args.pool_size:
            break
        if row["chembl_id"] in done_ids or row["chembl_id"] in rejected_ids:
            continue
        result, reason = check_molecule(row["canonical_smiles"])
        if result and result[0] in structures:
            reason = "duplicate_isomeric_structure"
        if reason:
            candidates.execute("INSERT INTO rejected_molecules VALUES (?,?)", (row["chembl_id"], reason))
            continue
        canonical, scaffold = result
        structures.add(canonical)
        evidence = []
        for record in source.execute(evidence_sql, (row["molregno"],)):
            a = dict(record)
            payload = {"activity_id": a["activity_id"], "molecule_chembl_id": row["chembl_id"],
                "canonical_smiles": row["canonical_smiles"], "parent_molecule_chembl_id": row["parent_chembl_id"],
                "standard_type": a["standard_type"], "standard_value": a["standard_value"],
                "standard_relation": "=", "standard_units": "nM", "standard_flag": 1,
                "potential_duplicate": 0, "data_validity_comment": None, "pchembl_value": a["pchembl_value"],
                "src_id": a["src_id"], "document_journal": a["journal"], "document_year": a["year"],
                **{k: a[k] for k in ["assay_chembl_id", "assay_type", "confidence_score", "assay_test_type",
                    "assay_organism", "assay_description", "document_chembl_id", "doi", "pubmed_id", "doc_type",
                    "target_chembl_id", "target_pref_name", "target_organism", "target_type"]}}
            evidence.append((a["activity_id"], row["chembl_id"], json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))
        if not evidence:
            raise RuntimeError("Population/evidence join mismatch")
        candidates.execute("INSERT INTO molecules VALUES (?,?,?,?,?,?)", (row["chembl_id"],row["canonical_smiles"],
            canonical,scaffold,row["parent_chembl_id"],evidence[0][0]))
        candidates.executemany("INSERT INTO activities VALUES (?,?,?)", evidence)
        selected += 1
        if scaffold:
            scaffold_counts[scaffold] += 1
            capacity += scaffold_counts[scaffold] <= args.scaffold_cap
        else:
            capacity += 1
        if selected % 5000 == 0:
            candidates.commit()
            print(f"Candidates={selected:,}; capacity after scaffold cap={capacity:,}", flush=True)
        if capacity >= args.pool_size:
            break
    if capacity < args.pool_size:
        raise RuntimeError("Not enough qualifying structures")
    candidates.commit()
    candidates.close()
    status = {"chembl_db_version": "ChEMBL_37", "chembl_release_date": "2026-05-01", "source": "official_postgresql_release"}
    plan = {"created_at_utc": utc_now(), "source_status": status,
        "source_url": URL, "source_sha256": SHA256,
        "eligible_activity_total": source.execute("SELECT COUNT(*) FROM activities").fetchone()[0],
        "activity_backed_molecule_population": len(population),
        "source_table_counts": [dict(r) for r in source.execute("SELECT * FROM imports")],
        "config": {"filters": FILTERS, "seed": args.seed, "scaffold_cap": args.scaffold_cap,
            "sampling": "seeded hash order over all qualifying molecule IDs, followed by structure filtering and scaffold cap",
            "source": "ChEMBL_PostgreSQL_release", "evidence_scope": "all qualifying activities for each selected molecule",
            "rdkit_version": rdBase.rdkitVersion,
            "structure_filters": {"molecule_type": "Small molecule", "mol_wt": [100,1000],"heavy_atoms": [8,80],
                "single_component": True,"carbon_required":True,"no_radicals":True,"allowed_atomic_numbers": sorted(ALLOWED_ATOMS)}}}
    save_json(raw / "download_plan.json", plan)
    save_json(raw / "source_status_end.json", status)
    source.close()
    print("Release-based candidate pool complete", flush=True)


if __name__ == "__main__":
    main()
