#!/usr/bin/env python3
"""Export a lossless, per-molecule graph fact layer without API calls.

Every serialized graph is reconstructed and compared with its source before it
is written. Eight worker processes prepare bounded chunks; one parent streams
deterministic gzip JSONL. The final artifact is published by atomic replacement
only after every selected row passes and both source hashes remain unchanged.
The existing property CSV and all annotation outputs are read-only inputs.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
from itertools import islice, zip_longest
import json
import multiprocessing
import os
from pathlib import Path
import time

from rdkit import Chem, RDLogger, rdBase

from download_chembl import ROOT, utc_now
from molecular_features import FEATURE_NAMES
from structural_facts_v2 import SCHEMA_VERSION as FACT_SCHEMA_VERSION
from structural_facts_v2 import mol_from_structural_facts_v2, structural_facts_v2

RECORD_SCHEMA_VERSION = "structural_fact_record/2"
METADATA_SCHEMA_VERSION = "structural_fact_export/2"
DEFAULT_OUTPUT = ROOT / "data/processed/chembl_100000_structural_facts_v2.jsonl.gz"


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atom_signature(atom):
    return (atom.GetAtomicNum(), atom.GetIsotope(), atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(), atom.GetNumExplicitHs(), atom.GetNumImplicitHs(),
            atom.GetNoImplicit(), atom.GetIsAromatic(), str(atom.GetHybridization()),
            atom.GetAtomMapNum(), str(atom.GetChiralTag()),
            atom.GetIntProp("_chiralPermutation") if atom.HasProp("_chiralPermutation") else None,
            tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors()))


def bond_signature(bond):
    return (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType()),
            bond.GetIsAromatic(), bond.GetIsConjugated(), str(bond.GetBondDir()),
            str(bond.GetStereo()), tuple(bond.GetStereoAtoms()))


def stereo_group_signature(mol):
    return [(str(group.GetGroupType()), tuple(atom.GetIdx() for atom in group.GetAtoms()),
             tuple(bond.GetIdx() for bond in group.GetBonds())) for group in mol.GetStereoGroups()]


def verify_graph(original, facts):
    rebuilt = mol_from_structural_facts_v2(facts)
    if original.GetNumAtoms() != rebuilt.GetNumAtoms() or original.GetNumBonds() != rebuilt.GetNumBonds():
        raise ValueError("atom_or_bond_count_mismatch")
    if [atom_signature(atom) for atom in original.GetAtoms()] != [atom_signature(atom) for atom in rebuilt.GetAtoms()]:
        raise ValueError("indexed_atom_or_stereo_mismatch")
    if [bond_signature(bond) for bond in original.GetBonds()] != [bond_signature(bond) for bond in rebuilt.GetBonds()]:
        raise ValueError("indexed_bond_or_stereo_mismatch")
    if stereo_group_signature(original) != stereo_group_signature(rebuilt):
        raise ValueError("enhanced_stereo_group_mismatch")
    if Chem.MolToCXSmiles(original) != Chem.MolToCXSmiles(rebuilt):
        raise ValueError("canonical_cxsmiles_mismatch")
    return {"status": "exact_graph_roundtrip", "atoms": original.GetNumAtoms(), "bonds": original.GetNumBonds(),
            "indexed_atoms_bonds_hydrogens_charge_isotopes_stereo_match": True,
            "canonical_cxsmiles_match": True}


def make_record(row):
    """Make one checked record; failures identify only the molecule, never credentials."""
    cid = row["chembl_id"]
    original = Chem.MolFromSmiles(row["smiles"])
    if original is None:
        raise ValueError(f"{cid}: invalid_input_smiles")
    Chem.AssignStereochemistry(original, cleanIt=True, force=True)
    # Reconstruct the JSON-round-tripped representation, not an in-memory shortcut.
    encoded_facts = json_text(structural_facts_v2(original, original_smiles=row["smiles"]))
    facts = json.loads(encoded_facts)
    try:
        validation = verify_graph(original, facts)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"{cid}: {exc}") from exc
    return {"schema_version": RECORD_SCHEMA_VERSION, "chembl_id": cid, "smiles": row["smiles"],
            "facts_sha256": hashlib.sha256(encoded_facts.encode("utf-8")).hexdigest(),
            "structural_facts": facts, "graph_validation": validation}


def prepare_chunk(rows):
    encoded = []
    atom_total = bond_total = 0
    fact_hash_lines = []
    for row in rows:
        record = make_record(row)
        encoded.append((json_text(record) + "\n").encode("utf-8"))
        atom_total += record["graph_validation"]["atoms"]
        bond_total += record["graph_validation"]["bonds"]
        fact_hash_lines.append(f"{record['chembl_id']}\t{record['facts_sha256']}\n")
    return b"".join(encoded), len(rows), atom_total, bond_total, "".join(fact_hash_lines).encode("utf-8")


def init_worker():
    RDLogger.DisableLog("rdApp.warning")


def source_rows(input_path, properties_path, limit=0):
    """Check identifier alignment; properties are never converted or rewritten."""
    seen = set()
    with input_path.open(newline="", encoding="utf-8") as mf, properties_path.open(newline="", encoding="utf-8") as pf:
        molecules, properties = csv.DictReader(mf), csv.DictReader(pf)
        if not {"chembl_id", "smiles"} <= set(molecules.fieldnames or []):
            raise ValueError("Molecule CSV requires chembl_id and smiles")
        if not {"chembl_id", *FEATURE_NAMES} <= set(properties.fieldnames or []):
            raise ValueError("Property CSV must retain the full 24-feature schema")
        for index, (molecule, prop) in enumerate(zip_longest(molecules, properties), 1):
            if limit and index > limit:
                return
            if molecule is None or prop is None or molecule["chembl_id"] != prop["chembl_id"]:
                raise ValueError(f"Molecule/property rows do not align at row {index}")
            cid, smiles = molecule["chembl_id"], molecule["smiles"]
            if not cid or not smiles or cid in seen:
                raise ValueError(f"Missing structure or duplicate identifier at row {index}")
            seen.add(cid)
            yield {"chembl_id": cid, "smiles": smiles}


def chunks(rows, size):
    iterator = iter(rows)
    while chunk := list(islice(iterator, size)):
        yield chunk


def prepared_chunks(rows, workers, chunk_size, max_pending):
    work = iter(chunks(rows, chunk_size))
    if workers == 1:
        for chunk in work:
            yield prepare_chunk(chunk)
        return
    # A fixed submission window bounds parent and child result buffers. Ordered
    # consumption also preserves input order and deterministic gzip bytes.
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=init_worker) as pool:
        pending = deque()
        try:
            for _ in range(max_pending):
                chunk = next(work, None)
                if chunk is None:
                    break
                pending.append(pool.submit(prepare_chunk, chunk))
            while pending:
                yield pending.popleft().result()
                chunk = next(work, None)
                if chunk is not None:
                    pending.append(pool.submit(prepare_chunk, chunk))
        finally:
            for future in pending:
                future.cancel()


def metadata_path(output):
    name = output.name.removesuffix(".jsonl.gz")
    return output.with_name(name + ".metadata.json")


def export(args):
    output = args.output
    meta_path = metadata_path(output)
    for source in (args.input, args.properties):
        if output.resolve() == source.resolve() or meta_path.resolve() == source.resolve():
            raise ValueError("Output must not overwrite a source CSV")
    if not args.force and (output.exists() or meta_path.exists()):
        raise FileExistsError("Fact output or metadata already exists; use a new path or explicit --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    meta_temporary = meta_path.with_name(meta_path.name + f".tmp.{os.getpid()}")
    source_hashes = {"molecules": file_hash(args.input), "properties": file_hash(args.properties)}
    code_hashes = {"fact": file_hash(ROOT / "scripts/structural_facts_v2.py"),
                   "exporter": file_hash(Path(__file__))}
    started_at = utc_now()
    started = time.monotonic()
    rows = atoms = bonds = 0
    content_hash, facts_hash = hashlib.sha256(), hashlib.sha256()
    report_at = 1000
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=args.compression_level, mtime=0) as zipped:
                for data, count, atom_count, bond_count, fact_lines in prepared_chunks(
                    source_rows(args.input, args.properties, args.limit), args.workers, args.chunk_size, args.max_pending
                ):
                    zipped.write(data)
                    content_hash.update(data)
                    facts_hash.update(fact_lines)
                    rows += count
                    atoms += atom_count
                    bonds += bond_count
                    if rows >= report_at:
                        print(f"Verified and written {rows} molecules; elapsed={time.monotonic()-started:.1f}s", flush=True)
                        report_at += 1000
        if rows == 0:
            raise ValueError("No molecules selected")
        after_hashes = {"molecules": file_hash(args.input), "properties": file_hash(args.properties)}
        if source_hashes != after_hashes:
            raise RuntimeError("A source file changed during export; refusing publication")
        if code_hashes != {"fact": file_hash(ROOT / "scripts/structural_facts_v2.py"),
                           "exporter": file_hash(Path(__file__))}:
            raise RuntimeError("Exporter or fact code changed during export; refusing publication")
        metadata = {"schema_version": METADATA_SCHEMA_VERSION, "record_schema_version": RECORD_SCHEMA_VERSION,
                    "fact_schema_version": FACT_SCHEMA_VERSION, "started_at_utc": started_at, "completed_at_utc": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3), "molecule_count": rows,
                    "verified_graph_count": rows, "total_atoms": atoms, "total_bonds": bonds,
                    "selected_limit": args.limit, "workers": args.workers, "chunk_size": args.chunk_size,
                    "max_pending_chunks": args.max_pending, "rdkit_version": rdBase.rdkitVersion,
                    "sources": {"molecules": {"path": str(args.input.resolve()), "sha256": source_hashes["molecules"]},
                                "properties": {"path": str(args.properties.resolve()), "sha256": source_hashes["properties"]}},
                    "feature_names": FEATURE_NAMES, "source_files_unchanged": True,
                    "fact_code_sha256": code_hashes["fact"],
                    "exporter_code_sha256": code_hashes["exporter"],
                    "artifact": {"path": str(output.resolve()), "sha256": file_hash(temporary),
                                 "bytes": temporary.stat().st_size, "compression": "gzip", "compression_level": args.compression_level,
                                 "jsonl_sha256": content_hash.hexdigest()},
                    "ordered_facts_sha256": facts_hash.hexdigest(),
                    "ordered_facts_hash_format": "UTF-8 chembl_id + tab + facts_sha256 + newline, in input order",
                    "facts_hash_format": "SHA256 of UTF-8 JSON; sorted keys, compact separators, ensure_ascii=False",
                    "validation_scope": "Every JSON-serialized graph reconstructed; indexed atoms, bonds, hydrogens, "
                        "charges, isotopes, stereo and canonical CXSMILES compared with the source RDKit molecule. "
                        "This does not verify natural-language descriptions or experimental chemical identity."}
        meta_temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, output)
        os.replace(meta_temporary, meta_path)
    finally:
        temporary.unlink(missing_ok=True)
        meta_temporary.unlink(missing_ok=True)
    print(json.dumps({"status": "completed", "molecule_count": rows, "output": str(output),
                      "metadata": str(meta_path), "bytes": metadata["artifact"]["bytes"],
                      "elapsed_seconds": metadata["elapsed_seconds"]}), flush=True)
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-pending", type=int, default=None, help="Bound submitted chunks; default twice worker count")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 exports all rows")
    parser.add_argument("--force", action="store_true", help="Explicitly replace an existing fact artifact")
    args = parser.parse_args(argv)
    args.max_pending = args.workers * 2 if args.max_pending is None else args.max_pending
    if min(args.workers, args.chunk_size, args.max_pending) < 1 or args.limit < 0:
        parser.error("Worker/chunk/window sizes must be positive; limit must be nonnegative")
    if not 1 <= args.compression_level <= 9:
        parser.error("Compression level must be in [1, 9]")
    if not args.output.name.endswith(".jsonl.gz"):
        parser.error("Output filename must end in .jsonl.gz")
    return args


def main(argv=None):
    init_worker()
    export(parse_args(argv))


if __name__ == "__main__":
    main()
