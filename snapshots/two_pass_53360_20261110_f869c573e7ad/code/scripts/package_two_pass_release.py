#!/usr/bin/env python3
"""Build an allowlisted, bounded-size release after the complete two-pass audit.

No network, API calls, raw model reasoning, credentials or database files are
published. The model's second pass is not independent chemical verification.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from audit_natural_provenance import FEATURE_NAMES, file_hash

SCHEMA = "two_pass_release_manifest/1"
TARGET = 100000
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_TOTAL_BYTES = 1536 * 1024 * 1024
CSV_FIELDS = ["source_index", "chembl_id", "smiles", *FEATURE_NAMES, "property_vector", "first_description", "description",
              "first_description_sha256", "description_sha256", "facts_sha256", "text_changed", "first_pass", "second_pass", "chemistry_verified"]
META_FIELDS = {"schema_version", "status", "generation_mode", "model", "model_requested", "response_id",
              "run_fingerprint", "prompt_sha256", "selected_attempt_index", "selected_pass_number", "origin",
              "chemical_review_status", "review_reasons", "word_count", "word_count_in_target"}
TRAINING_FIELDS = {"source_index","chembl_id","smiles","facts_sha256","properties","property_vector","schema_version",
                   "first_description","description","first_description_sha256","description_sha256","text_changed",
                   "first_pass","second_pass","chemistry_verified"}
SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\b(?:ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~-]{24,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r'''(?i)["']?(?:api[_-]?key|secret[_-]?key|password|access[_-]?token)["']?\s*[:=]\s*["'][A-Za-z0-9_./+\-=]{24,}["']'''),
]
FORBIDDEN_NAMES = {".env", ".git", ".ssh", ".aws", "credentials", "credentials.json", "environment.yml", "environment.yaml", "keys", "keys.txt", "api_keys.json", "api_keys.txt"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_relative(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Unsafe release path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or ".." in path.parts or "\\" in value or ":" in value or any(ord(c)<32 for c in value):
        raise ValueError("Unsafe release path")
    if any(part.lower() in FORBIDDEN_NAMES or part.lower().startswith(".env") for part in path.parts) or path.suffix.lower() in {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".log", ".zip", ".tar", ".tgz", ".7z"} or value.lower().endswith(".tar.gz"):
        raise ValueError("Forbidden release asset")
    return path.as_posix()


def secret_scan(path):
    """Conservative pattern scan; never return matched credential values."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("Release contains symlink or non-file")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                raise ValueError(f"Potential secret in {path.name}, line {number}; matching value withheld")


def check_gate(audit, fingerprint, target=None):
    target = TARGET if target is None else target
    required = {"schema_version": "two_pass_full_integrity_audit/1", "valid": True, "publish_ready": True,
                "target_count": target, "complete_pairs": target, "first_pass_accepted": target,
                "second_pass_accepted": target, "missing_pairs": 0, "run_fingerprint": fingerprint,
                "chemistry_verified": False}
    for name, value in required.items():
        if type(audit.get(name)) is not type(value) or audit.get(name) != value:
            raise ValueError(f"Release gate failed: {name}")
    if audit.get("error_count") != 0:
        raise ValueError("Integrity audit reports errors")


def phase_metadata(row):
    return {key: row[key] for key in sorted(META_FIELDS) if key in row}


def training_row(row):
    """Explicit projection prevents accidental raw-response/reasoning leakage."""
    first, second = row.get("first_pass"), row.get("second_pass")
    if row.get("pair_status") != "complete_pair" or not isinstance(first, dict) or not isinstance(second, dict):
        raise ValueError("Incomplete pair reached release")
    for stage in (first, second):
        if stage.get("status") != "accepted_format" or not isinstance(stage.get("description"), str) or not stage["description"].strip():
            raise ValueError("Missing accepted description")
        if hashlib.sha256(stage["description"].encode()).hexdigest() != stage.get("description_sha256"):
            raise ValueError("Description digest mismatch at release")
        if stage.get("chemistry_verified") is not False:
            raise ValueError("Unsubstantiated chemical verification field")
    out = {key: row[key] for key in ("source_index", "chembl_id", "smiles", "facts_sha256", "properties", "property_vector")}
    out.update(schema_version="two_pass_training_record/1", first_description=first["description"],
               description=second["description"], first_description_sha256=first["description_sha256"],
               description_sha256=second["description_sha256"], text_changed=first["description"] != second["description"],
               first_pass=phase_metadata(first), second_pass=phase_metadata(second), chemistry_verified=False)
    return out


def cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else "" if value is None else str(value)


def open_gzip_text(path):
    raw = Path(path).open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, io.TextIOWrapper(compressed, encoding="utf-8", newline="")


def verify_training_shards(directory, entries):
    json_entries = sorted((e for e in entries if e.get("role") == "paired_training_jsonl"), key=lambda e:e["path"])
    csv_entries = sorted((e for e in entries if e.get("role") == "paired_training_csv"), key=lambda e:e["path"])
    if not json_entries or csv_entries and len(csv_entries) != len(json_entries):
        raise ValueError("Missing or unequal training shard sets")
    count, seen = 0, set()
    for shard, entry in enumerate(json_entries):
        local = 0
        with gzip.open(directory/entry["path"],"rt",encoding="utf-8") as jh:
            csv_handle = gzip.open(directory/csv_entries[shard]["path"],"rt",encoding="utf-8",newline="") if csv_entries else None
            try:
                reader = csv.DictReader(csv_handle) if csv_handle else None
                if reader and reader.fieldnames != CSV_FIELDS: raise ValueError("Released CSV schema differs")
                paired = itertools.zip_longest(jh, reader) if reader else ((line,None) for line in jh)
                for line, flat in paired:
                    if line is None or reader and flat is None: raise ValueError("Released CSV and JSONL lengths differ")
                    row = json.loads(line); count += 1; local += 1
                    if set(row) != TRAINING_FIELDS or row.get("schema_version") != "two_pass_training_record/1":
                        raise ValueError("Released training record has unexpected fields/schema")
                    for phase in ("first_pass","second_pass"):
                        metadata=row.get(phase)
                        if not isinstance(metadata,dict) or not set(metadata) <= META_FIELDS or metadata.get("status") != "accepted_format":
                            raise ValueError("Released phase metadata contains unexpected/private fields")
                        origin=metadata.get("origin")
                        if origin is not None and (not isinstance(origin,dict) or set(origin) != {"kind","journal_path","batch_id","record_index","original_record_sha256","run_fingerprint"}):
                            raise ValueError("Released origin metadata contains unexpected fields")
                    if type(row.get("source_index")) is not int or row.get("source_index") != count or row.get("chembl_id") in seen or not isinstance(row.get("chembl_id"),str):
                        raise ValueError("Released source coverage/order/uniqueness mismatch")
                    seen.add(row["chembl_id"])
                    props, vec = row.get("properties"), row.get("property_vector")
                    if not isinstance(props,dict) or set(props) != set(FEATURE_NAMES) or any(type(v) not in (float,int) or not math.isfinite(v) for v in props.values()):
                        raise ValueError("Released properties are not 24 finite numbers")
                    if not isinstance(vec,list) or len(vec) != 24 or any(type(v) not in (float,int) or not math.isfinite(v) for v in vec) or vec != [props[k] for k in FEATURE_NAMES]:
                        raise ValueError("Released property vector differs")
                    for text_key, hash_key in (("first_description","first_description_sha256"),("description","description_sha256")):
                        if not isinstance(row.get(text_key),str) or not row[text_key].strip() or hashlib.sha256(row[text_key].encode()).hexdigest() != row.get(hash_key):
                            raise ValueError("Released description/hash mismatch")
                    if row.get("chemistry_verified") is not False or type(row.get("text_changed")) is not bool or row["text_changed"] != (row["first_description"] != row["description"]):
                        raise ValueError("Released chemical/change status differs")
                    if reader:
                        values = {**row,**props}
                        if any(flat[k] != cell(values.get(k)) for k in CSV_FIELDS): raise ValueError("Released CSV differs from JSONL")
                if local != entry.get("rows") or csv_entries and local != csv_entries[shard].get("rows"):
                    raise ValueError("Actual released shard row count differs from manifest")
            finally:
                if csv_handle: csv_handle.close()
    if count != TARGET: raise ValueError("Actual released data does not contain the fixed 100000 rows")


def verify_release(directory, expected_fingerprint=None):
    """Recheck a finished release; useful to offline reviewers and publisher."""
    directory = Path(directory)
    if (directory/"release_manifest.json").is_symlink() or (directory/"release_manifest.json").stat().st_size >= MAX_FILE_BYTES:
        raise ValueError("Manifest is not an ordinary file below the publication size ceiling")
    manifest = read_json(directory / "release_manifest.json")
    if manifest.get("schema_version") != SCHEMA or expected_fingerprint and manifest.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("Release manifest schema/fingerprint mismatch")
    files = manifest.get("files", [])
    names = [safe_relative(entry["path"]) for entry in files]
    if len(names) != len({name.casefold() for name in names}):
        raise ValueError("Duplicate release file path")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    if actual != set(names) | {"release_manifest.json"}:
        raise ValueError("Release contains missing or unlisted files")
    for entry in files:
        path = directory / entry["path"]
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != directory.parent):
            raise ValueError("Release path traverses a symlink")
        if path.stat().st_size != entry["size_bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError("Release file hash/size mismatch")
        if path.stat().st_size >= MAX_FILE_BYTES:
            raise ValueError("Release file exceeds normal Git size budget")
        secret_scan(path)
    for role in ("paired_training_jsonl", "paired_training_csv"):
        selected = [e for e in files if e.get("role") == role]
        if (selected or role == "paired_training_jsonl") and (any(type(e.get("rows")) is not int or e["rows"] <= 0 for e in selected) or sum(e["rows"] for e in selected) != TARGET):
            raise ValueError("Release shard row totals differ from the fixed denominator")
    if sum((directory/name).stat().st_size for name in actual) >= MAX_TOTAL_BYTES:
        raise ValueError("Release total exceeds ordinary Git publication budget")
    verify_training_shards(directory, files)
    audit_ref = manifest["integrity_audit"]
    audit_path = directory / safe_relative(audit_ref["path"])
    if audit_ref.get("schema") != "two_pass_full_integrity_audit/1" or audit_ref.get("valid") is not True or audit_ref.get("run_fingerprint") != manifest["run_fingerprint"]:
        raise ValueError("Integrity audit reference differs from release contract")
    if audit_ref["sha256"] != file_hash(audit_path):
        raise ValueError("Audit reference hash mismatch")
    audit = read_json(audit_path)
    check_gate(audit, manifest["run_fingerprint"])
    if manifest.get("publish_ready") is not True:
        raise ValueError("Manifest is not publish ready")
    for field in ("target_count", "complete_pairs", "first_pass_accepted", "second_pass_accepted", "missing_pairs", "chemistry_verified"):
        if type(manifest.get(field)) is not type(audit[field]) or manifest.get(field) != audit[field]:
            raise ValueError("Manifest and audit counts disagree")
    secret_scan(directory / "release_manifest.json")
    return manifest


def package(run_dir, audit_path, output_dir, assets_manifest=None, rows_per_shard=2000, include_csv=False):
    run_dir, audit_path, output_dir = Path(run_dir).resolve(), Path(audit_path).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise ValueError("Release destination must be new; existing releases are never overwritten")
    if not 1 <= rows_per_shard <= 10000:
        raise ValueError("Shard row count must be in 1..10000")
    audit = read_json(audit_path)
    config = read_json(run_dir / "run_config.json")
    check_gate(audit, config["run_fingerprint"])
    paired = run_dir / "paired/annotations.jsonl"
    snapshots = {str(Path(path).resolve()): value for path, value in audit.get("input_hashes", {}).items()}
    exports = {str(Path(entry["path"]).resolve()): entry["sha256"] for entry in audit.get("data_exports", [])}
    if exports.get(str(paired)) != file_hash(paired):
        raise ValueError("Paired export is not the audited file")
    for path, sha in {**snapshots, **exports}.items():
        if file_hash(path) != sha:
            raise ValueError("Audited input changed before packaging")
    audit_sha = file_hash(audit_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=output_dir.name + ".building-", dir=output_dir.parent))
    entries = []
    def register(relative, role, rows=None):
        path = temporary / relative
        secret_scan(path)
        if path.stat().st_size >= MAX_FILE_BYTES:
            raise ValueError("Release file too large; reduce rows-per-shard")
        entries.append({"path": relative, "role": role, "sha256": file_hash(path), "size_bytes": path.stat().st_size,
                        **({"rows": rows} if rows is not None else {})})
    try:
        (temporary / "data").mkdir()
        fields = CSV_FIELDS
        count, seen, handles, current_rows, shard = 0, set(), None, 0, 0
        def close_shard():
            if handles:
                for raw, handle in handles:
                    handle.close(); raw.close()
                register(f"data/paired-{shard:05d}.jsonl.gz", "paired_training_jsonl", current_rows)
                if include_csv: register(f"data/paired-{shard:05d}.csv.gz", "paired_training_csv", current_rows)
        with paired.open(encoding="utf-8") as source:
            for line in source:
                row = training_row(json.loads(line))
                count += 1
                if type(row.get("source_index")) is not int or row["source_index"] != count or row["chembl_id"] in seen:
                    raise ValueError("Release source order/uniqueness mismatch")
                seen.add(row["chembl_id"])
                if (count - 1) % rows_per_shard == 0:
                    close_shard(); shard += 1; current_rows = 0
                    handles = [open_gzip_text(temporary / f"data/paired-{shard:05d}.{ext}.gz") for ext in (("jsonl", "csv") if include_csv else ("jsonl",))]
                    if include_csv:
                        writer = csv.DictWriter(handles[1][1], fieldnames=fields)
                        writer.writeheader()
                handles[0][1].write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                flattened = {**row, **row["properties"]}
                if include_csv: writer.writerow({key: cell(flattened.get(key)) for key in fields})
                current_rows += 1
        close_shard()
        if count != TARGET:
            raise ValueError("Release must contain exactly 100000 unique complete pairs")
        shutil.copyfile(audit_path, temporary / "integrity_audit.tmp")
        (temporary / "metadata").mkdir()
        (temporary / "integrity_audit.tmp").rename(temporary / "metadata/integrity_audit.json")
        register("metadata/integrity_audit.json", "integrity_audit")
        schema = {"schema_version": "two_pass_training_record/1", "record_count": TARGET,
                  "feature_names": FEATURE_NAMES, "property_vector_order": "feature_names",
                  "description": "Exact second-pass final API text", "first_description": "Exact accepted first-pass final API text",
                  "text_changed": "Exact string inequality, not a correctness measure", "chemistry_verified": False,
                  "raw_api_reasoning": "Retained privately in original journals; excluded from this release",
                  "source_file_hashes": {Path(config[path_key]).name:config[hash_key] for path_key,hash_key in
                      (("input_path","input_sha256"),("molecules_path","molecules_sha256"),("properties_path","properties_sha256")) if path_key in config and hash_key in config},
                  "limitations": ["Format and provenance verification do not establish chemical accuracy.",
                                  "Model self-review is not independent chemical verification.",
                                  "Stored descriptor values describe the source graph and are separate from generated text."]}
        write_json(temporary / "metadata/data_schema.json", schema)
        register("metadata/data_schema.json", "data_schema")
        asset_roles = set()
        if assets_manifest:
            assets = read_json(assets_manifest)
            for asset in assets["files"]:
                name = safe_relative(asset["path"])
                if not name.startswith(("code/", "prompts/", "docs/", "licenses/")) or name in {e["path"] for e in entries}:
                    raise ValueError("Assets must have unique code/prompts/docs/licenses destinations")
                source = Path(asset["source"])
                if source.is_symlink() or file_hash(source) != asset["sha256"]:
                    raise ValueError("Allowlisted asset hash mismatch")
                destination = temporary / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                register(name, asset.get("role", "supporting_asset"))
                asset_roles.add(asset.get("role"))
        if not {"source_license","source_attribution","first_prompt","review_prompt","generator_code","source_documentation"} <= asset_roles:
            raise ValueError("Release assets omit source licensing, attribution, prompts, generation code or source documentation")
        readme = ("# ChEMBL two-pass molecular descriptions\n\n"
                  "100,000 source molecules with an initial English description and its minimal-review output.\n\n"
                  "Files in `data/` are gzip JSONL shards in original source order; matching CSV shards are optional. "
                  "`description` is the second-pass text; `first_description` preserves the exact first pass. "
                  "The 24-property vector order and source hashes are in `metadata/data_schema.json`.\n\n"
                  "All records passed source, request, response, and export integrity checks. This is not a certificate of chemical accuracy. "
                  "Model self-review is not independent verification; `chemistry_verified` remains false. "
                  "Changed/unchanged counts measure text equality only. Raw response reasoning, credentials and local databases are excluded.\n\n"
                  "SMILES and underlying activity sources are ChEMBL 37; numeric properties were computed with RDKit, "
                  "and descriptions were generated and reviewed by the model identified in each record. "
                  "See `licenses/` for the preserved ChEMBL license and required attribution; no new code license is asserted here.\n\n"
                  "Check `release_manifest.json` for the complete allowlist and SHA-256 hashes.\n")
        (temporary / "README.md").write_text(readme, encoding="utf-8")
        register("README.md", "readme")
        manifest = {"schema_version": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    **{key: audit[key] for key in ("run_fingerprint", "target_count", "complete_pairs", "first_pass_accepted", "second_pass_accepted", "missing_pairs", "chemistry_verified")},
                    "publish_ready":True,
                    "integrity_audit": {"path": "metadata/integrity_audit.json", "sha256": audit_sha, "schema": audit["schema_version"], "valid": True,"run_fingerprint":config["run_fingerprint"]},
                    "files": entries, "secret_scan": {"passed": True, "scope": "Every allowlisted UTF-8 file, including decompressed shards; conservative credential patterns, not an exhaustive secret detector"},
                    "maximum_file_bytes_exclusive": MAX_FILE_BYTES,"maximum_total_bytes_exclusive":MAX_TOTAL_BYTES}
        write_json(temporary / "release_manifest.json", manifest)
        verify_release(temporary, config["run_fingerprint"])
        for path, sha in {**snapshots, **exports, str(audit_path): audit_sha}.items():
            if file_hash(path) != sha:
                raise ValueError("Audited input changed during packaging")
        temporary.rename(output_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assets-manifest", type=Path)
    parser.add_argument("--rows-per-shard", type=int, default=2000)
    parser.add_argument("--include-csv",action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-run-fingerprint")
    args = parser.parse_args(argv)
    if args.verify_only:
        result = verify_release(args.output_dir, args.expected_run_fingerprint)
    else:
        if not args.run_dir or not args.audit:
            parser.error("--run-dir and --audit are required for packaging")
        result = package(args.run_dir, args.audit, args.output_dir, args.assets_manifest, args.rows_per_shard,args.include_csv)
    print(json.dumps({"schema_version": result["schema_version"], "complete_pairs": result["complete_pairs"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
