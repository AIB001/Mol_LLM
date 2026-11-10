#!/usr/bin/env python3
"""Publish an explicitly authorized incomplete two-pass snapshot, never a full release.

Only common Git/path/credential-scan helpers are reused from the frozen full
publisher. Its full-completion verifier and COUNT_FIELDS are never modified.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

import publish_two_pass_release as safety


SCHEMA_VERSION = "two_pass_snapshot_publisher/1"
MANIFEST_NAME = "snapshot_manifest.json"
TARGET_COUNT = 100000
EXPECTED_PAIRS = 53360
EXPECTED_FIRST_ONLY = 2190


def verify_snapshot_rows(directory, entries, expected_pairs, expected_first_only):
    """Read actual data, retaining only IDs/indices; never infer chemical quality."""
    seen_indices, seen_ids = set(), set()
    forbidden = {"reasoning_content", "request_payload", "attempt_log", "response"}
    def no_raw(value):
        if isinstance(value, dict):
            safety.require(not forbidden.intersection(value), "Raw API payload appeared in published data")
            for nested in value.values():no_raw(nested)
        elif isinstance(value, list):
            for nested in value:no_raw(nested)
    def text_hash_matches(row, text_key, hash_key):
        value = row.get(text_key)
        safety.require(isinstance(value, str) and bool(value.strip()) and
                       hashlib.sha256(value.encode()).hexdigest() == row.get(hash_key), "Snapshot description SHA mismatch")
    for role, expected in (("paired_training_jsonl", expected_pairs), ("first_pass_only_jsonl", expected_first_only)):
        total, previous = 0, 0
        for entry in sorted((e for e in entries if e["role"] == role), key=lambda e:e["path"]):
            path = directory / entry["path"]
            opener = gzip.open if path.suffix == ".gz" else open
            count = 0
            with opener(path, "rt", encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line); count += 1; total += 1
                    safety.require(isinstance(row, dict), "Snapshot data row is not an object")
                    no_raw(row)
                    index, cid = row.get("source_index"), row.get("chembl_id")
                    safety.require(type(index) is int and previous < index <= TARGET_COUNT and index not in seen_indices,
                                   "Snapshot source indices are duplicated, unordered or outside the population")
                    safety.require(isinstance(cid, str) and cid and cid not in seen_ids, "Snapshot IDs overlap or are missing")
                    previous = index; seen_indices.add(index); seen_ids.add(cid)
                    safety.require(row.get("chemistry_verified") is False, "Data row claims chemical certification")
                    safety.require(isinstance(row.get("first_pass"), dict) and row["first_pass"].get("status") == "accepted_format",
                                   "Snapshot row lacks accepted first pass")
                    text_hash_matches(row, "description", "description_sha256")
                    if role == "paired_training_jsonl":
                        safety.require(row.get("schema_version") == "two_pass_training_record/1" and
                                       isinstance(row.get("second_pass"), dict) and row["second_pass"].get("status") == "accepted_format",
                                       "Paired snapshot row lacks accepted review")
                        text_hash_matches(row, "first_description", "first_description_sha256")
                        safety.require(type(row.get("text_changed")) is bool and
                                       row["text_changed"] == (row["description"] != row["first_description"]), "Paired exact-change flag mismatch")
                    else:
                        safety.require(row.get("schema_version") == "two_pass_snapshot_first_only_record/1" and
                                       row.get("pair_status") == "awaiting_second_pass" and
                                       row.get("description_origin") == "first_pass_unreviewed" and
                                       row.get("review_completed") is False and row.get("second_pass") is None and
                                       row.get("second_description") is None, "First-only row is mislabeled as reviewed")
            safety.require(count == entry["rows"], "Actual snapshot shard count differs from manifest")
        safety.require(total == expected, "Actual snapshot dataset count differs from authorization")


def verify_snapshot(directory: Path, expected_fingerprint: str,
                    expected_pairs=EXPECTED_PAIRS, expected_first_only=EXPECTED_FIRST_ONLY):
    directory = Path(directory)
    require = safety.require
    require(re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint) is not None, "Expected fingerprint must be an exact SHA256")
    require(type(expected_pairs) is int and 0 < expected_pairs < TARGET_COUNT, "Snapshot must be explicitly incomplete")
    require(type(expected_first_only) is int and 0 <= expected_first_only <= TARGET_COUNT - expected_pairs, "Invalid first-only expectation")
    require(directory.is_dir() and not directory.is_symlink(), "Snapshot directory missing or a symlink")
    manifest_path = directory / MANIFEST_NAME
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Missing ordinary snapshot_manifest.json")
    require(manifest_path.stat().st_size < safety.MAX_FILE_BYTES, "Manifest reaches file-size ceiling")
    manifest = safety.read_json(manifest_path)
    require(manifest.get("schema_version") == "two_pass_snapshot_manifest/1", "This is not a snapshot manifest")
    expected = {"target_count": TARGET_COUNT, "exported_pair_count": expected_pairs,
                "exported_first_pass_only_count": expected_first_only,
                "first_pass_accepted": expected_pairs + expected_first_only,
                "second_pass_accepted": expected_pairs, "remaining_pairs": TARGET_COUNT - expected_pairs}
    for key, value in expected.items():
        require(type(manifest.get(key)) is int and manifest[key] == value, f"Snapshot {key} mismatch")
    require(manifest.get("run_fingerprint") == expected_fingerprint, "Snapshot fingerprint mismatch")
    require(manifest.get("snapshot_complete") is False and manifest.get("chemistry_verified") is False,
            "Snapshot must not claim full completion or chemical certification")
    require(manifest.get("publish_ready") is not True, "Snapshot must not claim full-release publish readiness")
    relative = str(safety.safe_relative(manifest.get("snapshot_path")))
    pattern = rf"snapshots/two_pass_{expected_pairs}_[0-9]{{8}}_{expected_fingerprint[:12]}"
    require(re.fullmatch(pattern, relative) is not None, "Snapshot repository path is not the expected dated namespace")
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "Snapshot file manifest is empty")
    names, by_name = {MANIFEST_NAME}, {}
    for entry in entries:
        require(isinstance(entry, dict), "Invalid snapshot file entry")
        name = str(safety.safe_relative(entry.get("path")))
        require(name.casefold() not in {x.casefold() for x in names}, "Duplicate/case-colliding/self-referencing snapshot path")
        names.add(name); by_name[name] = entry
        role = entry.get("role")
        require(isinstance(role, str) and role and not any(x in role.lower() for x in
                ("raw_api", "reasoning", "request_payload", "secret", "credential")), "Private/raw payload role is not publishable")
        path = directory / name
        require(path.is_file() and not path.is_symlink(), f"Missing ordinary snapshot file: {name}")
        for parent in path.parents:
            if parent == directory:
                break
            require(not parent.is_symlink(), "Snapshot path traverses a symlink")
        require(type(entry.get("size_bytes")) is int and entry["size_bytes"] == path.stat().st_size,
                f"Snapshot file-size mismatch: {name}")
        require(entry["size_bytes"] < safety.MAX_FILE_BYTES, "Snapshot file reaches 40 MiB ceiling")
        require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid snapshot file SHA256")
        require(safety.digest(path) == entry["sha256"], f"Snapshot file hash mismatch: {name}")
        safety.scan_file(path)
    actual = set()
    for path in directory.rglob("*"):
        require(not path.is_symlink(), "Snapshot contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
        else:
            require(path.is_dir(), "Snapshot contains a special file")
    require(actual == names, "Snapshot has missing or unlisted files")
    require(sum(e["size_bytes"] for e in entries) + manifest_path.stat().st_size < safety.MAX_PAYLOAD_BYTES,
            "Snapshot reaches 1.5 GiB push budget")
    for role, expected_rows in (("paired_training_jsonl", expected_pairs), ("first_pass_only_jsonl", expected_first_only)):
        shards = [e for e in entries if e["role"] == role]
        require(all(type(e.get("rows")) is int and 0 < e["rows"] <= expected_rows for e in shards), "Invalid snapshot shard row counts")
        require(sum(e["rows"] for e in shards) == expected_rows, f"Incomplete {role} shard coverage")
    reference = manifest.get("integrity_audit")
    require(isinstance(reference, dict), "Snapshot lacks an integrity audit")
    audit_name = str(safety.safe_relative(reference.get("path")))
    require(audit_name in by_name and reference.get("sha256") == by_name[audit_name]["sha256"], "Snapshot audit reference hash mismatch")
    require(reference.get("schema") == "two_pass_full_integrity_audit/1" and reference.get("valid") is True
            and reference.get("publish_ready") is False and reference.get("run_fingerprint") == expected_fingerprint,
            "Snapshot audit reference must be valid but not ready for full release")
    audit = safety.read_json(directory / audit_name)
    require(audit.get("schema_version") == "two_pass_full_integrity_audit/1" and audit.get("valid") is True
            and audit.get("publish_ready") is False and audit.get("chemistry_verified") is False,
            "Snapshot audit is invalid or falsely claims full readiness")
    require(audit.get("valid_source_and_provenance") is True and type(audit.get("error_count")) is int
            and audit["error_count"] == 0, "Snapshot source/provenance audit must have exactly zero errors")
    require(audit.get("run_fingerprint") == expected_fingerprint, "Snapshot audit fingerprint mismatch")
    for key, expected_value in {"target_count": TARGET_COUNT, "complete_pairs": expected_pairs,
                                "first_pass_accepted": expected_pairs + expected_first_only,
                                "second_pass_accepted": expected_pairs, "missing_pairs": TARGET_COUNT - expected_pairs}.items():
        require(type(audit.get(key)) is int and audit[key] == expected_value, f"Snapshot audit {key} mismatch")
    verify_snapshot_rows(directory, entries, expected_pairs, expected_first_only)
    safety.scan_file(manifest_path)
    return manifest, safety.digest(manifest_path), sorted(names)


def stage_snapshot(checkout, directory, manifest, names):
    relative = manifest["snapshot_path"]
    target = checkout / relative
    for name in names:
        source, destination = directory / name, target / name
        for parent in destination.parents:
            if parent == checkout:
                break
            safety.require(not parent.is_symlink(), "Snapshot destination traverses a symlink")
        if destination.exists():
            safety.require(destination.is_file() and not destination.is_symlink()
                           and safety.digest(destination) == safety.digest(source), "Existing snapshot bytes differ; refusing overwrite")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        safety.require(safety.digest(destination) == safety.digest(source), "Snapshot copy differs")
    safety.require({p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == set(names), "Destination snapshot has extra files")
    marker = f"<!-- two-pass-snapshot:{relative} -->".encode()
    readme = checkout / "README.md"
    safety.require(not readme.is_symlink(), "README is a symlink")
    old = readme.read_bytes() if readme.exists() else b"# Mol_LLM\n"
    if marker not in old:
        pairs, first_only = manifest["exported_pair_count"], manifest["exported_first_pass_only_count"]
        previews = [e["path"] for e in manifest["files"] if e["role"] in {"preview_json", "preview_csv"}]
        links = " · ".join(f"[{name}]({relative}/{name})" for name in previews)
        addition = (f"\n\n{marker.decode()}\n## 当前结果 / Current snapshot — 全量尚未完成\n\n"
                    f"本次包含 **{pairs:,} 条双轮完整配对**，另有 **{first_only:,} 条仅第一轮已接受**的描述，单独保存且尚无已接受的二轮结果（可能已尝试但未成功）。"
                    f"目标为 {TARGET_COUNT:,} 条双轮配对，目前仍缺 {TARGET_COUNT - pairs:,} 条；这是阶段快照，不是最终全量发布。\n\n"
                    f"- [快照说明 / Snapshot README]({relative}/README.md)\n"
                    f"- [数据目录 / Data]({relative}/data/)\n"
                    f"- [文件清单及校验值 / Manifest]({relative}/{MANIFEST_NAME})\n"
                    + (f"- 直接预览 / Preview: {links}\n" if links else "") +
                    f"\nThis snapshot contains {pairs:,} completed first-pass/review pairs and a separate {first_only:,} first-pass-only set with no accepted second-pass result. "
                    "Source/provenance checks passed for the saved state; they do not certify chemical accuracy. "
                    "Raw reasoning, request logs and credentials are excluded.\n").encode()
        readme.write_bytes(old + addition)
    safety.git(checkout, "add", "--", relative, "README.md")
    required = {relative + "/" + name for name in names}
    actual = set(filter(None, safety.git(checkout, "ls-files", "-z", "--", relative).stdout.split("\0")))
    safety.require(actual == required, "Git ignored or omitted a snapshot file")
    staged = set(filter(None, safety.git(checkout, "diff", "--cached", "--name-only", "-z").stdout.split("\0")))
    safety.require(staged <= required | {"README.md"}, "Unexpected staged files")
    statuses = safety.git(checkout, "diff", "--cached", "--name-status").stdout.splitlines()
    safety.require(all(x.split("\t", 1)[0] in {"A", "M"} for x in statuses), "Staged deletion/rename is forbidden")
    return relative, bool(staged)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--expected-run-fingerprint", required=True)
    parser.add_argument("--expected-pairs", type=int, default=EXPECTED_PAIRS)
    parser.add_argument("--expected-first-only", type=int, default=EXPECTED_FIRST_ONLY)
    parser.add_argument("--checkout-dir", type=Path, default=Path("/mnt/e/My_Project/Mol_LLM_github_publish"))
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)
    directory, checkout, state_path = args.snapshot_dir.absolute(), args.checkout_dir.absolute(), args.state_file.absolute()
    safety.require(not directory.resolve().is_relative_to(checkout.resolve()) and not checkout.resolve().is_relative_to(directory.resolve()), "Snapshot and checkout must be separate")
    safety.require(not state_path.resolve().is_relative_to(directory.resolve()) and not state_path.resolve().is_relative_to(checkout.resolve()), "State must be outside payload directories")
    state_path.parent.mkdir(parents=True, exist_ok=True); checkout.parent.mkdir(parents=True, exist_ok=True)
    # Same checkout lock as the full publisher prevents overlapping publications.
    with state_path.with_suffix(state_path.suffix + ".lock").open("a") as lock, (checkout.parent / ("." + checkout.name + ".publication.lock")).open("a") as checkout_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(checkout_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another publication process owns the lock", file=sys.stderr); return 2
        previous = safety.read_json(state_path) if state_path.exists() else {}
        state = {"schema_version": SCHEMA_VERSION, "repository": safety.REPOSITORY, "branch": "main",
                 "started_at_utc": safety.utc_now(), "snapshot_dir": str(directory), "checkout_dir": str(checkout),
                 "expected_run_fingerprint": args.expected_run_fingerprint, "expected_pairs": args.expected_pairs,
                 "expected_first_only": args.expected_first_only, "snapshot_complete": False, "chemistry_verified": False,
                 "push_requested": args.push, "published": False, "publisher_sha256": safety.digest(Path(__file__)),
                 "shared_git_helper_sha256": safety.digest(Path(safety.__file__))}
        try:
            manifest, checksum, names = verify_snapshot(directory, args.expected_run_fingerprint, args.expected_pairs, args.expected_first_only)
            state.update(status="verified", manifest_sha256=checksum, snapshot_path=manifest["snapshot_path"], payload_files=len(names))
            if previous.get("manifest_sha256") == checksum and previous.get("commit_sha"):
                state["commit_sha"] = previous["commit_sha"]
            safety.atomic_json(state_path, state)
            if not args.push:
                print('{"status":"verified","published":false,"snapshot_complete":false}'); return 0
            state["remote_head_before"] = safety.synchronize_checkout(checkout, previous, checksum)
            relative, changed = stage_snapshot(checkout, directory, manifest, names)
            safety.require(verify_snapshot(directory, args.expected_run_fingerprint, args.expected_pairs, args.expected_first_only)[1] == checksum, "Snapshot changed during staging")
            safety.require(verify_snapshot(checkout / relative, args.expected_run_fingerprint, args.expected_pairs, args.expected_first_only)[1] == checksum, "Staged snapshot differs")
            if changed:
                name, email = safety.authenticated_identity()
                safety.git(checkout, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m",
                           f"Publish partial two-pass snapshot: {args.expected_pairs} pairs and {args.expected_first_only} first-only")
            commit = safety.object_id(checkout, "HEAD")
            safety.require(commit is not None, "Snapshot commit is missing")
            state.update(status="committed_pending_push", commit_sha=commit)
            safety.atomic_json(state_path, state)
            safety.git(checkout, "push", "--no-follow-tags", "origin", "HEAD:refs/heads/main")
            remote = safety.git(checkout, "ls-remote", "origin", "refs/heads/main").stdout.split()
            safety.require(remote and remote[0] == commit, "Remote did not expose the exact snapshot commit")
            state.update(status="published", published=True, completed_at_utc=safety.utc_now(), remote_commit_sha=commit,
                         snapshot_url=f"https://github.com/{safety.REPOSITORY}/tree/{commit}/{relative}")
            safety.atomic_json(state_path, state)
            print(json.dumps({"status":"published","snapshot_complete":False,"commit_sha":commit,"snapshot_url":state["snapshot_url"]}))
            return 0
        except Exception as exc:
            error = str(exc) if isinstance(exc, safety.PublicationBlocked) else type(exc).__name__
            state.update(status="blocked", error=error, updated_at_utc=safety.utc_now())
            safety.atomic_json(state_path, state)
            print("Snapshot publication blocked: " + error, file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
