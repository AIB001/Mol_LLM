#!/usr/bin/env python3
"""Publish only a complete, audited two-pass release via a separate Git checkout.

No network mutation occurs without --push. The package must already have been
built from the final 100000-pair snapshot; this program never annotates data.
"""
from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone


SCHEMA_VERSION = "two_pass_release_publisher/1"
REPOSITORY = "AIB001/Mol_LLM"
REMOTE_URL = "https://github.com/AIB001/Mol_LLM.git"
MAX_FILE_BYTES = 40 * 1024 * 1024
MAX_PAYLOAD_BYTES = 1536 * 1024 * 1024
COUNT_FIELDS = {
    "target_count": 100000,
    "complete_pairs": 100000,
    "first_pass_accepted": 100000,
    "second_pass_accepted": 100000,
    "missing_pairs": 0,
}
SECRET_PATTERNS = [
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb'(?i)["\']?(?:api[_-]?key|access[_-]?token|client[_-]?secret)["\']?\s*[:=]\s*["\'][A-Za-z0-9_/-]{20,}["\']'),
]


class PublicationBlocked(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise PublicationBlocked(message)


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(path)


def safe_relative(value):
    require(isinstance(value, str) and value, "Manifest path must be a nonempty string")
    p = PurePosixPath(value)
    require(not p.is_absolute() and "\\" not in value and value == p.as_posix(), "Noncanonical manifest path")
    require(all(x not in {"", ".", ".."} for x in p.parts), "Unsafe manifest path")
    require(not any(ord(x) < 32 for x in value), "Control character in manifest path")
    for part in p.parts:
        name = part.lower()
        require(name not in {".git", ".ssh", ".aws", "credentials", "credentials.json", "keys", "keys.txt", "api_keys.json", "api_keys.txt", "id_rsa", "id_ed25519"}, "Credential/control path in release")
        require(not name.startswith(".env") and not name.endswith((".pem", ".key", ".log")), "Credential/log file in release")
        require(not name.endswith((".tar", ".tar.gz", ".tgz", ".zip", ".7z")), "Opaque archives are not publishable; use enumerated files or gzip data shards")
    return p


def scan_file(path: Path):
    # Scan both compressed dataset text and ordinary code/docs. Report only the
    # offending filename, never matched bytes (which might be a credential).
    opener = gzip.open if path.name.endswith(".gz") else open
    carry = b""
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            data = carry + block
            require(not any(p.search(data) for p in SECRET_PATTERNS), f"Potential credential found in {path.name}")
            carry = data[-1024:]


def check_counts(value, label, fingerprint):
    require(isinstance(value, dict), f"{label} must be an object")
    for key, expected in COUNT_FIELDS.items():
        require(type(value.get(key)) is int and value[key] == expected, f"{label}: {key} must equal {expected}")
    require(value.get("run_fingerprint") == fingerprint, f"{label}: run fingerprint mismatch")
    require(value.get("chemistry_verified") is False, f"{label}: chemistry verification scope must remain explicit")


def verify_release(release_dir: Path, expected_fingerprint: str):
    require(re.fullmatch(r"[0-9a-f]{64}", expected_fingerprint) is not None, "Expected run fingerprint must be an exact SHA256")
    require(release_dir.is_dir() and not release_dir.is_symlink(), "Release directory is missing or a symlink")
    manifest_path = release_dir / "release_manifest.json"
    require(manifest_path.is_file() and not manifest_path.is_symlink(), "Missing ordinary release_manifest.json")
    require(manifest_path.stat().st_size < MAX_FILE_BYTES, "Manifest reaches 40 MiB ceiling")
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == "two_pass_release_manifest/1", "Unexpected manifest schema")
    check_counts(manifest, "Manifest", expected_fingerprint)
    require(manifest.get("publish_ready") is True, "Manifest is not publish_ready")
    files = manifest.get("files")
    require(isinstance(files, list) and len(files) > 0, "Empty release file list")
    seen = {"release_manifest.json"}
    by_name = {}
    for entry in files:
        require(isinstance(entry, dict), "Malformed file entry")
        name = str(safe_relative(entry.get("path")))
        require(name not in seen, "Duplicate or self-referential manifest file")
        require(name.casefold() not in {s.casefold() for s in seen}, "Case-colliding release paths")
        seen.add(name)
        by_name[name] = entry
        require(isinstance(entry.get("role"), str) and entry["role"], "File role is required")
        require(not any(x in entry["role"].lower() for x in ("raw_api", "reasoning", "request_payload", "secret", "credential")), "Private/raw payload role is not publishable")
        path = release_dir / name
        require(path.is_file() and not path.is_symlink(), f"Missing ordinary release file: {name}")
        require(type(entry.get("size_bytes")) is int and entry["size_bytes"] == path.stat().st_size, f"File size mismatch: {name}")
        require(entry["size_bytes"] < MAX_FILE_BYTES, f"File reaches 40 MiB shard ceiling: {name}")
        require(isinstance(entry.get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid file SHA256")
        require(digest(path) == entry["sha256"], f"File hash mismatch: {name}")
        scan_file(path)
    actual = set()
    for path in release_dir.rglob("*"):
        require(not path.is_symlink(), "Release contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(release_dir).as_posix())
        else:
            require(path.is_dir(), "Release contains a special file")
    require(actual == seen, "Release contains unlisted or missing files")
    require(sum(x["size_bytes"] for x in files) + manifest_path.stat().st_size < MAX_PAYLOAD_BYTES, "Release reaches 1.5 GiB single-push budget")
    for role in ("paired_training_jsonl", "paired_training_csv"):
        entries = [entry for entry in files if entry["role"] == role]
        if role == "paired_training_csv" and not entries:
            continue
        target = COUNT_FIELDS["target_count"]
        require(entries and all(type(entry.get("rows")) is int and 0 < entry["rows"] <= target for entry in entries), f"Missing or invalid {role} shard counts")
        require(sum(entry["rows"] for entry in entries) == target, f"Incomplete {role} coverage")
    scan_file(manifest_path)
    audit_ref = manifest.get("integrity_audit")
    require(isinstance(audit_ref, dict), "Manifest lacks integrity_audit")
    audit_name = str(safe_relative(audit_ref.get("path")))
    require(audit_name in by_name, "Integrity audit must be a listed payload file")
    require(audit_ref.get("sha256") == by_name[audit_name]["sha256"], "Integrity audit reference hash mismatch")
    require(audit_ref.get("valid") is True and audit_ref.get("run_fingerprint") == expected_fingerprint, "Invalid integrity audit reference")
    require(audit_ref.get("schema") == "two_pass_full_integrity_audit/1", "Invalid integrity audit reference schema")
    audit = read_json(release_dir / audit_name)
    require(audit.get("schema_version") == "two_pass_full_integrity_audit/1", "Unexpected integrity audit schema")
    check_counts(audit, "Integrity audit", expected_fingerprint)
    require(audit.get("valid") is True and audit.get("publish_ready") is True, "Final integrity audit did not approve publication")
    return manifest, digest(manifest_path), sorted(seen)


def git(checkout, *arguments, check=True):
    cmd = ["git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential", "-C", str(checkout), *arguments]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    if check and proc.returncode:
        # No command output is persisted: avoid leaking a credential accidentally
        # emitted by an external credential helper or unusual Git configuration.
        raise PublicationBlocked(f"Git {arguments[0]} failed (exit {proc.returncode}); inspect the checkout locally")
    return proc


def object_id(checkout, ref):
    proc = git(checkout, "rev-parse", "--verify", ref, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else None


def remote_is_expected(url):
    return url.strip() in {REMOTE_URL, REMOTE_URL.removesuffix(".git"), "git@github.com:AIB001/Mol_LLM.git", "ssh://git@github.com/AIB001/Mol_LLM.git"}


def synchronize_checkout(checkout: Path, previous, manifest_hash):
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(["gh", "repo", "clone", REPOSITORY, str(checkout)], capture_output=True, text=True, timeout=600)
        require(proc.returncode == 0, "Clone failed; no repository was published")
    require((checkout / ".git").is_dir(), "Checkout must be a separate ordinary Git repository")
    require(remote_is_expected(git(checkout, "remote", "get-url", "origin").stdout), "Unexpected Git origin")
    # origin.pushurl overrides the fetch URL, so it must be checked separately.
    push_urls = git(checkout, "remote", "get-url", "--push", "--all", "origin").stdout.splitlines()
    require(len(push_urls) == 1 and remote_is_expected(push_urls[0]), "Unexpected or multiple Git push URLs")
    require(git(checkout, "config", "--bool", "--get", "remote.origin.mirror", check=False).stdout.strip() != "true", "Mirror remotes are forbidden")
    require(not git(checkout, "status", "--porcelain").stdout, "Publishing checkout has uncommitted/untracked changes")
    require(git(checkout, "symbolic-ref", "--quiet", "HEAD").stdout.strip() == "refs/heads/main", "Publishing checkout must be on main")
    git(checkout, "fetch", "origin")
    remote_head = object_id(checkout, "refs/remotes/origin/main")
    local_head = object_id(checkout, "HEAD")
    if remote_head and not local_head:
        # Initialize an unborn local branch without overwriting an existing head.
        git(checkout, "read-tree", "--reset", "-u", remote_head)
        git(checkout, "update-ref", "refs/heads/main", remote_head, "0" * 40)
    elif remote_head and local_head != remote_head:
        if git(checkout, "merge-base", "--is-ancestor", local_head, remote_head, check=False).returncode == 0:
            git(checkout, "merge", "--ff-only", remote_head)
        else:
            require(previous.get("commit_sha") == local_head and previous.get("manifest_sha256") == manifest_hash,
                    "Local branch has unrecognized unpublished history or divergence")
            require(git(checkout, "merge-base", "--is-ancestor", remote_head, local_head, check=False).returncode == 0,
                    "Remote advanced incompatibly; no force push is allowed")
    elif not remote_head and local_head:
        require(previous.get("commit_sha") == local_head and previous.get("manifest_sha256") == manifest_hash,
                "Nonempty local history for an empty remote needs matching publication state")
    return remote_head


def stage_release(checkout: Path, release_dir: Path, fingerprint: str, names):
    relative = Path("releases") / ("two_pass_100000_" + fingerprint[:12])
    target = checkout / relative
    require(not target.is_symlink(), "Release destination is a symlink")
    for name in names:
        source = release_dir / name
        destination = target / name
        for parent in destination.parents:
            if parent == checkout:
                break
            require(not parent.is_symlink(), "Symlink in publication destination")
        if destination.exists():
            require(destination.is_file() and not destination.is_symlink() and digest(destination) == digest(source), "Existing release bytes differ; refusing overwrite")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        require(digest(destination) == digest(source), "Copied file differs from package")
    if target.exists():
        require({p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == set(names), "Destination release contains extra files")
    marker = f"<!-- two-pass-release:{fingerprint} -->".encode()
    readme = checkout / "README.md"
    require(not readme.is_symlink(), "README is a symlink")
    old = readme.read_bytes() if readme.exists() else b"# Mol_LLM\n"
    if marker not in old:
        addition = (f"\n\n{marker.decode()}\n"
                    f"## Two-pass ChEMBL annotation release\n\n"
                    f"[{relative.as_posix()}]({relative.as_posix()}/release_manifest.json) contains 100,000 completed first-pass/review pairs, "
                    "the prompts and code snapshot, property vectors, final response references and integrity reports. "
                    "The integrity checks cover data alignment and API provenance; they do not certify complete chemical accuracy. "
                    "Model reasoning and raw request logs are excluded.\n").encode()
        readme.write_bytes(old + addition)
    git(checkout, "add", "--", relative.as_posix(), "README.md")
    tracked_release = set(filter(None, git(checkout, "ls-files", "-z", "--", relative.as_posix()).stdout.split("\0")))
    require(tracked_release == {f"{relative.as_posix()}/{name}" for name in names},
            "Git ignored or omitted a manifest-listed file; publication must not be partial")
    staged = git(checkout, "diff", "--cached", "--name-only", "-z").stdout.split("\0")
    allowed = {"README.md"} | {f"{relative.as_posix()}/{name}" for name in names}
    require(set(filter(None, staged)) <= allowed, "Unexpected staged files")
    statuses = git(checkout, "diff", "--cached", "--name-status").stdout.splitlines()
    require(all(line.split("\t", 1)[0] in {"A", "M"} for line in statuses), "Staged deletion/rename is forbidden")
    return relative.as_posix(), bool(list(filter(None, staged)))


def authenticated_identity():
    proc = subprocess.run(["gh", "api", "user"], capture_output=True, text=True, timeout=60)
    require(proc.returncode == 0, "GitHub CLI authentication unavailable")
    data = json.loads(proc.stdout)
    login, user_id = data.get("login"), data.get("id")
    require(isinstance(login, str) and re.fullmatch(r"[A-Za-z0-9-]+", login) and type(user_id) is int, "Cannot identify authenticated GitHub account")
    return login, f"{user_id}+{login}@users.noreply.github.com"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--expected-run-fingerprint", required=True)
    parser.add_argument("--checkout-dir", type=Path, default=Path("/mnt/e/My_Project/Mol_LLM_github_publish"))
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument("--push", action="store_true", help="After the full gate passes, commit and perform an ordinary non-force push to main")
    args = parser.parse_args(argv)
    # The state/lock live outside both immutable package and repository payload.
    release = args.release_dir.absolute()
    checkout = args.checkout_dir.absolute()
    state_path = args.state_file.absolute()
    require(not release.resolve().is_relative_to(checkout.resolve()) and not checkout.resolve().is_relative_to(release.resolve()), "Release and checkout must be independent directories")
    require(not state_path.resolve().is_relative_to(release.resolve()) and not state_path.resolve().is_relative_to(checkout.resolve()), "State must be outside package and checkout")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    checkout.parent.mkdir(parents=True, exist_ok=True)
    checkout_lock = checkout.parent / ("." + checkout.name + ".publication.lock")
    with state_path.with_suffix(state_path.suffix + ".lock").open("a") as lock, checkout_lock.open("a") as global_lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another publication process owns the lock", file=sys.stderr)
            return 2
        previous = read_json(state_path) if state_path.exists() else {}
        state = {"schema_version": SCHEMA_VERSION, "repository": REPOSITORY, "branch": "main", "started_at_utc": utc_now(),
                 "expected_run_fingerprint": args.expected_run_fingerprint, "push_requested": args.push, "published": False,
                 "release_dir": str(release), "checkout_dir": str(checkout), "publisher_sha256": digest(Path(__file__))}
        try:
            manifest, manifest_hash, names = verify_release(release, args.expected_run_fingerprint)
            state.update(status="verified", manifest_sha256=manifest_hash, payload_files=len(names),
                         payload_bytes=sum(x["size_bytes"] for x in manifest["files"]))
            if previous.get("manifest_sha256") == manifest_hash and previous.get("commit_sha"):
                state["commit_sha"] = previous["commit_sha"]
            if not args.push:
                atomic_json(state_path, state)
                print(json.dumps({"status": "verified", "published": False, "manifest_sha256": manifest_hash}))
                return 0
            # Preserve a pending commit marker before any external operation.
            if previous.get("manifest_sha256") == manifest_hash and previous.get("commit_sha"):
                state["commit_sha"] = previous["commit_sha"]
            atomic_json(state_path, state)
            state["remote_head_before"] = synchronize_checkout(checkout, previous, manifest_hash)
            release_path, changed = stage_release(checkout, release, args.expected_run_fingerprint, names)
            # Detect package changes during copying before committing or pushing.
            require(verify_release(release, args.expected_run_fingerprint)[1] == manifest_hash, "Package changed during staging")
            verify_release(checkout / release_path, args.expected_run_fingerprint)
            if changed:
                name, email = authenticated_identity()
                git(checkout, "-c", f"user.name={name}", "-c", f"user.email={email}", "commit", "-m",
                    f"Publish complete two-pass ChEMBL 100000 release {args.expected_run_fingerprint[:12]}")
            commit = object_id(checkout, "HEAD")
            require(commit is not None, "No publication commit exists")
            state.update(status="committed_pending_push", commit_sha=commit, release_path=release_path)
            atomic_json(state_path, state)
            # The ordinary push rejects any incompatible remote advance. No
            # automatic force, reset, remote deletion, PR, issue or comment.
            git(checkout, "push", "--no-follow-tags", "origin", "HEAD:refs/heads/main")
            remote = git(checkout, "ls-remote", "origin", "refs/heads/main").stdout.split()
            require(remote and remote[0] == commit, "Remote verification did not observe the exact publication commit")
            state.update(status="published", published=True, completed_at_utc=utc_now(), remote_commit_sha=commit,
                         release_url=f"https://github.com/{REPOSITORY}/tree/{commit}/{release_path}")
            atomic_json(state_path, state)
            print(json.dumps({"status": "published", "commit_sha": commit, "release_url": state["release_url"]}))
            return 0
        except Exception as exc:
            # Only controlled exception messages are persisted; arbitrary JSON,
            # subprocess or filesystem diagnostics may contain private bytes.
            error = str(exc) if isinstance(exc, PublicationBlocked) else type(exc).__name__
            state.update(status="blocked", error=error, updated_at_utc=utc_now())
            atomic_json(state_path, state)
            print(f"Publication blocked: {error}", file=sys.stderr)
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
