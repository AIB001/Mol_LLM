#!/usr/bin/env python3
"""Create readable JSON arrays from JSONL without changing the JSONL sources.

Default conversion is safe while the generator writes other atomic exports: an
opened source file remains the chosen snapshot. Batch/metadata formatting is
explicitly opt-in and refused while the runtime's writer process is still alive.
Metadata formatting touches only single-line JSON; existing multiline layouts
are retained byte-for-byte. run_config.json is preserved and receives a separate
run_config.pretty.json copy. No keys, network, generator imports, or chemistry
changes are involved.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUFFER_BYTES = 8 * 1024 * 1024


def source_signature(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def pretty_array(source, destination=None):
    """Stream exact JSON values into an indented standard JSON array atomically."""
    source = Path(source)
    destination = Path(destination) if destination is not None else source.with_suffix(".pretty.json")
    if source.resolve() == destination.resolve():
        raise ValueError("Source JSONL and destination JSON must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".format.tmp")
    count, statuses, hasher = 0, Counter(), hashlib.sha256()
    try:
        with source.open("rb", buffering=BUFFER_BYTES) as original, \
             temporary.open("w", encoding="utf-8", newline="\n", buffering=BUFFER_BYTES) as output:
            opened_stat = os.fstat(original.fileno())
            output.write("[\n")
            for line_number, raw_line in enumerate(original, 1):
                hasher.update(raw_line)
                if not raw_line.strip():
                    raise ValueError(f"{source}: blank JSONL record at line {line_number}")
                value = json.loads(raw_line)
                if not isinstance(value, dict):
                    raise ValueError(f"{source}: record {line_number} must be a JSON object")
                if count:
                    output.write(",\n")
                output.write("  " + json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).replace("\n", "\n  "))
                count += 1
                statuses[str(value.get("status", "missing"))] += 1
            output.write("\n]\n")
            closed_stat = os.fstat(original.fileno())
            if source_signature(opened_stat) != source_signature(closed_stat):
                raise RuntimeError("The opened JSONL file changed in place; readable output was not published")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {"source": str(source.resolve()), "output": str(destination.resolve()), "records": count,
            "status_counts": dict(statuses), "source_sha256": hasher.hexdigest(),
            "source_bytes": opened_stat.st_size, "output_bytes": destination.stat().st_size,
            "source_replaced_during_conversion": source_signature(opened_stat) != source_signature(source.stat())}


def ensure_writer_stopped(directory):
    """Protect live journals/metadata; preview-array conversion needs no stop."""
    runtime = Path(directory) / "runtime.json"
    if not runtime.exists():
        return
    state = json.loads(runtime.read_text(encoding="utf-8"))
    pid = state.get("pid")
    if type(pid) is not int or pid <= 0:
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    except PermissionError:
        raise RuntimeError(f"Cannot establish that writer PID {pid} has stopped; batch/metadata formatting refused") from None
    raise RuntimeError(f"Writer PID {pid} is still alive; batch/metadata formatting must wait until it exits")


def reformat_json(path, destination=None):
    """Preserve JSON values/order; optionally create a copy without changing source."""
    path = Path(path)
    destination = Path(destination) if destination is not None else path
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".format.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n", buffering=BUFFER_BYTES) as handle:
            handle.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def format_directory(directory, source=None, destination=None, rounds=False, batches=False, metadata=False):
    directory = Path(directory)
    if batches or metadata:
        ensure_writer_stopped(directory)
    source = Path(source) if source is not None else directory / "annotations.jsonl"
    results = [pretty_array(source, destination)]
    if rounds:
        for path in sorted(directory.glob("round_*.jsonl")):
            if path.resolve() != source.resolve():
                results.append(pretty_array(path))
    reformatted = {"batches": 0, "metadata": 0, "metadata_pretty_copies": 0, "metadata_skipped_multiline": 0}
    if batches:
        for path in sorted((directory / "batches").glob("*.json")):
            if not path.name.endswith(".pretty.json"):
                reformat_json(path)
                reformatted["batches"] += 1
    if metadata:
        for path in sorted(directory.rglob("*.json")):
            if path.name.endswith(".pretty.json") or "batches" in path.relative_to(directory).parts:
                continue
            if path.name == "run_config.json":
                # Policy histories refer to the original configuration bytes.
                reformat_json(path, path.with_name("run_config.pretty.json"))
                reformatted["metadata_pretty_copies"] += 1
                continue
            with path.open("r", encoding="utf-8") as handle:
                nonempty_lines = sum(bool(line.strip()) for line in handle)
            if nonempty_lines > 1:
                # Keep existing layouts and copied supporting-file hashes intact.
                reformatted["metadata_skipped_multiline"] += 1
                continue
            reformat_json(path)
            reformatted["metadata"] += 1
    return {"converted_arrays": results, "reformatted_files": reformatted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/chembl_100000_controlled")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format-rounds", action="store_true")
    parser.add_argument("--format-batches", action="store_true")
    parser.add_argument("--format-metadata", action="store_true")
    args = parser.parse_args(argv)
    report = format_directory(args.output_dir, args.input, args.output, args.format_rounds,
                              args.format_batches, args.format_metadata)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
