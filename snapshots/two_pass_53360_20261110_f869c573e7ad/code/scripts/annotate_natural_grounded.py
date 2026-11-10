#!/usr/bin/env python3
"""Add verified structural evidence to the frozen free-prose annotation runner.

Only the input context, system prompt, output directory, and their provenance
change. Descriptions remain model-written natural language. The original batch
storage, partial repair, response history, stop handling and exports are reused.
No API call occurs at import or during --dry-run.
"""
from __future__ import annotations

from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import sys

import annotate_natural_pilot as runner

SYSTEM_PATH = runner.ROOT / "prompts/natural_description_system_v5.txt"
DEFAULT_OUTPUT = runner.ROOT / "results/chembl_10000_natural_v5"
GENERATION_MODE = "deepseek_free_natural_language_from_smiles_and_verified_context"
WRAPPER_SCHEMA_VERSION = "grounded_natural_annotation_wrapper/1"
ORIGINAL_MESSAGES = runner.messages_for
ORIGINAL_CONFIG = runner.make_config


def context_module():
    return importlib.import_module("natural_structure_context")


def messages_for(items, config, errors=None):
    messages = ORIGINAL_MESSAGES(items, config, errors)
    payload = json.loads(messages[1]["content"])
    builder = context_module().natural_structure_context
    for row, item in zip(payload["molecules"], items):
        row["structural_context"] = builder(item["structural_facts"])
        if runner.digest(item["structural_facts"]) != item["facts_sha256"]:
            raise RuntimeError("Structural context construction changed source facts or their hash")
    messages[1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return messages


def make_config(args):
    config = ORIGINAL_CONFIG(args)
    context = context_module()
    config.pop("run_fingerprint", None)
    config.update(generation_mode=GENERATION_MODE, grounding_wrapper_schema=WRAPPER_SCHEMA_VERSION,
                  grounding_wrapper_code_sha256=runner.file_hash(Path(__file__)),
                  structure_context_schema_version=context.SCHEMA_VERSION,
                  structure_context_code_sha256=runner.file_hash(Path(context.__file__)),
                  readable_fact_code_sha256=runner.file_hash(runner.ROOT / "scripts/readable_facts_v3.py"),
                  model_input_fields=["key", "chembl_id", "smiles", "structural_context"],
                  output_text_origin="model-generated free prose; structural context is input evidence, not output composition")
    config["run_fingerprint"] = runner.digest(config)
    return config


@contextmanager
def patched_runner():
    """Use reversible in-memory overrides; never modify frozen source files."""
    overrides = {"messages_for": messages_for, "make_config": make_config,
                 "SYSTEM_PATH": SYSTEM_PATH, "DEFAULT_OUTPUT": DEFAULT_OUTPUT, "GENERATION_MODE": GENERATION_MODE}
    original = {key: getattr(runner, key) for key in overrides}
    try:
        for key, value in overrides.items():
            setattr(runner, key, value)
        yield runner
    finally:
        for key, value in original.items():
            setattr(runner, key, value)


def forwarded_arguments(argv):
    forwarded = list(argv)
    defaults = {"--thinking": "disabled", "--max-tokens": "2500", "--account-caps": "3,3",
                "--max-pending-batches": "6", "--limit": "25"}
    supplied = {argument.split("=", 1)[0] for argument in forwarded}
    for flag, value in defaults.items():
        if flag not in supplied:
            forwarded.extend([flag, value])
    return forwarded


def main(argv=None):
    forwarded = forwarded_arguments(sys.argv[1:] if argv is None else argv)
    with patched_runner():
        args = runner.parse_args(forwarded)
        old_output = runner.ROOT / "results/chembl_10000_natural_pilot"
        if args.output_dir.resolve() == old_output.resolve():
            raise RuntimeError("The SMILES-only pilot is preserved; use a separate grounded output directory")
        return runner.main(forwarded)


if __name__ == "__main__":
    main()
