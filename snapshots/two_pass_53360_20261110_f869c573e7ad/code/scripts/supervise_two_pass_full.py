#!/usr/bin/env python3
"""Supervise the authorized full two-pass run; never remove STOP or push by itself.

Reuses the existing tested supervisor policy in an isolated module namespace.
Explicit funding stops wait for a fresh balance check every 60 seconds; ordinary
missing records receive at most two additional invocations. Completion commands
are optional argv arrays supplied by the operator and pinned across restarts.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import signal
import sys

import run_two_pass_full as full

ROOT = full.ROOT
OUTPUT = full.DEFAULT_OUTPUT
POLICY_PATH = ROOT / "scripts/supervise_natural_trial.py"
SCHEMA = "two_pass_full_supervisor/1"


def matches_worker(argv, cwd, directory):
    if not any(Path(token).name in {"run_two_pass_full.py", "run_two_pass_full"} for token in argv):
        return False
    for index, token in enumerate(argv):
        if token == "--output-dir" and index+1 < len(argv):
            value=argv[index+1]
        elif token.startswith("--output-dir="):
            value=token.split("=",1)[1]
        else:continue
        path=Path(value)
        return full.same_path(path if path.is_absolute() else cwd/path,directory)
    return full.same_path(directory,OUTPUT)


def worker_command(config,directory,caps="400,400",pending=800,python=sys.executable):
    return [python,str(ROOT/"scripts/run_two_pass_full.py"),"--input",config["input_path"],
        "--molecules",config["molecules_path"],"--properties",config["properties_path"],
        "--first-prompt",config["first_prompt_path"],"--review-prompt",config["review_prompt_path"],
        "--seed-dir",config["seed_directory"],"--output-dir",str(directory),"--limit",str(full.TARGET),
        "--account-caps",caps,"--max-pending-batches",str(pending),"--http-retries","6","--request-timeout","600",
        "--repair-passes",str(config["first_validation_passes_per_invocation"]),
        "--review-passes",str(config["review_validation_passes_per_invocation"])]


def load_policy(config,directory):
    spec=importlib.util.spec_from_file_location("_two_pass_full_supervisor_policy",POLICY_PATH)
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    policy.TARGET,policy.OUTPUT=full.TARGET,Path(directory)
    policy.EXPECTED_FINGERPRINT=config["run_fingerprint"]
    policy.SCHEMA_VERSION=SCHEMA
    policy.matches_worker=matches_worker
    def terminal(summary):
        accepted=summary.get("status_counts",{}).get("accepted_format",0)
        stored=summary.get("stored_molecules")
        if (summary.get("run_fingerprint")!=config["run_fingerprint"] or summary.get("target_molecules")!=full.TARGET
                or summary.get("in_flight_batches")!=0 or type(accepted)is not int or type(stored)is not int
                or not 0<=accepted<=stored<=full.TARGET):
            raise policy.Blocked("invalid_two_pass_terminal_summary")
        if accepted==full.TARGET:
            if summary.get("paused") is not False or summary.get("stop_reason") is not None:
                raise policy.Blocked("completed_but_explicitly_stopped_no_completion_hooks")
            if (summary.get("complete")is not True or summary.get("all_descriptions_present")is not True
                    or summary.get("first_pass_accepted")!=full.TARGET or summary.get("second_pass_accepted")!=full.TARGET
                    or summary.get("finished_batches")!=full.TARGET//5 or summary.get("saved_batches")!=full.TARGET//5):
                raise policy.Blocked("inconsistent_two_pass_completion")
            return "complete"
        reasons=summary.get("account_disabled_reasons",[])
        if any(reason in {"http_401","http_403"} for reason in reasons):
            raise policy.Blocked("worker_account_authorization_failed")
        if summary.get("stop_reason")=="seed_prepared":
            if (accepted!=full.SEED_COUNT or stored!=full.SEED_COUNT or (Path(directory)/"runtime.json").exists()
                    or summary.get("finished_batches")!=full.SEED_COUNT//5):
                raise policy.Blocked("invalid_prepared_seed_bootstrap")
            return "funding"
        disabled=summary.get("account_disabled",[])
        if (summary.get("stop_reason") in {"all_accounts_unavailable","http_402"} and disabled
                and len(disabled)==len(reasons) and all(value is True for value in disabled)
                and all(reason in {"http_402","balance_explicitly_unavailable"} for reason in reasons)):
            return "funding"
        if (summary.get("stop_reason")is None and summary.get("paused")is False and stored==full.TARGET
                and summary.get("saved_batches")==full.TARGET//5):
            return "repair"
        raise policy.Blocked("two_pass_worker_stopped_requires_inspection")
    policy.terminal_policy=terminal
    return policy


def create_supervisor(config,directory,*,caps="400,400",pending=800,hooks=None,hooks_hash=None,
                      validate=full.validate_existing_configuration,**dependencies):
    if config.get("schema_version")!=full.SCHEMA or config.get("dataset_size")!=full.TARGET:
        raise ValueError("Unexpected full two-pass configuration")
    policy=load_policy(config,directory)
    def validate_pinned(path):
        current=validate(path)
        if current!=config:raise policy.Blocked("full_configuration_changed_after_start")
        return current
    class TwoPassSupervisor(policy.Supervisor):
        def save(self,status,reason=None,**updates):
            if reason=="10000_accepted_format":reason="100000_paired_descriptions_saved"
            if reason=="launched_frozen_v6_worker":reason="launched_full_two_pass_worker"
            return super().save(status,reason,**updates)
    defaults={"workers":policy.find_workers,"balances":lambda:policy.balance_snapshot(Path(directory)/"STOP")}
    defaults.update(dependencies)
    supervisor=TwoPassSupervisor(Path(directory),worker_command(config,directory,caps,pending),validate=validate_pinned,
                                 hooks=hooks,hooks_hash=hooks_hash,**defaults)
    supervisor.session_metadata.update(two_pass_supervisor_code_sha256=policy.sha256(Path(__file__)),
                                        base_supervisor_code_sha256=policy.sha256(POLICY_PATH))
    supervisor.state.update(supervisor.session_metadata)
    return supervisor,policy


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=OUTPUT)
    parser.add_argument("--account-caps",default="400,400")
    parser.add_argument("--max-pending-batches",type=int,default=800)
    parser.add_argument("--on-complete-commands-file",type=Path)
    args=parser.parse_args(argv)
    caps=full.core.account_caps(args.account_caps)
    if len(caps)!=2 or args.max_pending_batches<1:parser.error("Two account caps and a positive task bound required")
    config=full.validate_existing_configuration(args.output_dir)
    policy=load_policy(config,args.output_dir)
    hooks,checksum=policy.completion_commands(args.on_complete_commands_file)
    supervisor,_=create_supervisor(config,args.output_dir,caps=",".join(map(str,caps)),pending=args.max_pending_batches,
                                  hooks=hooks,hooks_hash=checksum)
    for signum in (signal.SIGINT,signal.SIGTERM):signal.signal(signum,supervisor.request_stop)
    return supervisor.run()


if __name__=="__main__":raise SystemExit(main())
