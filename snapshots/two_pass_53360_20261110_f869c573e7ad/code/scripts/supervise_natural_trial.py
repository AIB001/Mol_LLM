#!/usr/bin/env python3
"""Supervise only the authorized frozen v6 10,000-molecule trial.

This transport supervisor never removes STOP and never passes --resume. It
waits for an existing worker, resumes an explicit all-account HTTP 402 pause
after a fresh CNY balance check, and permits at most two additional full passes
for holes left by normal completion. Accepted rows and all recovery logic remain
the frozen runner's responsibility. Importing this module performs no API calls.

SIGINT/SIGTERM writes this trial's STOP and exits; the worker's existing STOP
watcher drains requests and saves results. No process is forcibly terminated.
Completion means accepted_format, not independently verified chemistry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/chembl_10000_natural_v6"
TARGET = 10000
EXPECTED_FINGERPRINT = "4951335561e1c28dd6ec5d2949413b0ddc9e997d706b7531ccafe3e30320cf91"
SCHEMA_VERSION = "natural_trial_supervisor/1"
MAX_REPAIR_RUNS = 2
CORE_FILES = {
    "runner_code_sha256": "annotate_natural_pilot.py",
    "validation_code_sha256": "natural_description_checks.py",
    "shared_validation_code_sha256": "annotation_checks_v2.py",
    "shared_runner_code_sha256": "annotate_deepseek.py",
    "shared_store_code_sha256": "annotate_deepseek_v2.py",
    "format_code_sha256": "format_controlled_json.py",
    "grounding_wrapper_code_sha256": "annotate_natural_grounded.py",
    "structure_context_code_sha256": "natural_structure_context.py",
    "readable_fact_code_sha256": "readable_facts_v3.py",
}


class Blocked(RuntimeError):
    """A terminal condition requiring inspection, with a non-secret reason."""


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise Blocked("missing_or_invalid_" + Path(path).name) from None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def validate_config(directory=OUTPUT):
    config = read_json(directory / "run_config.json")
    unsigned = {key: value for key, value in config.items() if key != "run_fingerprint"}
    actual = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
    if actual != config.get("run_fingerprint") or actual != EXPECTED_FINGERPRINT:
        raise Blocked("unexpected_frozen_run_fingerprint")
    required = {"model": "deepseek-v4-pro", "thinking": "enabled", "reasoning_effort": "low",
                "max_tokens": 32768, "temperature": 0.6, "dataset_size": TARGET, "batch_size": 5}
    if any(config.get(key) != value for key, value in required.items()):
        raise Blocked("unexpected_frozen_model_configuration")
    for key, filename in CORE_FILES.items():
        if sha256(ROOT / "scripts" / filename) != config.get(key):
            raise Blocked("frozen_dependency_changed_" + filename)
    prompt = Path(config["prompt_path"]).read_text(encoding="utf-8").strip()
    if (prompt != config.get("system_prompt") or
            hashlib.sha256(prompt.encode()).hexdigest() != config.get("prompt_sha256")):
        raise Blocked("frozen_prompt_changed")
    return config


def worker_command(config, caps="128,128", pending=256, python=sys.executable, directory=OUTPUT):
    """Preserve exact config paths; only transport concurrency is configurable."""
    return [python, str(ROOT / "scripts/annotate_natural_grounded.py"),
            "--input", config["input_path"], "--prompt", config["prompt_path"],
            "--output-dir", str(directory), "--limit", str(TARGET),
            "--model", "deepseek-v4-pro", "--thinking", "enabled", "--temperature", "0.6",
            "--max-tokens", "32768", "--request-timeout", "600",
            "--account-caps", caps, "--max-pending-batches", str(pending),
            "--repair-passes", "3", "--http-retries", "6"]


def matches_worker(argv, cwd, directory):
    """Match actual argv tokens, not a shell command containing this filename."""
    if not any(Path(token).name in {"annotate_natural_grounded.py", "annotate_natural_pilot.py"} for token in argv):
        return False
    for index, token in enumerate(argv):
        if token == "--output-dir" and index + 1 < len(argv):
            value = argv[index + 1]
        elif token.startswith("--output-dir="):
            value = token.split("=", 1)[1]
        else:
            continue
        path = Path(value)
        path = path if path.is_absolute() else cwd / path
        try:
            # DrvFs preserves spelling in resolve() although differently cased
            # paths can refer to the same directory. Identity must use inode/dev.
            return path.samefile(directory)
        except OSError:
            return path.resolve() == directory.resolve()
    return False


def find_workers(directory=OUTPUT):
    """Scan /proc as runtime.json can precede exit or lag process startup."""
    workers = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            argv = [part.decode() for part in (process / "cmdline").read_bytes().split(b"\0") if part]
            if argv and matches_worker(argv, (process / "cwd").resolve(), directory):
                workers.append(int(process.name))
        except (OSError, UnicodeError):
            continue
    return sorted(workers)


def _fetch_balances():
    """Only this explicit runtime function accesses credentials/the balance API."""
    from annotate_deepseek import get_balance, read_keys
    try:
        keys = read_keys()
    except Exception:
        raise Blocked("credentials_unavailable") from None
    result = []
    for index, key in enumerate(keys, 1):
        try:
            raw = get_balance(key)
        except Exception as exc:
            message = str(exc)
            if "401" in message or "403" in message:
                raise Blocked("balance_authorization_failed_account_" + str(index)) from None
            raise Blocked("balance_check_failed_account_" + str(index)) from None
        balances = []
        for entry in raw.get("balance_infos", []):
            if entry.get("currency") in {"CNY", "USD"}:
                # Never copy arbitrary API response content or credentials into state.
                try:
                    value = Decimal(str(entry["total_balance"]))
                    if value.is_finite():
                        balances.append({"currency": entry["currency"], "total_balance": str(value)})
                except (InvalidOperation, KeyError):
                    pass
        result.append({"account": "account_" + str(index), "is_available": raw.get("is_available") is True,
                       "balance_unavailable": bool(raw.get("balance_unavailable")), "balances": balances})
    return result


def balance_snapshot(stop_file=OUTPUT / "STOP"):
    """Keep STOP responsive even while a balance HTTP request is waiting."""
    result, error, done = [], [], threading.Event()

    def fetch():
        try:
            result.append(_fetch_balances())
        except Exception as exc:
            error.append(exc)
        finally:
            done.set()

    if stop_file.exists():
        return []
    threading.Thread(target=fetch, daemon=True).start()
    while not done.wait(0.25):
        if stop_file.exists():
            return []
    if error:
        raise error[0]
    return result[0]


def funds_ready(accounts):
    for account in accounts:
        if not account.get("is_available") or account.get("balance_unavailable"):
            continue
        for entry in account.get("balances", []):
            try:
                amount = Decimal(str(entry["total_balance"]))
                if entry.get("currency") == "CNY" and amount.is_finite() and amount >= Decimal("1"):
                    return True
            except (InvalidOperation, KeyError):
                continue
    return False


def terminal_policy(summary):
    """Pure classifier; normal incomplete passes are bounded separately."""
    accepted = summary.get("status_counts", {}).get("accepted_format", 0)
    stored = summary.get("stored_molecules")
    if (summary.get("target_molecules") != TARGET or summary.get("in_flight_batches") != 0 or
            not isinstance(accepted, int) or not 0 <= accepted <= TARGET or
            not isinstance(stored, int) or not accepted <= stored <= TARGET):
        raise Blocked("invalid_or_nonterminal_summary")
    if accepted == TARGET:
        if (summary.get("all_descriptions_present") is not True or
                summary.get("finished_batches") != TARGET // 5 or summary.get("saved_batches") != TARGET // 5):
            raise Blocked("inconsistent_completion_counts")
        return "complete"
    reasons = summary.get("account_disabled_reasons", [])
    if any(reason in {"http_401", "http_403"} for reason in reasons):
        raise Blocked("worker_account_authorization_failed")
    stop_reason = summary.get("stop_reason")
    disabled = summary.get("account_disabled", [])
    if (stop_reason in {"all_accounts_unavailable", "http_402"} and disabled and
            len(disabled) == len(reasons) and all(value is True for value in disabled) and
            all(reason == "http_402" for reason in reasons)):
        return "funding"
    if (stop_reason is None and summary.get("paused") is False and stored == TARGET and
            summary.get("saved_batches") == TARGET // 5):
        return "repair"
    raise Blocked("worker_stopped_requires_inspection")


def completion_commands(path):
    if path is None:
        return [], None
    document = read_json(path)
    value = document.get("commands") if isinstance(document, dict) and set(document) == {"commands"} else None
    if (not isinstance(value, list) or any(not isinstance(command, list) or not command or
            any(not isinstance(arg, str) or not arg or "\0" in arg for arg in command) for command in value)):
        raise Blocked("completion_commands_must_be_commands_object_with_argv_arrays")
    return value, sha256(path)


class Supervisor:
    def __init__(self, directory, command, *, balances=balance_snapshot, workers=find_workers,
                 launch=subprocess.Popen, sleep=time.sleep, clock=time.monotonic,
                 validate=validate_config, hooks=None, hooks_hash=None, run_hook=None,
                 hook_launch=subprocess.Popen):
        self.directory, self.command = Path(directory), command
        self.balances, self.workers, self.launch = balances, workers, launch
        self.sleep, self.clock, self.validate = sleep, clock, validate
        self.hooks, self.hooks_hash, self.run_hook = hooks or [], hooks_hash, run_hook
        self.hook_launch = hook_launch
        self.stop_file = self.directory / "STOP"
        self.policy_dir = self.directory / "supervisor"
        self.state_path = self.policy_dir / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema_version": SCHEMA_VERSION, "created_at_utc": now(), "target_molecules": TARGET,
            "run_fingerprint": EXPECTED_FINGERPRINT, "worker_launches": 0, "repair_runs_used": 0,
            "events": [], "completed_hooks": 0, "awaiting_worker_result": False}
        self.check_state(self.state)
        self.session_metadata = dict(supervisor_pid=os.getpid(), started_at_utc=now(), worker_command=command,
                                     supervisor_code_sha256=sha256(__file__), on_complete_commands_sha256=hooks_hash)
        if (self.directory / "run_config.json").exists():
            self.session_metadata["run_config_file_sha256"] = sha256(self.directory / "run_config.json")
            self.session_metadata["frozen_code_sha256"] = {
                key: value for key, value in read_json(self.directory / "run_config.json").items()
                if key.endswith("code_sha256")}
        self.state.update(self.session_metadata)
        self.child = None
        self.next_balance = 0.0
        self.signal_received = None

    def check_state(self, state):
        if (state.get("schema_version") != SCHEMA_VERSION or state.get("target_molecules") != TARGET or
                state.get("run_fingerprint") != EXPECTED_FINGERPRINT):
            raise Blocked("supervisor_state_identity_mismatch")
        prior_hash = state.get("on_complete_commands_sha256")
        if prior_hash is not None and prior_hash != self.hooks_hash:
            raise Blocked("completion_commands_changed_or_omitted")

    def save(self, status, reason=None, **updates):
        changed = status != self.state.get("status") or reason != self.state.get("reason")
        self.state.update(status=status, reason=reason, updated_at_utc=now(), **updates)
        if changed:
            self.state["events"].append({"at_utc": now(), "status": status, "reason": reason,
                                          "worker_launches": self.state["worker_launches"]})
            print(json.dumps({"status": status, "reason": reason, "at_utc": now()}, indent=2), flush=True)
        atomic_json(self.state_path, self.state)

    def request_stop(self, signum, _frame=None):
        self.signal_received = signal.Signals(signum).name
        # Leave STOP for the frozen worker's own graceful drain; never signal-kill it.
        self.stop_file.touch(exist_ok=True)

    def finish(self):
        if self.state.get("hook_in_progress") is not None:
            self.save("blocked_finalize", "completion_command_interrupted_requires_inspection",
                      final_completed=False)
            return True
        for index in range(self.state.get("completed_hooks", 0), len(self.hooks)):
            if self.stop_file.exists():
                self.save("stopped", "stop_file_before_completion_command")
                return True
            self.save("completing", "running_authorized_completion_command", hook_in_progress=index,
                      final_completed=False)
            log = self.policy_dir / f"completion_{index + 1:02d}.log"
            with log.open("ab") as handle:
                options = dict(cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, shell=False)
                if self.run_hook is not None:  # Injectable synchronous fake used by offline tests.
                    returncode = self.run_hook(self.hooks[index], check=False, **options).returncode
                else:
                    process = self.hook_launch(self.hooks[index], **options)
                    self.save("completing", "running_authorized_completion_command", hook_pid=process.pid)
                    while process.poll() is None:
                        if self.stop_file.exists():
                            self.save("stopped", "stop_file_during_completion_command", final_completed=False)
                            return True
                        self.sleep(0.25)
                    returncode = process.returncode
            if returncode != 0:
                self.save("blocked_finalize", "completion_command_failed_" + str(index + 1),
                          final_completed=False)
                return True
            self.state.update(completed_hooks=index + 1, hook_in_progress=None)
            atomic_json(self.state_path, self.state)
        if self.stop_file.exists():
            self.save("stopped", "stop_file_before_completion", final_completed=False)
            return True
        self.save("delivery_complete" if self.hooks else "generation_complete", "10000_accepted_format",
                  worker_pid=None, final_completed=True)
        return True

    def step(self):
        """One non-sleeping scheduling step, injectable for offline tests."""
        if self.stop_file.exists() or self.signal_received:
            self.save("stopped", self.signal_received or "stop_file", worker_pid=None)
            return True
        child_code = self.child.poll() if self.child is not None else None
        pids = self.workers(self.directory)
        if self.child is not None and child_code is None and self.child.pid not in pids:
            pids.append(self.child.pid)
        if len(pids) > 1:
            raise Blocked("multiple_workers_detected_no_launch")
        if pids:
            if self.state.get("worker_pid") != pids[0] or self.state.get("status") != "worker_running":
                if self.child is None and not self.state.get("awaiting_worker_result"):
                    self.state["observed_external_pid"] = pids[0]
                    self.state["external_summary_before_sha256"] = (
                        sha256(self.directory / "summary.json") if (self.directory / "summary.json").exists() else None)
                self.save("worker_running", "waiting_for_existing_worker", worker_pid=pids[0])
            return False
        if self.child is not None:
            self.child = None
            self.state["last_worker_exit_code"] = child_code
            if child_code != 0:
                raise Blocked("worker_nonzero_exit_requires_inspection")
        summary_path = self.directory / "summary.json"
        summary = read_json(summary_path)
        summary_hash = sha256(summary_path)
        if self.state.get("observed_external_pid") is not None:
            if summary_hash == self.state.get("external_summary_before_sha256"):
                # A worker may be observed during its final exit. Accept that
                # unchanged summary only when runtime timestamps prove it is new.
                runtime_path = self.directory / "runtime.json"
                runtime = read_json(runtime_path) if runtime_path.exists() else {}
                try:
                    fresh = (runtime.get("pid") == self.state["observed_external_pid"] and
                             datetime.fromisoformat(summary["updated_at_utc"]) >=
                             datetime.fromisoformat(runtime["started_at_utc"]))
                except (KeyError, TypeError, ValueError):
                    fresh = False
                if not fresh:
                    raise Blocked("external_worker_exited_without_fresh_summary")
            self.state["observed_external_pid"] = None
        if self.state.get("awaiting_worker_result"):
            if summary_hash == self.state.get("summary_before_launch_sha256"):
                raise Blocked("worker_exited_without_fresh_summary")
            self.state["awaiting_worker_result"] = False
        policy = terminal_policy(summary)
        self.state.update(accepted_molecules=summary["status_counts"].get("accepted_format", 0),
                          stored_molecules=summary["stored_molecules"], worker_pid=None,
                          last_summary_sha256=summary_hash, last_worker_stop_reason=summary.get("stop_reason"))
        if policy == "complete":
            return self.finish()
        if policy == "repair" and self.state["repair_runs_used"] >= MAX_REPAIR_RUNS:
            raise Blocked("two_repair_passes_exhausted_requires_inspection")
        if self.clock() < self.next_balance:
            return False
        accounts = self.balances()
        self.next_balance = self.clock() + 60
        if self.stop_file.exists():
            self.save("stopped", self.signal_received or "stop_file")
            return True
        self.save("waiting_for_funds", "fresh_balance_below_threshold_or_unavailable",
                  last_balances=accounts, last_balance_check_utc=now())
        if not funds_ready(accounts):
            return False
        self.validate(self.directory)
        # Recheck immediately before launching: STOP wins and external workers win.
        if self.stop_file.exists():
            self.save("stopped", "stop_file")
            return True
        if self.workers(self.directory):
            self.save("worker_running", "worker_appeared_before_launch")
            return False
        if policy == "repair":
            self.state["repair_runs_used"] += 1
        self.state["worker_launches"] += 1
        launch_id = self.state["worker_launches"]
        log = self.policy_dir / f"worker_{launch_id:04d}.log"
        self.save("launching", "funding_continuation" if policy == "funding" else "bounded_hole_repair",
                  awaiting_worker_result=True, summary_before_launch_sha256=summary_hash,
                  last_worker_log=str(log))
        with log.open("ab") as handle:
            self.child = self.launch(self.command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self.save("worker_running", "launched_frozen_v6_worker", worker_pid=self.child.pid)
        return False

    def run(self):
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        with (self.policy_dir / "supervisor.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another supervisor holds this trial's lock", file=sys.stderr)
                return 2
            try:
                # Another supervisor may have updated counters while this object
                # was being constructed. Only the state read under flock wins.
                if self.state_path.exists():
                    latest = read_json(self.state_path)
                    try:
                        self.check_state(latest)
                    except Blocked as exc:
                        # Preserve the newer supervisor's state and requirements.
                        atomic_json(self.policy_dir / f"startup_blocked_{os.getpid()}.json",
                                    {"status": "blocked", "reason": str(exc), "at_utc": now()})
                        return 2
                    self.state = {**latest, **self.session_metadata}
                self.save("starting", "checking_existing_worker_and_terminal_summary")
                while not self.step():
                    self.sleep(1)
                return 2 if self.state["status"] == "blocked_finalize" else 0
            except Blocked as exc:
                self.save("blocked", str(exc))
                return 2
            except Exception as exc:
                # Do not log exception strings: third-party exceptions may contain credentials.
                self.save("blocked", "supervisor_exception_" + type(exc).__name__)
                return 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--account-caps", default="128,128")
    parser.add_argument("--max-pending-batches", type=int, default=256)
    parser.add_argument("--on-complete-commands-file", type=Path,
                        help='Authorized JSON {"commands": [["python", "/abs/script.py", ...], ...]}; no shell')
    args = parser.parse_args(argv)
    try:
        caps = [int(value) for value in args.account_caps.split(",")]
        if len(caps) != 2 or any(not 1 <= value <= 500 for value in caps) or args.max_pending_batches < 1:
            raise ValueError
    except ValueError:
        parser.error("Use two account caps in 1..500 and a positive pending-batch limit")
    args.account_caps = ",".join(map(str, caps))
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        config = validate_config()
        hooks, hooks_hash = completion_commands(args.on_complete_commands_file)
        supervisor = Supervisor(OUTPUT, worker_command(config, args.account_caps, args.max_pending_batches),
                                hooks=hooks, hooks_hash=hooks_hash)
    except Blocked as exc:
        print(json.dumps({"status": "blocked", "reason": str(exc)}, indent=2), file=sys.stderr)
        return 2
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, supervisor.request_stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
