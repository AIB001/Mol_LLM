#!/usr/bin/env python3
"""Annotate five molecules per request, with process workers and durable batch/round outputs."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sqlite3
import time

import requests
from rdkit import Chem, RDLogger, rdBase

from annotation_checks import validate_annotation
from download_chembl import ROOT, save_json, utc_now
from molecular_features import FEATURE_NAMES, qualitative_context
from structure_facts import structural_facts

SYSTEM_PATH = ROOT / "prompts/molecular_annotation_system.txt"
KEY_PATH = Path.home() / ".config/mol_llm/deepseek.key"
WORKER = {}


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_keys():
    keys = []
    for env_name, path in [("DEEPSEEK_API_KEY",KEY_PATH),
                          ("DEEPSEEK_API_KEY_SECONDARY",KEY_PATH.with_name("deepseek_secondary.key"))]:
        key = os.environ.get(env_name, "").strip()
        if not key and path.exists():
            key = path.read_text().strip()
        if key and key not in keys:
            keys.append(key)
    if not keys:
        raise RuntimeError("Set DEEPSEEK_API_KEY or store credentials in owner-only key files")
    return keys


def get_balance(key):
    for attempt in range(3):
        try:
            r = requests.get("https://api.deepseek.com/user/balance",
                             headers={"Authorization": f"Bearer {key}"}, timeout=(10, 25))
            if r.status_code in (401, 403):
                raise RuntimeError(f"DeepSeek authorization failed (HTTP {r.status_code})")
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 2:
                return {"balance_unavailable": True}
            time.sleep(2 ** attempt)


def messages_for(items, system):
    payload = {"molecules": [{"chembl_id": r["chembl_id"], "smiles": r["smiles"],
        "verified_structural_facts":r["facts"],
        "qualitative_properties":{k:r["context"][k] for k in ["formal_charge_state","calculated_partition_tendency",
            "calculated_polar_surface","rotatable_connections"]}} for r in items]}
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]


def property_context(f):
    """The same four categories as qualitative_context, without unused graph work."""
    logp = f["crippen_logp"]
    charge = ("net positive formal charge" if f["formal_charge"] > 0 else
              "net negative formal charge" if f["formal_charge"] < 0 else
              "net neutral with separated formal charges" if f["positive_charge_centers"] or f["negative_charge_centers"] else
              "no formal charges in the supplied structure")
    return {"formal_charge_state": charge,
            "calculated_partition_tendency": ("water-favoring partition tendency" if logp < 0 else
                "modest octanol-favoring partition tendency" if logp < 2 else
                "moderate octanol-favoring partition tendency" if logp < 4 else
                "strong octanol-favoring partition tendency" if logp < 6 else
                "very strong octanol-favoring partition tendency"),
            "calculated_polar_surface": ("limited" if f["tpsa"] < 40 else "moderate" if f["tpsa"] < 90 else
                "substantial" if f["tpsa"] < 140 else "extensive"),
            "rotatable_connections": ("few or no freely rotatable bonds" if f["rotatable_bonds"] <= 3 else
                "some rotatable connections" if f["rotatable_bonds"] <= 8 else "many rotatable connections")}


def validate_batch(result, identifiers):
    if not isinstance(result, dict) or set(result) != {"annotations"} or not isinstance(result["annotations"], list):
        return {}, ["invalid_batch_schema"]
    rows = result["annotations"]
    if any(not isinstance(r, dict) for r in rows):
        return {}, ["invalid_annotation_object"]
    ids = [r.get("chembl_id") for r in rows]
    if len(ids) != len(identifiers) or len(set(ids)) != len(ids) or set(ids) != set(identifiers):
        return {}, ["missing_duplicate_or_unexpected_identifier"]
    mapped = {r["chembl_id"]: r for r in rows}
    errors = [f"{cid}:{error}" for cid in identifiers for error in validate_annotation(mapped[cid], cid)]
    return mapped, errors


def initialize_worker(keys, config, semaphores, account_stops, stop_event):
    WORKER.update(keys=keys, config=config, semaphores=semaphores, account_stops=account_stops,
                  stop=stop_event, session=requests.Session())


def api_request(messages):
    c = WORKER["config"]
    payload = {"model": c["model"], "messages": messages, "response_format": {"type": "json_object"},
               "thinking": {"type": c.get("thinking","disabled")},
               "max_tokens": c.get("max_tokens",4096), "stream": False}
    if c.get("thinking")=="enabled":payload["reasoning_effort"]="low"
    else:payload["temperature"]=0.0
    for attempt in range(5):
        if WORKER["stop"].is_set():
            return None, "account_stopped", None
        active = [i for i,e in enumerate(WORKER["account_stops"]) if not e.is_set()]
        if not active:
            WORKER["stop"].set()
            return None, "all_accounts_unavailable", None
        preferred = WORKER["preferred_account"]
        account = preferred if preferred in active else active[0]
        try:
            with WORKER["semaphores"][account]:
                if WORKER["stop"].is_set():
                    return None, "account_stopped", None
                if WORKER["account_stops"][account].is_set():
                    continue
                response = WORKER["session"].post("https://api.deepseek.com/chat/completions", json=payload,
                    headers={"Authorization": "Bearer " + WORKER["keys"][account]}, timeout=(15, 180))
            if response.status_code in {401, 402, 403}:
                WORKER["account_stops"][account].set()
                continue
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    return None, f"http_{response.status_code}_after_retries", account
                time.sleep(min(2 ** attempt, 30) + random.random())
                continue
            if not response.ok:
                # Invalid model/configuration should stop the run, not repeat for thousands of batches.
                WORKER["stop"].set()
                return None, f"configuration_http_{response.status_code}", account
            try:
                return response.json(), None, account
            except ValueError:
                if attempt == 4:
                    return None, "invalid_http_json", account
        except requests.RequestException as exc:
            if attempt == 4:
                return None, f"transport_{type(exc).__name__}_after_retries", account
        time.sleep(min(2 ** attempt, 30) + random.random())
    return None, "request_failed", None


def annotate_batch(task):
    batch_id, round_id, items = task
    config = WORKER["config"]
    WORKER["preferred_account"] = (batch_id - 1) % len(WORKER["keys"])
    messages = messages_for(items, config["system_prompt"])
    usage = Counter()
    account_usage = {}
    mapped, errors, generated = {}, [], None
    finish_reason = model_returned = None
    for attempt in range(3):
        response, request_error, account = api_request(messages)
        if request_error:
            return {"batch_id": batch_id, "round_id": round_id, "status": "request_failed",
                    "error": request_error, "chembl_ids": [r["chembl_id"] for r in items],
                    "usage": dict(usage), "usage_by_account":account_usage,"created_at_utc": utc_now()}
        account_label = f"account_{account+1}"
        account_usage.setdefault(account_label, Counter())
        for name, value in response.get("usage", {}).items():
            if isinstance(value, int):
                usage[name] += value
                account_usage[account_label][name] += value
        model_returned = response.get("model")
        choices = response.get("choices") or [{}]
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        try:
            generated = json.loads(choice.get("message", {}).get("content") or "")
            mapped, errors = validate_batch(generated, [r["chembl_id"] for r in items])
        except (ValueError, TypeError, KeyError):
            errors = ["invalid_or_empty_json"]
            mapped = {}
        if finish_reason != "stop":
            errors.append("incomplete_generation")
            # A repeated request with the same token cap cannot repair a truncated generation.
            if finish_reason=="length":break
        if not errors:
            break
        messages = messages_for(items, config["system_prompt"])
        messages.append({"role": "user", "content": "Correct the following validation failures while retaining the original evidence and constraints: " + "; ".join(errors)})
    output = []
    for item in items:
        cid = item["chembl_id"]
        annotation = mapped.get(cid)
        issues = validate_annotation(annotation, cid)
        if finish_reason != "stop":
            issues.append("incomplete_generation")
        valid = not issues
        output.append({"chembl_id": cid, "smiles": item["smiles"],
            "properties": item["properties"], "property_vector": [item["properties"][n] for n in FEATURE_NAMES],
            "description": annotation["description"] if valid else None,
            "status": annotation["status"] if valid else "validation_failed",
            "issues": annotation["issues"] if valid else issues,
            "batch_id": batch_id, "round_id": round_id, "model": model_returned,"account":account_label,
            "prompt_sha256": config["prompt_sha256"], "chembl_release": item["chembl_release"],
            "rdkit_version": config["rdkit_version"],
            "representative_activity_id": item["representative_activity_id"],
            "validation_scope": "format and numeric-property checks; chemistry not independently verified"})
    return {"batch_id": batch_id, "round_id": round_id, "status": "completed",
            "records": output, "usage": dict(usage), "usage_by_account":account_usage,"attempts": attempt + 1,
            "model_requested": config["model"], "model_returned": model_returned,
            "failed_output": generated if errors else None, "created_at_utc": utc_now()}


def read_items(molecules_path, properties_path):
    from itertools import zip_longest
    with open(molecules_path, newline="", encoding="utf-8") as m, open(properties_path, newline="", encoding="utf-8") as p:
        for row, props in zip_longest(csv.DictReader(m), csv.DictReader(p)):
            if row is None or props is None or row["chembl_id"] != props["chembl_id"]:
                raise RuntimeError("Molecule/property tables do not align")
            features = {n: float(props[n]) for n in FEATURE_NAMES}
            yield {"chembl_id": row["chembl_id"], "smiles": row["smiles"], "properties": features,
                   "chembl_release": row["chembl_release"], "representative_activity_id": row["representative_activity_id"]}


def persist_batch(db, directory, result):
    folder = directory / ("batches" if result["status"] == "completed" else "failures")
    folder.mkdir(exist_ok=True)
    path=folder / f"batch_{result['batch_id']:06d}.json"
    if path.exists() and result["status"]=="completed":
        previous=json.loads(path.read_text())
        if previous!=result:
            history=directory/"batch_history";history.mkdir(exist_ok=True)
            stamp=previous.get("created_at_utc",str(time.time_ns())).replace(":","-")
            save_json(history/f"batch_{result['batch_id']:06d}_{stamp}.json",previous)
            old={r["chembl_id"]:r for r in previous["records"]}
            result["records"]=[old[r["chembl_id"]] if r["chembl_id"] in old and old[r["chembl_id"]].get("description")
                and old[r["chembl_id"]]["status"] in {"ok","needs_review"} else r for r in result["records"]]
            usage=Counter(previous.get("usage",{}));usage.update(result.get("usage",{}));result["usage"]=dict(usage)
    save_json(path, result)
    if result["status"] != "completed":
        return
    db.execute("INSERT OR REPLACE INTO batches VALUES (?,?,?)",
               (result["batch_id"], result["round_id"], json.dumps(result["usage"])))
    for r in result["records"]:
        db.execute("INSERT OR REPLACE INTO annotations VALUES (?,?,?,?,?)",
                   (r["chembl_id"],r["batch_id"],r["round_id"],r["status"],json.dumps(r,ensure_ascii=False,separators=(",", ":"))))
    db.commit()


def export_rows(db, directory, round_id=None):
    name = f"round_{round_id:04d}" if round_id is not None else "annotations"
    sql = "SELECT payload FROM annotations"
    params = ()
    if round_id is not None:
        sql += " WHERE round_id=?"
        params = (round_id,)
    sql += " ORDER BY batch_id,chembl_id"
    target = directory / f"{name}.csv"
    temporary = Path(str(target) + ".tmp")
    fields = ["chembl_id", "smiles"] + FEATURE_NAMES + ["description", "status", "issues", "batch_id", "round_id", "model"]
    with open(temporary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (payload,) in db.execute(sql, params):
            r = json.loads(payload)
            writer.writerow({"chembl_id": r["chembl_id"], "smiles": r["smiles"], **r["properties"],
                "description": r["description"], "status": r["status"], "issues": json.dumps(r["issues"],ensure_ascii=False),
                "batch_id": r["batch_id"], "round_id": r["round_id"], "model": r["model"]})
    temporary.replace(target)
    if round_id is None:
        target = directory / "annotations.jsonl"
        temporary = Path(str(target) + ".tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            for (payload,) in db.execute(sql):
                f.write(payload + "\n")
        temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/chembl_100000_molecules.csv")
    parser.add_argument("--properties", type=Path, default=ROOT / "data/processed/chembl_100000_properties.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/chembl_100000_deepseek_v4_pro")
    parser.add_argument("--processes", type=int, default=2000)
    parser.add_argument("--max-inflight", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--round-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=100, help="Maximum molecules from the start of input; 0 selects all")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--thinking",choices=["disabled","enabled"],default="disabled")
    parser.add_argument("--max-tokens",type=int,default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.processes,args.max_inflight,args.batch_size,args.round_size) < 1 or args.limit < 0:
        parser.error("Invalid size argument")
    if args.batch_size != 5 or args.round_size % args.batch_size:
        parser.error("Use five molecules per batch and a round size divisible by five")
    if args.max_inflight > 500:
        parser.error("The documented V4 Pro account limit is 500; use --max-inflight <= 500")
    RDLogger.DisableLog("rdApp.warning")
    system = SYSTEM_PATH.read_text(encoding="utf-8")
    config = {"model":args.model, "batch_size":args.batch_size,"round_size":args.round_size,
        "prompt_sha256":file_hash(SYSTEM_PATH),"input_sha256":file_hash(args.input),
        "properties_sha256":file_hash(args.properties),"rdkit_version":rdBase.rdkitVersion,
        "context_code_sha256":file_hash(ROOT/"scripts/molecular_features.py"),
        "structure_facts_sha256":file_hash(ROOT/"scripts/structure_facts.py"),
        "validation_code_sha256":file_hash(ROOT/"scripts/annotation_checks.py"),
        "feature_names":FEATURE_NAMES,"system_prompt":system,"thinking":args.thinking,
        "reasoning_effort":"low" if args.thinking=="enabled" else None,
        "max_tokens":args.max_tokens or (32768 if args.thinking=="enabled" else 4096)}
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        parser.error("Prompt, model or input changed. Use a new output directory to avoid mixing annotations.")
    save_json(config_path,config)
    if args.dry_run:
        path=out/"dry_run_requests.jsonl"
        count=0; batch=[]
        with open(path,"w",encoding="utf-8") as f:
            for item in read_items(args.input,args.properties):
                if args.limit and count>=args.limit:break
                item["context"]=property_context(item["properties"])
                item["facts"]=structural_facts(Chem.MolFromSmiles(item["smiles"]),item["properties"])
                batch.append(item);count+=1
                if len(batch)==args.batch_size:
                    f.write(json.dumps({"status":"not_submitted","messages":messages_for(batch,system)},ensure_ascii=False)+"\n");batch=[]
            if batch:f.write(json.dumps({"status":"not_submitted","messages":messages_for(batch,system)},ensure_ascii=False)+"\n")
        print(f"Dry run: {count} molecules; no API calls; {path}")
        return
    keys=read_keys()
    balances=lambda:{f"account_{i+1}":get_balance(k) for i,k in enumerate(keys)}
    initial_balances=balances()
    if not (out/"balance_before.json").exists():save_json(out/"balance_before.json",initial_balances)
    save_json(out/"balance_latest.json",initial_balances)
    db=sqlite3.connect(out/"annotations.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""CREATE TABLE IF NOT EXISTS annotations(chembl_id TEXT PRIMARY KEY,batch_id INTEGER,round_id INTEGER,status TEXT,payload TEXT);
        CREATE TABLE IF NOT EXISTS batches(batch_id INTEGER PRIMARY KEY,round_id INTEGER,usage TEXT);""")
    # Atomic batch files recover the narrow crash window before the SQLite transaction commits.
    existing={r[0] for r in db.execute("SELECT batch_id FROM batches")}
    for path in sorted((out/"batches").glob("batch_*.json")):
        result=json.loads(path.read_text())
        if result["batch_id"] not in existing:persist_batch(db,out,result)
    done={r[0] for r in db.execute("SELECT batch_id FROM batches WHERE batch_id NOT IN (SELECT batch_id FROM annotations WHERE status='validation_failed')")}
    all_items=[]
    for item in read_items(args.input,args.properties):
        if args.limit and len(all_items)>=args.limit:break
        all_items.append(item)
    total=len(all_items)
    # Worker processes do network I/O only. Structure interpretation happens in the parent.
    # Linux fork plus frozen GC keeps imported libraries and the input dataset shared.
    gc.collect();gc.freeze()
    context=mp.get_context("fork")
    semaphores=[context.BoundedSemaphore(args.max_inflight) for _ in keys]
    account_stops=[context.Event() for _ in keys]
    stop=context.Event()
    worker_count=min(args.processes,max(1,(total+args.batch_size-1)//args.batch_size))
    if (out/"runtime.json").exists():
        runtime_history=out/"runtime_history";runtime_history.mkdir(exist_ok=True)
        save_json(runtime_history/f"runtime_{time.time_ns()}.json",json.loads((out/"runtime.json").read_text()))
    save_json(out/"runtime.json",{"requested_processes":args.processes,"created_processes":worker_count,
        "max_inflight_per_account":args.max_inflight,"accounts":len(keys),"selected_molecules":total,"started_at_utc":utc_now()})
    print(f"Processes={worker_count}; accounts={len(keys)}; HTTP concurrency per account<={args.max_inflight}; molecules={total}; batch_size=5",flush=True)
    failures=Counter()
    remaining=Counter((start//args.round_size+1) for start in range(0,total,args.batch_size)
                      if start//args.batch_size+1 not in done)
    round_returns=Counter()
    expected=dict(remaining)
    def task_stream():
        # The pool feeder prepares bounded, small tasks while earlier API calls run.
        # A slow request in one round must not block dispatch of later rounds.
        for start in range(0,total,args.batch_size):
            if stop.is_set():return
            batch_id=start//args.batch_size+1
            if batch_id in done:continue
            items=[]
            for original in all_items[start:min(start+args.batch_size,total)]:
                item=dict(original)
                item["context"]=property_context(item["properties"])
                item["facts"]=structural_facts(Chem.MolFromSmiles(item["smiles"]),item["properties"])
                items.append(item)
            yield batch_id,start//args.round_size+1,items
    try:
        with context.Pool(worker_count,initializer=initialize_worker,initargs=(keys,config,semaphores,account_stops,stop)) as pool:
            for n,result in enumerate(pool.imap_unordered(annotate_batch,task_stream(),chunksize=1),1):
                round_id=result["round_id"]
                persist_batch(db,out,result)
                if result["status"]=="completed":done.add(result["batch_id"])
                else:failures[result["error"]]+=1
                remaining[round_id]-=1;round_returns[round_id]+=1
                if n%20==0 or remaining[round_id]==0:
                    statuses=dict(db.execute("SELECT status,COUNT(*) FROM annotations GROUP BY status"))
                    save_json(out/"progress.json",{"updated_at_utc":utc_now(),"round_id":round_id,
                        "finished_batches":len(done),"status_counts":statuses,"request_failures":failures,
                        "account_stopped":stop.is_set(),"target_molecules":total})
                    print(f"Round {round_id}: {round_returns[round_id]}/{expected[round_id]} batches returned; statuses={statuses}; failures={dict(failures)}",flush=True)
                if remaining[round_id]==0:
                    export_rows(db,out,round_id)
                    export_rows(db,out)
                    save_json(out/"balance_latest.json",balances())
    finally:
        export_rows(db,out)
        counts=dict(db.execute("SELECT status,COUNT(*) FROM annotations GROUP BY status"))
        usage=Counter()
        for (raw,) in db.execute("SELECT usage FROM batches"):usage.update(json.loads(raw))
        summary={"updated_at_utc":utc_now(),"target_molecules":total,"status_counts":counts,
            "stored_molecules":sum(counts.values()),"request_failures":failures,"usage":usage,
            "account_stopped":stop.is_set(),"all_rows_stored":sum(counts.values())==total,
            "all_descriptions_present":sum(counts.values())==total and not counts.get("validation_failed",0),
            "chemical_verification":"Not independently verified; self-reported status and automated format/numeric screens only"}
        save_json(out/"summary.json",summary)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()
        gc.unfreeze()
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
