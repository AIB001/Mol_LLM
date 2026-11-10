#!/usr/bin/env python3
"""Derive a CSV view from published paired JSONL gzip shards, without API calls."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath


def export_view(release_dir, output, view="paired"):
    release_dir, output = Path(release_dir), Path(output)
    if output.exists(): raise ValueError("Output already exists; refusing to overwrite")
    manifest = json.loads((release_dir/"release_manifest.json").read_text())
    schema = json.loads((release_dir/"metadata/data_schema.json").read_text())
    features = schema["feature_names"]
    entries = sorted((e for e in manifest["files"] if e.get("role") == "paired_training_jsonl"), key=lambda e:e["path"])
    fields = ["source_index","chembl_id","smiles",*features,"property_vector","description","description_sha256","chemistry_verified"]
    if view == "paired": fields += ["first_description","first_description_sha256","text_changed"]
    count=0
    opener = gzip.open if output.suffix == ".gz" else open
    try:
        with opener(output,"wt",encoding="utf-8",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for entry in entries:
                relative=PurePosixPath(entry["path"])
                if relative.is_absolute() or ".." in relative.parts:raise ValueError("Unsafe shard path")
                path=release_dir/relative
                with path.open("rb") as raw:sha=hashlib.file_digest(raw,"sha256").hexdigest()
                if sha != entry["sha256"]:raise ValueError("Shard digest differs from release manifest")
                with gzip.open(path,"rt",encoding="utf-8") as data:
                    for line in data:
                        row=json.loads(line);count+=1
                        if row["source_index"] != count:raise ValueError("Source sequence has a gap")
                        values={**row,**row["properties"]}
                        if view == "first":
                            values["description"],values["description_sha256"]=row["first_description"],row["first_description_sha256"]
                        writer.writerow({k:json.dumps(values[k],ensure_ascii=False,separators=(",",":")) if isinstance(values.get(k),(dict,list)) else values.get(k) for k in fields})
        if count != manifest["target_count"]:raise ValueError("View does not cover complete release")
    except BaseException:
        output.unlink(missing_ok=True);raise
    return count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--view",choices=["paired","first","second"],default="paired")
    args=parser.parse_args()
    print(json.dumps({"rows":export_view(args.release_dir,args.output,args.view),"view":args.view}))


if __name__=="__main__":main()
