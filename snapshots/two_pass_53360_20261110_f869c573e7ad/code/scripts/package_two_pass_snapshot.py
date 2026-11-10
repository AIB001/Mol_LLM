#!/usr/bin/env python3
"""Package an explicitly incomplete saved two-pass snapshot without changing full gates.

Complete pairs and accepted first-pass texts with no accepted second-pass result
are separate datasets. Only explicit training projections and allowlisted frozen
workflow assets are included; raw responses, reasoning and credentials stay out.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

import package_two_pass_release as release

SCHEMA = "two_pass_snapshot_manifest/1"
MANIFEST_NAME = "snapshot_manifest.json"
TARGET = 100000
EXPECTED_PAIRS, EXPECTED_FIRST_ONLY = 53360, 2190
FIRST_ONLY_SCHEMA = "two_pass_snapshot_first_only_record/1"
SOURCE_FIELDS = ("source_index", "chembl_id", "smiles", "facts_sha256", "properties", "property_vector")
FIRST_ONLY_FIELDS = set(SOURCE_FIELDS) | {"schema_version", "pair_status", "description_origin", "review_completed",
    "first_description", "description", "first_description_sha256", "description_sha256", "second_description",
    "second_description_sha256", "first_pass", "second_pass", "chemistry_verified"}
REQUIRED_ASSET_ROLES = {"source_license", "source_attribution", "first_prompt", "review_prompt", "generator_code", "source_documentation"}


def check_gate(audit, fingerprint, expected_pairs, expected_first_only):
    if type(expected_pairs) is not int or not 0 < expected_pairs < TARGET or type(expected_first_only) is not int or not 0 <= expected_first_only <= TARGET-expected_pairs:
        raise ValueError("Expected snapshot counts must describe a strictly incomplete population")
    expected = {"schema_version": "two_pass_full_integrity_audit/1", "valid": True, "valid_source_and_provenance": True,
        "publish_ready": False, "error_count": 0, "target_count": TARGET, "complete_pairs": expected_pairs,
        "first_pass_accepted": expected_pairs+expected_first_only, "second_pass_accepted": expected_pairs,
        "missing_pairs": TARGET-expected_pairs, "run_fingerprint": fingerprint, "chemistry_verified": False}
    for key, value in expected.items():
        if type(audit.get(key)) is not type(value) or audit[key] != value:
            raise ValueError("Snapshot audit gate failed: " + key)


def accepted(row):
    return isinstance(row, dict) and row.get("status") == "accepted_format" and isinstance(row.get("description"), str) and bool(row["description"].strip())


def check_source_row(row):
    for phase in ("first_pass", "second_pass"):
        stage = row.get(phase)
        if not accepted(stage):
            continue
        for key in SOURCE_FIELDS:
            if stage.get(key) != row.get(key):
                raise ValueError("Accepted stage identity/properties differ from paired source: " + key)
        if hashlib.sha256(stage["description"].encode()).hexdigest() != stage.get("description_sha256"):
            raise ValueError("Accepted source description hash mismatch")
        if stage.get("chemistry_verified") is not False:
            raise ValueError("Unsupported chemical verification claim")


def first_only_row(row):
    first, second = row.get("first_pass"), row.get("second_pass")
    if row.get("pair_status") != "awaiting_second_pass" or not accepted(first) or accepted(second):
        raise ValueError("First-only projection requires no accepted second-pass result")
    check_source_row(row)
    text, checksum = first["description"], first["description_sha256"]
    return {**{key: row[key] for key in SOURCE_FIELDS}, "schema_version": FIRST_ONLY_SCHEMA,
        "pair_status": "awaiting_second_pass", "description_origin": "first_pass_unreviewed", "review_completed": False,
        "first_description": text, "description": text, "first_description_sha256": checksum, "description_sha256": checksum,
        "second_description": None, "second_description_sha256": None, "first_pass": release.phase_metadata(first),
        "second_pass": None, "chemistry_verified": False}


def validate_row(row, role):
    fields = release.TRAINING_FIELDS if role == "paired_training_jsonl" else FIRST_ONLY_FIELDS
    schema = "two_pass_training_record/1" if role == "paired_training_jsonl" else FIRST_ONLY_SCHEMA
    if set(row) != fields or row.get("schema_version") != schema:
        raise ValueError("Unexpected/private snapshot record fields")
    if type(row.get("source_index")) is not int or not 1 <= row["source_index"] <= TARGET or not isinstance(row.get("chembl_id"), str) or not row["chembl_id"]:
        raise ValueError("Invalid original source identity")
    props, vector = row.get("properties"), row.get("property_vector")
    if not isinstance(props, dict) or set(props) != set(release.FEATURE_NAMES) or any(type(v) not in (int,float) or not math.isfinite(v) for v in props.values()):
        raise ValueError("Snapshot properties must be the 24 finite source values")
    if not isinstance(vector,list) or len(vector)!=24 or any(type(v) not in (int,float) or not math.isfinite(v) for v in vector) or vector != [props[k] for k in release.FEATURE_NAMES]:
        raise ValueError("Snapshot property vector differs from the source values/order")
    for text_key, hash_key in [("first_description","first_description_sha256"),("description","description_sha256")]:
        text=row.get(text_key)
        if not isinstance(text,str) or not text.strip() or hashlib.sha256(text.encode()).hexdigest()!=row.get(hash_key):
            raise ValueError("Snapshot description/hash mismatch")
    phases=("first_pass","second_pass") if role=="paired_training_jsonl" else ("first_pass",)
    for phase in phases:
        value=row.get(phase)
        if not isinstance(value,dict) or not set(value)<=release.META_FIELDS or value.get("status")!="accepted_format":
            raise ValueError("Snapshot phase metadata is invalid or includes private fields")
        origin=value.get("origin")
        if origin is not None and (not isinstance(origin,dict) or set(origin)!={"kind","journal_path","batch_id","record_index","original_record_sha256","run_fingerprint"}):
            raise ValueError("Unexpected origin metadata")
    if row.get("chemistry_verified") is not False:
        raise ValueError("Snapshot must not claim chemical verification")
    if role=="paired_training_jsonl":
        if type(row.get("text_changed")) is not bool or row["text_changed"]!=(row["first_description"]!=row["description"]):
            raise ValueError("Exact paired change flag differs")
    elif (row.get("pair_status")!="awaiting_second_pass" or row.get("description_origin")!="first_pass_unreviewed"
          or row.get("review_completed")is not False or row.get("second_description")is not None
          or row.get("second_description_sha256")is not None or row.get("second_pass")is not None
          or row["description"]!=row["first_description"] or row["description_sha256"]!=row["first_description_sha256"]):
        raise ValueError("First-only text is mislabeled as reviewed or differs from its original first pass")


def verify_snapshot_payload(directory, manifest):
    """Stream both real datasets; gaps in original source indices are intentional."""
    directory=Path(directory);all_ids=set();all_indices=set();preview=[];counts={}
    for role, expected in [("paired_training_jsonl",manifest["exported_pair_count"]),("first_pass_only_jsonl",manifest["exported_first_pass_only_count"])]:
        entries=sorted((entry for entry in manifest["files"] if entry.get("role")==role),key=lambda entry:entry["path"])
        total,last=0,0
        for entry in entries:
            local=0
            with gzip.open(directory/entry["path"],"rt",encoding="utf-8") as handle:
                for line in handle:
                    row=json.loads(line);validate_row(row,role)
                    index,cid=row["source_index"],row["chembl_id"]
                    if index<=last or index in all_indices or cid in all_ids:
                        raise ValueError("Snapshot source order/uniqueness or paired/first-only disjointness mismatch")
                    last=index;all_indices.add(index);all_ids.add(cid);local+=1;total+=1
                    if role=="paired_training_jsonl" and len(preview)<100:preview.append(row)
            if local!=entry.get("rows"):raise ValueError("Actual snapshot shard row count differs from manifest")
        if total!=expected:raise ValueError("Actual snapshot data row totals differ from manifest")
        counts[role]=total
    if len(all_ids)!=manifest["first_pass_accepted"]:raise ValueError("Snapshot does not cover all accepted first-pass identities")
    entries={entry["role"]:entry for entry in manifest["files"] if entry["role"] in {"preview_json","preview_csv"}}
    if set(entries)!={"preview_json","preview_csv"}:raise ValueError("Missing paired preview")
    displayed=release.read_json(directory/entries["preview_json"]["path"])
    if displayed.get("sample_only")is not True or displayed.get("records")!=preview or displayed.get("sample_count")!=len(preview):
        raise ValueError("Readable preview differs from the first paired records")
    with (directory/entries["preview_csv"]["path"]).open(newline="",encoding="utf-8") as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=release.CSV_FIELDS:raise ValueError("Preview CSV schema mismatch")
        for flat,row in itertools.zip_longest(reader,preview):
            if flat is None or row is None:raise ValueError("Preview CSV row count differs")
            values={**row,**row["properties"]}
            if any(flat[key]!=release.cell(values.get(key)) for key in release.CSV_FIELDS):raise ValueError("Preview CSV differs from paired JSONL")
    return counts


def verify_snapshot(directory, expected_fingerprint=None, expected_pairs=EXPECTED_PAIRS, expected_first_only=EXPECTED_FIRST_ONLY):
    directory=Path(directory)
    if directory.is_symlink():raise ValueError("Snapshot directory cannot be a symlink")
    path=directory/MANIFEST_NAME
    if path.is_symlink() or not path.is_file() or path.stat().st_size>=release.MAX_FILE_BYTES:raise ValueError("Invalid snapshot manifest file")
    manifest=release.read_json(path)
    if manifest.get("schema_version")!=SCHEMA or expected_fingerprint and manifest.get("run_fingerprint")!=expected_fingerprint:
        raise ValueError("Snapshot manifest schema/fingerprint mismatch")
    counts={"target_count":TARGET,"exported_pair_count":expected_pairs,"exported_first_pass_only_count":expected_first_only,
            "first_pass_accepted":expected_pairs+expected_first_only,"second_pass_accepted":expected_pairs,"remaining_pairs":TARGET-expected_pairs}
    if any(type(manifest.get(k))is not int or manifest[k]!=v for k,v in counts.items()) or manifest.get("snapshot_complete")is not False or manifest.get("chemistry_verified")is not False or manifest.get("publish_ready")is not False:
        raise ValueError("Snapshot count/completion scope mismatch")
    entries=manifest.get("files",[]);names=[release.safe_relative(e["path"]) for e in entries]
    if len(names)!=len({name.casefold() for name in names}) or MANIFEST_NAME.casefold() in {name.casefold() for name in names}:raise ValueError("Duplicate/colliding snapshot paths")
    actual={p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=set(names)|{MANIFEST_NAME}:raise ValueError("Snapshot contains missing or unlisted files")
    for entry in entries:
        asset=directory/entry["path"]
        if asset.is_symlink() or any(parent.is_symlink() for parent in asset.parents if parent!=directory.parent):raise ValueError("Snapshot path traverses symlink")
        if asset.stat().st_size!=entry["size_bytes"] or release.file_hash(asset)!=entry["sha256"]:raise ValueError("Snapshot file hash/size mismatch")
        if asset.stat().st_size>=release.MAX_FILE_BYTES:raise ValueError("Snapshot file exceeds ordinary Git size budget")
        release.secret_scan(asset)
    if sum((directory/name).stat().st_size for name in actual)>=release.MAX_TOTAL_BYTES:raise ValueError("Snapshot exceeds ordinary Git size budget")
    reference=manifest["integrity_audit"];audit_name=release.safe_relative(reference["path"])
    if audit_name not in names or reference.get("sha256")!=release.file_hash(directory/audit_name) or reference.get("schema")!="two_pass_full_integrity_audit/1" or reference.get("valid")is not True or reference.get("publish_ready")is not False or reference.get("run_fingerprint")!=manifest["run_fingerprint"]:
        raise ValueError("Snapshot audit reference mismatch")
    check_gate(release.read_json(directory/audit_name),manifest["run_fingerprint"],expected_pairs,expected_first_only)
    verify_snapshot_payload(directory,manifest)
    release.secret_scan(path)
    return manifest


class Shards:
    def __init__(self,root,stem,role,limit,register):
        self.root,self.stem,self.role,self.limit,self.register=root,stem,role,limit,register
        self.raw=self.handle=None;self.count=self.local=self.number=0
    def close(self):
        if self.handle is not None:
            self.handle.close();self.raw.close()
            self.register(f"data/{self.stem}-{self.number:05d}.jsonl.gz",self.role,self.local)
            self.raw=self.handle=None
    def add(self,row):
        if self.handle is None or self.local==self.limit:
            self.close();self.number+=1;self.local=0
            self.raw,self.handle=release.open_gzip_text(self.root/f"data/{self.stem}-{self.number:05d}.jsonl.gz")
        self.handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
        self.count+=1;self.local+=1


def package_snapshot(run_dir,audit_path,output_dir,assets_manifest=None,rows_per_shard=2000,
                     expected_pairs=EXPECTED_PAIRS,expected_first_only=EXPECTED_FIRST_ONLY,snapshot_path=None):
    run_dir,audit_path,output_dir=Path(run_dir).resolve(),Path(audit_path).resolve(),Path(output_dir).resolve()
    if output_dir.exists():raise ValueError("Snapshot destination must be new; existing releases are preserved")
    if not 1<=rows_per_shard<=10000:raise ValueError("Shard size must be in 1..10000")
    config=release.read_json(run_dir/'run_config.json');audit=release.read_json(audit_path)
    fingerprint=config['run_fingerprint'];check_gate(audit,fingerprint,expected_pairs,expected_first_only)
    paired=run_dir/'paired/annotations.jsonl'
    audited_exports={str(Path(e['path']).resolve()):e['sha256'] for e in audit.get('data_exports',[])}
    audited_inputs={str(Path(p).resolve()):sha for p,sha in audit.get('input_hashes',{}).items()}
    paired_hash=release.file_hash(paired);config_hash=release.file_hash(run_dir/'run_config.json');audit_hash=release.file_hash(audit_path)
    if audited_exports.get(str(paired))!=paired_hash or audited_inputs.get(str(run_dir/'run_config.json'))!=config_hash:
        raise ValueError("Snapshot paired export or configuration is not the audited file")
    assets_manifest=Path(assets_manifest) if assets_manifest else run_dir/'release_assets.json'
    asset_list=release.read_json(assets_manifest)['files'];assets_hash=release.file_hash(assets_manifest)
    repo_path=snapshot_path or f"snapshots/two_pass_{expected_pairs}_{datetime.now(timezone.utc):%Y%m%d}_{fingerprint[:12]}"
    if not re.fullmatch(rf"snapshots/two_pass_{expected_pairs}_[0-9]{{8}}_{fingerprint[:12]}",release.safe_relative(repo_path)):
        raise ValueError("Snapshot repository path must identify this dated partial population")
    output_dir.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=output_dir.name+'.building-',dir=output_dir.parent));entries=[]
    def register(name,role,rows=None):
        release.safe_relative(name);path=temporary/name;release.secret_scan(path)
        if path.stat().st_size>=release.MAX_FILE_BYTES:raise ValueError("Snapshot shard/asset too large")
        if name.casefold() in {e['path'].casefold() for e in entries}:raise ValueError("Duplicate asset destination")
        entries.append({'path':name,'role':role,'sha256':release.file_hash(path),'size_bytes':path.stat().st_size,**({'rows':rows} if rows is not None else {})})
    writers=[]
    try:
        (temporary/'data').mkdir()
        paired_writer=Shards(temporary,'paired','paired_training_jsonl',rows_per_shard,register)
        first_writer=Shards(temporary,'first-pass-only','first_pass_only_jsonl',rows_per_shard,register)
        writers=[paired_writer,first_writer];preview=[];seen=set();last=0;source_count=0
        with paired.open(encoding='utf-8') as handle:
            for line in handle:
                original=json.loads(line);source_count+=1
                index,cid=original.get('source_index'),original.get('chembl_id')
                if type(index)is not int or not last<index<=TARGET or not isinstance(cid,str) or cid in seen:raise ValueError("Audited source order/identity mismatch")
                last=index;seen.add(cid)
                if original.get('pair_status')=='complete_pair':
                    check_source_row(original);row=release.training_row(original);validate_row(row,'paired_training_jsonl');paired_writer.add(row)
                    if len(preview)<100:preview.append(row)
                elif accepted(original.get('first_pass')):
                    row=first_only_row(original);validate_row(row,'first_pass_only_jsonl');first_writer.add(row)
                elif accepted(original.get('second_pass')):
                    raise ValueError("Accepted second pass exists without an accepted first pass")
        for writer in writers:writer.close()
        if source_count!=TARGET or paired_writer.count!=expected_pairs or first_writer.count!=expected_first_only:raise ValueError("Actual snapshot populations disagree with the audited counts")
        (temporary/'metadata').mkdir();shutil.copyfile(audit_path,temporary/'metadata/integrity_audit.json');register('metadata/integrity_audit.json','integrity_audit')
        release.write_json(temporary/'preview.json',{'schema_version':'two_pass_snapshot_preview/1','sample_only':True,'sample_count':len(preview),
            'note':'First complete-pair records for display only; already included in the paired data shards. Not additional dataset rows.','records':preview})
        register('preview.json','preview_json',len(preview))
        with (temporary/'preview.csv').open('w',newline='',encoding='utf-8') as handle:
            writer=csv.DictWriter(handle,fieldnames=release.CSV_FIELDS);writer.writeheader()
            for row in preview:
                flat={**row,**row['properties']};writer.writerow({key:release.cell(flat.get(key)) for key in release.CSV_FIELDS})
        register('preview.csv','preview_csv',len(preview))
        schema={'schema_version':'two_pass_snapshot_data_schema/1','target_count':TARGET,'exported_pair_count':expected_pairs,
            'exported_first_pass_only_count':expected_first_only,'feature_names':release.FEATURE_NAMES,'property_vector_order':'feature_names',
            'paired_schema':'two_pass_training_record/1','paired_description':'Exact accepted second-pass text; first_description preserves its first pass.',
            'first_only_schema':FIRST_ONLY_SCHEMA,'first_only_description':'Exact accepted first-pass text with no accepted second-pass result; second_description/second_pass are null.',
            'source_index':'Original 1..100000 source index, retained without renumbering. Each released subset has gaps.',
            'preview':'Display sample from paired shards, not additional rows.','chemistry_verified':False}
        release.write_json(temporary/'metadata/data_schema.json',schema);register('metadata/data_schema.json','data_schema')
        roles=set()
        for asset in asset_list:
            name=release.safe_relative(asset['path']);source=Path(asset['source'])
            if not name.startswith(('code/','prompts/','docs/','licenses/')) or source.is_symlink() or release.file_hash(source)!=asset['sha256']:
                raise ValueError("Frozen allowlisted workflow asset mismatch")
            destination=temporary/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,destination)
            if release.file_hash(destination)!=asset['sha256']:raise ValueError("Copied workflow asset hash mismatch")
            register(name,asset.get('role','supporting_asset'));roles.add(asset.get('role'))
        if not REQUIRED_ASSET_ROLES<=roles:raise ValueError("Snapshot lacks required licensing, attribution, prompts or source/workflow documentation")
        readme=f'''# 当前两轮标注快照 / Partial two-pass annotation snapshot

目标为 {TARGET:,} 个分子的两轮完整配对。本次保存并发布 **{expected_pairs:,} 条完整配对**，以及独立的 **{expected_first_only:,} 条已有接受的第一轮正文、尚无已接受的第二轮结果**。共覆盖 {expected_pairs+expected_first_only:,} 条已接受的第一轮结果；仍有 {TARGET-expected_pairs:,} 个分子未完成配对。**这是阶段快照，不是最终全量发布。**

- `data/paired-*.jsonl.gz`：完整配对。`description` 为原样保留的第二轮正文，`first_description` 为同分子的第一轮正文。
- `data/first-pass-only-*.jsonl.gz`：尚无已接受的第二轮结果；可能已尝试第二轮但失败或未通过格式检查。`description` 仅为第一轮原文，`description_origin=first_pass_unreviewed`，`review_completed=false`，`second_description` 和 `second_pass` 均为空。此组不得误作已审校数据。
- `preview.json` / `preview.csv`：前 {len(preview)} 条完整配对的可读样本，已包含在数据碎片中，不增加数据行数。
- `metadata/data_schema.json`：24 项数值性质与向量顺序。`source_index` 保留原始来源索引，各发布子集存在间隔，绝不重新编号为 1..{expected_pairs}。
- `metadata/integrity_audit.json`：当前保存状态的来源、请求、响应、性质和导出一致性审计，`valid=true`、`publish_ready=false`。完整 100,000 条发布门槛仍然保留。
- `snapshot_manifest.json`：全部文件的白名单、行数、字节数和 SHA-256。

`code/`、`prompts/`、`docs/`、`licenses/` 原样复制已固定的 {len(asset_list)} 个工作流资产。运行文档中的 100,000 是整个流程目标，不表示本快照已完成全量。原代码和配置中的路径是来源记录；不要从这些副本无意恢复原任务。ChEMBL 许可和 required attribution 保留在 `licenses/`；未另行授予新的代码许可证。

This is an explicitly incomplete saved-state snapshot. It contains {expected_pairs:,} accepted first-pass/review pairs and a separate {expected_first_only:,} first-pass-only set with **no accepted second-pass result**. IDs are disjoint between these sets. Source indices retain their original values and may have gaps. Preview files repeat a sample already present in the paired shards.

Integrity checks are not chemical accuracy certification. Model self-review is not independent expert verification; `chemistry_verified` remains false. Raw API responses/reasoning, credentials, request logs and SQLite files are excluded from the data release. Text changes are not correctness scores.
'''
        (temporary/'README.md').write_text(readme,encoding='utf-8');register('README.md','readme')
        manifest={'schema_version':SCHEMA,'created_at_utc':datetime.now(timezone.utc).isoformat(),'run_fingerprint':fingerprint,
            'snapshot_path':repo_path,'target_count':TARGET,'exported_pair_count':expected_pairs,'exported_first_pass_only_count':expected_first_only,
            'first_pass_accepted':expected_pairs+expected_first_only,'second_pass_accepted':expected_pairs,'remaining_pairs':TARGET-expected_pairs,
            'snapshot_complete':False,'publish_ready':False,'chemistry_verified':False,'workflow_asset_count':len(asset_list),
            'workflow_assets_manifest_sha256':assets_hash,'audited_paired_export_sha256':paired_hash,'audited_run_config_sha256':config_hash,
            'integrity_audit':{'path':'metadata/integrity_audit.json','sha256':audit_hash,'schema':audit['schema_version'],'valid':True,'publish_ready':False,'run_fingerprint':fingerprint},
            'files':entries,'secret_scan':{'passed':True,'scope':'All listed UTF-8 files, including decompressed data shards; conservative credential-pattern scan, not exhaustive.'},
            'package_code_sha256':release.file_hash(Path(__file__)),'shared_projection_code_sha256':release.file_hash(Path(release.__file__))}
        release.write_json(temporary/MANIFEST_NAME,manifest)
        verify_snapshot(temporary,fingerprint,expected_pairs,expected_first_only)
        for path,checksum in [(paired,paired_hash),(run_dir/'run_config.json',config_hash),(audit_path,audit_hash),(assets_manifest,assets_hash)]:
            if release.file_hash(path)!=checksum:raise ValueError("Audited snapshot input changed during packaging")
        temporary.rename(output_dir)
        return manifest
    except BaseException:
        for writer in writers:
            if writer.handle is not None:
                writer.handle.close();writer.raw.close();writer.handle=writer.raw=None
        shutil.rmtree(temporary,ignore_errors=True)
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--assets-manifest',type=Path)
    parser.add_argument('--rows-per-shard',type=int,default=2000)
    parser.add_argument('--expected-pairs',type=int,default=EXPECTED_PAIRS)
    parser.add_argument('--expected-first-only',type=int,default=EXPECTED_FIRST_ONLY)
    parser.add_argument('--snapshot-path')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--expected-run-fingerprint')
    args=parser.parse_args(argv)
    if args.verify_only:result=verify_snapshot(args.output_dir,args.expected_run_fingerprint,args.expected_pairs,args.expected_first_only)
    else:
        if not args.run_dir or not args.audit:parser.error('--run-dir and --audit are required')
        result=package_snapshot(args.run_dir,args.audit,args.output_dir,args.assets_manifest,args.rows_per_shard,args.expected_pairs,args.expected_first_only,args.snapshot_path)
    print(json.dumps({key:result[key] for key in ['schema_version','target_count','exported_pair_count','exported_first_pass_only_count','snapshot_complete']}))


if __name__=='__main__':main()
