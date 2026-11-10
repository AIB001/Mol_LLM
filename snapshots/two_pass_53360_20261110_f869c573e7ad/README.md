# 当前两轮标注快照 / Partial two-pass annotation snapshot

目标为 100,000 个分子的两轮完整配对。本次保存并发布 **53,360 条完整配对**，以及独立的 **2,190 条已有接受的第一轮正文、尚无已接受的第二轮结果**。共覆盖 55,550 条已接受的第一轮结果；仍有 46,640 个分子未完成配对。**这是阶段快照，不是最终全量发布。**

- `data/paired-*.jsonl.gz`：完整配对。`description` 为原样保留的第二轮正文，`first_description` 为同分子的第一轮正文。
- `data/first-pass-only-*.jsonl.gz`：尚无已接受的第二轮结果；可能已尝试第二轮但失败或未通过格式检查。`description` 仅为第一轮原文，`description_origin=first_pass_unreviewed`，`review_completed=false`，`second_description` 和 `second_pass` 均为空。此组不得误作已审校数据。
- `preview.json` / `preview.csv`：前 100 条完整配对的可读样本，已包含在数据碎片中，不增加数据行数。
- `metadata/data_schema.json`：24 项数值性质与向量顺序。`source_index` 保留原始来源索引，各发布子集存在间隔，绝不重新编号为 1..53360。
- `metadata/integrity_audit.json`：当前保存状态的来源、请求、响应、性质和导出一致性审计，`valid=true`、`publish_ready=false`。完整 100,000 条发布门槛仍然保留。
- `snapshot_manifest.json`：全部文件的白名单、行数、字节数和 SHA-256。

`code/`、`prompts/`、`docs/`、`licenses/` 原样复制已固定的 51 个工作流资产。运行文档中的 100,000 是整个流程目标，不表示本快照已完成全量。原代码和配置中的路径是来源记录；不要从这些副本无意恢复原任务。ChEMBL 许可和 required attribution 保留在 `licenses/`；未另行授予新的代码许可证。

This is an explicitly incomplete saved-state snapshot. It contains 53,360 accepted first-pass/review pairs and a separate 2,190 first-pass-only set with **no accepted second-pass result**. IDs are disjoint between these sets. Source indices retain their original values and may have gaps. Preview files repeat a sample already present in the paired shards.

Integrity checks are not chemical accuracy certification. Model self-review is not independent expert verification; `chemistry_verified` remains false. Raw API responses/reasoning, credentials, request logs and SQLite files are excluded from the data release. Text changes are not correctness scores.
