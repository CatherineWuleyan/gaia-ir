# E0：Ground-truth calibrated claim graph（pilot）

这是 Paper2IR 一个月路线的第一步：先用确定性 checker 检查“答案对不对”之外的科学证据约束是否被保留。

当前目录只包含一个小型、可审计的 pilot fixture，不把它当成最终 benchmark 结果。`questions.jsonl` 中的第一题从 `weekly_reports/out/domain_graph_lkm/gaia.ir.json` 的真实 `conjunction` operator 自动抽取，答案是图编译器的结构化结论；它用于验证 schema 和 checker，不用于宣称模型能力。后续题目必须来自人工核验的 LTH 论文 claim、observation、condition 和 source anchor。

## 运行

```bash
python benchmark/e0/strict_check.py \
  --questions benchmark/e0/questions.jsonl \
  --responses benchmark/e0/sample_responses.jsonl
```

输出为 JSON，包含每题 `strict_success` 和诊断项。checker 不调用 LLM；任何未被 gold 明确允许的前提、边类型、条件或 provenance 都会被标记。

## 目标 gold schema

- `answer`: 规范化答案或 `null`（unknown）
- `status`: `supported` / `unknown` / `conflict`
- `required_premises`: 必须出现的 claim/observation ID
- `evidence_path`: 有序的 `{source_id, relation}` 边
- `conditions`: 精确的条件键值集合
- `provenance`: 允许回溯的 source/anchor ID

最终 E0 需要扩展到 5–10 篇 LTH 论文、120–200 题，并由领域人工核验；当前 pilot 只验证 checker 的行为。
