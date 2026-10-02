# LTH 首轮精简实验：目标论文原文 vs 可检索跨论文图

版本：2026-10-02（已批准 v2）  
状态：先落实协议和 plan-only，再开始 live pilot  
完整方案：[`autoresearch-icml-plan.md`](./autoresearch-icml-plan.md)
候选题池：[`lth-pilot-10-question-catalog.md`](./lth-pilot-10-question-catalog.md)

## 1. 这一轮到底要实现什么

在同一个有争议的 LTH 问题上，给 agent 看目标论文原文，或者让它在同一目标论文上检索跨论文图，哪一种更能让它找到真正需要外部证据的方向，并写出有证据、条件匹配、能区分假设的下一步实验？

模型不知道评分器的答案，也看不到预先列好的动作或候选实验。它先发散给出 3–5 个方向，再选一个输出实验卡。首轮先做协议和证据质量 pilot；真正的“最值得做”要等有可执行 LTH 入口和盲审/执行结果后才能声称。

## 2. 输入和输出

两种条件都看到同一份带段落锚点的目标论文原文、研究问题、当前现象、硬约束、证据截止时间和预算。

| 条件 | 额外能力 | 目的 |
|---|---|---|
| `paper_only` | 只能在目标论文原文内检索段落 | 单论文基线 |
| `external_text` | 检索同一外部论文池的普通文本片段 | 控制“多了外部信息” |
| `reasoning_graph` | 检索同一论文池的节点、typed edge、条件和 source anchor | 测图结构是否比文本更有用 |

`reasoning_graph` 必须是真检索，不是把全图 JSON 一次性塞进 prompt。最小接口是：

```text
search_claims(query, k)
expand_neighbors(node_or_edge_id, k)
get_source_anchor(anchor_id)
```

每次调用都保存 query、返回 ID、选中 ID、调用次数和 token。目标论文在两种条件中都相同；只有外部证据访问方式改变。首轮只做两组也可以，但那时只能说“跨论文信息有帮助”，不能说“图结构优于文本”。正式结论需要三组。

输出不是固定 action enum。模型输出：

- 3–5 个自己提出的 `next_step` 方向；
- 选中的方向和一句短理由；
- 论文段落或图节点/边的引用（含短 quote 和 anchor ID）；
- 条件、baseline、treatment、control、metrics、entrypoint、预算；
- 预期观察、falsifier、什么结果会改变下一步选择、不确定性；
- `retrieval_trace`，即它实际查了什么、看到了什么、用了什么。

动作类别由评分器事后归类，不出现在模型输入中；`abstain` 仍是合法的自由文本决定。

## 3. 先锁定一个重点 bundle

首轮先只做一个重点 bundle，避免样本空间和评分一起膨胀：**rewinding vs initialization / mask semantics**。它天然需要跨论文证据：目标论文容易让模型直接建议复现 early-rewinding；只有补进外部论文关于 initialization、large-batch、transfer 和 layer-wise pruning 的结果，才有理由提出匹配的 2×2 对照。

建议 focal paper 是 `Stabilizing the Lottery Ticket Hypothesis`（本地 paper id `867769152067076667`），图条件从 89 篇全图中检索：

- `Towards Understanding Iterative Magnitude Pruning`（`867750791409041515`）：初始化、batch size 与稳定训练；
- `One ticket to win them all`（`867771291879342656`）：mask 的迁移证据；
- 可选 `When Layers Play the Lottery`（`867756402284691503`）：layer-wise initialization/pruning 条件。

问题只描述可观察现象和目标，不直接告诉模型 initialization、rewinding、warmup 或 2×2 方向。候选机制和可接受干预族放在 hidden rubric，避免题面把答案范围泄漏给模型。例如题面可以只写：**“在 ResNet-18/CIFAR-10、30K iterations、batch size 128、最终 80% sparsity 的设置下，现有结果互相矛盾。下一步最值得做什么实验来区分可能原因？请给实验卡，并写出会推翻当前判断的结果。”**

hidden rubric 可以把 matched rewinding/schedule 对照作为一种可接受干预族，但 P01 等外部结果必须标作机制线索和条件限制，不能直接当作 ResNet-18 结果。

首轮可以把同一 bundle 做成 3 个独立 task seed；若这组 bridge audit 不通过，再退回到“暂缓判断”，而不是临时添加第二个科学问题。

每道题先由人工锁定任务事实和评分器信息；模型输入不含 allowed action：

```json
{
  "task_id": "lth_bridge_01",
  "question": "...",
  "focal_paper": {"paper_id": "...", "full_text": "..."},
  "graph_snapshot": "sha256:...",
  "external_corpus": "same frozen corpus for text and graph retrieval",
  "required_conditions": ["dataset", "model", "sparsity", "rewind_point"],
  "evidence_cutoff": "YYYY-MM-DD",
  "budget": {"max_seconds": 600, "max_gpu_hours": 0}
}
```

每道题至少有 `paper_only` 和 `reasoning_graph` 两份输入；正式图优势实验再加入 `external_text`。任务 id、问题文字、截止时间、输出 schema、模型、seed、检索调用预算都相同。评分器可以有 hidden rubric，但模型看不到。

## 4. 输出保留开放想法和一张可审计实验卡

每次输出 3–5 个短方向、一个最终方向和一张卡，并附公开、结构化的证据 trace。trace 只记录证据、关系、条件、决定和不确定性，不要求冗长的内部思维链：

```json
{
  "task_id": "...",
  "next_step": "自由文本",
  "reason": "一句短理由",
  "evidence": [
    {"anchor_id": "...", "quote": "...", "source": "paper|graph", "role": "support|limitation|context", "relation_id": "..."}
  ],
  "conditions": [{"key": "...", "value": "..."}],
  "retrieval_trace": [
    {"query": "...", "returned_ids": [], "chosen_ids": [], "tool": "search_claims|expand_neighbors|get_source_anchor"}
  ],
  "experiment": {
    "baseline": "...",
    "treatment": "...",
    "controls": ["..."],
    "metrics": ["..."],
    "entrypoint": "... or null",
    "expected_observation": "...",
    "falsifier": "...",
    "budget": {"max_seconds": 600, "max_gpu_hours": 0}
  },
  "decision_rule": "什么结果会改变下一步选择"
}
```

`abstain` 是合法输出：如果 evidence cutoff、条件或 entrypoint 不够，安全暂缓比编造实验更好。

## 5. 评分先看“有没有用对证据”

首轮只报五项，避免指标过多：

- `CardValid@1`：schema、自由文本 next_step、实验字段和引用闭包都合法；
- `EvidenceIntegrity`：anchor、引用短文、relation 真存在，且没有跨过截止时间；
- `ConditionPreservation`：模型没有把不同数据集、架构、稀疏率或 rewind 条件混在一起；
- `CrossPaperUse@1`：在 `reasoning_graph` 下是否引用了真正必要的外部 claim/关系，并用于改变实验设计；
- `UnsupportedDecisionRate`：没有足够证据仍给出具体实验的比例。

对非 `abstain` 卡做一个盲审分数：科学必要性、baseline/control、falsifier、预算可执行性各 0–2 分。两名标注者只审 6 个任务卡，报告一致性；LLM judge 只作辅助，不作为唯一金标。

主结果是每道题每个 seed 的配对差：

```text
reasoning_graph − paper_only
```

首轮只把“图条件的 `CrossPaperUse@1` 上升，同时 `UnsupportedDecisionRate` 不上升”作为值得继续扩大的信号。3 道题太小，结果只能称为 pilot，不能直接写成 ICML 主结论。

## 6. 当前工作区图的实际版本

最终图只认 [`merge.html`](../merge.html)。它的嵌入 `view-data` 已锁定到
`paper2ir_experiment_copy/evidence/lth_final_graph_20261002/`，当前审计为 240 个节点、235 条边、103 个无向连通分量，其中 102 个节点孤立。这个 viewer 图没有节点 source anchors、没有 `source_targets`，也没有可从 payload 直接验证的 paper id；当前 anchor coverage 为 0/240。因此 `get_source_anchor` 对未补回的 anchor 必须返回 unavailable，不能为了让卡片通过而编造 provenance。

已经做了一次保守的 sidecar 回溯，但它不能改写 `merge.html`：从每个 cluster 的 `gaia.ir.json` strategy 反向追到论文 claim，再连接本地 `formalization.json` 和 `ocr.md`。结果是 220 个节点为 `derived`、20 个节点 unresolved；235 条边中 125 条只有两端的 derived closure、110 条 unresolved；direct provenance 仍为 0。报告在 [`provenance_mapping_report.md`](../paper2ir_experiment_copy/evidence/lth_final_graph_20261002/provenance_mapping_report.md)，因此当前主 pilot 仍把图 anchor credit 关闭。

图里的 `K02`、`K03` 等 domain summary 是展示层的人工整理节点。它们不能直接放入主任务的模型输入，否则题目会把“mask/init”或“stability/rewinding”答案写给模型。首轮只允许检索原始 claim/observation 及其经 audit 的关系；summary 节点留作 evaluator-only 或单独的 answer-leakage 对照。235 条边虽然在 viewer 中都标为 `confirmed`，但只有 99 条保留 `original_relation`、96 条有 confidence；`confirmed` 不能代替 source-grounded provenance。

在生成三个 seed 任务前做一次 bridge audit：回到原文为目标 claim、外部 claim 和 relation 补齐 source anchor、paper id、条件和 cutoff。最终图负责提供结构和候选路径，原文负责提供可评分的证据来源；两者不能混为一个金标准。

## 7. 怎么复用现有代码和 benchmark

现有实现直接复用：

1. `pipelines/single_paper` 负责单论文 claim/observation/anchor；
2. `pipelines/merge` 负责跨论文 relation/conclusion；
3. `paper2ir_experiment_copy/research_triage` 负责 manifest、proposal schema、确定性校验、配对聚合和报告。

也就是说，不再新增一条“第三种抽取 pipeline”；新增的只是 LTH bridge task fixture、`paper_only`/`reasoning_graph` 两种 renderer 和 `CrossPaperUse` 评分。这一点与完整计划保持一致。

MLAgentBench 先只作接口 sanity check：它的 evaluator 能判断“生成代码后指标是否改善”，不能判断 LTH 方向本身是否值得做。因此首轮 LTH 结果只报告实验卡和证据评分；等找到固定的 LTH/OpenLTH entrypoint 后，才填 `ExecutionPass`。ScienceAgentBench 放到第二阶段，必须先锁 official verified split 和官方 harness，不能用现有 parquet 代替官方结果。

## 8. 第一批实际运行规模

先做一个不发模型请求的 plan-only 检查：

```text
1 bundle × 3 conditions × 3 seeds = 9 cells
```

确认每个 cell 的任务 hash、graph snapshot hash、token budget、prompt 差异和输出目录后，再发起 live call。当前三模型 plan-only 产物为 [`lth_graph_v2_plan_20261002`](../paper2ir_experiment_copy/evidence/lth_graph_v2_plan_20261002/)，共 27 个计划 cell；它只写 prompt 和 manifest，没有发模型请求。第一轮 live 仍只生成设计卡，不执行训练代码；没有合法 LTH entrypoint 时允许 `experiment.entrypoint=null`，但不得伪造 `ExecutionPass` 或训练结果。

该 manifest 还保存了自动 fairness audit：三个条件看到相同的目标论文（SHA256 `4a250a...63afa`），共用 8 次调用/4000 返回 token 的总预算，并从同一份排除 focal paper 的 88 篇外部语料检索。审计同时保留两个警告：`merge.html` 当前没有可评分的 graph source anchor；目标论文原文已经出现 rewinding、initialization、warmup 和 schedule 等可能的答案词。因此这组 bundle 不能单独证明跨论文图的增益。

为避免把这组 warning 当成主结果，另外冻结了一个更适合做 bridge audit 的候选任务：[`lth_bridge_v2_bingo_task.json`](../paper2ir_experiment_copy/research_triage/fixtures/lth_bridge_v2_bingo_task.json)。它以只报告单一 MNIST/feed-forward 设置的 BINGO 论文为 focal paper，问题要求判断效率优势能否迁移到其他数据集和模型；自动审计将它标为 `bridge_suitability=candidate`。对应的 27-cell plan 在 [`lth_bingo_v2_plan_20261002`](../paper2ir_experiment_copy/evidence/lth_bingo_v2_plan_20261002/)；它仍是 plan-only，且 graph provenance warning 仍然保留。

旧版三模型版本已经冻结在
`paper2ir_experiment_copy/evidence/lth_graph_vs_paper_pilot_multimodel_20261001/`；它是 legacy baseline，不能冒充 v2 结果。v2 运行必须新建目录：

```text
1 bundle × 3 conditions × 3 seeds × 3 model profiles = 27 cells
model profiles: deepseek_v4_1, trisol_qwen7b, trisol_qwen35b
```

每个 cell 的目录身份是
`<task>/<model_profile>/<condition>/seed-<seed>`。profile 只保存环境变量名，
不把 API key 写入任务、输出或日志；模型、provider、endpoint host、task/prompt/
response hash 和 seed 写入 `metadata.json`。三模型之间分别做配对差，不把不同模型
混成一个样本。

产物目录建议为：

```text
paper2ir_experiment_copy/evidence/lth_graph_vs_paper_pilot_YYYYMMDD/
  manifest.json
  tasks.jsonl
  prompts/
  proposals/
  scores/
  report.md
```

每次运行追加到 [`research_triage/CHANGELOG.md`](../paper2ir_experiment_copy/research_triage/CHANGELOG.md)，记录代码 commit、graph snapshot hash、task/condition/seed、模型响应 hash、validator result 和失败类型；不覆盖历史 evidence。

三模型 manifest 的状态仍是 `planned_authorization_required`、`api_requests_made=0`；
缺失某个平台的 endpoint 或 credential 时，runner 在请求前失败并保留 bounded
error metadata，不自动换模型或重试到另一个 provider。

## 9. 这轮完成的判定

完成标准很简单：

1. 一个 bundle 的三个 seed 都有人工复核过的 target/external/bridge/condition/cutoff；
2. 三个条件只差外部证据表示，且共用目标论文原文和检索预算；
3. 9 个 plan-only cell 全部能展开，且 prompt 没有 hidden rubric、allowed action 或候选列表；
4. live pilot 能逐项审计引用和条件；
5. 报告 paired difference，并明确说明没有 LTH 执行结果。

如果图只让卡片更完整，却没有提高跨论文证据使用或降低无依据决策，结果仍然有价值：它会把论文主张收窄为“改善可审计性”，而不是声称图能自动选出更好的科学实验。

## 10. 网络访问和模型先验

主 pilot 使用 `knowledge_access=closed_book`：关闭 browser、web search 和未冻结的外部网络，只允许 provider API 连接；`external_text` 与 `reasoning_graph` 使用本地冻结语料和受控 retrieval API。三种条件使用相同的模型、prompt、seed、输出 token、检索调用数和返回 token；`paper_only` 的外部调用数记为 0，不能把 live web search 只给 graph 条件。

关闭网络不能删除基础模型的预训练记忆。主结论应表述为“在闭卷、冻结语料和受控检索下，图相对目标论文/文本检索的增益”；模型说出输入外的论文、数字或事实时，不得获得 evidence credit，并计入 `unanchored_claim`/unsupported 审计。以后如需研究开放检索，另设 `open_web` stress arm，不能混入主结果。

## 11. 首轮仍未解决的严谨性问题

这轮代码已经把输入边界和三种条件固定下来，但下面几件事还不能被结果替代：

1. **还没有真实 provider 结果。** 离线 `live_tool_loop.py` 已经实现工具调用、预算和真实 trace 校验，但 27 个 cell 仍是 plan-only，尚未向 DeepSeek/Trisol 发请求；因此现在不能报告图相对文本的性能提升。
2. **最终图的 provenance 仍不够。** `merge.html` 的 240 个节点没有直接 source anchor。sidecar 只能把一部分节点和边回溯为 derived，不能把 derived closure 当成原文直接证据，所以当前图证据分数必须保持关闭。
3. **没有可执行的 LTH 入口。** 当前任务是 `design_only`、GPU 预算为 0；它只能评实验卡和证据使用，不能证明 agent 选出的实验真的更有用或能跑通。
4. **开放式“最好下一步”没有唯一金标。** 必须先锁定可接受的干预族，再由不知道条件的多名专家盲评科学必要性、区分力、控制变量、falsifier 和成本，并报告一致性。单靠 schema 通过不够。
5. **首轮样本太小。** 一个 bundle、三个 seed 只能发现协议问题。即使三种条件都跑完，也只能叫 pilot；跨论文结论要扩展到多个 bridge-required task，并按 task 配对统计。
6. **图与文本的公平性还要用实际日志检查。** 三种条件必须使用同一冻结论文池、截止时间、返回 token、调用次数和延迟预算；否则 `reasoning_graph` 的收益可能只是看到了更多文本或调用了更多次。
7. **仍需反事实和负例。** 删除关键 bridge edge、加入条件不匹配的 decoy paper、以及不依赖外部论文的 negative control，才能判断图是否真的被使用，而不是只输出一个听起来合理的 LTH 实验。

所以，首轮的可接受结论只有两类：先验证协议、检索和证据审计是否工作；或者在满足上述执行和盲评门槛后，报告 `reasoning_graph − external_text` 的配对差。不能用当前 plan-only 或旧版六节点结果宣称“图让实验选择更好”。
