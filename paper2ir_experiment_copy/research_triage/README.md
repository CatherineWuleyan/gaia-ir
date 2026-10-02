# Research Triage Harness

这是 Paper2IR 的最小可审计实验分诊接口。它不自建 benchmark，也不负责生成隐藏 chain-of-thought；它只约束 agent 输出一个可验证 proposal，并把证据、条件、实验入口和预算保存成 artifact。

## 当前能力

### LTH v2：最终图检索 pilot

首轮 LTH v2 固定 [`merge.html`](../../merge.html) 为唯一图源。`v2_protocol.py`
给三种条件都传目标论文全文；`paper_only` 只能检索论文，`external_text` 检索
同一份冻结外部语料，`reasoning_graph` 才能调用完整图的检索工具。图节点和边
不会预先塞进 prompt。

最终图的配套索引已冻结在
`evidence/lth_final_graph_20261002/`：`graph_index.json` 是完整
`view_model.reference.json` 的精确副本，`knowledge.index.json` 是同一次 merge
run 的 claim/source-anchor 索引，`artifact_index.json` 是运行 artifact 注册表。
对 `merge.html` 做计划时，`v2_plan.py` 会传入并校验 `graph_index.json` 的
canonical hash；旧索引会直接拒绝加载。索引只服务于只读检索和审计，不能改变
`merge.html` 的权威图或把间接 anchor 变成 direct provenance。

离线工具循环在 `live_tool_loop.py`：模型先请求工具，runner 在本地执行并记录
typed retrieval trace，最后用真实 trace 调 `validate_v2`。`ScriptedProvider` 只做
离线 preflight，不产生网络请求。`v2_fairness.py` 会检查目标论文 hash、focal
paper 排除、8 次调用/4000 返回 token 预算、资源隔离和 merge.html hash。

```bash
PYTHONPATH=paper2ir_experiment_copy .venv-gaia-a7/bin/python -m pytest -q \
  paper2ir_experiment_copy/research_triage/test_live_tool_loop.py \
  paper2ir_experiment_copy/research_triage/test_v2_fairness.py
```

当前原始 rewinding bundle 的 `bridge_suitability` 是 `warning`，BINGO 跨设置
任务是下一道 `bridge_required` 候选题；两者都先 plan-only，不能把离线 preflight
当成模型效果结果。

```bash
python research_triage/validate_proposal.py \
  --task research_triage/fixtures/task.json \
  --proposal research_triage/fixtures/proposal_good.json
```

验证包括：

- JSON schema 必填字段；
- action 是否在允许集合；
- claim/source anchor 是否属于 task evidence；
- condition key 是否属于 task condition vocabulary；
- entrypoint 是否为相对路径且没有路径穿越；
- 预算是否为非负有限值；
- `abstain` 是否真的没有伪造实验结果。
- 若 task 声明 `evidence_cutoff`，冻结证据集合中的每个 source anchor 必须有
  `source_anchor_metadata[anchor].available_at`，且不晚于 cutoff；否则即使模型
  不引用该 anchor 也会拒绝任务，避免未来证据偷偷进入提示。
- 非 `abstain` 卡的 baseline/treatment 必须不同，controls 不能为空，
  expected observation 和 falsifier 不能完全相同；metric 与 falsifier 的
  词面重叠只作为 warning，不是语义正确性的证明。

若已有官方 sandbox，可额外传 `--entrypoint-root`；validator 会确认卡片中的
相对入口确实落在 sandbox 内且文件存在。默认不执行入口，也不会把字符串检查
冒充执行成功。若任务没有声明任何允许的入口，非 `abstain` 卡会直接失败，
防止模型编造不可执行的入口。

它只验证 proposal，不执行任意 agent 代码。执行要通过 ScienceAgentBench/MLAgentBench 官方容器，并由现有 `agent/pipeline_harness` 的 artifact/checkpoint 机制托管。

要让 agent 从冻结任务生成实验卡，可运行：

```bash
PYTHONPATH=. python -m research_triage.model_proposer \
  --task research_triage/fixtures/task.json \
  --out evidence/model_proposer_fixture
```

该命令只保存短理由、结构化字段、模型/提示哈希和原始响应；不保存隐藏思维过程。task 可以用 `action_requirements` 声明每种动作必须出现的 evidence role；例如 `contrast` 可要求同时出现 `support` 和 `limitation`。这只是可机械检查的必要条件；通过它不证明论据蕴含结论，也不证明所选实验最有价值。

### 测量“最值得做哪个实验”

证据闭包只能说明实验卡可审计，不能说明所选动作在候选集合中最有信息价值。需要测量选择质量的任务可以声明模型可见的 `candidate_experiments` 列表，并把 `a_tier_candidate_ids`、可接受选择、优先顺序和成本放在单独的 evaluator-only oracle artifact 中。非 `abstain` 卡必须填 `experiment.candidate_id`；`candidate_evaluator.py` 随后输出 `DecisionChoiceValid@1`、`DecisionQuality@1`（Top-1 A-tier）、接受率、弃权正确率和成本 regret。oracle 从不进入模型请求；没有这类标注的任务对应指标保持 `null`，不会把卡片合法性当成决策质量代理。

本地候选选择 fixture 可直接复核：

```bash
PYTHONPATH=. python -m research_triage.candidate_evaluator \
  --task research_triage/fixtures/candidate_choice_task.json \
  --proposal research_triage/fixtures/candidate_choice_proposal.json \
  --oracle research_triage/fixtures/candidate_choice_oracle.json
```

同一任务的 evidence-mode 对照可运行：

```bash
PYTHONPATH=. python -m research_triage.evidence_mode_pilot \
  --task research_triage/fixtures/mlagentbench_vectorization_task.json \
  --out /tmp/triage_fresh_run \
  --condition paper2ir_gate --request-budget 1
```

输出目录必须为空，防止覆盖原始模型输出。运行器输出逐条件 proposal、hash、validator metrics 和汇总；paired aggregate 由 `research_triage.aggregate` 单独计算。

预注册的 `text_rag` baseline 和单项消融也可用重复的 `--condition` 显式选择：
`text_rag` 只暴露检索到的原文片段，不暴露 claim/关系/条件/provenance 闭包；
`without_condition`、`without_provenance`、`without_source_gate`、
`without_dependency`。默认命令只运行五个基础 condition；消融不会因为省略
参数而意外 fan-out。消融的模型输入会被削弱，但 paired reference scoring
仍使用完整冻结任务，避免把评分规则一起删掉。

在任何 live call 之前，可以用 `prompt_audit.py` 离线核对表示是否保留同一证据
闭包、以及 prompt 长度差异：

```bash
PYTHONPATH=. python -m research_triage.prompt_audit \
  --task research_triage/fixtures/mlagentbench_vectorization_task.json \
  --condition paper2ir_gate --condition flat_compact
```

两个预注册协议也可以先由 plan-only runner 展开并检查；默认不会创建输出目录、
不会调用模型：

```bash
PYTHONPATH=. python -m research_triage.run_protocol \
  --manifest evidence/information_matched_compact_pilot/manifest.json \
  --cells evidence/information_matched_compact_pilot/cells.jsonl \
  --root . --output-root /tmp/triage-plan
```

只有在明确批准 live 请求后，才同时加上
`--execute --acknowledge-live-requests`。每个 cell 固定一个请求预算，且输出目录
必须为空；runner 会拒绝把新响应混入旧 artifact。

论文-derived task 也复用同一格式：

```bash
PYTHONPATH=. python -m research_triage.validate_proposal \
  --task research_triage/fixtures/paper_lottery_ticket_task.json \
  --proposal evidence/paper_lottery_ticket_reference_card/proposal.json
```

该任务冻结了 *Playing Lottery Tickets in Style Transfer Models* 的 OCR 行锚点和
2022-10-14 cutoff。当前 reference card 选择 `abstain`，因为输入闭包没有可执行的
AdaIN/SANet 实现和匹配数据；它不是模型生成结果，也不把论文的 proxy metric 当作
独立感知质量证据。

评分统一使用完整任务的离线 reference checker，不随输入消融而删除评分规则。
关系 ID 可不引用，不能因 flat-claims 看不到图结构就扣分；一旦引用，仍需验证其有效性。
不得在看过输出之后添加偏向某种表示的评分门槛。

当前 pilot 同时删除了关系、required conditions 或证据内容，因此不是信息等量的纯表示对照。
特别是 direct 被移除了所有 claim，而提示要求有证据才能行动，其弃权不能归因于缺少图。
`seed=0` 只是运行配对标签，API 未传采样 seed；temperature=0 不保证跨调用精确复现。

## 设计原则

1. 先固定 benchmark 和输入，再比较 evidence layer；
2. proposal 是 JSON artifact，方便复跑和 diff；
3. provenance/condition 错误直接判失败；
4. LLM judge 只做小样本盲评，不做唯一主指标；
5. 原始 proposal、验证结果和失败 trace 永久保留。

## 主实验口径

正式 benchmark 的第一主指标是同一 task、同一 seed 下 Paper2IR 相对 control 的
`TaskSuccess@1` paired delta；同时报告 `ExecutionPass@1` 和成本，避免把“程序跑通”
误当成“任务做对”。证据层单独报告 `CheckerEvidenceIntegrity` 和
`CheckerUnsupportedInference`，不把 checker 分数和 SAB 分数相加。比较矩阵固定模型、
prompt 和 seed，按 task×seed 做 paired bootstrap；fixture 和没有 evaluator log 的运行只
能填 `null`，不能当作失败或成功。

证据分诊 pilot 的 `CardValid@1` 是共享契约通过率；历史命名
`DecisionSupported@1` 仅指机械契约支持，不能代替科学有效性或实验价值。
无 action 规则、无合法卡或弃权时 `DecisionSupported@1` 为 `null`。
合法弃权的 `UnsupportedDecisionRate` 记为零，但它不是一个支持的实验决策；
分析时必须同时按 `audit.action` 报告弃权与非弃权分母，
不能靠增加弃权宣称决策变好。`EvidenceIntegrity` 目前也是整卡检查的粗粒度指标，
不是语义蕴含分数。单个 pair 不报告置信区间；没有执行日志时官方
`ExecutionPass@1`/`TaskSuccess@1` 保持 `null`。


`adapters/task_manifest.py` is the handoff point for official benchmark data.
Create one JSONL row per official task with an explicit container argv and a
sandbox. The adapter rejects duplicate IDs, empty commands and invalid time
limits; it never derives a command from natural-language task text.

`adapters/science_agent_bench.py` matches the current official evaluator shape:
`python -m evaluation.harness.run_evaluation --benchmark_path ...
--pred_program_path ... --log_fname ... --run_id ... --instance_ids ...`.
It only converts a locally available verified annotation export into explicit
argv rows. The password-protected benchmark archive stays outside this copy;
its files are never redistributed or committed.
