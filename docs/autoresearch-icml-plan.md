# Cross-Paper Evidence Graphs for Next-Experiment Decisions

版本：2026-10-02  
目标 venue：ICML（先完成可复现的研究结果，再决定投稿轮次）  
当前实验沙箱：[`paper2ir_experiment_copy`](../paper2ir_experiment_copy/)

## 已批准的新版本（v2，2026-10-02）

这一版要实现的功能只有一句话：**给 agent 一个有争议的研究问题、目标论文原文和一个可检索的跨论文证据图，让它自己发散提出下一步实验，再选出一张有证据、条件匹配、能推翻当前判断的实验卡。**

输入不再给“可选动作”，也不再给预先列好的候选实验。模型看到的是：

1. 任务问题、当前已知现象、实验预算和硬约束；
2. 带段落/source-anchor 标记的目标论文原文；
3. 一个可调用的检索接口。`paper_only` 只能在目标论文内检索；`cross_paper_graph` 在同一目标论文基础上，还能检索冻结的 LTH 全图（节点、边和来源锚点），而不是把一小段“答案图”直接塞进 prompt；
4. 证据截止时间。隐藏 rubric、gold bridge path 和候选答案只留给评分器。

输出也不再受 action enum 限制。模型先给若干个自己想到的方向，再选一个方向，输出：自由文本 `next_step`、引用的论文段落/图节点/边、适用条件、baseline/treatment/control、指标、入口和预算、预期观察、falsifier、改变决定的规则及不确定性。动作类别只在输出后由评分器归类，不能在输入中限制思考范围。

要证明“图结构有优势”，正式比较至少三组：目标论文原文、同一外部论文池的普通文本检索、同一论文池的图检索；三组使用相同模型、检索调用数、返回 token、输出预算和随机种子。两组比较只能证明“额外跨论文证据有用”，不能证明“图优于文本”。

当前代码里的 6 节点/4 条边、`design_only` 运行仍是旧版协议的 pilot，只能说明格式和引用链路能跑通，不能说明最终功能已经实现，也不能据此声称图让实验选择更好。v2 的实现门是：接入真实 retrieval API、改成全文输入、去掉 action enum、锁定 80+ LTH 图 snapshot，并加入 bridge-required、within-paper、no-bridge、conflict 四类任务和隐藏决策评分。

## 研究问题与可证伪主张

给定一个有争议的科学或机器学习结论、时间截点前可见的证据，以及证据之间的关系，agent 是否能在固定预算内提出并选择**更值得做的下一步实验**，并把决定限制在可回溯的 claim、source anchor 和适用条件之内？本项目的主测试单元不是单篇论文，而是一个 **bridge-required controversy bundle**：目标论文提出争议，决定实验所需的定义、反例、限制或可执行方法分布在其他论文中。如果只给目标论文，问题必须变成信息不足或允许暂缓判断；只有跨论文检索和关系图才能恢复下一步。

主假设分成三层，避免把“格式更规整”误写成“科学决策更好”：

1. **检索假设**：在目标论文之外，LKM 的 typed relation、condition 和 provenance 扩展能比等 token 预算的文本检索找回更多必要的外部 claim 和 bridge path。
2. **审计假设**：带 relation、condition 和 provenance gate 的跨论文图，相比等信息的 flat ledger，能提高 `CardValid@1`、`EvidenceIntegrity`，并降低 `UnsupportedDecisionRate`。
3. **开放式决策假设**：图输入能让 agent 生成更多覆盖不同干预家族的可执行想法，并在其中选出具有更高信息价值/预算比的下一步；模型不接收预先枚举的候选实验集合。
4. **执行假设**：在存在合法 executable entrypoint 的任务上，决策质量的提升能传导到 `ExecutionPass@1`、官方 `TaskSuccess@1` 或任务原生得分；若不能传导，论文只报告过程可靠性和边界，而不宣称自动科研。

真正的论文主张必须同时满足：hidden rubric 有足够的专家一致性、官方 evaluator 可运行、paired CI 不跨 0，且 condition/provenance/source-gate 消融能解释增益来源。没有这些条件时，结果只称为 pilot 或 safety/abstention result。

## 输入、输出和信息边界

模型侧输入和评分器侧输入严格分开。模型侧协议固定为：

```json
{
  "task": {
    "task_id": "...",
    "question": "只描述待解释的现象和研究目标",
    "current_state": "已知结果和不确定性",
    "hard_constraints": ["dataset/model/training/evaluation constraints"],
    "budget": {"max_seconds": 600, "max_gpu_hours": 0},
    "evidence_cutoff": "YYYY-MM-DD"
  },
  "focal_paper": {
    "paper_id": "...",
    "title": "...",
    "full_text": "带 [anchor:a_001] 标记的冻结原文"
  },
  "retrieval": {
    "corpus_snapshot": "sha256:...",
    "tools": [
      "search_claims(query, k)",
      "expand_neighbors(node_or_edge_id, k)",
      "get_source_anchor(anchor_id)"
    ],
    "max_calls": 8,
    "max_return_tokens": 4000
  }
}
```

`paper_only` 的检索工具只返回目标论文原文的段落；`external_text` 返回同一外部论文池的普通文本片段；`cross_paper_graph` 返回同一论文池中的节点、typed edge、条件和 source anchor。三者都必须看到同一份目标论文原文。模型侧 ID 使用随机 opaque ID，不能从 `p01_claim_5` 之类的名字猜到答案；完整映射只留给评分器。每次查询、返回结果、选中结果、调用次数和 token 都写入 retrieval trace。

评分器可以看到 hidden rubric、gold bridge path、专家判断和执行结果，但模型看不到 `allowed_claims`、候选实验列表或 gold action。这样评分器可以判断一个任意生成的方案是否有用，而不会把答案范围提前塞进输入。

agent 先给 3–5 个自己提出的、彼此不同的短方向，再选择一个输出 JSON experiment card；不保存隐藏思维过程。`next_step` 是自由文本，事后才由评分器归类为复现、消融、对比、独立证据、暂缓或其他：

```json
{
  "task_id": "...",
  "question": "...",
  "hypothesis": "...",
  "reason": "一句可审计理由",
  "generated_ideas": [{"idea_id": "i1", "next_step": "自由文本方向", "one_line": "..."}],
  "selected_idea_id": "i1",
  "next_step": "自由文本的下一步行动",
  "evidence": [{"anchor_id": "...", "quote": "短引文", "source": "paper|graph", "role": "support|limitation|context", "relation_id": "..."}],
  "conditions": [{"key": "...", "value": "..."}],
  "experiment": {
    "dataset_refs": ["..."],
    "baseline": "...",
    "treatment": "...",
    "controls": ["..."],
    "metrics": ["..."],
    "entrypoint": "relative/path.py",
    "expected_observation": "...",
    "falsifier": "...",
    "budget": {"max_seconds": 600, "max_gpu_hours": 0}
  },
  "decision_rule": "...",
  "uncertainty": "..."
}
```

`retrieval_trace` 作为同级字段保存查询和选择记录。评分器对任意生成的 proposal 计算：跨论文证据使用、条件闭包、可执行性、区分假设的能力、预算成本、与 hidden acceptable intervention family 的匹配、结果信息增益和 hindsight regret。主任务不提供 `candidate_experiments`；候选列表只可用于单独的 calibration，不进入主结论。

## 评测对象与版本锁定

第一阶段主 benchmark 使用工作区的最终图 [`merge.html`](../merge.html)，后端读取其锁定的 JSON snapshot，而不是执行 HTML。snapshot 当前为 240 个节点、235 条边、103 个无向连通分量，其中 102 个节点孤立；节点没有 `source_anchor_ids`，payload 也没有可验证的 `paper_id`，anchor coverage 为 0/240。因此它目前只能作为结构候选底座，不能直接作为带 provenance 的 gold evidence graph。6–10 个 LTH bridge task 的 target/external/bridge/condition/cutoff 仍需回到原文另行审计，不能把图中的 `K` 总结节点或未锚定边直接当作答案标签。

第二阶段外部执行 benchmark 采用 ScienceAgentBench verified split。其论文明确给出 102 个任务、来自 44 篇同行评审论文、覆盖四个学科，并把目标统一为可执行 Python 程序；官方摘要报告三次尝试下最佳 agent 独立完成率为 32.4%，加入专家知识为 34.3%。见 [ICLR 2025 官方摘要页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f12b4df26344f3be803c06b555252efe-Abstract-Conference.html)、[官方代码](https://github.com/OSU-NLP-Group/ScienceAgentBench) 和 [verified split](https://huggingface.co/datasets/osunlp/ScienceAgentBench)。它可验证选定实验的程序结果与成本，但不是跨论文方向的金标。

执行顺序是 12–20 个分层任务 pilot → 30 个任务主实验 → 102 个任务扩展。2026-04-30 起官方提供 `verified` split；实验必须从 [Hugging Face verified split](https://huggingface.co/datasets/osunlp/ScienceAgentBench) 和官方 harness 加载，锁定数据集 revision、代码 commit、容器 digest、数据许可和 evaluator 版本。当前工作树仍未下载该 verified 运行包；在下载、校验并复现官方日志前，所有 SAB 数字只能称为 schema/执行链 sanity check，不能称为官方 `TaskSuccess`。

第三阶段采用 MLAgentBench 做 ML-only 外部 sanity check。其 ICML 2024 论文是 13 个任务、允许 agent 读写文件、执行代码和查看输出，Claude 3 Opus 报告平均成功率 37.5%；见 [PMLR 官方页](https://proceedings.mlr.press/v235/huang24y.html)。它的 task/evaluator 可以判断“生成的改动是否让程序指标改善”，不能单独判断开放式研究方向是否值得做。当前先锁定 vectorization task 和 commit，再扩展其余任务。

## 检索协议：让 LKM 成为必要条件

目标论文原文是共同输入；外部论文不直接全量塞进 prompt，而是通过有界、可记录的检索接口进入上下文。流程固定为：

1. 用标题、摘要、claim embedding 和关键词从全部 LTH 论文得到 lexical/embedding top-k 初始池；
2. 以 target claim 为锚点，在 LKM/merge 图上沿 `support`、`limitation`、`challenge`、`bridge`、`condition` 等 typed edge 做 1-hop/2-hop 扩展；
3. 对每条路径保留 paper id、claim id、source anchor、relation type、条件和 cutoff，删除无 anchor、越过 cutoff 或条件冲突的节点；
4. 将节点/边/anchor 路径压缩为固定 token 预算的 evidence bundle，再交给 agent 生成开放式 ideas；paper-only 不调用外部接口。

LKM 原图目前没有独立的 observation/data 节点，也没有明确的 `support`/`infer`/`induction` 语义；因此检索层使用 `clean_claims → single_paper formalization → merge` 补出的 observation、条件和 source anchor，而不是把原生关系名直接当作金标。跨论文路径在首轮定义为 `paper_i → integration hypothesis K → paper_j`，须经过 Step3/4 formalization 和 anchor audit 才能进入 hidden gold。

按同步的 LKM 接入说明，检索层可以对应四个只读接口：`POST /search`（keyword/vector/hybrid）、`GET /claims/{id}/evidence`（证据链）、`POST /variables/batch`（批量变量）和 `POST /papers/graph`（论文图）。对 agent 暴露的最小工具是 `search_claims(query,k)`、`expand_neighbors(id,k)`、`get_source_anchor(id)`；实验必须记录 query、scope、返回排序、被选节点、扩展 hop、source class、调用次数、token 和 graph snapshot hash。LKM score 只作排序信号，不能当作概率、Gaia prior 或 gold utility；`official > heuristic > provisional` 的 source trust 只用于 gate 和分层分析。

LKM 的 claim-driven / contradiction-driven explorer 应分别作为两种检索策略：claim-driven 从 target claim 向 support/condition 扩展，contradiction-driven 从 limitation/challenge/previous-conclusion 扩展。两者都要经过同一 cutoff、source-anchor 和条件 gate，再交给 Gaia compile/check/merge；不要让检索过程直接改写全局图。

首轮主比较保持同一论文池、同一检索调用/返回 token 预算和同一模型：

| 条件 | agent 可见内容 | 用途 |
|---|---|---|
| `S0_paper_only` | 目标论文完整原文；只允许论文内段落检索 | 单论文基线 |
| `S1_external_text` | 同一目标论文 + 同一外部论文池的普通文本检索 | 控制“多了外部信息” |
| `S2_cross_paper_graph` | 同一目标论文 + 同一外部论文池的节点/边/anchor 检索 | 测图结构增益 |
| `S3_graph_gate` | S2 + 条件、来源和依赖 gate | 主系统，第二阶段加入 |

另设 `decoy_connected`：保持节点数、边数和 token 数相同，但把桥接边随机重连，防止模型只利用“图很密”而不是语义路径。检索指标是 `Recall@K`、bridge-edge recall、condition match、independent-evidence rate、contradiction recall、token/latency；主决策指标只在 `cross_paper_dependency=true` 的子集上比较 S2/S3 相对 S0/S1。若首轮只跑 S0 与 S2，结论只能写“外部跨论文证据有用”，不能写“图结构优于文本检索”。

### LKM 在整条实验链里的位置

LKM 不只是一个“把更多论文放进 prompt”的上下文源，而是四个可审计模块：

1. **候选召回**：对 LTH 论文标题、摘要和 clean claim 做 BM25、dense、hybrid/rerank，得到同一候选池；
2. **关系扩展**：从 target claim 沿 typed edge 做 1-hop/2-hop 扩展，优先保留 support、challenge、limitation、condition 和跨论文桥；
3. **证据编译**：把每条路径编译成 `paper_id → claim_id → relation_id → source_anchor → condition`，再按 token 预算压缩；
4. **检索审计**：保存 query、初始命中、扩展节点、被剪枝原因和最终 bundle，供 `BridgePathRecall@K`、`ExternalAnchorPrecision@K`、`ConditionPreservation` 和反事实删边测试使用。

因此实验分两步，避免把“召回器变好了”和“图表示变好了”混在一起：第一步固定 flat renderer，比较 BM25、dense、hybrid/rerank、LKM graph 的检索质量；第二步固定相同的 evidence pool，比较 flat claims、typed graph、graph+gate 的决策质量。LKM 只有在这两步都能复现增益，才可以被写成系统贡献。

## 对照、消融与公平性

所有条件使用同一模型、同一 system/user 模板、temperature 0、同一任务与 seed、同一最大输出 token、同一检索调用/返回 token 和同一时间预算；只替换模型能看到的外部证据表示。主任务不提供 `candidate_experiments`；closed-world candidate evaluator 只用于 calibration，不进入主结论。

| 条件 | 输入 | 目的 |
|---|---|---|
| `S0_paper_only` | 目标论文完整原文 | 单论文基线 |
| `S1_external_text` | 目标论文 + 外部文本检索 | 文本 RAG 基线 |
| `S2_cross_paper_graph` | 目标论文 + 外部节点/边/anchor 检索 | 图结构主对照 |
| `S3_graph_gate` | S2 + condition/source/dependency gate | 主系统 |
| `decoy_connected` | 节点/边/token 匹配但桥接边随机 | 检验是否真的用关系语义 |
| `flat_claims` | claim + anchor，无 relation/typed gate | 测 claim 列表收益 |
| `flat_compact` | 信息等量的 flat ledger | 主表示对照，控制 prompt 长度 |
| `paper2ir` | claim graph + provenance + dependency | 完整图表示 |
| `paper2ir_gate` | 完整图 + condition/source/action gate | 主系统 |
| `without_condition` | 去掉适用条件 | 条件闭包因果消融 |
| `without_provenance` | 去掉 source anchor | provenance 因果消融 |
| `without_source_gate` | 放宽 source role | 来源污染消融 |
| `without_dependency` | 去掉 typed relation | 依赖关系消融 |

每个 task×condition 至少 3 个 seed；主结果按 task×seed 配对 bootstrap 95% CI，并同时报告 domain macro average、任务难度分层、失败类型和实际 token/cost。`flat_compact` 的 prompt 字符数在 live pilot 中仍有轻微差异，正式实验必须将字符/token gap 记录为预注册变量，必要时做长度匹配复跑。

## LTH 起步任务与跨论文桥

先以 [`merge.html`](../merge.html) 作为唯一最终图；抽取后的锁定快照保存在 `paper2ir_experiment_copy/evidence/lth_final_graph_20261002/`。它的 240 个节点、235 条边是 viewer 的 merge/finalization 结果，但 provenance 当前缺失，必须在任务层补回原文 anchor 后才能计入 `EvidenceIntegrity`。首批 6 个任务覆盖不同类型的跨论文依赖：

| 争议 | 必须桥接的论文/图节点 | 下一步实验的最小干预 |
|---|---|---|
| 稳定性与 rewind | P01 vs P53/P55/P62，K03 | batch size × rewind-k × iterative/one-shot，并做 isolation retraining |
| clean accuracy 与 robustness | P47/P63 vs P03/P23，K09 | 同一 mask/训练法同时测 clean、OOD、adversarial |
| ticket transfer | P36/P61/P37/P23，K05 | source dataset/目标域/optimizer 的交叉迁移 |
| mask 还是 initialization | P09/P69/P23，K02 | 固定 mask 比较 original init、random init、rewind |
| 结构与真实硬件加速 | P13/P69/P71，K06 | unstructured、block、layer/token 在同一 GPU kernel 测 latency |
| IMP 的发现成本 | P56/P62/P66，K07 | IMP、ETT、one-shot、early-bird 在同目标 sparsity/预算下比较 |

每题 hidden rubric 要求至少引用两篇论文，包含一条支持和一条限制/挑战路径，保留条件，写出 controls、falsifier、预算，并说明结果如何区分 competing hypotheses。另加三类反事实检查：删除 gold bridge edge 后方案应改变或 abstain；加入条件不兼容的 decoy paper 后不应产生 false bridge；1-hop 不依赖外部论文的题作为 negative control。核心指标为 `BridgeNecessityScore = StrictSuccess(S2) - StrictSuccess(S0)`，只在 `cross_paper_dependency=true` 题上计算。

## 文献驱动的对照修改

- **检索质量单独评估**：LitSearch（EMNLP 2024）显示 BM25、dense retriever 和 LLM reranking 在科学文献检索上差异很大，因此先做 BM25/dense/hybrid/LKM 的 `Recall@K` leaderboard，再做图表示对照；不能把检索质量差异误写成图推理收益。见 [LitSearch](https://aclanthology.org/2024.emnlp-main.840/)。
- **证据立场与反例**：SciFact（EMNLP 2020）把 scientific claim verification 分成 SUPPORTS/REFUTES/NOINFO，并要求 rationale；SciFact-Open 进一步显示从小语料到开放语料会出现明显性能下降。我们的 hidden rubric 要求至少一条 support 和一条 limitation/challenge，并新增 `StanceCoverage`、`ContradictionRecall` 和 anchor-level rationale 检查。见 [SciFact](https://aclanthology.org/2020.emnlp-main.609/)、[SciFact-Open](https://aclanthology.org/2022.findings-emnlp.347/)。
- **图检索不是“连通就算”**：G-Retriever（NeurIPS 2024）把 textual-graph RAG 的子图选择建模为带预算的 Steiner Tree，并同时报告图 QA 和 hallucination；因此加入固定 token 的 subgraph budget、decoy-connected 和删边反事实。见 [G-Retriever](https://proceedings.neurips.cc/paper_files/paper/2024/hash/efaf1c9726648c8ba363a5c927440529-Abstract.html)。
- **开放式 idea 不能用单一 exact match**：DiscoveryBench（ICLR 2025）使用 context/variables/relationships 的 facet 评测；BLADE（Findings EMNLP 2024）用专家分析的多表示匹配；ICLR 2025 的 100+ 研究者人评显示 idea novelty 和 feasibility 需要盲评，而且模型存在 self-evaluation 失真和生成同质化。由此主任务输出 K=3–5 个 ideas，报告 diversity、coverage、novelty/feasibility 盲评和选择后验，而不是只报一个候选命中率。见 [DiscoveryBench](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0d70af566e69f1dfb687791ecf955e28-Abstract-Conference.html)、[BLADE](https://aclanthology.org/2024.findings-emnlp.815/)、[ICLR 2025 idea-generation study](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ea94957d81b1c1caf87ef5319fa6b467-Abstract-Conference.html)。
- **设计、实现、执行分开报**：ScienceAgentBench、MLAgentBench、EXP-Bench（ICLR 2026）都说明完整可执行实验远难于单独的设计或代码步骤；MLR-Bench（NeurIPS 2025）进一步将 idea generation、proposal、experiment、paper writing 分阶段评估。因此我们的主表分开 `EvidenceIntegrity`、`DesignQuality`、`ExecutionPass`、`TaskSuccess` 和成本，不能用一个端到端分数掩盖失败阶段。见 [EXP-Bench](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c411f5b2d9c55f1685e72db224ad8b0e-Abstract-Conference.html)、[MLR-Bench](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab8dd000d6f87f40061a73f8bca7fae4-Abstract-Datasets_and_Benchmarks_Track.html)。
- **“最值得做”要有信息价值基线**：Sequential Information Maximization（COLT 2015）、Robust Bayesian Active Learning（AISTATS 2026）和 cost-aware active experiment selection（CoRL 2025）都把信息增益、代表性、模型错配和成本纳入实验选择。我们的 hidden rubric 增加 `DiscriminativePower`、`UtilityPerCost`，并和 random、uncertainty、cost-only、information-gain oracle 做 calibration 对照。见 [Sequential Information Maximization](https://proceedings.mlr.press/v40/Chen15b.html)、[R-IDeA](https://proceedings.mlr.press/v300/tang26d.html)、[CoRL active experiment selection](https://proceedings.mlr.press/v305/anwar25a.html)。

ScienceAgentBench 和 MLAgentBench 提供可执行任务与结果评估，但不是跨论文检索或“最值得做”方向的金标。EviGraph 已把 typed evidence graph 用于自主科研流程，因此本文贡献应收窄为 **cross-paper、condition-aware、provenance/independence 校准的 bridge-required next-step planning，并在 LTH 上做可执行评测**，不声称“首个 evidence-graph agent”。

## 指标与判定

硬指标分为三组，不相加：

- **卡片与证据**：`CardValid@1`、`EvidenceIntegrity`、`DecisionSupported@1`、`UnsupportedDecisionRate`、`RelationCitation@1`、cutoff/condition/provenance closure；
- **开放式选择价值**：`IdeaCount`、idea diversity/coverage、`BridgeUse@1`、`DecisionQuality@1`（hidden rubric + 可执行结果）、information gain、hindsight regret、cost regret、abstention coverage/selective risk。closed-world 的 `DecisionChoiceValid@1`/Top-1 A-tier 只作为 calibration 指标；
- **执行结果**：`ExecutionPass@1`、官方 `TaskSuccess@1`、任务原生分数、CodeBERTScore/API cost（按 benchmark 原生定义）。

对“下一步实验是否值得做”增加一个隐藏的决策 rubric：实验必须能区分至少两个 competing hypotheses，且给出预期观察、falsifier、controls、条件和成本。可执行任务报告真实结果；没有 runnable entrypoint 的 LTH 任务只报告 rubric 分数、`BridgePathRecall`、`ConditionPreservation` 和 abstention，不能填 `ExecutionPass`。成本归一化效用定义为 `UtilityPerCost = DiscriminativePower × Executability / (1 + max_seconds + λ·max_gpu_hours)`，同时报告与 hidden acceptable intervention family 的 hindsight regret。

无合法入口或证据不足时，`abstain` 是合法动作。报告 abstention rate、abstention precision 和非弃权分母，不能通过多弃权制造“unsupported 下降”。20% 分层任务做双人盲审，检查实验是否区分假设、控制变量、falsifier 和条件一致性，报告 Cohen κ 或 Krippendorff α；LLM judge 只能做辅助分析。

## 执行架构

复用现有链，不在 Gaia IR 内核中重写运行时：仓库正式维护的证据工作流是 `pipelines/single_paper` 和 `pipelines/merge`，`clean_claims` 是预处理；隔离副本的 `research_triage` 是评测编排 harness，不是第三条论文抽取 pipeline。single-paper 产出 claim/observation/anchor，merge 产出跨论文 typed edge/conclusion，新增只在 cross-paper retrieval/bridge task 和开放式 decision layer。

六阶段运行器为：

```text
freeze_inputs → propose → validate → execute_official → score → report
```

`RunStore` 冻结输入、配置和 SHA-256；`validate` 只做确定性 schema/closure/gate；`execute_official` 只接受显式 argv、sandbox 和 env allowlist，禁止 shell 拼接；`score` 将官方 evaluator 结果与 checker 指标分开；`report` 保留 proposal、metadata、stdout/stderr、失败类型、成本和 artifact hash。任何模型输出都不能直接在宿主机执行。

## 已落实的代码与证据

截至 2026-10-01，隔离副本已完成：

- 实验卡 schema、时间截点、source/condition/entrypoint gate、六阶段 pipeline、ScienceAgentBench/MLAgentBench adapter、paired aggregate/bootstrap；
- 新增 `experiment.candidate_id`、独立的 `candidate_evaluator.py` 和 evaluator-only candidate oracle 接口，并把 `DecisionChoiceValid@1`/`DecisionQuality@1` 接入 score artifact；没有 oracle 的旧任务指标保持 `null`；
- 离线回归：`127 passed`；剩余 1,504 条 NumPy 弃用警告来自外部 MLAgentBench vectorization 代码；
- plan-only 协议：12-cell compact-flat 和 6-cell paper-derived abstention 协议均能展开，不创建输出目录、不发起模型请求；
- 已有 live pilot 只作描述性记录：2 tasks × 2 conditions × 3 seeds 中，`paper2ir_gate` 为 6/6 合法卡，compact-flat 为 4/6；paired CardValid 差值为 -1/3，bootstrap 区间 [-2/3, 0]，不能据此宣称普遍图优势；
- paper-derived task 中两种表示均 3/3 合法 abstain，说明入口/截点安全门生效，不证明论文结论。

## 下一步执行门

1. **本地门（已满足）**：全量离线测试、manifest/task hash、plan-only 展开、模型 prompt 不包含 oracle。
2. **LTH bridge audit（下一步）**：从现有 5–10 篇可验证 LTH 图开始，标注 6 个 bridge-required task；每题写 target/external/bridge relation/condition/cutoff/decoy，并确认 target-only 只能得到不确定或 abstain。
3. **开放式 pilot**：10 个 task × 3 seeds 跑 S0–S3；agent 生成 K=3–5 个 arbitrary ideas，评分器做 bridge-use、hidden rubric、信息增益、预算和执行后验；不把候选列表放入 prompt。
4. **真实执行门**：先锁定 MLAgentBench vectorization 子任务和 OpenLTH/LTH entrypoint，再扩到 ScienceAgentBench verified split；官方 evaluator 只验证选定代码的运行结果和成本，不能单独判定研究方向。
5. **主实验门**：pilot 中 S2/S3 相对 S0/S1 在 bridge-required 子集的 `BridgeNecessityScore`、`DecisionQuality@1` 或官方任务分数至少一项有正向 paired CI，且 `UnsupportedDecisionRate` 不升高；否则转为“图改善可审计性/安全弃权，但未证明实验选择更优”的边界论文。
6. **写作门**：只有在 LTH bridge 子集与外部 MLAgentBench/ScienceAgentBench 执行后验方向一致时，才写“减少无依据实验决策”；Related Work 中明确不声称首个 evidence-graph agent。

## 研究日志与修改记录

每次实验追加到 `paper2ir_experiment_copy/research_triage/CHANGELOG.md`，并保存：代码 commit、task/condition/seed、prompt/model/response hash、request/cost、validator result、官方 evaluator log、失败分类和解释。禁止覆盖已有 evidence 目录；新的 live run 必须使用空输出目录和显式 `--execute --acknowledge-live-requests`。
