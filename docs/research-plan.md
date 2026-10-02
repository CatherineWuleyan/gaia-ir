# Gaia-IR 研究计划与文献调研报告

> 版本：2026-10-02 · 状态：v2 协议已批准，进入落地阶段
> 依据：本仓库 `pipeline_merge_v9` 封存结果、`docs/` 契约层、以及四条文献线的调研。
> 目的：回答三个问题 —— **创新点在哪 / 下一步具体怎么做 / 能从已有文献复用哪些东西**。

> **当前执行口径**：本报告中的 AutoResearch/LTH 实验以 [`docs/autoresearch-icml-plan.md`](./autoresearch-icml-plan.md) 的 v2 为准。模型输入是目标论文全文 + 受控检索接口，不含 allowed action 或候选实验；正式图优势比较使用目标论文、外部文本检索、跨论文图检索三组。LTH pilot 的逐项协议见 [`docs/lth-graph-vs-paper-pilot.md`](./lth-graph-vs-paper-pilot.md)。

---

## 0. 一页结论

**一句话定位**

> Gaia-IR 不是"再做一个科学知识图谱"，而是把一个**条件感知（condition-aware）的跨论文 claim 推理图**做成**带执行语义的中间表示**，并用**过程级指标 + 真实 AutoResearch 效用**证明它有用。

**创新点落在三层，缺一不可**

| 层 | 创新内容 | 直接对标物 | 差异化 |
|---|---|---|---|
| 表示层 | claim 级跨论文图；关系不是标签而是**有 canonical 微观骨架的推理策略**（9 种）；condition 与 provenance 是一等公民 | CodRED 谱系的关系类别体系；SciNets 的 query-local concept graph；citation GraphRAG | 关系**可展开为确定性算子网络**，因此可编译、可校验、可边际化 |
| 执行层 | 图作为**可执行 IR**；checkpoint / replay / artifact hash / 失败隔离 / schema 与 compiler 门禁 | HypER 的非持久化 reasoning chain；AI-Researcher 的 agent 流水线 | 证明的不是"答得对"，而是**可复现、可审计、可恢复** |
| 评测层 | StrictSuccess + 无依据推断率 + 条件违反率 + trace 有效性 + 恢复率/hash 一致性；并定义 **next-step research planning** 任务 | ScienceAgentBench / PaperBench 的可执行评测；SCHEMA 的拓扑感知幻觉评估 | 把"科研推理质量"和"运行可靠性"**分开计量**，而不是揉成一个分数 |

**最重要的判断**

> 现有工作里，**没有一篇同时具备** claim 级跨论文图 + 类型化推理关系 + 条件保持 + provenance 闭包 + 图作为可执行中间表示 + checkpoint/replay + 在真实 AutoResearch 任务上的效用证明。这是 Gaia-IR 的窗口，但它是"首批（among the first）"而不是"首次（the first）"，措辞必须谨慎。

**核验后的两项重要修正（见附录 A）**

> 1. **novelty 的对照物要收紧。** 最初把 SciNets / SciR / SCHEMA 与 HypER 并列为"最接近的四篇"，但前三篇都是 **2026 年未评审预印本**。能撑住 novelty 边界的**同行评审**工作实际只有 **HypER（EMNLP 2025）**。因此 novelty 应主要对比 **HypER + AI-Researcher + CodRED 谱系**，2026 预印本只放在"同期动态"。
> 2. **"刘知远组路线"这个框定不成立。** ReactXT 的作者单位是 **NUS/USTC/Hokkaido**（含同名 "Zhiyuan Liu"，但不是清华那位），PRESTO 来自 **IDEA**。只有 EDRM / Denoising DS-QA / SciKGAT / Toolink 一组归属成立。正文已改为"知识增强检索与科学事实核验这一脉工作"。

**下一步的最高优先级**

> **先把 E0 做成一个严格的受控 benchmark，再碰 103 篇论文和真实 AutoResearch。**
> 如果 E0 不能证明"图"和"harness"在受控任务上各自带来增益，后面所有结果都无法归因。

**当前最大的具体技术缺口（本次调研新发现）**

> `docs/pipelines/pipeline-8-merge.md` Step 2 明确规定："直接命中集合确定后，只围绕这些 Knowledge 做**一跳扩展**……扩展得到的新节点**不再递归扩展**。"
> 而 MR.COD / PILOT 这一整条线的核心贡献恰恰是**多跳证据路径检索与排序**。Gaia-IR 目前在 Step 2 主动放弃了多跳。这既是风险（跨论文多跳关系召回不足），也是**最容易拿到确定性收益的改进点**。

---

## 1. 文献调研结果

### 1.1 线一：跨文档关系抽取（CodRED 谱系）

> ⚠️ **引用规范**：下表中第二列加粗项（MR.COD、PILOT、REIC 等）大多是**系统/方法名，不是论文标题**。只有最后一列是可直接进参考文献的完整标题。经逐条核验，本表最初版本中的多条"标题"实为描述性转写，已按真实标题更正。

| 年份 | 系统/方法名 | **真实论文标题** | 核心动作 | ID |
|---|---|---|---|---|
| 2021 | CodRED | *CodRED: A Cross-Document Relation Extraction Dataset for Acquiring Knowledge in the Wild* | 建立跨文档 RE 数据集与任务 | `2021.emnlp-main.366`（EMNLP 2021 main） |
| 2022 | （无系统名） | *Entity-centered Cross-document Relation Extraction* | 实体中心文档上下文过滤 + cross-path entity relation attention，路径间交互而非独立处理 | `2022.emnlp-main.671`（EMNLP 2022 main） |
| 2023 | MR.COD | *Multi-hop Evidence Retrieval for Cross-document Relation Extraction* | 多跳证据检索 + contextual dense retriever 对**证据**排序 | `2023.findings-acl.657`（Findings of ACL 2023；arXiv:2212.10786） |
| 2023 | PILOT | *Explore the Way: Exploring Reasoning Path by Bridging Entities for Effective Cross-Document Relation Extraction* | 把桥接实体作为 stepstone 探索推理路径 | `2023.findings-emnlp.450`（Findings of EMNLP 2023） |
| 2023 | LGCR | *Local-to-Global Causal Reasoning for Cross-Document Relation Extraction* | 局部因果估计 + 因果引导的全局推理，抑制语义相似带来的噪声 | DOI `10.1109/JAS.2023.123540`（IEEE/CAA JAS 10(7):1608–1621） |
| 2024 | REIC | *Reward-based Input Construction for Cross-document Relation Extraction* | 以 RE 分数为 reward 的**强化学习句子选择器**（该方向首个 learning-based selector） | `2024.acl-long.501`（ACL 2024 long） |
| 2024 | KXDocRE | *Knowledge-Driven Cross-Document Relation Extraction* | 注入实体领域知识，并输出解释性证据文本 | `2024.findings-acl.227`（Findings of ACL 2024） |
| 2024 | CoRE-NEPD | *Towards Better Graph-based Cross-document Relation Extraction via Non-bridge Entity Enhancement and Prediction Debiasing* | 非桥接实体增强 + GNN 统一实体图 + 预测去偏 | `2024.findings-acl.38`（Findings of ACL 2024；arXiv:2406.16529） |
| 2025 | CLEAR | *CLEAR: Cross-Document Link-Enhanced Attention for Relation Extraction with Relation-Aware Context Filtering* | relation-aware context filter + 路径距离加权 attention + 跨路径实体矩阵 | DOI `10.3390/app15137435`（Applied Sciences 15(13):7435） |
| 2026 | HCRE | *HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction with a Prediction-then-Verification Strategy* | 关系类别组织成**层级树**，LLM 逐层分类 + prediction-then-verification | `2026.findings-acl.1985`（Findings of ACL 2026） |

**核验结论（重要）**

- 上表 **10 篇全部真实存在**，ID 与 venue 均可访问；最初版本中 **9 条的"标题"是描述性转写而非真实标题**（缩写被误当作标题），已全部更正。
- **"ECRIM" 这个缩写在原文中并不存在**——该文没有系统名。此前把它当作系统名引用是不成立的。
- MR.COD 的准确展开是 **"Multi-hop evidence retrieval for Cross-document RE"**，不是 "Multi-hop Reasoning over ... passages"。
- CoRE-NEPD 声称"超过 GPT-3.5 和 InstructUIE"需回到原文核对，本报告不将其作为论据。
- 本线全部为 2023–2026 年成果，早期条目（CodRED）比最初描述晚一年定位：**CodRED 是 EMNLP 2021 main，不是 ACL 2021**。

**演进主线**

```text
读所有文本（CodRED 2021）
  → 围绕实体筛选文本（Entity-centered Cross-doc RE 2022）
    → 检索多跳证据路径（MR.COD / PILOT 2023）
      → 压缩长文档 + 注入外部知识（REIC / KXDocRE / CoRE-NEPD 2024）
        → 关系感知过滤（CLEAR 2025）
          → LLM + 层级决策 + 预测后验证（HCRE 2026）
```

**对 Gaia-IR 的三条直接启示**

1. **检索必须是一等公民。** MR.COD 之后，"先检索多跳证据再排序"被证明优于"用已有文档直接推理"。Gaia-IR 的 Step 2 目前只有一跳扩展，等于停在了 2022 年 Entity-centered 工作之前的水平。
2. **"没有关系"是一个需要校准的决策。** CoRE-NEPD 专门处理 NA 样本造成的预测偏置。Gaia-IR 的 Step 3 大量走"证据不足则不写入"路径，但**从不报告这条路径的准确率**——这是一个现成的、廉价的诊断指标。
3. **关系类别多的时候，flat 分类会失败。** HCRE 的层级分类 + prediction-then-verification 直接对应 Gaia-IR 的"先判确定性 Operator / 再判 weakpoint 类型 / 最后机械封口"两阶段设计——**这个设计在文献上是有支撑的**，可以作为方法论引用。

**注意边界**：这条线全部在 **Wikipedia + CodRED** 上验证，**不是学术论文场景**。Gaia-IR 若声称"论文版 CrossDocRE"，需要自己造 benchmark，不能直接迁移结论。

---

### 1.2 线二：知识增强的科学推理模块

> ⚠️ **归属更正（核验后）**：本节最初被表述为"刘知远组路线"。逐条核验后发现**这个归属不能整体成立**：
> - **EDRM**（*Entity-Duet Neural Ranking*，ACL 2018，Liu, Xiong, Sun, Liu）、**Denoising DS Open-Domain QA**（ACL 2018，Lin, Ji, Liu, Sun）、**SciKGAT**（Findings of EMNLP 2020，Liu, Xiong, Dai, Sun, Sun, Liu）、**Toolink**（NAACL 2024，Qian, Xiong, Liu, Liu）—— 这一组作者构成与清华系一致，归属**成立**。
> - **ReactXT**（Findings of ACL 2024）的作者含 "Zhiyuan Liu"，但其单位是 **NUS**（合作者来自 NUS / USTC / Hokkaido），与清华刘知远**不是同一归属**；同名 ≠ 同人，**不要当作该组工作引用**。
> - **PRESTO**（Findings of EMNLP 2024）作者为 Cao, Shao, Liu, Liu, Tang, Yao, Li，单位为 **IDEA**，同样**不属于**该组。
>
> 结论：可以写"知识增强检索与科学事实核验这一脉工作"，**不要写"刘知远组的 AutoResearch 路线"**。

| 工作 | 真实标题 | 做了什么 | 对应 AutoResearch 的哪个模块 |
|---|---|---|---|
| **EDRM** | *Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval*（ACL 2018，`P18-1223`） | 把 query 和 document 中的实体映射到知识图谱表示，与文本表示联合排序 | **检索**：论文/方法/数据集检索、找桥接概念 |
| **Denoising DS-QA** | *Denoising Distantly Supervised Open-Domain Question Answering*（ACL 2018，`P18-1161`） | paragraph selector 过滤噪声 → reader 回答 | **证据筛选**："检索—筛选—阅读—回答"骨架 |
| **SciKGAT** | *Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling*（Findings of EMNLP 2020，`2020.findings-emnlp.216`） | 域内语言建模适配开放域事实抽取与核验；**SciKGAT 是该文提出的系统名，不是标题** | **证据核验**：hypothesis sanity check、结论支持/反驳 |
| **ReactXT** | *ReactXT: Understanding Molecular "Reaction-ship" via Reaction-Contextualized Molecule-Text Pretraining*（Findings of ACL 2024，`2024.findings-acl.318`；arXiv:2405.14225） | reaction-text modeling；OpenExp 数据集（约 27.4 万条反应—实验步骤对）；实验步骤预测、分子描述生成、逆合成 | **实验方案生成**：把自然语言实验描述转成结构化动作序列 |
| **PRESTO** | *PRESTO: Progressive Pretraining Enhances Synthetic Chemistry Outcomes*（Findings of EMNLP 2024，`2024.findings-emnlp.597`） | 渐进式跨模态预训练，分子图—文本对齐 | 科学领域模型（注意：**分子图 ≠ 知识图谱**） |
| **Toolink** | *Toolink: Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model*（NAACL 2024 long，`2024.naacl-long.48`） | CoS-GPT 数据集 + LLaMA-CoS，工具创建与调用链 | **工具编排**（但不以 KG 为核心） |

**两条必须在引用时纠正的事实错误**

1. **EDRM ≠ "entity-disambiguated"。** 正确展开是 **Entity-Duet Neural Ranking Model**。用错展开会让缩写完全失真。
2. **SciKGAT 的核验对象是 COVID-FACT，不是 SciFact。** 最初"在 SciFact 上报告约 30 个百分点 precision 提升"的说法**未能核验**，本报告**不再把它当作论据**。若确需引用，请回到原文确认数据集与指标。（另注意三个易混对象：**SciFact** = Wadden et al. EMNLP 2020, `2020.emnlp-main.609`；**SciClaim** = Magnusson & Friedman EMNLP 2021, `2021.emnlp-main.381`；**COVID-FACT** = SciKGAT 一文使用的数据集。三者不同。）

**评估**：这条线提供了 AutoResearch 的三个关键模块（**找证据 / 组织知识 / 生成实验方案**），但**没有形成完整闭环**：

```text
科研问题提出 → 文献知识图谱检索 → 新假设生成 → 实验设计
  → 代码/实验执行 → 结果解释 → 反思迭代
```

ReactXT 特别值得单独说明：它是**分子图 + 文本 + 反应序列的多模态语言模型**，三处训练上下文（forward / backward / random molecule context）解决"reversal curse"，OpenExp 提供结构化动作序列。它能承担科研 agent 中的"实验步骤生成"子模块，但**没有假设提出、没有文献综合、没有闭环迭代**。

**对 Gaia-IR 的价值**：ReactXT 的 **OpenExp 动作序列格式**（ADD / STIR / HEAT / CONCENTRATE / DRY / PURIFY）可以直接借鉴为 Gaia-IR next-step planning 输出里"最小实验设计"的**结构化模板**——否则 LLM 会用自然语言自由发挥，评测无法自动化。

---

### 1.3 线三：AutoResearch agent 与 benchmark 范式

| 论文 | 工作流 | 如何证明更好 | 关键数字 |
|---|---|---|---|
| **MLAgentBench**（ICML 2024，PMLR 235:20271–20309） | ReAct agent 读写文件、运行代码、检查结果、反复改进；13 个实验任务（CIFAR-10、BabyLM 等） | 比较 GPT-4 / Claude / Gemini 的任务成功率 | Claude 3 Opus 平均 ≈ 37.5%；遇到训练数据中不存在的新任务时成功率可降到 0% |
| **MLE-bench**（ICLR 2025；arXiv:2410.07095） | 75 个真实 Kaggle 竞赛，agent 自己数据处理、训练、调参、提交 | Kaggle 排名 + 人类 baseline；另测算力扩展与训练数据污染 | 最佳 o1-preview + AIDE 达到铜牌及以上的比例为 **16.9%** |
| **ScienceAgentBench**（ICLR 2025） | 44 篇论文的 102 个真实科研任务转成可执行 Python 程序 | 可执行性优先：代码能否运行、结果是否正确、token/时间成本；9 位领域专家验证 + 污染控制 | 最佳 agent 独立完成率仅 **32.4%** |
| **AI-Researcher + Scientist-Bench**（NeurIPS 2025 main，*AI-Researcher: Autonomous Scientific Innovation*） | 三阶段流水线：文献/代码检索 → Resource Analyst 把论文公式映射到代码 → 生成想法 → Code Agent 实现 → Advisor Agent 反复审查 → 自动写论文 | 22 篇论文；分开统计代码完整性、实现正确性、论文质量；多 LLM judge 交叉比较 | Claude 系列完整执行率 93.8%，正确性平均仅 **2.65/5** |
| **AutoDiscovery**（NeurIPS 2025 main） | 不给定研究问题，agent 自己探索假设；用 LLM 先验/后验变化定义 **Bayesian surprise** + progressive widening **MCTS** | 21 个真实数据集、固定预算，比对比策略多发现 **5–29%** 惊奇结果；专家判断约 2/3 确实令人惊讶 | 证明的是"探索效率"，不是"科学价值" |
| **PaperBench**（ICML 2025 poster 43586，*PaperBench: Evaluating AI's Ability to Replicate AI Research*） | 从零复现 20 篇 ICML 2024 Spotlight/Oral | 拆成 **8,316** 个可评分子任务；rubric 由原论文作者参与制定；单独验证 LLM judge；与顶尖 ML PhD baseline 比较 | 最佳测试 agent 平均复现分数仅 **21%** |

> **URL 可用性提示（核验发现）**：`openai.com/index/mle-bench/` 与 `openai.com/index/paperbench/` 对自动抓取返回 **HTTP 403**（反爬，非 404），无法编程确认。正式引用请改用：
> - MLE-bench → ICLR 2025 proceedings：`https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html`
> - PaperBench → ICML 2025 poster：`https://icml.cc/virtual/2025/poster/43586`
>
> **AI-Researcher 的 NeurIPS 链接必须修正**：`/file/` 路径会 **404**，正确路径是 `/hash/`：
> `https://proceedings.neurips.cc/paper_files/paper/2025/hash/0d904d300a105809a2114d727851e759-Abstract-Conference.html`

**顶会认可的六条证据标准**（这就是我们 E0–E3 必须遵守的协议）

1. **从文本评价转向可执行结果** —— 代码能不能跑、算法有没有实现、指标有没有达到、能不能复现、隐藏测试集/排行榜表现。
2. **固定预算下比较** —— token、wall-clock、GPU/CPU 数量、实验次数、API 成本；比较"单位预算完成多少科研任务"。
3. **做 scaffold ablation** —— 直接 prompt / ReAct / OpenHands / self-debug / 有无文献检索 / 有无 reviewer / 有无多轮 / 有无树搜索，判断提升来自 harness 还是来自更强的底层模型。
4. **硬指标与软指标分开** —— 硬：execution pass rate、implementation completeness、correctness、benchmark score、Kaggle medal、reproduction score；软：novelty、rigor、significance、scientific insight。
5. **用人类专家校准评价器** —— PaperBench 单独做 judge benchmark；AI-Researcher 多模型交叉打分；ScienceAgentBench 领域专家参与设计与人工验证。
6. **报告失败率与污染风险** —— 负结果比"生成了一篇漂亮论文"更有说服力。

---

### 1.4 线四：Gaia-IR 的最近邻与 novelty 边界

| 方向 | 代表工作（真实标题） | 已经做了什么 | Gaia-IR 尚未被覆盖的部分 |
|---|---|---|---|
| 科学 claim 图谱抽取 | *Extracting Fine-Grained Knowledge Graphs of Scientific Claims: Dataset and Transformer-Based Results*（EMNLP 2021，`2021.emnlp-main.381`；提出 **SciClaim** 数据集） | 从论文抽取细粒度 claim、实体、关系、属性 | 是抽取数据集，不是跨论文推理系统 |
| Claim provenance | *"Who said it, and Why?" Provenance for Natural Language Claims*（ACL 2020，`2020.acl-main.406`）；*What is Your Article Based On? Inferring Fine-grained Provenance*（ACL-IJCNLP 2021，`2021.acl-long.458`） | 建立 claim 的来源、演化、相互依赖 | 没把 provenance 变成可执行的研究 pipeline |
| 科学 claim 因果推理 | *Characterizing and Verifying Scientific Claims: Qualitative Causal Structure is All You Need*（EMNLP 2023，`2023.emnlp-main.828`） | 异构图表示 claim 的因果结构 + GNN 做核验 | 面向 claim verification，不是跨论文研究发现 |
| 文献图辅助 idea generation | ResearchAgent（→ NAACL 2025，arXiv:2404.07738）、SciMON（→ ACL 2024，`2024.acl-long.18`）、KG-CoI（arXiv:2411.02382） | 用 citation / KG / 检索结果辅助生成研究问题或假设 | 图是检索增强或 idea 辅助，不是带 deduction/abduction/contradiction 的正式 claim graph |
| 带 provenance 的科学假设生成 | **HypER**: *Literature-grounded Hypothesis Generation and Distillation with Provenance*（EMNLP 2025 main，`2025.emnlp-main.1292`） | 训练 SLM 做文献引导推理 + 有证据支撑的假设 | 非持久化领域推理图；无条件闭包、无图编译、无 checkpoint/replay harness |
| 图约束的跨论文多跳 synthesis | **SciNets**: *Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis*（arXiv:2601.09727，**预印本**） | 从文献构造 concept graph，探索多跳路径 | query-local concept graph；无 claim 级条件语义、无可执行 pipeline |
| 科学 reasoning benchmark | **SciR**: *A Controllable Benchmark for Scientific Reasoning in LLMs*（arXiv:2606.13020，**预印本**）；**PaperScope**（arXiv:2604.11307，**预印本**） | 分别控制 deduction/induction/abduction 难度；多论文、多模态 deep research | 是 benchmark，不是生产级研究图谱与 harness |
| 端到端科研 agent | **AI-Researcher: Autonomous Scientific Innovation**（NeurIPS 2025 main） | 文献 → idea → 代码 → 实验 → 论文的完整 workflow | 没把跨论文科学推理显式表示为可验证 IR |
| 最新相关工作 | *Graph-Native Reinforcement Learning Enables Traceable Scientific Hypothesis Generation through Conceptual Recombination*（arXiv:2607.00924，**预印本**；模型名 Graph-PRefLexOR）；**HyGRAIL**（arXiv:2609.02056，**预印本**）；*Tracing the Cascade: A Topology-Aware Evaluation Framework for Scientific Agent Hallucinations*（arXiv:2608.00711，**预印本**；框架名 SCHEMA）；**CrossTrace**（arXiv:2603.28924，**预印本**） | graph-native 假设生成；KG + LLM 假设发现；**拓扑感知幻觉评估**；带逐步证据的 reasoning trace | 各自只覆盖 Gaia-IR 的一部分 |

> **引用强度分级（核验后必须遵守）**
>
> | 级别 | 条目 | 可否作为论据 |
> |---|---|---|
> | **A. 同行评审** | CodRED、Entity-centered、MR.COD、PILOT、LGCR、REIC、KXDocRE、CoRE-NEPD、CLEAR、HCRE、SciClaim、两条 provenance、EMNLP 2023 因果结构、HypER、ResearchAgent、SciMON、AI-Researcher、MLAgentBench、MLE-bench、ScienceAgentBench、AutoDiscovery、PaperBench、EDRM、Denoising DS-QA、SciKGAT、ReactXT、PRESTO、Toolink、KG-CoI | ✅ 可作核心论据 |
> | **B. 未评审预印本（2026）** | SciNets、SciR、PaperScope、Graph-PRefLexOR、HyGRAIL、SCHEMA、CrossTrace | ⚠️ **只作为"最新动态"提及，不得作为 novelty 判断的唯一支撑** |
>
> **B 级条目的额外风险**：SciNets 为**单作者**预印本；CrossTrace 的合著者列出一个非人名（"OMS-II Research Fellow"），且全文为第一人称单数。二者均**未经同行评审**。在正式论文里若要引用，必须明确标注 "preprint, unrefereed"。
>
> **这也意味着 novelty 论证的结构要调整**：最初版本把 SciNets / SciR / SCHEMA 当作"最接近的四篇"——其中三篇是预印本。**真正能撑住 novelty 边界的同行评审工作只有 HypER（EMNLP 2025）**。因此 novelty 应主要对比 **HypER + AI-Researcher + CodRED 谱系**，把 2026 预印本放在"同期动态"一节。

**易混三件套（务必分清，否则引用会错）**

| 名称 | 是什么 | 出处 |
|---|---|---|
| **SciFact** | 数据集 | Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020, `2020.emnlp-main.609` |
| **SciClaim** | 数据集 | Magnusson & Friedman, EMNLP 2021, `2021.emnlp-main.381` |
| **COVID-FACT** | 数据集 | SciKGAT 一文（Findings of EMNLP 2020, `2020.findings-emnlp.216`）实际使用的核验数据集 |

**最接近的对照与逐项差异**（注意列中三项为未评审预印本，见上方分级）

| 能力 | HypER ✅A | SciNets ⚠️B | SciR ⚠️B | SCHEMA ⚠️B | **Gaia-IR** |
|---|:--:|:--:|:--:|:--:|:--:|
| claim / observation 级跨论文图 | 部分 | 部分 | — | — | ✅ |
| deduction / induction / abduction / contradiction 明确 relation type | 部分 | — | 部分 | — | ✅（9 种策略 + 7 种算子，各有 canonical 微观骨架） |
| condition / context preservation | — | — | — | — | ✅（`background` + 条件对比门禁） |
| source provenance closure | ✅ | — | — | 部分 | ✅（source anchor + PackageRef + `ir_hash`） |
| 图作为**可执行**中间表示 | — | — | — | — | ✅（compile → validate → lower → BP） |
| checkpoint / artifact hash / 失败隔离 / replay | — | — | — | — | ✅（`RunStore` + `run_pipeline` + `check_run`） |
| 过程指标（答案对但链错也要扣分） | 部分 | — | ✅ | ✅ | ✅（StrictSuccess） |
| 真实 AutoResearch downstream utility | — | — | — | — | 待证明（E2/E3） |

**Novelty 的稳妥表述**

> ❌ 不要写：*"We build a scientific knowledge graph for research."*
> ❌ 不要写：*"We are the first to generate hypotheses from papers."*
> ✅ 应该写：*"We introduce an execution-oriented, condition-aware scientific reasoning representation that compiles cross-paper claims, evidence, and typed inference relations into replayable research pipelines."*

中文版：

> 本文提出一种面向执行的、条件感知的科学推理表示，把跨论文的 claim、证据与类型化推理关系编译为**可回放的研究流水线**，并用过程指标与真实下游效用验证其价值。

---

## 2. 创新点定位

### 2.1 命题拆解：三个必须分开测的主张

| 命题 | 内容 | 测法 |
|---|---|---|
| **A：图结构有用** | Gaia 图比普通文本检索 / flat claim list 更能支持多跳推理 | B0–B4 对比，按 hop 数分桶 |
| **B：严格 schema 有用** | typed relation + condition + provenance 能减少"把相关当推理 / 忽略适用条件 / 错误合并不同实验结论 / 生成无来源中间结论" | 图/知识层消融 |
| **C：harness 有用** | 同一图 + 同一模型下，pipeline harness 比单体脚本更可恢复、可复现、可审计、更省重复调用 | harness 层消融，用工程指标而非 accuracy |

**三个命题必须分开测**，否则最后无法判断提升来自图、来自 prompt、还是来自运行框架。

### 2.2 Gaia-IR 已有的技术资产（作为 baseline 的起点）

| 资产 | 现状 | 位置 |
|---|---|---|
| 图谱规模 | v9：**318 节点 / 257 边 / 74 条跨论文 delta 边**（62 deduction · 10 abduction · 2 contradiction） | `example/view_model.json` |
| 结论结构 | 1 棵结论树覆盖 8 个 L1 结论 + 3 个独立 L1 结论（保留 `GRAPH_DISCONNECTED` 警告） | `README.md` |
| 单篇 → 合并链路 | LKM 原生 518 knowledge → clean claims 151 claim/note + 74 relation candidates → single formalization 335 knowledge / 92 operators / 128 strategies → merge 展示层 | `docs/validation/single-pipeline-comparison.md` |
| Harness | `RunStore`、`run_pipeline`、`check_run`、provenance、validation、compiler、indexing（约 5.7k 行） | `agent/pipeline_harness/` |
| 契约层 | 9 篇 Gaia IR 契约文档 + 3 篇 theory 文档（621 + 671 + 506 行）+ 2 篇 pipeline 文档 | `docs/` |
| 可读 | 自包含 Viewer；单篇与 merge 两条维护中的配置 | `example/viewer.html`、`pipelines/` |
| 语料 | `data/` 下已有 **102 篇**论文目录（含 `ocr.md`、`claims_final.json`、`graph.json`、`organized_content.json` 等） | `data/` |
| 评测 | **尚不存在任何 benchmark 目录** —— E0 需要从零建立 | `benchmark/`（待创建） |

**必须承认的现实**：以上只能证明 pipeline 在迭代，**不能证明 pipeline 有科学价值**。边数从 26 增到 74、节点从少变多，都不是效果指标。

### 2.3 创新点的三级递进（写论文时的贡献排序）

1. **任务定义**：`next-step research planning` —— 给定截至时间 \(t\) 的论文集合、领域推理图和一个未解决研究问题，生成并排序**未来可执行的研究方案**（每个方案含假设 / 证据来源 / 适用条件 / 最小实验设计 / 预期观察 / 证伪条件 / 资源估计）。
2. **表示与系统**：Gaia-IR 的条件感知跨论文推理表示 + pipeline harness。
3. **Benchmark 与评测协议**：StrictSuccess、无依据推断率、条件违反率、trace 有效性、恢复率、hash 一致性；三层评分（答案分 / 科学证据分 / 轨迹分）。

论文类型因此**不是纯模型论文**，而是 **system + benchmark + evaluation protocol** 论文。

---

## 3. 下一步具体怎么做

### 阶段总览

```text
P0 冻结与规格化（1 周）
   ↓  gate: benchmark spec + checker 可实现
P1 E0 受控 benchmark（3–4 周）
   ↓  gate: 图与 harness 各自有可测增益
P2 E1 消融（2–3 周，可与 P1 后期并行）
   ↓  gate: 归因清晰，无混淆变量
P3 E2 真实 AutoResearch 任务（6–8 周）
   ↓  gate: 下游效用可测
P4 E3 next-step research planning + 人类 baseline（4–6 周）
   ↓
P5 论文 / 开源产品化
```

---

### P0：冻结与规格化（第 1 周）

**做什么**

1. **冻结评测语料**：LTH 领域 5–10 篇论文（先用已有 5 篇打样，再扩到 10 篇）。冻结 `gaia.ir`、`formalization`、`knowledge.index` 与各自 SHA-256。
2. **写 benchmark spec 文档**（`docs/benchmark/e0-spec.md`），必须包含：
   - 问题分桶规则：1-hop / 2-hop / 3-hop / 4-hop / 完整合取 / 条件不匹配 / 冲突证据 / unknown / 跨论文替代解释 / 相关但不等价
   - 每题的 gold：答案 + 必需前提集合 + 证据路径（每条边的 relation type）+ 适用条件 + 冲突/未知标记 + 来源 claim ID
   - 题目发布流程（复用 Agentic Challenge 已有的 discipline）：`challenge_id`、`sourceQuestionId`、`package SHA`、strict preflight、baseline、oracle、private/hidden test、package readiness
3. **写 StrictSuccess checker 的确定性实现**（不依赖 LLM judge）。

**StrictSuccess 定义（全部满足才算成功）**

- 最终答案正确；
- 必需前提没有漏掉（premise recall = 1）；
- 证据路径中每条边的 relation type 正确；
- 条件没有被删除、放宽或偷换；
- 每个结论都有合法 provenance；
- 不把 unsupported inference 当作确定结论；
- 对 unknown 问题能正确拒答或保留不确定性。

**产出**：`docs/benchmark/e0-spec.md`、`benchmark/e0/questions.jsonl`、`benchmark/e0/strict_check.py`

**出题可以直接从现有 artifact 半自动生成**（不用从零人工写题）

每篇论文目录（如 `data/<paper_id>/`）已经包含：

| 文件 | 内容 | 出题用途 |
|---|---|---|
| `claims_final.json` | `claim` / `note` / `relation` 三类，claim 带 `conclusion`、`text`、`is_pure_data`、`needs_more_context`、`number` | 直接作为 gold premise 候选与 relation type 标注来源 |
| `conclusions_labeled.json` | 论文级结论，带 `global_id`、`order`、`title`、`content`、`labeled_segments`、`labeling_status` | 直接作为 L1 结论与答案骨架 |
| `index.json` | `segments` / `headings` / `char_offset_unit` / `line_numbering` / `formulas` | 生成 source anchor 与证据句定位 |
| `graph.json`、`step2_output_claims.json`、`organized_content.json` | 中间结构 | 交叉校验与消融输入构造 |

**做法**：从 `conclusions_labeled.json` 取 L1 结论做 gold answer，从 `claims_final.json` 的 `relation` 取 gold 证据路径，从 `index.json` 的 `segments` 取 gold source anchor；然后**人工只做三件事**——确认前提必要性、补写 4-hop 题、标注条件与冲突桶。这样 120–200 题的构造成本可以压到可控范围。

**Gate**：checker 能在 oracle 答案上 100% 通过、在故意植入的 5 类错误上 100% 拒绝。

---

### P1：E0 受控 benchmark（第 2–5 周）

**任务规模**：5–10 篇 LTH 论文 → **120–200 道问题**。每题不仅要求答案，还要求：给出证据路径、指出适用条件、标注冲突或未知、返回来源 claim ID。

**五组系统（必须固定同一模型、相同 token/context 预算、相同证据数量、相同问题、3 次重复运行）**

| 系统 | 输入 | 作用 |
|---|---|---|
| **B0** | Qwen-7B 直接读论文文本 | 下界 |
| **B1** | Qwen-7B + BM25/向量检索 | 检索增益 |
| **B2** | Qwen-7B + flat claims | 结构化但无图 |
| **B3** | Qwen-7B + Gaia 图，**去掉 provenance/condition** | 图结构增益 |
| **B4** | 完整 Gaia-IR pipeline harness | 完整系统 |

> ⚠️ **最容易犯的错误**：让 B4 获得更多上下文后再声称是 harness 的功劳。必须把"图带来的提升"和"运行机制带来的提升"分开。

**主指标（不要只用 accuracy）**

- 主指标：`StrictSuccess`
- 诊断指标：answer accuracy、evidence path precision/recall、premise recall、**unsupported inference rate**、**condition violation rate**、contradiction precision/recall、unknown calibration、token/时间/API 成本、trace validity
- 组织方式：**主指标 + 诊断指标**，不要只给一个总分

**理想结果不是"B4 绝对分最高"，而是**：

> 随着 hop 数、前提数量和干扰比例增加，**B4 的性能下降得最慢**。

**Go / No-Go 门槛**

- 完整系统相对最强 baseline 的 StrictSuccess **≥ +10 个百分点**；
- unsupported inference **下降 30–50%**；
- 3-hop / 4-hop 上性能衰减明显更慢；
- 所有主要结论有 **paired bootstrap 95% CI**，而不是只报一次运行。

---

### P2：E1 组件消融（第 6–8 周）

**图 / 知识层消融** —— 回答"知识表示是否有用"

- 去掉 typed relation（全部降级为 `infer`）
- 去掉 condition（丢弃 `background` 与条件对比门禁）
- 去掉 provenance（丢弃 source anchor）
- 去掉结论树
- 用 flat claim 替代 graph

**Harness 层消融** —— 回答"pipeline 工程机制是否有用"

- 不使用 checkpoint
- 不使用 replay
- 失败产物允许进入下一步
- 不做 schema validation
- 不做 official compiler 校验
- 只跑一次，不允许 retry / fork

**关键**：Harness 层**不一定直接提升 answer accuracy**，它应该提升的是：

| 指标 | 测法 |
|---|---|
| 中断恢复成功率 | 中途随机 kill 一个 stage，完整 pipeline 恢复成功的比例 |
| 重复成本节省 | checkpoint/replay 是否比从头重跑少 ≥ 30% token 与时间 |
| 结果一致性 | 同一输入同一配置重复运行，最终 artifact hash 是否一致 |
| 无效产物拦截率 | 故意注入非法产物，被 validator 拦下的比例 |
| 失败 stage 可定位率 | 报错能否定位到具体 stage/attempt |
| 审计完整度 | 每个结论能否回溯到 source anchor |

**这才是"harness 有用"的直接证据**，而不是让它去比拼 accuracy。

---

### P3：E2 真实 AutoResearch 任务（第 9–16 周）

对 103 篇 LTH 论文做三类任务。

#### 任务一：跨论文综合

输入若干论文，要求 agent 找出：多篇共同支持的结论 / 条件不同导致的差异 / 互相替代的解释 / 当前无法确定的结论。

#### 任务二：矛盾发现（时间切分）

```text
早期论文 → 构图
新论文   → 作为查询（禁止把新论文内容直接放入图）
         → 判断新论文与已有结论是矛盾、补充，还是限定
```

指标：contradiction precision / recall、condition mismatch accuracy、alert precision、每篇论文人工复核后的**有效 alert 数**。

#### 任务三：研究问题生成

比较"有图"与"无图"时，agent 是否更容易提出：有明确证据缺口的问题 / 能区分冲突和条件差异的问题 / 可以实验验证的问题 / 不只是重组已有结论的问题。

需要盲评（1–5 分 rubric，不告诉专家方案来自哪个系统）：novelty、evidence completeness、scientific plausibility、testability、是否真的来自跨论文结构。

---

### P4：E3 next-step research planning（第 17–22 周）

**任务定义**：给定论文集合、领域推理图和未解决研究问题，输出排序后的研究方案，每个方案包含：研究假设、证据来源、适用条件、最小实验设计、预期观察、**可能证伪它的结果**、资源与成本估计。

**Baseline 矩阵**

| Baseline | 能力 |
|---|---|
| Direct LLM | 只给问题 |
| Text-RAG | 检索论文文本 |
| Flat-claim RAG | 检索 claim 列表 |
| Graph-RAG | 用 Gaia 图，但不带完整 condition/provenance |
| **Gaia-agent** | 完整图、关系、条件、provenance 和 pipeline harness |
| Human baseline | 领域研究者提出方案 |

**指标**

1. **方案质量**：`Accepted Plan Recall@K`（专家预先标注可接受方案集合）、专家 pairwise win rate、actionability、feasibility、novelty、falsifiability。
2. **证据与推理质量**（Gaia 最有优势的地方）：Evidence Support Precision、Provenance Coverage、Condition Preservation、Evidence Path Precision/Recall、Contradiction Detection F1、Unsupported Recommendation Rate、错误跨条件泛化率。
3. **实际科研收益**：推荐实验成功率、达到目标指标所需实验次数、time-to-first-valid-plan、每个有效方案的 token/API/GPU 成本、单位成本带来的性能提升。
4. **Harness 自身价值**：同 P2 的工程指标。

**三种数据来源结合**

- **A. 专家构造题**：50–100 个真实领域问题，由专家提供可接受方向、必须引用的证据、不可违反的条件、合理实验设计。
- **B. 时间切分 retrospective benchmark**：只给某时间点以前的论文，让它提出下一步方向，然后检查后来是否真的出现类似方法 / 是否解决了当时的矛盾 / 是否与后续实验结果一致。
- **C. 可执行实验 benchmark**：把推荐方案转成可运行任务，直接检查代码是否运行、实验是否完成、是否达到目标、是否真的验证了原始假设。

---

### P5：论文与产品

**建议顺序**：先产品化验证真实需求，再把产品中稳定的任务抽象成论文。

- 作为产品：证明研究者**确实愿意用、能节省时间、能找到有价值的下一步**；
- 作为论文：最大难点是"下一步解决方案"通常**没有唯一 gold answer**，容易变成主观评价——所以必须靠 `Accepted Plan Recall@K` + 盲评 + 可执行验证三者撑住。

**路线**：开源 Gaia-agent MVP → 收集真实研究问题与反馈 → 固化成 Next-Step Research Benchmark → 写 system/benchmark paper。

**最终论文主张（不要过度）**

> ❌ 不写："Gaia-agent 能自主完成科学发现。"
> ✅ 写："在固定模型和证据预算下，结构化跨论文推理图能够让研究 agent 生成更有证据支持、更少条件错误、更加可执行的下一步研究方案。"

---

## 4. 可以复用已有文献的哪些东西（重点）

### 4.1 直接可复用的方法组件

| 文献 | 复用内容 | 用在 Gaia-IR 的哪一步 | 复用方式 |
|---|---|---|---|
| **MR.COD**<br>*Multi-hop Evidence Retrieval for Cross-doc RE*, Findings of ACL 2023 | 多跳证据检索 + contextual dense retriever 对证据排序 | **Pipeline 8 Step 2**（目前只做一跳扩展） | 把一跳扩展改成"多跳路径枚举 + 路径排序"，路径分数只用于召回、不进入 prior |
| **PILOT**<br>*Explore the Way: Exploring Reasoning Path by Bridging Entities*, Findings of EMNLP 2023 | 把桥接实体作为 stepstone 探索推理路径；可作为桥接合法性校验 | Step 3 的 analogy `BridgeClaim`、abduction `AltExp` 校验 | 作为机械前置检查：桥接不成立的候选直接不写入 |
| **CLEAR**<br>*Cross-Document Link-Enhanced Attention ...*, Applied Sciences 15(13):7435 | relation-aware context filter + 路径距离加权 attention | Step 2 → Step 3 之间的候选分组过滤 | 用目标关系类型反向过滤候选 group，减少 Step 3 的 LLM 调用量 |
| **REIC**<br>*Reward-based Input Construction for Cross-doc RE*, ACL 2024 | 以下游 RE 分数为 reward 的 RL 句子选择器（该方向首个 learning-based selector） | Step 1 的 `source_excerpts` 选择 | 训练/蒸馏一个 excerpt selector，直接降低 token 成本（也服务 P2 的成本指标） |
| **KXDocRE**<br>*Knowledge-Driven Cross-Document Relation Extraction*, Findings of ACL 2024 | 实体领域知识注入 + 解释性证据输出 | Step 1/2 的实体消歧与 claim 归一 | 用实体类型与 KG 路径辅助 canonicalization；解释性证据作为答案的强制字段 |
| **CoRE-NEPD**<br>*...Non-bridge Entity Enhancement and Prediction Debiasing*, Findings of ACL 2024 | NA 样本偏置校准 + 非桥接实体的利用 | Step 3 的"证据不足不写入"决策 | 把"不建立关系"当作**可评估的预测类别**，报告其 precision/recall 与 calibration |
| **LGCR** | 因果推理区分"相似但无关"文档 | Step 2 召回后的噪声过滤 | 作为 B1/B2 baseline 的噪声对照，突出条件门禁的增益 |
| **HCRE**<br>*LLM-based Hierarchical Classification ... with Prediction-then-Verification*, Findings of ACL 2026 | 关系类别层级化 + 逐层分类 + prediction-then-verification | Step 3 的 operator/weakpoint 判定 | 为"确定性 Operator / weakpoint 类型 / 不建立"设计层级决策树，每层后做 verification |
| **SciKGAT**<br>*Adapting Open Domain Fact Extraction and Verification to COVID-FACT*, Findings of EMNLP 2020 | 声明 → 检索 → 支持/反驳证据 → 判断 的核验流水线 | E0 的冲突题、E2 任务二的矛盾发现 | 作为 contradiction detection 的方法参照。⚠️ **原调研称其用 SciFact 且提升约 30 点，核验未能成立**（该文数据集是 COVID-FACT），**不要沿用该数字**。需外部 sanity check 请直接用 SciFact（`2020.emnlp-main.609`） |
| **EDRM**<br>*Entity-Duet Neural Ranking: Understanding the Role of KG Semantics in Neural IR*, ACL 2018 | 实体映射到 KG 表示后与文本表示联合排序（⚠️ EDRM = **Entity-Duet**，不是 entity-disambiguated） | Step 2 检索打分 | 在 BM25/向量之上加一路 entity-linked 分数，作为 B1 的增强 |
| **Denoising DS QA** | paragraph selector → reader 两阶段 | 整体架构 | 作为 B1 的架构模板；selector 的噪声过滤指标可复用 |

### 4.2 可直接复用的数据集与格式

| 来源 | 复用内容 | 用途 |
|---|---|---|
| **CodRED** | 任务框架、桥接实体多跳**评测设计**、NA 样本问题意识 | 借评测结构，**不借数据**（Wikipedia 场景 ≠ 论文场景） |
| **SciFact**（Wadden et al., EMNLP 2020, `2020.emnlp-main.609`） | 科学声明核验任务与指标 | E0 冲突/核验题的外部对照。⚠️ **不要写成「经 SciKGAT」**——SciKGAT 用的是 COVID-FACT |
| **ReactXT / OpenExp**<br>*Understanding Molecular "Reaction-ship"...*, Findings of ACL 2024 | 结构化实验动作序列格式（ADD/STIR/HEAT/CONCENTRATE/DRY/PURIFY）、约 27.4 万条反应—步骤对的构造方法 | E3 "最小实验设计"字段的**输出模板**，让评测可自动化 |
| **SciR** | deduction / induction / abductive 难度分桶控制 | E0 分桶设计的直接参照（与 Gaia 的 typed relation 天然对齐） |
| **PaperBench** | 8,316 子任务的 rubric 分解法 + 原作者参与制定 rubric + 单独验证 LLM judge | E3 专家 rubric 的方法论 |
| **ScienceAgentBench** | 102 个真实科研任务转可执行程序的构造流程 + 9 位专家验证 + 污染控制 | E3 可执行实验 benchmark 的构造流程 |
| **MLE-bench** | Kaggle 排名 + 人类 baseline + 算力扩展测试 + 训练数据污染测试 | 固定预算比较与污染风险报告的模板 |

### 4.3 可复用的评测协议与指标

| 来源 | 复用内容 |
|---|---|
| **ScienceAgentBench / MLE-bench / PaperBench** | 真实任务 + 沙箱执行 + 固定预算 + 隐藏评测 + 人类 baseline 这套成熟范式 |
| **PaperBench** | 硬指标与软指标分开报告；judge 单独做 benchmark 验证 |
| **AI-Researcher** | 多 LLM judge 交叉打分；advisor/reviewer 的多阶段 agent 架构 |
| **AutoDiscovery** | Bayesian surprise（LLM 先验/后验变化）+ progressive widening MCTS 作为**假设空间搜索**策略 |
| **SCHEMA** | 拓扑感知幻觉评估 —— 直接对应"最终答案正确但中间推理链错误"的过程级指标 |
| **HypER** | 有效/无效推理链的判别训练 —— 可作为关系类型判定器的训练信号与 baseline |
| **Agentic Challenge（已有实践）** | 三层评分（答案分 / 科学证据分 / 轨迹分）、题目发布流程、strict preflight、hidden test —— 复用现成 discipline 而不是重新发明 |

### 4.4 三条"不要复用"的教训

1. **不要复用 CodRED 的数据与关系标签体系** —— 276 个扁平关系类别 + Wikipedia 场景，和论文场景的条件结构不匹配。
2. **不要复用"让 LLM 一次在全部关系里选"的做法** —— HCRE 已经证明这会失败。
3. **不要复用"只看最终答案"的评测** —— SCHEMA 与 PaperBench 都表明必须看过程。

---

## 5. 实验协议要点（所有阶段共用）

### 5.1 必须固定的量

同一个 backbone；相同 token 与检索预算；相同论文集合；相同问题；多次随机运行（≥3 seed）。

### 5.2 三层评分（复用 Agentic Challenge 经验）

| 层 | 内容 | 评分方式 |
|---|---|---|
| 1. **答案分** | 最终回答是否正确 | 确定性 checker |
| 2. **科学证据分** | 证据路径、条件、关系类型是否正确 | 确定性 checker + 人工抽样复核 |
| 3. **轨迹分** | 是否真正使用了图、是否保留来源、是否可复现 | artifact/hash/trace 机械检查 |

> **不要让 LLM judge 单独决定主结果。** LLM judge 做辅助评分，且必须像 PaperBench 那样先单独验证 judge 本身。

### 5.3 已知的评测陷阱（来自 Agentic Challenge 的真实教训）

- checker 容差漏洞；
- 相同 bundle 重复评分不稳定；
- 元数据提示与真实 grader 不一致；
- LLM 规则分只能作参考；
- 轨迹污染会影响 trace 分；
- 静态审计无法证明 agent 真的执行了大规模搜索。

---

## 6. 风险与对策

| 风险 | 具体表现 | 对策 |
|---|---|---|
| **归因混淆** | B4 拿到更多上下文/更多 token 才赢 | 固定预算；把图增益与 harness 增益**分开消融**；报告 per-budget 曲线 |
| **Benchmark 主观性** | "下一步研究方案"没有唯一 gold | `Accepted Plan Recall@K` + 专家可接受方案集合 + 盲评 + 可执行验证三管齐下 |
| **数据污染** | LTH 论文很可能在 LLM 预训练数据里 | 时间切分；保留近期论文作 held-out；报告污染测试；改用改写/新论文子集复测 |
| **图质量本身是变量** | 图由 LLM pipeline 产出，图错了结论就不可信 | 人工抽检节点/边的正确率并作为**独立报告项**；把图质量作为 covariate |
| **规模不匹配** | 5 篇 → 103 篇，Step 3 的 LLM 调用与成本可能爆炸 | 先跑 10 篇；用 CLEAR 式关系过滤 + REIC 式 selector 压成本；记录每步成本 |
| **Step 2 多跳缺失** | 跨论文多跳关系召回不足，图会漏边 | 引入 MR.COD 式多跳路径枚举与排序（P1 的必做改进项） |
| **查询/评测泄漏** | 建图时用了新论文内容 | 时间切分协议写进 spec；Step 2 只从基线召回 |
| **关系表示能力不足** | Gaia 只有 9 策略 + 7 算子，可能表达不了真实科学论证 | 报告"无法表达"的样本比例与具体缺口，作为 limitation 与后续工作 |

---

## 7. 待办清单

### 未来 2 周（P0）

- [ ] 写 `docs/benchmark/e0-spec.md`：分桶规则、gold 结构、发布流程
- [ ] 实现 `benchmark/e0/strict_check.py` 并通过 oracle / 错误注入测试
- [ ] 冻结 5 篇 LTH 论文的 `gaia.ir` + `formalization` + SHA-256
- [ ] 从 v9 图中导出 B2/B3/B4 三种输入的构造脚本
- [ ] 明确 Qwen-7B 的部署与成本计量方式

### 未来 4 周（P1 启动）

- [ ] 出题 120 道（先 60 道打样），按 10 个桶分层
- [ ] 跑通 B0–B4 全链路，固定预算与 3 seed
- [ ] 实现 paired bootstrap CI 的报告脚本
- [ ] **Step 2 多跳改进**：基于 MR.COD 思路实现多跳路径枚举 + 排序（作为 B4 的增强版 B4+）

### 未来 8 周（P1 完成 + P2）

- [ ] E0 结果与 Go/No-Go 判定
- [ ] 图/知识层 5 项消融
- [ ] Harness 层 6 项消融 + 中断恢复实验
- [ ] 出第一版内部报告：三张表（Scientific utility / Ablation / Engineering reliability）

### 未来 6 个月（P3 + P4）

- [ ] 扩到 103 篇论文，构建 E2 三个任务的数据
- [ ] 时间切分 retrospective benchmark
- [ ] E3 专家题 50–100 道 + 人类 baseline
- [ ] 开源 Gaia-agent MVP，收集真实研究者反馈
- [ ] 论文投稿

---

## 8. 论文标题与摘要骨架（可直接迭代）

**候选标题**

1. *Gaia-IR: An Execution-Oriented, Condition-Aware Representation for Cross-Paper Scientific Reasoning*
2. *From Claims to Pipelines: Compiling Cross-Paper Scientific Reasoning into Replayable Research Agents*
3. *Condition-Aware Claim Graphs Beat Flat Retrieval for Multi-Hop Scientific Reasoning*

**摘要骨架（六句）**

1. **问题**：科研 agent 需要在不丢失适用条件与证据来源的前提下跨论文推理，而现有做法要么是扁平文本检索，要么是无类型的 citation/concept graph。
2. **缺口**：已有科学 claim 图谱工作（SciClaim、HypER、SciNets）既没有条件语义，也不是可执行表示；已有 AutoResearch agent（AI-Researcher 等）没有把跨论文推理显式表示为可验证 IR。
3. **方法**：我们提出 Gaia-IR —— 把跨论文 claim、证据与类型化推理关系（9 种推理策略，各自可展开为确定性算子骨架）编译为可回放的研究流水线。
4. **系统**：pipeline harness 提供 checkpoint/replay、artifact hash、失败隔离与 compiler/validator 门禁。
5. **评测**：在 LTH 领域的受控 benchmark（5–10 篇论文 / 120–200 题，10 个难度桶）上，与 text-RAG、flat-claim RAG、无 condition/provenance 的图三种 baseline 在固定模型与预算下对比；另在 103 篇论文语料上评测跨论文综合、矛盾发现与研究问题生成。
6. **结果主张**：StrictSuccess 相对最强 baseline 提升 ≥ 10 个百分点，unsupported inference 下降 30–50%，且随 hop 数增加性能衰减更慢；harness 在中断恢复与重复运行一致性上显著优于单体脚本。

**三个可证伪假设（写进论文的 pre-registration）**

- **H1**：在 3-hop 与 4-hop 问题上，完整系统的 StrictSuccess 相对 flat-claim RAG 的优势**大于**在 1-hop 上的优势。
- **H2**：移除 condition 字段导致的 unsupported inference 上升幅度，**大于**移除 typed relation 导致的上升幅度。
- **H3**：harness 消融**不显著**影响 answer accuracy，但**显著**影响中断恢复成功率与重复运行 hash 一致性。

> H1–H3 的价值在于：即使完整系统没有拿到绝对最高分，只要这三个方向性假设成立，论文的贡献依然清晰。

---

## 附录 A：文献引用核验结果（已完成）

**核验方式**：37 条引用逐条对着 ACL Anthology / arXiv / PMLR / ICLR / NeurIPS proceedings / Crossref **实时抓取**验证。

**总体结论**：**没有一条是伪造的** —— 37 条全部指向真实、可定位的文献。系统性问题有两类：

1. **标题被替换成描述性转写或缩写**（37 条中 15 条）—— 缩写是系统/方法名，不是论文标题；
2. **一个 URL 路径错误**（AI-Researcher 的 NeurIPS `/file/` → 应为 `/hash/`）。

### A.1 必须修改的引用（按严重度）

| # | 条目 | 问题 | 修正 |
|---|---|---|---|
| 1 | **AI-Researcher**（NeurIPS 2025） | 链接 `/file/` 路径 **404** | 改为 `/hash/0d904d300a105809a2114d727851e759-Abstract-Conference.html`；正确标题为 *AI-Researcher: Autonomous Scientific Innovation* |
| 2 | **EDRM**（ACL 2018） | 缩写展开**错误** | EDRM = **Entity-Duet** Neural Ranking Model，**不是** "entity-disambiguated"。正确标题 *Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval*（`P18-1223`） |
| 3 | **组归属**（线二） | "刘知远组"整体归属**不成立** | ReactXT 作者单位是 **NUS/USTC/Hokkaido**（同名 ≠ 同人）；PRESTO 来自 **IDEA**。只有 EDRM / Denoising DS-QA / SciKGAT / Toolink 一组归属成立 |
| 4 | **SciKGAT** | 标题与数据集**均被写错** | 正确标题 *Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling*（`2020.findings-emnlp.216`）；核验数据集是 **COVID-FACT**，不是 SciFact。"~30 个百分点提升"**未能核验，已从正文删除** |
| 5 | **"ECRIM"** | 该缩写在原文中**不存在** | 该文无系统名，只能写 *Entity-centered Cross-document Relation Extraction*（`2022.emnlp-main.671`） |
| 6 | **CodRED** | venue 与标题双错 | 是 **EMNLP 2021 main**（不是 ACL 2021），完整标题含 *for Acquiring Knowledge in the Wild*，ID `2021.emnlp-main.366`（原缺 ID） |
| 7 | **MR.COD** | 缩写展开错误 | 正确展开 *Multi-hop **Evidence Retrieval** for Cross-document Relation Extraction*，已正式发表于 `2023.findings-acl.657` |
| 8 | **PILOT / REIC / KXDocRE / LGCR / CoRE-NEPD / HCRE** | 标题为杜撰或转写 | 真实标题见 §1.1 表。PILOT = Path wIth expLOraTion；REIC = REward-based Input Construction |
| 9 | **Graph-PRefLexOR / SCHEMA** | 当作标题引用 | 二者分别是**模型名**与**框架名**；真实标题见 §1.4 |
| 10 | **SciClaim** | 数据集名当标题 | 论文标题 *Extracting Fine-Grained Knowledge Graphs of Scientific Claims: Dataset and Transformer-Based Results*（`2021.emnlp-main.381`） |
| 11 | **预印本升级** | 未用已发表版本 | MR.COD → `2023.findings-acl.657`；CoRE-NEPD → `2024.findings-acl.38`；ResearchAgent → NAACL 2025；SciMON → `2024.acl-long.18` |
| 12 | **MLE-bench / PaperBench** | openai.com 链接**无法确认** | 两者对自动抓取返回 **HTTP 403**（反爬，**不是 404**）。改用 ICLR 2025 proceedings 与 `https://icml.cc/virtual/2025/poster/43586` |

### A.2 一个被误判为"可能有错"、实际正确的条目

最初怀疑 *Characterizing and Verifying Scientific Claims*（EMNLP 2023）的 Anthology ID 与标题不符。**核验结果：ID 正确**，`2023.emnlp-main.828` 确实承载该标题（Wu, Chao, Zhou, Luo；pp. 13428–13439）。

但这里有一个**必须记住的三方区分**：

| 名称 | 是什么 | 出处 |
|---|---|---|
| **SciFact** | 数据集 | Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020, `2020.emnlp-main.609` |
| **SciClaim** | 数据集 | Magnusson & Friedman, EMNLP 2021, `2021.emnlp-main.381` |
| **COVID-FACT** | 数据集 | SciKGAT 一文实际使用 |

### A.3 引用强度分级

| 级别 | 条目 | 使用限制 |
|---|---|---|
| **A：同行评审** | 30 条（完整清单见 §1.4 分级表） | ✅ 可作核心论据 |
| **B：未评审预印本** | SciNets、SciR、PaperScope、Graph-PRefLexOR、HyGRAIL、SCHEMA、CrossTrace（均 2026） | ⚠️ 只作"同期动态"；**不得作为 novelty 判断的唯一支撑** |

B 级还有两条额外风险：**SciNets 是单作者预印本**；**CrossTrace 把非人名（"OMS-II Research Fellow"）列为合著者**且全文第一人称单数。二者均未评审。

**对报告结论的影响**：最初版本把 SciNets / SciR / SCHEMA 与 HypER 并列为"最接近的四篇"——其中三篇是预印本。**能撑住 novelty 边界的同行评审工作实际只有 HypER（EMNLP 2025）**，因此 novelty 论证已改为主要对比 **HypER + AI-Researcher + CodRED 谱系**。

---

## 附录 B：核心论文清单（按优先级，标题已核验）

**第一优先：决定 novelty 边界（以同行评审为主）**

1. **HypER**: *Literature-grounded Hypothesis Generation and Distillation with Provenance* — EMNLP 2025 main, `2025.emnlp-main.1292` ✅
2. **AI-Researcher: Autonomous Scientific Innovation** — NeurIPS 2025 main ✅
3. *Characterizing and Verifying Scientific Claims: Qualitative Causal Structure is All You Need* — EMNLP 2023 main, `2023.emnlp-main.828` ✅
4. **SciNets**（arXiv:2601.09727）、**SciR**（arXiv:2606.13020）、**SCHEMA**（arXiv:2608.00711）⚠️ 均为 2026 未评审预印本，仅作同期动态

**第二优先：决定方法可复用性**

5. **MR.COD**: *Multi-hop Evidence Retrieval for Cross-document Relation Extraction* — Findings of ACL 2023, `2023.findings-acl.657` ✅
6. **PILOT**: *Explore the Way: Exploring Reasoning Path by Bridging Entities for Effective Cross-Document Relation Extraction* — Findings of EMNLP 2023, `2023.findings-emnlp.450` ✅
7. **CLEAR**: *Cross-Document Link-Enhanced Attention for Relation Extraction with Relation-Aware Context Filtering* — Applied Sciences 15(13):7435 ✅
8. **HCRE**: *LLM-based Hierarchical Classification for Cross-Document Relation Extraction with a Prediction-then-Verification Strategy* — Findings of ACL 2026, `2026.findings-acl.1985` ✅
9. **REIC**: *Reward-based Input Construction for Cross-document Relation Extraction* — ACL 2024 long, `2024.acl-long.501` ✅
10. **CoRE-NEPD**: *Towards Better Graph-based Cross-document Relation Extraction via Non-bridge Entity Enhancement and Prediction Debiasing* — Findings of ACL 2024, `2024.findings-acl.38` ✅

**第三优先：决定评测协议**

11. **ScienceAgentBench**: *Toward Rigorous Assessment of Language Agents for Data-Driven Scientific Discovery* — ICLR 2025 ✅
12. **PaperBench**: *Evaluating AI's Ability to Replicate AI Research* — ICML 2025 poster 43586 ✅
13. **MLE-bench**: *Evaluating Machine Learning Agents on Machine Learning Engineering* — ICLR 2025；arXiv:2410.07095 ✅
14. **AI-Researcher** / Scientist-Bench — NeurIPS 2025 main ✅
15. **AutoDiscovery: Open-ended Scientific Discovery via Bayesian Surprise** — NeurIPS 2025 main ✅
16. **MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation** — ICML 2024, PMLR 235:20271–20309 ✅

**第四优先：提供领域模块**

17. **SciKGAT**: *Adapting Open Domain Fact Extraction and Verification to COVID-FACT through In-Domain Language Modeling* — Findings of EMNLP 2020, `2020.findings-emnlp.216` ✅
18. **EDRM**: *Entity-Duet Neural Ranking: Understanding the Role of Knowledge Graph Semantics in Neural Information Retrieval* — ACL 2018, `P18-1223` ✅
19. **ReactXT**: *Understanding Molecular "Reaction-ship" via Reaction-Contextualized Molecule-Text Pretraining* — Findings of ACL 2024, `2024.findings-acl.318` ✅（注意：**NUS 团队**，非清华刘知远组）
20. **Toolink**: *Linking Toolkit Creation and Using through Chain-of-Solving on Open-Source Model* — NAACL 2024 long, `2024.naacl-long.48` ✅

**第五优先：数据集与基础**

21. **CodRED**: *A Cross-Document Relation Extraction Dataset for Acquiring Knowledge in the Wild* — EMNLP 2021 main, `2021.emnlp-main.366` ✅
22. *Entity-centered Cross-document Relation Extraction* — EMNLP 2022 main, `2022.emnlp-main.671` ✅
23. **SciClaim**: *Extracting Fine-Grained Knowledge Graphs of Scientific Claims* — EMNLP 2021 main, `2021.emnlp-main.381` ✅
24. **LGCR**: *Local-to-Global Causal Reasoning for Cross-Document Relation Extraction* — IEEE/CAA JAS 10(7):1608–1621, DOI `10.1109/JAS.2023.123540` ✅
25. **KXDocRE**: *Knowledge-Driven Cross-Document Relation Extraction* — Findings of ACL 2024, `2024.findings-acl.227` ✅
26. *"Who said it, and Why?" Provenance for Natural Language Claims* — ACL 2020, `2020.acl-main.406` ✅
27. *What is Your Article Based On? Inferring Fine-grained Provenance* — ACL-IJCNLP 2021, `2021.acl-long.458` ✅
28. *Denoising Distantly Supervised Open-Domain Question Answering* — ACL 2018, `P18-1161` ✅
29. **PRESTO**: *Progressive Pretraining Enhances Synthetic Chemistry Outcomes* — Findings of EMNLP 2024, `2024.findings-emnlp.597` ✅（IDEA 团队）
30. **ResearchAgent**（NAACL 2025）、**SciMON**（`2024.acl-long.18`）、**KG-CoI**（arXiv:2411.02382）、**PaperScope**（arXiv:2604.11307）、**Graph-PRefLexOR**（arXiv:2607.00924）、**HyGRAIL**（arXiv:2609.02056）、**CrossTrace**（arXiv:2603.28924）
