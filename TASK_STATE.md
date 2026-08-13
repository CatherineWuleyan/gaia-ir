# 任务状态：证据台账与 HTML 可视化

状态：本轮已完成并可交付  
保存日期：2026-08-10  
工作目录：`/Users/catherinewu/Documents/gaia-ir`

## 已完成

1. 已生成证据台账工作簿：
   - `outputs/evidence-ledger-step1-20260810/evidence_ledger_step1.xlsx`
   - 共 89 条命题、7 个规范结论组、38 条关系。
   - 命题构成：23 条 clean claims、7 个粗粒度结论、8 个 highlight、5 个 weakpoint、7 个问题、39 个推理步骤。
   - 所有记录均包含稳定 ID、类型、论文章节、图表、OCR 行号、原文锚点、支持/限制对象和正式 IR 状态。

2. 已保留原始 HTML：
   - `02_lkm_reference/lkm_reasoning_graph_visualizer.html`

3. 已基于原始 HTML 和 vis-network 新建证据台账可视化：
   - `02_lkm_reference/evidence_ledger_visualizer.html`
   - HTML 已内嵌工作簿中的 89 条台账记录、7 个结论映射、38 条关系及原始 graph 数据，因此可作为单文件使用。

4. 已实现的交互：
   - “证据台账（细化视图）”与“原始 graph（粗粒度视图）”切换。
   - 按 C01—C07 结论组筛选。
   - 按命题、图号、章节或 ID 检索。
   - 分层显示主结论、clean claims、highlight、weakpoint、问题和推理步骤。
   - 推理步骤默认折叠，可勾选后展开 ↝ 子网络。
   - 点击节点查看稳定 ID、命题文本、认识状态、论文位置、原文锚点、支持/限制对象、来源文件、JSON Path 和正式 IR 状态。

## 已完成的运行检查

- 默认 C01 证据概览：11 个节点、13 条关系。
- C06 + `Fig. A1` 检索：9 个节点、12 条关系。
- C06 勾选推理步骤后：14 个节点、17 条关系。
- 已点击并核对 `OTWTA-WP-P05` 详情面板，来源、原文锚点和限制对象显示正常。
- 已验证可切换到原始粗粒度 graph。
- 首轮页面检查未发现浏览器脚本错误或警告。

## 重新连接后的最终复核

- 已复核原始粗粒度全图：27 个节点、7 条关系。
- 已确认检索框可由“重置筛选”清空并恢复默认 C01。
- 已在 736px 与 360px 视口检查首次加载、筛选和推理步骤展开：无横向溢出；图谱、控件和指标均可用。
- 360px 下 C06 展开推理步骤后仍为 14 个节点、17 条关系。
- 最终 HTML 为 311 KB，不包含 `fetch`、XHR 或 WebSocket，数据已全部内嵌。

## 后续语义工作

从 C01—C07 中选择一个结论组，把 weakpoint 与推理步骤进一步改写成可进入 Gaia IR 的显式子网络。当前工作完成了证据可追溯性整理和已有细化内容的可视化，但没有凭空增加新的论文事实，也尚未完成下一轮 IR formalization。

## 中文化更新（2026-08-10）

- 已为 7 个粗粒度结论、8 个实验观察、5 个 weakpoint、7 个研究问题和 39 个推理步骤补齐中文工作文本。
- Evidence_Ledger 的 89 条记录均已有中文工作文本；英文源文本仍保留用于审计。
- 新增“论文原文锚点_中文译文”列，89 条记录均已填写。
- Conclusion_Map 的标题、粗粒度结论和 LKM 评述已改为中文。
- Raw_Graph_Nodes 新增 Title_ZH、Content_ZH 和 Steps_ZH_JSON，不覆盖英文原始数据。
- 已同步重建 evidence_ledger_visualizer.html，默认 C01 及结论筛选显示中文。
- 这次更新只完成翻译与可读性增强，尚未把每个实验系统拆为 S/A/B/M/R/U 和“设置—操作—预测—观测—解释”五层；该工作仍属于步骤二。

## 步骤二：LTH weakpoint 标注

- 已完整读取 `05-formalization-methodology.md` 的三步方法及落体示例。
- 已只读核对廖悦辛关于当前任务的飞书表述：粗图缺少可靠推理类型；本论文主要通过实验进行 abduction/induction；现有 clean claims 缺实验 claim。
- 已生成 `01_paper_sample_one_ticket/lth_coarse_graph_weakpoints.json`：标注 13 个连接级 weakpoint，不填写主观 `(p₁,p₂)`。
- 已生成 `01_paper_sample_one_ticket/LTH_WEAKPOINT_ANNOTATIONS.md`：供人工审阅的汇总清单。
- 已明确：原粗图中的 P1–P5 是限制命题，不是 weakpoint 连接；H1–H8 是证据摘要。
- 下一步应按优先级补实验 claim，然后逐个把这些 `↝` 展开为 abduction/induction 子网络。

## 步骤二：Figure 1 实验 claim 草稿

- 已从 Figure 1 图注、§3.1 正文、通用结果说明和 Appendix A.1 抽取 S/A/B/M/R/U 候选。
- 已生成 `01_paper_sample_one_ticket/figure1_experiment_claims_draft.json`，包含 2 个核心实验 claim 和 2 个可选随机基线 claim。
- 已生成 `01_paper_sample_one_ticket/FIGURE1_EXPERIMENT_CLAIMS_REVIEW.md`，包含 Mermaid 可视化和 6 个待人工确认项。
- 已根据用户定位的中文正文，将 Figure 1a 的观测修正为“作者在所测条件中始终观察到全局剪枝性能高于局部（逐层）剪枝”。
- 已将浅层表达能力机制保留为作者所称的“一种直观解释”，归类为 abduction，而不是实际观测。
- 该初版草稿当时仍不进入正式 IR；其后已按下述“Figure 1 claim 结构修订”重新处理不明确字段。

## Figure 1 claim 结构修订（2026-08-10）

- 已保留“在条件 S 下，方法 A 与基线 B 比较，在指标 M 上观察到结果 R，重复次数或误差为 U”的实验 claim 结构；六个字段共同写进 self-contained observation 节点，不在字段之间创建推理边。
- 已核对实验操作的正文依据：训练—剪枝—延迟重置—复训流程、每轮 20% 量级剪枝、全局/逐层定义、30 轮日程、收敛准确率和六随机种子均有论文锚点。
- 已把随机彩票从 C05 的“全局 vs. 逐层”核心比较移出，仅保留为第三比较臂；后续如研究“是否为中奖彩票”再拆独立 claim。
- 已明确 M 是 metric/measurement；U 是重复次数和不确定性报告。按用户要求，Figure 1b 在 O2 中明确保留 `U=not reported`，但不因此主观调整概率。
- 已删除非正式“解释”边，并将 `OTWTA-WC-09` 拟拆为性能规律 `09A` 与逐层稀疏规律 `09B`，分别使用 induction（重复 abduction）结构展开。
- 已按用户复核意见把作者的浅层表达能力直觉 `OTWTA-WC-10` 恢复为 active causal-attribution weakpoint：作者解释作为 `H_mech`，与层敏感性、跨层幅值分布和优化动态等替代解释共同进入独立 abduction 子网络；Figure 1 的 O1/O2 提供相容性支持，但仍需机制消融来区分解释。

## Pipeline 示例回推（2026-08-11）

- 已修改 Obsidian 文件：`/Users/catherinewu/Library/Mobile Documents/iCloud~md~obsidian/Documents/working/pipeline.md`。
- 已把 LTH Figure 1 的最终 Step 5 细命题图原样保留，并回推补齐 Step 1–4 的累计 Mermaid 图。
- Step 1：来源锚点、raw observation、作者解释摘要和合并粗结论；只保留非正式粗脉络。
- Step 2：O1/O2 补齐 S/A/B/M/R/U，随机彩票移出核心 A/B 比较，观察与作者解释分开。
- Step 3：仅重新识别 induction/abduction 候选类型，不提前展开逻辑结构。
- Step 4：拆分 G1/G2，标出 W09A/W09B/W10、范围限制和替代解释候选，仍保留 ↝。
- Step 5：用 conjunction/disjunction/entailment/equivalence 展开三个 weakpoint；作者直观解释作为竞争性机制假说进入 abduction 子网络。
- 已验证目标文件与暂存版本一致，包含 5 个 Mermaid block、10 个成对代码围栏。
- 覆盖前版本备份：`tmp/pipeline.before-20260811.md`；最终暂存副本：`tmp/pipeline.md`。
