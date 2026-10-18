# R1–R5 典型支持链与可靠性评估

日期：2026-10-18
对象：`outputs/question_domain_aircraft/answer_layer.json`（当前 311 claim 冻结图上的 5 个候选回答）
辅助材料：`typical_chains.json`、`domain_view_question.json`、
`weekly_reports/out/domain_graph_aircraft_reviewed/audit.md`、
`outputs/aircraft_domain_merge/run_20261013T093612Z_c7fc9087` 的 Step 3 artifact

> **2026-10-18 typed-edge update（当前图）**
> 本报告的 R1–R5 深链现已被明确归类为 `exploratory_paths`，不再计入 support。
> 当前图有 311 claims / 223 edges，补回 15 条 Step 3 `special_case`/`bounds`
> display links；边角色为 `infer 202 / special_case 13 / bounds 5 / reproduces 3`，
> **deduction 0**，因此 typed support chains 为 **0**。R1–R5 的 direct support
> 仍为 10/8/8/10/9；exploratory paths 为 18/24/24/20/14。下面的链分析仍然成立：
> 它们说明这些路径为什么不能当证据，而不是说它们现在是证据。

## 0. 一句话结论

**这 5 个根作为“问题导航/候选路线”是成立的；作为“证据链/已证结论”不可直接引用。**

- 每个根都有一个可辨认的**单篇核心证据簇**：R1 的 CACDS/ModelCenter、R2 的嵌套 MDO
  与 NASA 结构约束、R3 的 014abc68 DAE 模型、R4 的 066432fe 飞行数据 delta 模型、
  R5 的 26dddb44 联合 HIL。
- 但 **R1–R5 的“多论文支持”很薄**：每个根都是 1 篇主论文 + 1 篇边缘论文；第二篇
  通常只贡献 1 条直接支持。
- **多跳链不是证明链**。Step 3 的 `infer` 表达大量使用 “needs / requires /
  shared object / what follows”，它们是**依赖、使能、实例或类比**关系，不是
  “前提为真则结论为真”的蕴含。当前 chain enumerator 把它们全部当成支持步骤。
- 人工复核（227 图）对同一批原始 claim 的判断更严厉：
  - 本轮 86 个被连接/链上的 claim，只有 **8 个 keep**；**40 个 note、20 个 rewrite、
    16 个 split、2 个 merge**。
  - 46 条直接 support 中，只有 **3 个 keep**（P211/P229/P213，均属 R2）；R1/R3/R4/R5
    的直接支持里**没有 keep**。
  - 人工复核明确写道：**“All 86 original cross-paper edges remain excluded without
    sufficient transfer premises.”** 当前深链恰恰建立在这 86 条跨论文边上。
- 因此分级：
  - **导航可靠性：Medium（B-）**——可以用来发现路线、组织阅读顺序；
  - **证据可靠性：Low（C）**——不能把链或 `primary` 标签当作事实证明；
  - 5 个根中 R2 最弱（两个独立发现被 “while” 拼在一起），R3/R4 的核心较强但跨论文
    部分弱，R1/R5 的核心方法描述强但人工复核把它们大量归为 note。

## 1. 评估方法

### 1.1 链的提取

从 `answer_layer.json` 的 `roots[*].chains` 中，每个根选 2–3 条：

- **direct**：`claim → root` 的单步 support，代表该根最直接的证据；
- **deep**：跨 3 篇以上论文、depth ≥ 4 的链，代表当前 chain enumerator 的最强形态；
- 必要时加一条 **method bridge**（同方法、不同机型/数据的桥接链）。

完整路径、claim 文本、边类型、Step 3 表达、人工复核 action 已写入
`typical_chains.json`，本报告引用其中的编号和原文。

### 1.2 边类型的判定

对每一步同时看三个来源：

1. 当前 view 的 edge label（`support` / `bounds` / `infer`）；
2. 跨论文边的 Step 3 `weakpoint.payload.expression`（原始判断理由）；
3. 人工复核 `audit.md` 对该 claim 的 disposition（keep/rewrite/split/note/merge）。

据此把链步骤分为：

| 类型 | 含义 | 是否能作为支持链步骤 |
| --- | --- | --- |
| **direct support** | answer edge；claim 直接支持 root | 可以，但仍需查 claim 本身 |
| **entailment** | 前提语义上推出结论 | 可以 |
| **independent reproduction** | 独立方法复现同一量 | 弱支持，需同一对象/条件 |
| **enabling / representation** | A 能表示/提供 B 所需的输入 | 不是支持 |
| **dependency / needs** | B 要完成需要 A（A 是必要条件） | 不是支持，只是前置条件 |
| **instance / example** | A 是 B 的一个例子 | 弱支持，不能证明 B 的选择 |
| **analogy / method bridge** | 不同机型/方法之间的类比 | 不是支持 |
| **paper-internal infer** | 单篇论文内部的 infer，当前 IR 无 expression | 需要回原文/正式化策略，不能默认信任 |

### 1.3 分级

- **A / High**：核心证据来自同一对象/条件，直接支持 root 的主要断言，边界清楚；
- **B / Medium**：核心成立，但 root 是跨对象/跨论文的综合，或链不是证明；
- **C / Low**：支持主要是主题相关、依赖、类比，或关键 claim 已被人工复核改写/排除；
- **D / Very Low**：无直接支持或与 root 断言冲突。

本报告分别给：**direct（直接支持）**、**chain（多跳链）**、**source（来源质量）**、
**overall（作为回答的总体可靠性）**。

## 2. 总体结果

### 2.1 结构数字

- 5 个根，50 条 answer edge（45 support + 5 bounds，0 oppose）；
- 46 条 claim 被直接连接，86 条 claim 进入连接或链；
- 12/13 篇论文被触及；`paper_0fc191d1`（18 claims）未被触及；
- 4 条 claim 同时支持两个根：
  - `paper_c6f44cb4::claim_1/5/25`（MDO 架构）→ R1/R2；
  - `paper_26dddb44::claim_4`（实时仿真计算动力学）→ R3/R5。

### 2.2 直接支持的“主论文/边缘论文”

| 根 | 主论文（support 条数） | 边缘论文（support 条数） | bounds |
| --- | --- | --- | --- |
| R1 | `aec49f8f`（7）CACDS/ModelCenter | `c6f44cb4`（3）另一套 MDO 框架 | 0 |
| R2 | `c6f44cb4`（5）嵌套 MDO | `9c2fb427`（3）NASA 结构约束 | 2（9c2fb427） |
| R3 | `014abc68`（7）DAE 模型 | `26dddb44`（1）实时仿真 | 2（014abc68） |
| R4 | `066432fe`（9）delta 气动模型 | `53166ac1`（1）模拟双脉冲参数辨识 | 0 |
| R5 | `26dddb44`（8）联合 HIL | `755906eb`（1）低成本 SITL/HITL | 1（755906eb） |

“2 篇论文”在结构上成立，但**第二篇大多只贡献 1 条直接支持**；R1 的 3 条
`c6f44cb4` 是另一套 MDO 框架，不是 CACDS 的同一系统。把每个根称为
“multi-paper support”会高估跨论文共识。

### 2.3 人工复核口径（关键）

`weekly_reports/out/domain_graph_aircraft_reviewed/audit.md` 的 311 条处置：

```text
note 149 / rewrite 77 / split 43 / merge 16 / keep 26
```

它是把原始 311 图清理成 227 条事实 claim 的人工复核，不是对“方法路线”的复核。
把它作为**来源质量**参照，得到：

| 集合 | keep | note | rewrite | split | merge |
| --- | ---: | ---: | ---: | ---: | ---: |
| 46 条直接 support | 3 | 25 | 8 | 9 | 1 |
| 83 条非根链节点 | 8 | 38 | 19 | 16 | 2 |
| 86 条去重后的连接/链 claim | 8 | 40 | 20 | 16 | 2 |

- 直接 support 中只有 **R2** 拥有 keep 的 claim（P211/P229；P213 是 keep 的 bound）；
  R1/R3/R4/R5 的直接支持**全部**是 note/split/rewrite。
- 这**不等于**这些 claim 都是错的：人工口径把工具能力、方法描述、上下文等归为
  note；对“怎么做仿真设计”这种问题，note 中的方法描述往往正是有用的内容。
  但它意味着：**这些支持没有被复核图承认为事实命题**，而回答层把它们当成了
  “claim → answer” 的证据。
- `split` / `rewrite` 更直接：当前 native 图的 claim 不是最精确的版本，人工复核
  已经把同一条原文拆成多条或改写了措辞；回答层使用的是**未拆/未改写的 native 文本**。

### 2.4 人工复核对跨论文边的判断

审计原文：

> All 86 original cross-paper edges remain excluded without sufficient transfer premises.

这条对回答层是决定性的：R3/R4/R5 的深链、以及 R1/R2 的部分深链，全部依赖这 86 条
跨论文边。人工复核认为这些边**没有足够的迁移前提**，因此没有进入 227 图。
当前回答层的新 answer edge 也没有经过人工复核；它们是 LLM 在冻结候选池里新加的。

## 3. R1–R5 逐根分析

### 3.1 R1 · 集成概念设计环境

**Root**：Aircraft simulation design can be organized around an integrated
conceptual-design and simulation-optimization environment that runs an
end-to-end civil-airplane design case and uses an optimizer to minimize a
design objective such as direct operating cost.

**直接支持（10 条）**：

- 7 条 `aec49f8f`：P15/P36/P37/P41/P46/P47/P57 —— CACDS 的统一
  geometry→analysis→optimization 工作流、CACDS 与 ModelCenter 的 I/O 耦合、
  CJX 案例、DOC 最小化、优化后机翼几何变化。
- 3 条 `c6f44cb4`：P59/P63/P81 —— 另一套模块化 MDO 框架、Level 2 配置/翼盒优化。
- 人工复核：P41/P15/P36/P37/P57/P59/P63/P81 = note；P46/P47 = split。
  **直接支持没有 keep。**

**Chain A（direct core）：P36 → R1**

- P36：CACDS 以功能组件方式与 ModelCenter 耦合，通过定义好的输入/输出文件把
  ModelCenter 作为优化引擎。
- 边：answer support；理由 “CACDS is coupled with ModelCenter as optimization
  engine via defined inputs/outputs”。
- 评价：**强支持 “simulation-optimization environment” 这一半**；它说明 CACDS
  可以被外部优化器驱动，但不说明“端到端案例”或“DOC 最小化”本身。

**Chain B（direct case）：P41 → R1**

- P41：CJX 150 座单通道飞机案例在 CACDS 中端到端跑通。
- 边：answer support。
- 评价：直接支持 “runs an end-to-end civil-airplane design case”。
  但 P41 在人工复核中是 **note**，更像案例总结而不是原子事实；它不能独立证明
  “端到端”是 CACDS 的通用能力，只能证明该案例被跑过。

**Chain C（deep）：P17 → P81 → P63 → R1**

| 步骤 | 内容 | 边类型（依据） | 评价 |
| --- | --- | --- | --- |
| P17 → P81 | P17：CACDS 支持多段翼/机身/尾翼等建模；P81：MDO Level 2 优化翼盒布局、翼尖 span、tip chord | **enabling / representation**（Step3：CACDS 的多段翼建模“can represent” Level 2 要优化的参数） | 只说明 CACDS 能表示这些几何参数，不说明 CACDS 支持该 MDO 框架，也不说明两者被集成 |
| P81 → P63 | P81：Level 2 优化翼盒布局/tip chord；P63：Level 2 优化 configuration 几何变量 | **paper-internal infer，无 expression**；两条是并列的 Level 2 活动（翼盒 vs 构型），不是前提→结论 | 不能作为支持步骤；更像同一论文内的层级/并列关系 |
| P63 → R1 | Level 2 优化 configuration 变量，带目标函数/约束 | **direct support**（answer edge） | 只支持“存在一个优化环境”，不支持 CACDS 的端到端案例 |

**R1 可靠性判断**

- Direct：**A-/B**。CACDS 侧的 P36/P37/P41/P46/P47/P57 合起来确实支持
  “CACDS + ModelCenter 的集成环境与 CJX/DOC 案例”；但它们是 note/split，
  且 `c6f44cb4` 的 3 条是另一套 MDO 框架，不是同一个系统。
- Chain：**C**。P17→P81 是 representational；P81→P63 无 expression 且是并列
  活动；整条链没有把 CACDS 和 MDO 框架连接起来，反而把两套系统并置。
- Source：**C+**。0 keep，8 note，2 split；人工复核不承认这些为事实 claim。
- **Overall：B / Medium**。作为“CACDS 案例 + 外部优化器”的路线描述可信；
  作为“跨两个框架的统一集成环境”则证据不足。

### 3.2 R2 · 嵌套多学科优化

**Root**：Conceptual aircraft design can be pursued through a unified modular
MDO framework with nested optimization levels that optimize configuration and
wing-box design variables, while structural and static-margin constraints may
materially affect the resulting design.

**直接支持（8 + 2 bounds）**：

- `c6f44cb4`（5）：P59/P61/P62/P63/P81 —— 模块化 MDO 框架、两个嵌套层级、
  Level 1/2 的优化对象。
- `9c2fb427`（3 + 2 bounds）：P211/P215/P229 + bounds P213/P216 —— NASA
  lift-plus-cruise 的 plate buckling、8% static margin 导致约 6% 重量增加；
  bounds：位移约束不活跃、原问题省略 stress/displacement 约束。
- 人工复核：P61/P59/P63/P81/P62/P215 = note；P211/P229 = keep；P213 = keep；
  P216 = note。**R2 是唯一有 keep 直接支持的根。**

**Chain A（direct architecture）：P61 → R2**

- P61：该 MDO 框架支持两个嵌套优化层级。
- 边：answer support；理由 “matching the answer's core architecture”。
- 评价：直接支持第一句（nested levels），但 P61 是 **note**；人工复核不认为它是
  事实 claim。

**Chain B（direct constraint）：P211 → R2**

- P211：在 +3g/−1g 静载工况下，NASA lift-plus-cruise 的翼盒主要由 plate
  buckling 约束决定尺寸。
- 边：answer support。
- 评价：直接支持 “structural constraints may materially affect” 的
  **结构约束**部分；这是 R2 里最硬的 keep claim。它来自另一套 NASA MDO 问题，
  不是 `c6f44cb4` 的嵌套框架。

**Chain C（deep）：P17 → P81 → P23 → P229 → R2**

| 步骤 | 内容 | 边类型（依据） | 评价 |
| --- | --- | --- | --- |
| P17 → P81 | CACDS 多段翼建模 → MDO Level 2 翼盒参数 | **enabling / representation** | 只说明参数可表示 |
| P81 → P23 | Level 2 翼盒/构型变量 → CACDS 几何参数自动传给重量/CoG | **dependency / needs**（Step3：Level 2 评估重量/CoG “requires” 这种自动传递） | 前置条件，不是证据 |
| P23 → P229 | 重量/CoG 自动计算 → NASA 静稳定约束使重量增加约 6% | **dependency / needs**（Step3：约束驱动的重量最小化 “needs” 重量计算） | 前置条件，不是证据；而且 P23 是 CACDS，P229 是 NASA 的另一套 MDO 问题 |
| P229 → R2 | 静稳定约束使重量增加约 6% | **direct support**（answer edge） | 只支持“结构/静稳定约束会影响设计”，不支持“该嵌套 MDO 框架使用了这些约束” |

**R2 可靠性判断**

- Direct：两个分句分别 **A-/B**：嵌套 MDO 由 `c6f44cb4` 支持（但均为 note）；
  结构/静稳定约束由 `9c2fb427` 的 keep claims + bounds 支持。
- Composite：**C+**。root 用 “while” 把两套独立的 MDO 问题拼成一个方法路线；
  没有任何 claim 表明 `c6f44cb4` 的框架实际使用了 plate buckling / static margin。
- Chain：**C**。P81→P23、P23→P229 都是 “needs”，不是支持；整条链是依赖路径。
- Source：**C+**。2 个 keep 直接支持，但 6 个 note；人工复核把嵌套 MDO 架构判为
  note，把 NASA 结构约束判为 keep。
- **Overall：C+ / Medium-Low**。作为“两条设计原则的并列”可信；作为“一个统一的
  嵌套 MDO 方法”证据不足。

### 3.3 R3 · 非线性飞行动力学建模

**Root**：Aircraft simulation can be formulated as a coupled nonlinear
differential-algebraic flight-dynamics model with six-degree-of-freedom
equations expressed in body or wind axes, requiring state variables and
parameters to be specified.

**直接支持（8 + 2 bounds）**：

- `014abc68`（7）：P178（35 方程 DAE，39 变量，30 参数）、P181（正仿真输入）、
  P183（体轴角速度—欧拉角）、P184/P185/P186（风轴标量方程）、P187（体轴转动
  动力学）。
- `26dddb44`（1）：P164（HIL 地面计算机每周期计算动力学微分方程）。
- bounds：P182（逆仿真输入）、P197（左右对称飞机的化简）。
- 人工复核：P178/P183 = split；P185/P184/P186/P187 = rewrite；P181/P164 = note；
  bounds P182/P197 = note/rewrite。**0 keep。**

**Chain A（direct DAE）：P178 → R3**

- P178：模型是单个耦合非线性 DAE，35 个标量方程、39 个飞行变量、30 个独立常数
  参数。
- 边：answer support。
- 评价：直接、精确地支持 root 的 “coupled nonlinear DAE … requiring state
  variables and parameters”。这是 R3 最硬的直接支持；但 P178 是 **split**，
  native 版本可能混合了多个事实，人工复核拆分了它。

**Chain B（direct kinematics）：P183 → R3**

- P183：体轴角速度 (p,q,r) 由欧拉角及其导数显式给出。
- 边：answer support。
- 评价：支持 “six-degree-of-freedom equations expressed in body axes” 的
  一部分；P183 是 **split**，同样需要回看拆分后的版本。

**Chain C（deep）：P191 → P251 → P204 → P126 → P185 → P164 → R3**

| 步骤 | 内容 | 边类型（依据） | 评价 |
| --- | --- | --- | --- |
| P191 → P251 | 任意高度空气密度 → 推力系数定义 C_T=T/(ρn²D⁴) | **dependency / needs**（Step3：C_T 在给定高度求值 “needs” 高度相关 ρ） | 前置输入，不是证据 |
| P251 → P204 | 推力分量 C_Tx,prop → 升降舵双脉冲参数辨识 C_x | **dependency / needs**（Step3：辨识 C_x 需要把推进分量从气动 x 力中分离） | 前置条件，不是证据；P251 是 CZ-150 螺旋桨，P204 是另一架 UAV |
| P204 → P126 | UAV 模拟双脉冲参数辨识 → 450 kg 滑翔机 delta 气动模型 | **independent reproduction**（Step3：独立程序+残差说明该系数集可从飞行数据复现） | 弱支持，跨机型/数据；不能证明 450 kg 模型本身 |
| P126 → P185 | delta 模型复现 C_X/C_Z/C_m → 风轴 β 方程使用 C_x/C_z | **enabling**（Step3：好 fidelity 使系数“reliable for”方程中的 C_x/C_z） | 说明系数可用于方程，不推出方程本身 |
| P185 → P164 | 一条风轴 β 方程 → HIL 计算机每周期计算动力学方程 | **instance / example**（Step3：这条方程是目标计算的一个具体例子） | 例子关系；HIL 可能使用别的模型 |
| P164 → R3 | HIL 实时计算动力学 | **direct support**（answer edge） | 只支持“仿真计算动力学”，不支持 DAE/6-DOF 的具体选择 |

**R3 可靠性判断**

- Direct：**A-/B**。`014abc68` 的 7 条直接支持确实覆盖 DAE、体/风轴方程、
  状态变量与参数；但它们是 split/rewrite，且 P164 只是通用实时仿真 note。
- Cross-paper：**C**。P164 不证明 DAE 形式；root 的“跨论文”部分实际上是
  “一个 DAE 模型 + 一个 HIL 应用”，不是两个独立来源对同一方法的验证。
- Chain：**C**。这是一条依赖/实例链，跨 5 篇论文、4 种机型/条件；不能当证明。
- Source：**C+**。0 keep，2 split，4 rewrite，2 note；人工复核没有保留这些 claim。
- **Overall：B- / Medium**。作为“014abc68 的 DAE 建模路线”可信；
  作为“跨论文的完整仿真设计答案”证据不足。它更像一个**建模组件**，不是完整路线。

### 3.4 R4 · 飞行数据模型校正

**Root**：Simulation-model fidelity can be improved by identifying aerodynamic
delta or parameter models from flight-test or simulated doublet data and
updating the original aerodynamic database, followed by checking predictions
against measured flight parameters.

**直接支持（10 条）**：

- `066432fe`（9）：P122（delta model for C_X/C_Z/C_m）、P123/P124/P125
  （ΔC_X/ΔC_Z/ΔC_m 参数与置信区间）、P126（模型复现系数）、P127（更新数据库后
  预测显著改善）、P128（三次独立飞行测试验证）、P129（更新模型贴近实测 α/θ/
  altitude）、P130（原模型偏离实测，构成校正动机）。
- `53166ac1`（1）：P204（固定翼 UAV 的模拟升降舵双脉冲参数辨识，复现
  C_x/C_z/C_m 与残差）。
- 人工复核：P122/P128/P204 = note；P123/P127 = rewrite；P124/P125/P126/P129/P130
  = split。**0 keep。**

**Chain A（direct delta model）：P127 → R4**

- P127：原始气动数据库 + 估计的 delta model 的更新仿真，显著改善飞行参数预测。
- 边：answer support。
- 评价：直接支持 root 的 “updating the original database → improved predictions”；
  是 R4 最硬的核心支持。但 P127 是 **rewrite**，native 文本不是人工改写后的版本。

**Chain B（direct parameter ID）：P123 → R4**

- P123：ΔC_X 的估计参数与 95% 置信区间（α³ 项 57.120 ± 1.136，静态项
  −0.034 ± 0.004）。
- 边：answer support。
- 评价：支持 “identifying aerodynamic delta models”；本身不覆盖数据库更新或
  飞行验证。P123 是 **rewrite**。

**Chain C（deep dependency）：P191 → P251 → P204 → P126 → R4**

- 与 R3 的前四步相同：P191→P251 是 ρ 的依赖，P251→P204 是推进分量的依赖，
  P204→P126 是跨机型的独立复现（弱支持），P126→R4 是直接支持。
- 评价：这条链不能证明 R4 的 “update database + check against measured flight
  parameters”；它只把“参数辨识”这一前提交给 P126。P204 是另一架 UAV 的模拟
  数据，不是 450 kg 滑翔机的飞行数据；把它当作 R4 的跨论文证据会过度概括。

**Chain D（method bridge）：P116 → P115 → P256 → P125 → P122 → R4**

| 步骤 | 内容 | 边类型（依据） | 评价 |
| --- | --- | --- | --- |
| P116 → P115 | HCPAV 悬停纯偏航仿真指令 → 前后旋翼舵面反向偏转实现纯偏航 | paper-internal infer，无 expression | 同论文内的仿真观察，可能成立；但无正式化理由 |
| P115 → P256 | HCPAV 的推进偏航力矩 → CZ-150 的 C_n 保留推进项 | **analogy / generic justification**（Step3：推进能产生偏航力矩，支持 C_n 保留推进项） | 不同机型；只是“推进项可能需要保留”的类比 |
| P256 → P125 | CZ-150 的 C_m 控制面交互项 → 450 kg 滑翔机的 ΔC_m 舵面项 | **analogy**（Step3：保留 δ_e 交互项“supports keeping/interpreting”显著项） | 不同机型/数据；不能证明 450 kg 模型的估计 |
| P125 → P122 | ΔC_m 参数 → 完整 delta 模型 | paper-internal infer，无 expression | 参数→模型的聚合，可能成立 |
| P122 → R4 | delta 模型用于更新仿真 | direct support | 支持 R4 的核心 |

**R4 可靠性判断**

- Direct：**A**。`066432fe` 的证据簇是五个根中最连贯的：模型估计、参数、数据库
  更新、独立飞行测试验证、原模型偏差动机；P127/P128/P129 形成完整的验证链。
- Cross-paper：**B-**。P204 支持“参数辨识”这一泛化，但不支持“更新原数据库 +
  用实测飞行参数验证”；P204 是不同 UAV、模拟数据、无数据库更新。
- Chain：**C+**。P204→P126 是唯一有点支持意味的“独立复现”，但跨机型；
  P191/P251 段是依赖；P116/P115 段是方法类比。
- Source：**C+**。0 keep，5 split，2 rewrite，3 note；人工复核把很多核心 claim
  拆分/改写/归为 note。
- **Overall：B / Medium**。核心飞行数据校正路线可信；把它泛化为“flight-test or
  simulated doublet data”并覆盖数据库更新/飞行验证，证据只来自 066432fe 一侧。

### 3.5 R5 · 硬件在环验证

**Root**：Before flight test, aircraft simulation design can be validated
through real-time hardware-in-the-loop testing in which a ground computer
solves the aircraft dynamics and the integrated control and propulsion systems
are exercised under specified tracking-error criteria.

**直接支持（9 + 1 bound）**：

- `26dddb44`（8）：P162（闭环 HIL 平台，集飞控/推进/仿真设备/数据库）、
  P163（硬件组成）、P164（地面计算机每周期计算动力学）、P172（姿态稳定）、
  P173（轨迹跟踪）、P174（发动机燃油流量在允许误差内）、P175（飞控/推进
  一体化有效）、P176（系统稳定可靠，可作为飞前测试平台）。
- `755906eb`（1）：P158（低成本 SITL/HITL 工作流，HITL 与真实飞行足够一致，
  可用于快速控制器开发/调参/航电验证）。
- bound：P159（HITL 与真实飞行差异约 10%）。
- 人工复核：P162/P163/P164/P172/P173/P175/P176/P158 = note；P174 = rewrite；
  P159 = merge。**0 keep。**

**Chain A（direct platform）：P162 → R5**

- P162：联合实时 HIL 系统被设计为闭环平台，集成飞控、推进、仿真设备和数据库。
- 边：answer support。
- 评价：直接支持 root 的 “real-time HIL … integrated control and propulsion”；
  是 R5 最核心的直接支持。但 P162 是 **note**，人工复核不认为它是事实 claim。

**Chain B（direct workflow）：P158 → R5**

- P158：低成本 SITL/HITL 工作流，HITL 与真实飞行足够一致，可用于快速控制器
  开发、调参、航电验证和初步失效检查。
- 边：answer support。
- 评价：支持 “HIL 可以用于验证” 的泛化；但它不是 `26dddb44` 的联合实时
  飞控/推进平台，也没有 root 说的 “specified tracking-error criteria”。
  P158 是 **note**。

**Chain C（deep）：P191 → P251 → P204 → P126 → P185 → P164 → R5**

- 与 R3 的深链完全相同，只把最后一步改为 P164→R5。
- 评价：前半段是 ρ/C_x/参数辨识的依赖与类比，P185→P164 是“一条风轴方程是
  HIL 计算的一个例子”，P164→R5 才是 HIL 支持。整条链不能证明 R5 的联合
  飞控/推进/误差判据；它只是把 HIL 之前的气动建模链串了起来。
- P164 是 note；P159 的 10% 差异是 R5 最重要的边界，但它在链外，容易被忽略。

**R5 可靠性判断**

- Direct：**A-/B**。`26dddb44` 的 8 条直接支持确实覆盖平台、硬件、实时计算、
  姿态/轨迹/燃油流量结果、飞控/推进一体化和飞前测试结论；但全部是 note，
  人工复核没有把它们作为事实 claim 保留。
- Cross-paper：**C+**。P158 是另一套低成本 SITL/HITL 工作流，不是同一平台；
  它支持“HIL 方法”但不支持“联合实时飞控/推进 + 指定误差判据”。
- Chain：**C**。依赖/实例链，不能证明 R5。
- Source：**C**。0 keep，8 note，1 rewrite；人工复核的 R5 是 `755906eb` 的
  `claim_5_part2`（Minion HITL 行为），不是 `26dddb44` 的这组 HIL claims。
- **Overall：B- / Medium-Low**。作为“26dddb44 的联合 HIL 方法描述”可信；
  作为“跨论文、带误差判据的验证答案”，证据和复核支持都弱；10% 差异必须保留。

## 4. 链级问题（比单根问题更重要）

### 4.1 `infer` 不是 entailment

当前图的 `infer` 是 Step 3/4 的通用标签。原始 Step 3 expression 明确显示它们是：

- **dependency / needs**：
  - P17→P81：“CACDS can represent … that Level 2 optimizes.”
  - P81→P23：“evaluating weight/CoG … requires claim_23's automatic transfer.”
  - P23→P229：“claim_27 needs claim_23's weight calculation.”
  - P191→P251：“the conclusion needs an altitude-dependent ρ.”
  - P251→P204：“claim_6's C_x identification needs claim_4's propulsion term.”
  - P126→P185：“coefficients are reliable for the C_x/C_z terms” (enabling).
- **instance / example**：
  - P185→P164：“the evidence supplies one specific aircraft dynamics differential
    equation that the target's computer computes.”
- **analogy / method bridge**：
  - P115→P256：HCPAV 的偏航力矩 → CZ-150 的 C_n 推进项。
  - P256→P125：CZ-150 的 C_m 交互项 → 450 kg 滑翔机的 ΔC_m 舵面项。
- **independent reproduction**（唯一接近支持的类型）：
  - P204→P126：独立参数辨识程序复现同一组 C_x/C_z/C_m；但跨 UAV/滑翔机，
    且 P204 是 note。
- **paper-internal infer，无 expression**：
  - P81→P63、P116→P115、P125→P122。当前 IR 没有说明这些边为什么成立；
    不能默认它们是真蕴含。

结论：**深链是“图上的可达路径”，不是“证据上的推理链”**。把 chain 数量
（22/26/27/24/19）或 max depth（4/5/9/6/7）当作支持强度会严重高估。

### 4.2 R3/R4/R5 共用同一条“气动参数”脊柱

P191→P251→P204→P126（→P185→P164）同时出现在 R3/R4/R5 的深链里：

- P191 是任意高度空气密度；
- P251 是 CZ-150 螺旋桨推力系数；
- P204 是另一架 UAV 的模拟双脉冲参数辨识；
- P126 是 450 kg 滑翔机的 delta 气动模型；
- P185 是 014abc68 的风轴方程；
- P164 是 26dddb44 的 HIL 计算机。

这条脊柱跨 5 篇论文、至少 4 种对象/条件。它把“空气密度→推力系数→参数辨识→
delta 模型→方程→HIL”串成一条可达路径，但没有任何一步是“前一步为真则后一步
为真”。它不能同时作为 R3/R4/R5 的支持链；同一组节点被三个根共享，恰好说明
chain enumerator 在利用图的连通性，而不是在验证三个不同的回答。

### 4.3 直接支持也有“跨对象泛化”

- R3：P164 支持“HIL 计算动力学”，不支持 DAE/6-DOF 的具体选择；
- R4：P204 支持“模拟双脉冲参数辨识”，不支持“更新原数据库 + 用实测飞行参数
  验证”；
- R5：P158 支持“低成本 SITL/HITL 工作流”，不支持“联合实时飞控/推进平台 +
  指定误差判据”；
- R1：P59/P63/P81 支持“另一套 MDO 框架”，不支持 CACDS 的端到端案例；
- R2：P211/P215/P229 支持“NASA MDO 的结构/静稳定约束”，不支持“c6f44cb4 的
  嵌套 MDO 框架实际使用了这些约束”。

这些是 root 语句“把两篇论文的结论拼成一个方法路线”的典型过度概括。
直接 support 的 LLM reason 往往只说“supports X”，没有说明“X 与 root 的
其余部分是否同一对象/条件”。

### 4.4 “primary”标签只反映计数

当前 verdict 规则：

- `contested`：有 oppose；
- `primary`：support ≥3 且跨 ≥2 篇论文；
- `partial`：有 support 但未达 multi-paper；
- `exploratory`：无 support。

5 个根都是 `primary`，因为每个根都有 2 篇论文、≥8 条 support。但这个规则：

- 不区分主论文 7–9 条 vs 边缘论文 1–3 条；
- 不区分 support claim 是 keep/note/split/rewrite；
- 不区分 direct support 与深链；
- 不要求 oppose 存在；
- 不要求同一对象/条件。

所以 `primary` 应读作“**至少两个论文簇的 claim 被 LLM 连到了这个根**”，
而不是“多论文已证实”。

### 4.5 人工复核的 86 条跨论文边

审计原文：

> All 86 original cross-paper edges remain excluded without sufficient transfer premises.

这意味着：

- 深链的跨论文步骤在人工复核中**全部被排除**；
- 回答层的新 answer edge 是 LLM 新加的，不在人工复核范围内；
- R1–R5 的“跨论文”不是图已经承认的跨论文推理，而是回答层在 native 图上
  重新做的连接。

如果把“人工复核 227 图”当作当前最可信的领域图，那么回答层的跨论文部分
**没有被该图支持**；它需要重新论证迁移前提，而不是直接引用。

## 5. 可靠性分级

评分口径：

- **direct**：直接 support 是否覆盖 root 的主要断言；
- **chain**：多跳链是否由蕴含/独立复现组成；
- **source**：support/链 claim 在人工复核中的 keep/note/rewrite/split；
- **overall**：作为“候选回答/导航”的总体可靠性（不是科学真值）。

| 根 | direct | chain | source | overall | 一句话理由 |
| --- | --- | --- | --- | --- | --- |
| R1 | A-/B | C | C+ | **B / Medium** | CACDS 核心强；与 c6f44cb4 的另一套 MDO 是并置，深链是 representational + 并列 |
| R2 | A-/B（分句） | C | C+ | **C+ / Medium-Low** | 嵌套 MDO 与 NASA 结构约束各自成立，但 “while” 没有集成证据；深链是 needs |
| R3 | A-/B | C | C+ | **B- / Medium** | 014abc68 的 DAE 核心强；P164 只是通用 HIL 应用，深链是依赖/实例 |
| R4 | A | C+ | C+ | **B / Medium** | 066432fe 的 delta 校正 + 独立飞行验证最连贯；P204 是相关方法，不是数据库更新/飞行验证 |
| R5 | A-/B | C | C | **B- / Medium-Low** | 26dddb44 的联合 HIL 描述强但全是 note；P158 是另一套低成本工作流；10% 差异是硬边界 |
| **整个回答层** | — | — | — | **B- 导航 / C 证据** | 可用于发现问题路线和组织阅读，不可当作已证结论 |

补充：

- **R4 是五个根里最可靠的**：核心 claim 簇最完整，且有人工复核的独立飞行测试
  验证内容（虽然 native 版本被 split/rewrite）。
- **R2 是最弱的**：它的两个分句分别成立，但 root 把两套不同的 MDO 问题写成
  一个方法；人工复核只保留了结构约束侧（P211/P229），没有保留嵌套 MDO 架构侧。
- **R3 的“模型形式”本身可靠**，但作为“完成飞机仿真设计”的答案是过度收窄的：
  它是一个建模组件，不是完整流程；P164 不能补足流程。
- **R5 的“方法描述”可靠**，但人工复核的 R5 是另一条 Minion HITL 行为 claim；
  `26dddb44` 的 8 条 HIL 证据在复核图中是 note，不是事实 claim。
- **R1 的“CACDS 案例”可靠**，但把它和 c6f44cb4 的 MDO 框架合成一个“统一环境”
  没有直接证据；P59/P63/P81 是另一套框架。

## 6. 建议

### 6.1 对当前回答层（不改数据也能做）

1. **给链步骤加语义标签**：把 `dependency / enabling / instance / analogy /
   independent reproduction / entailment` 显示在 viewer 的链上；当前只有
   `infer`，读者会把依赖当证据。
2. **过滤深链**：chain 视图默认只显示 `depth=2` 的直接 support + 明确蕴含的
   `depth=3`；`depth≥4` 折叠为“依赖路径/探索路径”，不计入 support 统计。
3. **root 详情显示 Step 3 expression**：当前详情只显示 answer reason，没有显示
   每一步的 Step3 理由；把 “needs/requires” 暴露出来，读者能立即判断。
4. **verdict 增加依据列**：`primary` 旁标注“主论文 7 / 边缘论文 3 / keep 0 /
   note 8”，不要只显示“2 papers”。
5. **把 10% HIL 差异、NASA 省略 stress/displacement、R3 逆仿真/对称飞机条件
   提升到 root 级别**，不要只放在节点详情或 bounds 里。

### 6.2 对图/管线

1. **在复核后的 227 图上重跑回答层**，或在完整 `pipeline.question.json` 中先
   生成 K 层，再让回答连 K；不要用 native 311 图的 note/split 文本冒充 claim。
2. **人工确认 5 个根的直接 support**：尤其 R3/R4/R5 的第二篇论文，只保留
   同一对象/条件能覆盖的 claim；否则把 root 改成“单论文方法路线”。
3. **把 R2 拆成两个根**：`嵌套 MDO 框架` 与 `结构/静稳定约束`；或者补一条
   真正连接两者的集成 claim，再合成一个根。
4. **R1 拆成 CACDS 案例路线与 MDO 框架路线**；R4 把 P204 标为“相关工作”，
   不进入“数据库更新 + 飞行验证”的直接 support；R5 把 26dddb44 联合 HIL 与
   755906eb 低成本 SITL/HITL 分开。
5. **补 oppose/contradiction**：当前 0 oppose；没有反对证据的 `primary` 不应
   被读成“无争议”。
6. **原文核验**：当前来源多为 LKM 图节点文本；人工复核 README 已注明这一点。
   对 R1/R2/R4/R5 的核心 claim，至少回原论文页核验一次。

### 6.3 如果只做一件事

**把“链”从支持统计里拆出去。** 目前 36.5% 覆盖率、22–27 条链、depth 9 这些
数字会让人以为回答层有很深的证据；实际上多跳链主要是依赖/类比。把 direct
support 与 chain 分开统计，并把 chain 标注为“探索路径”，是成本最低、收益最大的
修正。

## 7. 相关文件

- `typical_chains.json`：R1–R5 典型链的路径、claim 文本、边类型、Step 3 expression、
  人工复核 action。
- `answer_layer.json`：原始 roots/links/chains。
- `weekly_reports/out/domain_graph_aircraft_reviewed/audit.md`：311→227 人工复核
  的逐条处置。
- `outputs/aircraft_domain_merge/run_20261013T093612Z_c7fc9087`：
  Step 3 proposals（weakpoint expression）与 Step 4/5 artifacts。
- `domain_viewer_question.html`：当前 viewer；链节点详情只显示 direct reason，
  不显示 Step 3 expression。
