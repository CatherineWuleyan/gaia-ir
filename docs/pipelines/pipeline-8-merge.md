## 1. 要解决什么问题

```text
论文 1 ─→ Pipeline 7.0 ─→ Paper Package 1 ─┐
论文 2 ─→ Pipeline 7.0 ─→ Paper Package 2 ─┼─→ Pipeline 8.0 ─→ 领域图
论文 3 ─→ Pipeline 7.0 ─→ Paper Package 3 ─┘

新论文 ─→ Pipeline 7.0 ─→ New Paper Package ─→ Pipeline 8.0 ─→ 更新后的领域图
```

两条 Pipeline 相互独立：Pipeline 7.0 只忠实表达论文自身，Pipeline 8.0 不回头修改论文图。

## 2. 领域图由什么组成

领域图不是把所有论文依次复制、改名、合并进一个越来越大的 `formalization.json`。

它包含两类知识：

### 论文知识

论文明确提出的命题、实验观察和内部推理，继续由各自的 Paper Package 保存。

### 领域知识

多篇论文共同形成的关系和综合结论，由领域集成层保存。例如：

```text
论文 A：跨模型深度迁移 ── abduction ─┐
论文 B：跨数据集迁移 ─── abduction ─┼─→ 领域结论 K
论文 C：跨 optimizer 迁移 ─ abduction ─┘

K：winning ticket 在多种实验设置变化下表现出迁移能力
```

这个领域结论不属于 A、B、C 中的任何一篇论文，因此不能写回任一 Paper Package，而应由集成层拥有。

最终看到的领域大图，是论文包与领域集成结果共同形成的视图：

```text
Paper Packages + Integration Packages
                  ↓
        Domain Graph / Index / Viewer
```

## 3. 同一条 Pipeline 的三种运行方式

Pipeline 8.0 不为“首次建图”和“新增论文”维护两套逻辑，而是用同一套步骤处理三种场景。

### Bootstrap：首次建立领域

先让所有论文分别完成 Pipeline 7.0，再在同一个领域范围内联合寻找跨论文关系。

首次建图不能简单地按论文顺序滚动合并，应以同一批论文为整体，在相关主题簇中联合判断。

### Incremental：导入新论文

新论文先独立形成 Paper Package，然后只检索它可能影响的领域局部：

```text
New Paper Package
        ↓
召回相关论文节点、领域结论和推理结构
        ↓
判断新论文应接入哪里
        ↓
更新领域图
```

新论文可能：

- 与已有命题等价或矛盾；
- 为已有领域结论增加新证据；
- 扩大或缩小已有结论的适用范围；
- 与其他论文共同形成一个新的领域结论；
- 与当前领域没有足够强的推理关系。

增量模式只处理受影响的局部，不重新运行整个领域。

### Reconcile：局部重整

长期增量更新后，领域图可能出现早期结论过窄、重复综合、漏召回或加入顺序影响。

Reconcile 会选中受影响的局部子图，重新收集其中的原始 Paper Packages，再运行同一套跨论文判断。它不是另一条 Pipeline，而是 Pipeline 8.0 的局部重算模式。

## 4. Pipeline 8.0 主流程

```text
Step 0  确定本次处理范围
   ↓
Step 1  冻结并检查输入
   ↓
Step 2  找到相关的领域局部
   ↓
Step 3  识别跨论文 Operator 和 weakpoint
   ↓
Step 4  展开 weakpoint 并形式化
   ↓
Step 5  校验并发布更新后的领域状态
```

### Step 0：手动选择运行方式

开始运行前，由人选择本次属于：

- Bootstrap：首次建立领域；
- Incremental：导入新论文；
- Reconcile：局部重整。

同时指定本次处理的论文或领域范围。系统不自动推断运行方式，只检查所选模式与输入是否匹配。Step 0 不进行语义判断，也不产生领域 Artifact。

### Step 1：冻结并检查输入

Pipeline 8.0 以经过 Gaia compiler 编译并通过 validator 的 Package 为输入单位。已有领域可以由 Pipeline 7.0 生成，也可以只保留已编译的 Gaia Package；领域集成不要求所有旧 Package 都具有 Pipeline 7.0 的作者侧文件。

每个 Package 按以下层级使用已有 Artifact：

| Artifact           | 要求  | 用途                                                             |
| ------------------ | --- | -------------------------------------------------------------- |
| 已验证的 `gaia.ir`     | 必需  | 领域集成的语义主输入；提供 Knowledge、Operator、Strategy、完整 QID 和跨 Package 引用 |
| 最终 `formalization` | 可选  | 提供 weakpoint、revision、完整 source anchor 等作者侧上下文                 |
| `knowledge.index`  | 可选  | 加速 Step 2 检索；缺失时可由 `gaia.ir` 重建，不作为事实来源                        |

`gaia.ir` 是 Step 1 唯一必需的内容输入。缺少 `formalization` 时，该 Package 仍可参加 Knowledge、Operator 和 Strategy 层面的领域集成，但不能复用其中未编译的 weakpoint 或作者侧工作流信息。缺少 `knowledge.index` 时不阻塞集成，只需在 Step 2 使用现有检索能力从 `gaia.ir` 建立临时检索视图。

复用现有 `inputs/manifest.json`、ArtifactRef 和 SHA-256 冻结输入，不建立另一套输入包装。系统机械检查 `gaia.ir` 的 Artifact 完整性、hash、package identity 和 Gaia validator 结果；可选 Artifact 存在时，再检查其完整性、hash 及其与 `gaia.ir` 的绑定一致性。任一必需检查失败时，该 Package 不进入 Step 2；可选 Artifact 缺失本身不构成失败。

三种模式分别冻结：

- Bootstrap：本次用于首次建域的至少两个已验证 Gaia Packages；
- Incremental：一个确定的当前领域基线，以及本次新增的已验证 Gaia Package；
- Reconcile：一个确定的当前领域基线，以及人工选定的重整范围。

Incremental 中的当前领域基线表示新论文要接入的、由已验证 Gaia Packages 构成的确定领域版本，不表示把全部旧 Package 重新执行一次领域集成。Step 2 只从该基线召回与新论文相关的旧知识局部。当前领域基线具体由什么现有引用标识，留待后续审核；本 Step 不预设新的 Domain Snapshot 文件。

### Step 2：检索领域局部

Step 2 先限定召回范围，再选择检索方法。检索方法不能扩大已经确定的范围：

```text
召回范围 = Step 0 指定的总领域
         ∩ 检索命中的相关领域分区
         ∩ 当前运行方式允许访问的 packages
```

先根据待处理 Paper Package 中的公开 Knowledge，定位相关领域分区；再只在命中分区中检索相关命题。不能先读取所有其他 Paper Packages，再进行全量两两比较。

三种运行方式的查询范围分别是：

- Bootstrap：先将本批论文路由到相关主题分区，再在同一分区内进行跨 package 召回；
- Incremental：只以新增 Paper Package 为查询方，检索当前领域基线及本次其他新增 Paper Packages 中的相关知识，不重新检索旧知识之间的关系；
- Reconcile：只以人工选定的 Knowledge 或局部子图为起点，在指定的重整范围内召回。

公开的 claim、observation claim 和已有领域综合命题可以发起检索。note、形式化辅助节点、空内容接口以及纯 Viewer 派生节点不单独发起检索。Strategy、Operator 和 weakpoint 通过命中的 Knowledge 进入局部子图，而不是作为无边界的独立检索入口。

直接命中集合确定后，只围绕这些 Knowledge 做一跳扩展，包括与它们直接相连的 Operator、Strategy、weakpoint、被直接引用的 Knowledge、背景 note 和 source anchors。扩展得到的新节点不再递归扩展。

召回结果必须跨 Paper Package 或 Integration Package；同一论文内部的关系已经由 Pipeline 7.0 负责。后续一个 weakpoint 可以包含同一论文的多个节点，但整体必须跨越至少两个 package。

领域集成以已批准的 Paper Package 为事实边界。本步骤使用形式化命题及其图上下文，不导入或重新阅读论文全文。命题信息不足时，不通过原文临时补全，也不建立关系；如有必要，应先返回 Pipeline 7.0 修正对应 Paper Package，再重新集成。source anchors 只随结果保留，用于溯源和人工审核。

检索结果只表示“可能相关”，不代表关系已经成立。

### Step 3：识别跨论文结构并建立 weakpoint

Step 3 只处理 Step 2 已冻结的 `groups`，每个 group 独立判断，不建立跨 group 的候选组合，不继续检索，也不回到论文全文。候选只由 group 的 `proposition_qids`、一跳结构及其直接引用构成。判断依据仅限于 Knowledge 的 canonical content、类型、package provenance、已有 Operator、Strategy、weakpoint 和 Step 2 保留的一跳上下文。

Step 3 采用两阶段 LLM 判断，随后进行机械封口：阶段一判断候选是否存在直接成立的确定性 Operator；阶段二对 Operator 候选判定具体类型，对非 Operator 候选判定 weakpoint 类型；最后机械检查 QID、Package 边界、arity、端点、循环、重复和表达式引用。LLM 只能使用冻结 group 中的内容和结构，不能生成新的事实、规则、桥梁、替代解释或 Knowledge identity。证据不足、方向不清或依赖外部常识时，候选直接不写入 Step 3 artifact。

#### 3.1 直接成立的确定性关系

LLM 只有在确认关系可以直接成立时，才将其写入 Integration Package 的 Operator，不建立 weakpoint。`equivalence` 也必须经过语义判断，不能由字符串相等机械决定。

|情况|处理|
|---|---|
|两个命题内容、范围、条件和不确定性一致|`equivalence`|
|两个命题在兼容范围内对同一对象作出不可同时成立的断言|`contradiction`|
|一个现有命题明确是另一个命题的否定|`negation`|
|现有命题之间已经构成明确的合取或析取|`conjunction` / `disjunction`|

等价不修改原始 Package 的 Knowledge identity。若两个等价 Knowledge 的类别相同（`O`、`E` 或 `非OE`），则在 Integration Package 的集成图中机械归并为一个代表节点：保留双方的 source anchors 和 Package provenance，所有入边和出边迁移到代表节点，重复边去重，自环删除。原始 QID 保留为溯源别名，等价关系不作为重复独立证据计数。类别必须来自已有 metadata/type，不能由 LLM 新造。

矛盾必须同时满足：讨论同一对象或变量、比较范围和实验条件兼容、两个断言无法同时为真。不同数据集、模型、时间、指标或实验条件下得到不同结果，不自动构成矛盾；条件不清楚时也不建立 `contradiction`。

如果新论文与已有领域结论矛盾，只增加 contradiction Operator，不立即改写或删除已有结论；是否收缩结论范围留给后续 Reconcile。

#### 3.2 复用 weakpoint 表达有方向的推理

不能直接写成 Operator、但存在明确推理方向的关系，复用 Pipeline 7.0 的 weakpoint：

```json
{
  "evidence_claim_ids": ["..."],
  "target_claim_id": ["..."],
  "reasoning_type": "deduction | abduction | analogy | infer",
  "evidence_anchor_ids": [],
  "expression": "..."
}
```

端点必须来自 Step 2 冻结的局部，或是本步骤创建的 Integration-owned 候选结论。weakpoint 整体必须跨越至少两个 package。

##### 补充 deduction 前提

如果论文 B 已有 `P → C`，但缺少规则或条件 `R`，而论文 A 提供了现成的 `R`，则建立：

```text
P + R ── deduction weakpoint ─→ C
```

`R` 必须是 Step 2 已召回的正式 claim。Step 3 不自行生成缺失规则。

##### 补充 abduction 的替代解释

如果论文 B 使用假说 `H_B` 解释现象 `Obs_B`，论文 A 的 claim `H_A` 也能在兼容条件下解释同一个现象，则 `H_A` 可以作为显式替代解释：

```text
(H_B ∨ H_A) ≡ Obs_B
```

对应 weakpoint：

```json
{
  "evidence_claim_ids": ["Obs_B", "H_A"],
  "target_claim_id": ["H_B"],
  "reasoning_type": "abduction",
  "evidence_anchor_ids": [],
  "expression": "([H_B] 或 [H_A]) 等价 [Obs_B]"
}
```

`expression` 明确区分现象和替代解释，不能仅凭数组位置猜测角色。`H_A` 必须能够解释 `Obs_B`；仅仅与 `H_B` 矛盾、主题相关或来自另一个实验，不足以成为替代解释。

没有显式替代解释时，`evidence_claim_ids` 只保存 `Obs_B`，由 Gaia formalizer 在 Step 4 补齐通用 `AltExp` 接口。

如果同时找到多个已有替代假说，不直接向 abduction 填入三个以上 premises。只有当它们都能解释同一现象时，才可先用现有 `disjunction` Operator 形成一个组合替代解释 claim，再作为 abduction 的第二个 premise；否则保留待审查。

##### 补充 analogy 桥梁

如果一个 package 提供源域规律 `G_src`，另一个 package 提供明确的跨域映射 `M`，并且已有目标域结论 `V_target`，则建立：

```text
G_src + M ── analogy weakpoint ─→ V_target
```

`M` 必须明确说明哪些变量、约束或结构在两个领域间保持成立。只有“两个方法相似”不能建立 analogy；缺少现成 bridge claim 时，Step 3 不自行创造。

##### 明确认定的 infer

只有现有命题之间已经存在明确方向，但无法归入 deduction、abduction 或 analogy 时，才允许建立 `infer` weakpoint。检索相关但无法确定方向的候选不写入 artifact，不能由 `null` 自动转换为 `infer`。

#### 3.3 形成新的领域结论

如果多个命题需要指向尚不存在的领域结论，先复用 group 中或已有领域的 Knowledge。无法复用时，由 Integration Package 建立只供 Step 4 使用的候选 claim `K` 占位；Step 3 不填充其 canonical content，也不把它当作已发布事实。

`K` 占位必须具有稳定 ID、候选类型、来源 QID 和 `placeholder` 标记。Step 4 只有在至少一条指向 `K` 的 weakpoint 成功形式化时，才填充内容并将其纳入发布结果；全部失败时丢弃 `K`。

如果 `K` 只压缩已有命题，没有扩大适用范围，则建立一条有界总结：

```text
A + B + C ── deduction weakpoint ─→ K
```

如果 `K` 是能够解释或预测多个具体实例的一般性假说，则每个实例分别建立一条 abduction weakpoint，共享 `K`：

```text
实例 A ── abduction weakpoint ─┐
实例 B ── abduction weakpoint ─┼─→ K
实例 C ── abduction weakpoint ─┘
```

不同 Paper Package 不自动等于独立证据；复用同一实验、数据或结果的命题不能重复计数。所有共享 `K` 的合格 abduction 可以在索引或 Viewer 中派生为归纳性视图，但不持久化 `induction` Strategy、Induction Group 或额外字段。

#### 3.4 全局不变量：复用、去重与图安全

复用、去重和图安全贯穿候选构造、两阶段 LLM 判断和最终写入，不是独立的处理阶段。Step 3 优先复用已有 Knowledge、领域结论、Paper Package weakpoint、已有 Strategy 的 evidence/target/background、已有 Operator 以及 source anchor；不修改原始 Paper Package。

相同端点、方向、类型和表达式的重复 weakpoint 机械合并；相同 Operator 机械去重。来自同一实验、数据或结果的命题不能作为多份独立 evidence。所有写入结果必须跨越至少两个 Package，不产生自依赖、明显推理环或重复路径。

#### 3.5 不写入条件

以下情况不建立 Operator 或 weakpoint，也不保留单独的 `rejected_relations` 字段：只有关键词或主题相似；对象、条件或指标不兼容；推理方向不清；缺少规则、bridge 或替代解释；需要论文全文或外部常识；会产生自依赖、循环或重复证据；或 LLM 输出无法通过 QID、arity、端点和表达式校验。

Step 3 不建立正式 Strategy，不调用原文，不补造隐藏前提。Step 3 proposal 只包含确定性 Operator、分类后的 weakpoint、必要的 `K` 占位，以及同类 equivalence 的归并映射。


### Step 4：展开 weakpoint 并形式化

Step 4 复用 Pipeline 7.0 的 weakpoint 展开方法，只处理 Step 3 已经分类的 weakpoint。它不再判断哪些领域材料相关，也不回到论文全文寻找新事实。

每个 weakpoint 先机械检查：evidence 和 target 均存在，跨 package 引用可以解析，evidence 与 target 不重合，`expression` 中的 Knowledge ID 与 weakpoint 一致，不产生自依赖、重复 Strategy 或推理环，并且 target 的范围与 evidence、条件兼容。检查通过后，按 `reasoning_type` 写入正式 Strategy。

#### deduction

所有严格推导需要的 claim 必须已经出现在 `evidence_claim_ids` 中：

```text
A₁ ∧ A₂ ∧ ... ∧ Aₖ → C
```

保存 `type: "deduction", premises: [A₁, ..., Aₖ], conclusion: C`。Step 4 不再从原文补规则、条件或中间命题，缺少必要前提时不展开。有界领域总结只有在 premises 严格覆盖 `K` 的全部内容时才能使用 deduction，不得把经验性推广伪装成严格蕴含。

#### abduction

abduction 的接口为：

```text
(H ∨ AltExp) ≡ Obs
```

没有显式替代解释时，保存 `type: "abduction", premises: [Obs], conclusion: H`，由 Gaia official formalizer 补齐通用 `AltExp`。

有显式替代解释时，根据 weakpoint 的 `expression` 确定角色，保存 `type: "abduction", premises: [Obs, AlternativeHypothesis], conclusion: H`，顺序不能互换。

多个实例指向同一 `K` 时，每个实例分别形成一条 abduction Strategy，不合并成多前提 abduction，也不建立 `induction`。

如果 Paper Package 中已经存在相同的 `(type=abduction, observation=Obs, conclusion=H)`，但使用自动 `AltExp`，Integration Package 新增的显式替代解释版本视为对原推理的细化，不作为第二份独立证据重复计算。原 Paper Package 保持不变，领域活动视图优先使用显式替代解释版本。

#### analogy

只有 source law、bridge claim 和 target 均已存在时，才保存 `type: "analogy", premises: [G_src, BridgeClaim], conclusion: V_target`。目标域条件只能复用 Step 2 局部中已有的 note，写入 `background`；缺少 bridge claim 时不创建 Strategy。

#### infer

只有 Step 3 明确分类为 `infer` 的 weakpoint 才能展开。`reasoning_type=null`、检索相关但关系不足、其他类型展开失败或缺少必要前提时，都不能机械转换成 infer。

#### 新 Knowledge 边界

Step 4 不创建新的领域事实。它只能使用 Paper Packages 中已有的 Knowledge、已有 Integration Knowledge、Step 3 创建的候选 `K`，以及 Gaia formalizer 必需的结构型 helper 和 abduction 通用 `AltExp` 接口。缺少其他语义命题时保持未展开，不根据常识或原文临时补齐。

#### weakpoint 生命周期

成功展开后，写入正式 Strategy，并从当前 Step 4 结果中移除对应 weakpoint；原 Step 3 Artifact 保留完整审计来源。

如果一个新建候选 `K` 的所有 weakpoint 都未能展开，`K` 不进入已发布的 Integration Package。如果至少一条指向 `K` 的 Strategy 成功，`K` 与成功的 Strategy 可以进入发布结果；其他失败 weakpoint 仍保留在 Step 3 Artifact 中，不伪装成正式关系。

#### 确定性合并

所有 weakpoint 基于同一个冻结的 Step 3 Artifact 独立处理，最后按冻结顺序确定性汇总。每条 Strategy 使用官方字段 `scope: "local"`、`type`、`premises`、`conclusion` 和 `background`，并由官方 Strategy 模型生成 `strategy_id`。相同 `strategy_id` 机械去重；显式替代解释只细化同一次 abduction，不重复计数；Step 3 已建立的 Operator 保持不变；派生 FormalExpr 和 helper 不回写作者态真源。

### Step 5：校验与发布
复用pipeline_7.0中的编译插件即可。

校验通过后，更新当前领域状态，再由它生成搜索索引和 Viewer。索引和 Viewer 都是派生结果，不是另一份领域真源。

## 5. 复用原则

Pipeline 8.0 优先复用现有能力：

- Pipeline 7.0 的运行、输入冻结、revision、Artifact、校验和 Viewer 机制；
- Gaia 的 package、Knowledge、Operator、Strategy、QID、PackageRef、compiler 和 validator；
- Bohrium LKM v2 和 `gaia search lkm` 的知识、推理链和 Paper Graph 检索；
- 必要时复用 `gaia-research` 和 Paper2tools 的研究路由与 reasoning-chain 聚类。

只有现有工具或字段确实无法表达需求时，才讨论新增合同。检索排名只用于召回，不作为概率、prior 或 Gaia belief。

## 6. 不变原则

无论采用哪种运行方式，都遵守以下原则：

1. 不修改 Pipeline 7.0 生成的 Paper Package。
2. 不因文本相似就合并不同论文的 Knowledge identity。
3. 跨论文综合结论由集成层拥有，并保留到来源论文的追溯路径。
4. 找候选、判断关系、补齐知识和写入 Gaia 分开进行。
5. 能机械判断的事项不用 LLM；证据不足时不猜。
6. 领域图、索引和 Viewer 只有一条可重建的真源链。
7. 同一组输入不应因论文加入顺序不同而产生本质不同的结果。
