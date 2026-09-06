# 实验 Claim 处理方案

状态：已批准，作为后续实现依据。

## 目标

原始实验 claim 不丢失、不重复导入，同时要求 Step2 回到论文原文和图片，对原始实验 claim 做细化、补全和必要拆分。

## 核心规则

系统只保留两种正式知识类型：

- `claim`：非实验性知识；
- `observation_claim`：实验设置、操作、测量和结果构成的实验事实。

不再使用 `obsevation_candidate` 作为正式中间层，也不把 candidate 作为独立 claim 再次送入 Step2。

原始实验 claim 直接升级为 `observation_claim`。原始 claim 的 ID 保留，避免因为重新抽取而产生第二个同义节点。

## Step2 的职责

Step2 必须在原文、图、图注和相关上下文中核对每一个原始实验 claim，并将其细化为可验证的实验事实。

细化至少覆盖以下信息（原文存在时必须保留）：

- Setting：数据集、任务、模型、训练/评估条件；
- Action：剪枝、训练、对比或其他实验操作；
- Measurement：指标、参数比例、稀疏率等测量项；
- Result：与 baseline 或其他对照的具体比较结果。

Step2 可以：

1. 补全原始 claim 缺失的实验条件或指标；
2. 将一个包含多个独立实验结果的原始 claim 拆成多个 `observation_claim`；
3. 从原文或图片中新增原始 claim 未覆盖的实验结果。

Step2 不可以：

1. 把原始实验 claim 作为另一套输入节点重新导入；
2. 因为文本改写而同时保留原始 claim 和一个同义 observation claim；
3. 用模型常识补充原文没有给出的实验数值或结论。

## 拆分规则

如果一个原始 claim 包含多个相互独立的实验结果，允许拆分。例如：

```text
claim_7  ->  claim_7a, claim_7b, claim_7c
```

拆分后的节点都属于 `observation_claim`，并保留原始 claim 的来源信息。拆分只在确实存在独立的 Setting、Measurement 或 Result 时进行，不能为了格式统一而机械拆分一句话。

如果原始 claim 已经是一个完整、不可再拆的实验事实，则直接在原 ID 上补全内容，不新增同义节点。

## 新增实验 claim

只有当 Step2 在原文或图片中发现原始输入完全没有覆盖的新实验结果时，才创建新的 ID：

```text
claim_O01
claim_O02
```

每个新增节点必须带有明确的 `source_anchor_ids`，且这些 ID 必须非空、唯一、可解析。

## 去重规则

去重只发生在 observation claim 内部：

- 原始实验 claim 与其细化结果不是两个节点，不做二次语义去重；
- 拆分结果之间如果语义完全重复，只保留一个；
- 新增结果与已有 observation claim 重复时，合并来源锚点，不创建重复节点；
- 无法判断是否重复时，保留节点并记录待审核状态。

## 最终 graph 规则

最终 graph 中：

- 原始实验 claim（经 Step2 细化后）作为 `observation_claim` 使用；
- Step2 新发现的实验事实作为新增 `observation_claim` 使用；
- 不出现 `obsevation_candidate` 节点；
- Step4 只使用细化后的 observation claim，不使用未处理的原始 candidate 文本。

## 验收标准

一次合格的 Step2 结果必须满足：

1. 每个原始实验 claim 都有处理状态：`kept`、`refined`、`split` 或 `not_experimental`；
2. 所有被保留为实验事实的 claim 都是 `observation_claim`；
3. 每个 observation claim 都能回到原文、图片或图注中的 source anchor；
4. 多任务、多指标、多数据集结果没有被无理由压缩成一条总括 claim；
5. 没有同一事实的 candidate/claim/observation 三重重复；
6. 原文没有出现的内容不能被 Step2 臆造。

## 本方案明确不处理的事项

本方案只规定原始实验 claim 与 Step2 的边界，不改变：

- Step3 relation 的分类规则；
- Step4 weakpoint 的展开规则；
- viewer 的展示样式；
- Gaia runtime 版本。

