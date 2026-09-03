# Round 4 跨文章审计

## 数据集与来源

- Bohr 运行目录：`/workspace/gaia-round3/results_fixed`（30 个 manifest；唯一失败的运行已重试成功）。
- 冻结的 pipeline commit：`516134cb515acec780e39dcdc5cd17d43363adf0`。
- 本地副本：`graph_single/round4/results_fixed`。

## 观测统计

最终 Step 4 formalization 共覆盖 30 篇文章、1,018 个 claim Knowledge、352 个 weakpoint，其中 176 个仍未分类（`reasoning_type=null`），最终只持久化了 6 个命名 strategy。30 个结果中的 `graph.composes` 全部为空。Step 4 共生成 309 个 operator，但其中大多数只是未解决关系表达式的 conjunction helper，并不是完整的 claim-to-claim 推理策略。全语料中 canonical claim 重复项多出的数量为 8（平均每篇 0.27 个，单篇最多 4 个）。

Step 2 生成了 260 个实验观察相关条目：123 个 `obsevation_candidate` 和 137 个 `observation_claim`。只有 9/30 篇文章包含 `observation_claim`；21 篇虽然有候选条目，却没有提升为实验 claim。候选类型在公开 contract 中还存在拼写错误（`obsevation_candidate`），使下游筛选和审计容易出错。

## Pipeline 根因

1. **关系被放在 workflow 旁路中。** Step 3 将候选关系写入 `workflow.weakpoints`；Step 4 过滤掉所有 null 分类，并且只有额外的 LLM 扩展成功后才写入 strategy。因此，一条关系可能已经被抽取并通过校验，却仍然从 `graph.strategies` 和最终渲染的推理层中消失。
2. **null 分类没有安全的机械降级。** Step 4 的 prompt 明确说 null 应在工具外机械映射为 `infer`，但实现实际把它保留为 unresolved，从未完成该映射。这直接造成关系数量少、null 数量大。
3. **关系扩展是全有或全无，而且过于严格。** 只要缺少一个 anchor/rule，或发生 official strategy-ID 冲突，整次扩展就会被丢弃。本轮失败文章出现了 `source_anchor_ids` 唯一性错误和 strategy-ID 冲突；同一策略也会让本来有效的部分关系无法表示。
4. **实验抽取没有闭环提升机制。** 文本候选可以生成，但提升为 observation 依赖语义工具为每个 observation 返回恰好一个等价 claim。因此候选记录可能保留下来，却没有可用的实验 claim；vision fallback 只在文本扩展失败后触发，而且依赖图片文件名精确匹配。
5. **去重只发生在局部抽取阶段。** 本轮出现 8 个 canonical claim 重复项，说明 imported claim、observation equivalent 和新增扩展 claim 之间没有统一的最终合并去重。去重应合并 provenance 并保留 alias，而不是简单删除记录。

本轮实验 claim 缺失还有一个明确的运行时原因：30 篇中有 22 篇在 Step 2 语义调用时收到 DeepSeek HTTP 402（`Insufficient Balance`）。这不是“没有实验结果”，而是 API 计费失败；另有 2 篇返回了无效 JSON，2 篇未通过 schema/content 校验。pipeline 保留了候选，但在没有模型调用的情况下不能凭空断言实验结果。

## 下一步拟实现的通用修复

- 对证据和 ID 均通过门槛的 null weakpoint，确定性地降级为 `infer` strategy（同时支持单目标和多目标）；未通过门槛时保留 weakpoint 和 source anchor。
- 为 observation candidate 使用 schema 兼容的 canonical 类型；语义抽取不足时保留候选 provenance，并将图片匹配从精确文件名扩展为 basename/label 归一化匹配。
- 通过统一 premise 顺序、background 以及每个 weakpoint 的稳定命名空间，确定性处理 strategy-ID 冲突；单条扩展失败不能丢弃其他无关关系。
- 在最终合并边界执行全语料 claim canonicalization，同时保留 provenance 和 alias。

这些改动只依赖 source anchor、claim ID、关系元数和 schema 约束，不依赖某篇文章的措辞或固定阈值，因此可以泛化到其他文章和领域。
