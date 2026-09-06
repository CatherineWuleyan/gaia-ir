# Gaia IR 改进计划

状态：持续维护。

## 1. Step4 响应内容回退与重置机制

### 背景

DeepSeek thinking mode 的 assistant message 同时包含 `content` 和 `reasoning_content`。当前 run 中模型把合法的 Step4 JSON 放在 `reasoning_content`，而 `content` 为空，导致 pipeline 将有效结果误判为空 expansion。

### 目标行为

按以下顺序读取模型结果：

1. `content` 非空：解析 `content`；
2. `content` 为空且 `reasoning_content` 非空：尝试解析 `reasoning_content`；
3. 两者都为空：判定为空响应，触发一次重置/重试；
4. 重置后仍为空：将调用标记为失败，并记录明确的响应为空错误；
5. `reasoning_content` 非空但不是合法 Step4 JSON：标记为响应解析失败，进入既有 repair/重试流程，不得伪装成成功的空 expansion。

### 不变约束

- `content` 始终优先于 `reasoning_content`；
- 只有能解析且通过 Step4 schema 校验的 JSON 才能进入 formalization；
- 不能把整段 reasoning 文本直接当作正式结果；
- “模型明确返回 `{"knowledges":{},"strategies":[]}`”与“响应字段为空/解析失败”必须区分；
- 本改进不放宽 source anchor、strategy、证据充分性等逻辑门禁。

### 验收标准

- `content` 有效时，结果与当前行为一致；
- `content` 为空、`reasoning_content` 有效 JSON 时，能恢复 strategy/knowledge；
- 两者都为空时，不产生 `succeeded` 的空 expansion；
- `reasoning_content` 为普通思考文本时，不写入 formalization；
- 相关响应状态、重试次数和失败原因可在 run artifact 中审计。

## 2. 实验 Claim 处理

按照[实验 Claim 处理方案](experimental-claim-handling.md)实施：原始实验 claim 直接作为 `observation_claim`，Step2 回到原文和图片进行细化、补全和必要拆分，不再引入独立 candidate 层。

## 3. Step4 Anchor 与近重复关系

### Anchor 规则

Step4 新增的 claim 或 note 只能引用本次请求 `source_excerpts` 中出现的原始论文 anchor。

- 不得使用 `anchor_claim_*` 等 claims 文件内部 anchor 代替原文 anchor；
- `source_anchor_ids` 必须非空、唯一，并且属于 `source_excerpts[].anchor_id`；
- 校验失败时保留原始模型响应和错误信息，不得在重试后覆盖为 `null`。

### 近重复关系规则

如果两个 claim 只是具体化/概括化或高度近重复关系，不进入 abduction expansion。

处理顺序为：

1. 先判断两端是否表达同一事实；
2. 确认是近重复时归并为一个 canonical claim；
3. 合并来源 anchor 和 provenance；
4. 删除重复 relation；
5. 只有存在真实解释性缺口时，才进入 Step4 abduction。
