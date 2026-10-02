# v2 live tool-loop preflight（离线）

## 目的

这一版补上了 v2 协议中缺少的“模型请求检索工具，再继续生成实验卡”这一段运行逻辑。它只定义 provider 接口并使用离线 `ScriptedProvider`，没有向 DeepSeek、Trisol 或其他网络端点发请求。

## 一轮交互

1. `build_prompt(task, condition)` 生成初始输入。输入包含完整目标论文、任务条件和检索接口说明；图条件只包含 `merge.html` 的快照文件、节点数、边数和 hash，不包含节点或边列表。
2. runner 把条件允许的工具定义传给 provider。模型可以先返回一个或多个工具调用。
3. runner 校验工具名和参数，在本地 `V2RetrievalSession` 中执行调用，把短结果作为下一轮 `tool` 消息返回。
4. 模型返回最终 v2 JSON 实验卡。
5. runner 将 session 的真实 `calls` 传给 `validate_v2(..., actual_retrieval_trace=...)`。模型自己填写的 `retrieval_trace.returned_ids` 不参与证据闭包。

## 图条件的关键保证

LTH fixture 的 model-facing JSON 只有图 metadata，没有 240 个节点和 235 条边的枚举。图节点只有在模型实际调用 `search_claims` 或 `expand_neighbors` 后才进入对话。隐藏 trace 同时保存 `returned_ids` 和带类型的 `returned_refs`，因此即使 task JSON 没有节点 ID，validator 也能确认某个 `graph_node` 或 `graph_edge` 确实由真实调用返回。

当前 `merge.html` 没有可审计 source anchor，所以图引用会保留 `graph provenance unavailable` warning；这不等于给图证据 provenance credit。

## 离线验证

`test_live_tool_loop.py` 覆盖三件事：

- 两轮工具调用 + 最终卡片，且 fake provider 伪造返回 ID 时，validator 仍使用 runner 的真实 trace；
- `paper_only` 不能调用外部文本工具；
- retrieval call budget 由 runner-owned session 强制执行。

定向测试命令：

```text
PYTHONPATH=paper2ir_experiment_copy .venv-gaia-a7/bin/python -m pytest -q \
  paper2ir_experiment_copy/research_triage/test_live_tool_loop.py \
  paper2ir_experiment_copy/research_triage/test_v2_protocol.py \
  paper2ir_experiment_copy/research_triage/test_v2_retrieval.py
```

结果：`11 passed`。

## 尚未完成

这不是 live provider 结果，也不声称模型已经在真实 API 上完成研究决策。后续接 DeepSeek/Trisol 时，只需实现 `CompletionProvider.complete(messages, tools, seed)`，保留本 runner 的检索、trace 和 validator 边界；必须把 provider 的工具调用模式映射到这个接口，不能退回把全图预加载到 prompt 的旧协议。
