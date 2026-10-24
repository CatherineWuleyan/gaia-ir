# LTH 最终图冻结 bundle

这个目录是 LTH 首轮 pilot 的冻结证据包，2026-10-18 从已删除的
`paper2ir_experiment_copy/evidence/lth_final_graph_20261002/` 原样迁入。

权威 viewer 是 [`outputs/lth/final/merge.html`](../../outputs/lth/final/merge.html)：

- HTML SHA-256：`3c7e9df131305d2e6326e24b97e3db7132c2c68aed5f8ba824c3960c126cd1bf`
- 节点 / 边：240 / 235
- 精确配套源：[`outputs/lth/final/view_model.reference.json`](../../outputs/lth/final/view_model.reference.json)（SHA-256 `5011c3aa…e59e84c1`，与 `graph_index.json` 相同）

| 文件 | 用途 |
|---|---|
| `graph_index.json` | 与 `merge.html` 完全一致的独立图索引 |
| `graph_snapshot.json` | 冻结快照（含 canonical payload） |
| `knowledge.index.json` | 同次运行的 claim 检索索引（71 条 claim / 57 个 source anchor） |
| `artifact_index.json` | 23 个 artifact 的 lineage |
| `manifest.json` | hash、provenance policy、检索接口说明 |
| `provenance_mapping.json` | derived provenance sidecar；不是 direct provenance |
| `verify/` | `merge.html`、索引和快照的一致性测试 |

## 验证

```bash
cd evidence/lth_final_graph_20261002/verify
../../../.venv-gaia-a7/bin/python -m pytest -q test_final_graph.py
```

当前 15 个测试全部通过。改动 `merge.html` 或索引后必须重跑这组测试。
