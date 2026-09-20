# pipeline_merge 领域图迭代对比（v1 → v9）

输入：`test1`–`test5` 五篇论文包（bootstrap 模式，Step 0–5），跑在 Bohrium 上。

| 文件 | 内容 |
| --- | --- |
| `index.html` | 迭代对比总览：每版的改动、delta 边数/构成、结果；含构成图 |
| `viewer_final.html` | **最终版领域图**（v9，完整三层视图，自包含可直接打开） |
| `delta_v*.html` | 各版本**只看集成 delta** 的 viewer（论文层默认关闭） |
| `delta_chart.svg` | delta 边构成堆叠柱状图 |
| `delta_stats.json` | 各版本 delta 统计原始数据 |

最终版（v9，BOHR_ID `20732204`）：318 节点 / 257 边，集成 delta 74 条
（contradiction 2、deduction 62、abduction 10），结论树 12 个节点（L1 11 + L2 根 1），
每个结论 ≥2 条前提、≤8 条（`CONCLUSION_CAP`），方向为 `premises → Ki`（文档 §3.3）。
