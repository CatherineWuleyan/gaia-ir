# Aircraft 领域图历史版本

最终问题图在 [`../final/domain_viewer_question.html`](../final/domain_viewer_question.html)。
本目录只保留与最终版不同的历史版本，按时间排序：

| 版本 | 文件/目录 | 说明 |
|---|---|---|
| 01 | `01-flow-original.html` | 流程原版（原根目录 `domain_graph_aircraft_original.html`） |
| 02 | `02-native-311-2026-10-13/` | 311 节点原生图运行；viewer 在 `views/viewer.html` |
| 03 | `03-reviewed-handcrafted-2026-10-14/` | 手工清洗复核版；`domain_viewer.html` + `source_snapshot/`，是 question 层 pipeline 的默认输入 |
| 04 | `04-domain-2026-10-17.html` | 从 git HEAD 的 `gaia/airplane/domain_viewer.html` 恢复；原目录已删除 |

已去重：`gaia/airplane/domain_viewer.v2.backup.html` 与 02 的 viewer 完全相同，
`airplane_final/domain_viewer_question.html` 与最终版完全相同，均未重复保留。
