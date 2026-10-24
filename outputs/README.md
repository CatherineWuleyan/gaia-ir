# outputs

所有生成的图和配套数据都放在这里，按主题分目录。每个主题的最终版在 `final/`，
历史版本在 `versions/`（版本目录用日期前缀，`final/` 始终指向最新可用版本）。

| 主题 | 最终图 | 历史版本 |
|---|---|---|
| LTH 跨论文图 | [`lth/final/merge.html`](lth/final/merge.html) + [`view_model.reference.json`](lth/final/view_model.reference.json) | [`lth/versions/01-v9-example/`](lth/versions/01-v9-example/)（冻结 v9 示例） |
| Physics 领域图 | [`physics/final/domain_graph_full.reviewed.html`](physics/final/domain_graph_full.reviewed.html) + JSON | 复核前版本已删除；构建脚本与 LKM 输入在 [`physics/build/`](physics/build/) |
| Airplane 领域图 | [`aircraft/final/domain_viewer_question.html`](aircraft/final/domain_viewer_question.html) + answer/typed JSON | [`aircraft/versions/`](aircraft/versions/)（4 个历史版本，见其中 README） |
| 89 篇聚合图 | [`aggregate/viewer.html`](aggregate/viewer.html) + [`view_model.json`](aggregate/view_model.json) | 无 |
| Ariadne 运行输出 | `ariadne_runs/<timestamp>_<slug>/` | 每次运行一个目录 |

## 命名约定

- 主题目录用领域名：`lth/`、`physics/`、`aircraft/`、`aggregate/`。
- 最终版统一叫 `final/`，里面保留该主题的权威 viewer 和必要 JSON。
- 历史版本放在 `versions/`，目录/文件名以 `01-`、`02-` 或日期开头，便于排序。
- 不再把生成图写到仓库根目录或 `data/`。

## 相关验证

- LTH：`evidence/lth_final_graph_20261002/verify/test_final_graph.py`（15 个测试）。
- Airplane：`./ariadne/run.sh demo` 直接读取 `aircraft/final/` 的 JSON。
- Physics：`physics/final/` 为人工复核后的最终版。

## 与 git 的关系（重要）

仓库根目录的 `.gitignore` 忽略整个 `/outputs/`，因为 `ariadne_runs/` 是生成物。
但**冻结的示例数据必须入库**，否则 `git clone` 之后 `./ariadne/run.sh demo` 和
`ariadne doctor` 的 demo 就绪检查都会失败（这两个 bug 在 0.5.0 里修过一次）。

已入库的路径：

- `aircraft/final/`（demo 的 view / edge_roles / fixture 来源）
- `physics/final/`、`physics/build/target_question.json`
- 本文件、`aircraft/versions/README.md`

这些路径新增文件时要显式 `git add -f`：

```bash
git add -f outputs/aircraft/final/<new-file>
```

判断标准：**重新生成它需要花钱调用 LKM/LLM 的，入库；纯渲染产物，可以不入库。**
