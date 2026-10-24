# 工作区整理说明

整理时间：2026-10-18。所有生成图现在统一放在 `outputs/` 下，按主题（`aircraft/`、
`physics/`、`lth/`）分目录，同一主题的不同版本收在各自的 `versions/` 里。

## 根目录结构

根目录只保留 10 个文件夹：

```
agent/  archive/  ariadne/  benchmark/  data/
docs/   evidence/  outputs/  pipelines/  weekly_reports/
```

以及 `README.md`、`run_pipeline.sh`、`build_aggregate_view.py` 等顶层文件。
历史实验、一次性产物和旧代码统一收在 `archive/`。

## outputs 结构

```
outputs/
├── aircraft/
│   ├── final/                              # 2026-10-18 最终问题图
│   │   ├── domain_viewer_question.html
│   │   ├── domain_view_question.json / domain_view_typed.json
│   │   ├── edge_roles.json / edge_roles_typed.json
│   │   ├── answer_layer.json / answer_fixture.json
│   │   ├── ALL_CHAINS_TO_ROOTS.md / ALL_CHAINS_TO_ROOTS_FEISHU_DRAFT.md
│   │   └── evaluation、cross-paper audit、typical chains
│   └── versions/
│       ├── 01-flow-original.html            # 流程原版
│       ├── 02-native-311-2026-10-13/        # 311 节点原生图（run 目录，viewer 在 views/）
│       ├── 03-reviewed-handcrafted-2026-10-14/  # 手工清洗版（question 层默认输入）
│       └── 04-domain-2026-10-17.html        # 从 git HEAD 恢复的 domain viewer
├── physics/
│   ├── final/                               # 复核版图 + answer layer / graph JSON
│   └── build/                               # 构建脚本、LKM 输入、materials/target/theory
├── lth/
│   ├── final/                               # merge.html + view_model.reference.json + gaia.ir.json
│   └── versions/01-v9-example/              # 冻结 v9 viewer、view_model、domain_graph
├── aggregate/                               # 89 篇论文聚合 viewer + view_model + metrics + report
└── ariadne_runs/                            # ariadne 每次运行的输出
```

命名和版本约定见 [`outputs/README.md`](../outputs/README.md)；Airplane 各版本说明见
[`outputs/aircraft/versions/README.md`](../outputs/aircraft/versions/README.md)。

## 当前权威最终产物

| 路径 | 用途 | 处理规则 |
|---|---|---|
| [`outputs/lth/final/merge.html`](../outputs/lth/final/merge.html) | LTH 唯一最终图 | 只读，不用其他 viewer 替换 |
| [`outputs/lth/final/view_model.reference.json`](../outputs/lth/final/view_model.reference.json) | `merge.html` 的精确配套源（240 节点 / 235 边） | 检索与投影只读这一份 |
| [`evidence/lth_final_graph_20261002/`](../evidence/lth_final_graph_20261002/) | 冻结图快照、`graph_index.json`、artifact registry、provenance 审计、manifest | 从已删除的 `paper2ir_experiment_copy` 迁入；运行时不改写 |
| [`evidence/lth_final_graph_20261002/verify/`](../evidence/lth_final_graph_20261002/verify/) | `merge.html` 与快照一致性测试（`test_final_graph.py`） | 改动最终图后先跑这组测试 |
| [`outputs/physics/final/`](../outputs/physics/final/) | Physics 最终复核图（`domain_graph_full.reviewed.html`）+ answer layer / graph JSON | 只读；`pre_*` 中间版本已删 |
| [`outputs/aircraft/final/`](../outputs/aircraft/final/) | Airplane 最终问题图、typed view、answer layer、edge roles 与 chains JSON | `ariadne` demo 和 question 层 pipeline 的输入 |
| [`outputs/aircraft/versions/`](../outputs/aircraft/versions/) | Airplane 四个历史版本 | 只读比较 |
| [`outputs/lth/versions/01-v9-example/`](../outputs/lth/versions/01-v9-example/) | 冻结 v9 的公开示例 viewer | 只读示例 |
| [`ariadne/`](../ariadne/) | 问题驱动的端到端 pipeline（question → LKM → paper2ir → single → merge → answer layer） | 维护代码，入口 `ariadne/run.sh` |

LTH 最终图的核对值：HTML SHA-256 为
`3c7e9df131305d2e6326e24b97e3db7132c2c68aed5f8ba824c3960c126cd1bf`，canonical
payload 为 `9deaac0d4767d4f75e98f6383fc238513643708d37c7ed577e5bab68c629d74d`，
包含 240 个节点和 235 条边。

## 不要混用的相似产物

- `outputs/lth/final/merge.html` 是唯一权威 LTH 图；旧的 `outputs/merge_corpus/view_model.json`
  （207/154 版本）已删除，历史 `weekly_reports/out/*` 图也已删除。
- `outputs/aircraft/versions/03-reviewed-handcrafted-2026-10-14/source_snapshot/domain_view_model.json`
  是 airplane question 层 pipeline 的默认输入，不能替换成旧 viewer 或旧快照。
- physics 的 `outputs/physics/final/` 是最终版；`pre_candidate_rename` / `pre_logic_fix`
  中间版本已删除，不要再从备份 HTML 反推当前图。
- `outputs/aircraft/versions/04-domain-2026-10-17.html` 是从 git HEAD 恢复的
  `gaia/airplane/domain_viewer.html`；原 `gaia/airplane/` 目录已删除。

## 2026-10-24 docs 清理

`docs/` 之前混着两类东西：Gaia IR 契约，和隐藏论文实验的全套过程笔记（89 个文件）。
后者已整体归档到 `archive/superseded-docs/20261024-retired/`，原因见其中的 README：

- 实验已退役，`hidden-paper-*` 不再维护；
- 其中多份文档直接写着隐藏论文 A 的身份和 gold 实验，不能放在会被 `git add -A`
  扫到的位置；
- `worktree-*-summary`、`weekly-report-draft-*` 等是过程笔记，不是产品契约。

`docs/` 现在只保留 `gaia-ir/`（契约）、`theory/`（理论）、`pipelines/`（pipeline 协议）、
`validation/`（对比验证）和本文件。标准：读代码或产物时需要查的留下，记录思路的归档。

另：`origin/main` 上已经没有 `docs/`，这批文档只存在于本地工作区；不要把 `docs/`
当作已发布内容引用。

## 目录分类

| 类别 | 路径 | 说明 |
|---|---|---|
| 维护源码 | `agent/`, `pipelines/`, `ariadne/`, `docs/` | 可审查的代码、协议和文档（`docs/` 只留契约与理论） |
| 最终产物 | `outputs/aircraft/final/`, `outputs/physics/final/`, `outputs/lth/final/`, `outputs/lth/versions/`, `evidence/lth_final_graph_20261002/` | 报告和展示只读这些 |
| 本地论文语料 | `data/` | 80+ LTH 论文 OCR、图片和 formalization；不提交 Git |
| 历史封存 | `archive/`（`runs/`、`assets/figure/`、`lkm/`、`bohr-jobs/`、`merge-delta-charts/`、`aircraft-history-20261017/`、`rerun-inputs/`、`research-triage-minimal/`、`skills/`、`superseded-docs/`、`removed-paper2ir-20261018/`） | 只读比较，不进入主 pilot |
| 环境/缓存 | `.venv-gaia-a7/` | 本地运行环境，不纳入整理结果 |

## 2026-10-18 清理记录

| 清理项 | 处理 |
|---|---|
| `paper2ir_experiment_copy/`（约 587M） | 整个删除。删除前把 `evidence/lth_final_graph_20261002/` 与两个校验脚本迁到 `evidence/`；未提交的 tracked 改动存为 `archive/removed-paper2ir-20261018/worktree-changes.patch.gz` |
| `outputs/` 旧运行（约 1.6G） | 删除 `bohr_runs`、`pipeline_single_runs`、`harness_runs`、`test2_*`、`merge_*`、`reruns_*`、`fig2_*`、`weekly_report_*`、`lkm_*` 等 |
| `outputs/physics_domain/` | 删除旧 `build/` 运行、`work/`、`smoke*`、复核前图和 `pre_*` 中间件；复核版移到 `outputs/physics/final/`，构建脚本与 LKM 输入移到 `outputs/physics/build/` |
| `weekly_reports/out/` | 旧的 reviewed 图移到 `outputs/aircraft/versions/03-reviewed-handcrafted-2026-10-14/`，其余旧图删除 |
| `archive/`（约 312M） | 删除与 `archive/assets/figure/` 完全相同的 `legacy-paper-figures/`，删除 `working-outputs/` |
| 根目录 | 删除 `LKM.html`、`claims_final.html`、`formalization.html`、重复的 `domain_graph_full.reviewed.html`、`results/`、`converted_viewers/`、`.tmp_question_probe.py` |
| `docs/` | 同系列旧版本 63 个移到 `archive/superseded-docs/`，只保留最新版 |
| 缓存 | 全工作区删除 `__pycache__`、`.pytest_cache`、`.DS_Store`（约 38M） |
| 根目录归档 | 9 个历史/一次性目录移入 `archive/`：`lkm_native`→`archive/lkm/native`、`lkm_docs`→`archive/lkm/docs`、`lkm_probe`→`archive/lkm/probe`、`bohr`→`archive/bohr-jobs`、`merge_domain_graph`→`archive/merge-delta-charts`、`aircraft_history_review_20261017`→`archive/aircraft-history-20261017`、`rerun_inputs`→`archive/rerun-inputs`、`research_triage`→`archive/research-triage-minimal`、`skills`→`archive/skills` |
| 生成图归位 | `data/aggregate_*`→`outputs/aggregate/`（`viewer.html`、`view_model.json`、`metrics.json`），根目录 `compare.md`→`outputs/aggregate/report.md`；`merge.html`→`outputs/lth/final/`；`outputs/merge_corpus/*`→`outputs/lth/final/`；`example/`→`outputs/lth/versions/01-v9-example/`；`domain_graph_aircraft_original.html`→`outputs/aircraft/versions/01-flow-original.html`；`outputs/question_domain_aircraft/`→`outputs/aircraft/final/`；`outputs/aircraft_domain_merge/run_...`→`outputs/aircraft/versions/02-native-311-2026-10-13/`；`weekly_reports/out/domain_graph_aircraft_reviewed/`→`outputs/aircraft/versions/03-reviewed-handcrafted-2026-10-14/`；`outputs/physics_domain/reviewed/`→`outputs/physics/final/`；`outputs/physics_domain/*`→`outputs/physics/build/`；`outputs/airplane_final/` 合并进 `outputs/aircraft/final/` |

清理前的大小约 4.8G，清理后约 1.7G。

## Git 整理规则

当前工作区仍有一批未提交改动。不要使用 `git add .` 或整体提交。提交时按白名单添加：
最终图、冻结 evidence、`ariadne/`、`agent/`、`pipelines/`、对应测试和文档。
`outputs/` 在 `.gitignore` 里，但被 Git 跟踪的最终图（aircraft final、physics final、
v9 example）已经以 rename 形式暂存；提交前用 `git status` 确认，不要用 `git checkout`
把它们还原回旧路径。

## 下一步顺序

1. 先用 `evidence/lth_final_graph_20261002/verify/test_final_graph.py` 验证
   `outputs/lth/final/merge.html`、canonical payload 和 `graph_index.json` 三者一致；
2. 再跑 `./ariadne/run.sh demo`，确认 airplane question 层读取的 JSON 完整；
3. 最终图发生任何变化时，同步更新 `outputs/lth/final/merge.html`、
   `outputs/lth/final/view_model.reference.json` 和 `evidence/lth_final_graph_20261002/`，
   不要只改其中一处。
