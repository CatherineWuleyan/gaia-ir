# Gaia IR

Gaia IR turns paper evidence into a typed, traceable domain graph. The workspace
contains the maintained Gaia IR pipeline and a separate research-triage pilot
that evaluates whether a cross-paper evidence graph changes an agent's next
experiment choice.

## Final repository surface

| Path | Purpose |
|---|---|
| [`agent/`](agent/) | Harness runtime, artifact store, checkpoints, validation, provenance, and generic Viewer source. |
| [`pipelines/single_paper/`](pipelines/single_paper/) | Maintained single-paper pipeline, latest checked Step 1–5 configuration. |
| [`pipelines/merge/`](pipelines/merge/) | Stepwise merge configuration and the checked five-paper v9 configuration. |
| [`docs/gaia-ir/`](docs/gaia-ir/) | Contracts and architecture notes needed to interpret Gaia IR artifacts. |
| [`outputs/`](outputs/) | 所有生成图与配套 JSON，按主题分目录。 |
| [`outputs/lth/final/`](outputs/lth/final/) | LTH 唯一权威图 `merge.html` + 精确配套 `view_model.reference.json`。 |
| [`outputs/lth/versions/01-v9-example/`](outputs/lth/versions/01-v9-example/) | 冻结 v9 的 `viewer.html`、`view_model.json` 和 `domain_graph.html`。 |
| [`outputs/physics/final/`](outputs/physics/final/) | Physics 最终复核图与配套 JSON。 |
| [`outputs/aircraft/final/`](outputs/aircraft/final/) | Airplane 最终问题图、typed view、answer layer 与 chains Markdown。 |
| [`outputs/aircraft/versions/`](outputs/aircraft/versions/) | Airplane 历史版本（流程原版、native-311、reviewed、domain）。 |
| `outputs/aggregate/` | 89 篇论文的聚合 viewer、view model、metrics 和审计报告；55 MB，**仅本地生成**，由 `build_aggregate_view.py` 重建，不入库。 |
| [`evidence/lth_final_graph_20261002/`](evidence/lth_final_graph_20261002/) | LTH 冻结图快照、graph index、provenance 审计和 manifest。 |
| [`evidence/lth_final_graph_20261002/verify/`](evidence/lth_final_graph_20261002/verify/) | `merge.html` 与冻结快照的一致性测试。 |
| [`ariadne/`](ariadne/) | 问题驱动的端到端 reasoning pipeline。 |
| [`run_pipeline.sh`](run_pipeline.sh) | Repository entry point. |
| [`docs/workspace-map.md`](docs/workspace-map.md) | Which directories are authoritative, generated, historical, or local-only. |

工作区现在只保留三套最终图，全部在 `outputs/` 下：

- Physics：`outputs/physics/final/`；
- LTH：`outputs/lth/final/merge.html` + `outputs/lth/final/view_model.reference.json` + `evidence/lth_final_graph_20261002/`；
- Airplane：`outputs/aircraft/final/`（历史版本在 `outputs/aircraft/versions/`，`ariadne` demo 直接读取 final 里的 JSON）。

其余 `outputs/*` 旧运行、`weekly_reports/out/*` 旧图和 `paper2ir_experiment_copy/` 已清理；
`archive/`（含已归档的 LKM 三件套、Bohr 作业、merge 图表、aircraft 历史复核和旧 skill）
与 `data/` 作为历史/本地语料保留。根目录只保留维护代码、`outputs/` 和数据目录。

## Ariadne

[Ariadne](ariadne/) 是问题驱动的端到端产品层：输入一个领域问题，检索 LKM 论文，
依次跑 paper2ir、single、merge，最后生成候选答案 root、类型化推理链和自包含的
问题图 HTML。LKM 是主索引；检索阶段还会用 OpenAlex、Crossref、arXiv、
Semantic Scholar 交叉确认候选论文，降低单索引漏检和元数据错配。它复用本仓库
维护中的 pipeline，不另起一套实现。

![Ariadne aircraft question graph](ariadne/docs/images/aircraft-question-graph.png)

```bash
./ariadne/run.sh config init     # 配置自己的 API key 和 Bohr 账户
./ariadne/run.sh doctor
./ariadne/run.sh demo            # 离线飞机示例，不需要 key
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

文档：[English](ariadne/README.md) · [中文](ariadne/README.zh-CN.md)。
示例：[Airplane](outputs/aircraft/final/domain_viewer_question.html) ·
[Physics](outputs/physics/final/domain_graph_full.reviewed.html)。

## Run the merge pipeline

The launcher reports the selected configuration, Git revision, worktree state, and imported module path:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/pipelines/single_paper/pipeline.step1-5.json
./run_pipeline.sh merge /absolute/path/to/pipelines/merge/pipeline.test1-5.json
```

The published v9 artifacts are already available for inspection:

- [Open the v9 Viewer](outputs/lth/versions/01-v9-example/viewer.html)
- [Inspect the projected view model](outputs/lth/versions/01-v9-example/view_model.json)
- [Inspect the generated domain graph](outputs/lth/versions/01-v9-example/domain_graph.html)

The generic Viewer template is [`agent/pipeline_harness/view/viewer.html`](agent/pipeline_harness/view/viewer.html). The Viewer only projects stored artifacts; it is not a second source of truth.

## Verified `pipeline_merge_v9`

| Metric | Result |
|---|---:|
| Graph nodes | 318 |
| Graph edges | 257 |
| Integration-delta edges | 74 |
| Delta composition | 62 deduction · 10 abduction · 2 contradiction |
| Conclusion structure | 1 tree covering 8 L1 conclusions + 3 independent L1 conclusions |

The result is a verified forest and retains the expected `GRAPH_DISCONNECTED` warning for three independent conclusions.

## Validation

```bash
PYTHONPATH=agent .venv-gaia-a7/bin/python \
  -m unittest discover -s agent/tests -p 'test_*.py'
```

Generated runs, caches, intermediate datasets, and local secrets remain outside the committed final surface. The two maintained workflow sources are the single-paper pipeline and `pipeline_merge_v9`; older compatibility and batch-run wrappers are excluded.
