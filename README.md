# Gaia IR

Gaia IR turns paper evidence into a typed, traceable domain graph. The repository is frozen around the verified five-paper **`pipeline_merge_v9`** result.

## Final repository surface

| Path | Purpose |
|---|---|
| [`agent/`](agent/) | Harness runtime, artifact store, checkpoints, validation, provenance, and generic Viewer source. |
| [`pipelines/single_paper/`](pipelines/single_paper/) | Maintained single-paper pipeline, latest checked Step 1–5 configuration. |
| [`pipelines/merge/`](pipelines/merge/) | Stepwise merge configuration and the checked five-paper v9 configuration. |
| [`pipelines/clean_claims/`](pipelines/clean_claims/) | paper2ir stage: paper graph → claims, used by Ariadne. |
| [`ariadne/`](ariadne/) | Question-led product layer: configuration, retrieval, orchestration, answer layer, viewer. |
| [`docs/gaia-ir/`](docs/gaia-ir/) | Contracts and architecture notes needed to interpret Gaia IR artifacts. |
| [`example/viewer.html`](example/viewer.html) | Self-contained Viewer for the frozen v9 graph. |
| [`example/`](example/) | Frozen v9 `view_model.json` and `domain_graph.html`. |
| [`run_pipeline.sh`](run_pipeline.sh) | Repository entry point. |

`pipelines/clean_claims/` is the paper2ir stage required by Ariadne. The standalone
`paper2ir` repository remains a separate, intentionally untouched project.

## Ariadne

[Ariadne](ariadne/) 是问题驱动的端到端产品层：输入一个领域问题，检索 LKM 论文，
依次跑 paper2ir、single、merge，最后生成候选答案 root、类型化推理链和自包含的
问题图 HTML。它复用本仓库维护中的 pipeline，不另起一套实现。

![Ariadne aircraft question graph](ariadne/docs/images/aircraft-question-graph.png)

```bash
./ariadne/run.sh config init     # 配置自己的 API key 和 Bohr 账户
./ariadne/run.sh doctor
./ariadne/run.sh demo            # 离线飞机示例，不需要 key
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

文档：[English](ariadne/README.md) · [中文](ariadne/README.zh-CN.md)。
示例：[Airplane](outputs/question_domain_aircraft/domain_viewer_question.html) ·
[Physics](outputs/physics_domain/reviewed/domain_graph_full.reviewed.html)。

## Run the merge pipeline

The launcher reports the selected configuration, Git revision, worktree state, and imported module path:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/pipelines/single_paper/pipeline.step1-5.json
./run_pipeline.sh merge /absolute/path/to/pipelines/merge/pipeline.test1-5.json
```

The published v9 artifacts are already available for inspection:

- [Open the v9 Viewer](example/viewer.html)
- [Inspect the projected view model](example/view_model.json)
- [Inspect the generated domain graph](example/domain_graph.html)

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
