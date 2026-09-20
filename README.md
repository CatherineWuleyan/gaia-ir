# Gaia IR

Gaia IR turns paper evidence into a typed, traceable domain graph. The repository is frozen around the verified five-paper **`pipeline_merge_v9`** result.

## Final repository surface

| Path | Purpose |
|---|---|
| [`agent/`](agent/) | Harness runtime, artifact store, checkpoints, validation, provenance, and generic Viewer source. |
| [`pipelines/single_paper/`](pipelines/single_paper/) | Maintained single-paper pipeline, latest checked Step 1–5 configuration. |
| [`pipelines/merge/`](pipelines/merge/) | Stepwise merge configuration and the checked five-paper v9 configuration. |
| [`docs/gaia-ir/`](docs/gaia-ir/) | Contracts and architecture notes needed to interpret Gaia IR artifacts. |
| [`merge_domain_graph/viewer_final.html`](merge_domain_graph/viewer_final.html) | Self-contained Viewer for the frozen v9 graph. |
| [`outputs/bohr_runs/merge_test1-5_v9_final/`](outputs/bohr_runs/merge_test1-5_v9_final/) | Frozen v9 `view_model.json` and `domain_graph.html`. |
| [`run_pipeline.sh`](run_pipeline.sh) | Repository entry point. |

`paper2ir` is a separate repository and is intentionally untouched.

## Run the merge pipeline

The launcher reports the selected configuration, Git revision, worktree state, and imported module path:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/pipelines/single_paper/pipeline.step1-5.json
./run_pipeline.sh merge /absolute/path/to/pipelines/merge/pipeline.test1-5.json
```

The published v9 artifacts are already available for inspection:

- [Open the v9 Viewer](merge_domain_graph/viewer_final.html)
- [Inspect the projected view model](outputs/bohr_runs/merge_test1-5_v9_final/view_model.json)
- [Inspect the generated domain graph](outputs/bohr_runs/merge_test1-5_v9_final/domain_graph.html)

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
