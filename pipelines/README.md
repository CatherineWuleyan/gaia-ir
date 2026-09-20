# Maintained pipelines

The repository keeps two current workflows: the latest single-paper pipeline and the verified five-paper `pipeline_merge_v9` merge pipeline.

| Workflow | Configuration |
|---|---|
| `single-v2` | `single_paper/pipeline.step1-5.json` |
| `merge` | `merge/pipeline.test1-5.json` (v9) |

Run from the repository root:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/pipelines/single_paper/pipeline.step1-5.json
./run_pipeline.sh merge /absolute/path/to/pipelines/merge/pipeline.test1-5.json
```

The published v9 result is stored under `outputs/bohr_runs/merge_test1-5_v9_final/`; its self-contained Viewer is `merge_domain_graph/viewer_final.html`.
