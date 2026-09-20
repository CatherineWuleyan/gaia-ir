# Maintained pipelines

This directory contains the two maintained Gaia IR workflows: a single-paper pipeline and a multi-paper merge pipeline. Both use the shared Harness and produce traceable, validated artifacts.

| Workflow | Configuration |
|---|---|
| `single-v2` | `single_paper/pipeline.step1-5.json` |
| `merge` | `merge/pipeline.test1-5.json` |

Run from the repository root:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/pipelines/single_paper/pipeline.step1-5.json
./run_pipeline.sh merge /absolute/path/to/pipelines/merge/pipeline.test1-5.json
```

The checked merge example is in [`example/`](../example/), with a self-contained Viewer, projected view model, and generated domain graph.
