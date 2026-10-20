# Gaia IR

[English](README.md) · [简体中文](README.zh-CN.md)

Turn paper claims into Gaia IR packages, then merge those packages into a cross-paper reasoning graph. This repository contains the runtime, extraction stages, pipeline configurations, and their tests. For question-led retrieval and the end-to-end CLI, use [Ariadne](https://github.com/CatherineWuleyan/ariadne).

## Install

Python 3.11 or later is required. The compiler dependency is pinned to `gaia-lang==0.5.0a7`.

```bash
git clone https://github.com/CatherineWuleyan/gaia-ir.git
cd gaia-ir
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e agent

export PYTHONPATH="$PWD/agent:$PWD/pipelines/single_paper:$PWD/pipelines/merge"
```

Live extraction and reasoning stages need a model key. The tested client uses an OpenAI-compatible endpoint:

```bash
export DEEPSEEK_API_KEY="your-key"
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

## Run

The supported pipeline configurations are `pipelines/single_paper/pipeline.json`, `pipelines/merge/pipeline.json`, and `pipelines/merge/pipeline.question.json`. Stage-only configurations live under test fixtures.

### Single paper

Prepare an input manifest with paths relative to the manifest itself:

```json
{
  "artifacts": [
    {"path": "paper.md", "kind": "source.paper_text", "media_type": "text/markdown"},
    {"path": "claims_final.json", "kind": "source.claims_final", "media_type": "application/json"}
  ]
}
```

```bash
./run_pipeline.sh single-v2 /absolute/path/to/inputs.json
```

To extract claims from a paper graph first, place it at `pipelines/clean_claims/data/<paper-id>/graph.json` and run:

```bash
python pipelines/clean_claims/run_full_pipeline.py <paper-id>
```

### Merge papers

Export each completed single-paper run:

```bash
python pipelines/single_paper/export_artifacts.py /path/to/run paper-id --out /path/to/packages
```

The exporter writes `gaia.ir.json`, `formalization.json`, and `knowledge.index.json` under `graph_single/<paper-id>/`. Create a merge input manifest listing these files for each paper, with artifact kinds `gaia.ir`, `formalization`, and `knowledge.index`.

Set `scope.domain` and `scope.selection` in the first stage of `pipelines/merge/pipeline.json`; selection entries must match the packages' `namespace:name` identities. Then run:

```bash
./run_pipeline.sh merge /absolute/path/to/merge-inputs.json
```

For the question answer layer, configure the scope and question in `pipeline.question.json`, then pass that configuration to `pipeline_harness init --pipeline`.

### Inspect or resume a run

```bash
python -m pipeline_harness status --run /path/to/run
python -m pipeline_harness run --run /path/to/run
python -m pipeline_harness view --run /path/to/run
python -m pipeline_harness check --run /path/to/run
```

Open the generated `views/viewer.html` in a browser. Generated runs and local inputs are ignored by Git.

## Offline check

The built-in synthetic workflow checks the harness without model calls:

```bash
run_dir=$(python -m pipeline_harness init --pipeline synthetic --runs-root outputs/smoke)
python -m pipeline_harness run --run "$run_dir"
python -m pipeline_harness view --run "$run_dir"
python -m pipeline_harness check --run "$run_dir"
```

## Tests

With the environment and `PYTHONPATH` set as above:

```bash
python -m unittest discover -s agent/tests -p 'test_*.py'
python -m unittest discover -s pipelines/single_paper/tests -p 'test_*.py'
python -m unittest discover -s pipelines/merge/tests -p 'test_*.py'
```

## Layout

```text
agent/                         Harness, artifact store, compiler integration, viewer, tests
pipelines/clean_claims/         Paper graph → extracted claims
pipelines/single_paper/         Claims → Gaia IR package; package exporter
pipelines/merge/                Cross-paper merge and optional answer layer
requirements.txt               Runtime dependencies
run_pipeline.sh                Single-paper and merge entry point
```
