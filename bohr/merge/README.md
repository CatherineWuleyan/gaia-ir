# Running the domain-integration (pipeline_merge) on Bohrium

This directory submits the **current working-tree** `pipeline_merge` (Step 0–5
of Pipeline 8.0) as **one** Bohrium Job over the five minibatch paper packages
`test1`…`test5`, and projects the resulting **domain graph** viewer.

The merge is a single joint run, not one job per paper: `step0_select_scope`
freezes a `bootstrap` scope containing all five Packages, so Step 2/3 look for
cross-paper structure across the whole batch at once.

| Path | Role |
| --- | --- |
| `manifests/test1-5.json` | merge input manifest; paths are relative to the manifest directory, so it is only valid once staged under `<stage>/manifests/` |
| `build_input_manifest.py` | regenerates the manifest from `data/minibatch/<paper>/` (`gaia.ir.json` + `formalization.json` + `knowledge.index.json`) |
| `job_run.sh` | job-side runner: loads `agent/.env`, installs deps, preflights Gaia, runs the merge, then projects the domain graph |
| `stage_and_submit.sh` | local: assembles the upload payload and submits (or dry-runs) |
| `bohr_watch.sh` | local: polls the job to a terminal state, then pulls the driver log and result bundle |
| `fetch_results.sh` | local: downloads a finished job's `out.zip` |

## Usage

```bash
bash bohr/merge/stage_and_submit.sh stage       # build /tmp/gaia_bohr_merge_test1-5
bash bohr/merge/stage_and_submit.sh dry-run     # validate the Bohrium payload
bash bohr/merge/stage_and_submit.sh smoke       # 2-minute environment check, no pipeline run
bash bohr/merge/stage_and_submit.sh submit      # billable submission (stages if needed)

python3 bohr/merge/build_input_manifest.py test1 test2 test3 test4 test5 \
    --out bohr/merge/manifests/test1-5.json
bash bohr/merge/bohr_watch.sh <BOHR_ID> outputs/bohr_runs/merge_test1-5 240
```

Submit parameters are overridable with env vars: `PAPERS`, `PROJECT_ID`,
`MACHINE_TYPE`, `IMAGE_ADDRESS`, `MAX_RUN_TIME`, `RESULT_ROOT`, `STAGE_DIR`.

## The 5-paper scope

`pipelines/merge/pipeline.test1-5.json` freezes the manual Step 0 selection:

| folder | package | paper |
| --- | --- | --- |
| test1 | `papers:paper_24bac745` | Graph Neural Networks at a Fraction |
| test2 | `papers:paper_f519c93b` | Pruning and Quantization Impact on Graph Neural Networks |
| test3 | `papers:paper_607795ad` | Routing the Lottery: Adaptive Subnetworks for Heterogeneous Data |
| test4 | `papers:paper_7768dd46` | Evaluating Lottery Tickets Under Distributional Shifts |
| test5 | `papers:paper_af285cb1` | Finding Stable Subnetworks at Initialization with Dataset Distillation |

Domain: `lottery-ticket-hypothesis` (same string as the earlier two-paper merge
runs, so a later `incremental` run can treat this as the domain baseline).

## What the job produces

`job_run.sh` runs `pipeline_harness init` + `run`, then — because the harness
never projects a view itself — `pipeline_harness view --run <run_dir>` with the
`pipeline_merge.view:MergeViewAdapter` adapter declared by the config. Only
`results/` is declared in `backward_files`, and it contains:

```
results/domain_graph.html     self-contained domain graph (dagre + view inlined)
results/view_model.json       the projected view document
results/check.json            pipeline_harness check report
results/status.tsv            exit status + wall clock
results/logs/                 pip, init, run, stderr, view logs
results/runs/<run_id>/        the whole run: frozen inputs, artifacts, work files
results/merge_run/            a second copy of the run for convenience
```

`resultUrl` (via `fetch_results.sh`) is the source of truth for the bundle.

## Gotchas

The generic Bohrium gotchas found while building the single-paper driver still
apply — see `bohr/single/README.md` — in particular: install the published
`gaia-lang==0.5.0a7` wheel instead of the local package, parse `agent/.env`
tolerantly (it has a space after `=`), read the driver log from
`data.logFiles[0].url`, and do not wrap `bohr` in GNU `timeout` on macOS.

Merge-specific notes:

1. **The payload must carry three artifact kinds per paper.** `gaia.ir` is the
   only required input, but shipping `formalization` and `knowledge.index` lets
   Step 1 bind the author-side context and the pre-built retrieval index instead
   of rebuilding them.
2. **`pipelines/single_paper` must be staged too.** Step 3 imports
   `pipelines.single_paper.agent_pipeline_v2.step2._load_deepseek_env` and Step 4
   imports `...authoring`; `$ROOT` on `PYTHONPATH` resolves them as a namespace
   package.
3. **Bootstrap ignores the lexical Step 2 groups for judgment.** `step3` pairs
   claims semantically with the LLM and then gates batches of 20; Step 2's 52
   groups are still frozen and audited. Every Package is an equal anchor into
   one shared claim pool, so all cross-Package pairs are eligible regardless of
   which Package is smallest (see `RUNS.md` for the v1→v2 comparison).
4. **`DEEPSEEK_MERGE_MODEL`** overrides the Step 3/4 merge model
   (default `deepseek-v4-flash`); `DEEPSEEK_MODEL` still drives the single-paper
   vision model.
5. **Step 3/4 latency is remote HTTP, not local compute.** `job_run.sh` exports
   `GAIA_MERGE_RETRIEVAL_WORKERS` (default 16) and `GAIA_MERGE_JUDGE_WORKERS`
   (default 8). A GPU machine gives no speedup here — the process idles waiting
   on DeepSeek and the GPU would sit unused; only local embedding/inference
   would change that.

