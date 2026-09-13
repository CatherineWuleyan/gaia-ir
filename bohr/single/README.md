# Running the single-paper V2 pipeline on Bohrium

This directory submits the **current working-tree** `single-v2`
(`pipelines/single_paper/pipeline.step1-5.json`, pipeline id
`agent-pipeline-v2.1-step1-5`) as Bohrium Jobs, one job per minibatch paper.

## Layout

| Path | Role |
| --- | --- |
| `manifests/test{3,4,5}.json` | harness input manifests; paths are relative to the manifest directory, so they are only valid once staged under `<stage>/manifests/` |
| `build_input_manifest.py` | regenerates a manifest from `data/minibatch/<paper>/` — text, claims, and one `source.original_figure` per image |
| `job_run.sh` | job-side runner: loads `agent/.env`, installs deps, preflights Gaia, then runs one paper |
| `stage_and_submit.sh` | local: assembles the upload payload and submits (or dry-runs) |
| `bohr_watch.sh` | local: polls a job to a terminal state, then pulls the driver log and result bundle |
| `fetch_results.sh` | local: downloads a finished job's `out.zip` |
| `collect_results.py` | local: copies `formalization.json` / `gaia.ir.json` / `knowledge.index.json` out of a run into `data/minibatch/<paper>/` |

## Usage

```bash
bash bohr/single/stage_and_submit.sh stage test3       # build /tmp/gaia_bohr_single_test3
bash bohr/single/stage_and_submit.sh dry-run test3     # validate the Bohrium payload
bash bohr/single/stage_and_submit.sh smoke             # 2-minute environment check, no pipeline run
bash bohr/single/stage_and_submit.sh submit test3      # billable submission (stages if needed)

python3 bohr/single/build_input_manifest.py --paper test4 --out bohr/single/manifests/test4.json
bash bohr/single/bohr_watch.sh <BOHR_ID> outputs/bohr_runs/test3 180
python3 bohr/single/collect_results.py --run <unpacked run dir> --paper test3
```

Submit parameters are overridable with env vars: `PROJECT_ID`, `MACHINE_TYPE`,
`IMAGE_ADDRESS`, `MAX_RUN_TIME`, `RESULT_ROOT`, `STAGE_DIR`.

## What the job does

The uploaded directory root becomes the working directory. `job_run.sh`
starts from whatever directory it is invoked in, so the staged tree keeps the
same relative layout as the repository (`agent/`, `pipelines/single_paper/`,
`data/minibatch/…`), which matters because:

* `agent_pipeline_v2.step2._load_deepseek_env()` walks up from its own file
  looking for `agent/.env`;
* `agent_pipeline_v2.step4` reaches into
  `agent_pipeline_v2/claim_cleaner/data/` (shipped empty so the remote run does
  not inherit local working outputs);
* `pipeline_harness` resolves from `PYTHONPATH`, so the harness is **not**
  installed — only the published `gaia-lang` wheel is.

Only `results/` is declared in `backward_files`, so `agent/.env` (which holds
`DEEPSEEK_API_KEY`) is never copied into the result bundle.

## Gotchas found the hard way

1. **`agent/.env` cannot be `source`d.** It is written as
   `DEEPSEEK_API_KEY= sk-…` (a space after `=`), so POSIX `.` tries to execute
   the key as a command. The pipeline tolerates this because it splits on `=`
   and strips; `job_run.sh` uses an equivalently tolerant parser.

2. **`pip install -e .` fails in the job image.** The image's pip defaults to
   `https://pypi.tuna.tsinghua.edu.cn/simple` through `http://ga.dp.tech:8118`,
   which returns 403 for some wheels, and editable installs additionally need
   PEP 517 build isolation (`setuptools`). Install the published
   `gaia-lang==0.5.0a7` wheel instead and put the harness on `PYTHONPATH`.
   In practice `--index-url https://mirrors.aliyun.com/pypi/simple` works.

3. **`bohr job download` is broken in CLI 2.2.20.** It resolves the Bohr ID and
   then `GET /openapi/v4/job/<jobId>`, which 404s. The working endpoint is
   `GET /openapi/v4/job/detail/<bohrId>`, whose `resultUrl` is a signed URL for
   the job's `out.zip`. `fetch_results.sh` uses that.

4. **`bohr job log` also pages.** The driver log is a declared `log_file`; its
   signed URL is in `data.logFiles[0].url`, which is easier to read than the
   paginated `data.log` field.

5. **`-r /personal/...` does not populate the disk** when submitting from a
   local machine (`autoDownloadStatus` stays incomplete and the directory is
   created empty). Treat `resultUrl` as the source of truth.

6. **The upload endpoint rejects large payloads.** `POST
   https://tiefblue.dp.tech/api/upload/binary` fails with *"Client.Timeout
   exceeded while awaiting headers"* once the directory grows much past a few
   MB — an 11 MB payload (all three papers plus their figures) never completed,
   while 2–9 MB per-paper payloads did. Reachability is not the problem (the
   host answers `curl` in ~0.5 s). Stage one paper at a time, which is what
   `stage_and_submit.sh` now does by default. A failed upload never creates a
   job, so retrying `submit <paper>` is safe.

7. **Figures must be declared or step 2 cannot see them.** The pipeline only
   reads a figure through `source.original_figure`, matched by
   `metadata.source_filename` against the candidate's `image_name`
   (`step2.py:1385`). With text inputs alone, a paper whose numbers live in
   plots yields almost no observations: test4 produced 1 observation and a
   0-edge graph, and gained 29 edges once its 4 figures were supplied
   (`build_input_manifest.py` does this automatically). See `RUNS.md`.

8. **Do not wrap `bohr` in `timeout` on macOS.** GNU `timeout` is not installed,
   so `timeout 60 bohr job list … 2>/dev/null` runs nothing, exits 127, and the
   empty stdout looks exactly like an API outage. This produced several
   false "Bohr API is flaky" readings before it was spotted; `bohr job list`
   actually answers in ~1.4 s. Use a background job for a deadline, or check for
   `timeout` first.

9. **Job cost settles late.** `spendTime`/`cost` read at the moment a job leaves
   `Running` are interim — test3's figure run read 866s / ¥0.14 from the watcher
   but settled at 1155s / ¥0.20. Re-read `bohr job list` a few minutes later
   before recording costs.
