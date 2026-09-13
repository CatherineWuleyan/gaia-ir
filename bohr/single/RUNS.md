# Bohr single-pipeline batches — 2026-09-11

Two batches of `test3`, `test4`, `test5` through the **current working-tree**
single-paper pipeline (`agent-pipeline-v2.1-step1-5`), submitted as parallel
Bohrium Jobs.

* **Batch 1** — text inputs only (`ocr.md` + `claims_final.json`).
* **Batch 2** — same inputs plus the paper's figure images as
  `source.original_figure`, which enables step 2's vision fallback.

Everything else (image, machine type, project, command) is identical.

## Fixed environment

| Field | Value |
| --- | --- |
| image_address | `registry.dp.tech/dptech/ubuntu:ubuntu24.04-py3.12` |
| machine_type | `c8_m32_cpu` (8 cores / 32 GB, ¥0.64/h) |
| project_id | `4655935` |
| max_run_time | 240 min |
| nnode | 1 |
| command | `mkdir -p results && bash job_run.sh <paper> > results/<paper>.driver.log 2>&1` |
| backward_files | `results/` |

Runtime observed: Python 3.12.4, `gaia-lang==0.5.0a7`, deps from
`https://mirrors.aliyun.com/pypi/simple`.

## Source provenance

```
local_repo          /Users/catherinewu/Documents/gaia-ir
local_git_commit    300938c9d80c57b4000b12ac362da80f32873223
local_git_dirty     true
pipeline_config     pipelines/single_paper/pipeline.step1-5.json
config_sha256       031a65e87b02c5f20876372cb5ae860cc8bf27ff416f2e4da2a89bbcfc54c216
```

Source hashes verified identical to the working tree at submission time:

```
1edd700406459f5a7d51b80a3fcd5da9b4be5b84cb578ed7e721028131b05d5e  agent_pipeline_v2/step1.py
61cb510d1215efe367efc2f8586a30306dcd0fd344a3ff3ff5033887f4020fe3  agent_pipeline_v2/step2.py
2e1ca231827f3a3fb75d4f2851852f218a77643304e4c1dbdc8161ace706610d  agent_pipeline_v2/step3.py
0d7f8b259caa6309e6d2dff13e6485ac9567ecd484dc260d2a38a8a3bcc905a0  agent_pipeline_v2/step4.py
c1064c114027077c6b15f76e1bf0795ebaf7f344aa3de8f53aedff465dadfd3a  agent_pipeline_v2/view.py
```

## Jobs

| batch | paper | BOHR_ID | jobId | run_id | spendTime | cost |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | test3 | 20725139 | 23370553 | run_20260911T043911Z_e5a957f7 | 696s | ¥0.12 |
| 1 | test4 | 20725140 | 23370554 | run_20260911T044147Z_6871bc8d | 139s | ¥0.02 |
| 1 | test5 | 20725299 | 23370706 | run_20260911T061852Z_8b6a887a | 1706s | ¥0.30 |
| 2 | test3 | 20725913 | 23371367 | run_20260911T080347Z_607795ad | 1155s | ¥0.20 |
| 2 | test4 | 20725911 | 23371365 | run_20260911T080324Z_7768dd46 | 702s | ¥0.12 |
| 2 | test5 | 20725912 | 23371366 | run_20260911T080336Z_af285cb1 | 1933s | ¥0.34 |

All six: `exitCode=0`, `status=Finished`, harness `run_status=succeeded`, all six
stages on attempt 1, `pipeline_harness check` → `ok=True` with no errors.

Environment probes (`gaia-single-smoke`): `20725111` and `20725112` failed while
the container environment was being sorted out; `20725113` passed. ¥0.06 total.

All nine jobs together: **¥1.16** (batch 1 ¥0.44, batch 2 ¥0.66, probes ¥0.06).
These are Bohrium's settled figures; the platform finalises cost a few minutes
after a job leaves `Running`, so a watcher that reads the value at termination
sees an interim number (test3 batch 2 read 866s/¥0.14, settled at 1155s/¥0.20).

## Figures change the outcome only when the text alone is insufficient

| paper | batch | view nodes | view edges | step2 observ. | step4 strategies | vision calls | vision obs. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test3 | 1 | 167 | 118 | 34 | 17 | 0 | — |
| test3 | 2 | 172 | **125** | 33 | 16 | 2 | 0 |
| test4 | 1 | 26 | **0** | 8 | **0** | 0 | — |
| test4 | 2 | 53 | **29** | 11 | **3** | 3 | 2 figures |
| test5 | 1 | 343 | 384 | 45 | 37 | 0 | — |
| test5 | 2 | 352 | **393** | 44 | 45 | 1 | 0 |

* **test4 is the case that matters: 0 → 29 edges.** Its results exist only in
  plots, so with text alone 10 of 11 figure candidates returned
  `needed_context` ("the caption states the setup but not the measured result"),
  step 2 produced a single observation, and the relation pass found nothing.
* **test3 barely moves and test5 only mildly.** Their captions already state the
  numbers, so text extraction succeeds without vision (test3 got 33 of its
  figure observations from captions). Supplying figures is insurance, not a
  general improvement.
* Vision is not a rubber stamp: for test4 it read `_3.jpg` and `_4.jpg` and
  produced the Amazon-Reviews accuracy/transfer numbers, but rejected `_2.jpg`
  as "a schematic illustration of the subnetwork transfer process … contains no
  measured experimental result". test3's two vision calls and test5's one call
  also returned no knowledge rather than inventing any.

## Inputs

| paper | ocr.md sha256 | claims_final.json sha256 | figures |
| --- | --- | --- | --- |
| test3 | `ffaec335f2766dd2ce580c39ef355ed2e6b1d097dc9bdd5f072d96c2a5db2cdc` | `e751ff182105cf093cb421d263c8f42ad2ffdc97dd71b1b4e937569b0c50ccd9` | 12 |
| test4 | `1de82dc6c716fc0f19f12ebff1e0fb810d1dba8033f3ad2b9452fda0da6c5c77` | `84c99a097029cdda236fafded87b449c19e518e51a0eabd4e23e4bbe8e2f6892` | 4 |
| test5 | `186e0a2fa4304254c61ce0714257844273c83e5f0eec684046aeecf646d6b6cd` | `ae2617c66cacad458f5b6e005776861f81c9914645c2271c0205fb873ee9c31a` | 7 |

Manifests for batch 2 are generated by `build_input_manifest.py`, which adds one
`source.original_figure` per image with `metadata.source_filename` — the field
`step2._vision_candidate()` matches on. Every candidate `image_name` resolves:
test3 12/12, test4 4/4, test5 7/7.

## Outputs

`data/minibatch/<paper>/` now holds **batch 2** (same file set as test1/test2),
produced by `collect_results.py`:

| paper | ir_hash |
| --- | --- |
| test3 | `sha256:07699f4fb660227e0359d39efd8de4fa98fa0e76b7bfa8fb51806806757aa267` |
| test4 | `sha256:90e09a8a3c6685be3def6f97a63a0fb78f8840367f078feb09f8b89e0c84811f` |
| test5 | `sha256:5a5bb77ffabce276ce6234d01a3e7f43a719ef1eb26f8bfce1a6c3ca72fc6d8e` |

Raw bundles for both batches are kept separately under
`outputs/bohr_runs/<paper>/` (batch 1) and `outputs/bohr_runs/<paper>_fig/`
(batch 2), so the before/after comparison stays reproducible.

## Findings emitted at step 4 / step 5 (batch 2)

| paper | pruned observations | floating claims | graph components (largest) | other |
| --- | --- | --- | --- | --- |
| test3 | 9 | 1 | 7 (20%) | — |
| test4 | 1 | 3 | 5 (55%) | — |
| test5 | 4 | 2 | 4 (88%) | 7 × `STEP4_INSUFFICIENT_EVIDENCE` |

Warnings only; `pipeline_harness check` reports no errors for any run. test3 and
test4 are small papers (10 and 6 claims) so their reasoning graphs stay
fragmented; test5 (27 claims + 12 relations) is the best-connected.

## Viewer fix: contradictions rendered without an operand

test4's contradiction displayed only `claim_6` in the viewer, apparently
"missing its target". The run data was fine; the projection dropped the edge:

1. Step 4 records the relation `([claim_O03] 和 [claim_O04]) 与 [claim_6]
   不可同时成立` as a contradiction whose first operand is a synthesized helper
   with content `conjunction(claim_O03,claim_O04)`. Nothing derives that helper,
   so it had **no incoming edge** — its operands exist only in the content
   string.
2. The viewer hides helper nodes and contracts them by rewiring
   incoming × outgoing (`compactFormalGraph`, `viewer.html`). With zero incoming
   edges the product is empty, so the helper's only edge
   (`helper → contradiction`) was deleted with nothing replacing it.
3. The contradiction was left with a single input.

Fixed in `agent_pipeline_v2/view.py`: helper claims now get their operands
projected as input edges, so the contraction bridges `claim_O03` and `claim_O04`
onto the contradiction. Nested helpers (Step 4 folds long conjunctions into
`conjunction(helper_relation_1_1,claim_16)` chains) are flattened to non-helper
Knowledge IDs, because the viewer contracts hidden nodes in node order and a
helper-to-helper edge would make the result depend on that order.

The fix is projection-only — `pipeline_harness/runner.py` never projects a view,
so the compiled IR, the archived artifacts and the `ir_hash` values above are
unchanged. Effect on the three projected graphs:

| paper | view nodes | view edges | contradiction inputs |
| --- | --- | --- | --- |
| test3 | 172 → 188 | 125 → 146 | — |
| test4 | 53 → 56 | 29 → 38 | 1 → 3 (`claim_6`, `claim_O03`, `claim_O04`) |
| test5 | 352 → 382 | 393 → 453 | 2 (unchanged; no helper operand) |

Regression coverage: `pipelines/single_paper/tests/test_view.py` (operand
parsing, nested-chain flattening, cycles).
