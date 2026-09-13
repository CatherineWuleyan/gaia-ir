# Bohr pipeline_merge batches — test1–test5 domain graph

`pipeline_merge` (Pipeline 8.0, Step 0–5) Bohrium Jobs over the five minibatch
paper packages `test1`…`test5`, in `bootstrap` mode, projecting the domain graph
viewer.

| run | BOHR_ID | code | integration edges | status |
| --- | --- | --- | --- | --- |
| v1 | 20726924 | anchor-on-smallest-package retrieval | 26 | Finished, superseded |
| v2 | 20726977 | equal-pool retrieval (every Package anchors) | 199 | Finished, superseded |
| v3 | 20729724 | + relation gate / abduction shape / star merge | — | **Failed at Step 5** (multi-premise abduction) |
| v4 | 20730467 | canonical single-observation collapse | — | Killed (design superseded) |
| v5 | 20730474 | summarize-N-observations-into-A | — | Killed (concatenated A) |
| **v6** | **20730477** | **A = summarized observation premise + mechanical guard** | **102** | **Finished, current** |

## Fixed environment

| Field | Value |
| --- | --- |
| image_address | `registry.dp.tech/dptech/ubuntu:ubuntu24.04-py3.12` |
| machine_type | `c8_m32_cpu` (8 cores / 32 GB, ¥0.64/h) |
| project_id | `4655935` |
| max_run_time | 480 min |
| command | `mkdir -p results && bash job_run.sh > results/merge.driver.log 2>&1` |
| backward_files | `results/` |
| merge model | `deepseek-v4-flash` (`DEEPSEEK_MERGE_MODEL` unset) |

Runtime observed: Python 3.12.4, `gaia-lang==0.5.0a7`, deps from the default
Tsinghua mirror.

## Source provenance

```
local_repo          /Users/catherinewu/Documents/gaia-ir
local_git_commit    300938c9d80c57b4000b12ac362da80f32873223
local_git_dirty     true
pipeline_config     pipelines/merge/pipeline.test1-5.json
config_sha256       066fa283801eb636a3be9a441c6fec6baf0e73fd409505dcad7b0330bd0780f5
staged_payload      /tmp/gaia_bohr_merge_test1-5 (3.2 MB)
llm concurrency     GAIA_MERGE_RETRIEVAL_WORKERS=16, GAIA_MERGE_JUDGE_WORKERS=8
```

## Jobs

| run | BOHR_ID | jobId | run_id | stages | spendTime | cost |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 20726924 | 23372406 | run_20260911T112253Z_2472e671 | all 6, attempt 1 | 499 s | ¥0.08 |
| v2 | 20726977 | 23372454 | run_20260911T115913Z_429941da | all 6, attempt 1 | 2772 s | ¥0.49 |

Both: `exitCode=0`, `run_status=succeeded`, `pipeline_harness check` → `ok=true`.
A local preflight (repo `.venv-gaia-a7`, same staged tree) also ran steps 0–2 and
a full run for each generation to shake out input/config problems.

## Inputs (frozen)

| folder | package | gaia.ir sha256 |
| --- | --- | --- |
| test1 | `papers:paper_24bac745` | `6a5edbbe…c1ac5a` |
| test2 | `papers:paper_f519c93b` | `8e750140…6a8a2f` |
| test3 | `papers:paper_607795ad` | `919d1577…d10cd87` |
| test4 | `papers:paper_7768dd46` | `83f47892…dee66866` |
| test5 | `papers:paper_af285cb1` | `46bf8e13…e86d45ff` |

Each package also contributed `formalization.json` and `knowledge.index.json`
(15 artifacts total); Step 1 bound all of them to exactly one gaia.ir Package.

## v1 → v2: equal-pool retrieval

`step3._retrieve_related()` used to anchor only the **smallest** Package's public
claims against all other Packages. That covers every cross pair for a
two-Package bootstrap, but with five Packages it only judges pairs that touch
`test4`: 1,971 of 16,112 cross-package pairs (**12.2 %**). Relations such as
`test1 × test2` (GNN efficiency; 559 lexically overlapping claim pairs) or
`test3 × test5` (lottery tickets / stable subnetworks) were never retrieved.

v2 anchors **every Package** into the same shared pool. Nothing else about the
prompts changed: retrieval is still the "find this anchor's related claims"
form (high recall), and the existing sorted-`seen` set keeps each unordered pair
single when both sides nominate it. Concurrency became
`GAIA_MERGE_RETRIEVAL_WORKERS` / `GAIA_MERGE_JUDGE_WORKERS` because the cost is
remote HTTP latency, not local compute.

Also fixed: Step 3 now applies Step 4's analogy arity (`evidence ≥ 2`) itself,
instead of emitting single-premise analogy weakpoints that Step 4 always drops
with `STEP4_WEAKPOINT_NOT_EXPANDED`.

Result of the change:

| metric | v1 | v2 |
| --- | --- | --- |
| cross-package pairs eligible | 12.2 % | 100 % |
| Step 3 operators | 0 | 3 |
| Step 3 weakpoints | 21 (15 analogy, 5 abduction, 1 infer) | 141 (139 abduction, 1 deduction, 1 infer) |
| Step 4 integration edges | 26 | **199** |
| domain graph | 303 nodes / 209 edges | 303 nodes / **382 edges** |
| compiled IR | 37 knowledges / 26 strategies | 512 knowledges / **199 strategies** |
| findings | 15 analogy-arity warnings | 2 `STEP4_CYCLIC_STRATEGY_SKIPPED` + 1 `GRAPH_DISCONNECTED` (warning) |
| wall clock / cost | 499 s / ¥0.08 | 2772 s / ¥0.49 |

## Result

`outputs/bohr_runs/merge_test1-5_v2/result/unpacked/results/`:

| file | content |
| --- | --- |
| `domain_graph.html` | self-contained domain graph viewer (dagre + view inlined) |
| `view_model.json` | projected view document |
| `check.json` | `ok=true`; one `GRAPH_DISCONNECTED` warning (95/97 nodes in one component) |
| `merge_run/` | full run: frozen inputs, artifacts, work files, views |

Domain graph: **303 nodes / 382 edges** — 183 paper-internal operator edges plus
**199 integration edges** (97 nodes in the merge-delta layer). Compiled
integration Package (`integration:lottery_ticket_hypothesis`),
`ir_hash=sha256:90c20c23be94ba3bd3b5de5c079f15660c5c9d2bbd5bee3abf1c35ff32a5589b`.

### Integration relations (samples)

```
conjunction    paper_f519c93b::claim_O57  +  paper_607795ad::claim_O18
contradiction  paper_24bac745::claim_13   ×  paper_af285cb1::claim_O02
contradiction  paper_f519c93b::claim_E04  ×  paper_24bac745::claim_13
abduction      Obs: paper_f519c93b::claim_58 (LTH not supported for graph models)
               Hyp: paper_607795ad::claim_O05 (subnetwork collapse at high pruning ratios)
abduction      Obs: paper_af285cb1::claim_9  (resetting becomes critical at high sparsity)
               Hyp: paper_7768dd46::claim_4  (dense and IMP cases sit in separate minima)
```

### Domain conclusion (candidate K)

> Across the evaluated pruning and sparse-subnetwork studies spanning graph,
> convolutional, speech, and implicit neural architectures, sparsity yields
> baseline-comparable or improved behavior only within a bounded
> moderate-to-high regime, while beyond a task- and architecture-dependent
> critical point performance, mask specialization, and robustness to
> initialization or transfer degrade together; fine-tuning, retraining, or
> deliberately stable/distilled mask selection can extend the viable regime but
> does not eliminate the threshold.

## Open observations

1. **Abduction now dominates the delta** (139 of 141 weakpoints). It reads
   plausibly on samples, but the equal pool gives the judge many more
   observation/hypothesis pairs, so the delta may be over-generated relative to
   the conservative v1. Reviewing a sample before publishing is worthwhile.
2. **LLM variance remains material.** The local preflight of the *unchanged* v1
   code produced 2 weakpoints against the remote v1's 21, from identical frozen
   inputs at temperature 0. Tier 2 makes that worse, so treat a single run as
   one sample, not a fixed result.
3. **Only 3 operators** (1 conjunction, 2 contradiction) came out of the
   deterministic-operator gate; nearly all cross-paper structure went to
   weakpoints.

## v3 → v6: making the integration delta defensible

v2's 199 integration edges were dominated by 139 abductions, and a per-edge audit
(`abduction_review.md`) showed 76% of them were not abductions at all:

| shape (v2 abductions) | count | share |
| --- | --- | --- |
| observation → hypothesis (valid) | 34 | 24% |
| observation → observation | 17 | 12% |
| general claim → observation (reversed) | 30 | 22% |
| claim → claim (no observation premise) | 58 | 42% |

Three fixes now guard the delta, all in `pipeline_merge/step3.py`:

1. **Abduction shape.** The premise must be an observation (`claim_O*`/`claim_E*`
   or an explicit `observation_claim`), the conclusion must not be one, and there
   must be exactly one premise (the official compiler accepts only "observation
   plus optional alternative explanation").
2. **Summarized premise A.** A hypothesis supported by N observations is no
   longer published as N abduction edges. The observations are compressed into
   one integration-owned observation claim `A` (`candidate_A_*`), `A -> H` is the
   abduction, and `observations -> A` is kept as a bounded deduction so every
   source observation stays reachable. One `A` is computed per unique observation
   set and shared by every hypothesis that set supports.
3. **Mandatory relation kind.** The judge must label each candidate
   `explains` / `consistent_with` / `none`. Only `explains` can become an
   abduction; `consistent_with` between two observations becomes a conjunction
   operator (equivalence would merge identities, which is too strong); the rest
   are dropped.

A `"summary"` that is merely the observations glued together is rejected
mechanically (`_is_summary_acceptable`): more than 30 words, two or more clause
joiners (and/while/whereas/but), an 8-word verbatim span copied from a source
observation, or a study-specific number all fall back to a single deterministic
observation.

### v6 (current)

| metric | v2 | v6 |
| --- | --- | --- |
| cross-package pairs eligible | 100% | 100% |
| step3 operators | 0 | 64 (56 convergence conjunctions, 8 contradictions) |
| step3 weakpoints | 141 (139 abduction) | 35 (24 abduction, 10 summary deduction, 1 infer) |
| summarized premises A | — | 12 entries → 10 unique, one per observation set |
| integration edges | 199 | 102 |
| domain graph | 303 nodes / 209 edges | 313 nodes / 285 edges |
| compiled IR | 512 knowledges / 199 strategies | 250 knowledges / 83 strategies |
| `check` | ok, 15 arity warnings | ok, 1 `GRAPH_DISCONNECTED` (92/94 nodes) |
| wall clock / cost | 2772 s / ¥0.49 | 2307 s / ¥0.38 |

`ir_hash=sha256:3a085bfc2405286c27a7175d120ca0c889f29c0247e3802f2ddadd20dfcea047`.
The 10 summarized premises are listed in `summary_premises_A.md`; each is ≤22
words and names no dataset, model or metric.

### Still true after the fixes

1. **The gate is very noisy.** Three identical gate calls over the same 20
   candidates returned 20 / 13+7 / 20 `not_operator`, so run-to-run delta size
   varies materially even with the shape checks in place.
2. **`consistent_with` conjunctions are truthful but weak.** 56 of the 102 edges
   only assert that two observations both hold; they are not explanatory links.
3. **The 49 `infer` edges from the domain conclusion K** are the pre-existing
   K→supporting-claims encoding, not new cross-paper reasoning.

## v3 → v8: delta 迭代对比

每个版本的 **delta 子图**（论文层默认关闭）与构成图：

| 版本 | delta 边 | delta 节点 | 边构成 | 说明 |
| --- | --- | --- | --- | --- |
| v1 | 26 | 22 | abduction 5, infer 21 | 只锚最小包，覆盖 12.2% 跨论文对 |
| v2 | 199 | 97 | abduction 137, deduction 1, infer 61 | 平等池，abduction 泛滥 |
| v3 | 141 | 73 | conjunction 47, contradiction 1, abduction 50, infer 43 | 形状校验+relation；step5 编译失败 |
| v6 | 166 | 94 | conjunction 56, contradiction 8, abduction 24, deduction 29, infer 49 | A 概括前提；56 个 conjunction 是误用 |
| v7 | 221 | 161 | contradiction 3, abduction 22, deduction 45, infer 151 | 趋同汇入 K，K 星形 151 条 |
| **v8** | **77** | **67** | **conjunction 1, contradiction 5, deduction 52, abduction 19** | **多层结论树，premises→Ki** |

产物：`outputs/merge_delta_comparison/`
（`index.html` 汇总表+构成图，`delta_v{1,2,3,6,7,8}.html` 为各版本只看 delta 的 viewer，
`delta_chart.svg` 为边构成堆叠柱状图）。

### v8 结构验证

| 检查项 | 结果 |
| --- | --- |
| 结论树层数 | 2（L1 11 个 + L2 根 1 个） |
| 每层 premises 上限 | L1 最大 8 = `CONCLUSION_CAP` ✅ |
| 方向 | `premises ──deduction→ Ki` ✅（21 条 deduction） |
| conjunction 边 | 1（v6 的 56 降到 1） |
| delta 总量 | 77（v7 的 221 降到 77） |
| `check` | ok，1 条 `GRAPH_DISCONNECTED`（7 个分量，最大 39%） |

### v8 暴露的两个结论树缺陷

1. **6/11 个 L1 结论只有 1 条前提。** 单前提的"有界总结"不是总结，只是把一条 claim
   换个 id 重述。这些单点组来自"只出现在 A 概括 abduction 里的假设"（其观察侧是
   `candidate_A_*`，不在 `records` 中，被过滤后只剩假设本身）。修法：分组时跳过
   premises < 2 的组。
2. **根节点只覆盖 3/11 个 L1 结论。** L2 把 11 个成员按 cap 切成 [8, 3] 两组，其中
   8 前提那组合成返回 null，于是 `parents` 只剩 1 个，循环退出——另外 8 个 L1 结论
   悬挂无父节点，树不完整。修法：某组合成失败时把其成员提升到上一层（而不是丢弃），
   保证根覆盖全部存活节点；或在组过大失败时递归二分。
