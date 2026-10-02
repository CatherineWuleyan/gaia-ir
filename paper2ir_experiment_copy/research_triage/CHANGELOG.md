# Research triage change log

## 2026-10-02 — v2 tool-loop and fairness gate

- Added the provider-agnostic `live_tool_loop.py` preflight. It keeps the
  complete `merge.html` graph behind retrieval tools, records typed returned
  IDs, and validates against the runner-owned trace; the scripted tests make
  no network requests.
- Added `v2_fairness.py` and persisted its report in the v2 plan manifest.
  The three conditions now have mechanically checked paper hashes, corpus
  membership, resource isolation, graph identity, and equal 8-call/4000-token
  budgets. The current bundle is labelled `bridge_suitability=warning` because
  the focal paper already contains the likely answer terms.
- Added explicit opt-in loading of the provenance sidecar. Sidecar anchors are
  exposed as `derived` with a warning and never upgraded to direct
  `merge.html` provenance. The authoritative viewer remains `merge.html`.
- Added `lth_bridge_v2_bingo_task.json` and its 27-cell plan. The original
  rewinding bundle remains a leakage warning; the BINGO focal paper passes the
  current bridge-suitability diagnostic and is the next candidate for a live
  retrieval pilot.

## 2026-10-02

- Revised `reasoning_graph` so it no longer filters the model input to a
  preselected cross-paper claim. The treatment now renders the complete
  candidate closure as `evidence_graph.nodes`, `evidence_graph.edges`, and
  `source_anchors`; the model must retrieve and cite its own minimal subgraph.
  The full validator closure stays in `task.json` for post-hoc auditing, and
  each cell now saves the exact model-facing JSON as `model_input.json`.
- The revised six-cell DeepSeek run is in
  `evidence/lth_graph_vs_paper_pilot_deepseek_runs_20261002_graph_retrieval_escalated/`.
  All six cards passed validation; graph cards selected the external
  `p01_claim_5` bridge in every seed, while the output JSON and graph input
  are embedded in the Chinese report. The earlier un-escalated retry is kept
  as a DNS transport-failure audit and is not scored.
- Regenerated per-model reports with pretty-printed JSON. Reports for v4 and
  the two Qwen runs are explicitly labelled legacy direct-serialization when
  their saved artifacts predate `model_input.json`; they are not presented as
  measurements of the revised graph-retrieval protocol.
- Added a Chinese reading layer before the raw JSON: each cell now shows the
  task boundary, candidate claims/relations, the model's selected evidence,
  conditions, baseline/treatment/controls, metrics, expected observation,
  falsifier, decision rule, and auditable decision trace. Raw JSON remains
  unchanged in the cell directories for exact audit; the main report keeps
  only one integrated reading view to avoid repeated JSON blocks.
- After explicit destination-and-payload authorization, the DeepSeek v4.1
  profile completed the authoritative six-cell run in
  `evidence/lth_graph_vs_paper_pilot_deepseek_runs_20261002_v4/`. All six
  cards passed validation; the graph condition added `p01_claim_5` and
  `rel_large_batch_recovers_init` on every seed while preserving the
  `contrast` action.
- Trisol login and team quota were verified read-only. The team has 16 NVIDIA
  L20 GPUs on cluster `b1`; public cached checkpoints were fixed to
  `Qwen2.5-7B-Instruct` (`a09a3545`) and `Qwen3.5-35B-A3B-FP8` (`9d1823d2`).
  Both services were team-private and stopped after their runs; final service
  metadata is retained under the corresponding evidence directory.
- Added a private Trisol transport to `model_proposer.py`. It calls
  `trisol inference exec` and the Pod-local OpenAI-compatible endpoint instead
  of exposing a public URL or copying a provider token. The adapter strips the
  CLI's trailing `exit_code: 0` marker, records bounded transport metadata, and
  keeps each cell's prompt/response hashes.
- Added profile-specific Qwen context budgets: 7B uses 5,000 completion tokens;
  35B uses 1,500 because its served `max_model_len=4096` and graph prompt is
  about 2,436 tokens. The DeepSeek-only `thinking` field is omitted for Trisol
  vLLM requests. A first 35B request with 5,000 tokens returned HTTP 400 and is
  preserved as a transport diagnostic; it was not counted as a scientific
  result.
- The private Qwen 7B six-cell run is in
  `evidence/lth_graph_vs_paper_pilot_trisol_qwen7b_runs_20261002_v3/`: all
  cards passed, all chose `contrast`, and no graph card cited a cross-paper
  claim or relation. The Qwen 35B run is in
  `evidence/lth_graph_vs_paper_pilot_trisol_qwen35b_runs_20261002_v2/`: all
  cards passed, all chose `ablate`, and graph cards added only target-paper
  context (`p53_claim_16`) with no cross-paper relation. This is a model-aware
  evidence-use result, not a benchmark-wide causal gain; the fixture remains
  design-only because no executable LTH entrypoint is available.
- Added `research_triage/lth_report.py` and the combined three-model report at
  `evidence/lth_graph_vs_paper_pilot_multimodel_runs_20261002/`. It preserves
  paired cards, metrics, claim/relation differences and interpretation for
  DeepSeek v4.1, Qwen 7B and Qwen 35B.
- Added an offline regression test for the private Trisol adapter's CLI output
  framing. The full research-triage suite now passes: 147 tests (the existing
  1,504 NumPy deprecation warnings remain unrelated to this pilot).
- 将总报告及三个模型的 Markdown 报告改为中文；模型名、claim/relation ID、
  指标键和原始 JSON 结果保持不变，报告生成脚本也同步更新。
- 在报告中补充了输入闭包、JSON 输出结构、实验卡字段和数字指标定义；同时
  修正旧版汇总器没有读取 DeepSeek `claim_ids` 而导致 claim 行为空的问题。
- Preserved failed attempts instead of overwriting them: DeepSeek v1 (hidden
  reasoning consumed all 5,000 completion tokens), Trisol v1/v2 sandbox
  transport/config failures, and the Qwen 35B context-limit run. The full
  planned outputs are additive and versioned (`v1`, `v2`, `v3`).

## 2026-10-01

- Ran the 18-cell protocol with the explicit live-request acknowledgement.
  All cells reached the runner and wrote bounded failure artifacts, but no
  provider returned a model response: six DeepSeek calls failed at DNS
  resolution and twelve Qwen calls failed before transport because the
  `TRISOL_API_KEY` profile variable is absent. The run is therefore a
  connectivity/configuration diagnostic, not a scientific result.
- Fixed the protocol cell-directory layout after this run exposed a collision:
  `run_protocol` now passes `--out-is-cell`, so each invocation writes exactly
  `<task>/<model_profile>/<condition>/seed-<seed>` without duplicating the
  condition directory. Added regression coverage for the 18 unique paths and
  direct cell output mode.
- Transport and configuration failures now retain bounded provider metadata
  (profile, model, endpoint host, seed, task/prompt hashes, and request
  attempted) without storing credentials or response text. Failed cells leave
  `EvidenceIntegrity` and paired scientific deltas undefined instead of
  converting an outage into a false evidence-condition result.
- A controlled escalated connectivity probe reached `api.deepseek.com` and
  returned HTTP 401 without sending task content. The subsequent authorized
  18-cell execution was rejected by automatic review because the command
  would export the local LTH task/claims/prompt to the external endpoint;
  no DeepSeek prompt or response was stored. Explicit destination-and-payload
  authorization is pending.

- Live execution remains pending external-request authorization. The first
  direct DeepSeek probe was rejected by the local automatic review because it
  would send the frozen LTH task and evidence packet to an external endpoint.
  No provider request or model result is counted; the 18-cell manifest remains
  `planned_authorization_required`.
- Endpoint preflight found the configured DeepSeek environment at
  `agent/.env` resolves `DEEPSEEK_MODEL=deepseek-v4-flash-vision-exp` (the
  profile alias is `deepseek_v4_1`), while no `TRISOL_*` or Qwen endpoint
  variables are present in the workspace environment. The Trisol CLI also
  reports DNS failure for `trisol.dp.tech`; no Qwen request was attempted.

- Added the three-model LTH pilot manifest
  `evidence/lth_graph_vs_paper_pilot_multimodel_20261001/`: one concrete task,
  `paper_only`/`reasoning_graph`, seeds 0/1/2, and profiles
  `deepseek_v4_1`, `trisol_qwen7b`, and `trisol_qwen35b` (18 planned cells,
  zero API requests). Output identity includes the model profile so raw cards
  cannot overwrite another model's result.
- Added endpoint-agnostic model profiles to `model_proposer.py`. Credentials,
  URLs, and model IDs are environment-resolved and never written to metadata;
  audit metadata now records profile/provider/model, endpoint host, task/prompt/
  response hashes, seed, usage, and closed-book/tool flags.
- Added model-aware protocol validation, matrix expansion, and aggregation.
  `summarize_by_model` and `paired_bootstrap(..., model_profile=...)` preserve
  paired comparisons independently for each model while keeping legacy rows
  compatible. No model call was made while implementing this change.
- The evidence runner now persists the exact model-facing `prompt.txt` beside
  each task/proposal/metadata artifact (or a bounded `prompt_error.json`), so
  every live card can be regenerated against the recorded prompt hash.
- Offline verification: `py_compile`, plan-only expansion (18 cells), and the
  dependency-free matrix/runner subset pass. The full model/evidence pytest
  subset could not be collected in the current interpreter because
  `jsonschema` is absent; `paper2ir_experiment_copy/pyproject.toml` already
  declares `jsonschema>=4.18,<5`, and no network install was attempted.

- Added `docs/lth-pilot-10-question-catalog.md` with ten LTH bridge-required
  experiment questions, ranked by how stable their literature reference answer
  is. The first live candidate pool is still closed-book and keeps the main
  causal contrast to `paper_only` versus `reasoning_graph`.
- Added an optional visible `decision_trace` to proposal cards. It contains at
  most six short, typed steps (`evidence`, `bridge`, `condition`,
  `alternative`, `decision`, `uncertainty`) with auditable claim/relation/anchor
  references. The validator checks closure; it does not request or persist a
  private hidden chain-of-thought.
- Tightened the model prompt for the pilot: no browsing or external tools,
  facts outside the frozen task packet are unavailable, and insufficient
  evidence should produce `abstain`. The pilot matrix records
  `knowledge_access=closed_book`, `web_tools_enabled=false`, `tool_calls=0`,
  and the provider endpoint allowlist. Pretrained model memory remains a shared
  background and cannot receive evidence credit without an allowed anchor.
- Added `execution_mode=design_only` support for cards whose scientific design
  can be audited before an LTH training entrypoint exists. Such cards may set
  `experiment.entrypoint=null`; execution metrics remain undefined and the
  validator still rejects invented paths.
- Added the first closure-complete LTH design fixture for the rewinding versus
  initialization question. The runner now supports `paper_only` and
  `reasoning_graph` claim closures, so the same frozen task can be rendered
  with or without the external initialization/large-batch bridge claims.
- Narrowed that fixture to a concrete design-only setting: ResNet-18 on
  CIFAR-10, 30K training iterations, batch size 128, and a pre-registered 80%
  sparsity point, with k=0, k=500, and warmup as the candidate interventions.
  The 80% point is a pilot condition, not a claim copied from the paper; the
  eventual runnable implementation still needs a separate entrypoint audit.
- Added a frozen `current_state` and remaining-budget block to the LTH fixture.
  The agent now sees the reported k=0 failure and the unresolved rewind-versus-
  stabilization decision, while the action space remains open-ended rather
  than a supplied candidate list.
- Added the concise first-phase plan `docs/lth-graph-vs-paper-pilot.md`: one
  bridge-required LTH bundle (`rewinding vs initialization / mask semantics`),
  two conditions (`paper_only` vs `reasoning_graph`), and six plan-only cells
  (one bundle × two conditions × three seeds). The 89-paper native LKM graph is
  the retrieval corpus; the 73-paper cross-paper view is navigation only. This
  phase scores auditable experiment cards and does not claim LTH execution
  because no LTH training entrypoint is present.
- Audited the workspace graph inventory: `outputs/lkm_nonbig_view_model.json`
  contains 89 papers, 3,268 nodes, and 3,161 native relations with per-paper
  claim/anchor/provenance artifacts; `outputs/lth_cross_paper/view_model.json`
  is a 73-paper, 10-hypothesis design preview with 151 candidate edges and
  `validated=false`.

- Extended the research plan with an LKM end-to-end retrieval role: separate
  BM25/dense/hybrid/LKM retrieval evaluation from flat-vs-graph rendering,
  preserve retrieval traces, add stance/contradiction/condition metrics, and
  compare information-gain and cost-aware selection baselines. Added design
  implications from LitSearch, SciFact/SciFact-Open, G-Retriever, DiscoveryBench,
  BLADE, the ICLR 2025 idea-generation study, EXP-Bench, MLR-Bench, COLT
  sequential information maximization, AISTATS robust active learning, and
  CoRL active experiment selection.
- Attempted read-only Feishu discovery via `lark-cli` as bot. Bot identity was
  verified, but Drive/Wiki search was blocked by missing application scope
  `search:docs:read`; Wiki/IM calls also encountered Feishu account-service DNS
  failure. No writes, permissions changes, or publication calls were made.
- Revised `docs/autoresearch-icml-plan.md` around cross-paper bridge-required LTH
  tasks. The main condition no longer exposes a closed candidate set: the agent
  generates 3–5 arbitrary ideas, selects one, and is scored with hidden rubric,
  information gain, diversity, cost, and execution outcomes. Added S0–S3
  retrieval controls, decoy-connected graphs, LTH bridge task families, and
  `BridgeNecessityScore`; frozen candidate evaluation remains calibration-only.
- Audited the synchronized LKM notes and recorded that the 73-paper LTH graph
  is `design-preview`/`validated=false`; its candidate P→K edges are not gold
  until Step3/4 formalization and source-anchor audit. Live Feishu access was
  unavailable because `lark-cli auth status --json --verify` had no usable
  user identity and bot verification failed on DNS, so no Feishu writes or
  claims of live workspace coverage were made.
- Completed the explicitly authorized six-cell paper-derived pilot (two
  serializations, three seeds) in `evidence/paper_compact_live_20261001/`.
  All six cards were valid abstentions with empty evidence and null
  experiments; this verifies the conservative entrypoint/cutoff policy, not
  the paper's scientific conclusion.
- Hardened the plan-only/live runner with fixture task-id, manifest
  condition/seed, and expected entrypoint-policy cross-checks. This reuses
  existing manifest and task fields; no new card or protocol schema field was
  added.
- Added a regression test for the cross-checks; the isolated suite now passes
  121 tests.
- Verified from the publisher repository that the evaluator harness and
  command are public; the remaining ScienceAgentBench blocker is the missing
  `benchmark_verified.zip` data/gold-program/scoring-rubric archive.
- Rechecked the publisher download link through the public repository; it
  redirects to Microsoft login and exposes no direct unauthenticated archive
  URL in this environment.
- Added an optional `experiment.candidate_id` contract for tasks that provide a
  frozen candidate-experiment set.
- Added `candidate_evaluator.py`, which scores candidate choice against a
  separate evaluator-only oracle (`a_tier_candidate_ids`, accepted choices,
  preferred order, and cost) while keeping oracle labels out of the model
  prompt. Tasks without this annotation keep the metric explicitly undefined.
- Integrated `DecisionChoiceValid@1`, `DecisionQuality@1` (Top-1 A-tier),
  acceptance, abstention, and cost-regret fields into the six-stage score
  artifact and added regression tests for correct, suboptimal, mismatched,
  unknown, and abstaining choices; the isolated suite now passes 127 tests.
- Candidate evaluation now also checks declared candidate conditions against the
  proposal's top-level condition tuple, preventing a candidate-ID/condition
  mismatch from receiving selection credit.

## 2026-09-30

- Completed the explicitly authorized 12-cell compact-flat pilot (two tasks,
  two serializations, three seeds) in a new result directory without touching
  historical outputs. Added per-cell audit retention, bounded protocol failure
  classes, paired bootstrap summaries, and a plain-language README. The graph
  condition was valid in 6/6 cells versus 4/6 for compact-flat; the result is
  descriptive and is not reported as ScienceAgentBench TaskSuccess.
- Rechecked the official ScienceAgentBench SharePoint artifact without making
  model requests: browser navigation timed out again and no local archive was
  produced, so the official-evaluator gate remains open.
- Searched the isolated checkout and workspace for a hidden SAB archive or
  `evaluation.harness.run_evaluation`; only the public annotation parquet,
  exports, and prepared argv manifests were present.
- Added `model_proposer.py`, an opt-in DeepSeek adapter that receives one
  frozen task/evidence JSON and writes `proposal.json`, `metadata.json`, and
  the raw response with prompt/response hashes.
- Added an explicit output skeleton to the model prompt after the first live
  call omitted required top-level fields and misplaced conditions inside the
  experiment object.
- Added JSON Schema validation through `jsonschema`, including the short
  `reason` field and per-evidence `relation_ids`.
- Added relation closure, source-anchor closure, required-condition closure,
  duplicate-condition checks, and a genuine `abstain` path.
- Added task-declared `action_requirements` so an action must cite the required
  evidence roles; `CheckerUnsupportedDecisionRate` now reflects this decision
  gate instead of treating every schema-valid card as supported.
- Added task-declared `allowed_entrypoint_refs`; the model must choose an
  executable entrypoint already present in the frozen task closure.
- Added a public MLAgentBench vectorization task fixture and a live generated
  card. It selects `ablate`, cites the API support and loop limitation claims,
  and makes numerical equivalence a falsifier before runtime is accepted.
- Added `evidence_mode_pilot.py` and a four-condition live pilot on that same
  task. Direct abstained under missing evidence; Flat-Claims, Paper2IR, and
  Paper2IR+Gate generated mechanically valid cards. The result is retained as a
  one-task pilot, not a general effect estimate.
- Extended the paired aggregate report with `CardValid@1` and
  `DecisionSupported@1`; the pilot's Direct-versus-Paper2IR+Gate card-validity
  delta is 0.0 because both cards passed, while their actions differ
  (abstain versus contrast).
- Corrected the evidence-mode ablation so `allowed_entrypoint_refs` stays
  fixed in every evidence condition; only graph/gate fields are changed.
- Scoring correction history: an intermediate revision made ungated input
  conditions unscorable. A subsequent revision added an offline reference
  scorer but also required relation IDs, producing a -1.0 flat->graph delta.
  That delta is WITHDRAWN: the post-hoc rule penalized a format that was
  unavailable to flat claims, not an unsupported scientific decision.
- Removed the relation-count requirement from the active fixtures and
  validator. Every condition now uses the same offline reference contract;
  optional relation citations earn no extra credit. Existing proposals are
  rescored, not regenerated. Both vectorization and synthetic fixture pilots
  now have a descriptive flat->graph delta of 0.0. Neither establishes graph
  benefit. Existing historical task/metadata/raw responses remain unchanged,
  including the fixture gated input that contained the withdrawn requirement.
- A separate four-condition DeepSeek fixture call was made before withdrawing
  that rule. It checks plumbing only, has no claim text to adjudicate, and is
  not an independent scientific replication or official benchmark result.
- Added metamorphic tests for identical scoring across conditions, optional
  relation IDs, fabricated anchors, missing action rules, malformed model
  values, transport failures, and safe abstention. The validator now rejects
  malformed JSON fields without crashing and separates schema validity from
  task-closure validity. Invalid evidence cannot satisfy a gate by role name.
- Preserved `null` for missing support rules and abstention in decision-support
  scoring and downstream pipeline metrics. Safe abstention is zero unsupported
  actions, not a supported experiment. Runner refuses to overwrite outputs.
- Bumped the persisted validator identity to `validate_proposal@0.3.0` for
  fail-closed malformed-card handling, undefined decision-support states, and
  the shared visible/reference closure scores.
- Single-pair aggregate CIs are now null; a degenerate interval is not a
  population uncertainty estimate. No new schema, artifact kind, or tool was
  retained; the experimental relation-count field and derived metric were removed.
- Added explicit optional API seeds to the model proposer and evidence-mode
  runner. Metadata and paired rows now preserve the requested seed; the
  existing live artifacts predate this option and must not be described as
  seeded replicates. Output directories refuse overwrite and invalid seeds are
  rejected before any network call; the single-card CLI also refuses to
  overwrite a non-empty output directory.
- Distinguished the row pairing seed from the API sampling seed. Historical
  live artifacts now record `pairing_seed=0` and `api_seed=null`; the model
  name is recovered from preserved metadata rather than guessed.
- Generated a complete 102-row ScienceAgentBench verified-split command
  manifest through the existing official adapter. It is explicitly marked
  `commands_prepared_not_executed`; no prediction, evaluator score, or
  TaskSuccess value was invented.
- Added a mocked transport regression test proving an explicit API seed is
  sent and persisted in metadata without making a network call.
- Added an optional `--entrypoint-root` audit input. When an official sandbox
  is supplied, the validator resolves the relative entrypoint inside that root
  and records `entrypoint_exists`; the model/task schema is unchanged and the
  default remains a non-executing closure check.
- Added optional task-side time-cutoff enforcement. A task can declare an
  ISO-date `evidence_cutoff` and per-anchor `available_at` metadata; future or
  missing anchor dates fail deterministically for the entire frozen evidence
  set, not only for anchors the model cites. Existing vectorization data has no
  invented cutoff, while the fixture now exercises the gate.
- Tightened experiment-card checks: baseline and treatment must differ,
  controls cannot be empty, and an identical expected observation/falsifier is
  rejected. A literal metric-name overlap in the falsifier is reported as an
  audit warning only; no lexical check is treated as scientific entailment.
- Normalized punctuation when checking that a falsifier mentions a metric;
  `validation_accuracy` and “validation accuracy” are treated as the same
  audit token, without changing the non-semantic warning boundary.
- Added a public ScienceAgentBench verified instance-1 (ClinTox) task fixture
  and one seed-17 DeepSeek card. It selects a controlled contrast between
  multitask ECFP and single-task baselines, cites a limitation that the
  annotation contains no comparative result, and remains explicitly
  annotation-derived rather than a gold plan. Official execution is still
  unavailable; the metric-overlap warning is retained for semantic review.
- Added a live fixture generation artifact under
  `evidence/model_proposer_fixture/`; the second call passed deterministic
  validation. The first invalid response remains documented in the run notes,
  rather than being counted as a successful card.
- Corrected the single-card CLI so a non-empty output directory is rejected
  before credentials are loaded or an API request is made; initialized the
  falsifier audit metric for valid abstentions so safe abstain cards cannot
  crash validation.
- Verified the pinned SAB evaluator source and documented its native
  `success_rate`/`valid_program`/`codebert_score` JSONL fields. The generic
  binary `task_success` parser intentionally does not coerce `success_rate`.
  Prepared command rows now use absolute evaluator/prediction/log paths, and
  the manifest test checks all 102 annotation IDs plus its SHA256.
- Added an audit README for the SAB instance-1 four-condition pilot. The
  flat/paper2ir failures are retained as exact-label closure mismatches, not
  reinterpreted as a graph benefit; requested API seed 17 is recorded but
  server-side reproducibility remains unverified.
- Tightened cutoff parsing to require the documented extended `YYYY-MM-DD`
  form rather than accepting compact dates accidentally accepted by Python.
- Added a frozen paper-derived task for *Playing Lottery Tickets in Style
  Transfer Models*. Its claims point to explicit OCR line anchors and its
  cutoff is the 2022-10-14 publication date. Because the allowed closure has
  no runnable AdaIN/SANet implementation or matched test assets, the retained
  reference card correctly abstains and names acquisition of a reproducibility
  package as the next experiment. A seed-23 DeepSeek request failed at DNS
  resolution before a response was received; this is recorded separately and
  is not counted as a model decision.
- Added `source_anchor_manifest.json` beside the paper reference card with the
  frozen OCR SHA256, declared line spans, and compact provenance snippets; the
  proposal schema remains unchanged.
- Recorded a post-cutoff external reproducibility search in
  `reproducibility_search.json`. It found no public official code listing, is
  excluded from the agent prompt, and only strengthens the audit rationale for
  the existing abstention card.
- Added a closure-complete synthetic contrast test for the paper task. It is a
  validator regression fixture, not a model response or executed experiment;
  it proves that a future runnable entrypoint can produce the required support,
  limitation, conditions, controls, falsifier, and budget fields.
- Added a prompt-contract regression test ensuring the paper task's cutoff,
  claim anchors, relations, and empty entrypoint closure are actually exposed
  to the proposer without adding hidden fields or chain-of-thought output.
- Tightened entrypoint closure: a non-abstaining proposal is rejected when the
  task declares no allowed executable entrypoint. The paper task's synthetic
  future-path test now proves it rejects the current empty closure and passes
  only after an explicit entrypoint is added.
- Updated the paper request failure trace with four seed-23 attempts, including
  the approved diagnostic: `finish_reason=length` and 5000 completion tokens.
  No response artifact or hidden reasoning was stored.
- Added an information-matched `flat_equivalent` proposer condition. It keeps
  the full claim/anchor/relation/condition/action-rule closure for validation,
  but renders a flat evidence ledger to the model; existing proposal and rows
  formats remain unchanged.
- Added an explicit `--condition` selector to the existing evidence-mode runner
  so a requested paired run cannot accidentally fan out to all conditions. The
  interrupted seed-1 directory is preserved and excluded from analysis.
- Bounded malformed-response diagnostics now record only content length and
  fenced-JSON presence, never response text or hidden reasoning.
- Completed the approved information-matched pilot: 12 target rows across two
  tasks, two representations, and three seeds, with 16 total requests after
  the runner fan-out correction. The audit manifest reports descriptive
  `CardValid@1`/`EvidenceIntegrity` deltas and keeps official TaskSuccess null.
- Added an offline prompt audit to the information-matched manifest. It records
  equal semantic closure counts for claims, anchors, and relations, while
  explicitly noting that the flat ledger was longer in both pilot tasks; the
  comparison therefore does not isolate token length or budget effects.
- Added a bounded `ModelProtocolError` path. Malformed or empty completions can
  now retain model/seed/usage/length metadata in `error.json` without retaining
  response text or hidden reasoning; two regression tests cover the exception
  and runner behavior. The isolated test suite now passes 104 tests.
- Implemented the four preregistered evidence ablations in the runner:
  `without_condition`, `without_provenance`, `without_source_gate`, and
  `without_dependency`. They are explicit selectable conditions, while the
  default runner remains limited to the five base conditions to prevent
  accidental API fan-out; paired scoring continues to use the complete task.
- Added an explicit, non-default `text_rag` baseline. It exposes retrieved
  prose but no claim, relation, condition, or provenance closure, making any
  structured citation in a non-abstaining card auditable as unsupported rather
  than silently granting the baseline graph fields.
- Added a non-default `flat_compact` condition that serializes each claim and
  relation once. Offline prompt audit shows a 126–159 character gap against
  the graph prompt on the two pilot tasks; no model calls have been made for
  this candidate yet, so the historical pilot statistics are unchanged.
- Added `prompt_audit.py`, a no-network audit that hashes normalized evidence
  closures and reports prompt length/word deltas for selected conditions. Its
  regression tests verify compact-flat closure equivalence and the historical
  flat-ledger length imbalance.
- Added an explicit runner `--request-budget` guard. A selected condition set
  that exceeds the declared request count now fails before credentials, output
  creation, or model invocation; the live pilot command can therefore enforce
  one API call per task/condition/seed.
- Added `evidence/information_matched_compact_pilot/` as a clearly marked
  authorization-required protocol for the next 12-cell graph vs compact-flat
  run. It contains no model outputs and is kept separate from the historical
  16-request result manifest.
- Added descriptive `RelationCitation@1` reporting for valid cards. Relation
  use remains optional and earns no validity or decision-support credit; the
  metric only records whether a valid card actually cited a graph relation.
- Added protocol tests for the pending compact pilot: all 12 cells are unique,
  fixtures exist, each cell has request budget 1, no model output is present,
  and both tasks pass the offline closure/length preflight. The isolated suite
  now passes 110 tests.
- Added deterministic `audit.failure_class` labels (`protocol_empty`,
  `protocol_non_json`, `transport`, `validator`, and related classes). These
  labels are diagnostic only and keep protocol failures separate from scientific
  decision metrics; the isolated suite now passes 111 tests.
- Tightened the model-facing prompt for empty executable closures: when
  `allowed_entrypoint_refs` is empty, the proposer is explicitly instructed to
  abstain rather than copy the skeleton's placeholder path. The validator and
  prompt now enforce the same paper-task boundary.
- Added a no-network 12-cell protocol smoke: a local fake proposer exercises
  every planned task/condition/seed directory and the one-request budget guard,
  without creating any live model artifact. The isolated suite now passes 112
  tests.
- Added a separate six-cell authorization-required protocol for the disputed
  paper task. It keeps the manual abstention card as the current reference and
  declares the expected no-entrypoint policy (`abstain`, no experiment) before
  any live request is made.
- Extended the no-network protocol tests to cover the paper six-cell plan and
  its explicit empty-entrypoint/abstention policy. The isolated suite now
  passes 113 tests.
- Added `evidence/goal_requirements_audit.json`, a checkpoint-level audit that
  marks the structured card, gates, modification log, and regression suite as
  verified while keeping the paper live check and official benchmark result
  explicitly incomplete until direct evidence is available.
- Documented the two authorization-required protocols and the checkpoint audit
  in `evidence/README.md`, so planned cells cannot be mistaken for model or
  benchmark results.
- Rechecked ScienceAgentBench access against the official repository README:
  `benchmark_verified.zip` is now documented as released, but its SharePoint
  link redirects to Microsoft login in this environment. Added a dated access
  record and kept official execution/TaskSuccess unverified.
- Added `evidence/STATUS.md`, a human-readable checkpoint that separates
  verified behavior from deliberately unclaimed results and lists the exact
  post-authorization protocol steps.
- Added `research_triage.run_protocol`, a plan-first protocol runner. It
  validates cell uniqueness, fixture paths, zero prior requests, and a
  one-request budget; live execution requires both `--execute` and an explicit
  `--acknowledge-live-requests` flag.
- The isolated regression suite now passes 116 tests after covering the
  plan-only runner and its live-execution guard.
- Ran plan-only preflight for both pending protocols: 12 benchmark cells and 6
  paper cells expanded successfully, with no output directories or model calls.
- Added a regression test proving `--execute` without
  `--acknowledge-live-requests` exits before creating any output directory.
- The isolated regression suite now passes 117 tests.
- The live runner now refuses a non-empty output root, with a regression test,
  so a repeat run cannot mix new responses into an existing evidence directory.
- The isolated regression suite now passes 118 tests.
- Rechecked the official SAB archive through the in-app browser: the zip is
  previewable, but no local download was confirmed (browser download event
  timed out and shell DNS resolution failed). The access records now state
  this more precisely; no evaluator claim was added.
- Documented the plan-only/live-confirmed `run_protocol` commands in the
  research-triage README for safe handoff.
- Added a no-network six-cell paper-protocol smoke proving that a valid
  abstention remains valid across both representations and all three seeds.
- The isolated regression suite now passes 119 tests.
- Added a manifest/preflight consistency test so recorded prompt lengths and
  graph/flat character gaps cannot drift from the current offline audit.
- The isolated regression suite now passes 120 tests.
- Tightened the model-facing decision rule to prefer the highest expected
  information gain per unit of declared budget, while retaining the strongest
  alternative and falsifier constraints.
- Refreshed the offline compact-prompt audit values after the decision-rule
  wording change; semantic closure hashes and graph/flat character gaps remain
  unchanged.
- Approved v2 research protocol and updated `docs/autoresearch-icml-plan.md`,
  `docs/lth-graph-vs-paper-pilot.md`, and `docs/research-plan.md`. The model
  input is now specified as focal-paper full text plus a controlled retrieval
  interface; fixed action lists and candidate experiments are evaluator-only.
  The formal comparison is paper-only vs external-text retrieval vs
  cross-paper-graph retrieval under matched call/token budgets. The historical
  six-node design-only run is explicitly labeled a legacy protocol and is not
  treated as evidence of decision-quality improvement.
- Adopted `/Users/catherinewu/Documents/gaia-ir/merge.html` as the sole final
  graph source and extracted a locked snapshot under
  `evidence/lth_final_graph_20261002/`. The audit records 240 nodes, 235 edges,
  103 connected components, 102 isolated nodes, zero source-anchor coverage,
  and unknown paper-level cross-paper provenance. Added deterministic
  `search_claims`, `expand_neighbors`, and non-fabricating `get_source_anchor`
  helpers; the graph is blocked from evidence scoring until anchors and paper
  provenance are restored.
## 2026-10-02 — v2 retrieval and provenance preflight

- Added `v2_protocol.py` and `proposal_v2.schema.json`: all conditions expose
  the full target-paper text, while the model chooses free-form candidate
  directions and a `next_step`; legacy action enums, candidate IDs and the
  decision oracle stay evaluator-only.
- Added `v2_retrieval.py` and `v2_plan.py`. The local tools separate paper
  search, frozen external-text search, and the complete `merge.html` graph
  backend. Every call records returned IDs and token/call budgets; the plan
  directory `evidence/lth_graph_v2_plan_20261002/` contains 27 no-network cells
  for three conditions, three seeds and three model profiles.
- Added `provenance_map.py` and the `lth_final_graph_20261002` sidecar audit.
  The graph has 220 derived nodes and 20 unresolved nodes; 125 derived edges
  and 110 unresolved edges; direct provenance remains zero. Graph evidence
  credit is therefore disabled until a source-grounded map is explicitly
  reviewed.
- Added `v2_fairness.py` and `test_v2_fairness.py`. The plan now persists a
  mechanical audit showing the three conditions share the same full target
  paper, 8-call/4000-token retrieval budget, and focal-excluded 88-paper
  corpus. The audit also records the current graph-provenance warning and the
  fact that the required focal paper itself contains rewinding, initialization,
  warmup, and schedule terms that may make this bundle answerable without a
  cross-paper bridge.
