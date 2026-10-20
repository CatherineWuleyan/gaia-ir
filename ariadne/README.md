# Ariadne

Personal-library mode reuses validated paper packages and domain relations before
searching LKM. Select your own PDF/Markdown documents with `--documents`:

```bash
./ariadne/run.sh ask "How can aircraft simulation be calibrated?" \
  --library outputs/my_library --top-k 2 --yes
./ariadne/run.sh ask "What do these documents establish?" \
  --library outputs/my_library --documents /path/paper.pdf /path/notes.md --yes
```

Warm questions run only the answer layer when local evidence covers the question.
Missing relations trigger bounded local reconciliation; missing materials trigger
one gap-directed retrieval round, processing only new or changed papers. Remaining
gaps are visible in answer caveats and `run.json`; freshness requests force retrieval.
Prior candidate answers are excluded from evidence. The library's atomic manifest
references immutable runs and compiled integration packages with content/version
checks. Keep those run directories; failed runs and concurrent update conflicts do
not overwrite the library. Existing runs are not automatically migrated.

Document import requires the `reportlab` and `pypdf` dependencies listed in
`ariadne/requirements.txt`. LKM accepts unencrypted PDFs up to 64 MiB / 50 pages.
Markdown is transported as a text-preserving PDF (including formula source), while
the original Markdown feeds paper processing. Remote document links/images are not
fetched. Extraction uploads do not imply publication in LKM's global search index.
Identical validated imports are reused without another upload. Omit `--library`
and `--documents` to retain the original full pipeline; `--dry-run` never uploads,
calls a model, or writes a run or library.

**Question-led domain reasoning graphs.**

[中文](README.zh-CN.md) · [Examples](#examples) · [Quick start](#quick-start) · [Configuration](#configuration) · [Commands](#commands)

![Aircraft simulation design question graph](docs/images/aircraft-question-graph.png)

Ariadne turns a domain question into a typed, inspectable map of papers,
claims, candidate answers, and reasoning chains. It does not answer with a
single paragraph. It returns:

- **candidate answers** (roots R1–R5) generated from the question;
- the **claims and evidence** that support, bound, or contradict each root;
- **typed relations** between claims, including cross-paper links;
- **every multi-step chain** that reaches a root, as Markdown and Mermaid.

The name is deliberate: Ariadne's thread is how you find your way through a
labyrinth. Here the thread is the answer layer and the reasoning paths that
lead to it.

```text
question
  -> LKM query planning + multi-query search + independent-source checks + graph-backed paper selection
  -> paper2ir      (paper graph -> claims_final)
  -> single-paper  (claims -> Gaia IR package)
  -> merge         (many packages -> domain graph)
  -> Step 6        (candidate answer roots + typed links + chains)
  -> HTML question graph + all-chains Markdown
```

Ariadne is a **thin product layer** over the maintained Gaia IR stages. It does
not fork the pipelines; it orchestrates them, isolates each run, and adds the
question-first answer layer and deliverables.

## Download

Prebuilt self-contained bundle (no clone needed):

[ariadne-0.4.2-bundle.tar.gz](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.4.2/ariadne-0.4.2-bundle.tar.gz) ·
[ariadne-0.4.2-bundle.zip](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.4.2/ariadne-0.4.2-bundle.zip) ·
[SHA256SUMS](https://github.com/CatherineWuleyan/gaia-ir/releases/download/v0.4.2/SHA256SUMS)

The bundle includes Ariadne, the three pipelines, the offline aircraft demo,
and `install.sh`. To build it yourself:

```bash
./packaging/build_release.sh
```

## Examples

### 1. How should aircraft simulation design be carried out?

> **Question:** 怎么完成飞机的仿真设计？
> **Domain:** conceptual design, flight dynamics, aerodynamic modeling, MDO,
> hardware-in-the-loop validation.

![Aircraft question graph](docs/images/aircraft-question-graph.png)

The frozen aircraft graph contains **13 papers, 311 claims, 258 relations,
93 integration deltas, and 5 candidate answers**. The exporter produces
**107 multi-step chains** to those answers.

| Root | Candidate answer | Chains |
| --- | --- | ---: |
| R1 | 集成概念设计环境 · integrated conceptual-design environment | 20 |
| R2 | 嵌套多学科优化 · nested multidisciplinary optimization | 26 |
| R3 | 非线性飞行动力学建模 · nonlinear flight-dynamics modeling | 25 |
| R4 | 飞行数据模型校正 · flight-data model correction | 21 |
| R5 | 硬件在环验证 · hardware-in-the-loop validation | 15 |

What to look at in the viewer:

- the five large red roots and the green **support** links that reach them;
- the distinction between **same-paper** claim sequences (reading order, not a
  logical proof) and cross-paper support;
- dashed **analogy** and **infer** relations: they are visible context, but
  they are **not** promoted to deduction and do not form support chains.

Inspect the artifacts:

- [Aircraft question viewer](../outputs/aircraft/final/domain_viewer_question.html)
- [All chains to R1–R5](../outputs/aircraft/final/ALL_CHAINS_TO_ROOTS.md)
- [Answer layer JSON](../outputs/aircraft/final/answer_layer.json)
- [Frozen demo dataset](../outputs/aircraft/final/)

Run it offline (no keys, no LKM):

```bash
./ariadne/run.sh demo
```

### 2. Which pair-breaking channel sets the in-plane B_c2 in 4H_a-NbSe2?

> **Question:** 在块体 4H_a-NbSe2（非中心对称多型）中，面内上临界场
> B_c2∥ 由哪条对破缺通道决定：塞曼（经 Ising SOC 抬高）还是轨道
>（含有限动量/FFLO）？

![Physics question graph](docs/images/physics-question-graph.png)

The reviewed physics graph contains **18 papers, 500 claims, 654 nodes,
1,298 edges, and 14 theory-layer nodes**. It is an example of a question where
the answer layer must keep two compatible but competing explanations separate:

- **R1**: Ising protection suppresses the Zeeman channel, leaving the orbital /
  interlayer channel to set the main scale;
- **R2**: a purely orbital channel can be numerically sufficient if
  γ ≈ 31/5.5 ≈ 5.6;
- the open question is whether Ising SOC is **necessary**, not whether it
  exists.

The physics case also shows the honesty boundary of the product: the graph
contains 9 contradiction edges, including cross-symmetry and cross-material
evidence. The viewer shows them; it does not silently turn them into support.

Inspect the artifacts:

- [Physics reviewed viewer](../outputs/physics/final/domain_graph_full.reviewed.html)
- [Answer layer JSON](../outputs/physics/final/answer_layer.reviewed.json)
- [Mentor review notes](../outputs/physics/final/MENTOR_REVIEW.md)
- [Target question](../outputs/physics/build/target_question.json)

GitHub shows HTML files as source. To view a graph, open the file locally after
cloning, or serve the repository with `python3 -m http.server` and open the file
in a browser.

## Quick start

### Requirements

| Requirement | Why |
| --- | --- |
| Python **3.11+** | the harness and pipeline code |
| `gaia-lang==0.5.0a7` | the Gaia IR compiler used by the single stage |
| `bohr` CLI **2.2+** | live LKM search, paper graphs, and optional PDF parsing |
| DeepSeek or OpenAI-compatible API key | paper2ir, single, merge, and Step 6 |
| Bohrium account + access key | live LKM calls (not needed for `demo`) |

The public checkout includes the pipeline code Ariadne needs:
`pipelines/clean_claims`, `pipelines/single_paper`, `pipelines/merge`, and
`weekly_reports/export_single_artifacts.py`.

### 1. Clone and install

```bash
git clone https://github.com/CatherineWuleyan/gaia-ir.git
cd gaia-ir

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install --pre -r ariadne/requirements.txt
pip install -e agent
```

`run.sh` prefers, in order: `$ARIADNE_PYTHON`, `.venv-gaia-a7`, `.venv`, then
`python3`. You can also pin the interpreter in the config (see below).

### 2. Install the Bohr CLI

```bash
npm install -g bohr-cli
bohr version
```

If `bohr` is already installed and logged in, Ariadne can use the active
profile. To keep a dedicated Ariadne profile, use `ariadne auth bohr` (next
step).

### 3. Configure your key and Bohr account

Interactive, recommended:

```bash
./ariadne/run.sh config init
```

Non-interactive:

```bash
export DEEPSEEK_API_KEY="sk-..."
export BOHR_ACCESS_KEY="..."

./ariadne/run.sh config init --no-input \
  --api-key "$DEEPSEEK_API_KEY" \
  --bohr-access-key "$BOHR_ACCESS_KEY"
```

The config is written to `~/.config/ariadne/config.json` with mode `0600`.
Ariadne never prints the key; `config show` and `config get` mask it.

Then log in to Bohr with the configured access key:

```bash
./ariadne/run.sh auth bohr
```

This creates or updates the Bohr profile `ariadne` and leaves your existing
default profile untouched. To use the browser/device flow instead:

```bash
./ariadne/run.sh auth bohr --device
```

Finally, check everything:

```bash
./ariadne/run.sh doctor
```

`doctor` reports the config source, model endpoint, Bohr login, pipeline files,
and whether demo / offline-full / live-LKM-full are ready. It never prints a
secret.

### 4. Run the offline demo

```bash
./ariadne/run.sh demo
```

This runs the question layer on the frozen aircraft graph: no LKM call and no
LLM call. It writes a run directory under `outputs/ariadne_runs/`:

```text
outputs/ariadne_runs/run_<timestamp>_<question>/
  domain_viewer_question.html   # self-contained question graph
  ALL_CHAINS_TO_ROOTS.md        # every multi-step chain to each root
  run.json                      # question, stages, runtime summary
  04_question/                  # answer_layer.json, view model, evaluation
```

Open the viewer:

```bash
open outputs/ariadne_runs/run_*/domain_viewer_question.html   # macOS
xdg-open outputs/ariadne_runs/run_*/domain_viewer_question.html  # Linux
```

### 5. Run a live question

Start with **two papers** while you check quality and cost:

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

Ariadne now chooses the papers itself: an LLM query planner expands the
question into several LKM queries, independent indexes cross-check the
candidate papers, and each candidate graph is probed. Empty or conclusion-less
graphs are skipped before `paper2ir` starts. The full audit is written to
`01_retrieval/selection.json`.

Preview first without spending anything:

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --dry-run
```

Notes:

- `--yes` authorizes billable LKM calls for this run. Without it, Ariadne asks
  for confirmation in an interactive terminal and fails fast in a batch job.
- `--top-k 2` keeps paper2ir cost and latency small. Each paper runs **14 LLM
  steps** in paper2ir, then the single and merge pipelines call models again.
- `--no-fulltext` skips arXiv PDF download/parsing and uses graph-derived text.
- `--no-query-planner` uses only deterministic fallback queries.
- `--queries N` / `--candidate-limit N` tune search breadth and graph probing.
- `--no-cross-check` skips the independent-source verification; `--external-discovery`
  lets papers that several sources agree on but LKM missed enter the pool.
- `--papers papers.json` replaces automatic retrieval with a local spec (see
  [Offline paper specs](#offline-paper-specs)).

## Automatic retrieval

Live mode does not ask you to choose papers. The retrieval stage is:

1. **Plan queries** — ask the configured LLM for several short English LKM
   queries covering the question's concepts, methods/systems, and evaluation
   setting. The English question and deterministic keyword query are always
   included as fallbacks.
2. **Search broadly** — run up to `retrieval.queries` hybrid LKM searches
   (`conclusion,abstract` scopes) and merge papers by LKM id.
3. **Cross-check sources** — send the first two planned queries to OpenAlex,
   Crossref, arXiv, and Semantic Scholar, then record which independent indexes
   resolve each candidate by DOI, arXiv id, or near-identical title
   (`corroboration`). Candidates missing from the search results get a direct
   DOI lookup (at most four). A 429, timeout, or malformed payload from one
   source is recorded as "unavailable" and never stops retrieval.
4. **Rank candidates** — combine multi-query hits, LKM rerank scores, a cheap
   lexical topic relevance score, independent-source agreement, and (after
   probing) graph size.
5. **Probe graphs** — fetch each candidate's paper graph. Empty graphs,
   graphs with no conclusion nodes, and very thin graphs are skipped. If the
   first pass does not yield enough usable papers, Ariadne extends the probe
   instead of failing.
6. **Build missing graphs** — when a paper has no native LKM graph, Ariadne
   resolves a PDF through arXiv, DOI landing pages (`citation_pdf_url`),
   OpenAlex, Unpaywall, Semantic Scholar, Crossref, and MDPI's CDN. It
   downloads the PDF first, then runs up to `retrieval.max_extractions`
   `bohr lkm extract` jobs in parallel (3 workers). Closed-access PDFs and
   PDFs over LKM's 50-page limit are recorded as skipped instead of consuming
   the budget. Extraction costs 1.00 CNY per paper (0.10 CNY cache hit);
   `--no-lkm-extract` disables the step.
7. **Select and audit** — prefer graph-backed, relevant papers confirmed by
   `retrieval.min_corroboration` independent sources; fall back to
   relevance-only, then to all graph-backed papers if too few are confirmed.
   Then write `01_retrieval/selection.json` with every query, candidate,
   cross-check result, graph probe, skip reason, relevance score, extraction
   task, and the final selection.

### Multi-source cross-validation

LKM is the only source of paper graphs, but a single index both misses papers
and occasionally misattributes metadata. Ariadne therefore treats LKM as the
**primary index** and OpenAlex, Crossref, arXiv, and Semantic Scholar as
**independent witnesses**: confirmed candidates rank and select higher. The
default `retrieval.min_corroboration = 1` means "LKM plus at least one
independent source"; set it to 2 to require two independent confirmations.

- Per-candidate agreement is written to `selection.json` under
  `cross_check`, summarised in `01_retrieval/sources.json`, and kept raw per
  source in `01_retrieval/sources/<source>.json`.
- `--no-cross-check` or `retrieval.cross_check=false` disables the step for
  offline or restricted networks. `retrieval.cross_check_sources` selects the
  participating sources, e.g. `openalex,crossref`.
- `--external-discovery` (`retrieval.external_discovery`) adds papers that at
  least two independent sources agree on but LKM missed. They have no native
  LKM graph, so they follow the same PDF → `bohr lkm extract` path and count
  against `retrieval.max_extractions`, which can add extraction cost.
- These metadata APIs are free and read-only, so cross-checking adds no LKM
  or model charges. Semantic Scholar rate-limits anonymous traffic; set
  `S2_API_KEY` to raise its quota. Ariadne only sends the first two queries.

`--papers ...` still bypasses both automatic retrieval and cross-validation
when you already know the corpus.

To tune retrieval without editing code:

```bash
./ariadne/run.sh ask "..." --queries 5 --candidate-limit 16
./ariadne/run.sh ask "..." --no-query-planner   # deterministic queries only
./ariadne/run.sh ask "..." --no-cross-check     # disable independent checks
./ariadne/run.sh ask "..." --external-discovery # recover papers LKM missed
./ariadne/run.sh config set retrieval.min_relevance 0.2
./ariadne/run.sh config set retrieval.min_corroboration 2
```

`selection.json` is the first file to inspect when the selected corpus looks
wrong. `queries.json` shows the plan; `candidates.json` shows the ranked pool;
`search_raw_*.json` keeps each LKM response; `sources.json` records what each
independent source returned and confirmed; `extract/*/submit.json` and
`result.json` keep the LKM extraction audit. Selection fills up to `--top-k`:
relevance and independent confirmation are preferences, not hard caps.

## Configuration

### File location

| Priority | Source |
| ---: | --- |
| 1 | CLI flags (`--api-key`, `--base-url`, `--model`, `--bohr-profile`, …) |
| 2 | environment variables |
| 3 | Ariadne config file |
| 4 | `agent/.env` (legacy, local only) |
| 5 | built-in defaults |

The config file is resolved in this order:

```text
$ARIADNE_CONFIG
$XDG_CONFIG_HOME/ariadne/config.json
~/.config/ariadne/config.json
```

On Windows, `%APPDATA%\ariadne\config.json` is used when `APPDATA` is set.

A minimal config looks like this; the full example is
[`config.example.json`](config.example.json):

```json
{
  "version": 1,
  "model": {
    "provider": "deepseek",
    "api_key": "sk-...",
    "base_url": "https://api.deepseek.com/v1",
    "model": "deepseek-v4-flash",
    "merge_model": "deepseek-v4-flash"
  },
  "bohr": {
    "profile": "ariadne",
    "access_key": "...",
    "host": "https://open.bohrium.com",
    "auto_approve_spend": false,
    "payment_source": ""
  },
  "runtime": {
    "python": ""
  }
}
```

The file is **sparse**: only values you set are written. Unset values keep
their defaults or fall through to environment / `agent/.env`.

### Config commands

```bash
./ariadne/run.sh config path                 # print the config path
./ariadne/run.sh config init                 # interactive setup
./ariadne/run.sh config init --no-input ...  # scripted setup
./ariadne/run.sh config show                 # effective config, secrets masked
./ariadne/run.sh config show --json
./ariadne/run.sh config get model.model
./ariadne/run.sh config set model.model deepseek-v4-flash
./ariadne/run.sh config unset model.model
./ariadne/run.sh config test --api           # one tiny chat completion
./ariadne/run.sh config test --bohr          # bohr auth status
```

Supported keys:

| Key | Meaning |
| --- | --- |
| `model.provider` | `deepseek` or `openai-compatible` |
| `model.api_key` | model key (masked in output) |
| `model.base_url` | OpenAI-compatible base URL, default `https://api.deepseek.com/v1` |
| `model.model` | paper2ir / single model |
| `model.merge_model` | merge / Step 6 model |
| `bohr.profile` | Bohr CLI profile; empty means the active profile |
| `bohr.access_key` | Bohrium access key (masked in output) |
| `bohr.host` | default `https://open.bohrium.com` |
| `bohr.auto_approve_spend` | if true, always pass `--yes` to billable LKM commands |
| `bohr.payment_source` | empty (gateway default), `photon`, or `balance` |
| `runtime.python` | Python used for pipeline subprocesses |
| `retrieval.use_query_planner` | use the LLM query planner (default true) |
| `retrieval.queries` | max LKM queries per question (default 5) |
| `retrieval.per_query` | results requested per query (default 8) |
| `retrieval.candidate_limit` | first-pass graph probes (default 16; extended if needed) |
| `retrieval.min_papers` | minimum usable graph-backed papers (default 2) |
| `retrieval.min_graph_nodes` | reject graphs below this node count (default 3) |
| `retrieval.min_relevance` | lexical topic gate for preferred papers (default 0.2) |
| `retrieval.use_lkm_extract` | build missing graphs with `bohr lkm extract` (default true) |
| `retrieval.max_extractions` | max papers to extract per run (default 5, 0 disables) |
| `retrieval.cross_check` | verify candidates against independent indexes (default true) |
| `retrieval.cross_check_sources` | comma-separated sources (default all four) |
| `retrieval.min_corroboration` | independent sources needed to prefer a paper (default 1, 0 disables the gate) |
| `retrieval.external_discovery` | add multi-source papers missed by LKM (default false) |
| `retrieval.external_candidates` | max external papers added per run (default 4) |

### Environment variables

| Variable | Purpose |
| --- | --- |
| `ARIADNE_CONFIG` | config file path |
| `ARIADNE_MODEL_API_KEY` / `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` | model key |
| `ARIADNE_MODEL_BASE_URL` / `DEEPSEEK_BASE_URL` / `OPENAI_BASE_URL` | endpoint |
| `ARIADNE_MODEL` / `DEEPSEEK_MODEL` / `OPENAI_MODEL` | model name |
| `ARIADNE_MERGE_MODEL` / `DEEPSEEK_MERGE_MODEL` | merge model |
| `ARIADNE_BOHR_ACCESS_KEY` / `BOHR_ACCESS_KEY` / `LKM_ACCESS_KEY` | Bohr key |
| `ARIADNE_BOHR_PROFILE` | Bohr profile |
| `ARIADNE_AUTO_APPROVE_SPEND` | `1` / `true` to approve billable calls |
| `ARIADNE_PAYMENT_SOURCE` | `photon` or `balance` |
| `ARIADNE_PYTHON` | interpreter for Ariadne and its subprocesses |
| `S2_API_KEY` | optional Semantic Scholar key for higher cross-check rate limits |

Environment variables win over the config file, so CI does not need a config
file:

```bash
DEEPSEEK_API_KEY=... BOHR_ACCESS_KEY=... \
  ./ariadne/run.sh ask "..." --top-k 2 --yes
```

### Model compatibility

The bundled paper2ir, single, and merge stages use an **OpenAI-compatible
chat-completions client**. DeepSeek is the default and the tested path.
Any endpoint that accepts `POST {base_url}/chat/completions` with
`Authorization: Bearer <key>` should work; set `model.base_url`,
`model.model`, and `model.merge_model` accordingly.

`ANTHROPIC_API_KEY` alone is **not** sufficient for a full run: the single and
merge stages require the OpenAI-compatible client. If you set only an Anthropic
key, `doctor` says so explicitly.

### Bohr account and billing

`bohr` uses profiles. Ariadne reads `bohr.profile`; when empty, it uses the
active Bohr profile. `ariadne auth bohr` creates a dedicated `ariadne` profile:

```bash
# access-key flow (non-interactive)
./ariadne/run.sh auth bohr --access-key "$BOHR_ACCESS_KEY"

# browser flow
./ariadne/run.sh auth bohr

# headless/device flow
./ariadne/run.sh auth bohr --device

# check only
./ariadne/run.sh auth bohr --check
./ariadne/run.sh auth status
```

If you prefer the Bohr CLI directly, the equivalent is:

```bash
bohr auth login --ak "$BOHR_ACCESS_KEY" --ak-only --profile ariadne
```

LKM `search`, `graph`, `reasoning`, and related calls are billable
(**0.05 CNY per call**, with a personal monthly 1,000-call quota at the time of
writing). PDF parsing is billed separately. Ariadne asks before the first
billable call; `--yes` or `bohr.auto_approve_spend=true` authorizes the run.
A `--dry-run` never spends. Check current prices with:

```bash
bohr billing pricing --resource lkm -o json
```

### Python interpreter

`run.sh` picks the interpreter before Ariadne starts. Once Ariadne is running,
`runtime.python` / `ARIADNE_PYTHON` is what pipeline subprocesses use. Pin it
when your venv is not named `.venv`:

```bash
./ariadne/run.sh config set runtime.python /opt/venvs/gaia/bin/python
```

## Commands

| Command | What it does |
| --- | --- |
| `ask "<question>"` | full question → LKM → paper2ir → single → merge → render |
| `ask ... --mode demo` | answer-layer step on the frozen aircraft graph |
| `ask ... --dry-run` | preflight and plan, no files, no spending |
| `ask ... --papers spec.json` | offline paper spec instead of LKM search |
| `ask ... --top-k N` | papers to select (start with 2) |
| `ask ... --queries N` | max LKM queries for automatic retrieval |
| `ask ... --candidate-limit N` | first-pass graph probes before extended probing |
| `ask ... --no-query-planner` | disable the LLM planner; use fallback queries |
| `ask ... --no-lkm-extract` | do not build graphs with `bohr lkm extract` |
| `ask ... --max-extractions N` | cap LKM graph extractions for this run |
| `ask ... --no-cross-check` | skip independent-source verification |
| `ask ... --min-corroboration N` | independent sources needed to prefer a paper |
| `ask ... --external-discovery` | add papers multiple sources found but LKM missed |
| `ask ... --yes` | approve billable LKM calls for this run |
| `ask ... --no-fulltext` | skip PDF parsing; use graph-derived text |
| `demo` | offline aircraft example |
| `doctor` | config, Python, model, Bohr, pipelines, readiness |
| `config init/show/set/get/unset/path/test` | manage configuration |
| `auth bohr/status` | Bohr login and auth state |

Global run flags:

```text
--config PATH
--api-key KEY  --base-url URL  --model NAME  --merge-model NAME
--bohr-profile P  --bohr-access-key AK  --payment-source photon|balance
--python PATH
```

## Outputs

Every full or demo run is isolated under `outputs/ariadne_runs/<timestamp>_<slug>/`:

| File | Meaning |
| --- | --- |
| `domain_viewer_question.html` | self-contained question-led graph; no server needed |
| `ALL_CHAINS_TO_ROOTS.md` | every multi-step chain ending at R1–R5, with Mermaid and content tables |
| `05_outputs/answer_layer.json` | roots, links, typed edges, exploratory paths (full) |
| `05_outputs/domain_view_question.json` | view model with the answer layer injected (full) |
| `run.json` | question, per-stage timings, config summary (no secrets), output paths |
| `logs/` | command logs for every stage |

The offline demo uses `04_question/` instead of `05_outputs/` for the answer
layer and view model; the two headline deliverables are still copied to the
run root.

For a full run, intermediate stages are kept:

```text
01_retrieval/   queries.json, candidates.json, selection.json, sources.json,
                sources/<source>.json, search_raw_*.json,
                extract/<paper>/submit.json|result.json,
                per-paper graph.json + ocr.md
02_paper2ir/    run-local clean_claims copy, claims_final.json per paper
03_single/      Gaia IR packages, formalization.json, knowledge.index.json
04_merge/       merge pipeline run, view_model.json
05_outputs/     answer layer and deliverables
```

The run-local `02_paper2ir/pipeline` copy means old data is never reused and a
failed paper2ir step does not contaminate the next run.

## How it works

| Stage | What it does | Implementation |
| --- | --- | --- |
| `01_retrieval` | plan multiple LKM queries, merge/rank candidates, cross-check them against independent indexes, probe graphs, skip empty or conclusion-less papers, then fetch full text when possible | [`ariadne/retrieval.py`](ariadne/retrieval.py), [`ariadne/sources.py`](ariadne/sources.py) |
| `02_paper2ir` | 14-step paper2ir pipeline: LKM graph + text → `claims_final.json` | [`pipelines/clean_claims`](../pipelines/clean_claims/) from a run-local copy |
| `03_single` | maintained single-paper pipeline: claims → `gaia.ir.json`, `formalization.json`, `knowledge.index.json` | [`pipelines/single_paper`](../pipelines/single_paper/) + harness |
| `04_merge` | merge paper packages into one domain graph, with K conclusions and Step 6 answer roots/links | [`pipelines/merge/pipeline.question.json`](../pipelines/merge/pipeline.question.json) |
| `05_render` | render the question viewer and export all chains | [`render_question_viewer.py`](../pipelines/merge/pipeline_merge_question/render_question_viewer.py), [`export_chains_markdown.py`](../pipelines/merge/pipeline_merge_question/export_chains_markdown.py) |

The three maintained pipelines Ariadne packages:

1. **paper2ir** — paper → claims;
2. **single** — claims → per-paper reasoning graph;
3. **merge** — many graphs → domain graph and question answer layer.

Retrieval is before the three pipelines; rendering is after them.

### What “support” means

- Only **deduction** edges are chainable. `infer`, `analogy`, `bounds`,
  `special_case`, and `same-paper` are retained as visible context but are not
  promoted to deduction.
- `same-paper` groups direct support claims by paper and claim number; it is a
  reading sequence, not a logical proof.
- The viewer shows all graph paths by default. A path being drawable does not
  make it evidence; the chain exporter labels the edge type.
- Roots are **candidate answers**, not truth labels. The graph is a map of the
  evidence and its gaps, not a verdict.

## Offline paper specs

Use `--papers` to skip LKM search. A spec can point at LKM ids or local files:

```json
{
  "papers": [
    {
      "lkm_id": "1234567890123456789",
      "title": "Paper title",
      "arxiv": "2501.00001",
      "graph_path": "path/to/graph.json",
      "ocr_path": "path/to/paper.md"
    },
    {
      "title": "Local graph without an LKM id",
      "graph_path": "path/to/another-graph.json",
      "ocr_path": "path/to/another-paper.md"
    }
  ]
}
```

Paths are relative to the repository root unless absolute. Accepted graph shapes:

- `{"code": 0, "data": {"papers": [...]}}` (pipeline-native);
- `{"data": {"items": [...]}}` (Bohr CLI response);
- a bare `{"nodes": [...], "edges": [...]}` graph;
- a single paper item with `graph` / `paper`.

A local graph without an LKM id receives a stable `local_<hash>` id, so the
rest of the pipeline can still run.

```bash
./ariadne/run.sh ask "your question" \
  --papers my-papers.json --top-k 2 --yes
```

Local graph/text entries do not spend LKM calls. Entries with only an
`lkm_id` still call `bohr lkm graph` and need `--yes`.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `bohr: command not found` | `npm install -g bohr-cli`, or use `--papers` with local files |
| `profile "ariadne" does not exist` | `./ariadne/run.sh auth bohr`, or set `bohr.profile` to an existing profile |
| `bohr lkm search failed` / `401` | check the access key, run `ariadne auth bohr --check`, or `bohr auth login` |
| `CONFIRMATION_REQUIRED` | re-run with `--yes`, or set `bohr.auto_approve_spend=true` |
| `No model API key is configured` | `config init`, `--api-key`, or `DEEPSEEK_API_KEY` |
| `ANTHROPIC_API_KEY` is set but full mode fails | use a DeepSeek/OpenAI-compatible key; the bundled single/merge stages need it |
| `ModuleNotFoundError: gaia` | `pip install --pre -r ariadne/requirements.txt` |
| `No module named pipeline_harness` | `pip install -e agent`, or run through `./ariadne/run.sh` |
| `--papers contains no usable entries` | each entry needs `lkm_id` or `graph_path` |
| `Ariadne needs at least two papers` | increase `--top-k` or add papers to the spec |
| `paper2ir stopped at ...` | inspect `02_paper2ir/logs/paper2ir_<id>.log`; run-local data is preserved for diagnosis |
| Demo works but live run is slow | start with `--top-k 2 --no-fulltext`; paper2ir is 14 LLM steps per paper |
| Windows path errors | use a venv inside the repo and run from the repository root; set `runtime.python` explicitly |

## Repository layout

| Path | Purpose |
| --- | --- |
| [`ariadne/`](.) | this product: CLI, config, retrieval, stage orchestration, render |
| [`agent/`](../agent/) | pipeline harness, artifact store, checkpoints, validation, viewer source |
| [`pipelines/clean_claims/`](../pipelines/clean_claims/) | paper2ir: paper graph → claims |
| [`pipelines/single_paper/`](../pipelines/single_paper/) | claims → per-paper Gaia IR |
| [`pipelines/merge/`](../pipelines/merge/) | many packages → domain graph + question layer |
| [`weekly_reports/export_single_artifacts.py`](../weekly_reports/export_single_artifacts.py) | export single-run artifacts for merge |
| [`outputs/aircraft/final/`](../outputs/aircraft/final/) | frozen aircraft demo dataset |
| [`outputs/physics/final/`](../outputs/physics/final/) | reviewed physics example graph |

## Status and limits

- `demo` is tested offline on the frozen aircraft graph and writes both
  deliverables.
- An 8-paper automatic aircraft run completed end-to-end: **184 claims, 105
  relations, 5 candidate answers, 33 answer links, and 17 chains**. It used
  `--top-k 8`; k is a runtime option, not a hard-coded paper count.
- Three of the eight graphs were built automatically by `bohr lkm extract`:
  Interactive Cockpit (62 nodes / 15 conclusions), Cybermatrix (46 / 10), and
  Flying-V (56 / 12). Native-graph papers and built graphs go through the same
  selection/materialisation path.
- On broad questions, many high-relevance candidates still have empty native
  graphs. Closed-access publishers (IEEE/AIAA) and PDFs over 50 pages are
  skipped with an explicit `extract_skip_reason` in `selection.json`; no
  synthetic graph is invented.
- The LKM extraction path was also exercised on a real arXiv paper (Maestro):
  87 nodes / 88 edges / 19 conclusions. The empty-native-graph → extraction
  decision is covered by unit tests; it only triggers when a PDF/arXiv source
  is available.
- `full` preflight-checks config, Python, pipeline files, and Bohr login before
  spending. Start with `--top-k 2 --no-fulltext` while checking quality and cost.
- The quality of full-mode chains depends on paper2ir, single, merge, and the
  question layer. Ariadne reports what those stages produce; it does not invent
  missing evidence or upgrade `infer` to `deduction`.
- The viewer shows all graph paths by default. A drawable path is not
  automatically a support chain.
- The physics example is a reviewed static graph, not a fresh live run. It
  demonstrates the shape of the answer layer and the review surface.

## Cost

| Resource | Cost |
| --- | --- |
| LKM search / graph / reasoning | 0.05 CNY per call, personal monthly 1,000-call quota |
| LKM PDF extraction | separate billing; check `bohr billing pricing` |
| LLM calls | billed by your provider; paper2ir is 14 steps per paper plus single/merge |

A two-paper automatic run makes up to `retrieval.queries` search calls plus
graph probes (first pass up to `retrieval.candidate_limit`, extended only if
too few graphs are usable), before the LLM pipeline. Missing graphs add up to
`retrieval.max_extractions` **1.00 CNY** extraction calls (0.10 CNY on a cache
hit). Cross-checking calls only free metadata APIs and adds no LKM or model
charges; external-discovery papers count against the extraction budget. Use
`--dry-run` to see the upper-bound plan and `--no-fulltext` to skip PDF parsing.

## License

This repository currently has no license file. Treat it as a research
prototype; add a license before redistributing it as a library.
