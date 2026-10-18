# Ariadne

**Ariadne — question-led domain reasoning graphs.**

Ariadne follows a thread from a research question to papers, claims, candidate
answers, and inspectable reasoning chains:

```text
question
  → LKM search + paper graphs
  → paper2ir      (paper graph → claims_final)
  → single-paper  (claims → Gaia IR package)
  → merge         (many packages → domain graph)
  → Step 6        (candidate answer roots + typed links + chains)
  → HTML question graph + all-chains Markdown
```

The name is deliberate: Ariadne's thread is how you find your way through a
labyrinth. Here the thread is the answer layer and the reasoning paths that lead
to it.

## Quick start

```bash
cd /Users/catherinewu/Documents/gaia-ir

# 1. Check prerequisites
./ariadne/run.sh doctor

# 2. Offline, no LKM or LLM: use the frozen aircraft graph
./ariadne/run.sh demo

# 3. Full run: question → LKM → paper2ir → single → merge → question layer
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --mode full --top-k 8
```

Outputs are written to `outputs/ariadne_runs/<timestamp>_<slug>/`:

| File | Meaning |
| --- | --- |
| `domain_viewer_question.html` | self-contained question-led domain graph |
| `ALL_CHAINS_TO_ROOTS.md` | every multi-step chain ending at R1–R5, with Mermaid + content tables |
| `05_outputs/answer_layer.json` | roots, links, typed edges, exploratory paths |
| `05_outputs/domain_view_question.json` | view model with the answer layer injected |
| `run.json` | question, stages, and output paths |

## Commands

### `doctor`

Checks the repo paths, the Gaia Python environment, the maintained pipeline
configs, the LKM CLI, and whether a model key is available.

### `demo`

Runs the question layer on the frozen aircraft dataset:

- input: `outputs/question_domain_aircraft/domain_view_typed.json`
- edge roles: `outputs/question_domain_aircraft/edge_roles_typed.json`
- answer fixture: `outputs/question_domain_aircraft/answer_fixture.json`
- no LKM call, no model call
- output: viewer + all-chains Markdown

This is the fastest way to see the final product and to test a change to the
viewer or the chain exporter.

### `ask --mode full`

The real end-to-end path. It has five stages:

| Stage | What it does | Main implementation |
| --- | --- | --- |
| `01_retrieval` | `bohr lkm search` for the question, then `bohr lkm graph --paper-id`; fetch full text when possible, otherwise build a graph-derived text fallback | `ariadne/retrieval.py` |
| `02_paper2ir` | 14-step paper2ir pipeline: LKM graph + text → `claims_final.json` | `pipelines/clean_claims` (run from a run-local copy) |
| `03_single` | maintained single-paper pipeline: claims → `gaia.ir.json`, `formalization.json`, `knowledge.index.json` | `pipelines/single_paper` + harness |
| `04_merge` | merge the paper packages into one domain graph, with K conclusions and Step 6 answer roots/links | `pipelines/merge/pipeline.question.json` + `pipeline_merge_question` |
| `05_outputs` | render the question viewer and export all chains | `render_question_viewer.py`, `export_chains_markdown.py` |

This is the "three pipelines" the product packages:

1. **paper2ir** — paper → claims;
2. **single** — claims → per-paper reasoning graph;
3. **merge** — many graphs → domain graph and question answer layer.

Retrieval is before the three pipelines; rendering is after them.

## Prerequisites for full mode

- `bohr` CLI installed and logged in (`bohr auth login`), or pass a local paper
  spec instead of using LKM search.
- A model key in `agent/.env` (`DEEPSEEK_API_KEY` or `ANTHROPIC_API_KEY`).
- The repo-local `.venv-gaia-a7` environment, or set `ARIADNE_PYTHON` to a
  compatible Python with Gaia 0.5.0a7.
- Full mode is expensive: paper2ir is 14 LLM steps per paper, then the single
  pipeline and merge also call models. Start with `--top-k 2` or `--top-k 3`
  when testing.

### Offline paper spec

`ask --mode full --papers weekly_reports/lkm_papers.json` skips LKM search but
still fetches each paper's LKM graph (and full text when an arXiv id is present).
A local spec can also point at existing graph/text files:

```json
{
  "papers": [
    {
      "lkm_id": "1234567890123456789",
      "title": "Paper title",
      "arxiv": "2501.00001",
      "graph_path": "path/to/graph.json",
      "ocr_path": "path/to/paper.md"
    }
  ]
}
```

Paths are relative to the repo root.

## Product vs. repo

Ariadne is a **thin product layer**, not a fork of the pipelines. It orchestrates
the maintained implementations in `pipelines/clean_claims`,
`pipelines/single_paper`, and `pipelines/merge`, adding:

- a question-first entry point;
- LKM retrieval and materialisation;
- isolated run directories (the clean_claims copy is per-run; old data is not
  reused);
- answer-root generation and typed-link validation via Step 6;
- the viewer and all-chains Markdown export.

Keeping the product thin makes each stage independently auditable: a failure in
paper2ir is not hidden inside a new monolith, and the merge/viewer code remains
the maintained code.

## Honest status

- `demo` is tested offline on the frozen aircraft graph.
- `full` is wired to the maintained stages, but a complete 2+ paper LKM run was
  not executed while packaging this product because it needs live LKM
  authentication and multiple model calls. Use `doctor` and `--top-k 2` for the
  first real run.
- The quality of full-mode chains depends on paper2ir/single/merge and the
  question layer. The viewer does not turn `infer` into `deduction`; only
  `deduction` is chainable, and cross-paper `analogy`/`infer` are retained as
  visible non-support relations.
