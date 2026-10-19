# Ariadne 0.4.0

**Question → LKM papers → graph → candidate answers.**

Ariadne is a self-contained research-graph runtime. You give it a domain
question; it searches the Bohr LKM corpus, builds or fetches paper graphs, runs
paper2ir / single / merge, and renders a question graph with candidate answer
roots and typed reasoning chains.

This bundle includes the Ariadne launcher, the maintained Gaia IR pipelines,
the offline aircraft demo, and an installer.

## Requirements

- macOS or Linux (Windows: use WSL)
- Python **3.11+**
- Network access for model calls and live LKM search
- An OpenAI-compatible model API key (DeepSeek is the tested path)
- A Bohr account for live LKM retrieval (the offline `demo` needs no key)

## 1. Install

```bash
tar -xzf ariadne-0.4.0-bundle.tar.gz
cd ariadne-0.4.0
./install.sh
```

`install.sh` creates `./.venv`, installs `ariadne/requirements.txt`, and
installs the bundled pipeline harness. Use `./install.sh --no-venv` to install
into an existing Python environment, or `--no-deps` to only create the venv.

## 2. Configure your own key

```bash
./ariadne/run.sh config init
```

The prompt asks for your model API key, base URL, model name, and (optionally)
a Bohr access key. The key is stored in `~/.config/ariadne/config.json` with
mode `0600`; it is never bundled with Ariadne.

You can also use environment variables:

```bash
export DEEPSEEK_API_KEY="sk-..."
export DEEPSEEK_BASE_URL="https://api.deepseek.com/v1"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export BOHR_ACCESS_KEY="..."
```

Check the setup:

```bash
./ariadne/run.sh doctor
./ariadne/run.sh config test --api
./ariadne/run.sh config test --bohr
```

## 3. Run the offline demo

No key is needed:

```bash
./ariadne/run.sh demo
open outputs/ariadne_runs/run_*/domain_viewer_question.html
```

The demo runs the question/answer layer on the bundled frozen aircraft graph.

## 4. Run a live question

```bash
./ariadne/run.sh ask "怎么完成飞机的仿真设计？" --top-k 2 --yes
```

Ariadne writes a self-contained run directory under
`outputs/ariadne_runs/` containing:

- `domain_viewer_question.html` — the question graph;
- `ALL_CHAINS_TO_ROOTS.md` — all multi-step chains;
- `05_outputs/answer_layer.json` — candidate roots and answer links;
- `01_retrieval/selection.json` — every query, candidate, graph probe, skip
  reason, and the final paper selection;
- `01_retrieval/extract/` — LKM extraction audit, when a missing graph was
  built with `bohr lkm extract`.

Use `--dry-run` to see the upper-bound plan and `--no-fulltext` to skip PDF
parsing. A first run with `--top-k 2 --no-fulltext` is the cheapest way to
check quality.

## What is included

```text
ariadne/                 launcher, CLI, retrieval, paper2ir, render, tests
agent/                   pipeline harness used by single/merge
pipelines/clean_claims/  paper2ir
pipelines/single_paper/  single-paper pipeline
pipelines/merge/         domain graph + question/answer layer
weekly_reports/          single-artifact exporter
outputs/aircraft/final/  frozen demo graph assets
```

## What is not included

- API keys, Bohr credentials, `.env` files
- The local paper corpus and generated `outputs/ariadne_runs/`
- Historical runs, private datasets, and the full research workspace

## Documentation

- [中文完整说明](ariadne/README.zh-CN.md)
- [English guide](ariadne/README.md)
- [Configuration example](ariadne/config.example.json)

## Troubleshooting

- **No Python 3.11+**: install one and rerun `./install.sh --python /path/to/python3`.
- **`doctor` says API key missing**: run `config init` or export `DEEPSEEK_API_KEY`.
- **Bohr not logged in**: run `./ariadne/run.sh auth bohr`.
- **A full run is slow**: paper2ir is 14 steps per paper; use `--top-k 2` first.
- **A paper has no native graph**: Ariadne tries `bohr lkm extract` (1.00 CNY
  per successful paper, capped by `--max-extractions`). Use
  `--no-lkm-extract` to disable it.
