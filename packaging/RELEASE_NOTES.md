# Ariadne 0.5.0

Ariadne turns a domain question into a graph-backed set of candidate answers:
LKM retrieval → paper2ir → single → merge → answer layer → self-contained
question graph.

This release ships the retrieval and library work that 0.4.x described but did
not contain. **The 0.4.x bundles were built from a code tree that had no
`sources.py`**, so the cross-validation section of those guides could not be
reproduced from the download. The 0.5.0 bundle matches its tag.

## Download

- `ariadne-0.5.0-bundle.tar.gz` — macOS/Linux
- `ariadne-0.5.0-bundle.zip` — macOS/Linux/Windows (WSL)
- `SHA256SUMS` — archive checksums

The bundle is self-contained: Ariadne, the three pipelines, the offline aircraft
demo, `install.sh`, and the Chinese/English guides. It does not contain API
keys, Bohr credentials, local corpora, or generated runs.

## Quick start

```bash
tar -xzf ariadne-0.5.0-bundle.tar.gz
cd ariadne-0.5.0
./install.sh
./ariadne/run.sh config init
./ariadne/run.sh doctor
./ariadne/run.sh demo
```

## New in 0.5.0

**Independent-source cross-validation.** LKM stays the primary index; OpenAlex,
Crossref, arXiv and Semantic Scholar act as independent witnesses. Each
candidate records how many of them resolve it by DOI, arXiv id, or
near-identical title. A 429, a timeout, or a malformed payload from any source
is recorded as unavailable and never stops retrieval.

**External discovery.** With `--external-discovery`, papers that at least two
independent sources agree on but LKM missed enter the pool. They have no native
graph, so they follow the same PDF → `bohr lkm extract` path and count against
`retrieval.max_extractions`.

**Personal library mode.** `--library` reuses validated paper packages before
searching LKM. The manifest is atomic and references immutable runs with
content and version checks; failed runs and concurrent writers do not overwrite
an existing entry. Warm questions run only the answer layer when local evidence
covers the question.

**Document import.** `--documents` accepts your own PDF/Markdown files. Requires
`reportlab` and `pypdf` from `ariadne/requirements.txt`.

**Spend preflight.** `--dry-run` prints the upper-bound LKM call count and cost
without writing files, calling a model, or uploading anything.

## What this release does not claim

- The aircraft question graph has **0 typed support chains**. Its 107 exported
  rows are 7 same-paper reading sequences plus 100 `infer` exploratory paths.
  See `outputs/aircraft/final/CHAIN_RELIABILITY.md`.
- Root verdicts are structural (how many claims and papers link to a root), not
  a truth vote. All five aircraft roots currently share one verdict label, so
  the label does not discriminate between them.
- The cross-paper relabelling in `outputs/aircraft/final/` is an **LLM audit**
  (`edge_roles.json` records the model), not a human review. The physics example
  is the one that had human review.

## Live aircraft result (unchanged from 0.4.2)

An 8-paper aircraft run (`--top-k 8 --max-extractions 6`) produced:

- 184 claims, 105 relations, 36 integration deltas
- 5 candidate roots
- 33 answer links (27 support, 6 bounds)
- 17 exported paths (5 same-paper, 12 graph paths)

Three graphs were built automatically: Interactive Cockpit (62 nodes), Cybermatrix
(46), and Flying-V (56).

## Requirements

Python 3.11+, network access, an OpenAI-compatible model API key, and a Bohr
account for live LKM search. The offline `demo` needs no key.

## Verify

```bash
shasum -a 256 -c SHA256SUMS
```
