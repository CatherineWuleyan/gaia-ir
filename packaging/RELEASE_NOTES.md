# Ariadne 0.4.2

Ariadne turns a domain question into a graph-backed set of candidate answers:
LKM retrieval → paper2ir → single → merge → answer layer → self-contained
question graph.

This release integrates **empty-graph resolution and parallel graph building**
into the normal retrieval stage.

## Download

- `ariadne-0.4.2-bundle.tar.gz` — macOS/Linux
- `ariadne-0.4.2-bundle.zip` — macOS/Linux/Windows (WSL)
- `SHA256SUMS` — archive checksums

The bundle is self-contained: Ariadne, the three pipelines, the offline aircraft
demo, `install.sh`, and the Chinese/English guides. It does not contain API
keys, Bohr credentials, local corpora, or generated runs.

## Quick start

```bash
tar -xzf ariadne-0.4.2-bundle.tar.gz
cd ariadne-0.4.2
./install.sh
./ariadne/run.sh config init
./ariadne/run.sh doctor
./ariadne/run.sh demo
```

## Empty graphs are now an automatic step

When a candidate has no native LKM graph:

1. Ariadne resolves a PDF through arXiv, DOI landing pages
   (`citation_pdf_url`), OpenAlex, Unpaywall, Semantic Scholar, Crossref, and
   MDPI's CDN.
2. It downloads and validates the PDF **before** spending an extraction call.
3. It runs `bohr lkm extract` in parallel (3 workers) for up to
   `retrieval.max_extractions` papers (default 5).
4. Successful graphs re-enter the same selection/materialisation path.
5. Closed-access PDFs, PDFs over 50 pages, and extraction failures are recorded
   in `selection.json` with an explicit `extract_skip_reason`.

`--top-k` is a runtime parameter; it controls the number of selected papers and
does not change the number of answer roots.

## Live aircraft result

An 8-paper aircraft run (`--top-k 8 --max-extractions 6`) produced:

- 184 claims, 105 relations, 36 integration deltas
- 5 candidate roots
- 33 answer links (27 support, 6 bounds)
- 17 chains (5 same-paper, 12 graph paths)

Three graphs were built automatically: Interactive Cockpit (62 nodes), Cybermatrix
(46), and Flying-V (56).

## Requirements

Python 3.11+, network access, an OpenAI-compatible model API key, and a Bohr
account for live LKM search. The offline `demo` needs no key.

## Verify

```bash
shasum -a 256 -c SHA256SUMS
```
