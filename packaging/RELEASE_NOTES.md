# Ariadne 0.4.0

Ariadne turns a domain question into a graph-backed set of candidate answers:
LKM retrieval → paper2ir → single → merge → answer layer → self-contained
question graph.

## Download

- `ariadne-0.4.0-bundle.tar.gz` — macOS/Linux
- `ariadne-0.4.0-bundle.zip` — macOS/Linux/Windows (WSL)
- `SHA256SUMS` — archive checksums

The bundle is self-contained: Ariadne, the three pipelines, the offline aircraft
demo, `install.sh`, and the Chinese/English guides. It does not contain API
keys, Bohr credentials, local corpora, or generated runs.

## Quick start

```bash
tar -xzf ariadne-0.4.0-bundle.tar.gz
cd ariadne-0.4.0
./install.sh
./ariadne/run.sh config init
./ariadne/run.sh doctor
./ariadne/run.sh demo
```

## Highlights

- Automatic multi-query LKM retrieval with auditable paper selection.
- `--top-k` is filled from graph-backed candidates; relevance is a preference,
  not a hard cap.
- Papers without a native LKM graph can be built with `bohr lkm extract`
  (arXiv/OpenAlex/DOI PDF resolution), with `--max-extractions` and
  `--no-lkm-extract` controls.
- paper2ir retries transient failures, isolates a failed paper, reuses cached
  claims across runs, and uses a 600s read timeout.

## Requirements

Python 3.11+, network access, an OpenAI-compatible model API key, and a Bohr
account for live LKM search. The offline `demo` needs no key.

## Verify

```bash
shasum -a 256 -c SHA256SUMS
```
