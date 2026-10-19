# Ariadne 0.4.1

Ariadne turns a domain question into a graph-backed set of candidate answers:
LKM retrieval → paper2ir → single → merge → answer layer → self-contained
question graph.

This release is a **relevance and packaging fix** for 0.4.0.

## Download

- `ariadne-0.4.1-bundle.tar.gz` — macOS/Linux
- `ariadne-0.4.1-bundle.zip` — macOS/Linux/Windows (WSL)
- `SHA256SUMS` — archive checksums

The bundle is self-contained: Ariadne, the three pipelines, the offline aircraft
demo, `install.sh`, and the Chinese/English guides. It does not contain API
keys, Bohr credentials, local corpora, or generated runs.

## Quick start

```bash
tar -xzf ariadne-0.4.1-bundle.tar.gz
cd ariadne-0.4.1
./install.sh
./ariadne/run.sh config init
./ariadne/run.sh doctor
./ariadne/run.sh demo
```

## What changed

- Fixed `_relevance()` word-boundary escaping; relevance scores were always
  zero, so `--top-k` could fill with off-topic LKM papers.
- Added aircraft/flight-simulation phrases to the relevance vocabulary.
- Added a fill floor (`min_relevance * 0.5`) so Ariadne does not pad `--top-k`
  with clearly off-topic papers.
- Selection audit now records `fill_floor` and `filled`.
- Rebuilt the release bundle and download links for 0.4.1.

## Live aircraft trial

A 5-paper aircraft-design run from the bundle selects five on-topic papers and
produces:

- 71 claims, 55 relations, 12 integration deltas
- 5 candidate roots
- 32 answer links (24 support, 8 bounds)
- 12 chains (5 same-paper, 7 graph paths)

The graph is a focused 5-paper map, not a complete 13-paper domain review. Empty
native LKM graphs remain the main coverage bottleneck.

## Requirements

Python 3.11+, network access, an OpenAI-compatible model API key, and a Bohr
account for live LKM search. The offline `demo` needs no key.

## Verify

```bash
shasum -a 256 -c SHA256SUMS
```
