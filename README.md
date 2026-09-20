# Gaia IR

Gaia IR is the current research pipeline for turning scientific papers into auditable reasoning graphs and Gaia IR packages.

The repository contains the maintained pipeline, execution harness, formalization rules, multi-paper merge flow, and reviewable viewers. Older one-off extraction code and generated experiment outputs are kept out of the main source surface.

## What the pipeline does

```text
paper input
  → evidence and claim extraction
  → claim normalization
  → reasoning and weak-point analysis
  → formalization
  → Gaia IR compilation and validation
  → audit artifacts and viewer
```

The single-paper flow and the merge flow share the same artifact and provenance contracts. Each successful stage records immutable inputs, configuration hashes, artifact hashes, checkpoints, and structured findings so a run can be resumed and inspected.

## Repository map

- `agent/` — reusable runner, artifact store, checkpointing, validation, and viewer harness.
- `pipelines/single_paper/` — maintained single-paper Pipeline 7.0 flow and its tests.
- `pipelines/merge/` — multi-paper integration flow and delta generation.
- `pipelines/clean_claims/` — compatibility entry points for the earlier claim-cleaning workflow.
- `docs/` — Gaia IR contract, identity, hashing, canonicalization, lowering, and validation notes.
- `examples/` — small reviewable examples and viewer artifacts.
- `bohr/` — scripts for the external experiment environment; these are auxiliary and are not the default local entry point.
- `archive/` — historical configurations and entry points retained for reference.

Generated runs, caches, virtual environments, local secrets, and large intermediate datasets are local artifacts and are intentionally excluded from version control.

## Quick start

The repository expects the bundled Gaia `0.5.0a7` environment:

```bash
./run_pipeline.sh list
./run_pipeline.sh single-v2 /absolute/path/to/manifest.json
./run_pipeline.sh merge /absolute/path/to/manifest.json
```

The launcher prints the selected pipeline, configuration hash, Git revision, worktree status, and imported module path before running. Run from the repository root.

For the compatibility workflow:

```bash
./run_pipeline.sh legacy <paper_id>
```

Use the full pipeline configuration when continuing a historical run. The step-specific JSON files are for isolated development and tests; they do not resume an existing run by themselves.

## Tests

Run the fast harness and pipeline tests with:

```bash
python -m unittest discover -s agent/tests -p 'test_*.py'
python -m unittest discover -s pipelines/single_paper/tests -p 'test_*.py'
```

The tests are offline by default. Semantic tools and external model calls are only used when explicitly configured.

## Design principles

The runner owns execution, persistence, checkpointing, and audit state. Domain plugins own paper reasoning and formalization. Plugins communicate through versioned artifacts rather than shared in-memory objects. The final Gaia IR is produced by an explicit official compiler and is accepted only after structural and provenance checks pass.

The Viewer is a projection of stored artifacts. It is never the authoring source of truth.

## Status

This repository is the active Gaia IR implementation. `paper2ir` remains a separate repository for the earlier standalone paper-to-claim-network workflow.
