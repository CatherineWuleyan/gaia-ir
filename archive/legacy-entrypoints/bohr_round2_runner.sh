#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$ROOT/agent:$ROOT/pipelines/single_paper"
trap 'rm -f "$ROOT/agent/.env"' EXIT

python3 -m pip install --no-cache-dir -e "$ROOT/agent[gaia]"

mkdir -p "$ROOT/results/runs" "$ROOT/results/graphs"

for manifest in "$ROOT/inputs/manifests"/*.json; do
  paper_id="$(basename "$manifest" .json)"
  run_dir="$(python3 -m pipeline_harness init \
    --pipeline "$ROOT/config/pipeline.json" \
    --inputs "$manifest" \
    --runs-root "$ROOT/results/runs")"
  set +e
  python3 -m pipeline_harness run --run "$run_dir"
  status=$?
  set -e
  cp -R "$run_dir" "$ROOT/results/graphs/$paper_id"
  printf '%s\t%s\t%s\n' "$paper_id" "$status" "$run_dir" >> "$ROOT/results/status.tsv"
done

cp "$ROOT/config/pipeline.json" "$ROOT/results/pipeline.json"
cp "$ROOT/config/pipeline.commit.txt" "$ROOT/results/pipeline.commit.txt"
