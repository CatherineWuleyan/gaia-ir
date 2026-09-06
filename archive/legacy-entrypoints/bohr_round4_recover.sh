#!/usr/bin/env bash
set -u
cd /workspace/gaia-round3
export DEEPSEEK_API_KEY="$(sed -n 's/^DEEPSEEK_API_KEY=[[:space:]]*//p' agent/.env)"
export DEEPSEEK_BASE_URL="$(sed -n 's/^DEEPSEEK_BASE_URL=[[:space:]]*//p' agent/.env)"
export DEEPSEEK_MODEL="$(sed -n 's/^DEEPSEEK_MODEL=[[:space:]]*//p' agent/.env)"
export PYTHONPATH="/home/user/.local/lib/python3.12/site-packages:/workspace/gaia-official:/workspace/gaia-round3/agent:/workspace/gaia-round3/pipelines/single_paper:/workspace/gaia-vendor"
mkdir -p results_fixed/runs results_fixed/graphs
touch results_fixed/status.tsv
for manifest in inputs/manifests/*.json; do
  paper_id="$(basename "$manifest" .json)"
  if awk -F '\t' -v p="$paper_id" '$1==p && $2==0 {ok=1} END{exit !ok}' results_fixed/status.tsv; then
    continue
  fi
  run_dir="$(python3 -m pipeline_harness init --pipeline config/pipeline.json --inputs "$manifest" --runs-root results_fixed/runs)" || { printf '%s\t1\tinit_failed\n' "$paper_id" >> results_fixed/status.tsv; continue; }
  if python3 -m pipeline_harness run --run "$run_dir"; then
    rm -rf "results_fixed/graphs/$paper_id"
    cp -R "$run_dir" "results_fixed/graphs/$paper_id"
    printf '%s\t0\t%s\n' "$paper_id" "$run_dir" >> results_fixed/status.tsv
  else
    printf '%s\t1\t%s\n' "$paper_id" "$run_dir" >> results_fixed/status.tsv
  fi
done
