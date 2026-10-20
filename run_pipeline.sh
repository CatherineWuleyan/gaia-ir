#!/usr/bin/env bash
set -euo pipefail

# One explicit entry point for the two supported workflows.  This script is
# deliberately an assembler/diagnostic wrapper: it does not add a Harness
# stage, artifact kind, schema field, or recovery rule.

ROOT="$(cd "$(dirname "$0")" && pwd)"
# The official compiler contract is tied to Gaia 0.5.0a7. Prefer the
# repository-local environment so an activated/older Gaia checkout cannot
# silently change compiler semantics.
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
# Both pipeline packages must be importable for both subcommands.  `merge` was
# missing `pipelines/merge`, so the documented `./run_pipeline.sh merge ...`
# entry point died in the preflight with
# `ModuleNotFoundError: No module named 'pipeline_merge'` -- the merge plugin
# spec is `pipeline_merge.step0:...`, which only resolves with that directory on
# the path.  Keeping one variable means the two subcommands cannot drift again.
HARNESS_PYTHONPATH="$ROOT/agent:$ROOT/pipelines/single_paper:$ROOT/pipelines/merge"

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh list
  ./run_pipeline.sh single-v2 <manifest.json> [runs-root]
  ./run_pipeline.sh merge <manifest.json> [runs-root]

Pipelines:
  single-v2  Pipeline Harness single-paper V2, canonical step1-5 config
  merge      Pipeline Merge step0-5, canonical integration config

Use pipeline_harness run --run <run-dir> to resume an existing run.
EOF
}

identity() {
  local name="$1"; local config="${2:-}"
  echo "pipeline_name=$name"
  echo "repo_root=$ROOT"
  echo "git_commit=$(git -C "$ROOT" rev-parse HEAD)"
  echo "git_dirty=$(test -n "$(git -C "$ROOT" status --porcelain)" && echo true || echo false)"
  echo "python=$PYTHON_BIN"
  if [[ -n "$config" ]]; then
    echo "config=$config"
    echo "config_sha256=$(shasum -a 256 "$config" | awk '{print $1}')"
  fi
}

module_preflight() {
  [[ -x "$PYTHON_BIN" ]] || {
    echo "Gaia Python environment not found: $PYTHON_BIN" >&2
    echo "Install requirements.txt into $ROOT/.venv (Gaia 0.5.0a7), or set PYTHON_BIN explicitly to that environment." >&2
    exit 2
  }
  PYTHONPATH="$HARNESS_PYTHONPATH" "$PYTHON_BIN" - <<'PY'
import sys
import pipeline_harness
import agent_pipeline_v2
from gaia._meta import get_library_version
print(f"pipeline_harness_module={pipeline_harness.__file__}")
print(f"agent_pipeline_v2_module={agent_pipeline_v2.__file__}")
print(f"python_executable={sys.executable}")
version = get_library_version()
print(f"gaia_version={version}")
if version != "0.5.0a7":
    raise SystemExit(f"Gaia version mismatch: required 0.5.0a7, found {version}")
PY
}

run_harness() {
  local name="$1"; local config="$2"; local manifest="$3"; local runs_root="${4:-$ROOT/outputs/harness_runs}"
  [[ -f "$manifest" ]] || { echo "manifest not found: $manifest" >&2; exit 2; }
  identity "$name" "$config"
  module_preflight
  local run_dir
  run_dir="$(PYTHONPATH="$HARNESS_PYTHONPATH" "$PYTHON_BIN" -m pipeline_harness init \
    --pipeline "$config" --inputs "$manifest" --runs-root "$runs_root")"
  echo "run_dir=$run_dir"
  PYTHONPATH="$HARNESS_PYTHONPATH" "$PYTHON_BIN" -m pipeline_harness run --run "$run_dir"
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 2; }
  case "$1" in
    list)
      usage
      ;;
    single-v2)
      [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
      run_harness "single-v2" "$ROOT/pipelines/single_paper/pipeline.json" "$2" "${3:-$ROOT/outputs/harness_runs}"
      ;;
    merge)
      [[ $# -ge 2 && $# -le 3 ]] || { usage >&2; exit 2; }
      run_harness "merge" "$ROOT/pipelines/merge/pipeline.json" "$2" "${3:-$ROOT/outputs/merge_runs}"
      ;;
    *)
      echo "unknown pipeline: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

main "$@"
