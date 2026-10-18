#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="${ARIADNE_PYTHON:-$REPO/.venv-gaia-a7/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="${ARIADNE_PYTHON:-python3}"
fi
export PYTHONPATH="$HERE:$REPO:$REPO/agent:$REPO/pipelines/single_paper:$REPO/pipelines/merge${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m ariadne "$@"
