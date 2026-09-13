#!/usr/bin/env bash
# Job-side runner: execute ONE pipeline_merge (domain-integration) run on Bohrium
# over the five minibatch paper packages test1..test5, then project the domain
# graph viewer.
#
# Usage (inside the Bohrium job container, cwd = uploaded input directory root):
#   bash job_run.sh          # full merge run + domain graph projection
#   bash job_run.sh smoke    # environment check only, no pipeline run
#
# Layout expected at the staging root (see stage_and_submit.sh):
#   agent/                       pipeline_harness kernel + .env with DeepSeek settings
#   pipelines/single_paper/      agent_pipeline_v2 code (merge reuses its helpers)
#   pipelines/merge/             pipeline_merge code + the 5-paper config
#   data/minibatch/<paper>/      gaia.ir.json + formalization.json + knowledge.index.json
#   manifests/test1-5.json       merge input manifest (relative paths)
#   PROVENANCE.txt               local git commit / dirty flag / file hashes
#
# Only results/ is uploaded back (backward_files = ["results/"]), so the DeepSeek
# key in agent/.env is never copied to the personal disk.
set -uo pipefail

MODE="${1:-run}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
cd "$ROOT"

RESULTS="$ROOT/results"
LOGS="$RESULTS/logs"
RUNS_ROOT="$RESULTS/runs"
mkdir -p "$LOGS" "$RUNS_ROOT"

CONFIG="$ROOT/pipelines/merge/pipeline.test1-5.json"
MANIFEST="$ROOT/manifests/test1-5.json"

say() { printf '[merge %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

say "mode=$MODE host=$(hostname) pwd=$ROOT"
python3 --version 2>&1 | sed 's/^/python3: /'
say "PROVENANCE:"
sed 's/^/    /' "$ROOT/PROVENANCE.txt" 2>/dev/null || true

# ---------------------------------------------------------------- DeepSeek env
# NOTE: agent/.env is written as `DEEPSEEK_API_KEY= sk-...` (a space after `=`),
# which the pipeline's own parser tolerates (`value.strip()`) but POSIX `source`
# does not.  Parse it the same tolerant way.
load_env_file() {
  local file="$1" line key value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*$ ]] && continue
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"
    value="${line#*=}"
    key="${key//[[:space:]]/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    value="${value%\"}"; value="${value#\"}"
    value="${value%\'}"; value="${value#\'}"
    [[ -n "$key" ]] && export "$key=$value"
  done < "$file"
}

if [[ -f "$ROOT/agent/.env" ]]; then
  load_env_file "$ROOT/agent/.env"
  say "loaded agent/.env"
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  say "FATAL DEEPSEEK_API_KEY is not set"
  exit 3
fi
say "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-<unset>} DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-<unset>} DEEPSEEK_MERGE_MODEL=${DEEPSEEK_MERGE_MODEL:-<default>}"

export PYTHONPATH="$ROOT/agent:$ROOT/pipelines/single_paper:$ROOT/pipelines/merge:$ROOT"
export PYTHONDONTWRITEBYTECODE=1

# Step 3/4 time is dominated by remote DeepSeek HTTP round-trips, not local
# compute, so the only useful parallelism is HTTP concurrency.  A GPU machine
# would idle here and cost more.
export GAIA_MERGE_RETRIEVAL_WORKERS="${GAIA_MERGE_RETRIEVAL_WORKERS:-16}"
export GAIA_MERGE_JUDGE_WORKERS="${GAIA_MERGE_JUDGE_WORKERS:-8}"
say "llm concurrency: retrieval=${GAIA_MERGE_RETRIEVAL_WORKERS} judge=${GAIA_MERGE_JUDGE_WORKERS}"

# ------------------------------------------------- egress + index reachability
python3 - <<'PY' 2>&1 | sed 's/^/    /'
import urllib.request
targets = {
    "pypi.org": "https://pypi.org/simple/",
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple/",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/",
    "deepseek": "https://api.deepseek.com/v1/models",
}
for name, url in targets.items():
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=25) as response:
            print(f"egress {name}: {response.status}")
    except Exception as exc:  # noqa: BLE001
        print(f"egress {name}: {type(exc).__name__}: {exc}")
PY
say "pip config:"
python3 -m pip config list 2>&1 | sed 's/^/    /'

# ------------------------------------------------------------------ install deps
# The image's pip points at a Tsinghua mirror that can return 403 for individual
# wheels, so install the published gaia-lang wheel instead of building the local
# harness (pipeline_harness already resolves from PYTHONPATH).
PINS=(
  "numpy==2.3.2" "scipy==1.18.1" "sympy==1.14.0" "Pint==0.25.3"
  "pydantic==2.13.4" "pydantic_core==2.46.4"
)

try_install() {
  local label="$1"; shift
  say "pip attempt: $label"
  {
    echo "=== $(date -u +%FT%TZ) attempt: $label ($* gaia-lang==0.5.0a7 ${PINS[*]}) ==="
  } >> "$LOGS/pip.log"
  python3 -m pip install --no-cache-dir --disable-pip-version-check --no-input -q \
    "$@" "gaia-lang==0.5.0a7" "${PINS[@]}" >> "$LOGS/pip.log" 2>&1
}

pip_status=1
try_install "default index + pinned" && pip_status=0
[[ "$pip_status" -ne 0 ]] && { try_install "default index + pinned + break-system-packages" --break-system-packages && pip_status=0; }
[[ "$pip_status" -ne 0 ]] && { try_install "pypi.org + pinned" --break-system-packages -i https://pypi.org/simple && pip_status=0; }
[[ "$pip_status" -ne 0 ]] && { try_install "aliyun + pinned" --break-system-packages -i https://mirrors.aliyun.com/pypi/simple && pip_status=0; }
[[ "$pip_status" -ne 0 ]] && { try_install "tuna + pinned" --break-system-packages -i https://pypi.tuna.tsinghua.edu.cn/simple && pip_status=0; }
[[ "$pip_status" -ne 0 ]] && { try_install "default index + unpinned" --break-system-packages && pip_status=0; }

if [[ "$pip_status" -ne 0 ]]; then
  say "FATAL pip install failed; last 40 lines of pip.log:"
  tail -40 "$LOGS/pip.log"
  exit 4
fi
say "pip install ok"

python3 -m pip freeze > "$RESULTS/pip-freeze.txt" 2>/dev/null
python3 - <<'PY'
import sys
from gaia._meta import get_library_version
version = get_library_version()
print(f"gaia_version={version} python={sys.version.split()[0]}")
if version != "0.5.0a7":
    raise SystemExit(f"gaia version mismatch: required 0.5.0a7, found {version}")
PY
if [[ $? -ne 0 ]]; then
  say "FATAL gaia preflight failed"
  exit 5
fi
say "gaia preflight ok"

# ------------------------------------------------------- DeepSeek reachability
python3 - <<'PY' 2>&1 | tail -5
import json, os, urllib.request
base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
body = json.dumps({
    "model": os.environ.get("DEEPSEEK_MERGE_MODEL", os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")),
    "messages": [{"role": "user", "content": "ok"}],
    "max_tokens": 1,
}).encode()
req = urllib.request.Request(
    f"{base}/chat/completions", data=body,
    headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
             "Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(req, timeout=60) as response:
        print(f"deepseek_reachable=true status={response.status}")
except Exception as exc:  # noqa: BLE001
    print(f"deepseek_reachable=false error={type(exc).__name__}: {exc}")
PY

if [[ "$MODE" == "smoke" ]]; then
  say "smoke check complete (environment verified, no pipeline run)"
  exit 0
fi

# ------------------------------------------------------------------ the run
[[ -f "$CONFIG" ]] || { say "FATAL missing config $CONFIG"; exit 6; }
[[ -f "$MANIFEST" ]] || { say "FATAL missing manifest $MANIFEST"; exit 6; }

started="$(date +%s)"
run_dir="$(python3 -m pipeline_harness init \
  --pipeline "$CONFIG" --inputs "$MANIFEST" --runs-root "$RUNS_ROOT" 2>>"$LOGS/merge.init.log")"
if [[ -z "$run_dir" ]]; then
  say "FATAL init failed"
  tail -20 "$LOGS/merge.init.log"
  printf 'merge\t%s\t%s\n' 6 "init_failed" >> "$RESULTS/status.tsv"
  exit 6
fi
say "run_dir=$run_dir"

# Progress reporter: the harness is quiet until a stage finishes, so surface the
# persisted run record every 60s to make `bohr job log` useful while it runs.
(
  while true; do
    sleep 60
    python3 - "$run_dir" <<'PY' 2>/dev/null
import json, sys, pathlib
record = json.loads((pathlib.Path(sys.argv[1]) / "run.json").read_text())
print(f"[progress] stage={record.get('current_stage')} "
      f"next_index={record.get('next_stage_index')} status={record.get('status')} "
      f"attempts={record.get('attempts')}", flush=True)
PY
  done
) &
reporter=$!
trap 'kill "$reporter" 2>/dev/null' EXIT

python3 -m pipeline_harness run --run "$run_dir" \
  > "$LOGS/merge.run.json" 2> "$LOGS/merge.stderr.log"
status=$?
kill "$reporter" 2>/dev/null
trap - EXIT

elapsed=$(( $(date +%s) - started ))
printf 'merge\t%s\t%s\t%ss\t%s\n' "$status" "$(date -u +%FT%TZ)" "$elapsed" "$run_dir" \
  >> "$RESULTS/status.tsv"
say "finished merge exit=$status elapsed=${elapsed}s"

# ---------------------------------------------------- domain graph projection
# `pipeline_harness run` never projects a view; the merge viewer (claims /
# operators / weakpoints layers) is produced explicitly here.  viewer.html is
# self-contained (dagre + view data are inlined), so it is the deliverable.
if [[ "$status" -eq 0 ]]; then
  say "projecting domain graph view"
  if python3 -m pipeline_harness view --run "$run_dir" > "$LOGS/view.log" 2>&1; then
    cp -f "$run_dir/views/viewer.html" "$RESULTS/domain_graph.html" 2>/dev/null || true
    cp -f "$run_dir/views/view_model.json" "$RESULTS/view_model.json" 2>/dev/null || true
    say "domain graph -> $RESULTS/domain_graph.html"
  else
    say "WARN view projection failed; see $LOGS/view.log"
    tail -20 "$LOGS/view.log" | sed 's/^/    /'
  fi
  python3 -m pipeline_harness check --run "$run_dir" > "$RESULTS/check.json" 2>"$LOGS/check.stderr.log" || true
fi

python3 - "$run_dir" <<'PY' 2>/dev/null | sed 's/^/    /'
import json, sys, pathlib
run_dir = pathlib.Path(sys.argv[1])
record = json.loads((run_dir / "run.json").read_text())
print(f"run_status={record.get('status')}")
for finding in record.get("findings", []):
    print(f"finding[{finding.get('severity')}] {finding.get('code')}: {finding.get('message')}")
PY

# Keep the run tree (artifacts, frozen inputs, work files, views) inside results/
# so the whole merge is reproducible from the fetched bundle.
if [[ "$status" -eq 0 ]]; then
  cp -a "$run_dir" "$RESULTS/merge_run" 2>/dev/null || true
  say "results bundle: $(du -sh "$RESULTS" | awk '{print $1}')"
fi

exit "$status"
