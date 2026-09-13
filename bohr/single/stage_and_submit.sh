#!/usr/bin/env bash
# Assemble the Bohrium job payload for the single-paper V2 pipeline and submit
# (or dry-run) one job per minibatch test.
#
#   ./stage_and_submit.sh stage                 # stage every paper in PAPERS
#   ./stage_and_submit.sh stage test4           # stage only test4
#   ./stage_and_submit.sh dry-run test4         # validate the Bohrium payload
#   ./stage_and_submit.sh submit  test4         # real submission (billable)
#   ./stage_and_submit.sh smoke                 # environment check, no pipeline
#
# `submit`/`dry-run` stage the paper first if its payload is not already built.
#
# The payload is staged PER PAPER on purpose: Bohr's file upload endpoint
# (`POST https://tiefblue.dp.tech/api/upload/binary`) fails with
# "Client.Timeout exceeded while awaiting headers" once the directory grows much
# past a few MB, and every job uploads the whole directory.  A payload holding
# all three papers plus their figures is ~11 MB and never completes.
#
# Everything the job needs is copied into STAGE_DIR; nothing else from the
# repository (no .git, no .venv-gaia-a7) is uploaded.
set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SELF_DIR/../.." && pwd)"

PROJECT_ID="${PROJECT_ID:-4655935}"
MACHINE_TYPE="${MACHINE_TYPE:-c8_m32_cpu}"
IMAGE_ADDRESS="${IMAGE_ADDRESS:-registry.dp.tech/dptech/ubuntu:ubuntu24.04-py3.12}"
MAX_RUN_TIME="${MAX_RUN_TIME:-240}"
RESULT_ROOT="${RESULT_ROOT:-/personal/bohr_single_latest}"

PAPERS=(${PAPERS:-test3 test4 test5})
STAGE_DIR="${STAGE_DIR:-}"   # empty => derived per paper below

die() { echo "error: $*" >&2; exit 1; }

stage_dir_for() {
  [[ -n "$STAGE_DIR" ]] && { printf '%s' "$STAGE_DIR"; return; }
  if [[ -n "${1:-}" ]]; then printf '/tmp/gaia_bohr_single_%s' "$1"
  else printf '/tmp/gaia_bohr_single_stage'; fi
}

copy_code() {
  local dest="$1"
  rsync -a \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.DS_Store' \
    --exclude 'tests/' \
    "$REPO/agent/" "$dest/agent/"

  mkdir -p "$dest/pipelines/single_paper"
  rsync -a \
    --exclude '__pycache__/' --exclude '.pytest_cache/' --exclude '.DS_Store' \
    --exclude 'tests/' \
    "$REPO/pipelines/single_paper/" "$dest/pipelines/single_paper/"
  # The cleaner writes into claim_cleaner/data; ship the directory empty so the
  # remote run does not inherit local working outputs.
  rm -rf "$dest/pipelines/single_paper/agent_pipeline_v2/claim_cleaner/data"
  mkdir -p "$dest/pipelines/single_paper/agent_pipeline_v2/claim_cleaner/data"

  cp "$REPO/run_pipeline.sh" "$dest/"
  cp "$SELF_DIR/job_run.sh" "$dest/"
  mkdir -p "$dest/manifests"
  cp "$SELF_DIR/manifests/"*.json "$dest/manifests/"
  chmod +x "$dest/job_run.sh"
}

copy_paper_data() {
  local dest="$1" paper="$2" source="$REPO/data/minibatch/$paper"
  mkdir -p "$dest/data/minibatch/$paper"
  cp "$source/ocr.md" "$source/claims_final.json" "$dest/data/minibatch/$paper/"
  # Figure images feed step2's vision fallback; without them a paper whose
  # results live only in plots yields no observations (see RUNS.md / README.md).
  if [[ "${INCLUDE_FIGURES:-1}" == "1" ]]; then
    if compgen -G "$source/*.jpg" > /dev/null; then
      cp "$source/"*.jpg "$dest/data/minibatch/$paper/"
    fi
    if compgen -G "$source/*.png" > /dev/null; then
      cp "$source/"*.png "$dest/data/minibatch/$paper/"
    fi
  fi
}

# Environment smoke check needs the code only, so its payload stays tiny.
stage_code_only() {
  local dest
  dest="$(stage_dir_for)"
  echo "== staging code only -> $dest"
  rm -rf "$dest"
  mkdir -p "$dest"
  copy_code "$dest"
  STAGE_DIR="$dest"
  write_provenance
  echo "== staged $(du -sh "$dest" | awk '{print $1}')"
}

stage() {
  local papers=("$@")
  [[ ${#papers[@]} -gt 0 ]] || papers=("${PAPERS[@]}")
  local dest
  if [[ ${#papers[@]} -eq 1 ]]; then
    dest="$(stage_dir_for "${papers[0]}")"
  else
    dest="$(stage_dir_for)"
  fi

  echo "== staging ${papers[*]} -> $dest"
  rm -rf "$dest"
  mkdir -p "$dest"
  copy_code "$dest"
  for paper in "${papers[@]}"; do
    copy_paper_data "$dest" "$paper"
  done
  STAGE_DIR="$dest"
  write_provenance
  echo "== staged $(du -sh "$dest" | awk '{print $1}')"
}

write_provenance() {
  local out="$STAGE_DIR/PROVENANCE.txt"
  {
    echo "pipeline_name=single-v2 (agent-pipeline-v2.1-step1-5)"
    echo "staged_at_utc=$(date -u +%FT%TZ)"
    echo "local_repo=$REPO"
    echo "local_git_commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "local_git_dirty=$(test -n "$(git -C "$REPO" status --porcelain)" && echo true || echo false)"
    echo "pipeline_config=pipelines/single_paper/pipeline.step1-5.json"
    echo "pipeline_config_sha256=$(shasum -a 256 "$STAGE_DIR/pipelines/single_paper/pipeline.step1-5.json" | awk '{print $1}')"
    echo
    echo "--- pipeline source hashes (sha256) ---"
    ( cd "$STAGE_DIR" && find agent/pipeline_harness pipelines/single_paper/agent_pipeline_v2 \
        -name '*.py' -not -path '*/data/*' | LC_ALL=C sort | xargs shasum -a 256 )
    echo
    echo "--- input hashes (sha256) ---"
    ( cd "$STAGE_DIR" && shasum -a 256 data/minibatch/*/ocr.md data/minibatch/*/claims_final.json data/minibatch/*/*.jpg 2>/dev/null )
  } > "$out"
}

write_job_json() {
  local paper="$1" dest="$2"
  cat > "$dest" <<EOF
{
  "job_name": "gaia-fig-$paper",
  "command": "mkdir -p results && bash job_run.sh $paper > results/$paper.driver.log 2>&1",
  "log_file": "results/$paper.driver.log",
  "backward_files": ["results/"],
  "project_id": $PROJECT_ID,
  "machine_type": "$MACHINE_TYPE",
  "image_address": "$IMAGE_ADDRESS",
  "max_run_time": $MAX_RUN_TIME,
  "result_path": "$RESULT_ROOT/$paper"
}
EOF
}

ensure_stage() {
  local paper="$1"
  if [[ ! -f "$STAGE_DIR/manifests/$paper.json" || ! -d "$STAGE_DIR/data/minibatch/$paper" ]]; then
    stage "$paper"
  fi
}

submit() {
  local mode="$1" paper="$2"
  [[ -f "$STAGE_DIR/manifests/$paper.json" ]] || die "unknown paper: $paper"
  local job_json="$STAGE_DIR/job.$paper.json"
  write_job_json "$paper" "$job_json"
  echo "== $mode $paper (payload $(du -sh "$STAGE_DIR" | awk '{print $1}'))"
  if [[ "$mode" == "dry-run" ]]; then
    bohr job submit -i "$job_json" --input_directory "$STAGE_DIR" --dry-run -o json
  else
    bohr job submit -i "$job_json" --input_directory "$STAGE_DIR" -y -o json
  fi
}

submit_smoke() {
  local job_json="$STAGE_DIR/job.smoke.json"
  cat > "$job_json" <<EOF
{
  "job_name": "gaia-single-smoke",
  "command": "mkdir -p results && bash job_run.sh smoke > results/smoke.driver.log 2>&1",
  "log_file": "results/smoke.driver.log",
  "backward_files": ["results/"],
  "project_id": $PROJECT_ID,
  "machine_type": "$MACHINE_TYPE",
  "image_address": "$IMAGE_ADDRESS",
  "max_run_time": 30,
  "result_path": "$RESULT_ROOT/_smoke"
}
EOF
  echo "== submit smoke (environment check only, no pipeline run)"
  bohr job submit -i "$job_json" --input_directory "$STAGE_DIR" -y -o json
}

case "${1:-}" in
  stage)
    shift
    if [[ $# -ge 1 ]]; then stage "$@"; else stage; fi
    ;;
  dry-run)
    [[ $# -eq 2 ]] || die "usage: $0 dry-run <paper>"
    STAGE_DIR="$(stage_dir_for "$2")"; ensure_stage "$2"; submit dry-run "$2"
    ;;
  submit)
    [[ $# -eq 2 ]] || die "usage: $0 submit <paper>"
    STAGE_DIR="$(stage_dir_for "$2")"; ensure_stage "$2"; submit submit "$2"
    ;;
  smoke)
    STAGE_DIR="$(stage_dir_for)"; stage_code_only; submit_smoke
    ;;
  *)
    echo "usage: $0 {stage [paper...]|dry-run <paper>|submit <paper>|smoke}" >&2
    exit 2
    ;;
esac
