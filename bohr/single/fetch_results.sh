#!/usr/bin/env bash
# Download a finished Bohr job's result bundle.
#
# `bohr job download` in CLI 2.2.20 resolves the Bohr ID and then GETs
# /openapi/v4/job/<jobId>, which returns 404.  The working endpoint is
# /openapi/v4/job/detail/<bohrId>, which carries a signed `resultUrl` for the
# job's out.zip.  Fetch that instead.
#
# usage: fetch_results.sh <BOHR_ID> <out_dir>
set -euo pipefail

JOB="${1:?usage: fetch_results.sh <BOHR_ID> <out_dir>}"
OUT="${2:?out dir}"
mkdir -p "$OUT"

detail="$(bohr api GET "/openapi/v4/job/detail/$JOB" -o json 2>/dev/null)"
url="$(printf '%s' "$detail" | python3 -c "
import json,sys
try:
    data = json.load(sys.stdin)['data']
except Exception:
    print(''); raise SystemExit
print(data.get('resultUrl') or data.get('result') or '')
")"

if [[ -z "$url" ]]; then
  echo "no resultUrl for job $JOB (job may still be running)" >&2
  exit 1
fi

zip="$OUT/out.zip"
curl -fsSL "$url" -o "$zip"
echo "downloaded $(du -h "$zip" | cut -f1) -> $zip"
rm -rf "$OUT/unpacked"
mkdir -p "$OUT/unpacked"
unzip -q -o "$zip" -d "$OUT/unpacked"
echo "unpacked into $OUT/unpacked"
find "$OUT/unpacked" -maxdepth 2 -type d | head -10
