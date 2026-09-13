#!/usr/bin/env bash
# Poll one Bohr job until it reaches a terminal state, then fetch log + result files.
# usage: bohr_watch.sh <BOHR_ID> <out_dir> [max_minutes]
set -uo pipefail
JOB="${1:?usage: bohr_watch.sh <BOHR_ID> <out_dir> [max_minutes]}"
OUT="${2:?out dir}"
MAX_MIN="${3:-60}"
mkdir -p "$OUT"

status_of() {
  bohr job list -n 50 -o json 2>/dev/null | python3 -c "
import json,sys
try:
    data = json.load(sys.stdin).get('data', [])
except Exception:
    print('unknown'); raise SystemExit
for item in data:
    if str(item.get('bohrId')) == '$JOB':
        print(item.get('status')); raise SystemExit
print('unknown')
"
}

deadline=$(( $(date +%s) + MAX_MIN * 60 ))
while :; do
  status="$(status_of)"
  echo "$(date -u +%H:%M:%S) job=$JOB status=$status"
  case "$status" in
    Finished|Failed|Stopped|Killed|Cancel|Cancelled|Deleted|Timeout|Zombie)
      echo "=== terminal status: $status"
      break
      ;;
  esac
  if [[ $(date +%s) -ge $deadline ]]; then
    echo "=== monitor timeout after ${MAX_MIN}m (job still $status)"
    exit 2
  fi
  sleep 30
done

echo "=== job describe"
bohr job describe -i "$JOB" -o json 2>/dev/null | python3 -c "
import json,sys
data = json.load(sys.stdin).get('data')
item = data[0] if isinstance(data, list) and data else data
for key in ('bohrId','jobName','status','webStatus','spendTime','cost','endTime','errorInfo'):
    print(f'{key}={item.get(key)}')
"

echo "=== job log -> $OUT/driver.log"
bohr job log -i "$JOB" -o json 2>/dev/null | python3 -c "
import json,sys
try:
    files = json.load(sys.stdin)['data'].get('logFiles') or []
except Exception:
    files = []
print(files[0]['url'] if files else '')
" > "$OUT/logurl.txt"
if [[ -s "$OUT/logurl.txt" ]]; then
  curl -fsSL "$(cat "$OUT/logurl.txt")" -o "$OUT/driver.log" && echo "driver.log: $(wc -c < "$OUT/driver.log") bytes"
fi

echo "=== results"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SELF_DIR/fetch_results.sh" "$JOB" "$OUT/result" || echo "result fetch failed"
echo "=== files"
find "$OUT" -maxdepth 2 -type f | head -20
