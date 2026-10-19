#!/usr/bin/env bash
# Convenience wrapper for ./ariadne/run.sh at the bundle root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/ariadne/run.sh" "$@"
