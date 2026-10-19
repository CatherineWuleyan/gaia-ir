#!/usr/bin/env bash
# Build a self-contained Ariadne release directory, tarball, and zip.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$("$ROOT/.venv-gaia-a7/bin/python" -c 'import sys; sys.path.insert(0, "'"$ROOT"'/ariadne"); from ariadne import __version__; print(__version__)' 2>/dev/null || python3 -c 'import sys; sys.path.insert(0, "'"$ROOT"'/ariadne"); from ariadne import __version__; print(__version__)')"
NAME="ariadne-$VERSION"
DIST="$ROOT/dist"
STAGE="$DIST/$NAME"

rm -rf "$STAGE"
mkdir -p "$STAGE"

cp "$ROOT/packaging/README.bundle.md" "$STAGE/README.md"
cp "$ROOT/packaging/QUICKSTART.zh-CN.md" "$STAGE/QUICKSTART.zh-CN.md"
cp "$ROOT/packaging/install.sh" "$STAGE/install.sh"
cp "$ROOT/packaging/run.sh" "$STAGE/run.sh"
cp "$ROOT/packaging/.env.example" "$STAGE/.env.example"
printf '%s\n' "$VERSION" > "$STAGE/VERSION"

# Ariadne itself (code, docs, configs, tests).
rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.DS_Store' \
  "$ROOT/ariadne/" "$STAGE/ariadne/"

# Pipeline harness and the three Ariadne pipelines.
rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude '*.egg-info/' --exclude '.DS_Store' --exclude '.env' --exclude '.env.*' \
  "$ROOT/agent/" "$STAGE/agent/"

rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude 'data/' --exclude '.DS_Store' \
  "$ROOT/pipelines/clean_claims/" "$STAGE/pipelines/clean_claims/"

rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude 'data/' --exclude '.DS_Store' \
  "$ROOT/pipelines/single_paper/" "$STAGE/pipelines/single_paper/"

rsync -a \
  --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/' \
  --exclude '.DS_Store' \
  "$ROOT/pipelines/merge/" "$STAGE/pipelines/merge/"

mkdir -p "$STAGE/weekly_reports"
cp "$ROOT/weekly_reports/export_single_artifacts.py" "$STAGE/weekly_reports/"

# Only the frozen demo assets needed by `ariadne demo`.
mkdir -p "$STAGE/outputs/aircraft/final"
for asset in domain_view_typed.json edge_roles_typed.json answer_fixture.json; do
  cp "$ROOT/outputs/aircraft/final/$asset" "$STAGE/outputs/aircraft/final/$asset"
done

# Remove platform junk and local absolute paths from bundled docs.
find "$STAGE" -name '.DS_Store' -delete
grep -RIlF "$ROOT" "$STAGE" 2>/dev/null | while IFS= read -r file; do
  sed -i.bak "s#${ROOT}#<repo>#g" "$file"
  rm -f "$file.bak"
done
find "$STAGE" -type f -exec chmod 644 {} +
chmod +x "$STAGE/install.sh" "$STAGE/run.sh" "$STAGE/ariadne/run.sh"

# Per-file manifest inside the bundle.
(
  cd "$STAGE"
  if command -v shasum >/dev/null 2>&1; then
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 shasum -a 256 > MANIFEST.sha256
  else
    find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
  fi
)

mkdir -p "$DIST"
tar -C "$DIST" -czf "$DIST/$NAME-bundle.tar.gz" "$NAME"
(
  cd "$DIST"
  rm -f "$NAME-bundle.zip"
  zip -qr "$NAME-bundle.zip" "$NAME"
)
(
  cd "$DIST"
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$NAME-bundle.tar.gz" "$NAME-bundle.zip" > SHA256SUMS
  else
    sha256sum "$NAME-bundle.tar.gz" "$NAME-bundle.zip" > SHA256SUMS
  fi
)

echo "Built:"
ls -lh "$DIST/$NAME-bundle.tar.gz" "$DIST/$NAME-bundle.zip" "$DIST/SHA256SUMS"
