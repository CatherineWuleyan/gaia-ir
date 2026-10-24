# Packaging Ariadne

Ariadne is not a pure Python package: the CLI orchestrates the maintained
pipeline trees and needs demo graph assets. The release bundle is therefore a
self-contained directory with Ariadne, `agent/`, the three pipelines, the
offline aircraft demo, `install.sh`, and a per-file manifest.

## Build

```bash
./packaging/build_release.sh
```

Outputs in `dist/`:

```text
ariadne-<version>/
ariadne-<version>-bundle.tar.gz
ariadne-<version>-bundle.zip
SHA256SUMS
```

The script:

- copies only runtime code and demo assets;
- excludes local corpora, generated runs, virtualenvs, caches, and secrets;
- sanitizes local absolute paths from bundled docs;
- writes `MANIFEST.sha256` and archive checksums.

## Publish

```bash
gh release create v0.5.0 \
  dist/ariadne-0.5.0-bundle.tar.gz \
  dist/ariadne-0.5.0-bundle.zip \
  dist/SHA256SUMS \
  --title "Ariadne 0.5.0" \
  --notes-file packaging/RELEASE_NOTES.md
```

## Test the bundle before publishing

```bash
rm -rf /tmp/ariadne-bundle-test
mkdir -p /tmp/ariadne-bundle-test
tar -xzf dist/ariadne-0.5.0-bundle.tar.gz -C /tmp/ariadne-bundle-test
cd /tmp/ariadne-bundle-test/ariadne-0.5.0
./install.sh
./ariadne/run.sh doctor
./ariadne/run.sh demo
```
