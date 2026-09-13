#!/usr/bin/env python3
"""Build a single-paper V2 input manifest for one minibatch paper.

Emits `source.paper_text` + `source.claims_final` + one `source.original_figure`
per figure in the paper directory.  Supplying the figures is what lets
`agent_pipeline_v2.step2._vision_candidate()` attach an image to a figure
candidate and fall back to the DeepSeek vision tool when the caption alone is
evidence-insufficient (step2.py:1523 / :1327 / :1385).

Paths are written relative to the manifest's own directory, matching the
staging layout produced by stage_and_submit.sh:

    <stage>/manifests/test4.json          <- the manifest
    <stage>/data/minibatch/test4/ocr.md   <- ../data/minibatch/test4/ocr.md

usage:
    build_input_manifest.py --paper test4 --out bohr/single/manifests/test4.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

FIGURE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def natural_key(path: Path) -> tuple:
    """Sort figure_2 before figure_10."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", path.name)
    )


def build(paper: str, data_root: Path, manifest_dir: Path) -> dict:
    paper_dir = (data_root / paper).resolve()
    if not paper_dir.is_dir():
        raise SystemExit(f"paper directory not found: {paper_dir}")

    required = {"ocr.md": "source.paper_text", "claims_final.json": "source.claims_final"}
    artifacts: list[dict] = []

    # Paths are written relative to the manifest directory so the manifest is
    # only valid once staged; validate the shape here rather than the absolute path.
    rel_prefix = f"../data/minibatch/{paper}"
    for filename, kind in required.items():
        source = paper_dir / filename
        if not source.is_file():
            raise SystemExit(f"missing required input: {source}")
        artifacts.append({
            "path": f"{rel_prefix}/{filename}",
            "kind": kind,
            "media_type": "text/markdown" if filename.endswith(".md") else "application/json",
        })

    figures = sorted(
        (p for p in paper_dir.iterdir() if p.is_file() and p.suffix.lower() in FIGURE_SUFFIXES),
        key=natural_key,
    )
    for figure in figures:
        artifacts.append({
            "path": f"{rel_prefix}/{figure.name}",
            "kind": "source.original_figure",
            "media_type": MEDIA_TYPES[figure.suffix.lower()],
            # _vision_candidate matches the candidate's image_name against
            # metadata.source_filename, so this field is load-bearing.
            "metadata": {"source_filename": figure.name, "figure": figure.name},
        })

    return {"schema_version": "1", "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/minibatch"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    manifest = build(args.paper, args.data_root, args.out.parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for item in manifest["artifacts"]:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    print(f"wrote {args.out}  {counts}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
