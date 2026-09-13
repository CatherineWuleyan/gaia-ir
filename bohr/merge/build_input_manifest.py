#!/usr/bin/env python3
"""Build a pipeline_merge input manifest from minibatch paper folders.

Each selected paper contributes the three artifacts merge Step 1 understands:

    data/minibatch/<paper>/gaia.ir.json          kind: gaia.ir          (required)
    data/minibatch/<paper>/formalization.json    kind: formalization    (optional)
    data/minibatch/<paper>/knowledge.index.json  kind: knowledge.index  (optional)

`gaia.ir` is the only required content input.  The other two are supplied when
present so merge can reuse the author-side weakpoint/revision context and the
pre-built retrieval index instead of rebuilding one.

Paths are written relative to the manifest's own directory, matching the staging
layout produced by `stage_and_submit.sh`:

    <stage>/manifests/test1-5.json                     <- the manifest
    <stage>/data/minibatch/test1/gaia.ir.json          <- ../data/minibatch/test1/gaia.ir.json

usage:
    build_input_manifest.py --out bohr/merge/manifests/test1-5.json test1 test2 test3 test4 test5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "minibatch"

ARTIFACTS = (
    ("gaia.ir.json", "gaia.ir", "application/json", True),
    ("formalization.json", "formalization", "application/json", False),
    ("knowledge.index.json", "knowledge.index", "application/json", False),
)


def _identity(document: dict, field: str) -> str:
    return f"{document.get(field, '')}"


def _check_package_alignment(paper_dir: Path) -> None:
    """Fail closed when the three artifacts do not describe one Package."""
    ir = json.loads((paper_dir / "gaia.ir.json").read_text(encoding="utf-8"))
    namespace, name = ir.get("namespace"), ir.get("package_name")
    if not namespace or not name:
        raise SystemExit(f"{paper_dir}/gaia.ir.json has no Package identity")
    formalization_path = paper_dir / "formalization.json"
    if formalization_path.is_file():
        package = json.loads(formalization_path.read_text(encoding="utf-8")).get("package") or {}
        if (package.get("namespace"), package.get("name")) != (namespace, name):
            raise SystemExit(
                f"{paper_dir}/formalization.json belongs to "
                f"{package.get('namespace')}:{package.get('name')}, expected {namespace}:{name}"
            )


def build(papers: list[str], data_root: Path) -> dict:
    artifacts: list[dict] = []
    for paper in papers:
        paper_dir = (data_root / paper).resolve()
        if not paper_dir.is_dir():
            raise SystemExit(f"paper directory not found: {paper_dir}")
        _check_package_alignment(paper_dir)
        rel_prefix = f"../data/minibatch/{paper}"
        for filename, kind, media_type, required in ARTIFACTS:
            source = paper_dir / filename
            if not source.is_file():
                if required:
                    raise SystemExit(f"missing required input: {source}")
                print(f"note: skipping absent optional input {source}", file=__import__("sys").stderr)
                continue
            artifacts.append({
                "path": f"{rel_prefix}/{filename}",
                "kind": kind,
                "media_type": media_type,
                "required": required,
            })
    return {"schema_version": "1", "artifacts": artifacts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", nargs="+", help="minibatch folder names, e.g. test1 test2")
    parser.add_argument("--out", required=True, type=Path, help="manifest path to write")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    manifest = build(args.papers, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(manifest['artifacts'])} artifacts for {len(args.papers)} papers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
