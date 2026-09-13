#!/usr/bin/env python3
"""Copy the final artifacts of a harness run back into data/minibatch/<paper>/.

Mirrors how data/minibatch/test1 and test2 were recorded:
  formalization.json + gaia.ir.json + knowledge.index.json + correct_run_provenance.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

KIND_TO_OUTPUT = {
    "formalization": "formalization.json",
    "gaia.ir": "gaia.ir.json",
    "knowledge.index": "knowledge.index.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(run_dir: Path) -> list[dict]:
    index_path = run_dir / "artifacts" / "index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"] if isinstance(payload, dict) else payload
    return [item for item in artifacts if item.get("kind") in KIND_TO_OUTPUT]


def pick(artifacts: list[dict], kind: str) -> dict:
    candidates = [item for item in artifacts if item.get("kind") == kind]
    if not candidates:
        raise SystemExit(f"no artifact of kind {kind!r} in this run")
    if kind == "formalization":
        # Prefer the step-4 revision; fall back to the last one produced.
        candidates.sort(key=lambda item: item.get("metadata", {}).get("step") or 0)
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path, help="downloaded run directory")
    parser.add_argument("--paper", required=True, help="e.g. test3")
    parser.add_argument("--dest-root", type=Path, default=Path("data/minibatch"))
    args = parser.parse_args()

    run_dir = args.run.resolve()
    run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    artifacts = load_artifacts(run_dir)

    dest = (args.dest_root / args.paper).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, object] = {"run_id": run_record.get("run_id")}
    id_keys = {
        "formalization": "formalization_artifact_id",
        "gaia.ir": "gaia_ir_artifact_id",
        "knowledge.index": "knowledge_index_artifact_id",
    }
    sha_keys = {
        "formalization": "formalization",
        "gaia.ir": "gaia.ir",
        "knowledge.index": "knowledge.index",
    }
    source_sha256s: dict[str, str] = {}
    for kind, filename in KIND_TO_OUTPUT.items():
        artifact = pick(artifacts, kind)
        blob = run_dir / "artifacts" / "blobs" / f"{artifact['artifact_id']}.json"
        if not blob.is_file():
            raise SystemExit(f"missing blob for {artifact['artifact_id']}: {blob}")
        target = dest / filename
        shutil.copyfile(blob, target)
        digest = sha256_file(target)
        provenance[id_keys[kind]] = artifact["artifact_id"]
        source_sha256s[sha_keys[kind]] = digest
        print(f"{filename:26s} <- {artifact['artifact_id']}  sha256={digest[:16]}…")
    provenance["source_sha256s"] = source_sha256s

    (dest / "correct_run_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {dest}/correct_run_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
