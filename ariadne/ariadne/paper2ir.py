"""Stage 2: paper graph -> claims_final via the repo's paper2ir pipeline.

The 14 steps are not reimplemented.  A run-local copy of
``pipelines/clean_claims`` is created (scripts only, no old data), its own
``run_full_pipeline.STEPS`` list is imported, and ``run_step`` is called exactly
as the maintained runner does.  This keeps the stage isolated and resumable.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import Config, Runtime, resolve_runtime
from .util import AriadneError, REPO, python_bin, run, stage_log, write_json

EMPTY_INSTANCE_OUTPUTS = frozenset({
    "instance_target_positions.json",
    "claim_instance_associations.json",
})
IGNORE = shutil.ignore_patterns(
    "data", "__pycache__", "*.pyc", ".DS_Store", "*.log", "tests", "test_*.py",
)


def _copy_pipeline(dest: Path) -> Path:
    source = REPO / "pipelines" / "clean_claims"
    if not source.is_dir():
        raise AriadneError(f"paper2ir pipeline not found: {source}")
    dest = dest.resolve()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest, ignore=IGNORE)
    return dest


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location("ariadne_paper2ir_core", path)
    if spec is None or spec.loader is None:
        raise AriadneError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_one(paper: Mapping[str, Any], stage_dir: Path, pipeline_dir: Path,
            module: Any, run_root: Path, python: str) -> dict[str, Any]:
    pid = str(paper["lkm_id"])
    stage_dir = stage_dir.resolve()
    pipeline_dir = pipeline_dir.resolve()
    run_root = run_root.resolve()
    paper_dir = stage_dir / "data" / pid
    paper_dir.mkdir(parents=True, exist_ok=True)
    for name in ("graph.json", "ocr.md"):
        source = Path(str(paper.get("graph_path") if name == "graph.json" else paper.get("ocr_path")))
        if not source.is_absolute():
            source = run_root / source
        if not source.is_file():
            raise AriadneError(f"{pid}: missing {name} at {source}")
        shutil.copyfile(source, paper_dir / name)

    log = stage_log(stage_dir, f"paper2ir_{pid}")
    print(f"[paper2ir] {paper.get('title', pid)[:70]}", flush=True)
    for index, (script_rel, expected_output, _needs_api) in enumerate(module.STEPS, start=1):
        script = (pipeline_dir / script_rel).resolve()
        expected = paper_dir / expected_output
        # Keep the child's output in the product log, not in the terminal.
        proc = run(
            [python, str(script), pid],
            cwd=pipeline_dir, env=dict(os.environ, PYTHONPATH=str(pipeline_dir)),
            log=log, timeout=7200, check=False,
        )
        if proc.returncode == 0 and expected.is_file():
            print(f"  [{index:02d}/{len(module.STEPS)}] {script_rel} ok", flush=True)
            continue
        if expected_output in EMPTY_INSTANCE_OUTPUTS and proc.returncode == 0:
            expected.write_text("[]\n", encoding="utf-8")
            print(f"  [{index:02d}/{len(module.STEPS)}] {script_rel} empty -> []", flush=True)
            continue
        tail = (proc.stderr or proc.stdout or "").strip()[-800:]
        raise AriadneError(f"{pid}: paper2ir stopped at {script_rel} (see {log})\n{tail}")
    claims = paper_dir / "claims_final.json"
    if not claims.is_file():
        raise AriadneError(f"{pid}: paper2ir finished without claims_final.json")
    return {
        "lkm_id": pid,
        "title": paper.get("title"),
        "claims_final": str(claims.relative_to(stage_dir)),
        "graph": str((paper_dir / "graph.json").relative_to(stage_dir)),
        "text": str((paper_dir / "ocr.md").relative_to(stage_dir)),
    }


def run_all(papers: Iterable[Mapping[str, Any]], stage_dir: Path, run_root: Path,
            runtime: Runtime | None = None) -> list[dict[str, Any]]:
    if runtime is None:
        runtime = resolve_runtime(Config.load())
    runtime.apply_env()
    stage_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = _copy_pipeline(stage_dir / "pipeline")
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise AriadneError(
            "No OpenAI-compatible model key is configured. Run "
            "`ariadne config init`, set DEEPSEEK_API_KEY, or use `ariadne demo`."
        )
    os.environ.setdefault("DEEPSEEK_MODEL", "deepseek-v4-flash")
    python = runtime.python or python_bin()
    # The 14 step scripts each derive ``PROJECT_ROOT/data`` from their own
    # location.  Keep the run's data in the stage dir but expose it at the
    # pipeline copy's expected ``data/`` path via a symlink, otherwise the
    # imported STEPS list and the step scripts disagree about where to read.
    data_dir = stage_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    pipeline_data = pipeline_dir / "data"
    if not pipeline_data.exists():
        pipeline_data.symlink_to(data_dir, target_is_directory=True)
    module = _load_module(pipeline_dir / "run_full_pipeline.py")
    module.PROJECT_ROOT = pipeline_dir
    module.DATA_DIR = pipeline_data
    results = []
    for paper in papers:
        results.append(run_one(paper, stage_dir, pipeline_dir, module, run_root, python))
    write_json(stage_dir / "papers.json", {"papers": results})
    return results
