"""Small shared utilities: paths, env, subprocess logging, run directories."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[2]
PRODUCT = REPO / "ariadne"
DEFAULT_CONFIG = PRODUCT / "configs" / "aircraft.demo.json"
VENV_PY = REPO / ".venv-gaia-a7" / "bin" / "python"
HARNESS_PATH = ":".join(str(p) for p in (
    REPO, REPO / "agent", REPO / "pipelines" / "single_paper", REPO / "pipelines" / "merge",
))


class AriadneError(RuntimeError):
    """A user-facing failure with a concrete next step."""


def python_bin() -> str:
    explicit = os.environ.get("ARIADNE_PYTHON")
    if explicit:
        return explicit
    if VENV_PY.is_file():
        return str(VENV_PY)
    return sys.executable


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_env(path: Path | None = None) -> None:
    """Load repo-local keys without overwriting an explicit environment."""
    path = path or (REPO / "agent" / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value:
            os.environ.setdefault(key, value)


def slugify(value: str, limit: int = 42) -> str:
    text = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", value.strip()).strip("-")
    return (text[:limit] or "question").strip("-")


def new_run_dir(question: str, out: Path | None = None) -> Path:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        return out
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = REPO / "outputs" / "ariadne_runs" / f"run_{stamp}_{slugify(question)}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def harness_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = HARNESS_PATH + ((":" + existing) if existing else "")
    if extra:
        env.update(extra)
    return env


def run(
    cmd: Iterable[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    log: Path | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    argv = [str(item) for item in cmd]
    proc = subprocess.run(
        argv, cwd=str(cwd) if cwd else None, env=dict(env) if env else None,
        capture_output=True, text=True, timeout=timeout,
    )
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(
            "$ " + " ".join(argv) + "\n\n[stdout]\n" + proc.stdout +
            "\n[stderr]\n" + proc.stderr,
            encoding="utf-8",
        )
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1200:]
        raise AriadneError(f"command failed ({proc.returncode}): {' '.join(argv)}\n{tail}")
    return proc


def which(name: str) -> str | None:
    return shutil.which(name)


def stage_log(run_dir: Path, name: str) -> Path:
    return run_dir / "logs" / f"{name}.log"


def fail(message: str) -> None:
    raise AriadneError(message)
