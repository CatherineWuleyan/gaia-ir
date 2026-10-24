from pathlib import Path
import json
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_oracle_pilot_passes():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmark/e0/strict_check.py"),
         "--questions", str(ROOT / "benchmark/e0/questions.jsonl"),
         "--responses", str(ROOT / "benchmark/e0/sample_responses.jsonl")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads(proc.stdout)
    assert out["strict_success_rate"] == 1.0


def test_wrong_relation_is_rejected(tmp_path):
    bad = tmp_path / "bad.jsonl"
    row = json.loads((ROOT / "benchmark/e0/sample_responses.jsonl").read_text().splitlines()[0])
    row["evidence_path"][0]["relation"] = "infer"
    bad.write_text(json.dumps(row) + "\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "benchmark/e0/strict_check.py"),
         "--questions", str(ROOT / "benchmark/e0/questions.jsonl"),
         "--responses", str(bad)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert json.loads(proc.stdout)["results"][0]["diagnostics"]["evidence_path_exact"] is False
