"""Create a no-network plan for the v2 LTH pilot.

The plan is intentionally separate from the legacy evidence-mode runner.  It
materialises the exact model inputs for ``paper_only``, ``external_text`` and
``reasoning_graph`` without making a provider request.  The provider-agnostic
tool-loop preflight lives in ``live_tool_loop.py``; this module remains the
no-network plan and authorization gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .final_graph import load_final_graph
from .v2_fairness import audit_v2_fairness
from .v2_protocol import V2_CONDITIONS, build_prompt, condition_input


DEFAULT_PROFILES = ("deepseek_v4_1", "trisol_qwen7b", "trisol_qwen35b")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    return _sha_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def create_plan(
    *,
    fixture: Path,
    graph_html: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = (0, 1, 2),
    model_profiles: tuple[str, ...] = DEFAULT_PROFILES,
) -> dict[str, Any]:
    task = json.loads(fixture.read_text(encoding="utf-8"))
    graph = load_final_graph(graph_html)
    fairness_audit = audit_v2_fairness(task, graph=graph, repo_root=graph_html.parent)
    if not fairness_audit["ok"]:
        raise ValueError("v2 fairness audit failed: " + "; ".join(fairness_audit["errors"]))
    declared_hash = ((task.get("graph") or {}).get("snapshot_sha256"))
    if declared_hash and declared_hash != graph.embedded_sha256:
        raise ValueError("fixture graph snapshot hash does not match merge.html")
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    if not model_profiles or any(not isinstance(item, str) or not item for item in model_profiles):
        raise ValueError("model_profiles must be non-empty strings")
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    cells: list[dict[str, Any]] = []
    for profile in model_profiles:
        for condition in V2_CONDITIONS:
            visible = condition_input(task, condition)
            prompt = build_prompt(task, condition, visible_decision_trace=True)
            for seed in seeds:
                cell_id = f"{task['task_id']}/{profile}/{condition}/seed-{seed}"
                cell_dir = prompts_dir / profile / condition / f"seed-{seed}"
                cell_dir.mkdir(parents=True, exist_ok=True)
                input_path = cell_dir / "model_input.json"
                prompt_path = cell_dir / "prompt.txt"
                input_path.write_text(json.dumps(visible, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                prompt_path.write_text(prompt, encoding="utf-8")
                cells.append({
                    "cell_id": cell_id,
                    "task_id": task["task_id"],
                    "model_profile": profile,
                    "condition": condition,
                    "seed": seed,
                    "model_input": str(input_path.relative_to(output_dir)),
                    "prompt": str(prompt_path.relative_to(output_dir)),
                    "model_input_sha256": _sha_bytes(input_path.read_bytes()),
                    "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
                    "request_budget": 1,
                    "retrieval_calls": 0,
                    "status": "planned",
                })
    manifest = {
        "schema_version": "v2-pilot-plan-v1",
        "status": "planned_authorization_required",
        "api_requests_made": 0,
        "task_id": task["task_id"],
        "fixture": str(fixture),
        "fixture_sha256": _sha_bytes(fixture.read_bytes()),
        "graph_html": str(graph_html),
        "graph_html_sha256": graph.html_sha256,
        "graph_snapshot_sha256": graph.embedded_sha256,
        "graph_canonical_snapshot_sha256": graph.canonical_sha256,
        "graph_provenance_status": ((task.get("graph") or {}).get("provenance_status") or "unknown"),
        "fairness_audit": fairness_audit,
        "conditions": list(V2_CONDITIONS),
        "seeds": list(seeds),
        "model_profiles": list(model_profiles),
        "target_cells": len(cells),
        "request_budget_total": len(cells),
        "request_budget_per_invocation": 1,
        "tool_call_loop": "implemented_interface_only; plan-only",
        "evaluation_policy": {
            "no_allowed_actions_in_model_input": True,
            "no_candidate_experiments_in_model_input": True,
            "graph_anchor_credit": "disabled until provenance sidecar is explicitly enabled",
            "knowledge_access": "closed_book",
            "web_tools_enabled": False,
        },
        "cells": cells,
        "task_sha256": _sha_json(task),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "cells.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in cells), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--graph-html", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = create_plan(fixture=args.fixture, graph_html=args.graph_html, output_dir=args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "target_cells", "graph_snapshot_sha256", "graph_provenance_status")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
