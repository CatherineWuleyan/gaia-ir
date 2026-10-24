#!/usr/bin/env python3
"""Deterministic evaluator for the E0 claim-graph pilot.

The checker intentionally treats conditions, relation types, and provenance as
part of the answer. It is a guardrail against a superficially correct answer
whose evidence path is wrong or whose applicability condition was dropped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip())
    if isinstance(value, list):
        return [_norm(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _norm(v) for k, v in value.items()}
    return value


def _set_of(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(v) for v in values}


def _path(values: Any) -> list[tuple[str, str]]:
    if not isinstance(values, list):
        return []
    result: list[tuple[str, str]] = []
    for item in values:
        if isinstance(item, dict):
            result.append((str(item.get("source_id", "")), str(item.get("relation", ""))))
    return result


def evaluate(question: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    gold = question.get("gold") or {}
    diagnostics: dict[str, Any] = {}

    gold_answer = _norm(gold.get("answer"))
    got_answer = _norm(response.get("answer"))
    diagnostics["answer_exact"] = gold_answer == got_answer
    diagnostics["status_exact"] = str(gold.get("status", "")) == str(response.get("status", ""))

    required = _set_of(gold.get("required_premises"))
    got_premises = _set_of(response.get("premises"))
    diagnostics["premise_recall"] = (len(required & got_premises) / len(required)) if required else 1.0
    diagnostics["premise_set_exact"] = got_premises == required

    gold_path = _path(gold.get("evidence_path"))
    got_path = _path(response.get("evidence_path"))
    diagnostics["evidence_path_exact"] = got_path == gold_path
    diagnostics["evidence_path_precision"] = (
        len(set(got_path) & set(gold_path)) / len(set(got_path)) if got_path else (1.0 if not gold_path else 0.0)
    )

    gold_conditions = _norm(gold.get("conditions") or [])
    got_conditions = _norm(response.get("conditions") or [])
    diagnostics["conditions_exact"] = got_conditions == gold_conditions

    allowed_provenance = _set_of(gold.get("provenance"))
    got_provenance = _set_of(response.get("provenance"))
    diagnostics["provenance_closed"] = got_provenance.issubset(allowed_provenance)
    diagnostics["provenance_complete"] = set(x[0] for x in gold_path).issubset(got_provenance)

    # Extra evidence/premises are treated as unsupported inference. This keeps
    # the pilot conservative and makes the failure visible for later analysis.
    diagnostics["unsupported_inference"] = bool(
        (got_premises - required) or (set(got_path) - set(gold_path))
    )
    if str(gold.get("status")) == "unknown":
        diagnostics["unknown_is_abstained"] = got_answer is None and not got_premises and not got_path
    else:
        diagnostics["unknown_is_abstained"] = True

    hard_checks = [
        diagnostics["answer_exact"],
        diagnostics["status_exact"],
        diagnostics["premise_set_exact"],
        diagnostics["evidence_path_exact"],
        diagnostics["conditions_exact"],
        diagnostics["provenance_closed"],
        diagnostics["provenance_complete"],
        not diagnostics["unsupported_inference"],
        diagnostics["unknown_is_abstained"],
    ]
    return {
        "question_id": question.get("question_id"),
        "strict_success": all(hard_checks),
        "diagnostics": diagnostics,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    args = parser.parse_args()

    questions = {q["question_id"]: q for q in load_jsonl(args.questions)}
    responses = load_jsonl(args.responses)
    results = []
    for response in responses:
        qid = response.get("question_id")
        if qid not in questions:
            results.append({"question_id": qid, "strict_success": False, "error": "unknown_question_id"})
            continue
        results.append(evaluate(questions[qid], response))
    print(json.dumps({"results": results, "strict_success_rate": sum(r.get("strict_success", False) for r in results) / len(results) if results else 0.0}, ensure_ascii=False, indent=2))
    return 0 if all(r.get("strict_success", False) for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
