"""Mechanical fairness audit for the v2 three-condition pilot.

The audit deliberately treats the final ``merge.html`` as read-only.  It
checks the parts of the comparison that can be established before a model is
called: common target-paper input, matched retrieval budgets, equal external
corpus membership, focal-paper exclusion, graph snapshot identity, and
absence of evaluator-only fields.  It also reports (without silently fixing)
answer terms that are already present in the required full target paper.

This is a protocol audit, not a claim that the model's scientific choice is
correct.  A missing graph provenance mapping is reported as a warning and
keeps graph evidence credit disabled; it is never filled in from a display
label or an inferred paper id.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .final_graph import FinalGraph, audit_graph
from .v2_protocol import V2_CONDITIONS, condition_input
from .v2_retrieval import external_index


# These terms are intentionally only a diagnostic for the current LTH bridge
# task.  The full target paper is required by v2, so their presence is not a
# protocol error; it means the task cannot by itself establish a graph gain.
DEFAULT_LTH_LEAKAGE_TERMS = (
    "rewinding",
    "initialization",
    "warmup",
    "learning rate schedule",
)

_FORBIDDEN_VISIBLE_KEYS = {
    "allowed_actions",
    "candidate_experiments",
    "evaluation_oracle",
    "bridge_constraints",
    "gold_utility",
    "oracle",
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _walk_keys(value: Any, prefix: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key)
            yield from _walk_keys(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_keys(item, f"{prefix}[{index}]")


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})


def audit_v2_fairness(
    task: dict[str, Any],
    *,
    graph: FinalGraph | None = None,
    repo_root: Path | None = None,
    likely_answer_terms: Iterable[str] = DEFAULT_LTH_LEAKAGE_TERMS,
) -> dict[str, Any]:
    """Return a deterministic fairness report for one v2 task fixture.

    ``ok`` means all hard protocol checks pass.  ``warnings`` deliberately
    include two scientifically material limitations: the graph has no audited
    source anchors, and the required focal paper already contains likely
    answer terms.  Callers should persist the report with the plan rather than
    treating warnings as successful evidence.
    """

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(task, dict):
        return {"ok": False, "errors": ["task must be an object"], "warnings": [], "checks": [], "metrics": {}}

    try:
        visible = {condition: condition_input(task, condition) for condition in V2_CONDITIONS}
    except Exception as exc:  # preserve a machine-readable audit failure
        return {"ok": False, "errors": [f"condition_input failed: {exc}"], "warnings": [], "checks": [], "metrics": {}}

    # All conditions must receive the exact same focal paper and task facts.
    target = task.get("target_paper") or task.get("paper") or {}
    paper_text = target.get("text") if isinstance(target, dict) else None
    paper_hash = _sha256_text(paper_text) if isinstance(paper_text, str) else None
    metadata_hash = target.get("text_sha256") if isinstance(target, dict) else None
    _check(checks, "target_paper_nonempty", bool(paper_text and paper_text.strip()), "full target paper text is present")
    hash_ok = bool(paper_hash and (metadata_hash is None or metadata_hash == paper_hash))
    _check(checks, "target_paper_hash", hash_ok, f"sha256={paper_hash}; declared={metadata_hash}")
    visible_hashes = {
        condition: _sha256_text(value["target_paper"].get("text", ""))
        for condition, value in visible.items()
    }
    same_visible_paper = len(set(visible_hashes.values())) == 1 and next(iter(visible_hashes.values()), None) == paper_hash
    _check(checks, "same_full_target_paper", same_visible_paper, _canonical(visible_hashes))
    common_fields = ("task_id", "question", "hypothesis", "context", "execution", "allowed_dataset_refs", "allowed_entrypoint_refs", "evidence_cutoff", "knowledge_access", "web_tools_enabled")
    common_ok = True
    for field in common_fields:
        values = [_canonical(value.get(field)) for value in visible.values()]
        field_ok = len(set(values)) == 1
        common_ok = common_ok and field_ok
        _check(checks, f"same_{field}", field_ok, "all three conditions match" if field_ok else "condition values differ")
    if not common_ok:
        errors.append("task facts differ across conditions")

    # Retrieval budget is a shared total budget, not a per-tool budget.
    interfaces = {condition: value.get("retrieval_interface", {}) for condition, value in visible.items()}
    budget_pairs = {
        condition: (interface.get("max_calls"), interface.get("max_return_tokens"))
        for condition, interface in interfaces.items()
    }
    budget_ok = len(set(budget_pairs.values())) == 1
    _check(checks, "matched_retrieval_budget", budget_ok, _canonical(budget_pairs))
    if not budget_ok:
        errors.append("retrieval call/token budgets differ across conditions")
    expected_tools = {
        "paper_only": {"search_paper"},
        "external_text": {"search_paper", "search_external_text"},
        "reasoning_graph": {"search_paper", "search_external_text", "search_claims", "expand_neighbors", "get_source_anchor"},
    }
    for condition, expected in expected_tools.items():
        actual = set(interfaces[condition].get("tools", []))
        _check(checks, f"tools_{condition}", actual == expected, f"actual={sorted(actual)} expected={sorted(expected)}")
        if actual != expected:
            errors.append(f"tool set mismatch in {condition}")
    resource_isolation = {
        "paper_only_no_external_resource": "external_corpus" not in interfaces["paper_only"] and "graph_resource" not in interfaces["paper_only"],
        "external_text_no_graph_resource": "graph_resource" not in interfaces["external_text"] and "external_corpus" in interfaces["external_text"],
        "reasoning_graph_has_graph_resource": "graph_resource" in interfaces["reasoning_graph"] and "external_corpus" in interfaces["reasoning_graph"],
    }
    for name, ok in resource_isolation.items():
        _check(checks, name, ok, "condition-specific resources are isolated" if ok else "condition received an unexpected resource")
        if not ok:
            errors.append(name)

    # Corpus membership must be equal for flat text and graph retrieval, and
    # must exclude the focal paper to avoid giving graph/text conditions a
    # second copy of the target paper.
    sources = task.get("external_sources") or []
    source_ids = [item.get("paper_id") or item.get("source_id") for item in sources if isinstance(item, dict)]
    source_set = {item for item in source_ids if isinstance(item, str)}
    unique_sources = len(source_set) == len(source_ids) and bool(source_set)
    focal_id = target.get("paper_id") if isinstance(target, dict) else None
    focal_excluded = not focal_id or focal_id not in source_set
    corpus_meta = task.get("external_corpus") if isinstance(task.get("external_corpus"), dict) else {}
    count_ok = corpus_meta.get("paper_count") == len(source_set)
    _check(checks, "external_sources_unique", unique_sources, f"declared={len(source_ids)} unique={len(source_set)}")
    _check(checks, "external_corpus_count", count_ok, f"metadata={corpus_meta.get('paper_count')} declared={len(source_set)}")
    _check(checks, "focal_paper_excluded", focal_excluded, f"focal={focal_id} in_external={focal_id in source_set if focal_id else None}")
    if not unique_sources or not count_ok or not focal_excluded:
        errors.append("external corpus membership is not a unique focal-excluded set")
    external_visible = {
        condition: interfaces[condition].get("external_corpus")
        for condition in ("external_text", "reasoning_graph")
    }
    _check(checks, "same_external_corpus_metadata", external_visible["external_text"] == external_visible["reasoning_graph"], "external_text and reasoning_graph advertise the same frozen corpus")
    if external_visible["external_text"] != external_visible["reasoning_graph"]:
        errors.append("external corpus metadata differs between text and graph conditions")
    corpus_paper_ids: set[str] = set()
    if repo_root is not None:
        try:
            index = external_index(task, repo_root=repo_root)
            corpus_paper_ids = {record.source_paper_id for record in index.records}
            index_ok = corpus_paper_ids == source_set and focal_id not in corpus_paper_ids
            _check(checks, "external_index_matches_manifest", index_ok, f"indexed={len(corpus_paper_ids)} declared={len(source_set)}")
            if not index_ok:
                errors.append("external text index does not match declared corpus")
        except Exception as exc:
            errors.append(f"external index audit failed: {exc}")

    # Graph resource identity and counts are compared with merge.html when the
    # caller supplies the already-loaded final graph.  Never infer provenance
    # from node labels.
    graph_input = interfaces["reasoning_graph"].get("graph_resource")
    graph_meta = task.get("graph") if isinstance(task.get("graph"), dict) else {}
    graph_identity_ok = isinstance(graph_input, dict) and graph_input.get("snapshot_file") == graph_meta.get("snapshot_file") and graph_input.get("node_count") == graph_meta.get("node_count") and graph_input.get("edge_count") == graph_meta.get("edge_count")
    if graph is not None:
        graph_audit = audit_graph(graph.payload)
        graph_identity_ok = graph_identity_ok and graph.embedded_sha256 == graph_meta.get("snapshot_sha256") and graph_audit["node_count"] == graph_meta.get("node_count") and graph_audit["edge_count"] == graph_meta.get("edge_count")
    _check(checks, "graph_snapshot_identity", graph_identity_ok, f"declared={graph_meta.get('snapshot_sha256')} actual={graph.embedded_sha256 if graph else None}")
    if not graph_identity_ok:
        errors.append("reasoning_graph resource does not match merge.html snapshot")
    graph_provenance_status = graph_meta.get("provenance_status", "unknown")
    graph_anchor_count = graph_audit.get("nodes_with_source_anchor_ids", 0) if graph is not None else None
    if graph_provenance_status == "missing" or graph_anchor_count == 0:
        warnings.append("graph provenance is missing; graph nodes/edges must receive zero evidence credit")
    elif graph_provenance_status not in {"available", "partial"}:
        warnings.append(f"graph provenance status is {graph_provenance_status!r}")

    # Model-facing values must not contain evaluator-only closure or candidate
    # IDs.  This check recursively catches a leak nested in context or a tool
    # descriptor, while allowing the words to occur in the prompt instructions.
    leak_paths: list[str] = []
    for condition, value in visible.items():
        for path, key in _walk_keys(value):
            if key in _FORBIDDEN_VISIBLE_KEYS:
                leak_paths.append(f"{condition}:{path}")
    _check(checks, "no_evaluator_fields", not leak_paths, "; ".join(leak_paths) if leak_paths else "no hidden evaluator keys")
    if leak_paths:
        errors.append("evaluator-only fields leaked into model input")
    serialized = {condition: _canonical(value) for condition, value in visible.items()}
    forbidden_literals = ("p53_claim_", "K02", "K03")
    literal_hits = [f"{condition}:{literal}" for condition, text in serialized.items() for literal in forbidden_literals if literal in text]
    _check(checks, "no_known_answer_ids", not literal_hits, "; ".join(literal_hits) if literal_hits else "no known answer IDs")
    if literal_hits:
        errors.append("known answer identifiers leaked into model input")

    leakage_terms = [term.casefold() for term in likely_answer_terms if isinstance(term, str) and term.strip()]
    leakage_counts = {
        term: paper_text.casefold().count(term) for term in leakage_terms
    } if isinstance(paper_text, str) else {}
    present = {term: count for term, count in leakage_counts.items() if count > 0}
    bridge_suitability = "warning" if present else "candidate"
    if present:
        warnings.append("full target paper contains likely answer terms: " + ", ".join(f"{term}={count}" for term, count in present.items()))
    _check(checks, "answer_term_diagnostic", not bool(present), "no configured likely answer terms in focal paper" if not present else _canonical(present))

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "metrics": {
            "target_paper_sha256": paper_hash,
            "external_paper_count": len(source_set),
            "indexed_external_paper_count": len(corpus_paper_ids) if repo_root is not None else None,
            "focal_paper_id": focal_id,
            "retrieval_budget": budget_pairs,
            "graph_provenance_status": graph_provenance_status,
            "graph_anchor_count": graph_anchor_count,
            "likely_answer_terms_present": present,
            "bridge_suitability": bridge_suitability,
        },
    }
