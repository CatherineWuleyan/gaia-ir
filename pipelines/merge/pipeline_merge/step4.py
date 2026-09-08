"""Step 4: merge equivalent nodes and formalize frozen Step 3 weakpoints."""
from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from typing import Any, Mapping

from pipeline_harness.models import Finding, JSONDict
from pipeline_harness.plugins import ArtifactDraft, StageContext, StageResult
from pipeline_harness.store import atomic_write_json

from pipelines.single_paper.agent_pipeline_v2.authoring import (
    canonical_strategy,
    content_hash,
    emit_formalization,
)

from .step0 import validate_step0_options
from .step1 import Step1InputError, _read_json, _step0_options, validate_step1_inputs
from .step3 import PROPOSAL_SCHEMA, SCHEMA_VERSION, _category, _post_json


STEP_NAME = "step4_formalize_integration"
_BRACKET_ID = re.compile(r"\[([^\]]+)\]")


def _load_records(context: StageContext) -> tuple[dict[str, JSONDict], dict[str, JSONDict]]:
    records: dict[str, JSONDict] = {}
    anchors: dict[str, JSONDict] = {}
    for ref in context.find_all("gaia.ir"):
        payload = _read_json(context, ref)
        package = f"{payload.get('namespace')}:{payload.get('package_name')}"
        for index, item in enumerate(payload.get("knowledges") or []):
            if not isinstance(item, Mapping):
                continue
            qid, content = item.get("id"), item.get("content")
            if not isinstance(qid, str) or not qid or not isinstance(content, str) or not content.strip():
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
            if metadata.get("helper_visibility") == "formal_internal":
                continue
            source_anchor_ids = [str(value) for value in item.get("source_anchor_ids") or [] if isinstance(value, str)]
            anchor_ids = [f"anchor_ir_{ref.artifact_id}_{index}_{position}"
                          for position, _ in enumerate(source_anchor_ids)]
            if not anchor_ids:
                anchor_ids = [f"anchor_ir_{ref.artifact_id}_{index}"]
            for position, anchor_id in enumerate(anchor_ids):
                anchor = {
                    "anchor_id": anchor_id,
                    "artifact_id": ref.artifact_id,
                    "source_kind": "gaia.ir",
                    "locator": {"type": "json_pointer", "pointer": (
                        f"/knowledges/{index}/source_anchor_ids/{position}" if source_anchor_ids
                        else f"/knowledges/{index}"
                    )},
                    "relevance": "imported_knowledge",
                }
                if source_anchor_ids:
                    anchor["source_anchor_id"] = source_anchor_ids[position]
                anchors[anchor_id] = anchor
            records[qid] = {
                "qid": qid,
                "package": package,
                "content": content,
                "type": item.get("type", "claim"),
                "metadata": dict(metadata),
                "category": _category(item),
                "source_anchor_ids": anchor_ids,
            }
    return records, anchors


def _load_source_graph(context: StageContext, records: Mapping[str, JSONDict]) -> tuple[list[str], list[JSONDict], list[JSONDict]]:
    """Keep each paper's frozen graph; integration relations are added on top."""
    nodes: list[str] = []
    operators: list[JSONDict] = []
    strategies: list[JSONDict] = []
    for ref in context.find_all("formalization"):
        payload = _read_json(context, ref)
        knowledge_map = payload.get("knowledges") if isinstance(payload.get("knowledges"), Mapping) else {}
        for item in payload.get("graph", {}).get("nodes", []):
            if isinstance(item, str) and item in knowledge_map:
                nodes.append(item)
                knowledge = knowledge_map[item]
                content = knowledge.get("content", {}) if isinstance(knowledge, Mapping) else {}
                canonical = content.get("canonical") if isinstance(content, Mapping) else None
                if item not in records and isinstance(canonical, str) and canonical.strip():
                    records[item] = {"qid": item, "package": "papers:source", "content": canonical,
                                      "type": knowledge.get("type", "claim"), "metadata": {},
                                      "category": None, "source_anchor_ids": []}
        for item in payload.get("graph", {}).get("operators", []) or []:
            if not isinstance(item, Mapping):
                continue
            variables = [str(value) for value in item.get("variables") or []]
            conclusion = item.get("conclusion")
            endpoints = [*variables, str(conclusion)] if conclusion is not None else variables
            if endpoints and all(endpoint in records for endpoint in endpoints):
                operators.append({"id": str(item.get("operator_id") or item.get("id") or f"source_operator_{len(operators)}"),
                                  "type": str(item.get("operator") or item.get("type") or "infer"),
                                  "variables": variables, "conclusion": str(conclusion) if conclusion is not None else None,
                                  "metadata": {"source": "paper_internal", **dict(item.get("metadata") or {})}})
        for item in payload.get("graph", {}).get("strategies", []) or []:
            if not isinstance(item, Mapping):
                continue
            premises = [str(value) for value in item.get("premises") or []]
            conclusion = item.get("conclusion")
            if isinstance(conclusion, str) and premises and all(key in records for key in [*premises, conclusion]):
                strategies.append({"scope": "local", "type": str(item.get("type") or "infer"),
                                   "premises": premises, "conclusion": conclusion,
                                   "background": [str(value) for value in item.get("background") or [] if str(value) in records]})
    return list(dict.fromkeys(nodes)), operators, strategies


def _equivalence_aliases(
    operators: list[JSONDict], records: Mapping[str, JSONDict],
) -> tuple[dict[str, str], list[JSONDict]]:
    """Merge only LLM-approved equivalences whose O/E/non-OE categories match."""
    parent = {qid: qid for qid in records}

    def find(qid: str) -> str:
        while parent[qid] != qid:
            parent[qid] = parent[parent[qid]]
            qid = parent[qid]
        return qid

    for operator in operators:
        variables = operator.get("variables")
        if operator.get("type") != "equivalence" or not isinstance(variables, list) or len(variables) != 2:
            continue
        left, right = map(str, variables)
        if left not in records or right not in records:
            continue
        category = records[left].get("category")
        if category is None or category != records[right].get("category"):
            continue
        first, second = find(left), find(right)
        if first != second:
            representative, alias = min(first, second), max(first, second)
            parent[alias] = representative

    aliases = {qid: find(qid) for qid in records}
    classes: dict[str, list[str]] = defaultdict(list)
    for qid, representative in aliases.items():
        classes[representative].append(qid)
    merges = [{
        "representative_qid": representative,
        "merged_qids": sorted(members),
        "category": records[representative].get("category"),
        "source_anchor_ids": sorted({anchor for qid in members for anchor in records[qid]["source_anchor_ids"]}),
        "package_provenance": sorted({records[qid]["package"] for qid in members}),
    } for representative, members in sorted(classes.items()) if len(members) > 1]
    return aliases, merges


def _rewrite_operator(operator: Mapping[str, Any], aliases: Mapping[str, str]) -> JSONDict | None:
    variables = [aliases.get(str(qid), str(qid)) for qid in operator.get("variables") or []]
    variables = list(dict.fromkeys(variables))
    conclusion = operator.get("conclusion")
    conclusion = aliases.get(str(conclusion), str(conclusion)) if conclusion is not None else None
    if conclusion is not None and conclusion in variables:
        return None
    if operator.get("type") in {"equivalence", "contradiction"} and len(variables) != 2:
        return None
    if operator.get("type") == "negation" and len(variables) != 1:
        return None
    if operator.get("type") in {"conjunction", "disjunction"} and len(variables) < 2:
        return None
    result = copy.deepcopy(dict(operator))
    result["variables"] = variables
    result["conclusion"] = conclusion
    return result


def _rewrite_weakpoint(weakpoint: Mapping[str, Any], aliases: Mapping[str, str]) -> JSONDict | None:
    result = copy.deepcopy(dict(weakpoint))
    payload = result.get("payload")
    if not isinstance(payload, dict):
        return None
    evidence = list(dict.fromkeys(aliases.get(str(qid), str(qid)) for qid in payload.get("evidence_claim_ids") or []))
    targets = list(dict.fromkeys(aliases.get(str(qid), str(qid)) for qid in payload.get("target_claim_id") or []))
    if not evidence or not targets or set(evidence) & set(targets):
        return None
    payload["evidence_claim_ids"], payload["target_claim_id"] = evidence, targets
    expression = str(payload.get("expression") or "")
    payload["expression"] = _BRACKET_ID.sub(
        lambda match: f"[{aliases.get(match.group(1), match.group(1))}]", expression,
    )
    return result


def _candidate_prompt(candidate: Mapping[str, Any], weakpoints: list[JSONDict], records: Mapping[str, JSONDict]) -> str:
    evidence_ids = sorted({qid for weakpoint in weakpoints for qid in weakpoint["payload"]["evidence_claim_ids"]})
    evidence = [{"qid": qid, "content": records[qid]["content"]} for qid in evidence_ids if qid in records]
    modes = sorted({str(weakpoint["payload"]["reasoning_type"]) for weakpoint in weakpoints})
    return ("Materialize exactly one integration-owned candidate claim using only the supplied frozen evidence. "
            "For deduction, write only a bounded compression fully covered by every premise and do not broaden scope. "
            "For abduction, write a cautious common hypothesis that can explain or predict the supplied instances, never a proven fact. "
            "Do not add facts, conditions, mechanisms, metrics, or assumptions absent from the evidence. If impossible, return null. "
            "Return JSON only: {\"candidate_id\":\"...\",\"canonical_content\":\"...|null\"}.\n"
            f"CANDIDATE={json.dumps(candidate, ensure_ascii=False, sort_keys=True)}\n"
            f"REASONING_TYPES={json.dumps(modes)}\nEVIDENCE={json.dumps(evidence, ensure_ascii=False, sort_keys=True)}")


def _materialize_candidates(
    candidates: list[JSONDict], weakpoints: list[JSONDict], records: Mapping[str, JSONDict],
) -> dict[str, str]:
    by_target: dict[str, list[JSONDict]] = defaultdict(list)
    for weakpoint in weakpoints:
        for target in weakpoint["payload"]["target_claim_id"]:
            by_target[str(target)].append(weakpoint)
    result: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = candidate.get("id")
        if not isinstance(candidate_id, str) or candidate_id not in by_target:
            continue
        response = _post_json(_candidate_prompt(candidate, by_target[candidate_id], records))
        if not isinstance(response, dict) or response.get("candidate_id") != candidate_id:
            continue
        content = response.get("canonical_content")
        if isinstance(content, str) and content.strip():
            result[candidate_id] = content.strip()
    return result


def _abduction_premises(payload: Mapping[str, Any], records: Mapping[str, JSONDict]) -> list[str]:
    evidence = [str(qid) for qid in payload["evidence_claim_ids"]]
    if len(evidence) == 1:
        # The formalizer supplies the generic AltExp interface for an
        # observation-only abduction.  Do not invent a second premise here.
        return evidence
    expression = str(payload.get("expression") or "")
    mentioned = _BRACKET_ID.findall(expression)
    if "或" in expression or " or " in expression.lower():
        separator = "等价" if "等价" in expression else " equivalent "
        left, right = expression.split(separator, 1) if separator in expression else (expression, "")
        alternatives = list(dict.fromkeys(qid for qid in mentioned if qid in evidence and f"[{qid}]" in left))
        if len(alternatives) > 1:
            raise ValueError("abduction alternatives must be combined into one disjunction")
        if alternatives and right:
            observations = list(dict.fromkeys(qid for qid in mentioned if qid in evidence and f"[{qid}]" in right))
            if not observations:
                raise ValueError("abduction expression does not identify its observation")
            return [*observations, alternatives[0]]
    # Several observation premises are valid; only explicit alternatives are
    # restricted, and their disjunction must already be represented upstream.
    return evidence


def _background_from_expression(payload: Mapping[str, Any], records: Mapping[str, JSONDict]) -> list[str]:
    return list(dict.fromkeys(
        qid for qid in _BRACKET_ID.findall(str(payload.get("expression") or ""))
        if qid in records and records[qid].get("type") in {"note", "setting", "context"}
    ))


def _strategies_from_weakpoint(weakpoint: Mapping[str, Any], records: Mapping[str, JSONDict]) -> list[JSONDict]:
    payload = weakpoint["payload"]
    kind = payload["reasoning_type"]
    evidence = [str(qid) for qid in payload["evidence_claim_ids"]]
    targets = [str(qid) for qid in payload["target_claim_id"]]
    if kind == "abduction":
        premises = _abduction_premises(payload, records)
        return [{"scope": "local", "type": kind, "premises": premises,
                 "conclusion": target, "background": []} for target in targets]
    if kind == "analogy":
        if len(evidence) < 2:
            raise ValueError("analogy requires at least a source law and a bridge claim")
        background = _background_from_expression(payload, records)
        if not background:
            raise ValueError("analogy requires an explicit target-condition note")
        return [{"scope": "local", "type": kind, "premises": evidence,
                 "conclusion": target, "background": background} for target in targets]
    if kind not in {"deduction", "infer"}:
        raise ValueError(f"unsupported weakpoint reasoning type: {kind}")
    return [{"scope": "local", "type": kind, "premises": evidence,
             "conclusion": target, "background": []} for target in targets]


def _validate_acyclic(strategies: list[JSONDict]) -> None:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for strategy in strategies:
        for premise in strategy["premises"]:
            adjacency[str(premise)].add(str(strategy["conclusion"]))
    visiting: set[str] = set()
    visited: set[str] = set()
    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("formalized strategies contain a reasoning cycle")
        if node in visited:
            return
        visiting.add(node)
        for target in adjacency.get(node, set()):
            visit(target)
        visiting.remove(node)
        visited.add(node)
    for node in sorted(adjacency):
        visit(node)


def _build_document(
    context: StageContext, proposals: Mapping[str, Any], records: Mapping[str, JSONDict],
    anchors: Mapping[str, JSONDict], aliases: Mapping[str, str], merges: list[JSONDict],
    operators: list[JSONDict], weakpoints: list[JSONDict], candidate_contents: Mapping[str, str],
    source_nodes: list[str], source_operators: list[JSONDict], source_strategies: list[JSONDict],
) -> tuple[JSONDict, list[Finding]]:
    findings: list[Finding] = []
    active_weakpoints = [weakpoint for weakpoint in weakpoints
                         if all(not target.startswith("candidate_K_") or target in candidate_contents
                                for target in weakpoint["payload"]["target_claim_id"])]
    skipped = {weakpoint["id"] for weakpoint in weakpoints} - {weakpoint["id"] for weakpoint in active_weakpoints}
    for weakpoint_id in sorted(skipped):
        findings.append(Finding("STEP4_CANDIDATE_K_UNRESOLVED", "warning", f"Skipped {weakpoint_id}: candidate K was not materialized"))

    raw_strategies: list[JSONDict] = []
    for weakpoint in active_weakpoints:
        try:
            raw_strategies.extend(_strategies_from_weakpoint(weakpoint, {**records, **{
                key: {"type": "claim", "content": value} for key, value in candidate_contents.items()}}))
        except ValueError as exc:
            findings.append(Finding("STEP4_WEAKPOINT_NOT_EXPANDED", "warning", f"Retained {weakpoint['id']}: {exc}"))
    # LLM weakpoint judgments can point in both directions across overlapping
    # retrieval groups.  A single cycle must not discard the entire merge: keep
    # a deterministic maximal acyclic subset and report the omitted strategy.
    acyclic_strategies: list[JSONDict] = []
    for strategy in raw_strategies:
        try:
            _validate_acyclic([*acyclic_strategies, strategy])
        except ValueError as exc:
            findings.append(Finding(
                "STEP4_CYCLIC_STRATEGY_SKIPPED", "warning",
                f"Skipped cyclic integration strategy {strategy.get('strategy_id', '<unbound>')}: {exc}",
            ))
            continue
        acyclic_strategies.append(strategy)
    raw_strategies = acyclic_strategies
    supported_candidates = {str(strategy["conclusion"]) for strategy in raw_strategies} & set(candidate_contents)
    for candidate_id in sorted(set(candidate_contents) - supported_candidates):
        findings.append(Finding("STEP4_CANDIDATE_K_UNSUPPORTED", "warning",
                                f"Discarded {candidate_id}: no weakpoint expanded into a supporting Strategy"))

    used = set(source_nodes)
    used.update(qid for operator in source_operators for qid in [*operator["variables"], operator.get("conclusion")] if isinstance(qid, str))
    used.update(qid for strategy in source_strategies for qid in [*strategy["premises"], strategy["conclusion"], *strategy["background"]])
    used.update(qid for operator in operators for qid in [*operator["variables"], operator.get("conclusion")] if isinstance(qid, str))
    used.update(qid for strategy in raw_strategies
                for qid in [*strategy["premises"], strategy["conclusion"], *strategy["background"]])
    used.update(qid for merge in merges for qid in merge["merged_qids"])
    knowledges: dict[str, JSONDict] = {}
    nodes: list[str] = []
    source_anchor_ids: set[str] = set()
    merge_by_representative = {merge["representative_qid"]: merge for merge in merges}
    for qid in sorted(used):
        if qid in supported_candidates:
            knowledges[qid] = {"type": "claim", "content": {"canonical": candidate_contents[qid]}, "source_anchor_ids": []}
            nodes.append(qid)
            continue
        representative = aliases.get(qid, qid)
        if representative != qid or representative in knowledges or representative not in records:
            continue
        record = records[representative]
        anchor_ids = merge_by_representative.get(representative, {}).get("source_anchor_ids", record["source_anchor_ids"])
        source_anchor_ids.update(anchor_ids)
        knowledge_type = record["type"] if record["type"] in {"claim", "observation_claim", "note"} else "claim"
        knowledges[representative] = {"type": knowledge_type, "content": {"canonical": record["content"]},
                                      "source_anchor_ids": list(anchor_ids),
                                      "metadata": {"package_provenance": merge_by_representative.get(representative, {}).get(
                                          "package_provenance", [record["package"]])}}
        if knowledge_type != "note":
            nodes.append(representative)

    strategies: list[JSONDict] = []
    seen_strategy_ids: set[str] = set()
    for strategy in [*source_strategies, *raw_strategies]:
        bound = canonical_strategy(strategy)
        if bound["strategy_id"] not in seen_strategy_ids:
            seen_strategy_ids.add(bound["strategy_id"])
            strategies.append(bound)

    options = validate_step0_options(_step0_options(context))
    domain_name = re.sub(r"[^A-Za-z0-9_]+", "_", options["scope"]["domain"]).strip("_") or "domain"
    document: JSONDict = {
        "schema_version": "1.1.0",
        "revision": {"revision_id": f"revision_{context.run_id}_step_4", "supersedes": None,
                     "parent_hash": proposals["source_context_artifact"]["sha256"], "content_hash": ""},
        "package": {"paper_id": f"integration:{options['scope']['domain']}", "namespace": "integration",
                    "name": domain_name, "version": "1"},
        "graph": {"nodes": sorted(set(nodes)), "operators": [*source_operators, *operators], "strategies": strategies, "composes": []},
        "knowledges": knowledges,
        "workflow": {"source_records": [], "source_anchors": [copy.deepcopy(anchors[key]) for key in sorted(source_anchor_ids)],
                     "weakpoints": [], "gaps": [], "non_reasoning_links": [], "revisions": []},
    }
    document["revision"]["content_hash"] = content_hash(document)
    return document, findings


def _integration_delta(document: JSONDict, source_nodes: list[str], source_strategies: list[JSONDict]) -> JSONDict:
    """Derive a review-only subgraph: integration relations and their endpoints."""
    source_ids = set(source_nodes)
    source_keys = {(tuple(item.get("premises", [])), item.get("conclusion"), item.get("type")) for item in source_strategies}
    strategies = [item for item in document["graph"]["strategies"]
                  if (tuple(item.get("premises", [])), item.get("conclusion"), item.get("type")) not in source_keys]
    delta_ids = {qid for item in strategies for qid in [*item.get("premises", []), item.get("conclusion")] if isinstance(qid, str)}
    knowledges = {qid: document["knowledges"][qid] for qid in delta_ids if qid in document["knowledges"]}
    anchors = {anchor["anchor_id"]: anchor for anchor in document["workflow"]["source_anchors"]}
    used_anchors = {anchor for knowledge in knowledges.values() for anchor in knowledge.get("source_anchor_ids", [])}
    delta = copy.deepcopy(document)
    delta["package"] = {**document["package"], "name": f"{document['package']['name']}_integration_delta"}
    delta["graph"] = {"nodes": sorted(delta_ids), "operators": [], "strategies": strategies, "composes": []}
    delta["knowledges"] = knowledges
    delta["workflow"] = {**document["workflow"], "source_anchors": [anchors[key] for key in sorted(used_anchors)]}
    delta["revision"] = {**document["revision"], "revision_id": f"{document['revision']['revision_id']}_integration_delta", "content_hash": ""}
    delta["revision"]["content_hash"] = content_hash(delta)
    return delta


class Step4FormalizeIntegrationPlugin:
    stage_name = STEP_NAME

    def run(self, context: StageContext) -> StageResult:
        try:
            validate_step1_inputs(context)
            proposal_ref = context.require_one("integration.step3_proposals")
            proposals = _read_json(context, proposal_ref)
            if proposals.get("schema_name") != PROPOSAL_SCHEMA or proposals.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Step 4 requires the supported Step 3 proposal schema")
            records, anchors = _load_records(context)
            source_nodes, source_operators, source_strategies = _load_source_graph(context, records)
            raw_operators = [copy.deepcopy(item) for item in proposals.get("operators") or [] if isinstance(item, dict)]
            aliases, merges = _equivalence_aliases(raw_operators, records)
            operators = [rewritten for item in raw_operators if (rewritten := _rewrite_operator(item, aliases)) is not None]
            operators = list({json.dumps(item, sort_keys=True): item for item in operators}.values())
            weakpoints = [rewritten for item in proposals.get("weakpoints") or []
                          if isinstance(item, dict) and (rewritten := _rewrite_weakpoint(item, aliases)) is not None]
            candidates = [copy.deepcopy(item) for item in proposals.get("candidate_knowledges") or [] if isinstance(item, dict)]
            candidate_contents = _materialize_candidates(candidates, weakpoints, records)
            document, findings = _build_document(
                context, proposals, records, anchors, aliases, merges, operators, weakpoints, candidate_contents,
                source_nodes, source_operators, source_strategies,
            )
            emitted = emit_formalization(context, document, step=4, step_name=STEP_NAME)
            delta_path = context.work_dir / "integration_delta_formalization.json"
            atomic_write_json(delta_path, _integration_delta(document, source_nodes, source_strategies))
            artifacts = [*emitted.artifacts, ArtifactDraft(delta_path, "formalization.integration_delta", "application/json", {
                "schema_name": "gaia.formalization.v2", "schema_version": "1.1.0", "step": 4,
                "step_name": STEP_NAME, "view_variant": "integration_delta", "validation_status": "passed",
            })]
            return StageResult(emitted.status, artifacts, [*findings, *emitted.findings], {
                **emitted.metadata, "equivalence_merge_count": len(merges),
                "materialized_candidate_count": len(candidate_contents),
            })
        except (Step1InputError, ValueError, KeyError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            return StageResult("failed", findings=[Finding("STEP4_FORMALIZATION_FAILED", "error", str(exc))])


__all__ = ["Step4FormalizeIntegrationPlugin"]
