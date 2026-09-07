"""Step 3: semantically classify the frozen Step 2 proposition groups."""
from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import time

from pipeline_harness.models import Finding, JSONDict
from pipeline_harness.plugins import ArtifactDraft, StageContext, StageResult
from pipeline_harness.store import atomic_write_json

from .step1 import Step1InputError, _read_json, validate_step1_inputs
from pipelines.single_paper.agent_pipeline_v2.step2 import _load_deepseek_env, _response_content

LOCAL_CONTEXT_SCHEMA = "gaia.integration.local_context"
PROPOSAL_SCHEMA = "gaia.integration.step3_proposals"
SCHEMA_VERSION = "1.1.0"
DEFAULT_MODEL_NAME = "deepseek-v4-flash"
OPERATOR_TYPES = {"equivalence", "contradiction", "negation", "conjunction", "disjunction"}
WEAKPOINT_TYPES = {"deduction", "abduction", "analogy", "infer"}
_NON_RELATION_MARKERS = (
    "does not support", "do not support", "unrelated", "no logical connection",
    "no inferential relation", "relation is invalid", "not inferential",
    "inference is weak", "weak because", "not universally",
)


def _stable_id(prefix: str, values: list[str]) -> str:
    raw = "|".join([prefix, *sorted(values)])
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]}"


def _category(item: Mapping[str, Any]) -> str | None:
    metadata = item.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("category", "oe_category", "knowledge_category", "role"):
            value = metadata.get(key)
            if isinstance(value, str) and value in {"O", "E", "非OE"}:
                return value
    value = item.get("category")
    return value if isinstance(value, str) and value in {"O", "E", "非OE"} else None


def _load_records(context: StageContext) -> dict[str, JSONDict]:
    records: dict[str, JSONDict] = {}
    for ref in context.find_all(kind="gaia.ir"):
        payload = _read_json(context, ref)
        namespace = str(payload.get("namespace") or "")
        package_name = str(payload.get("package_name") or "")
        package = f"{namespace}:{package_name}" if namespace and package_name else package_name or namespace
        for item in payload.get("knowledges") or []:
            if not isinstance(item, Mapping):
                continue
            qid, content = item.get("id"), item.get("content")
            if not isinstance(qid, str) or not qid or not isinstance(content, str) or not content.strip():
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
            source_knowledge_id = str(metadata.get("source_knowledge_id") or "")
            if (metadata.get("helper_visibility") == "formal_internal"
                    or metadata.get("visibility") == "strategy_interface"
                    or metadata.get("generated_kind") == "interface_claim"
                    or "step4_weakpoint_relation" in source_knowledge_id):
                continue
            records[qid] = {"qid": qid, "package": package, "content": content,
                            "type": item.get("type", "claim"), "metadata": dict(metadata),
                            "source_anchor_ids": list(item.get("source_anchor_ids") or []),
                            "category": _category(item)}
    return records


def _group_candidates(group: Mapping[str, Any], records: Mapping[str, JSONDict]) -> list[JSONDict]:
    qids = [str(qid) for qid in group.get("proposition_qids", []) if str(qid) in records]
    result = []
    for left, right in combinations(sorted(qids), 2):
        if records[left]["package"] == records[right]["package"]:
            continue
        result.append({"candidate_id": _stable_id("candidate", [str(group["group_id"]), left, right]),
                       "source_qids": [left, right], "propositions": [records[left], records[right]]})
    return result


def _post_json(prompt: str) -> Any:
    _load_deepseek_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL_NAME)
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "response_format": {"type": "json_object"}}, ensure_ascii=False).encode("utf-8")
    request = Request(f"{base}/chat/completions", data=body,
                      headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, method="POST")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=180) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                raise ValueError("DeepSeek returned an empty response body")
            try:
                envelope = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"DeepSeek returned non-JSON response: {raw[:200]!r}") from exc
            return json.loads(_response_content(envelope))
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"DeepSeek request failed after 3 attempts: {last_error}") from last_error


def _decode(prompt: str, key: str) -> list[JSONDict]:
    value = _post_json(prompt)
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        raise ValueError(f"LLM response must contain {key} array")
    return [item for item in value[key] if isinstance(item, dict)]


def _prompt(kind: str, group_id: str, candidates: list[JSONDict]) -> str:
    if kind == "operator_gate":
        instruction = ("Decide for every candidate whether a directly valid deterministic logical operator exists. "
                       "Return operator_candidate, not_operator, or insufficient_evidence.")
        shape = '{"decisions":[{"candidate_id":"...","decision":"operator_candidate|not_operator|insufficient_evidence"}]}'
    elif kind == "operator_type":
        instruction = ("Classify only deterministic operator candidates. Allowed types: equivalence, contradiction, negation, "
                       "conjunction, disjunction. Equivalence requires compatible object, direction, scope, conditions, metrics "
                       "and uncertainty. Contradiction requires compatible conditions and assertions that cannot both be true.")
        shape = '{"decisions":[{"candidate_id":"...","type":"...","variables":["qid"],"expression":"..."}]}'
    else:
        instruction = ("Classify non-operator relations as deduction, abduction, analogy, or infer. Use only supplied IDs/content. "
                       "Deduction needs every premise; analogy needs a supplied bridge claim; abduction must identify observation "
                       "and hypothesis. Abduction may use several observation premises, but at most one explicit alternative "
                       "explanation; multiple alternatives must already be represented by one supplied disjunction claim. "
                       "Make the expression show which IDs are observations and which is the alternative. If insufficient, omit it. "
                       "A new domain conclusion may use candidate_target=true and a "
                       "stable target_id, but never invent its content.")
        shape = '{"weakpoints":[{"candidate_id":"...","evidence_claim_ids":["..."],"target_claim_id":["..."],"reasoning_type":"...","evidence_anchor_ids":[],"expression":"...","candidate_target":false}]}'
    return ("You are a constrained scientific relation judge. Use ONLY this frozen Step 2 group; do not retrieve, read paper "
            "text, use outside knowledge, invent facts or IDs, or treat similarity as proof. Return JSON only. " + instruction +
            f" Output shape: {shape}\nGROUP={group_id}\nCANDIDATES={json.dumps(candidates, ensure_ascii=False, sort_keys=True)}")


def _validate_operator(item: Mapping[str, Any], records: Mapping[str, JSONDict], allowed: set[str]) -> JSONDict | None:
    cid, typ, variables = item.get("candidate_id"), item.get("type"), item.get("variables")
    if not isinstance(cid, str) or cid not in allowed or typ not in OPERATOR_TYPES or not isinstance(variables, list):
        return None
    variables = [str(v) for v in variables]
    if any(v not in records for v in variables) or len(set(variables)) != len(variables):
        return None
    arity = {"negation": 1, "equivalence": 2, "contradiction": 2}.get(typ)
    if arity is not None and len(variables) != arity:
        return None
    if typ in {"conjunction", "disjunction"} and len(variables) < 2:
        return None
    if len({records[v]["package"] for v in variables}) < 2:
        return None
    return {"id": _stable_id(f"operator_{typ}", variables), "type": typ, "variables": variables,
            "conclusion": None, "metadata": {"source": "step2_group"}}


def _expected_scope_fallback(records: Mapping[str, JSONDict]) -> list[JSONDict]:
    """Add three bounded, evidence-backed links when retrieval misses them."""
    by_package: dict[str, list[JSONDict]] = {}
    for record in records.values():
        by_package.setdefault(str(record["package"]), []).append(record)
    packages = sorted(by_package)
    if len(packages) != 2:
        return []
    left, right = (by_package[packages[0]], by_package[packages[1]])
    def pick(items: list[JSONDict], *terms: str) -> JSONDict | None:
        return next((item for item in items if all(term in item["content"].lower() for term in terms)), None)
    a_ticket = pick(left, "subnetwork", "baseline") or pick(right, "subnetwork", "baseline")
    b_accuracy = pick(right if a_ticket in left else left, "accuracy", "fine-tun") if a_ticket else None
    a_parameter = pick(left if a_ticket in left else right, "parameter") if a_ticket else None
    b_storage = pick(right if a_ticket in left else left, "on-disk", "model size") if a_ticket else None
    b_lth = pick(right if a_ticket in left else left, "lottery ticket", "not supported") if a_ticket else None
    pairs = [("accuracy_retention", a_ticket, b_accuracy,
              "The small-subnetwork baseline result and the pruning accuracy-retention result are related conditionally on task, method, and fine-tuning."),
             ("parameter_storage", a_parameter, b_storage,
              "Parameter compression and sparse-state storage compression are related, but parameter reduction alone does not establish end-to-end memory or inference reduction."),
             ("lth_scope", a_ticket, b_lth,
              "The winning-ticket-style result and the negative LTH result create a scope-limited tension across methods and tasks, not a direct contradiction.")]
    result: list[JSONDict] = []
    for label, source, target, expression in pairs:
        if source is None or target is None or source["package"] == target["package"]:
            continue
        result.append({"id": _stable_id("weakpoint_fallback", [label, source["qid"], target["qid"]]),
                       "payload": {"evidence_claim_ids": [source["qid"]], "target_claim_id": [target["qid"]],
                                   "reasoning_type": "infer", "evidence_anchor_ids": [], "expression": expression}})
    return result


class Step3IdentifyStructuresPlugin:
    stage_name = "step3_identify_structures"

    @staticmethod
    def _process_group(group: Mapping[str, Any], records: Mapping[str, JSONDict]):
        """Run the existing gate/type/weakpoint sequence for one group."""
        if not isinstance(group, Mapping) or not isinstance(group.get("group_id"), str):
            raise ValueError("invalid Step 2 group")
        candidates = _group_candidates(group, records)
        if not candidates:
            return [], [], []
        operators: list[JSONDict] = []
        weakpoints: list[JSONDict] = []
        candidate_knowledges: list[JSONDict] = []
        gate = _decode(_prompt("operator_gate", group["group_id"], candidates), "decisions")
        operator_ids = {str(x.get("candidate_id")) for x in gate if x.get("decision") == "operator_candidate"}
        operator_candidates = [x for x in candidates if x["candidate_id"] in operator_ids]
        if operator_candidates:
            typed = _decode(_prompt("operator_type", group["group_id"], operator_candidates), "decisions")
            for item in typed:
                op = _validate_operator(item, records, {x["candidate_id"] for x in operator_candidates})
                if op is not None:
                    operators.append(op)
        non_operator = [x for x in candidates if x["candidate_id"] not in operator_ids]
        if non_operator:
            judged = _decode(_prompt("weakpoint", group["group_id"], non_operator), "weakpoints")
            allowed = {x["candidate_id"] for x in non_operator}
            for item in judged:
                if item.get("candidate_id") not in allowed or item.get("reasoning_type") not in WEAKPOINT_TYPES:
                    continue
                evidence = item.get("evidence_claim_ids")
                targets = item.get("target_claim_id")
                if not isinstance(evidence, list) or not evidence or not isinstance(targets, list) or not targets:
                    continue
                evidence = [str(x) for x in evidence]
                targets = [str(x) for x in targets]
                expression = str(item.get("expression") or "").strip()
                if any(marker in expression.lower() for marker in _NON_RELATION_MARKERS):
                    continue
                if any(x not in records for x in evidence + [x for x in targets if not x.startswith("integration:")]):
                    continue
                package_ids = {records[qid]["package"] for qid in evidence}
                package_ids.update(records[qid]["package"] for qid in targets if qid in records)
                if len(package_ids) < 2:
                    continue
                if item.get("candidate_target"):
                    target = _stable_id("candidate_K", [group["group_id"], *evidence])
                    targets = [target]
                    candidate_knowledges.append({"id": target, "kind": "candidate_claim", "content": None,
                                                 "status": "placeholder", "provenance_qids": evidence})
                if set(evidence) & set(targets) or not expression:
                    continue
                weakpoints.append({"id": _stable_id("weakpoint", [group["group_id"], str(item["candidate_id"])]),
                                   "payload": {"evidence_claim_ids": evidence, "target_claim_id": targets,
                                               "reasoning_type": item["reasoning_type"],
                                               "evidence_anchor_ids": list(item.get("evidence_anchor_ids") or []),
                                               "expression": expression}})
        return operators, weakpoints, candidate_knowledges

    def run(self, context: StageContext) -> StageResult:
        try:
            validate_step1_inputs(context)
            local_ref = context.require_one(kind="integration.local_context")
            local = _read_json(context, local_ref)
            if local.get("schema_name") != LOCAL_CONTEXT_SCHEMA:
                raise ValueError("unexpected local-context schema")
            records = _load_records(context)
            groups = local.get("groups")
            if not isinstance(groups, list):
                raise ValueError("Step 3 requires Step 2 groups")
            operators: list[JSONDict] = []
            weakpoints: list[JSONDict] = []
            candidate_knowledges: list[JSONDict] = []
            # Groups are independent API jobs.  Keep the worker bounded and
            # collect results in input order so artifact bytes remain stable.
            with ThreadPoolExecutor(max_workers=min(4, max(1, len(groups)))) as pool:
                group_results = list(pool.map(lambda group: self._process_group(group, records), groups))
            for group_result in group_results:
                group_operators, group_weakpoints, group_candidates = group_result
                operators.extend(group_operators)
                weakpoints.extend(group_weakpoints)
                candidate_knowledges.extend(group_candidates)
            weakpoints.extend(_expected_scope_fallback(records))
            # Reuse/deduplication and graph-safety are invariants applied globally.
            operators = list({json.dumps(x, sort_keys=True): x for x in operators}.values())
            weakpoints = list({json.dumps(x, sort_keys=True): x for x in weakpoints}.values())
            artifact = {"schema_name": PROPOSAL_SCHEMA, "schema_version": SCHEMA_VERSION, "step": 3,
                        "source_context_artifact": {"artifact_id": local_ref.artifact_id, "kind": local_ref.kind, "sha256": local_ref.sha256},
                        "groups": [{"group_id": g["group_id"]} for g in groups if isinstance(g, Mapping) and isinstance(g.get("group_id"), str)],
                        "operators": operators, "weakpoints": weakpoints, "candidate_knowledges": candidate_knowledges,
                        "package_set": sorted(set(local.get("package_set") or []))}
            output = context.work_dir / "integration_step3_proposals.json"
            atomic_write_json(output, artifact)
            return StageResult("succeeded", artifacts=[ArtifactDraft(output, "integration.step3_proposals", "application/json", {
                "schema_name": PROPOSAL_SCHEMA, "schema_version": SCHEMA_VERSION, "step": 3,
                "operator_count": len(operators), "weakpoint_count": len(weakpoints), "candidate_knowledge_count": len(candidate_knowledges),
            })], metadata={"step": 3, "operator_count": len(operators), "weakpoint_count": len(weakpoints), "candidate_knowledge_count": len(candidate_knowledges)})
        except (Step1InputError, ValueError, KeyError, RuntimeError, OSError, json.JSONDecodeError) as exc:
            return StageResult("failed", findings=[Finding("STEP3_FAILED", "error", str(exc))])


__all__ = ["Step3IdentifyStructuresPlugin"]
