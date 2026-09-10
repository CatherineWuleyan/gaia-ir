"""Step 3: semantically classify the frozen Step 2 proposition groups."""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
GATE_BATCH_SIZE = 20
_RESOURCE_SPAM = re.compile(
    r"\b(CPU|energy|memory|latency|runtime|power|inference time)\b", re.IGNORECASE
)
# Compiler-generated helper / interface names.  Some Paper Packages ship them
# without the `helper_visibility` marker, so match the name as well.
_HELPER_ID_RE = re.compile(
    r"(^|::)__|helper[_-]?relation|operator[_-]?result|disjunction[_-]?result|"
    r"alternative[_-]?explanation|step4_weakpoint_",
    re.I,
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
                    or "step4_weakpoint_relation" in source_knowledge_id
                    or _HELPER_ID_RE.search(qid)):
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


def _is_resource_observation(record: Mapping[str, Any]) -> bool:
    """Skip numeric resource-measurement spam (CPU/energy/memory/...)."""
    content = str(record.get("content") or "")
    return len(content) < 320 and bool(_RESOURCE_SPAM.search(content))


def _retrieval_prompt(anchor: JSONDict, candidates: list[JSONDict]) -> str:
    payload = [{"qid": claim["qid"], "content": claim["content"]} for claim in candidates]
    return (
        "You are a cross-paper relation finder. Given ONE anchor claim (from one paper) and a list of candidate "
        "claims (from other papers), return the IDs of candidate claims that have a specific, defensible relation "
        "to the anchor: a shared object with a real inferential link, not mere topic overlap. For each, give the "
        "direction (anchor_supports_candidate, candidate_supports_anchor, or tension) and a one-line reason. "
        'Return JSON only: {"related":[{"qid":"...","direction":"...","reason":"..."}]}\n'
        f"ANCHOR={json.dumps({'qid': anchor['qid'], 'content': anchor['content']}, ensure_ascii=False)}\n"
        f"CANDIDATES={json.dumps(payload, ensure_ascii=False)}"
    )


def _retrieve_related(records: Mapping[str, JSONDict], max_workers: int = 4) -> list[JSONDict]:
    """Retrieve plausible cross-Package claim pairs with the LLM.

    Asking the model to classify every pair of a large all-pairs batch makes it
    default to "no relation"; asking it to *find* each anchor's related claims is
    far more reliable and yields a small, judgeable candidate set.  Every
    cross-Package relation has an endpoint in some Package, so anchoring each
    Public claim against all other Packages' claims covers the pair space.
    """
    by_package: dict[str, list[str]] = defaultdict(list)
    for qid, record in records.items():
        if record.get("type") == "note" or _is_resource_observation(record):
            continue
        by_package[str(record["package"])].append(qid)
    packages = sorted(by_package)
    if len(packages) < 2:
        return []

    collected: list[JSONDict] = []
    seen: set[tuple[str, str]] = set()
    lock = threading.Lock()

    def retrieve(anchor_qid: str, candidate_qids: list[str]) -> None:
        response = _post_json(_retrieval_prompt(records[anchor_qid], [records[q] for q in candidate_qids]))
        if not isinstance(response, dict):
            return
        found: list[JSONDict] = []
        for item in response.get("related") or []:
            if not isinstance(item, Mapping):
                continue
            cid = item.get("qid")
            if not isinstance(cid, str) or cid not in records or cid == anchor_qid:
                continue
            key = tuple(sorted((anchor_qid, cid)))
            with lock:
                if key in seen:
                    continue
                seen.add(key)
            found.append({
                "candidate_id": _stable_id("candidate", [anchor_qid, cid]),
                "source_qids": [anchor_qid, cid],
                "propositions": [records[anchor_qid], records[cid]],
                "hint": {"direction": str(item.get("direction") or ""), "reason": str(item.get("reason") or "")},
            })
        with lock:
            collected.extend(found)

    jobs = [
        (anchor_qid, [q for other in packages if other != package for q in by_package[other]])
        for package in packages
        for anchor_qid in by_package[package]
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        list(pool.map(lambda job: retrieve(*job), jobs))
    collected.sort(key=lambda item: item["candidate_id"])
    return collected


def _post_json(prompt: str) -> Any:
    _load_deepseek_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not configured")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("DEEPSEEK_MERGE_MODEL", DEFAULT_MODEL_NAME)
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
                       "Return operator_candidate if yes; not_operator only if a non-deterministic reasoning relation "
                       "(deduction/abduction/analogy/infer) is supported by the supplied content; otherwise "
                       "insufficient_evidence. Superficial topical similarity alone is insufficient_evidence.")
        shape = '{"decisions":[{"candidate_id":"...","decision":"operator_candidate|not_operator|insufficient_evidence"}]}'
    elif kind == "operator_type":
        instruction = ("Classify only deterministic operator candidates. Allowed types: equivalence, contradiction, negation, "
                       "conjunction, disjunction. Equivalence requires compatible object, direction, scope, conditions, metrics "
                       "and uncertainty. Contradiction requires compatible conditions and assertions that cannot both be true.")
        shape = '{"decisions":[{"candidate_id":"...","type":"...","variables":["qid"],"expression":"..."}]}'
    else:
        instruction = ("Classify non-operator relations between the supplied candidates as deduction, abduction, analogy, or infer. "
                       "Use only supplied IDs/content. Deduction needs every premise; analogy needs a supplied bridge claim; "
                       "abduction must identify observation and hypothesis. Abduction may use several observation premises, but at "
                       "most one explicit alternative explanation; multiple alternatives must already be represented by one supplied "
                       "disjunction claim. Make the expression show which IDs are observations and which is the alternative. "
                       "Emit a relation only when it is specific and defensible from the supplied content; being about the same topic "
                       "or sharing vocabulary is NOT a relation — omit it. Use infer only when the direction is clear but no stronger "
                       "family fits; do not default to infer. Never propose a new conclusion here.")
        shape = '{"weakpoints":[{"candidate_id":"...","evidence_claim_ids":["..."],"target_claim_id":["..."],"reasoning_type":"...","evidence_anchor_ids":[],"expression":"..."}]}'
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


def _synthesis_prompt(claims: list[JSONDict]) -> str:
    payload = [
        {"qid": claim["qid"], "package": claim["package"], "content": claim["content"]}
        for claim in claims
    ]
    return (
        "You are a domain-integration synthesizer. Below are claims from multiple paper packages that were "
        "judged cross-paper related. Write ONE domain conclusion stating the shared regularity they jointly "
        "support.\n"
        "Rules:\n"
        "1. Abstract, never list. Do NOT restate, quote or concatenate the input claims; a conclusion whose "
        "clauses are joined by 'and' / 'while' is a copy, not a synthesis.\n"
        "2. It must be a single coherent proposition about the common pattern the inputs share (what holds "
        "across the methods, tasks and conditions they cover).\n"
        "3. Keep it conditional on the tested scope and invent nothing beyond the supplied claims.\n"
        "4. If the claims share no common regularity, return an empty supporting_qids list.\n"
        'BAD (a copy): "A holds, and B holds, while C holds."\n'
        'GOOD (a synthesis): "Across the evaluated <methods/tasks>, <common regularity>, conditional on <scope>."\n'
        'Return JSON only: {"summary":"...", "supporting_qids":["qid", ...]}'
        f"\nCLAIMS={json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )


def _synthesize_domain_conclusion(
    weakpoints: list[JSONDict], records: Mapping[str, JSONDict],
) -> tuple[list[JSONDict], list[JSONDict]]:
    """Synthesize one bounded domain conclusion (candidate K) from the cross-paper relations.

    Pairwise judgment stays conservative (a mere topical overlap must not become a relation), but
    the merge still needs a summary proposition: gather the claims the cross-paper relations touch
    and let the LLM decide whether they jointly support one domain conclusion.
    """
    involved: dict[str, JSONDict] = {}
    for weakpoint in weakpoints:
        payload = weakpoint.get("payload") or {}
        for qid in [*payload.get("evidence_claim_ids", []), *payload.get("target_claim_id", [])]:
            if isinstance(qid, str) and qid in records:
                involved[qid] = records[qid]
    if len(involved) < 2:
        return [], []
    claims = [involved[qid] for qid in sorted(involved)]
    response = _post_json(_synthesis_prompt(claims))
    if not isinstance(response, dict):
        return [], []
    supporting = [qid for qid in response.get("supporting_qids") or [] if isinstance(qid, str) and qid in records]
    supporting = list(dict.fromkeys(supporting))
    summary = str(response.get("summary") or "").strip()
    if len(supporting) < 2 or not summary:
        return [], []
    target = _stable_id("candidate_K", supporting)
    # The synthesis prompt is the authoritative domain-conclusion writer, so its
    # abstracted summary is carried through as the candidate's content; Step 4
    # must not re-write it with the coarser per-pair candidate prompt.
    candidate = {"id": target, "kind": "candidate_claim", "content": summary,
                 "status": "materialized", "provenance_qids": supporting}
    # The domain conclusion is the general proposition; it infers the finer
    # Paper claims it was abstracted from (K -> finer), rather than being a sink
    # that merely aggregates them.
    weakpoint = {"id": _stable_id("weakpoint", ["synthesis", target]),
                 "payload": {"evidence_claim_ids": [target], "target_claim_id": supporting,
                             "reasoning_type": "infer", "evidence_anchor_ids": [],
                             "expression": f"[{target}] 推出 " + " 与 ".join(f"[{qid}]" for qid in supporting)}}
    return [candidate], [weakpoint]


class Step3IdentifyStructuresPlugin:
    stage_name = "step3_identify_structures"

    @staticmethod
    def _judge_candidates(
        candidates: list[JSONDict], group_id: str, records: Mapping[str, JSONDict],
    ) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict]]:
        """Run the gate/type/weakpoint sequence over one candidate batch."""
        if not candidates:
            return [], [], []
        operators: list[JSONDict] = []
        weakpoints: list[JSONDict] = []
        candidate_knowledges: list[JSONDict] = []
        gate = _decode(_prompt("operator_gate", group_id, candidates), "decisions")
        decisions = {str(x.get("candidate_id")): x.get("decision") for x in gate}
        operator_ids = {cid for cid, decision in decisions.items() if decision == "operator_candidate"}
        weakpoint_ids = {cid for cid, decision in decisions.items() if decision == "not_operator"}
        operator_candidates = [x for x in candidates if x["candidate_id"] in operator_ids]
        if operator_candidates:
            typed = _decode(_prompt("operator_type", group_id, operator_candidates), "decisions")
            for item in typed:
                op = _validate_operator(item, records, {x["candidate_id"] for x in operator_candidates})
                if op is not None:
                    operators.append(op)
        non_operator = [x for x in candidates if x["candidate_id"] in weakpoint_ids]
        if non_operator:
            judged = _decode(_prompt("weakpoint", group_id, non_operator), "weakpoints")
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
                # Pairwise judgment only emits direct relations.  Domain
                # conclusions (K) are synthesized once, from all the cross-paper
                # relations together, so the merge does not mint a redundant K
                # per candidate pair.
                if set(evidence) & set(targets) or not expression:
                    continue
                weakpoints.append({"id": _stable_id("weakpoint", [group_id, str(item["candidate_id"])]),
                                   "payload": {"evidence_claim_ids": evidence, "target_claim_id": targets,
                                               "reasoning_type": item["reasoning_type"],
                                               "evidence_anchor_ids": list(item.get("evidence_anchor_ids") or []),
                                               "expression": expression}})
        return operators, weakpoints, candidate_knowledges

    @staticmethod
    def _process_group(group: Mapping[str, Any], records: Mapping[str, JSONDict]):
        """Judge the cross-Package candidate pairs inside one Step 2 group."""
        if not isinstance(group, Mapping) or not isinstance(group.get("group_id"), str):
            raise ValueError("invalid Step 2 group")
        return Step3IdentifyStructuresPlugin._judge_candidates(
            _group_candidates(group, records), str(group["group_id"]), records,
        )

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
            # Bootstrap judges the whole batch jointly: the candidate set is every
            # cross-Package public claim pair (semantic pairing via the LLM),
            # because Step 2's lexical cosine misses semantically related pairs.
            # Incremental/reconcile keep the bounded Step 2 groups.
            if local.get("mode") == "bootstrap":
                all_candidates = _retrieve_related(records)
                batches = [all_candidates[i:i + GATE_BATCH_SIZE]
                           for i in range(0, len(all_candidates), GATE_BATCH_SIZE)]
                jobs = [(batch, "bootstrap") for batch in batches]
            else:
                jobs = [(_group_candidates(group, records), str(group["group_id"])) for group in groups]
            with ThreadPoolExecutor(max_workers=min(4, max(1, len(jobs)))) as pool:
                results = list(pool.map(lambda job: self._judge_candidates(job[0], job[1], records), jobs))
            for group_operators, group_weakpoints, group_candidates in results:
                operators.extend(group_operators)
                weakpoints.extend(group_weakpoints)
                candidate_knowledges.extend(group_candidates)
            # Synthesize a bounded domain conclusion (K) from the cross-paper
            # relations so the merge yields a summary proposition even though
            # pairwise judgment stays conservative.
            if weakpoints:
                synth_candidates, synth_weakpoints = _synthesize_domain_conclusion(weakpoints, records)
                candidate_knowledges.extend(synth_candidates)
                weakpoints.extend(synth_weakpoints)
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
