"""Pipeline 7.0 Step 2: anchor and review experimental observations."""
from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pipeline_harness.domain.stages import (
    _cosine,
    _figure_aliases,
    _paper_paragraphs,
    _tokens,
)
from pipeline_harness.domain.tools import (
    DomainTool,
    ToolCallRequest,
    ToolCallResponse,
    validate_tool_response,
)
from pipeline_harness.models import Finding, JSONDict
from pipeline_harness.plugins import ArtifactDraft, StageContext, StageResult, instantiate
from pipeline_harness.store import atomic_write_json, read_json

from .authoring import content_hash, emit_formalization
from .step1 import INPUT_BUNDLE_KIND, PAPER_TEXT_KIND


STEP_NAME = "step2_complete_observations"
MODEL_NAME = "deepseek-v4-flash"
_IMAGE_PATTERN = re.compile(r"!\[[^]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)")
_PARAGRAPH_ANCHOR_PATTERN = re.compile(r"^anchor_paragraph_p(\d+)$")


def _best_paragraph_anchor(content: str, paragraphs: list[JSONDict]) -> JSONDict | None:
    """Apply Pipeline 7.0's deterministic anchoring threshold."""
    query = _tokens(content)
    if not query:
        return None
    eligible: list[tuple[float, int, JSONDict]] = []
    for index, paragraph in enumerate(paragraphs):
        text = str(paragraph.get("text", ""))
        nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
        if nonempty_lines and all(line.startswith("#") for line in nonempty_lines):
            continue
        paragraph_tokens = _tokens(text)
        shared = set(query) & set(paragraph_tokens)
        score = _cosine(query, paragraph_tokens)
        if len(shared) >= 2 and score >= 0.2:
            eligible.append((score, index, paragraph))
    return max(eligible, default=(0.0, 0, None), key=lambda item: (item[0], -item[1]))[2]


def _anchors_share_local_context(left: list[str], right: list[str], distance: int = 2) -> bool:
    """Treat same or nearby paper paragraphs as one local experiment context."""
    if set(left) & set(right):
        return True
    left_numbers = {
        int(match.group(1))
        for anchor_id in left
        if (match := _PARAGRAPH_ANCHOR_PATTERN.match(anchor_id))
    }
    right_numbers = {
        int(match.group(1))
        for anchor_id in right
        if (match := _PARAGRAPH_ANCHOR_PATTERN.match(anchor_id))
    }
    return any(abs(a - b) <= distance for a in left_numbers for b in right_numbers)


def _load_deepseek_env() -> None:
    """Load only the supported DeepSeek settings without exposing values."""
    env_path = Path(__file__).resolve().parents[2] / "agent" / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"}:
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def _response_content(raw: Any) -> str:
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("DeepSeek response has no choices[0].message.content") from exc
    if not isinstance(content, str):
        raise ValueError("DeepSeek response content must be a string")
    return content


class DeepSeekV4FlashObservationTool:
    """Extract text-grounded S/A/B/M/R/U claims with DeepSeek V4 Flash."""

    name = "deepseek-v4-flash-observation"
    version = "7"
    model = MODEL_NAME

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        if request.operation == "normalize_weakpoint_clusters":
            return self._normalize_weakpoint_clusters(request)
        if request.operation == "classify_weakpoints":
            return self._classify_weakpoints(request)
        _load_deepseek_env()
        candidates = request.parameters.get("experiment_candidates")
        if not isinstance(candidates, list):
            raise ValueError("experiment_candidates must be a list")
        if not candidates:
            return ToolCallResponse(
                request.call_id,
                "succeeded",
                {"skipped": "no_image_references"},
                {"status": "claims_extracted", "claims": [], "equivalent_claims": [], "relations": []},
                metadata={"candidate_count": 0, "model": self.model},
            )
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        selected_claims = request.parameters.get("selected_claims")
        frozen_claims = request.parameters.get("frozen_claims")
        jobs = [
            (
                candidate,
                self._claims_for_candidate(selected_claims, candidate),
                self._claims_for_candidate(frozen_claims, candidate),
            )
            for candidate in candidates
        ]
        if isinstance(selected_claims, list):
            jobs = [job for job in jobs if job[1]]
            if not jobs:
                raise ValueError("selected claims do not map to any experiment candidate")
        raw_responses: list[Any] = []
        claims: list[JSONDict] = []
        equivalent_claims: list[JSONDict] = []
        relations: list[JSONDict] = []

        def normalization_failure(exc: Exception) -> ToolCallResponse:
            return ToolCallResponse(
                request.call_id,
                "failed",
                {"responses": list(raw_responses)},
                error={"type": type(exc).__name__, "message": str(exc)},
                metadata={
                    "candidate_count": len(candidates),
                    "request_count": len(raw_responses),
                    "model": self.model,
                },
            )

        for candidate, selected_for_candidate, frozen_for_candidate in jobs:
            prompt = self._prompt(
                [candidate],
                selected_for_candidate if isinstance(selected_claims, list) else None,
                request.parameters.get("review_advice"),
                frozen_for_candidate,
            )
            body = json.dumps(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                },
                ensure_ascii=False,
            ).encode("utf-8")
            http_request = Request(
                f"{base_url}/chat/completions",
                data=body,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(http_request, timeout=180) as response:  # noqa: S310
                    raw: Any = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace").strip()
                suffix = f": {detail[:2000]}" if detail else ""
                raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}{suffix}") from exc
            except URLError as exc:
                raise RuntimeError(f"DeepSeek request failed: {exc.reason}") from exc
            raw_responses.append(raw)
            try:
                normalized = self._normalize(
                    _response_content(raw),
                    [candidate],
                    selected_for_candidate if isinstance(selected_claims, list) else None,
                )
            except Exception as exc:
                return normalization_failure(exc)
            if normalized.get("status") == "insufficient_context":
                return ToolCallResponse(
                    request.call_id,
                    "succeeded",
                    {"responses": raw_responses},
                    normalized,
                    metadata={"candidate_count": len(candidates), "request_count": len(raw_responses), "model": self.model},
                )
            claims.extend(normalized["claims"])
            phenomenon_request = Request(
                f"{base_url}/chat/completions",
                data=json.dumps({
                    "model": self.model,
                    "messages": [{"role": "user", "content": self._equivalent_claim_prompt(normalized["claims"], candidate)}],
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                }, ensure_ascii=False).encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(phenomenon_request, timeout=180) as response:  # noqa: S310
                    phenomenon_raw: Any = json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                raise RuntimeError(f"DeepSeek equivalent-claim request failed with HTTP {exc.code}") from exc
            except URLError as exc:
                raise RuntimeError(f"DeepSeek equivalent-claim request failed: {exc.reason}") from exc
            raw_responses.append(phenomenon_raw)
            try:
                candidate_equivalents = self._normalize_equivalent_claims(
                    _response_content(phenomenon_raw), normalized["claims"]
                )
            except Exception as exc:
                return normalization_failure(exc)
            equivalent_claims.extend(candidate_equivalents)
            groups = self._relation_groups(candidate_equivalents, candidate)
            if groups:
                relation_request = Request(
                    f"{base_url}/chat/completions",
                    data=json.dumps({
                        "model": self.model,
                        "messages": [{"role": "user", "content": self._relation_prompt(groups)}],
                        "temperature": 0,
                        "response_format": {"type": "json_object"},
                    }, ensure_ascii=False).encode("utf-8"),
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urlopen(relation_request, timeout=180) as response:  # noqa: S310
                        relation_raw: Any = json.loads(response.read().decode("utf-8"))
                except HTTPError as exc:
                    raise RuntimeError(f"DeepSeek relation request failed with HTTP {exc.code}") from exc
                except URLError as exc:
                    raise RuntimeError(f"DeepSeek relation request failed: {exc.reason}") from exc
                raw_responses.append(relation_raw)
                try:
                    relations.extend(self._normalize_group_relations(_response_content(relation_raw), groups))
                except Exception as exc:
                    return normalization_failure(exc)
        if isinstance(selected_claims, list):
            expected_ids = {item["id"] for item in selected_claims if isinstance(item, dict) and isinstance(item.get("id"), str)}
            actual_ids = [item.get("id") for item in claims]
            if set(actual_ids) != expected_ids or len(actual_ids) != len(set(actual_ids)):
                return normalization_failure(ValueError("targeted rewrite must return exactly the selected claim ids"))
        normalized = {
            "status": "claims_extracted",
            "claims": claims,
            "equivalent_claims": equivalent_claims,
            "relations": relations,
        }
        return ToolCallResponse(
            request.call_id,
            "succeeded",
            {"responses": raw_responses},
            normalized,
            metadata={"candidate_count": len(candidates), "request_count": len(raw_responses), "model": self.model},
        )

    def _classify_weakpoints(self, request: ToolCallRequest) -> ToolCallResponse:
        """Classify only the supplied weakpoint records; never infer new graph data."""
        _load_deepseek_env()
        records = request.parameters.get("weakpoints")
        if not isinstance(records, list) or not records:
            raise ValueError("weakpoints must be a non-empty list")
        record_ids = [item.get("weakpoint_id") for item in records if isinstance(item, dict)]
        if len(record_ids) != len(records) or any(not isinstance(item, str) or not item for item in record_ids):
            raise ValueError("every weakpoint requires a non-empty weakpoint_id")
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("weakpoint IDs must be unique")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        prompt = (
            "Classify every weakpoint using only its supplied evidence claims, one or more target claims, expression, and "
            "evidence-anchor excerpts. Do not use outside knowledge or invent author intent, hidden premises, "
            "alternative explanations, mappings, or conditions.\n\n"
            "Allowed labels only:\n"
            "- deduction: a supplied general rule, definition, model, or computation plus supplied conditions makes the target follow. "
            "Never label an experiment supporting a general conclusion as deduction.\n"
            "- abduction: observations support an explanatory hypothesis, mechanism, or general law that goes beyond the observations, "
            "with alternatives not ruled out by the supplied material.\n"
            "- analogy: a supplied source-domain law/structure is transferred to a target domain through an identifiable but unexpanded mapping or condition.\n"
            "- null: the supplied material cannot identify one of the above.\n\n"
            "Return JSON only: {\"classifications\":[{\"weakpoint_id\":\"...\",\"reasoning_type\":\"abduction|analogy|deduction|null\"}]}. "
            "Return every supplied ID exactly once and no other fields.\n\n"
            f"Weakpoints:\n{json.dumps(records, ensure_ascii=False)}"
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        http_request = Request(
            f"{base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(http_request, timeout=180) as response:  # noqa: S310
                raw: Any = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f": {detail[:2000]}" if detail else ""
            raise RuntimeError(f"DeepSeek weakpoint classification failed with HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RuntimeError(f"DeepSeek weakpoint classification failed: {exc.reason}") from exc
        try:
            payload = json.loads(_response_content(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek weakpoint classification response is not valid JSON") from exc
        classifications = payload.get("classifications") if isinstance(payload, dict) else None
        if not isinstance(classifications, list) or len(classifications) != len(records):
            raise ValueError("DeepSeek must classify every weakpoint exactly once")
        allowed = {"abduction", "analogy", "deduction", None}
        resolved: dict[str, str | None] = {}
        for item in classifications:
            if not isinstance(item, dict) or set(item) != {"weakpoint_id", "reasoning_type"}:
                raise ValueError("weakpoint classifications must contain only weakpoint_id and reasoning_type")
            weakpoint_id, reasoning_type = item["weakpoint_id"], item["reasoning_type"]
            if not isinstance(weakpoint_id, str) or weakpoint_id not in record_ids or weakpoint_id in resolved:
                raise ValueError("weakpoint classification has an unknown or duplicate ID")
            # JSON `null` is the contract value.  Models sometimes serialize that
            # literal as the string "null" despite the response schema; normalize
            # this representation without assigning a substantive category.
            if reasoning_type == "null":
                reasoning_type = None
            if reasoning_type not in allowed:
                raise ValueError("weakpoint classification has an unsupported reasoning_type")
            resolved[weakpoint_id] = reasoning_type
        if set(resolved) != set(record_ids):
            raise ValueError("DeepSeek weakpoint classification omitted an ID")
        return ToolCallResponse(
            request.call_id, "succeeded", {"response": raw},
            {"classifications": [{"weakpoint_id": item, "reasoning_type": resolved[item]} for item in record_ids]},
            metadata={"weakpoint_count": len(records), "model": self.model},
        )

    def _normalize_weakpoint_clusters(self, request: ToolCallRequest) -> ToolCallResponse:
        """Jointly orient and partition bounded candidate-relation clusters."""
        _load_deepseek_env()
        clusters = request.parameters.get("clusters")
        if not isinstance(clusters, list) or not clusters:
            raise ValueError("clusters must be a non-empty list")
        cluster_ids = [item.get("cluster_id") for item in clusters if isinstance(item, dict)]
        if (len(cluster_ids) != len(clusters)
                or any(not isinstance(item, str) or not item for item in cluster_ids)
                or len(cluster_ids) != len(set(cluster_ids))):
            raise ValueError("clusters require unique non-empty cluster_id values")
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
        prompt = (
            "Normalize each bounded scientific argument cluster into its complete minimal set of reasoning weakpoints. "
            "Candidate relations are coarse extraction hints: you may merge them, reverse their direction, or replace a transitive shortcut with adjacent relations, but only within the supplied cluster and only when the supplied claims and original-paper excerpts justify it. "
            "Do not create claims, notes, facts, IDs, conditions, mappings, or intermediate propositions. Step 4 handles source-grounded intermediate propositions. "
            "Fixed equivalence/contradiction relations are outside this task. relation_context_id only bounds review context and never proves that relations should merge.\n\n"
            "Prefer a minimal adjacent hierarchy. Do not retain A-to-C when the cluster supports A-to-B and B-to-C, unless an independent supplied relation explicitly asserts A-to-C. "
            "A multi-target weakpoint is allowed only when the same premises, scope, and reasoning family directly support every target. "
            "When a non-experimental claim B is a self-contained composite phenomenon that strictly entails several atomic phenomenon claims E, represent B -> [E1,E2] as one deduction weakpoint. "
            "If those E claims were separately extracted as evidence for the same explanatory hypothesis A, prefer the non-duplicative adjacent structure B -> A when B faithfully contains the relevant phenomena; do not also emit redundant E1 -> A and E2 -> A weakpoints. "
            "Experimental observations or E claims supporting a general hypothesis are abduction, never deduction.\n\n"
            "Labels: deduction means supplied premises plus source-stated rules/conditions can strictly entail every target; abduction means a phenomenon supports an explanatory hypothesis, mechanism, or generalization while alternatives remain; analogy requires a source-domain law and identifiable cross-domain mapping; null means none can be established from supplied material. "
            "For deduction, do not hide missing rules or empirical assertions. For abduction, premises are phenomena and targets are non-experimental explanations.\n\n"
            "Return JSON only as {\"clusters\":[{\"cluster_id\":\"...\",\"weakpoints\":[{\"member_relation_ids\":[\"...\"],\"evidence_claim_ids\":[\"...\"],\"target_claim_id\":[\"...\"],\"reasoning_type\":\"deduction|abduction|analogy\" OR null,\"expression\":\"...\"}],\"rejected_relation_ids\":[\"...\"]}]}. "
            "Return every cluster exactly once. Every candidate relation ID must occur exactly once, either in one weakpoint's member_relation_ids or in rejected_relation_ids. "
            "Every weakpoint must consume at least one candidate relation. Use only supplied claim IDs, keep evidence and targets non-empty, unique, and disjoint, and cite every endpoint as [claim_id] in expression. "
            "Reject a candidate only when no supported reasoning relation remains after joint normalization.\n\nClusters:\n"
            + json.dumps(clusters, ensure_ascii=False)
        )
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }, ensure_ascii=False).encode("utf-8")
        http_request = Request(
            f"{base_url}/chat/completions", data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(http_request, timeout=180) as response:  # noqa: S310
                raw: Any = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f": {detail[:2000]}" if detail else ""
            raise RuntimeError(f"DeepSeek weakpoint cluster normalization failed with HTTP {exc.code}{suffix}") from exc
        except URLError as exc:
            raise RuntimeError(f"DeepSeek weakpoint cluster normalization failed: {exc.reason}") from exc
        try:
            payload = json.loads(_response_content(raw))
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek weakpoint cluster response is not valid JSON") from exc
        items = payload.get("clusters") if isinstance(payload, dict) else None
        if not isinstance(items, list) or len(items) != len(clusters):
            raise ValueError("DeepSeek must normalize every weakpoint cluster exactly once")
        source_by_id = {item["cluster_id"]: item for item in clusters}
        normalized_by_id: dict[str, JSONDict] = {}
        allowed = {"abduction", "analogy", "deduction", None}
        for item in items:
            if (not isinstance(item, dict)
                    or set(item) != {"cluster_id", "weakpoints", "rejected_relation_ids"}
                    or item.get("cluster_id") not in source_by_id
                    or item["cluster_id"] in normalized_by_id):
                raise ValueError("weakpoint cluster result has invalid fields or ID")
            source = source_by_id[item["cluster_id"]]
            relation_ids = {
                relation["relation_id"] for relation in source.get("candidate_relations", [])
                if isinstance(relation, dict) and isinstance(relation.get("relation_id"), str)
            }
            node_ids = {
                node["claim_id"] for node in source.get("claims", [])
                if isinstance(node, dict) and isinstance(node.get("claim_id"), str)
            }
            weakpoints, rejected = item["weakpoints"], item["rejected_relation_ids"]
            if (not isinstance(weakpoints, list) or not isinstance(rejected, list)
                    or any(not isinstance(key, str) for key in rejected)):
                raise ValueError("weakpoint cluster requires weakpoints and rejected relation IDs")
            consumed: list[str] = list(rejected)
            normalized_weakpoints: list[JSONDict] = []
            for weakpoint in weakpoints:
                if not isinstance(weakpoint, dict) or set(weakpoint) != {
                    "member_relation_ids", "evidence_claim_ids", "target_claim_id", "reasoning_type", "expression",
                }:
                    raise ValueError("normalized weakpoint has invalid fields")
                members = weakpoint["member_relation_ids"]
                evidence = weakpoint["evidence_claim_ids"]
                targets = weakpoint["target_claim_id"]
                if any(not isinstance(values, list) or not values or len(values) != len(set(values))
                       or any(not isinstance(key, str) or not key for key in values)
                       for values in (members, evidence, targets)):
                    raise ValueError("normalized weakpoint IDs must be non-empty and unique")
                reasoning_type = weakpoint["reasoning_type"]
                if reasoning_type == "null":
                    reasoning_type = None
                expression = weakpoint["expression"]
                if (reasoning_type not in allowed or not set(evidence + targets) <= node_ids
                        or set(evidence) & set(targets) or not set(members) <= relation_ids
                        or not isinstance(expression, str) or not expression.strip()):
                    raise ValueError("normalized weakpoint has unsupported endpoints, type, or expression")
                if any(f"[{key}]" not in expression for key in [*evidence, *targets]):
                    raise ValueError("normalized weakpoint expression must cite every endpoint")
                consumed.extend(members)
                normalized_weakpoints.append({**weakpoint, "reasoning_type": reasoning_type})
            if len(consumed) != len(set(consumed)) or set(consumed) != relation_ids:
                raise ValueError("every cluster relation must be consumed or rejected exactly once")
            normalized_by_id[item["cluster_id"]] = {
                "cluster_id": item["cluster_id"], "weakpoints": normalized_weakpoints,
                "rejected_relation_ids": rejected,
            }
        if set(normalized_by_id) != set(source_by_id):
            raise ValueError("DeepSeek weakpoint cluster response omitted a cluster")
        return ToolCallResponse(
            request.call_id, "succeeded", {"response": raw},
            {"clusters": [normalized_by_id[item] for item in cluster_ids]},
            metadata={"cluster_count": len(clusters), "model": self.model},
        )

    @staticmethod
    def _equivalent_claim_prompt(observations: list[JSONDict], candidate: JSONDict) -> str:
        paragraphs = "\n\n".join(
            f"[{item['anchor_id']}] {item['text']}"
            for item in candidate.get("paragraphs", [])
            if isinstance(item, dict)
        )
        return (
            "Return JSON only. For every supplied experimental observation O, write exactly one English phenomenon "
            "claim E with the same truth conditions. E must be usable as a consequence of a hypothesis: state the "
            "observable result under the same setting, treatment or object, optional baseline, measure, comparison "
            "direction, result, conditions, and uncertainty, but remove report-event wording such as 'the experiment "
            "showed', 'we observed', or 'the paper reports'. Do not turn a measured result into a latent, population, "
            "causal, or more general claim. Do not add, omit, merge, split, explain, or reinterpret facts. Use only the "
            "supplied observations and paper paragraphs. Even when O already omits report-event wording, express E with "
            "different surface wording while preserving exactly the same truth conditions; never copy O's content "
            "verbatim. Return every observation_key exactly once and no other fields. "
            "Return {\"equivalent_claims\":[{\"observation_key\":\"...\",\"content\":\"...\"}]}.\n\n"
            f"Observations:\n{json.dumps(observations, ensure_ascii=False)}\n\nPaper paragraphs:\n{paragraphs}"
        )

    @staticmethod
    def _normalize_equivalent_claims(content: str, observations: list[JSONDict]) -> list[JSONDict]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek equivalent-claim response is not valid JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"equivalent_claims"}:
            raise ValueError("DeepSeek equivalent-claim response must contain only equivalent_claims")
        items = payload["equivalent_claims"]
        expected = {
            item["_extraction_key"]: item
            for item in observations
            if isinstance(item, dict) and isinstance(item.get("_extraction_key"), str)
        }
        if not isinstance(items, list) or len(items) != len(expected):
            raise ValueError("DeepSeek must return exactly one equivalent claim per observation")
        normalized: dict[str, JSONDict] = {}
        for item in items:
            if not isinstance(item, dict) or set(item) != {"observation_key", "content"}:
                raise ValueError("equivalent claims must contain only observation_key and content")
            key, equivalent_content = item["observation_key"], item["content"]
            if not isinstance(key, str) or key not in expected or key in normalized:
                raise ValueError("equivalent claim has an unknown or duplicate observation_key")
            if not isinstance(equivalent_content, str) or not equivalent_content.strip():
                raise ValueError("equivalent claim content must be non-empty")
            observation = expected[key]
            if equivalent_content.strip() == str(observation["content"]).strip():
                raise ValueError("equivalent claim must not copy the experimental observation verbatim")
            normalized[key] = {
                "observation_key": key,
                "content": equivalent_content.strip(),
                "paragraph_anchor_ids": list(observation["paragraph_anchor_ids"]),
            }
        if set(normalized) != set(expected):
            raise ValueError("DeepSeek omitted an equivalent claim")
        return [normalized[item["_extraction_key"]] for item in observations]

    @staticmethod
    def _relation_groups(phenomena: list[JSONDict], candidate: JSONDict) -> list[JSONDict]:
        """Offer each local target one set of eligible phenomenon claims."""
        groups: list[JSONDict] = []
        for item in candidate.get("related_claims", []):
            if (not isinstance(item, dict) or item.get("type") != "claim"
                    or not isinstance(item.get("id"), str) or not isinstance(item.get("content"), str)):
                continue
            evidence_candidates = [
                {
                    "phenomenon_key": phenomenon["observation_key"],
                    "phenomenon": phenomenon["content"],
                }
                for phenomenon in phenomena
                if _anchors_share_local_context(
                    phenomenon["paragraph_anchor_ids"], item.get("source_anchor_ids", [])
                )
            ]
            if evidence_candidates:
                groups.append({
                    "group_id": f"group_{len(groups) + 1}",
                    "target_claim_id": item["id"],
                    "target_claim_or_note": item["content"],
                    "evidence_candidates": evidence_candidates,
                })
        return groups

    @staticmethod
    def _relation_prompt(groups: list[JSONDict]) -> str:
        return (
            "Return JSON only. Classify every supplied target-centered candidate group exactly once; do not add or omit groups. "
            "Each group contains one target claim and the non-experimental phenomenon claims E that share a source anchor or local experiment context with it. "
            "Use only the supplied text and IDs; do not invent premises, mechanisms, relations, or IDs. "
            "relation_type must be exactly one of evidence, conjunction, disjunction, negation, implication, equivalence, contradiction, none. "
            "Use fixed logical expressions only: 且 / 和 / 与 for conjunction, 或 for disjunction, 非 for negation, "
            "推出 for implication direction, 等价 for equivalence, and 矛盾 or 不可同时成立 for contradiction. "
            "When phenomenon claims support a general claim, default to evidence, never implication: do not write '[E:...] 推出 [target]'. "
            "For each group, return relations as an array. A relation may select one or more eligible phenomenon keys. "
            "Select multiple phenomena only when they jointly support the target and removing one changes the relation's meaning; otherwise use a single phenomenon. "
            "For evidence, write '[E:key] 是 [target] 的例子或证据' or '([E:key1] 和 [E:key2]) 是 [target] 的例子或证据'. "
            "Every non-none expression must cite every selected [E:key] and the group's exact [target_claim_id]. "
            "Use an empty relations array if no relation is supported. "
            "Return {\"classifications\":[{\"group_id\":\"group_1\",\"relations\":[{\"source_phenomenon_keys\":[\"e_1\",\"e_2\"],"
            "\"relation_type\":\"evidence\",\"expression\":\"([E:e_1] 和 [E:e_2]) 是 [claim_7] 的例子或证据\"}]}]}.\nGroups:\n"
            + json.dumps(groups, ensure_ascii=False)
        )

    @staticmethod
    def _normalize_group_relations(content: str, groups: list[JSONDict]) -> list[JSONDict]:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek relation response is not valid JSON") from exc
        classifications = payload.get("classifications") if isinstance(payload, dict) else None
        if not isinstance(classifications, list) or len(classifications) != len(groups):
            raise ValueError("DeepSeek relation response must classify every candidate group exactly once")
        by_id = {group["group_id"]: group for group in groups}
        seen: set[str] = set()
        relations: list[JSONDict] = []
        allowed = {"evidence", "conjunction", "disjunction", "negation", "implication", "equivalence", "contradiction", "none"}
        for item in classifications:
            if not isinstance(item, dict) or set(item) != {"group_id", "relations"} or item.get("group_id") not in by_id or item["group_id"] in seen:
                raise ValueError("DeepSeek relation response has invalid or duplicate group_id")
            seen.add(item["group_id"])
            group_relations = item["relations"]
            if not isinstance(group_relations, list):
                raise ValueError("DeepSeek group relations must be an array")
            group = by_id[item["group_id"]]
            eligible = {candidate["phenomenon_key"] for candidate in group["evidence_candidates"]}
            relation_sets: set[tuple[str, ...]] = set()
            for relation in group_relations:
                if not isinstance(relation, dict) or set(relation) != {"source_phenomenon_keys", "relation_type", "expression"}:
                    raise ValueError("DeepSeek group relation has an invalid shape")
                source_keys = relation["source_phenomenon_keys"]
                relation_type = relation["relation_type"]
                expression = relation["expression"]
                if not isinstance(source_keys, list) or not source_keys or not all(isinstance(key, str) for key in source_keys):
                    raise ValueError("DeepSeek group relation requires source phenomenon keys")
                if len(source_keys) != len(set(source_keys)) or not set(source_keys) <= eligible:
                    raise ValueError("DeepSeek group relation references an ineligible phenomenon")
                if relation_type not in allowed - {"none"}:
                    raise ValueError("DeepSeek relation response has an unsupported relation_type")
                if not isinstance(expression, str) or f"[{group['target_claim_id']}]" not in expression:
                    raise ValueError("DeepSeek relation expression must cite its target claim")
                if any(f"[E:{key}]" not in expression for key in source_keys):
                    raise ValueError("DeepSeek relation expression must cite every selected phenomenon")
                source_signature = tuple(sorted(source_keys))
                if source_signature in relation_sets:
                    raise ValueError("DeepSeek group relation duplicates an evidence set")
                relation_sets.add(source_signature)
                relations.append({"phenomenon_keys": source_keys, "claim_id": group["target_claim_id"], "expression": expression})
        if seen != set(by_id):
            raise ValueError("DeepSeek relation response omitted a candidate group")
        return relations

    @staticmethod
    def _claims_for_candidate(claims: Any, candidate: JSONDict) -> list[JSONDict]:
        if not isinstance(claims, list):
            return []
        anchor_ids = {
            item.get("anchor_id")
            for item in candidate.get("paragraphs", [])
            if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
        }
        return [
            item
            for item in claims
            if isinstance(item, dict)
            and isinstance(item.get("source_anchor_ids"), list)
            and bool(anchor_ids & set(item["source_anchor_ids"]))
        ]

    @staticmethod
    def _prompt(
        candidates: list[JSONDict],
        selected_claims: Any,
        review_advice: Any,
        frozen_claims: Any,
    ) -> str:
        evidence = []
        for candidate in candidates:
            allowed_anchor_ids = [
                item["anchor_id"]
                for item in candidate.get("paragraphs", [])
                if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
            ]
            focus_anchor_ids = candidate.get("focus_anchor_ids", allowed_anchor_ids)
            paragraphs = "\n\n".join(
                f"[{item['anchor_id']}] {item['text']}"
                for item in candidate.get("paragraphs", [])
                if isinstance(item, dict)
            )
            evidence.append(
                f"Candidate {candidate.get('candidate_id')} / image {candidate.get('image_name')}. "
                f"Allowed paragraph_anchor_ids: {json.dumps(allowed_anchor_ids)}. "
                f"Focus paragraph_anchor_ids: {json.dumps(focus_anchor_ids)}.\n{paragraphs}"
            )
        rewrite = ""
        if isinstance(selected_claims, list):
            rewrite = (
                "\nThis is a targeted human-requested rewrite. Return exactly one corrected claim for "
                "each supplied id, retain each id verbatim, and do not return any unselected claim. "
                "Frozen claims are immutable context: do not rewrite them, and ensure the corrected claims "
                "remain distinct from them.\n"
                f"Selected claims: {json.dumps(selected_claims, ensure_ascii=False)}\n"
                f"Frozen claims: {json.dumps(frozen_claims or [], ensure_ascii=False)}\n"
                f"Human advice: {review_advice}\n"
            )
        return (
            "Return JSON only. Use only the supplied paper paragraphs. Do not use outside knowledge or infer missing facts. "
            "Extract each evidence-supported experimental observation as S/A/(optional B)/M/R/(optional U): S is setting "
            "or scope; A is the treatment, method, intervention, or experimental object; B is an optional "
            "baseline or control and may be omitted when the supplied evidence states an observation without one; "
            "M is the measure; "
            "R is the observed numerical or directional comparison result; U is optional and contains only an explicitly reported "
            "uncertainty value directly corresponding to R. Omit B or U when the evidence does not report it; never emit "
            "'baseline not reported', 'uncertainty not reported', or another absence placeholder. Use English. "
            "For each claim, directly write content as one fluent, complete English sentence. You may improve grammar and flow, "
            "but must preserve the meaning, scope, comparison direction, conditions, result, and any reported uncertainty of "
            "S/A/B/M/R and optional U. Do not mechanically concatenate a fixed template. Do not add, omit, merge, generalize, "
            "narrow, reinterpret, or move information between fields. "
            "For one image, produce one claim per distinct evidence-supported experimental unit, not a Cartesian "
            "expansion of treatments and conditions. Group items only when the paper reports them as one inseparable "
            "unit; otherwise represent separately reported treatments, conditions, or comparisons as separate claims, "
            "even if some fields coincide. Split whenever combining items would obscure a reported distinction. Do not "
            "invent distinctions unsupported by the evidence. "
            "Return every distinct, explicitly supported experimental result; do not omit a broad result merely "
            "because more specific conditional results are also returned. "
            "Each claim must cite one or more paragraph_anchor_ids from its own candidate and include its "
            "candidate_id. Cite only IDs in that candidate's Allowed paragraph_anchor_ids list; never invent "
            "an anchor ID or cite an ID from another candidate. Each claim must cite at least one ID from its "
            "Focus paragraph_anchor_ids list; use expanded context only to complete the experiment associated "
            "with that focus evidence. The expanded window can contain related work, background, or results for a "
            "different figure. Never extract a result attributed to cited studies or prior work. Extract only a result "
            "that the supplied text explicitly attributes to this paper's experiment represented by the candidate "
            "image or its caption; if that result is not yet present, return insufficient_context. "
            "Do not assess relations between observations and existing claims or notes in this call. "
            "If evidence is insufficient, return "
            "{\"status\":\"insufficient_context\",\"needed_context\":\"...\"}; never guess. Otherwise "
            "return {\"status\":\"claims_extracted\",\"claims\":[{\"candidate_id\":\"...\","
            "\"S\":\"...\",\"A\":\"...\",\"M\":\"...\","
            "\"R\":\"...\",\"content\":\"...\"," 
            "\"paragraph_anchor_ids\":[\"...\"]}]}."
            + rewrite
            + "\n\nEvidence:\n"
            + "\n\n".join(evidence)
        )

    @staticmethod
    def _normalize(content: str, candidates: list[JSONDict], selected_claims: Any = None) -> JSONDict:
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek response is not valid JSON") from exc
        if not isinstance(result, dict):
            raise ValueError("DeepSeek response must be a JSON object")
        if result.get("status") == "insufficient_context":
            return {
                "status": "insufficient_context",
                "needed_context": str(result.get("needed_context", "More paper context is required.")),
            }
        claims = result.get("claims")
        if result.get("status") != "claims_extracted" or not isinstance(claims, list):
            raise ValueError("DeepSeek response must declare claims_extracted with a claims array")
        if "relations" in result and result["relations"] not in ([], None):
            raise ValueError("observation extraction must not return relations")
        candidate_order = {
            candidate["candidate_id"]: index
            for index, candidate in enumerate(candidates)
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
        }
        anchors_by_candidate = {
            candidate["candidate_id"]: {
                item["anchor_id"]
                for item in candidate.get("paragraphs", [])
                if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
            }
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
        }
        focus_anchors_by_candidate = {
            candidate["candidate_id"]: set(candidate.get("focus_anchor_ids", ()))
            for candidate in candidates
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str)
        }
        selected_ids = {
            item["id"]
            for item in selected_claims or []
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        normalized: list[JSONDict] = []
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                raise ValueError(f"claim {index} must be an object")
            candidate_id = claim.get("candidate_id")
            if candidate_id not in candidate_order:
                raise ValueError(f"claim {index} cites an unknown candidate_id")
            fields = {name: claim.get(name) for name in ("S", "A", "M", "R")}
            if not all(isinstance(value, str) and value.strip() for value in fields.values()):
                raise ValueError(f"claim {index} must provide non-empty S/A/M/R strings")
            baseline = claim.get("B")
            if isinstance(baseline, str) and not baseline.strip():
                baseline = None
            if baseline is not None and (not isinstance(baseline, str) or not baseline.strip()):
                raise ValueError(f"claim {index} B must be omitted or a non-empty string")
            if isinstance(baseline, str) and baseline.strip().casefold() == "baseline not reported":
                raise ValueError(f"claim {index} must omit B instead of using an absence placeholder")
            uncertainty = claim.get("U")
            if isinstance(uncertainty, str) and not uncertainty.strip():
                uncertainty = None
            if uncertainty is not None and (not isinstance(uncertainty, str) or not uncertainty.strip()):
                raise ValueError(f"claim {index} U must be omitted or a non-empty string")
            if isinstance(uncertainty, str) and uncertainty.strip().casefold() == "uncertainty not reported":
                raise ValueError(f"claim {index} must omit U instead of using an absence placeholder")
            claim_content = claim.get("content")
            if not isinstance(claim_content, str) or not claim_content.strip():
                raise ValueError(f"claim {index} must provide non-empty content")
            anchor_ids = claim.get("paragraph_anchor_ids")
            if (
                not isinstance(anchor_ids, list)
                or not anchor_ids
                or not all(isinstance(value, str) for value in anchor_ids)
                or not set(anchor_ids) <= anchors_by_candidate[candidate_id]
            ):
                raise ValueError(f"claim {index} cites paragraph anchors outside its candidate")
            if focus_anchors_by_candidate[candidate_id] and not (
                set(anchor_ids) & focus_anchors_by_candidate[candidate_id]
            ):
                raise ValueError(f"claim {index} cites no focus paragraph anchor for its candidate")
            claim_id = claim.get("id")
            if selected_ids and claim_id not in selected_ids:
                raise ValueError(f"rewritten claim {index} must retain a selected id")
            normalized.append(
                {
                    "_extraction_key": f"{candidate_id}:{index}",
                    **({"id": claim_id} if isinstance(claim_id, str) else {}),
                    "candidate_id": candidate_id,
                    **{name: str(value).strip() for name, value in fields.items()},
                    **({"B": baseline.strip()} if isinstance(baseline, str) else {}),
                    **({"U": uncertainty.strip()} if isinstance(uncertainty, str) else {}),
                    "content": claim_content.strip(),
                    "paragraph_anchor_ids": list(dict.fromkeys(anchor_ids)),
                }
            )
        if selected_ids and {item.get("id") for item in normalized} != selected_ids:
            raise ValueError("targeted rewrite must return exactly the selected claim ids")
        normalized.sort(
            key=lambda item: (
                candidate_order[item["candidate_id"]],
                item["paragraph_anchor_ids"],
                *(str(item.get(name, "")).casefold() for name in ("S", "A", "B", "M", "R")),
            )
        )
        return {"status": "claims_extracted", "claims": normalized, "relations": []}


def _latest_formalization(context: StageContext) -> tuple[Any, JSONDict]:
    refs = context.find_all("formalization")
    if len(refs) != 1:
        raise ValueError(f"Step 2 requires exactly one current formalization, found {len(refs)}")
    return refs[0], read_json(context.artifact_path(refs[0]))


def _paper_text(context: StageContext) -> str:
    bundles = context.find_all(INPUT_BUNDLE_KIND)
    if bundles:
        if len(bundles) != 1:
            raise ValueError("Step 2 requires at most one input.bundle")
        value = read_json(context.artifact_path(bundles[0]))
        sources = value.get("sources")
        entries = sources.get(PAPER_TEXT_KIND) if isinstance(sources, dict) else None
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise ValueError("input.bundle must contain exactly one source.paper_text")
        paper = entries[0].get("value")
    else:
        paper_ref = context.require_one(PAPER_TEXT_KIND)
        paper = context.artifact_path(paper_ref).read_text(encoding="utf-8-sig")
    if not isinstance(paper, str) or not paper.strip():
        raise ValueError("source.paper_text value must be non-empty text")
    return paper


def _experiment_candidates(paragraphs: list[JSONDict], radius: int = 1) -> list[JSONDict]:
    images: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for paragraph in paragraphs:
        for match in _IMAGE_PATTERN.finditer(paragraph["text"]):
            raw_name = match.group(1).strip("<>")
            path_name = PurePosixPath(raw_name.split("?", 1)[0].split("#", 1)[0]).name
            image_name = path_name or raw_name
            key = image_name.casefold()
            if key not in seen_names:
                seen_names.add(key)
                images.append((image_name, PurePosixPath(image_name).stem))
    candidates: list[JSONDict] = []
    for sequence, (image_name, stem) in enumerate(images, 1):
        aliases = _figure_aliases(stem) | _figure_aliases(image_name)
        matches = [
            index
            for index, paragraph in enumerate(paragraphs)
            if any(alias in re.sub(r"[^a-z0-9]", "", paragraph["text"].lower()) for alias in aliases)
        ]
        indexes = sorted(
            {
                neighbor
                for index in matches
                for neighbor in range(max(0, index - radius), min(len(paragraphs), index + radius + 1))
            }
        )
        candidates.append(
            {
                "candidate_id": f"image_{sequence:02d}_{re.sub(r'[^a-z0-9]+', '_', stem.lower()).strip('_') or 'unnamed'}",
                "image_name": image_name,
                "paragraphs": [
                    {
                        "anchor_id": paragraphs[index]["anchor_id"],
                        "tag": paragraphs[index]["tag"],
                        "text": paragraphs[index]["text"],
                    }
                    for index in indexes
                ],
            }
        )
    return candidates


def _tool_call(
    context: StageContext,
    candidates: list[JSONDict],
    *,
    call_number: int,
    selected_claims: list[JSONDict] | None = None,
    review_advice: str | None = None,
    frozen_claims: list[JSONDict] | None = None,
) -> tuple[JSONDict, ArtifactDraft]:
    spec = context.options.get("tool_plugin")
    if not isinstance(spec, str):
        raise ValueError("Step 2 requires options.tool_plugin in module:object form")
    tool = instantiate(spec)
    if not isinstance(tool, DomainTool):
        raise TypeError(f"{spec} does not implement DomainTool")
    request = ToolCallRequest(
        call_id=f"semantic_step_2_{context.run_id}_{context.attempt}_{call_number}",
        tool_name=tool.name,
        tool_version=tool.version,
        operation="extract_observation_claims" if selected_claims is None else "revise_observation_claims",
        inputs=[
            {"artifact_id": ref.artifact_id, "kind": ref.kind, "sha256": ref.sha256}
            for ref in context.inputs
        ],
        parameters={
            "experiment_candidates": candidates,
            **({"selected_claims": selected_claims} if selected_claims is not None else {}),
            **({"review_advice": review_advice} if review_advice is not None else {}),
            **({"frozen_claims": frozen_claims} if frozen_claims is not None else {}),
        },
    )
    try:
        response = tool.invoke(request)
        if not isinstance(response, ToolCallResponse):
            raise TypeError(f"{spec} returned {type(response).__name__}, not ToolCallResponse")
        validate_tool_response(request, response)
    except Exception as exc:
        response = ToolCallResponse(
            request.call_id,
            "failed",
            None,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    response_path = context.work_dir / f"semantic_step_2_tool_response_{call_number}.json"
    atomic_write_json(response_path, {"request": request.to_dict(), "response": response.to_dict()})
    draft = ArtifactDraft(
        response_path,
        "tool.semantic_review.response",
        "application/json",
        {
            "schema_version": "1.0.0",
            "step": 2,
            "tool_call_id": request.call_id,
            "tool_name": tool.name,
            "tool_version": tool.version,
            "status": response.status,
        },
    )
    if response.status != "succeeded" or not isinstance(response.normalized, dict):
        raise ValueError(str((response.error or {}).get("message", "Step 2 semantic tool failed")))
    return response.normalized, draft


def _tool_audit_drafts(context: StageContext) -> list[ArtifactDraft]:
    drafts: list[ArtifactDraft] = []
    for path in sorted(context.work_dir.glob("semantic_step_2_tool_response_*.json")):
        payload = read_json(path)
        request = payload.get("request", {})
        response = payload.get("response", {})
        drafts.append(
            ArtifactDraft(
                path,
                "tool.semantic_review.response",
                "application/json",
                {
                    "schema_version": "1.0.0",
                    "step": 2,
                    "tool_call_id": request.get("call_id", "unknown"),
                    "tool_name": request.get("tool_name", "unknown"),
                    "tool_version": request.get("tool_version", "unknown"),
                    "status": response.get("status", "failed"),
                },
            )
        )
    return drafts


def _extract_with_expansion(
    context: StageContext,
    paragraphs: list[JSONDict],
    document: JSONDict,
    *,
    selected_claims: list[JSONDict] | None = None,
    review_advice: str | None = None,
    frozen_claims: list[JSONDict] | None = None,
    first_call_number: int = 1,
) -> tuple[list[JSONDict], list[JSONDict], list[JSONDict], list[ArtifactDraft]]:
    drafts: list[ArtifactDraft] = []
    initial_candidates = _experiment_candidates(paragraphs)
    if not initial_candidates:
        return [], [], [], drafts
    if selected_claims is not None:
        selected_anchor_ids = {
            anchor_id
            for claim in selected_claims
            if isinstance(claim, dict)
            for anchor_id in claim.get("source_anchor_ids", [])
            if isinstance(anchor_id, str)
        }
        initial_candidates = [
            candidate
            for candidate in initial_candidates
            if selected_anchor_ids
            & {
                item["anchor_id"]
                for item in candidate["paragraphs"]
                if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
            }
        ]
        if not initial_candidates:
            raise ValueError("selected claims do not map to any experiment candidate")
    claims: list[JSONDict] = []
    equivalent_claims: list[JSONDict] = []
    relations: list[JSONDict] = []
    for initial_candidate in initial_candidates:
        candidate_id = initial_candidate["candidate_id"]
        focus_anchor_ids = [
            item["anchor_id"]
            for item in initial_candidate["paragraphs"]
            if isinstance(item, dict) and isinstance(item.get("anchor_id"), str)
        ]
        previous_signature: tuple[str, ...] | None = None
        attempts = 0
        # Start from the mechanical image ±1 window, then widen through ±10.
        for radius in range(1, 11):
            candidates = _experiment_candidates(paragraphs, radius)
            candidate = next((item for item in candidates if item["candidate_id"] == candidate_id), None)
            if candidate is None:
                raise ValueError(f"experiment candidate disappeared while expanding context: {candidate_id}")
            candidate["focus_anchor_ids"] = focus_anchor_ids
            candidate_anchor_ids = [item["anchor_id"] for item in candidate["paragraphs"]]
            candidate["related_claims"] = [
                {"id": knowledge_id, "type": "claim", "content": knowledge["content"]["canonical"], "source_anchor_ids": knowledge.get("source_anchor_ids", [])}
                for knowledge_id, knowledge in document["knowledges"].items()
                if knowledge.get("type") == "claim"
                and _anchors_share_local_context(candidate_anchor_ids, knowledge.get("source_anchor_ids", []))
            ]
            signature = tuple(item["anchor_id"] for item in candidate["paragraphs"])
            if signature == previous_signature:
                continue
            previous_signature = signature
            attempts += 1
            normalized, draft = _tool_call(
                context,
                [candidate],
                call_number=first_call_number + len(drafts),
                selected_claims=selected_claims,
                review_advice=review_advice,
                frozen_claims=frozen_claims,
            )
            drafts.append(draft)
            if normalized.get("status") == "claims_extracted":
                extracted = normalized.get("claims")
                if not isinstance(extracted, list) or not all(isinstance(item, dict) for item in extracted):
                    raise ValueError("semantic tool normalized claims must be objects")
                equivalents = normalized.get("equivalent_claims")
                if not isinstance(equivalents, list) or not all(isinstance(item, dict) for item in equivalents):
                    raise ValueError("semantic tool normalized equivalent claims must be objects")
                claims.extend(extracted)
                equivalent_claims.extend(equivalents)
                extracted_relations = normalized.get("relations", [])
                if not isinstance(extracted_relations, list) or not all(isinstance(item, dict) for item in extracted_relations):
                    raise ValueError("semantic tool normalized relations must be objects")
                relations.extend({**item, "relation_context_id": candidate_id} for item in extracted_relations)
                break
            if normalized.get("status") != "insufficient_context":
                raise ValueError("semantic tool returned an unsupported status")
        else:
            raise ValueError(
                f"Step 2 marked {candidate_id} insufficient_context after the initial window and "
                f"nine mechanical expansions (+1 through +9; {attempts} distinct windows); terminating run"
            )
    return claims, equivalent_claims, relations, drafts


def _observation_items(claims: list[JSONDict]) -> tuple[dict[str, JSONDict], dict[str, str]]:
    ids = {str(claim.get("_extraction_key", f"generated:{index}")): f"claim_O{index:02d}" for index, claim in enumerate(claims, 1)}
    return {
        f"claim_O{index:02d}": {
            "type": "observation_claim",
            "content": {"canonical": claim["content"]},
            "source_anchor_ids": list(claim["paragraph_anchor_ids"]),
        }
        for index, claim in enumerate(claims, 1)
    }, ids


def _equivalent_items(
    equivalent_claims: list[JSONDict],
    observation_ids: dict[str, str],
) -> tuple[dict[str, JSONDict], dict[str, str], list[JSONDict]]:
    if len(equivalent_claims) != len(observation_ids):
        raise ValueError("Step 2 requires exactly one equivalent claim per observation")
    by_key: dict[str, JSONDict] = {}
    for item in equivalent_claims:
        key = item.get("observation_key")
        if not isinstance(key, str) or key not in observation_ids or key in by_key:
            raise ValueError("equivalent claim references an unknown or duplicate observation")
        by_key[key] = item
    if set(by_key) != set(observation_ids):
        raise ValueError("equivalent claims do not cover every observation")
    phenomenon_ids: dict[str, str] = {}
    phenomena: dict[str, JSONDict] = {}
    operators: list[JSONDict] = []
    for key, observation_id in observation_ids.items():
        suffix = observation_id.removeprefix("claim_O")
        phenomenon_id = f"claim_E{suffix}"
        phenomenon_ids[key] = phenomenon_id
        phenomena[phenomenon_id] = {
            "type": "claim",
            "content": {"canonical": by_key[key]["content"]},
            "source_anchor_ids": list(by_key[key]["paragraph_anchor_ids"]),
        }
        operators.append({
            "id": f"operator_equivalence_E{suffix}_O{suffix}",
            "type": "equivalence",
            "variables": [phenomenon_id, observation_id],
        })
    return phenomena, phenomenon_ids, operators


def _relation_items(relations: list[JSONDict], phenomenon_ids: dict[str, str], document: JSONDict) -> list[JSONDict]:
    links = list(document["workflow"]["non_reasoning_links"])
    existing = {link["id"] for link in links}
    for index, relation in enumerate(relations, 1):
        phenomenon_keys = relation.get("phenomenon_keys")
        claim_id = relation.get("claim_id")
        if not isinstance(phenomenon_keys, list) or not phenomenon_keys or not isinstance(claim_id, str):
            raise ValueError("relation references an unknown phenomenon or claim")
        phenomenon_id_by_key = {key: phenomenon_ids.get(key) for key in phenomenon_keys}
        if any(not isinstance(key, str) or phenomenon_id is None for key, phenomenon_id in phenomenon_id_by_key.items()):
            raise ValueError("relation references an unknown phenomenon or claim")
        link_id = f"relation_step2_{index}"
        if link_id in existing:
            raise ValueError(f"duplicate relation id: {link_id}")
        expression = relation["expression"]
        relation_context_id = relation.get("relation_context_id")
        if not isinstance(relation_context_id, str) or not relation_context_id:
            raise ValueError("Step 2 relation requires an experiment context ID")
        for key, phenomenon_id in phenomenon_id_by_key.items():
            expression = expression.replace(f"[E:{key}]", f"[{phenomenon_id}]")
        if "[E:" in expression:
            raise ValueError("relation expression references an unknown phenomenon")
        links.append({
            "id": link_id, "link_type": "imported_relation", "sources": list(phenomenon_id_by_key.values()), "target": claim_id,
            "reasoning": False, "metadata": {"relation": {
                "expression": expression, "relation_context_id": relation_context_id,
            }},
        })
        existing.add(link_id)
    return links


def _revision(
    document: JSONDict,
    context: StageContext,
    *,
    revision_id: str,
    review_status: str,
    modified: list[str],
) -> None:
    previous_revision = document["revision"]
    document["revision"] = {
        "revision_id": revision_id,
        "supersedes": previous_revision.get("revision_id"),
        "parent_hash": previous_revision.get("content_hash"),
        "content_hash": "",
    }
    # Artifact identity plus revision.parent_hash is the history chain.  The
    # schema-1.1 field remains present only as an empty compatibility field.
    document["workflow"]["revisions"] = []
    document["revision"]["content_hash"] = content_hash(document)


def _with_extra_artifacts(result: StageResult, drafts: list[ArtifactDraft]) -> StageResult:
    return StageResult(
        result.status,
        [*result.artifacts, *drafts],
        list(result.findings),
        dict(result.metadata),
    )


class Step2ExtractObservationsPlugin:
    """Mechanically anchor Knowledge, extract observations, and write relations."""

    def run(self, context: StageContext) -> StageResult:
        try:
            _, source = _latest_formalization(context)
            document = copy.deepcopy(source)
            paper = _paper_text(context)
            paragraphs = _paper_paragraphs(paper.splitlines())
            modified: list[str] = []
            for knowledge_id, knowledge in document["knowledges"].items():
                best = _best_paragraph_anchor(knowledge["content"]["canonical"], paragraphs)
                if best is None:
                    continue
                source_ids = knowledge.setdefault("source_anchor_ids", [])
                if best["anchor_id"] not in source_ids:
                    source_ids.append(best["anchor_id"])
                    modified.append(knowledge_id)
            claims, equivalent_claims, relations, drafts = _extract_with_expansion(context, paragraphs, document)
            observations, observation_ids = _observation_items(claims)
            phenomena, phenomenon_ids, equivalence_operators = _equivalent_items(equivalent_claims, observation_ids)
            duplicate_ids = set(document["knowledges"]) & (set(observations) | set(phenomena))
            if duplicate_ids:
                raise ValueError(f"Step 2 claim ids already exist: {sorted(duplicate_ids)}")
            existing_operator_ids = {item.get("id") for item in document["graph"]["operators"]}
            duplicate_operator_ids = existing_operator_ids & {item["id"] for item in equivalence_operators}
            if duplicate_operator_ids:
                raise ValueError(f"Step 2 operator ids already exist: {sorted(duplicate_operator_ids)}")
            document["knowledges"].update(observations)
            document["knowledges"].update(phenomena)
            document["graph"]["nodes"].extend(
                knowledge_id
                for knowledge_id in [*observations, *phenomena]
                if knowledge_id not in document["graph"]["nodes"]
            )
            document["graph"]["operators"].extend(equivalence_operators)
            document["workflow"]["non_reasoning_links"] = _relation_items(relations, phenomenon_ids, document)
            _revision(
                document,
                context,
                revision_id=f"revision_{context.run_id}_step_2",
                review_status="automated",
                modified=[*modified, *observations, *phenomena, *(item["id"] for item in equivalence_operators)],
            )
            emitted = emit_formalization(context, document, step=2, step_name=STEP_NAME)
            return _with_extra_artifacts(emitted, drafts)
        except Exception as exc:
            return StageResult(
                "failed",
                artifacts=_tool_audit_drafts(context),
                findings=[Finding("STEP2_EXTRACTION_FAILED", "error", str(exc))],
                metadata={"step": 2},
            )
