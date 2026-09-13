"""Merge viewer: compose the input Paper Packages with the integration delta.

The integration Package references Paper knowledge by external QID and adds only
its own cross-paper relations plus candidate K.  The projection splits into three
viewer layers so the merge delta can be isolated:

* ``claims``     — Paper claims that the merge did NOT touch (context).
* ``operators``  — package-internal reasoning edges (each paper's own graph).
* ``weakpoints`` — the merge delta: candidate K, the Paper claims its new
  relations touch, and those new cross-paper relations.

Hide ``claims`` and ``operators`` to see only the delta; hide ``weakpoints`` to
see the untouched Paper graph.  (The shared viewer only recognises the layer ids
``claims``/``operators``/``weakpoints``/``strategies``/``registry``, so the delta
layer reuses ``weakpoints``.)

Compiler-internal helper / interface nodes (``__*``, ``*_result``, AltExp,
``step4_weakpoint_*``, ``helper_relation_*``) are plumbing and never rendered.
Node labels are shortened to ``{paper-letter}:{local-label}`` / ``K:{n}``.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from pipeline_harness.models import ArtifactRef, JSONDict
from pipeline_harness.store import RunStore
from pipeline_harness.view.model import ViewDocument

_INTEGRATION_STAGE = "step4_formalize_integration"
_STEP = 4
_DELTA_LAYER = "weakpoints"
_CONTEXT_LAYER = "claims"
_PAPER_EDGE_LAYER = "operators"
_HELPER_RE = re.compile(
    r"(^|::)__|helper[_-]?relation|operator[_-]?result|disjunction[_-]?result|"
    r"alternative[_-]?explanation|step4_weakpoint_",
    re.I,
)


def _is_helper(knowledge_id: str) -> bool:
    return bool(_HELPER_RE.search(str(knowledge_id)))


def _short_label(knowledge_id: str, letter: str) -> str:
    local = knowledge_id.split("::", 1)[1] if "::" in knowledge_id else knowledge_id
    if local.startswith("claim_"):
        local = local[len("claim_"):]
    return f"{letter}:{local}"


def _read_json(store: RunStore, ref: ArtifactRef) -> JSONDict:
    with store.artifact_path(ref).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class MergeViewAdapter:
    version = "4"

    def project(
        self,
        store: RunStore,
        artifacts: list[ArtifactRef],
        options: Mapping[str, Any] | None = None,
    ) -> ViewDocument:
        # 1. Collect the public Paper knowledge (helpers excluded).
        paper_payloads: list[JSONDict] = []
        known: dict[str, JSONDict] = {}
        for ref in artifacts:
            if ref.kind != "gaia.ir" or ref.producer_stage != "input":
                continue
            payload = _read_json(store, ref)
            paper_payloads.append(payload)
            for knowledge in payload.get("knowledges") or []:
                if not isinstance(knowledge, Mapping):
                    continue
                qid, content = knowledge.get("id"), knowledge.get("content")
                if not isinstance(qid, str) or not qid or _is_helper(qid):
                    continue
                known[qid] = {"content": content or "", "type": knowledge.get("type", "claim")}

        # 2. Pre-read the integration delta so its touched claims can be routed
        #    to the delta layer (the viewer filters nodes by layer).
        integration_doc: JSONDict | None = None
        for ref in artifacts:
            if ref.kind == "formalization" and ref.producer_stage == _INTEGRATION_STAGE:
                integration_doc = _read_json(store, ref)
        delta_qids: set[str] = set()
        k_ids: list[str] = []
        if integration_doc is not None:
            graph = integration_doc.get("graph", {})
            for operator in graph.get("operators", []):
                delta_qids.update(str(v) for v in operator.get("variables") or [])
                if isinstance(operator.get("conclusion"), str):
                    delta_qids.add(operator["conclusion"])
            for strategy in graph.get("strategies", []):
                delta_qids.update(str(p) for p in strategy.get("premises") or [])
                if isinstance(strategy.get("conclusion"), str):
                    delta_qids.add(strategy["conclusion"])
            for knowledge_id in (integration_doc.get("knowledges") or {}):
                if isinstance(knowledge_id, str) and knowledge_id not in known and not _is_helper(knowledge_id):
                    k_ids.append(knowledge_id)

        packages = sorted({str(payload.get("package_name") or "") for payload in paper_payloads})
        letters = {name: chr(ord("A") + index) for index, name in enumerate(packages)}

        nodes: list[JSONDict] = []
        edges: list[JSONDict] = []
        search_documents: list[JSONDict] = []
        source_artifacts: list[str] = []
        emitted: set[str] = set()
        labels: dict[str, str] = {}

        def paper_letter(qid: str) -> str:
            for name, letter in letters.items():
                if f":{name}::" in qid:
                    return letter
            return "?"

        def add_node(node_id: str, label: str, content: str, kind: str, layer: str, qid: str) -> str:
            if node_id in emitted:
                return node_id
            emitted.add(node_id)
            labels[qid] = label
            nodes.append({
                "id": node_id,
                "entity_id": label,
                "label": label,
                "display_label": label,
                "display_meta": kind,
                "kind": kind,
                "layer": layer,
                "step": _STEP,
                "visible_at": ["overview", "standard"],
                "min_granularity": "overview",
                "summary": content,
                "source_anchor_ids": [],
                "details": {"id": qid, "qid": qid, "type": kind, "content": content,
                            "integration_owned": layer == _DELTA_LAYER and qid not in known},
            })
            search_documents.append({
                "id": f"search:{qid}", "title": label, "text": content,
                "tags": [kind, layer], "refs": [node_id], "metadata": {"step": _STEP},
            })
            return node_id

        def knowledge_node(knowledge_id: str, content: str, kind: str) -> str | None:
            if knowledge_id not in known:
                return None
            layer = _DELTA_LAYER if knowledge_id in delta_qids else _CONTEXT_LAYER
            return add_node(f"knowledge:{knowledge_id}",
                            _short_label(knowledge_id, paper_letter(knowledge_id)),
                            content, kind, layer, knowledge_id)

        def relation_edge(source: str, target: str, label: str, layer: str) -> None:
            if f"knowledge:{source}" not in emitted or f"knowledge:{target}" not in emitted:
                return  # endpoint is a helper / unknown node: drop the edge
            edges.append({
                "id": f"edge:{len(edges)}:{source}:{target}:{label}:{layer}",
                "source": f"knowledge:{source}",
                "target": f"knowledge:{target}",
                "label": label,
                "layer": layer,
                "edge_class": "reasoning",
                "semantic_type": label,
                "reasoning_state": "confirmed",
                "step": _STEP,
                "visible_at": ["overview", "standard"],
                "min_granularity": "overview",
                "details": {"source": source, "target": target, "relation": label},
            })

        # 3. Paper claims + package-internal reasoning.
        for ref in artifacts:
            if ref.kind != "gaia.ir" or ref.producer_stage != "input":
                continue
            source_artifacts.append(ref.artifact_id)
            payload = _read_json(store, ref)
            package_name = payload.get("package_name")
            for qid, record in known.items():
                if f":{package_name}::" not in qid:
                    continue
                knowledge_node(qid, record["content"], record["type"])
            for operator in payload.get("operators") or []:
                variables = [str(v) for v in operator.get("variables") or []]
                op_type = str(operator.get("operator") or operator.get("type") or "")
                if op_type in {"equivalence", "implication"} and len(variables) >= 2:
                    relation_edge(variables[0], variables[1], op_type, _PAPER_EDGE_LAYER)
            for strategy in payload.get("strategies") or []:
                conclusion = strategy.get("conclusion")
                premises = [str(p) for p in strategy.get("premises") or []]
                if not isinstance(conclusion, str) or not premises:
                    continue
                strategy_type = str(strategy.get("type") or "infer")
                for premise in premises:
                    relation_edge(premise, conclusion, strategy_type, _PAPER_EDGE_LAYER)

        # 4. Integration delta: candidate nodes + new cross-paper edges.
        if integration_doc is not None:
            for ref in artifacts:
                if ref.kind == "formalization" and ref.producer_stage == _INTEGRATION_STAGE:
                    source_artifacts.append(ref.artifact_id)
            k_counter = 0
            a_counter = 0
            for knowledge_id in k_ids:
                knowledge = (integration_doc.get("knowledges") or {}).get(knowledge_id) or {}
                content = (knowledge.get("content") or {}).get("canonical", "") if isinstance(knowledge.get("content"), Mapping) else ""
                # `candidate_A_*` is the summarized observation premise of an
                # abduction star and `candidate_K_*` the domain conclusion; they
                # must not both render as "K:".
                if knowledge_id.startswith("candidate_A_"):
                    a_counter += 1
                    label = f"A:{a_counter}"
                else:
                    k_counter += 1
                    level = (knowledge.get("metadata") or {}).get("conclusion_level") if isinstance(knowledge.get("metadata"), Mapping) else None
                    # Level-tagged domain conclusions make the summary tree legible.
                    label = f"K:{k_counter}" + (f"·L{level}" if level else "")
                add_node(f"knowledge:{knowledge_id}", label, content,
                         knowledge.get("type", "claim"), _DELTA_LAYER, knowledge_id)
            for operator in integration_doc.get("graph", {}).get("operators", []):
                variables = [str(v) for v in operator.get("variables") or []]
                op_type = str(operator.get("type") or "")
                if len(variables) < 2 or not op_type:
                    continue  # negation has a single operand: no pairwise edge
                # Every operator type is projected (a bare equivalence/implication
                # filter left conjunction/contradiction operands with no edge at
                # all, so they looked floating).  A k-ary operator is drawn as a
                # star from its first operand, the only shape this viewer has.
                for operand in variables[1:]:
                    relation_edge(variables[0], operand, op_type, _DELTA_LAYER)
            for strategy in integration_doc.get("graph", {}).get("strategies", []):
                conclusion = strategy.get("conclusion")
                premises = [str(p) for p in strategy.get("premises") or []]
                if not isinstance(conclusion, str) or not premises:
                    continue
                strategy_type = str(strategy.get("type") or "infer")
                for premise in premises:
                    relation_edge(premise, conclusion, strategy_type, _DELTA_LAYER)

        return ViewDocument(
            title="Domain graph · Papers + Integration",
            source_artifacts=source_artifacts,
            nodes=nodes,
            edges=edges,
            search_documents=search_documents,
            layers=[
                {"id": _CONTEXT_LAYER, "label": "Paper knowledge (context)", "default_visible": True},
                {"id": _PAPER_EDGE_LAYER, "label": "Paper reasoning", "default_visible": True},
                {"id": _DELTA_LAYER, "label": "★ Integration delta", "default_visible": True},
            ],
            metadata={
                "latest_step": _STEP,
                "merge": True,
                "delta_node_count": sum(1 for node in nodes if node["layer"] == _DELTA_LAYER),
                "integration_edge_count": sum(1 for edge in edges if edge["layer"] == _DELTA_LAYER),
            },
            stages=[{"number": _STEP, "name": "integration", "validation_status": "passed"}],
        )


__all__ = ["MergeViewAdapter"]
