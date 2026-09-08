from __future__ import annotations

import json
from typing import Any, Mapping

from pipeline_harness.models import ArtifactRef
from pipeline_harness.store import RunStore
from pipeline_harness.view.model import ViewDocument


class MergeFormalizationViewAdapter:
    """Project the final merge formalization into the semantic graph viewer."""

    version = "1"

    def project(
        self,
        store: RunStore,
        artifacts: list[ArtifactRef],
        options: Mapping[str, Any] | None = None,
    ) -> ViewDocument:
        candidates = [
            ref for ref in artifacts
            if ref.kind == "formalization"
            and ref.metadata.get("step_name") == "step4_formalize_integration"
        ]
        if not candidates:
            raise ValueError("merge run has no step4 integration formalization")
        ref = candidates[-1]
        document = json.loads(store.artifact_path(ref).read_text(encoding="utf-8"))
        graph = document.get("graph", {})
        knowledges = document.get("knowledges", {})
        if not isinstance(graph, dict) or not isinstance(knowledges, dict):
            raise ValueError("integration formalization has invalid graph or knowledges")

        nodes: list[dict[str, Any]] = []
        search_documents: list[dict[str, Any]] = []
        node_ids = set(graph.get("nodes", []))
        for knowledge_id, knowledge in knowledges.items():
            if not isinstance(knowledge_id, str) or not isinstance(knowledge, dict):
                continue
            node_ids.add(knowledge_id)
            content = knowledge.get("content", {})
            canonical = content.get("canonical", "") if isinstance(content, dict) else ""
            label = knowledge_id.rsplit("::", 1)[-1]
            node_id = f"claim:{knowledge_id}"
            nodes.append({
                "id": node_id,
                "label": label,
                "kind": knowledge.get("type", "claim"),
                "layer": "claims",
                "step": 4,
                "min_granularity": "overview",
                "visible_at": ["overview", "standard"],
                "summary": canonical,
                "details": knowledge,
            })
            search_documents.append({
                "id": node_id,
                "title": label,
                "text": canonical,
                "tags": ["claim", knowledge_id.split("::", 1)[0]],
                "refs": [node_id],
                "metadata": {"knowledge_id": knowledge_id},
            })

        edges: list[dict[str, Any]] = []
        strategies = graph.get("strategies", [])
        if not isinstance(strategies, list):
            strategies = []
        for strategy in strategies:
            if not isinstance(strategy, dict):
                continue
            strategy_id = str(strategy.get("strategy_id", ""))
            if not strategy_id:
                continue
            strategy_type = str(strategy.get("type", "infer"))
            conclusion = strategy.get("conclusion")
            target = f"claim:{conclusion}" if isinstance(conclusion, str) else ""
            if target and target in {node["id"] for node in nodes}:
                for index, premise in enumerate(strategy.get("premises", [])):
                    source = f"claim:{premise}"
                    if source in {node["id"] for node in nodes}:
                        edges.append({
                            "id": f"strategy:{strategy_id}:{index}",
                            "source": source,
                            "target": target,
                            "label": "→",
                            "semantic_type": strategy_type,
                            "layer": "operators",
                            "edge_class": "reasoning",
                            "reasoning_state": "confirmed",
                            "step": 4,
                            "min_granularity": "standard",
                            "visible_at": ["standard"],
                        })

        return ViewDocument(
            title="Merged semantic graph",
            source_artifacts=[ref.artifact_id],
            nodes=nodes,
            edges=edges,
            search_documents=search_documents,
            layers=[
                {"id": "claims", "label": "Claims", "default_visible": True},
                {"id": "operators", "label": "Relations", "default_visible": True},
            ],
            metadata={
                "adapter": "merge_formalization",
                "package": document.get("package", {}),
                "claim_count": len(knowledges),
                "strategy_count": len(strategies),
            },
        )
