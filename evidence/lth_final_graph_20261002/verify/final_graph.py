"""Load and audit the embedded graph in the workspace's final ``merge.html``.

The HTML viewer is the authoritative graph artifact for the first pilot.  This
module extracts the ``view-data`` JSON without executing the page, records both
the HTML and embedded-payload hashes, and exposes a small deterministic
retrieval interface for later model adapters.

The current viewer graph is a presentation/merge artifact.  It deliberately
does not invent source anchors: ``get_source_anchor`` reports an unavailable
anchor when the graph has no audited provenance for that node or edge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VIEW_DATA_RE = re.compile(
    r'<script\s+id=["\']view-data["\']\s+type=["\']application/json["\']\s*>'
    r"(?P<payload>.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)


class FinalGraphError(ValueError):
    """Raised when the final graph artifact is missing or malformed."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def extract_view_data(html_path: Path) -> tuple[dict[str, Any], str, str]:
    """Return ``(payload, html_sha256, embedded_json_sha256)`` from merge.html."""

    if not html_path.is_file():
        raise FinalGraphError(f"final graph HTML does not exist: {html_path}")
    html = html_path.read_text(encoding="utf-8")
    match = VIEW_DATA_RE.search(html)
    if match is None:
        raise FinalGraphError("merge.html has no application/json view-data script")
    raw = match.group("payload")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FinalGraphError(f"embedded view-data is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FinalGraphError("embedded view-data must be a JSON object")
    return payload, sha256_text(html), sha256_text(raw)


def _string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _node_text(node: dict[str, Any]) -> str:
    details = node.get("details") or {}
    parts = [
        _string(node.get("display_label")),
        _string(node.get("label")),
        _string(node.get("summary")),
        _string(details.get("content")),
        _string(details.get("original_label")),
    ]
    return " ".join(part for part in parts if part).strip()


def _split_source_anchor_ref(value: Any) -> tuple[str | None, str]:
    """Split an optional ``paper_id:anchor_id`` source reference."""

    raw = str(value or "")
    if ":" not in raw:
        return None, raw
    paper_id, anchor_id = raw.split(":", 1)
    if paper_id and anchor_id:
        return paper_id, anchor_id
    return None, raw


def _paper_id_from_source_path(value: Any) -> str | None:
    pattern = r"(?:^|/)data/([^/]+)/ocr" + r"\.md(?:$|/)"
    match = re.search(pattern, str(value or ""))
    return match.group(1) if match else None


def audit_graph(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate basic graph integrity and report provenance limitations."""

    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise FinalGraphError("view-data must contain list fields nodes and edges")
    node_ids = [node.get("id") for node in nodes if isinstance(node, dict)]
    if len(node_ids) != len(nodes) or any(not isinstance(node_id, str) or not node_id for node_id in node_ids):
        raise FinalGraphError("every graph node must have a non-empty string id")
    if len(set(node_ids)) != len(node_ids):
        raise FinalGraphError("graph node ids are not unique")
    node_set = set(node_ids)
    bad_edges = []
    edge_ids = []
    for edge in edges:
        if not isinstance(edge, dict):
            bad_edges.append("non-object edge")
            continue
        edge_id = edge.get("id")
        edge_ids.append(edge_id)
        if not isinstance(edge_id, str) or not edge_id:
            bad_edges.append("edge without id")
        if edge.get("source") not in node_set or edge.get("target") not in node_set:
            bad_edges.append(edge_id or "edge without id")
    if len(set(edge_ids)) != len(edge_ids):
        bad_edges.append("duplicate edge ids")
    if bad_edges:
        raise FinalGraphError("invalid graph edges: " + ", ".join(map(str, bad_edges[:8])))

    adjacency = {node_id: set() for node_id in node_ids}
    for edge in edges:
        adjacency[edge["source"]].add(edge["target"])
        adjacency[edge["target"]].add(edge["source"])
    components: list[int] = []
    unseen = set(node_ids)
    while unseen:
        start = unseen.pop()
        stack = [start]
        size = 1
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    size += 1
        components.append(size)
    components.sort(reverse=True)

    anchor_nodes = sum(bool(node.get("source_anchor_ids")) for node in nodes if isinstance(node, dict))
    provenance_nodes = sum(
        any(key in (node.get("details") or {}) for key in ("provenance", "source_anchor", "source_anchor_ids", "paper_id"))
        for node in nodes
        if isinstance(node, dict)
    )
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "layer_counts": _counts(node.get("layer") for node in nodes if isinstance(node, dict)),
        "node_kind_counts": _counts(node.get("kind") for node in nodes if isinstance(node, dict)),
        "display_prefix_counts": _counts(
            (re.match(r"^[A-Z]+", str(node.get("display_label", ""))) or ["<unknown>"])[0]
            for node in nodes
            if isinstance(node, dict) and node.get("display_label")
        ),
        "edge_semantic_counts": _counts(
            edge.get("semantic_type") or (edge.get("details") or {}).get("relation")
            for edge in edges
            if isinstance(edge, dict)
        ),
        "edge_layer_counts": _counts(edge.get("layer") for edge in edges if isinstance(edge, dict)),
        "edge_reasoning_state_counts": _counts(edge.get("reasoning_state") for edge in edges if isinstance(edge, dict)),
        "edges_with_original_relation": sum(bool((edge.get("details") or {}).get("original_relation")) for edge in edges if isinstance(edge, dict)),
        "edges_with_confidence": sum((edge.get("details") or {}).get("confidence") is not None for edge in edges if isinstance(edge, dict)),
        "connected_component_count": len(components),
        "largest_component_size": components[0] if components else 0,
        "isolated_node_count": sum(len(adjacency[node_id]) == 0 for node_id in node_ids),
        # The viewer payload does not retain paper_id on nodes or edges.  Do
        # not infer cross-paper provenance from the shared integration prefix;
        # that would turn a presentation namespace into a scientific label.
        "cross_paper_edge_count": None,
        "cross_paper_status": "unknown_without_paper_provenance",
        "nodes_with_source_anchor_ids": anchor_nodes,
        "anchor_coverage": anchor_nodes / len(nodes) if nodes else 0.0,
        "nodes_with_provenance_fields": provenance_nodes,
        "source_targets_count": len(payload.get("source_targets") or []),
        "source_artifacts_count": len(payload.get("source_artifacts") or []),
        "metadata": payload.get("metadata") or {},
        "provenance_status": "missing" if anchor_nodes == 0 and provenance_nodes == 0 else "partial",
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = "<null>" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


@dataclass(frozen=True)
class FinalGraph:
    """Deterministic, read-only retrieval view over the final graph."""

    payload: dict[str, Any]
    html_sha256: str
    embedded_sha256: str
    provenance_sidecar: dict[str, Any] | None = None
    # Optional exact standalone copy of the viewer's embedded payload.  It is
    # useful for index-based retrieval, but it is accepted only after its
    # canonical content matches the authoritative merge.html payload.
    index_sha256: str | None = None

    @property
    def _provenance_nodes(self) -> dict[str, dict[str, Any]]:
        payload = self.provenance_sidecar or {}
        return {
            item.get("graph_node_id"): item
            for item in payload.get("nodes", [])
            if isinstance(item, dict) and isinstance(item.get("graph_node_id"), str)
        }

    @property
    def _provenance_edges(self) -> dict[str, dict[str, Any]]:
        payload = self.provenance_sidecar or {}
        return {
            item.get("graph_edge_id"): item
            for item in payload.get("edges", [])
            if isinstance(item, dict) and isinstance(item.get("graph_edge_id"), str)
        }

    @property
    def canonical_sha256(self) -> str:
        canonical = json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256_text(canonical)

    @property
    def nodes(self) -> list[dict[str, Any]]:
        return [node for node in self.payload.get("nodes", []) if isinstance(node, dict)]

    @property
    def edges(self) -> list[dict[str, Any]]:
        return [edge for edge in self.payload.get("edges", []) if isinstance(edge, dict)]

    def model_payload(
        self,
        *,
        eligible_paper_ids: Iterable[str] | None = None,
        include_summary_nodes: bool = True,
        serialization: str = "standard",
    ) -> dict[str, Any]:
        """Return the complete graph in a bounded, model-facing form.

        ``merge.html`` remains the source of truth.  The viewer payload also
        contains repeated search-document and rendering data, so the model
        input keeps every node and edge but drops those redundant caches.  The
        claim text, IDs, relation type, endpoints, visibility and provenance
        fields needed for citation/audit are preserved.
        """

        if not isinstance(include_summary_nodes, bool):
            raise FinalGraphError("include_summary_nodes must be a boolean")
        if serialization not in {"standard", "compact"}:
            raise FinalGraphError("serialization must be 'standard' or 'compact'")
        allowlist: set[str] | None = None
        if eligible_paper_ids is not None:
            values = list(eligible_paper_ids)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise FinalGraphError("eligible_paper_ids must contain non-empty strings")
            allowlist = set(values)

        def provenance_subset(ref_type: str, ref_id: Any) -> bool:
            """Return whether a sidecar item is safely inside the allowlist.

            A filtered control must fail closed for nodes without explicit
            sidecar paper ids.  In particular, an integration/display label
            is never treated as a paper provenance signal.
            """

            if allowlist is None or not isinstance(ref_id, str):
                return allowlist is None
            records = self._provenance_nodes if ref_type == "graph_node" else self._provenance_edges
            record = records.get(ref_id)
            if not isinstance(record, dict):
                return False
            paper_ids = {str(value) for value in (record.get("paper_ids") or []) if value is not None}
            if not paper_ids:
                return False
            return paper_ids.issubset(allowlist)

        selected_node_ids = {
            str(node.get("id"))
            for node in self.nodes
            if isinstance(node, dict)
            and provenance_subset("graph_node", node.get("id"))
        }
        if allowlist is None:
            selected_node_ids = {str(node.get("id")) for node in self.nodes if isinstance(node, dict)}
        if not include_summary_nodes:
            selected_node_ids = {
                node_id for node_id in selected_node_ids
                if not self._is_summary_node_id(node_id)
            }

        selected_nodes = [
            node for node in self.nodes
            if isinstance(node, dict) and str(node.get("id")) in selected_node_ids
        ]
        selected_edges = [
            edge for edge in self.edges
            if isinstance(edge, dict)
            and str(edge.get("source")) in selected_node_ids
            and str(edge.get("target")) in selected_node_ids
            and provenance_subset("graph_edge", edge.get("id"))
        ] if allowlist is not None or not include_summary_nodes else list(self.edges)

        if serialization == "compact":
            node_keys = ("id", "summary", "source_anchor_ids")
            edge_keys = ("id", "source", "target", "label", "semantic_type")
        else:
            node_keys = (
                "id", "label", "display_label", "display_meta", "entity_id", "kind",
                "layer", "min_granularity", "summary", "source_anchor_ids", "visible_at",
            )
            edge_keys = (
                "id", "source", "target", "label", "semantic_type", "reasoning_state",
                "layer", "min_granularity", "visible_at",
            )
        nodes = [
            {key: node[key] for key in node_keys if key in node}
            for node in selected_nodes
        ]
        edges = [
            {key: edge[key] for key in edge_keys if key in edge}
            for edge in selected_edges
        ]
        model_metadata = self.payload.get("metadata")
        if not include_summary_nodes and isinstance(model_metadata, dict):
            # These viewer bookkeeping fields are aliases for the pre-aggregated
            # K/candidate_K summaries.  Keeping them after removing summary
            # nodes would leak answer-like IDs into the supposedly summary-free
            # model input without adding useful graph structure.
            model_metadata = {
                key: value
                for key, value in model_metadata.items()
                if key not in {"reference_domain_nodes", "pipeline_conclusions_folded"}
            }
        payload_filter = self.payload.get("graph_filter")
        result = {
            "schema_version": (
                "merge_graph_model_payload_compact_v1"
                if serialization == "compact"
                else "merge_graph_model_payload_v1"
            ),
            "source": self.payload.get("source") or "authoritative_merge_html",
            "html_sha256": self.html_sha256,
            "embedded_payload_sha256": self.embedded_sha256,
            "canonical_payload_sha256": self.canonical_sha256,
            "nodes": nodes,
            "edges": edges,
        }
        if serialization == "standard":
            result.update({
                "title": self.payload.get("title"),
                "layers": self.payload.get("layers"),
                "metadata": model_metadata,
                "source_targets": self.payload.get("source_targets", []),
                "source_artifacts": self.payload.get("source_artifacts", []),
                "stages": self.payload.get("stages", []),
            })
        if allowlist is not None or not include_summary_nodes:
            canonical_ids = (
                json.dumps(sorted(allowlist), ensure_ascii=False, separators=(",", ":"))
                if allowlist is not None else ""
            )
            unknown_nodes = len(self.nodes) - len(selected_nodes)
            unknown_or_filtered_edges = len(self.edges) - len(selected_edges)
            if allowlist is not None and not include_summary_nodes:
                mode = "provenance_subset_without_summary_nodes"
            elif allowlist is not None:
                mode = "provenance_subset"
            else:
                mode = "summary_excluded"
            computed_filter = {
                "mode": mode,
                "eligible_paper_ids_sha256": sha256_text(canonical_ids) if allowlist is not None else None,
                "eligible_paper_count": len(allowlist) if allowlist is not None else None,
                "source": "explicit_provenance_sidecar" if allowlist is not None else "authoritative_merge_html",
                "unknown_or_out_of_scope_nodes_excluded": unknown_nodes,
                "unknown_or_out_of_scope_edges_excluded": unknown_or_filtered_edges,
                "full_graph_node_count": len(self.nodes),
                "full_graph_edge_count": len(self.edges),
                "filtered_node_count": len(selected_nodes),
                "filtered_edge_count": len(selected_edges),
                "include_summary_nodes": include_summary_nodes,
                "summary_nodes_excluded": sum(self._is_summary_node_id(str(node.get("id"))) for node in self.nodes),
                "fail_closed_for_missing_paper_ids": allowlist is not None,
            }
            # A separately materialized historical view has already applied a
            # stricter edge/provenance policy than the ordinary allowlist.
            # Preserve that policy in the model-facing payload instead of
            # relabelling it as a generic provenance subset.
            if isinstance(payload_filter, dict) and payload_filter.get("mode") == "temporal_filtered_derived_edges_only":
                result["graph_filter"] = {**payload_filter, **{
                    "eligible_paper_ids_sha256": computed_filter["eligible_paper_ids_sha256"],
                    "eligible_paper_count": computed_filter["eligible_paper_count"],
                    "fail_closed_for_missing_paper_ids": computed_filter["fail_closed_for_missing_paper_ids"],
                    "selected_node_count": computed_filter["filtered_node_count"],
                    "selected_edge_count": computed_filter["filtered_edge_count"],
                    "filtered_node_count": computed_filter["filtered_node_count"],
                    "filtered_edge_count": computed_filter["filtered_edge_count"],
                    "full_graph_node_count": computed_filter["full_graph_node_count"],
                    "full_graph_edge_count": computed_filter["full_graph_edge_count"],
                }}
            else:
                result["graph_filter"] = computed_filter
        return result

    @staticmethod
    def _is_summary_node_id(node_id: str) -> bool:
        """Identify graph-level integration summaries without using labels as provenance."""

        return (
            node_id.startswith("candidate_K_")
            or ":candidate_K_" in node_id
            or node_id.startswith("knowledge:ref:K")
        )

    def search_claims(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Return deterministic lexical matches, with no fabricated provenance."""

        if not isinstance(query, str) or not query.strip():
            return []
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        terms = [term.casefold() for term in re.findall(r"[\w-]+", query) if term.strip()]
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for node in self.nodes:
            text = _node_text(node).casefold()
            score = sum(text.count(term) for term in terms)
            if score:
                ranked.append((score, str(node.get("id")), node))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [self._public_node(node, score) for score, _, node in ranked[:k]]

    def expand_neighbors(self, node_or_edge_id: str, k: int = 10) -> list[dict[str, Any]]:
        """Return incident edges and adjacent nodes for one node or edge."""

        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        node_ids = {node.get("id") for node in self.nodes}
        edge = next((item for item in self.edges if item.get("id") == node_or_edge_id), None)
        if edge is not None:
            node_or_edge_id = str(edge.get("source"))
        if node_or_edge_id not in node_ids:
            return []
        incident = [
            item for item in self.edges
            if item.get("source") == node_or_edge_id or item.get("target") == node_or_edge_id
        ]
        incident.sort(key=lambda item: str(item.get("id")))
        result = []
        for item in incident[:k]:
            result.append({
                "edge": self._public_edge(item),
                "source": self._public_node(self._node(item.get("source"))),
                "target": self._public_node(self._node(item.get("target"))),
            })
        return result

    def get_source_anchor(self, anchor_id: str) -> dict[str, Any]:
        """Return an anchor only when the final graph actually contains one."""

        paper_id, local_anchor_id = _split_source_anchor_ref(anchor_id)

        # The presentation graph has no paper id on direct node fields.  A
        # qualified reference therefore resolves through the audited sidecar;
        # an unqualified reference keeps the legacy direct lookup behavior.
        for node in self.nodes if paper_id is None else []:
            anchors = node.get("source_anchor_ids") or (node.get("details") or {}).get("source_anchor_ids") or []
            if local_anchor_id in anchors:
                return {"anchor_id": anchor_id, "status": "available", "node_id": node.get("id")}
        for target in self.payload.get("source_targets") or []:
            if isinstance(target, dict) and target.get("anchor_id") == local_anchor_id and paper_id is None:
                return {"anchor_id": anchor_id, "status": "available", **target}
        # The viewer remains authoritative.  A sidecar is an explicit,
        # separately audited opt-in and is labelled ``derived`` so callers do
        # not mistake a reverse closure for a direct edge/node citation.
        matches: list[dict[str, Any]] = []
        for node_id, item in sorted(self._provenance_nodes.items()):
            for source in item.get("source_anchors", []):
                if not isinstance(source, dict):
                    continue
                source_paper_id = _paper_id_from_source_path(source.get("source_path"))
                if paper_id is not None and source_paper_id != paper_id:
                    continue
                # Sidecars created before namespaced anchors used a local
                # ``anchor_claim_*`` value; newer curated sidecars may store
                # the fully qualified ``paper_id:anchor_id``.  Accept both
                # forms, while retaining the paper-path check above so two
                # papers reusing the same local anchor cannot cross-match.
                source_anchor_id = source.get("anchor_id")
                if source_anchor_id not in {local_anchor_id, anchor_id}:
                    continue
                matches.append({"graph_node_id": node_id, **source})
        if matches:
            return {
                "anchor_id": anchor_id,
                "status": "available",
                "provenance_class": "derived",
                "source_matches": matches,
                "warning": "sidecar reverse closure; not direct merge.html provenance",
            }
        reason = (
            "qualified source anchor is absent from the audited paper path"
            if paper_id is not None
            else "final merge.html has no audited source anchor for this id"
        )
        return {"anchor_id": anchor_id, "status": "unavailable", "reason": reason}

    def _node(self, node_id: str | None) -> dict[str, Any]:
        return next((node for node in self.nodes if node.get("id") == node_id), {"id": node_id, "display_label": node_id, "summary": ""})

    def _public_node(self, node: dict[str, Any], score: int | None = None) -> dict[str, Any]:
        provenance = self._provenance_nodes.get(node.get("id"), {})
        result = {
            "node_id": node.get("id"),
            "label": node.get("display_label") or node.get("label"),
            "kind": node.get("kind"),
            "summary": node.get("summary") or (node.get("details") or {}).get("content", ""),
            "source_anchor_ids": sorted(set(
                list(node.get("source_anchor_ids") or [])
                + list(provenance.get("source_anchor_ids") or [])
            )),
        }
        if provenance:
            result["provenance_class"] = provenance.get("provenance_class", "derived")
            paper_ids = sorted({str(value) for value in provenance.get("paper_ids", []) if value is not None})
            if paper_ids:
                result["source_paper_ids"] = paper_ids
                result["source_paper_count"] = len(paper_ids)
                result["source_paper_scope"] = "node_closure" if result["provenance_class"] == "derived" else "direct"
        if score is not None:
            result["lexical_score"] = score
        return result

    def _public_edge(self, edge: dict[str, Any]) -> dict[str, Any]:
        provenance = self._provenance_edges.get(edge.get("id"), {})
        result = {
            "edge_id": edge.get("id"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "semantic_type": edge.get("semantic_type") or (edge.get("details") or {}).get("relation"),
            "reasoning_state": edge.get("reasoning_state"),
            "provenance": (edge.get("details") or {}).get("provenance"),
        }
        if provenance:
            result["provenance_class"] = provenance.get("provenance_class", "derived")
            paper_ids = sorted({str(value) for value in provenance.get("paper_ids", []) if value is not None})
            if paper_ids:
                result["source_paper_ids"] = paper_ids
                result["source_paper_count"] = len(paper_ids)
                result["source_paper_scope"] = "endpoint_closure" if result["provenance_class"] == "derived" else "direct"
        return result


class RetrievalSession:
    """Budgeted retrieval wrapper that records every graph access.

    This is intentionally local and read-only.  A model adapter can expose
    these three methods as tools later; until then, the session provides the
    exact trace and budget accounting needed by the pilot preflight.
    """

    def __init__(
        self,
        graph: FinalGraph,
        *,
        max_calls: int = 8,
        max_return_tokens: int = 4000,
        eligible_paper_ids: Iterable[str] | None = None,
    ) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative integer")
        if isinstance(max_return_tokens, bool) or not isinstance(max_return_tokens, int) or max_return_tokens < 0:
            raise ValueError("max_return_tokens must be a non-negative integer")
        self.graph = graph
        self.max_calls = max_calls
        self.max_return_tokens = max_return_tokens
        if eligible_paper_ids is None:
            self.eligible_paper_ids: set[str] | None = None
        else:
            values = list(eligible_paper_ids)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("eligible_paper_ids must contain non-empty strings")
            self.eligible_paper_ids = set(values)
        self.calls: list[dict[str, Any]] = []
        self.returned_tokens = 0

    def _allowed(self, ref_type: str, ref_id: Any) -> bool:
        """Fail closed for hidden-paper sessions without paper provenance."""

        if self.eligible_paper_ids is None:
            return True
        if not isinstance(ref_id, str):
            return False
        records = self.graph._provenance_nodes if ref_type == "graph_node" else self.graph._provenance_edges
        record = records.get(ref_id)
        if not isinstance(record, dict):
            return False
        paper_ids = {str(value) for value in (record.get("paper_ids") or []) if value is not None}
        return bool(paper_ids) and paper_ids.issubset(self.eligible_paper_ids)

    def _visible_node_ids(self) -> set[str]:
        return {
            str(node.get("id"))
            for node in self.graph.nodes
            if isinstance(node, dict) and self._allowed("graph_node", node.get("id"))
        }

    def _start(self, tool: str, arguments: dict[str, Any]) -> None:
        if len(self.calls) >= self.max_calls:
            raise FinalGraphError("retrieval call budget exceeded")
        self.calls.append({"sequence": len(self.calls) + 1, "tool": tool, "arguments": arguments})

    def _finish(self, result: Any) -> Any:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True)
        tokens = max(1, (len(encoded) + 3) // 4) if encoded else 0
        self.returned_tokens += tokens
        call = self.calls[-1]
        call["returned_tokens"] = tokens
        call["returned_ids"] = _returned_ids(result)
        call["cumulative_returned_tokens"] = self.returned_tokens
        if self.returned_tokens > self.max_return_tokens:
            raise FinalGraphError("retrieval return-token budget exceeded")
        return result

    def search_claims(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        self._start("search_claims", {"query": query, "k": k})
        visible = self._visible_node_ids()
        result = [item for item in self.graph.search_claims(query, k) if item.get("node_id") in visible]
        return self._finish(result)

    def expand_neighbors(self, node_or_edge_id: str, k: int = 10) -> list[dict[str, Any]]:
        self._start("expand_neighbors", {"node_or_edge_id": node_or_edge_id, "k": k})
        visible = self._visible_node_ids()
        if node_or_edge_id not in visible and not self._allowed("graph_edge", node_or_edge_id):
            return self._finish([])
        # The graph adapter truncates incident edges before the provenance
        # allowlist is applied below.  In a historical leave-one-paper-out
        # view, the first raw edge can be mixed/unknown while a later edge is
        # fully eligible; asking for only ``k`` raw edges would then report an
        # empty relation even though a valid relation exists.  Fetch the
        # complete incident list, filter it, and truncate after fail-closed
        # checks.
        raw_k = max(k, len(self.graph.edges)) if self.eligible_paper_ids is not None else k
        result = self.graph.expand_neighbors(node_or_edge_id, raw_k)
        filtered = []
        for item in result:
            edge = item.get("edge") or {}
            source = item.get("source") or {}
            target = item.get("target") or {}
            if (
                self._allowed("graph_edge", edge.get("edge_id"))
                and source.get("node_id") in visible
                and target.get("node_id") in visible
            ):
                filtered.append(item)
        return self._finish(filtered[:k])

    def get_source_anchor(self, anchor_id: str) -> dict[str, Any]:
        self._start("get_source_anchor", {"anchor_id": anchor_id})
        result = self.graph.get_source_anchor(anchor_id)
        if self.eligible_paper_ids is not None:
            if result.get("node_id") and not self._allowed("graph_node", result.get("node_id")):
                result = {"anchor_id": anchor_id, "status": "unavailable", "reason": "hidden-paper allowlist"}
            elif result.get("source_matches"):
                matches = [
                    item for item in result["source_matches"]
                    if self._allowed("graph_node", item.get("graph_node_id"))
                ]
                if not matches:
                    result = {"anchor_id": anchor_id, "status": "unavailable", "reason": "hidden-paper allowlist"}
                else:
                    result = {**result, "source_matches": matches}
        return self._finish(result)

    def trace(self) -> dict[str, Any]:
        sidecar = self.graph.provenance_sidecar or {}
        sidecar_nodes = [item for item in sidecar.get("nodes", []) if isinstance(item, dict)]
        sidecar_edges = [item for item in sidecar.get("edges", []) if isinstance(item, dict)]
        resolved_sidecar_nodes = sum(
            item.get("provenance_class") == "derived" and bool(item.get("source_anchor_ids"))
            for item in sidecar_nodes
        )
        return {
            "snapshot_sha256": self.graph.embedded_sha256,
            "canonical_snapshot_sha256": self.graph.canonical_sha256,
            "max_calls": self.max_calls,
            "max_return_tokens": self.max_return_tokens,
            "calls": list(self.calls),
            "call_count": len(self.calls),
            "returned_tokens": self.returned_tokens,
            "eligible_paper_ids_sha256": (
                sha256_text(json.dumps(sorted(self.eligible_paper_ids), separators=(",", ":")))
                if self.eligible_paper_ids is not None else None
            ),
            "fail_closed_filter": self.eligible_paper_ids is not None,
            "provenance_sidecar": {
                "present": bool(sidecar),
                "node_count": len(sidecar_nodes),
                "edge_count": len(sidecar_edges),
                "derived_nodes_with_source_anchors": resolved_sidecar_nodes,
            },
            "warnings": (
                ["graph provenance is derived from an explicit sidecar; it is not direct merge.html provenance"]
                if sidecar
                else ["graph provenance is unavailable; returned nodes and edges cannot receive evidence credit"]
            ) if self.graph.payload.get("source_targets") == [] else [],
        }


def _returned_ids(result: Any) -> list[str]:
    ids: list[str] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                for key in ("node_id", "edge_id", "anchor_id"):
                    value = item.get(key)
                    if isinstance(value, str):
                        ids.append(value)
                for nested_key in ("edge", "source", "target"):
                    nested = item.get(nested_key)
                    if isinstance(nested, dict):
                        ids.extend(_returned_ids(nested))
    elif isinstance(result, dict):
        for key in ("node_id", "edge_id", "anchor_id"):
            value = result.get(key)
            if isinstance(value, str):
                ids.append(value)
    return sorted(set(ids))


def load_provenance_sidecar(path: Path, graph: FinalGraph) -> dict[str, Any]:
    """Load an explicitly opted-in provenance sidecar for ``merge.html``.

    The sidecar must describe the same node/edge counts as the frozen viewer.
    It is allowed to add derived reverse-closure metadata, but never changes
    the authoritative graph payload or upgrades derived provenance to direct.
    """

    if not path.is_file():
        raise FinalGraphError(f"provenance sidecar does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalGraphError(f"invalid provenance sidecar: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalGraphError("provenance sidecar must be an object")
    summary = payload.get("summary") or {}
    hashes = payload.get("graph_hashes")
    expected_hashes = {
        "source_html_sha256": graph.html_sha256,
        "embedded_payload_sha256": graph.embedded_sha256,
        "canonical_payload_sha256": graph.canonical_sha256,
    }
    if not isinstance(hashes, dict) or any(hashes.get(key) != value for key, value in expected_hashes.items()):
        raise FinalGraphError(
            "provenance sidecar is not bound to this merge.html graph "
            f"(declared={hashes!r}, expected={expected_hashes!r})"
        )
    if summary.get("graph_node_count") != len(graph.nodes) or summary.get("graph_edge_count") != len(graph.edges):
        raise FinalGraphError("provenance sidecar graph counts do not match merge.html")
    if not isinstance(payload.get("nodes"), list) or not isinstance(payload.get("edges"), list):
        raise FinalGraphError("provenance sidecar must contain nodes and edges lists")
    node_records = payload["nodes"]
    edge_records = payload["edges"]
    if any(not isinstance(node, dict) or not isinstance(node.get("graph_node_id"), str) for node in node_records):
        raise FinalGraphError("provenance sidecar contains an invalid graph node record")
    if any(not isinstance(edge, dict) or not isinstance(edge.get("graph_edge_id"), str) for edge in edge_records):
        raise FinalGraphError("provenance sidecar contains an invalid graph edge record")
    node_id_list = [node["graph_node_id"] for node in node_records]
    edge_id_list = [edge["graph_edge_id"] for edge in edge_records]
    if len(set(node_id_list)) != len(node_id_list) or len(set(edge_id_list)) != len(edge_id_list):
        raise FinalGraphError("provenance sidecar contains duplicate graph IDs")
    node_ids = set(node_id_list)
    edge_ids = set(edge_id_list)
    expected_node_ids = {node.get("id") for node in graph.nodes}
    expected_edge_ids = {edge.get("id") for edge in graph.edges}
    if node_ids != expected_node_ids:
        missing = len(expected_node_ids - node_ids)
        extra = len(node_ids - expected_node_ids)
        raise FinalGraphError(f"provenance sidecar node coverage does not match merge.html (missing={missing}, extra={extra})")
    if edge_ids != expected_edge_ids:
        missing = len(expected_edge_ids - edge_ids)
        extra = len(edge_ids - expected_edge_ids)
        raise FinalGraphError(f"provenance sidecar edge coverage does not match merge.html (missing={missing}, extra={extra})")
    graph_edges = {edge.get("id"): edge for edge in graph.edges}
    for record in edge_records:
        graph_edge = graph_edges[record["graph_edge_id"]]
        for key in ("source", "target"):
            if key in record and record[key] != graph_edge.get(key):
                raise FinalGraphError(
                    f"provenance sidecar edge endpoint mismatch for {record['graph_edge_id']}:{key}"
                )
    return payload


def load_matching_index(path: Path, graph: FinalGraph) -> tuple[dict[str, Any], str]:
    """Load a standalone graph index only when it is an exact graph match.

    The index is a retrieval copy of the viewer model, not a second source of
    truth.  Comparing the canonical payload prevents accidentally pairing an
    older ``view_model.json`` with the final ``merge.html``.
    """

    if not path.is_file():
        raise FinalGraphError(f"graph index does not exist: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalGraphError(f"invalid graph index: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalGraphError("graph index must be a JSON object")
    if not isinstance(payload.get("nodes"), list) or not isinstance(payload.get("edges"), list):
        raise FinalGraphError("graph index must contain nodes and edges lists")
    audit_graph(payload)
    canonical = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if canonical != graph.canonical_sha256:
        raise FinalGraphError(
            "graph index canonical payload does not match merge.html "
            f"({canonical} != {graph.canonical_sha256})"
        )
    return payload, sha256_text(raw)


def validate_snapshot_file(path: Path, graph: FinalGraph) -> dict[str, str]:
    """Validate a declared JSON snapshot against the authoritative graph."""

    if not path.is_file():
        raise FinalGraphError(f"graph snapshot does not exist: {path}")
    try:
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalGraphError(f"invalid graph snapshot: {path}") from exc
    if not isinstance(payload, dict):
        raise FinalGraphError("graph snapshot must be a JSON object")
    audit_graph(payload)
    canonical = sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    if canonical != graph.canonical_sha256:
        raise FinalGraphError(
            "graph snapshot canonical payload does not match merge.html "
            f"({canonical} != {graph.canonical_sha256})"
        )
    return {"path": str(path), "sha256": sha256_text(raw), "canonical_sha256": canonical}


def load_final_graph(
    html_path: Path,
    *,
    provenance_path: Path | None = None,
    index_path: Path | None = None,
) -> FinalGraph:
    payload, html_sha256, embedded_sha256 = extract_view_data(html_path)
    audit_graph(payload)
    graph = FinalGraph(payload, html_sha256, embedded_sha256)
    index_sha256 = None
    if index_path is not None:
        indexed_payload, index_sha256 = load_matching_index(index_path, graph)
        # The bytes may differ from the HTML's embedded JSON formatting, but
        # canonical equality above guarantees the graph content is identical.
        graph = FinalGraph(indexed_payload, html_sha256, embedded_sha256, index_sha256=index_sha256)
    if provenance_path is not None:
        sidecar = load_provenance_sidecar(provenance_path, graph)
        graph = FinalGraph(graph.payload, html_sha256, embedded_sha256, sidecar, index_sha256)
    return graph


def write_snapshot(html_path: Path, output_dir: Path) -> dict[str, Any]:
    graph = load_final_graph(html_path)
    audit = audit_graph(graph.payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "graph_snapshot.json"
    snapshot_path.write_text(json.dumps(graph.payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "final_merge_graph_v1",
        "source_html": str(html_path),
        "source_html_sha256": graph.html_sha256,
        "embedded_payload_sha256": graph.embedded_sha256,
        "canonical_payload_sha256": graph.canonical_sha256,
        "source_html_mtime_ns": html_path.stat().st_mtime_ns,
        "snapshot_file": snapshot_path.name,
        "audit": audit,
        "retrieval_api": ["search_claims(query,k)", "expand_neighbors(node_or_edge_id,k)", "get_source_anchor(anchor_id)"],
        "provenance_policy": "Do not score an anchor unless get_source_anchor returns available.",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path, help="path to the authoritative merge.html")
    parser.add_argument("--output-dir", type=Path, required=True, help="directory for graph_snapshot.json and manifest.json")
    args = parser.parse_args()
    manifest = write_snapshot(args.html, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
