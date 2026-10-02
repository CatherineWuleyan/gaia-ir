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

        for node in self.nodes:
            anchors = node.get("source_anchor_ids") or (node.get("details") or {}).get("source_anchor_ids") or []
            if anchor_id in anchors:
                return {"anchor_id": anchor_id, "status": "available", "node_id": node.get("id")}
        for target in self.payload.get("source_targets") or []:
            if isinstance(target, dict) and target.get("anchor_id") == anchor_id:
                return {"anchor_id": anchor_id, "status": "available", **target}
        return {"anchor_id": anchor_id, "status": "unavailable", "reason": "final merge.html has no audited source anchor for this id"}

    def _node(self, node_id: str | None) -> dict[str, Any]:
        return next((node for node in self.nodes if node.get("id") == node_id), {"id": node_id, "display_label": node_id, "summary": ""})

    @staticmethod
    def _public_node(node: dict[str, Any], score: int | None = None) -> dict[str, Any]:
        result = {
            "node_id": node.get("id"),
            "label": node.get("display_label") or node.get("label"),
            "kind": node.get("kind"),
            "summary": node.get("summary") or (node.get("details") or {}).get("content", ""),
            "source_anchor_ids": list(node.get("source_anchor_ids") or []),
        }
        if score is not None:
            result["lexical_score"] = score
        return result

    @staticmethod
    def _public_edge(edge: dict[str, Any]) -> dict[str, Any]:
        return {
            "edge_id": edge.get("id"),
            "source": edge.get("source"),
            "target": edge.get("target"),
            "semantic_type": edge.get("semantic_type") or (edge.get("details") or {}).get("relation"),
            "reasoning_state": edge.get("reasoning_state"),
            "provenance": (edge.get("details") or {}).get("provenance"),
        }


class RetrievalSession:
    """Budgeted retrieval wrapper that records every graph access.

    This is intentionally local and read-only.  A model adapter can expose
    these three methods as tools later; until then, the session provides the
    exact trace and budget accounting needed by the pilot preflight.
    """

    def __init__(self, graph: FinalGraph, *, max_calls: int = 8, max_return_tokens: int = 4000) -> None:
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 0:
            raise ValueError("max_calls must be a non-negative integer")
        if isinstance(max_return_tokens, bool) or not isinstance(max_return_tokens, int) or max_return_tokens < 0:
            raise ValueError("max_return_tokens must be a non-negative integer")
        self.graph = graph
        self.max_calls = max_calls
        self.max_return_tokens = max_return_tokens
        self.calls: list[dict[str, Any]] = []
        self.returned_tokens = 0

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
        return self._finish(self.graph.search_claims(query, k))

    def expand_neighbors(self, node_or_edge_id: str, k: int = 10) -> list[dict[str, Any]]:
        self._start("expand_neighbors", {"node_or_edge_id": node_or_edge_id, "k": k})
        return self._finish(self.graph.expand_neighbors(node_or_edge_id, k))

    def get_source_anchor(self, anchor_id: str) -> dict[str, Any]:
        self._start("get_source_anchor", {"anchor_id": anchor_id})
        return self._finish(self.graph.get_source_anchor(anchor_id))

    def trace(self) -> dict[str, Any]:
        return {
            "snapshot_sha256": self.graph.embedded_sha256,
            "canonical_snapshot_sha256": self.graph.canonical_sha256,
            "max_calls": self.max_calls,
            "max_return_tokens": self.max_return_tokens,
            "calls": list(self.calls),
            "call_count": len(self.calls),
            "returned_tokens": self.returned_tokens,
            "warnings": [
                "graph provenance is unavailable; returned nodes and edges cannot receive evidence credit"
            ] if self.graph.payload.get("source_targets") == [] else [],
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


def load_final_graph(html_path: Path) -> FinalGraph:
    payload, html_sha256, embedded_sha256 = extract_view_data(html_path)
    audit_graph(payload)
    return FinalGraph(payload, html_sha256, embedded_sha256)


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
