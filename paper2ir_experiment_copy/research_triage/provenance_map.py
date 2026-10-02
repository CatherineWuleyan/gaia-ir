"""Resolve provenance for the authoritative ``merge.html`` graph.

The final viewer is a merged presentation graph.  Its claim nodes retain the
cluster/candidate identifier, while source-paper edges live in the per-cluster
``view_model.json`` files.  This tool follows those edges backwards to source
claims, then joins the source claims to the local ``data/<paper_id>`` bundles.

The mapper is deliberately conservative:

* exact IDs and existing source-anchor metadata are used where available;
* derived candidates are labelled ``resolved_derived`` rather than pretending
  that a generated summary is a verbatim paper claim;
* manually wired or otherwise unmatched graph nodes/edges remain ``unknown``;
* ``anchor_claim_N`` is reported as a claim-file anchor, not as an original
  paper location.  Original-paper evidence is supplied only by paragraph
  anchors with an actual ``ocr.md`` line range.

It writes a sidecar JSON and a short audit report.  It does not modify the
HTML graph, the graph snapshot, or any existing runner.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


class ProvenanceMapError(ValueError):
    """Raised when an input artifact is malformed."""


VIEW_DATA_RE = re.compile(
    r'<script\s+id=["\']view-data["\']\s+type=["\']application/json["\']\s*>'
    r"(?P<payload>.*?)</script>",
    re.DOTALL | re.IGNORECASE,
)
PAPER_CLAIM_RE = re.compile(r"^(papers:paper_[^:]+)::(.+)$")
FINAL_NODE_RE = re.compile(r"^knowledge:integration:(lth_cluster_\d+)::(.+)$")
PARAGRAPH_RE = re.compile(r"^anchor_paragraph_(p\d+)$")
CLAIM_RE = re.compile(r"^anchor_claim_(\d+)$")


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProvenanceMapError(f"missing input: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProvenanceMapError(f"invalid JSON {path}: {exc}") from exc


def _strip_id(value: Any) -> str:
    """Normalize viewer IDs to the IDs used by cluster graph edges."""

    value = str(value or "")
    if value.startswith("knowledge:"):
        value = value[len("knowledge:") :]
    value = re.sub(r"^integration:lth_cluster_\d+::", "", value)
    return value


def _read_view_data(html_path: Path) -> dict[str, Any]:
    text = html_path.read_text(encoding="utf-8")
    match = VIEW_DATA_RE.search(text)
    if match is None:
        raise ProvenanceMapError(f"{html_path} has no view-data JSON")
    payload = json.loads(match.group("payload"))
    if not isinstance(payload, dict) or not isinstance(payload.get("nodes"), list):
        raise ProvenanceMapError("view-data must contain a nodes list")
    return payload


def _paper_paragraphs(path: Path) -> dict[str, dict[str, Any]]:
    """Reproduce the source anchor paragraph segmentation used by Pipeline 7."""

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    paragraphs: dict[str, dict[str, Any]] = {}
    buffer: list[str] = []
    start_line: int | None = None
    tag: str | None = None
    used: set[str] = set()

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line, tag
        if start_line is None or not buffer:
            buffer, start_line, tag = [], None, None
            return
        paragraph_tag = tag or f"p{len(paragraphs) + 1:04d}"
        if paragraph_tag in used:
            raise ProvenanceMapError(f"duplicate paragraph tag {paragraph_tag} in {path}")
        used.add(paragraph_tag)
        paragraphs[f"anchor_paragraph_{paragraph_tag}"] = {
            "anchor_id": f"anchor_paragraph_{paragraph_tag}",
            "start_line": start_line,
            "end_line": end_line,
            "text": "\n".join(buffer).strip(),
        }
        buffer, start_line, tag = [], None, None

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            flush(line_number - 1)
            continue
        # Source OCR files may carry an explicit ``[pXXXX]`` tag.  Preserve it
        # but strip the tag from the excerpt, as the pipeline does.
        match = re.match(r"^\s*\[#(p\d{4})\]\s*", line)
        if match is None:
            match = re.match(r"^\[?(p\d{4})\]?\s*", line)
        if match and buffer:
            flush(line_number - 1)
        if start_line is None:
            start_line = line_number
        if match:
            tag = match.group(1)
            line = line[match.end() :]
        buffer.append(line)
    flush(len(lines))
    return paragraphs


def _source_anchor_registry(
    paper_id: str,
    gaia: dict[str, Any],
    formalization: dict[str, Any],
    ocr_path: Path,
) -> dict[str, dict[str, Any]]:
    """Collect anchor metadata without inventing paragraph locations."""

    registry: dict[str, dict[str, Any]] = {}
    workflow = formalization.get("workflow") if isinstance(formalization, dict) else None
    for item in (workflow or {}).get("source_anchors", []) if isinstance(workflow, dict) else []:
        if isinstance(item, dict) and isinstance(item.get("anchor_id"), str):
            registry[item["anchor_id"]] = {
                "anchor_id": item["anchor_id"],
                "locator": item.get("locator"),
                "source_kind": item.get("source_kind"),
                "relevance": item.get("relevance"),
            }
    paragraphs = _paper_paragraphs(ocr_path) if ocr_path.is_file() else {}
    for anchor_id, paragraph in paragraphs.items():
        existing = registry.setdefault(anchor_id, {"anchor_id": anchor_id})
        existing.update(
            {
                "source_path": str(ocr_path),
                "start_line": paragraph["start_line"],
                "end_line": paragraph["end_line"],
                "text": paragraph["text"],
                "source_kind": existing.get("source_kind") or "source.paper_text",
            }
        )

    # Claim and observation metadata are available in both gaia.ir and
    # formalization.  Keep the first complete record and never infer a line
    # range from claim text.
    for document in (gaia, formalization):
        knowledges = document.get("knowledges", {}) if isinstance(document, dict) else {}
        iterable = knowledges.values() if isinstance(knowledges, dict) else knowledges
        if not isinstance(iterable, Iterable):
            continue
        for item in iterable:
            if not isinstance(item, dict):
                continue
            # gaia.ir stores a full papers:paper_hash:: prefix; formalization
            # stores local claim IDs.  Normalize both to local IDs.
            raw_id = str(item.get("id") or "")
            local_id = raw_id.split("::")[-1] if "::" in raw_id else raw_id
            anchor_ids = item.get("source_anchor_ids") or []
            if not anchor_ids:
                anchor_ids = (item.get("metadata") or {}).get("source_anchor_ids") or []
            for anchor_id in anchor_ids:
                if not isinstance(anchor_id, str):
                    continue
                current = registry.setdefault(anchor_id, {"anchor_id": anchor_id})
                if local_id:
                    current.setdefault("claim_ids", []).append(local_id)
                    current.setdefault("claim_local_ids", []).append(local_id)
    for item in registry.values():
        for key in ("claim_ids", "claim_local_ids"):
            if key in item:
                item[key] = sorted(set(item[key]))
    return registry


def _load_data_index(data_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return ``(claim_qid -> source info, paper_id -> paper info)``."""

    claim_index: dict[str, dict[str, Any]] = {}
    paper_index: dict[str, dict[str, Any]] = {}
    for gaia_path in sorted(data_root.glob("*/gaia.ir.json")):
        paper_dir = gaia_path.parent
        paper_id = paper_dir.name
        gaia = _json(gaia_path)
        formalization_path = paper_dir / "formalization.json"
        claims_path = paper_dir / "claims_final.json"
        ocr_path = paper_dir / "ocr.md"
        formalization = _json(formalization_path) if formalization_path.is_file() else {}
        claims_final = _json(claims_path) if claims_path.is_file() else {}
        anchors = _source_anchor_registry(paper_id, gaia, formalization, ocr_path)
        paper_info = {
            "paper_id": paper_id,
            "directory": str(paper_dir),
            "gaia_path": str(gaia_path),
            "formalization_path": str(formalization_path) if formalization_path.is_file() else None,
            "claims_final_path": str(claims_path) if claims_path.is_file() else None,
            "ocr_path": str(ocr_path) if ocr_path.is_file() else None,
            "package": gaia.get("package") or {},
            "anchors": anchors,
            "claims_final": claims_final,
            "formalization": formalization,
        }
        paper_index[paper_id] = paper_info
        knowledges = gaia.get("knowledges", [])
        if isinstance(knowledges, dict):
            knowledges = [dict(value, id=key) for key, value in knowledges.items()]
        for claim in knowledges:
            if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
                continue
            claim_id = claim["id"]
            if not claim_id.startswith("papers:paper_"):
                continue
            local_id = claim_id.split("::", 1)[-1]
            number = None
            if local_id.startswith("claim_") and local_id[6:].isdigit():
                number = int(local_id[6:])
            final_record = None
            if number is not None:
                for record in claims_final.get("claim", []) if isinstance(claims_final, dict) else []:
                    if isinstance(record, dict) and record.get("number") == number:
                        final_record = record
                        break
                if final_record is None:
                    for record in claims_final.get("note", []) if isinstance(claims_final, dict) else []:
                        if isinstance(record, dict) and record.get("number") == number:
                            final_record = record
                            break
            formal_record = None
            fknow = formalization.get("knowledges", {}) if isinstance(formalization, dict) else {}
            if isinstance(fknow, dict):
                formal_record = fknow.get(local_id)
            claim_index[claim_id] = {
                "paper_id": paper_id,
                "local_claim_id": local_id,
                "claim_text": claim.get("content"),
                "claim_type": claim.get("type"),
                "claim_metadata": claim.get("metadata") or {},
                "claims_final": final_record,
                "formalization": formal_record,
                "gaia_path": str(gaia_path),
                "formalization_path": str(formalization_path) if formalization_path.is_file() else None,
                "claims_final_path": str(claims_path) if claims_path.is_file() else None,
                "ocr_path": str(ocr_path) if ocr_path.is_file() else None,
                "source_anchor_ids": list(
                    claim.get("source_anchor_ids")
                    or (claim.get("metadata") or {}).get("source_anchor_ids")
                    or (formal_record or {}).get("source_anchor_ids", [])
                ),
            }
    return claim_index, paper_index


def _cluster_candidate_sources(
    cluster_view: dict[str, Any], candidate_id: str, cluster_ir: dict[str, Any] | None = None
) -> tuple[list[str], list[str], str]:
    """Reverse-traverse cluster strategies (then edges) to paper claims.

    The strategy premises are the strongest available derivation record.  A
    cluster viewer can contain edges that were added during rendering, so the
    edge traversal is retained as a fallback and its method is reported.
    """

    if isinstance(cluster_ir, dict):
        strategy_reverse: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for strategy in cluster_ir.get("strategies", []):
            if not isinstance(strategy, dict):
                continue
            conclusion = _strip_id(strategy.get("conclusion"))
            if conclusion:
                strategy_reverse[conclusion].append(strategy)
        queue = deque([_strip_id(candidate_id)])
        seen: set[str] = set()
        source_claims: set[str] = set()
        strategy_ids: set[str] = set()
        while queue:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            if current.startswith("papers:paper_"):
                source_claims.add(current)
                continue
            for strategy in strategy_reverse.get(current, []):
                if strategy.get("strategy_id"):
                    strategy_ids.add(str(strategy["strategy_id"]))
                for premise in strategy.get("premises", []) or []:
                    parent = _strip_id(premise)
                    if parent and parent not in seen:
                        queue.append(parent)
        if source_claims:
            return sorted(source_claims), sorted(strategy_ids), "gaia_ir_strategies"

    reverse: dict[str, set[str]] = defaultdict(set)
    for edge in cluster_view.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = _strip_id(edge.get("source"))
        target = _strip_id(edge.get("target"))
        if source and target:
            reverse[target].add(source)
    queue = deque([_strip_id(candidate_id)])
    seen: set[str] = set()
    sources: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        if current.startswith("papers:paper_"):
            sources.add(current)
            continue
        for parent in sorted(reverse.get(current, set())):
            if parent not in seen:
                queue.append(parent)
    return sorted(sources), [], "cluster_view_edges" if sources else "unresolved"


def _node_key(node_id: str) -> tuple[str, str] | None:
    match = FINAL_NODE_RE.match(node_id)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _anchor_records(source_info: dict[str, Any], anchor_ids: list[str], paper_info: dict[str, Any] | None) -> list[dict[str, Any]]:
    records = []
    anchors = (paper_info or {}).get("anchors", {}) if paper_info else {}
    for anchor_id in sorted(set(anchor_ids)):
        base = {"anchor_id": anchor_id}
        if anchor_id in anchors:
            base.update(anchors[anchor_id])
        elif (match := PARAGRAPH_RE.match(anchor_id)):
            base.update({"status": "unavailable", "reason": "paragraph anchor missing from source bundle"})
        elif CLAIM_RE.match(anchor_id):
            base.update({"status": "claim_file_only", "reason": "claim anchor has no original-paper line range"})
        else:
            base.update({"status": "unavailable", "reason": "anchor not found in source bundle"})
        records.append(base)
    return records


def build_map(html_path: Path, cluster_root: Path, data_root: Path) -> dict[str, Any]:
    payload = _read_view_data(html_path)
    claim_index, paper_index = _load_data_index(data_root)
    clusters: dict[str, dict[str, Any]] = {}
    cluster_irs: dict[str, dict[str, Any]] = {}
    for path in sorted(cluster_root.glob("cluster_*/view_model.json")):
        cluster_name = path.parent.name
        clusters[cluster_name] = _json(path)
        gaia_path = path.parent / "gaia.ir.json"
        if gaia_path.is_file():
            cluster_irs[cluster_name] = _json(gaia_path)

    # Manual merge passes can drop the ``integration:lth_cluster_*`` prefix
    # (for example ``knowledge:lost:candidate_*``).  Keep an exact-content
    # index as a conservative fallback.  A content match is accepted only
    # when it identifies one candidate unambiguously; otherwise it remains
    # unknown rather than choosing a likely-looking paper.
    candidate_by_content: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for cluster_name, cluster_view in clusters.items():
        for item in cluster_view.get("nodes", []):
            if not isinstance(item, dict):
                continue
            details = item.get("details") or {}
            candidate_id = str(details.get("id") or "")
            if not candidate_id.startswith("candidate_"):
                continue
            content = str(item.get("summary") or details.get("content") or "").strip()
            if content:
                candidate_by_content[content].append((cluster_name, candidate_id))

    node_records: list[dict[str, Any]] = []
    node_by_id: dict[str, dict[str, Any]] = {}
    for node in payload.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        key = _node_key(node_id)
        source_claim_ids: list[str] = []
        cluster_name = candidate_id = None
        resolution = "unknown"
        strategy_ids: list[str] = []
        trace_method = "unresolved"
        if key is not None:
            cluster_name, candidate_id = key
            # Final IDs use ``lth_cluster_00``; the on-disk directory is
            # ``cluster_00``.  Keep the final namespace in the report while
            # resolving against the directory spelling here.
            cluster_view = clusters.get(cluster_name.replace("lth_", "", 1))
            if cluster_view is not None:
                source_claim_ids, strategy_ids, trace_method = _cluster_candidate_sources(
                    cluster_view, candidate_id, cluster_irs.get(cluster_name.replace("lth_", "", 1))
                )
                resolution = "resolved_derived" if source_claim_ids else "unknown"
        else:
            content = str(node.get("summary") or (node.get("details") or {}).get("content") or "").strip()
            matches = candidate_by_content.get(content, [])
            if len(matches) == 1:
                cluster_name, candidate_id = matches[0]
                cluster_view = clusters.get(cluster_name)
                source_claim_ids, strategy_ids, trace_method = _cluster_candidate_sources(
                    cluster_view or {}, candidate_id, cluster_irs.get(cluster_name)
                )
                resolution = "resolved_derived" if source_claim_ids else "unknown"
            elif len(matches) > 1:
                # Preserve ambiguity for the report; source_claim_ids stays
                # empty so no evidence credit can be assigned.
                resolution = "unknown_ambiguous"
        source_claims = [claim_index[qid] for qid in source_claim_ids if qid in claim_index]
        missing_claim_ids = [qid for qid in source_claim_ids if qid not in claim_index]
        paper_ids = sorted({item["paper_id"] for item in source_claims})
        anchor_ids = sorted({aid for item in source_claims for aid in item.get("source_anchor_ids", [])})
        anchors = []
        for paper_id in paper_ids:
            pinfo = paper_index.get(paper_id)
            anchors.extend(_anchor_records({}, [a for item in source_claims if item["paper_id"] == paper_id for a in item.get("source_anchor_ids", [])], pinfo))
        if source_claim_ids and not source_claims:
            resolution = "unknown"
        record = {
            "graph_node_id": node_id,
            "display_label": node.get("display_label") or node.get("label"),
            "summary": node.get("summary") or (node.get("details") or {}).get("content", ""),
            "graph_kind": node.get("kind"),
            "graph_layer": node.get("layer"),
            "cluster": cluster_name,
            "candidate_id": candidate_id,
            "resolution": resolution,
            "provenance_class": "derived" if resolution == "resolved_derived" else "unresolved",
            "resolution_detail": "cluster_candidate_reverse_closure" if resolution == "resolved_derived" else "no_unambiguous_source_claim_path",
            "trace_method": trace_method if resolution == "resolved_derived" else None,
            "strategy_ids": strategy_ids if resolution == "resolved_derived" else [],
            "source_claim_ids": source_claim_ids,
            "missing_source_claim_ids": missing_claim_ids,
            "paper_ids": paper_ids,
            "source_anchor_ids": anchor_ids,
            "source_anchors": anchors,
            "source_claims": source_claims,
        }
        node_records.append(record)
        node_by_id[node_id] = record

    edge_records: list[dict[str, Any]] = []
    for edge in payload.get("edges", []):
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        left = node_by_id.get(source)
        right = node_by_id.get(target)
        source_claim_ids = sorted(set((left or {}).get("source_claim_ids", []) + (right or {}).get("source_claim_ids", [])))
        paper_ids = sorted({claim_index[qid]["paper_id"] for qid in source_claim_ids if qid in claim_index})
        same_cluster_edge = False
        source_key, target_key = _node_key(source), _node_key(target)
        if source_key and target_key and source_key[0] == target_key[0]:
            cluster_view = clusters.get(source_key[0].replace("lth_", "", 1))
            if cluster_view:
                s_norm, t_norm = _strip_id(source_key[1]), _strip_id(target_key[1])
                same_cluster_edge = any(
                    _strip_id(item.get("source")) == s_norm and _strip_id(item.get("target")) == t_norm
                    for item in cluster_view.get("edges", []) if isinstance(item, dict)
                )
        both_endpoints_resolved = bool(left and right and left.get("provenance_class") == "derived" and right.get("provenance_class") == "derived")
        edge_records.append(
            {
                "graph_edge_id": edge.get("id"),
                "source": source,
                "target": target,
                "semantic_type": edge.get("semantic_type") or (edge.get("details") or {}).get("relation"),
                "direct_cluster_edge_match": same_cluster_edge,
                "resolution": "derived_from_endpoint_closure" if both_endpoints_resolved else "unresolved",
                "provenance_class": "derived" if both_endpoints_resolved else "unresolved",
                "resolution_detail": (
                    "cluster_edge_with_derived_endpoint_closure" if both_endpoints_resolved and same_cluster_edge
                    else "merge_edge_with_derived_endpoint_closure" if both_endpoints_resolved
                    else "at_least_one_endpoint_unresolved"
                ),
                "source_claim_ids": source_claim_ids,
                "paper_ids": paper_ids,
                "note": "Final merge edges do not retain a direct source edge anchor; endpoint closure is reported explicitly.",
            }
        )

    resolved_nodes = sum(item["provenance_class"] == "derived" for item in node_records)
    resolved_edges = sum(item["provenance_class"] == "derived" for item in edge_records)
    direct_nodes = sum(item["provenance_class"] == "direct" for item in node_records)
    direct_edges = sum(item["provenance_class"] == "direct" for item in edge_records)
    all_anchor_records = [anchor for item in node_records for anchor in item["source_anchors"]]
    return {
        "schema_version": "provenance-map-v1",
        "authoritative_graph": str(html_path),
        "cluster_root": str(cluster_root),
        "data_root": str(data_root),
        "graph_metadata": payload.get("metadata") or {},
        "summary": {
            "graph_node_count": len(node_records),
            "graph_edge_count": len(edge_records),
            "resolved_node_count": resolved_nodes,
            "unknown_node_count": len(node_records) - resolved_nodes,
            "resolved_edge_count": resolved_edges,
            "unknown_edge_count": len(edge_records) - resolved_edges,
            "direct_node_count": direct_nodes,
            "direct_edge_count": direct_edges,
            "derived_node_count": resolved_nodes,
            "derived_edge_count": resolved_edges,
            "source_claim_count": len({qid for item in node_records for qid in item["source_claim_ids"]}),
            "source_paper_count": len({pid for item in node_records for pid in item["paper_ids"]}),
            "original_paper_anchor_count": sum(
                1 for item in node_records for anchor in item["source_anchors"] if anchor.get("start_line") is not None
            ),
            "claim_file_anchor_occurrence_count": sum(
                1 for anchor in all_anchor_records if str(anchor.get("anchor_id", "")).startswith("anchor_claim_")
            ),
            "unique_source_anchor_count": len({anchor.get("anchor_id") for anchor in all_anchor_records}),
        },
        "nodes": node_records,
        "edges": edge_records,
    }


def _report(mapping: dict[str, Any]) -> str:
    summary = mapping["summary"]
    lines = [
        "# merge.html provenance audit",
        "",
        "该报告只使用最终 `merge.html`、`outputs/merge_clusters/cluster_*/view_model.json` 和本地 `data/*` 文件。",
        "生成的候选 claim 通过 cluster 推理边反向追踪到论文 claim；没有路径的节点和边保留为 unknown。",
        "",
        "## 统计",
        "",
        f"- 图节点：{summary['graph_node_count']}；direct：{summary['direct_node_count']}；derived：{summary['derived_node_count']}；unresolved：{summary['unknown_node_count']}。",
        f"- 图边：{summary['graph_edge_count']}；direct：{summary['direct_edge_count']}；derived：{summary['derived_edge_count']}；unresolved：{summary['unknown_edge_count']}。",
        f"- 源 claim：{summary['source_claim_count']}；源论文：{summary['source_paper_count']}；有原文行号的 paragraph anchor：{summary['original_paper_anchor_count']} 次（唯一 anchor {summary['unique_source_anchor_count']}）；claim-file anchor：{summary['claim_file_anchor_occurrence_count']} 次。",
        "",
        "## 解释",
        "",
        "`resolved_derived` 表示图中的汇总 claim 由 cluster 推理边指向的一组源 claim 支持；它不是源论文的逐字引用。",
        "`anchor_claim_N` 只证明 claim-file 中的第 N 条 claim，不等于原文位置。只有 `anchor_paragraph_pXXXX` 且能在 `ocr.md` 中解析出行号时，才计入原文 anchor。",
        "最终 merge 边没有保留单独的 source edge anchor，因此边只报告两端节点的 source-claim closure，并明确标记为 derived。",
        "",
        "## unknown 节点",
        "",
    ]
    unknown = [item for item in mapping["nodes"] if item["provenance_class"] == "unresolved"]
    if not unknown:
        lines.append("无。")
    else:
        for item in unknown:
            lines.append(f"- `{item['graph_node_id']}` ({item.get('display_label') or ''})")
    return "\n".join(lines) + "\n"


def _report_json(mapping: dict[str, Any]) -> dict[str, Any]:
    """Return a compact machine-readable coverage/ambiguity audit."""

    nodes = mapping["nodes"]
    edges = mapping["edges"]
    unresolved_nodes = [
        {
            "graph_node_id": item["graph_node_id"],
            "display_label": item.get("display_label"),
            "resolution": item.get("resolution"),
            "reason": item.get("resolution_detail"),
        }
        for item in nodes
        if item.get("provenance_class") == "unresolved"
    ]
    unresolved_edges = [
        {
            "graph_edge_id": item.get("graph_edge_id"),
            "source": item.get("source"),
            "target": item.get("target"),
            "resolution": item.get("resolution"),
            "reason": item.get("resolution_detail"),
        }
        for item in edges
        if item.get("provenance_class") == "unresolved"
    ]
    summary = mapping["summary"]
    return {
        "schema_version": "provenance-map-report-v1",
        "authoritative_graph": mapping["authoritative_graph"],
        "sidecar": "provenance_mapping.json",
        "policy": {
            "direct": "No final merge node/edge is marked direct unless the final viewer itself stores a source path; none do.",
            "derived": "A candidate node is derived only when cluster reverse traversal reaches exact papers:paper_*::claim_* IDs present in data/.",
            "unresolved": "Any unmatched, ambiguous, manually wired, or endpoint-partial item is unresolved and receives no evidence credit.",
        },
        "summary": summary,
        "coverage": {
            "node_derived_fraction": summary["derived_node_count"] / summary["graph_node_count"] if summary["graph_node_count"] else 0.0,
            "edge_derived_fraction": summary["derived_edge_count"] / summary["graph_edge_count"] if summary["graph_edge_count"] else 0.0,
            "direct_node_fraction": 0.0,
            "direct_edge_fraction": 0.0,
        },
        "unresolved_nodes": unresolved_nodes,
        "unresolved_edges": unresolved_edges,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--report-json", type=Path)
    args = parser.parse_args(argv)
    mapping = build_map(args.html, args.clusters, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(_report(mapping), encoding="utf-8")
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(_report_json(mapping), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(mapping["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
