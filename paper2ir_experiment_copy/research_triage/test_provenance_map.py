from pathlib import Path

from research_triage.provenance_map import build_map


ROOT = Path(__file__).resolve().parents[2]


def test_merge_graph_provenance_is_conservative_and_joined():
    mapping = build_map(
        ROOT / "merge.html",
        ROOT / "outputs" / "merge_clusters",
        ROOT / "data",
    )
    summary = mapping["summary"]
    assert summary["graph_node_count"] == 240
    assert summary["graph_edge_count"] == 235
    assert summary["derived_node_count"] == 220
    assert summary["direct_node_count"] == 0
    assert summary["unknown_node_count"] == 20
    assert summary["derived_edge_count"] == 125
    assert summary["unknown_edge_count"] == 110
    resolved = next(node for node in mapping["nodes"] if node["provenance_class"] == "derived")
    assert resolved["source_claim_ids"]
    assert resolved["paper_ids"]
    assert resolved["source_anchors"]
    assert any(anchor.get("start_line") is not None for anchor in resolved["source_anchors"])
    assert any(claim.get("formalization") for claim in resolved["source_claims"])
    assert any(claim.get("claims_final") for node in mapping["nodes"] for claim in node["source_claims"])
    unresolved = [node for node in mapping["nodes"] if node["provenance_class"] == "unresolved"]
    assert unresolved
    assert all(not node["source_claim_ids"] for node in unresolved)

