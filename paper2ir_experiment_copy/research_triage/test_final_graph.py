from pathlib import Path

from .final_graph import RetrievalSession, load_final_graph


ROOT = Path(__file__).resolve().parents[2]


def test_merge_html_is_the_locked_final_graph():
    graph = load_final_graph(ROOT / "merge.html")
    audit = graph.payload["metadata"]
    assert len(graph.nodes) == 240
    assert len(graph.edges) == 235
    assert graph.html_sha256
    assert graph.embedded_sha256
    assert audit["merge"] is True


def test_final_graph_does_not_fabricate_missing_provenance():
    graph = load_final_graph(ROOT / "merge.html")
    result = graph.get_source_anchor("p01:anchor_claim_5")
    assert result["status"] == "unavailable"
    assert graph.payload["source_targets"] == []


def test_search_and_neighbor_retrieval_are_deterministic():
    graph = load_final_graph(ROOT / "merge.html")
    hits = graph.search_claims("optimization stability rewinding", k=5)
    assert hits
    assert any(hit["label"] == "K03" for hit in hits)
    neighbors = graph.expand_neighbors(hits[0]["node_id"], k=5)
    assert isinstance(neighbors, list)
    assert all("edge" in item and "source" in item and "target" in item for item in neighbors)


def test_retrieval_session_records_budget_and_missing_provenance_warning():
    graph = load_final_graph(ROOT / "merge.html")
    session = RetrievalSession(graph, max_calls=2, max_return_tokens=2000)
    hits = session.search_claims("optimization stability", k=2)
    assert hits
    anchor = session.get_source_anchor("a_missing")
    assert anchor["status"] == "unavailable"
    trace = session.trace()
    assert trace["call_count"] == 2
    assert trace["returned_tokens"] > 0
    assert trace["warnings"]
