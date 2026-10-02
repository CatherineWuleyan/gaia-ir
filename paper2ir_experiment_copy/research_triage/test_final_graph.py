from pathlib import Path

import json
import hashlib

import pytest

from .final_graph import FinalGraphError, RetrievalSession, load_final_graph


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


def test_matching_standalone_graph_index_is_accepted():
    graph = load_final_graph(
        ROOT / "merge.html",
        index_path=ROOT / "outputs/merge_corpus/view_model.reference.json",
    )
    assert graph.index_sha256
    assert graph.canonical_sha256 == "9deaac0d4767d4f75e98f6383fc238513643708d37c7ed577e5bab68c629d74d"
    assert len(graph.nodes) == 240
    assert len(graph.edges) == 235


def test_stale_standalone_graph_index_is_rejected(tmp_path):
    stale = tmp_path / "stale.json"
    payload = json.loads((ROOT / "outputs/merge_corpus/view_model.reference.json").read_text(encoding="utf-8"))
    payload["title"] = "stale index"
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalGraphError, match="does not match merge.html"):
        load_final_graph(ROOT / "merge.html", index_path=stale)


def test_companion_knowledge_index_is_registered_with_the_matching_merge_ir():
    evidence = ROOT / "paper2ir_experiment_copy/evidence/lth_final_graph_20261002"
    knowledge = json.loads((evidence / "knowledge.index.json").read_text(encoding="utf-8"))
    artifacts = json.loads((evidence / "artifact_index.json").read_text(encoding="utf-8"))
    by_kind = {item["kind"]: item for item in artifacts["artifacts"]}
    assert knowledge["schema_name"] == "gaia.knowledge.index"
    assert len(knowledge["entries"]) == 71
    assert len(knowledge["source_anchors"]) == 57
    assert by_kind["knowledge.index"]["sha256"] == hashlib.sha256(
        (evidence / "knowledge.index.json").read_bytes()
    ).hexdigest()
    assert by_kind["gaia.ir"]["sha256"] == hashlib.sha256(
        (ROOT / "outputs/merge_corpus/gaia.ir.json").read_bytes()
    ).hexdigest()


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


def test_provenance_sidecar_is_explicit_and_remains_derived():
    graph = load_final_graph(
        ROOT / "merge.html",
        provenance_path=ROOT / "paper2ir_experiment_copy/evidence/lth_final_graph_20261002/provenance_mapping.json",
    )
    hits = graph.search_claims("initialization sparsity", k=1)
    assert hits
    assert hits[0]["source_anchor_ids"]
    assert hits[0]["provenance_class"] == "derived"
    anchor = graph.get_source_anchor(hits[0]["source_anchor_ids"][0])
    assert anchor["status"] == "available"
    assert anchor["provenance_class"] == "derived"
    assert "not direct" in anchor["warning"]
