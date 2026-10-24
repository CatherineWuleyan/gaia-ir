from pathlib import Path

import json
import hashlib
import re

import pytest

from final_graph import FinalGraphError, RetrievalSession, load_final_graph, validate_snapshot_file


ROOT = Path(__file__).resolve().parents[3]


def test_merge_html_is_the_locked_final_graph():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    audit = graph.payload["metadata"]
    assert len(graph.nodes) == 240
    assert len(graph.edges) == 235
    assert graph.html_sha256
    assert graph.embedded_sha256
    assert audit["merge"] is True


def test_final_graph_does_not_fabricate_missing_provenance():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    result = graph.get_source_anchor("p01:anchor_claim_5")
    assert result["status"] == "unavailable"
    assert graph.payload["source_targets"] == []


def test_model_payload_contains_every_authoritative_node_and_edge():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    payload = graph.model_payload()
    assert payload["source"] == "authoritative_merge_html"
    assert payload["canonical_payload_sha256"] == graph.canonical_sha256
    assert len(payload["nodes"]) == len(graph.nodes) == 240
    assert len(payload["edges"]) == len(graph.edges) == 235
    assert {item["id"] for item in payload["nodes"]} == {item["id"] for item in graph.nodes}
    assert {item["id"] for item in payload["edges"]} == {item["id"] for item in graph.edges}


def test_model_payload_can_exclude_summary_nodes_without_changing_authoritative_graph():
    graph = load_final_graph(
        ROOT / "outputs/lth/final/merge.html",
        provenance_path=ROOT / "evidence/lth_final_graph_20261002/provenance_mapping.json",
    )
    payload = graph.model_payload(include_summary_nodes=False)
    assert payload["graph_filter"]["mode"] == "summary_excluded"
    assert payload["graph_filter"]["include_summary_nodes"] is False
    assert payload["graph_filter"]["full_graph_node_count"] == 240
    assert len(payload["nodes"]) < 240
    assert all(not graph._is_summary_node_id(str(node["id"])) for node in payload["nodes"])
    assert len(payload["edges"]) <= 235
    assert len(graph.nodes) == 240
    metadata_text = json.dumps(payload.get("metadata"), ensure_ascii=False)
    assert "K02" not in metadata_text
    assert "K03" not in metadata_text


def test_compact_model_payload_preserves_evidence_and_strips_viewer_metadata():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    standard = graph.model_payload()
    compact = graph.model_payload(serialization="compact")
    assert compact["schema_version"] == "merge_graph_model_payload_compact_v1"
    assert len(compact["nodes"]) == len(standard["nodes"])
    assert len(compact["edges"]) == len(standard["edges"])
    assert {node["id"] for node in compact["nodes"]} == {node["id"] for node in standard["nodes"]}
    assert {edge["id"] for edge in compact["edges"]} == {edge["id"] for edge in standard["edges"]}
    assert all(set(node) == {"id", "summary", "source_anchor_ids"} for node in compact["nodes"])
    assert all(set(edge) == {"id", "source", "target", "label", "semantic_type"} for edge in compact["edges"])
    assert "metadata" not in compact
    assert "layers" not in compact
    assert compact["canonical_payload_sha256"] == standard["canonical_payload_sha256"]


def test_matching_standalone_graph_index_is_accepted():
    graph = load_final_graph(
        ROOT / "outputs/lth/final/merge.html",
        index_path=ROOT / "outputs/lth/final/view_model.reference.json",
    )
    assert graph.index_sha256
    assert graph.canonical_sha256 == "9deaac0d4767d4f75e98f6383fc238513643708d37c7ed577e5bab68c629d74d"
    assert len(graph.nodes) == 240
    assert len(graph.edges) == 235


def test_stale_standalone_graph_index_is_rejected(tmp_path):
    stale = tmp_path / "stale.json"
    payload = json.loads((ROOT / "outputs/lth/final/view_model.reference.json").read_text(encoding="utf-8"))
    payload["title"] = "stale index"
    stale.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalGraphError, match="does not match merge.html"):
        load_final_graph(ROOT / "outputs/lth/final/merge.html", index_path=stale)


def test_companion_knowledge_index_is_registered_with_the_matching_merge_ir():
    evidence = ROOT / "evidence/lth_final_graph_20261002"
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
        (ROOT / "outputs/lth/final/gaia.ir.json").read_bytes()
    ).hexdigest()


def test_search_and_neighbor_retrieval_are_deterministic():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    hits = graph.search_claims("optimization stability rewinding", k=5)
    assert hits
    assert any(hit["label"] == "K03" for hit in hits)
    neighbors = graph.expand_neighbors(hits[0]["node_id"], k=5)
    assert isinstance(neighbors, list)
    assert all("edge" in item and "source" in item and "target" in item for item in neighbors)


def test_retrieval_session_records_budget_and_missing_provenance_warning():
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    session = RetrievalSession(graph, max_calls=2, max_return_tokens=2000)
    hits = session.search_claims("optimization stability", k=2)
    assert hits
    anchor = session.get_source_anchor("a_missing")
    assert anchor["status"] == "unavailable"
    trace = session.trace()
    assert trace["call_count"] == 2
    assert trace["returned_tokens"] > 0
    assert trace["warnings"]


def test_retrieval_session_allowlist_blocks_hidden_paper_records():
    graph = load_final_graph(
        ROOT / "outputs/lth/final/merge.html",
        provenance_path=ROOT / "evidence/lth_final_graph_20261002/provenance_mapping.json",
    )
    sidecar = json.loads((ROOT / "evidence/lth_final_graph_20261002/provenance_mapping.json").read_text())
    target = "867750889056633390"
    all_ids = {
        str(paper_id)
        for record in [*sidecar["nodes"], *sidecar["edges"]]
        for paper_id in record.get("paper_ids", [])
    }
    session = RetrievalSession(graph, max_calls=100, max_return_tokens=1_000_000, eligible_paper_ids=all_ids - {target})
    results = session.search_claims("elastic transfer ticket", 10)
    assert all(target not in str(item) for item in results)
    for item in results:
        assert target not in str(session.expand_neighbors(item["node_id"], 10))
    target_node = next(
        record["graph_node_id"]
        for record in sidecar["nodes"]
        if target in {str(value) for value in record.get("paper_ids", [])}
    )
    assert session.expand_neighbors(target_node, 10) == []
    anchor_id = next(
        anchor["anchor_id"]
        for record in sidecar["nodes"]
        if target in {str(value) for value in record.get("paper_ids", [])}
        for anchor in record.get("source_anchors", [])
    )
    anchor = session.get_source_anchor(anchor_id)
    assert target not in json.dumps(anchor)
    assert anchor["status"] in {"available", "unavailable"}
    assert all(target not in json.dumps(match) for match in anchor.get("source_matches", []))
    assert session.trace()["fail_closed_filter"] is True


def test_provenance_sidecar_is_explicit_and_remains_derived():
    graph = load_final_graph(
        ROOT / "outputs/lth/final/merge.html",
        provenance_path=ROOT / "evidence/lth_final_graph_20261002/provenance_mapping.json",
    )
    hits = graph.search_claims("initialization sparsity", k=1)
    assert hits
    assert hits[0]["source_anchor_ids"]
    assert hits[0]["provenance_class"] == "derived"
    anchor = graph.get_source_anchor(hits[0]["source_anchor_ids"][0])
    assert anchor["status"] == "available"
    assert anchor["provenance_class"] == "derived"
    assert "not direct" in anchor["warning"]


def test_qualified_source_anchor_resolves_one_paper_path():
    evidence = ROOT / "evidence/lth_final_graph_20261002"
    graph = load_final_graph(
        ROOT / "outputs/lth/final/merge.html",
        provenance_path=evidence / "provenance_mapping.json",
    )
    sidecar = json.loads((evidence / "provenance_mapping.json").read_text(encoding="utf-8"))
    selected = None
    for record in sidecar["nodes"]:
        for source in record.get("source_anchors", []):
            path = str(source.get("source_path") or "")
            anchor_id = source.get("anchor_id")
            match = re.search(r"data/([^/]+)/ocr\.md", path)
            if match and isinstance(anchor_id, str):
                selected = (match.group(1), anchor_id)
                break
        if selected:
            break
    assert selected is not None
    paper_id, anchor_id = selected
    result = graph.get_source_anchor(f"{paper_id}:{anchor_id}")
    assert result["status"] == "available"
    assert result["provenance_class"] == "derived"
    assert result["source_matches"]
    assert all(paper_id in str(item.get("source_path")) for item in result["source_matches"])


def test_provenance_sidecar_must_be_hash_bound_and_complete(tmp_path):
    source = ROOT / "evidence/lth_final_graph_20261002/provenance_mapping.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["nodes"] = payload["nodes"][:-1]
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalGraphError, match="node coverage"):
        load_final_graph(ROOT / "outputs/lth/final/merge.html", provenance_path=partial)

    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["graph_hashes"]["canonical_payload_sha256"] = "0" * 64
    mismatched = tmp_path / "mismatched.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FinalGraphError, match="not bound"):
        load_final_graph(ROOT / "outputs/lth/final/merge.html", provenance_path=mismatched)


def test_declared_snapshot_is_bound_to_authoritative_graph(tmp_path):
    graph = load_final_graph(ROOT / "outputs/lth/final/merge.html")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(json.dumps(graph.payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    audit = validate_snapshot_file(snapshot, graph)
    assert len(audit["sha256"]) == 64
    assert audit["canonical_sha256"] == graph.canonical_sha256
    snapshot.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    with pytest.raises(FinalGraphError, match="canonical payload"):
        validate_snapshot_file(snapshot, graph)
