import json
from pathlib import Path

from research_triage.final_graph import load_final_graph
from research_triage.v2_fairness import audit_v2_fairness
from research_triage.v2_plan import create_plan


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research_triage" / "fixtures" / "lth_bridge_v2_task.json"
BINGO_FIXTURE = ROOT / "research_triage" / "fixtures" / "lth_bridge_v2_bingo_task.json"
GRAPH_HTML = ROOT.parent / "merge.html"


def _task():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_lth_v2_conditions_are_information_matched_except_declared_tools():
    task = _task()
    graph = load_final_graph(GRAPH_HTML)
    report = audit_v2_fairness(task, graph=graph, repo_root=ROOT.parent)
    assert report["ok"] is True, report["errors"]
    assert report["metrics"]["external_paper_count"] == 88
    assert report["metrics"]["indexed_external_paper_count"] == 88
    assert report["metrics"]["focal_paper_id"] not in {
        item.get("paper_id") or item.get("source_id")
        for item in task["external_sources"]
    }
    assert report["metrics"]["retrieval_budget"] == {
        "paper_only": (8, 4000),
        "external_text": (8, 4000),
        "reasoning_graph": (8, 4000),
    }


def test_full_paper_is_identical_and_evaluator_fields_are_hidden():
    task = _task()
    graph = load_final_graph(GRAPH_HTML)
    report = audit_v2_fairness(task, graph=graph, repo_root=ROOT.parent)
    assert report["checks"]
    by_name = {item["name"]: item for item in report["checks"]}
    assert by_name["same_full_target_paper"]["ok"] is True
    assert by_name["no_evaluator_fields"]["ok"] is True
    assert by_name["no_known_answer_ids"]["ok"] is True


def test_audit_makes_graph_provenance_and_paper_answer_leak_explicit():
    task = _task()
    report = audit_v2_fairness(task, graph=load_final_graph(GRAPH_HTML), repo_root=ROOT.parent)
    assert report["metrics"]["graph_provenance_status"] == "missing"
    assert report["metrics"]["graph_anchor_count"] == 0
    assert report["metrics"]["likely_answer_terms_present"]["rewinding"] > 0
    assert any("provenance" in warning for warning in report["warnings"])
    assert any("answer terms" in warning for warning in report["warnings"])
    assert report["metrics"]["bridge_suitability"] == "warning"


def test_plan_manifest_persists_fairness_audit(tmp_path):
    manifest = create_plan(
        fixture=FIXTURE,
        graph_html=GRAPH_HTML,
        output_dir=tmp_path / "plan",
        seeds=(0,),
        model_profiles=("test_model",),
    )
    assert manifest["fairness_audit"]["ok"] is True
    assert manifest["fairness_audit"]["metrics"]["external_paper_count"] == 88
    persisted = json.loads((tmp_path / "plan" / "manifest.json").read_text(encoding="utf-8"))
    assert persisted["fairness_audit"]["metrics"]["retrieval_budget"]["external_text"] == [8, 4000]


def test_bingo_fixture_is_a_bridge_candidate_without_answer_term_warning():
    task = json.loads(BINGO_FIXTURE.read_text(encoding="utf-8"))
    report = audit_v2_fairness(task, graph=load_final_graph(GRAPH_HTML), repo_root=ROOT.parent)
    assert report["ok"] is True, report["errors"]
    assert report["metrics"]["bridge_suitability"] == "candidate"
    assert report["metrics"]["external_paper_count"] == 87
    assert report["metrics"]["focal_paper_id"] == "1130114433545142316"
