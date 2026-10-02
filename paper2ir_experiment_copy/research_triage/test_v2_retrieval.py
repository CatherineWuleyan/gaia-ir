import json
from pathlib import Path

from research_triage.final_graph import load_final_graph
from research_triage.v2_protocol import build_prompt, condition_input
from research_triage.v2_retrieval import V2RetrievalSession, external_index, paper_index


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research_triage" / "fixtures" / "lth_bridge_v2_task.json"
GRAPH_HTML = ROOT.parent / "merge.html"


def load_task():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_v2_fixture_keeps_answers_out_of_model_input():
    task = load_task()
    for condition in ("paper_only", "external_text", "reasoning_graph"):
        value = condition_input(task, condition)
        text = json.dumps(value, ensure_ascii=False)
        assert "allowed_actions" not in text
        assert "evaluation_oracle" not in text
        assert "candidate_experiments" not in text
        assert "p53_claim_" not in text
        assert value["target_paper"]["text"] == task["target_paper"]["text"]


def test_text_tools_are_deterministic_and_budgeted():
    task = load_task()
    paper = paper_index(task)
    assert paper.search("Resnet-18 CIFAR10", 2)
    external = external_index(task, repo_root=ROOT.parent)
    result_a = external.search("lottery ticket rewinding", 3)
    result_b = external.search("lottery ticket rewinding", 3)
    assert result_a == result_b
    session = V2RetrievalSession(task=task, condition="external_text", repo_root=ROOT.parent, max_calls=2, max_return_tokens=4000)
    session.search_paper("Resnet-18", 1)
    session.search_external_text("rewinding", 1)
    trace = session.trace()
    assert trace["call_count"] == 2
    assert all(call["returned_ids"] for call in trace["calls"])


def test_graph_tool_preserves_unavailable_provenance():
    task = load_task()
    graph = load_final_graph(GRAPH_HTML)
    session = V2RetrievalSession(task=task, condition="reasoning_graph", repo_root=ROOT.parent, graph=graph, max_calls=3, max_return_tokens=3000)
    results = session.search_claims("initialization sparsity", 2)
    assert results
    anchor = session.get_source_anchor("anchor_that_does_not_exist")
    assert anchor["status"] == "unavailable"
    assert session.trace()["warnings"]
