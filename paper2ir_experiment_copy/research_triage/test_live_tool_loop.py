from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_triage.final_graph import load_final_graph
from research_triage.live_tool_loop import LiveToolLoopError, ScriptedProvider, run_live_tool_loop


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research_triage" / "fixtures" / "lth_bridge_v2_task.json"
GRAPH_HTML = ROOT.parent / "merge.html"


def _task() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _proposal(task: dict, node_id: str) -> dict:
    return {
        "protocol_version": "2",
        "task_id": task["task_id"],
        "candidate_next_steps": [
            "Run a matched intervention contrast under the fixed task conditions.",
            "Collect an independent result under a second training schedule.",
        ],
        "next_step": "Run a matched intervention contrast under the fixed task conditions.",
        "reason": "The contrast separates two explanations while holding the stated conditions fixed.",
        "evidence": [{"ref_type": "graph_node", "ref_id": node_id, "role": "support"}],
        "conditions": [{"key": "dataset", "value": "CIFAR-10"}],
        "experiment_card": {
            "baseline": "dense reference and the current sparse baseline",
            "treatment": "the proposed matched intervention",
            "controls": ["same seed, architecture, data order, and evaluation protocol"],
            "metrics": ["test accuracy and convergence gap"],
            "entrypoint": None,
            "expected_observation": "The treatment changes the gap in the predicted direction.",
            "falsifier": "The matched treatment leaves the gap unchanged across seeds.",
            "budget": {"max_seconds": 600, "max_gpu_hours": 0},
        },
        "decision_rule": "If the treatment does not change the gap, test the independent schedule explanation.",
        "uncertainty": "The graph has no audited source anchors, so this is a retrieval audit rather than provenance credit.",
        # Deliberately incorrect model-reported IDs.  The runner-owned trace,
        # rather than this field, must decide evidence closure.
        "retrieval_trace": [{
            "tool": "search_claims",
            "arguments": {"query": "initialization sparsity", "k": 2},
            "returned_ids": ["model-invented-id"],
            "purpose": "model's untrusted summary of its search",
        }],
    }


def test_graph_nodes_are_retrieved_after_prompt_and_hidden_trace_authorises_refs():
    task = _task()
    graph = load_final_graph(GRAPH_HTML)
    hit = graph.search_claims("initialization sparsity", 1)[0]
    node_id = hit["node_id"]
    provider = ScriptedProvider([
        {"message": {"role": "assistant", "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "search_claims", "arguments": json.dumps({"query": "initialization sparsity", "k": 1})},
        }]}},
        {"message": {"role": "assistant", "content": json.dumps(_proposal(task, node_id), ensure_ascii=False)}},
    ])
    result = run_live_tool_loop(task, "reasoning_graph", provider, repo_root=ROOT.parent, graph=graph, seed=7)

    assert result.status == "ok"
    assert result.validation["ok"] is True
    assert result.validation["metrics"]["retrieval_trace_verified"] is True
    assert result.audit["actual_trace_used_for_validation"] is True
    assert result.audit["retrieval_context_preloaded"] is False
    actual_call = result.retrieval_trace["calls"][0]
    assert {ref["ref_id"] for ref in actual_call["returned_refs"] if ref["ref_type"] == "graph_node"} == {node_id}

    # The first model turn receives only the frozen resource metadata and tool
    # definitions.  No graph node/edge payload is present before the call.
    initial_prompt = provider.calls[0]["messages"][1]["content"]
    assert all(graph_id not in initial_prompt for graph_id in (node["id"] for node in graph.nodes))
    assert provider.calls[0]["tools"]
    assert {item["function"]["name"] for item in provider.calls[0]["tools"]} == {
        "search_paper", "search_external_text", "search_claims", "expand_neighbors", "get_source_anchor"
    }


def test_disallowed_tool_is_rejected_before_provider_can_bypass_condition():
    task = _task()
    provider = ScriptedProvider([{
        "message": {"role": "assistant", "tool_calls": [{
            "id": "bad",
            "type": "function",
            "function": {"name": "search_external_text", "arguments": {"query": "anything", "k": 1}},
        }]},
    }])
    with pytest.raises(LiveToolLoopError, match="not available in paper_only"):
        run_live_tool_loop(task, "paper_only", provider, repo_root=ROOT.parent)


def test_tool_budget_is_enforced_by_runner_owned_session():
    task = _task()
    task["retrieval"] = {"max_calls": 0, "max_return_tokens": 4000}
    provider = ScriptedProvider([{
        "message": {"role": "assistant", "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "search_paper", "arguments": {"query": "ResNet", "k": 1}},
        }]},
    }])
    with pytest.raises(ValueError, match="retrieval call budget exceeded"):
        run_live_tool_loop(task, "paper_only", provider, repo_root=ROOT.parent)

