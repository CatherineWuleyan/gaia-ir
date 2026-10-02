from __future__ import annotations

from research_triage.v2_protocol import (
    V2_CONDITIONS,
    build_prompt,
    condition_input,
    validate_v2,
)


def _task():
    return {
        "task_id": "v2-demo",
        "question": "Which intervention should be tested next?",
        "hypothesis": "The intervention changes the measured outcome.",
        "target_paper": {
            "paper_id": "paper-target",
            "title": "Target",
            "anchors": [
                {"anchor_id": "paper-target:p1", "text": "The target result."},
            ],
        },
        "external_text": [
            {"anchor_id": "paper-other:p1", "text": "An external result."},
        ],
        "graph": {
            "snapshot_sha256": "snapshot",
            "nodes": [{"node_id": "n1", "summary": "A graph result."}],
            "edges": [{"edge_id": "e1", "source": "n1", "target": "n1"}],
        },
        "execution": {"max_seconds": 10, "max_gpu_hours": 0},
        "allowed_dataset_refs": ["dataset"],
        "allowed_entrypoint_refs": [],
    }


def _proposal(**updates):
    result = {
        "protocol_version": "2",
        "task_id": "v2-demo",
        "candidate_next_steps": ["replicate", "compare"],
        "next_step": "Compare the intervention against baseline.",
        "reason": "The contrast isolates the stated hypothesis.",
        "evidence": [{"ref_type": "paper_anchor", "ref_id": "paper-target:p1", "role": "support"}],
        "conditions": [{"key": "dataset", "value": "dataset"}],
        "experiment_card": {
            "baseline": "baseline",
            "treatment": "intervention",
            "controls": ["matched control"],
            "metrics": ["accuracy"],
            "entrypoint": None,
            "expected_observation": "treatment improves accuracy",
            "falsifier": "no accuracy improvement",
            "budget": {"max_seconds": 10, "max_gpu_hours": 0},
        },
        "decision_rule": "If accuracy does not improve, test an alternative.",
        "uncertainty": "The external result may not transfer.",
        "retrieval_trace": [],
    }
    result.update(updates)
    return result


def test_all_conditions_keep_full_paper_and_hide_legacy_action_space():
    for condition in V2_CONDITIONS:
        value = condition_input(_task(), condition)
        assert value["target_paper"]["anchors"][0]["text"] == "The target result."
        assert "allowed_actions" not in value
        assert "allowed_claims" not in value
        assert "candidate_experiments" not in value
        assert "action_requirements" not in value
        # The prose may mention the word action while explaining that it is
        # deliberately not a schema field; no JSON action key is emitted.
        assert '"action":' not in build_prompt(_task(), condition)


def test_v2_proposal_accepts_free_form_next_step():
    result = validate_v2(_task(), _proposal(), "paper_only")
    assert result["ok"] is True
    assert result["metrics"]["canonical_action"] == "contrast"


def test_graph_evidence_requires_a_retrieval_return():
    proposal = _proposal(
        evidence=[{"ref_type": "graph_node", "ref_id": "n1", "role": "support"}],
    )
    result = validate_v2(_task(), proposal, "reasoning_graph")
    assert result["ok"] is False
    assert any("without a retrieval_trace return" in error for error in result["errors"])
    proposal["retrieval_trace"] = [{
        "tool": "search_claims",
        "arguments": {"query": "intervention"},
        "returned_ids": ["n1"],
        "purpose": "find a cross-paper mechanism",
    }]
    result = validate_v2(_task(), proposal, "reasoning_graph")
    assert result["ok"] is True


def test_actual_retrieval_trace_is_authoritative_for_graph_refs():
    proposal = _proposal(
        evidence=[{"ref_type": "graph_node", "ref_id": "n1", "role": "support"}],
        retrieval_trace=[{
            "tool": "search_claims",
            "arguments": {"query": "intervention"},
            "returned_ids": ["n1"],
            "purpose": "model reported a hit",
        }],
    )
    result = validate_v2(
        _task(), proposal, "reasoning_graph", actual_retrieval_trace=[{
            "tool": "search_claims", "arguments": {"query": "intervention"},
            "returned_ids": [], "purpose": "runner trace",
        }],
    )
    assert result["ok"] is False
    assert result["metrics"]["retrieval_trace_verified"] is True
