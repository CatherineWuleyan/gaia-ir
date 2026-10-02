import json
from pathlib import Path

from research_triage.v2_plan import create_plan


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "research_triage" / "fixtures" / "lth_bridge_v2_task.json"
GRAPH = ROOT.parent / "merge.html"


def test_plan_materialises_three_conditions_without_requests(tmp_path):
    output = tmp_path / "plan"
    manifest = create_plan(
        fixture=FIXTURE,
        graph_html=GRAPH,
        output_dir=output,
        seeds=(0, 1, 2),
        model_profiles=("test_model",),
    )
    assert manifest["status"] == "planned_authorization_required"
    assert manifest["api_requests_made"] == 0
    assert manifest["target_cells"] == 9
    assert manifest["graph_provenance_status"] == "missing"
    for cell in manifest["cells"]:
        value = json.loads((output / cell["model_input"]).read_text(encoding="utf-8"))
        text = json.dumps(value, ensure_ascii=False)
        assert "allowed_actions" not in text
        assert "evaluation_oracle" not in text
        assert "candidate_experiments" not in text
