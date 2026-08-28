"""
test_run_pipeline_from_step2b.py

不需要真的调用14个步骤脚本,只验证:
  1. STEPS_FROM_STEP2B确实跳过了step1a/step1b/step2a,从step2b开始
  2. run_from_step2b()正确按顺序调用run_step(),任何一步失败就停止、
     不继续跑后面的步骤(复用run_full_pipeline.py已经测试过的run_step()
     本身,这里只测"从step2b开始跑"这一层薄薄的编排逻辑)

用法:
    python test_run_pipeline_from_step2b.py
"""

from unittest.mock import patch

import run_pipeline_from_step2b as from_step2b


def test_steps_from_step2b_skips_step1a_1b_2a():
    rel_paths = [path for path, _out, _api in from_step2b.STEPS_FROM_STEP2B]
    assert not any("step1a" in p for p in rel_paths)
    assert not any("step1b" in p for p in rel_paths)
    assert not any("step2a_check_pure_data" in p for p in rel_paths)
    assert rel_paths[0] == "step2_claim_completeness/step2b_complete_claims.py"


def test_steps_from_step2b_has_11_steps_ending_with_step4d():
    assert len(from_step2b.STEPS_FROM_STEP2B) == 11
    last_path = from_step2b.STEPS_FROM_STEP2B[-1][0]
    assert "step4d_normalize_instance_relations" in last_path


def test_run_from_step2b_calls_all_steps_in_order_when_all_succeed():
    called = []

    def fake_run_step(script_path, output_path, paper_id):
        called.append(script_path.name)
        return True

    with patch.object(from_step2b.sync_pipeline, "run_step", fake_run_step):
        ok = from_step2b.run_from_step2b("fake_paper")

    assert ok is True
    assert len(called) == 11
    assert called[0] == "step2b_complete_claims.py"
    assert called[-1] == "step4d_normalize_instance_relations.py"


def test_run_from_step2b_stops_at_first_failure():
    called = []

    def fake_run_step(script_path, output_path, paper_id):
        called.append(script_path.name)
        return script_path.name != "step2d_remove_instance_content.py"  # 这一步失败

    with patch.object(from_step2b.sync_pipeline, "run_step", fake_run_step):
        ok = from_step2b.run_from_step2b("fake_paper")

    assert ok is False
    # step2b, step2c, step2d 三步应该都调用过了(在第3步失败),
    # 后面的step2e及之后不该被调用
    assert called == [
        "step2b_complete_claims.py",
        "step2c_check_instance_containment.py",
        "step2d_remove_instance_content.py",
    ]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK: {t.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过")
