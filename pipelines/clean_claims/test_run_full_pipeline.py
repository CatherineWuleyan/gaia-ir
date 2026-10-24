"""
test_run_full_pipeline.py

不调用真实的14个步骤脚本(那需要真实API访问,而且沙盒没有网络),只测试
run_step()这个编排逻辑本身对不对——用真实写到磁盘上的、极简的"假步骤
脚本"(不是mock掉subprocess,是真的用subprocess跑一个真实存在的.py文件)
模拟三种关键场景:
  1. 正常成功(退出码0,且生成了预期文件)
  2. 退出码非0(模拟真实步骤跑失败的情况)
  3. 退出码是0,但没有生成预期文件(专门模拟step1b_organize_labeled_
     content.py在缺输入文件时sys.exit(0)的真实行为——这是run_step()
     存在的核心原因,必须单独测这一条)

用法:
    python test_run_full_pipeline.py
"""

import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import run_full_pipeline as pipeline


def _write_fake_step(dir_path: Path, filename: str, body: str) -> Path:
    path = dir_path / filename
    path.write_text(body, encoding="utf-8")
    return path


def test_success_case_returns_true():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        script = _write_fake_step(tmp_dir, "fake_ok.py", (
            "import sys\n"
            "paper_id = sys.argv[1]\n"
            "print(f'processed {paper_id}')\n"
        ))
        output_path = tmp_dir / "some_output.json"
        output_path.write_text("[]", encoding="utf-8")  # 模拟脚本自己产出了这个文件

        ok = pipeline.run_step(script, output_path, "fake_paper")
        assert ok is True
    finally:
        shutil.rmtree(tmp_dir)


def test_nonzero_exit_code_returns_false():
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        script = _write_fake_step(tmp_dir, "fake_fail.py", (
            "import sys\n"
            "print('something went wrong')\n"
            "sys.exit(1)\n"
        ))
        output_path = tmp_dir / "never_created.json"

        ok = pipeline.run_step(script, output_path, "fake_paper")
        assert ok is False
        assert not output_path.exists()
    finally:
        shutil.rmtree(tmp_dir)


def test_exit_zero_but_no_output_returns_false():
    """专门复现step1b_organize_labeled_content.py的真实行为:退出码是0
    (表示"正常跳过",不是出错),但因为上游输入缺失,压根没写出预期的
    输出文件——这种情况run_step()必须判定为失败,不能被"退出码是0"
    骗过去。"""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        script = _write_fake_step(tmp_dir, "fake_silent_skip.py", (
            "import sys\n"
            "print('跳过:找不到输入文件')\n"
            "sys.exit(0)\n"  # 关键:退出码是0,不是1
        ))
        output_path = tmp_dir / "should_have_been_created.json"  # 这个脚本不会创建它

        ok = pipeline.run_step(script, output_path, "fake_paper")
        assert ok is False  # 尽管退出码是0,也必须判失败
        assert not output_path.exists()
    finally:
        shutil.rmtree(tmp_dir)


def test_paper_id_is_passed_as_argv():
    """确认paper_id确实是通过命令行参数传给子进程的(跟全部14个真实
    步骤脚本"sys.argv[1]或input()"这个既定读取方式对应)。"""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        script = _write_fake_step(tmp_dir, "fake_echo_argv.py", (
            "import sys\n"
            "from pathlib import Path\n"
            "paper_id = sys.argv[1]\n"
            "Path('" + str(tmp_dir).replace('\\', '\\\\') + "/echoed.txt').write_text(paper_id)\n"
        ))
        output_path = tmp_dir / "echoed.txt"

        ok = pipeline.run_step(script, output_path, "my-specific-paper-id-123")
        assert ok is True
        assert output_path.read_text() == "my-specific-paper-id-123"
    finally:
        shutil.rmtree(tmp_dir)


def test_main_stops_at_first_failing_step_via_temp_pipeline():
    """构造一条3步的迷你流水线(第1步成功,第2步"退出码0但不产出文件",
    第3步理论上不该被跑到),验证main()式的"失败即停"逻辑——直接复用
    run_step()逐步调用,不重新实现一遍main()的参数解析部分。"""
    tmp_dir = Path(tempfile.mkdtemp())
    try:
        step1 = _write_fake_step(tmp_dir, "step1.py", (
            "from pathlib import Path\n"
            "Path(__file__).parent.joinpath('out1.json').write_text('[]')\n"
        ))
        step2 = _write_fake_step(tmp_dir, "step2.py", (
            "print('装作正常跳过')\n"
        ))  # 退出码默认是0,但不产出out2.json
        step3_called = tmp_dir / "step3_was_called.txt"
        step3 = _write_fake_step(tmp_dir, "step3.py", (
            f"from pathlib import Path\n"
            f"Path(r'{step3_called}').write_text('yes')\n"
        ))

        results = []
        for script, out_name in [(step1, "out1.json"), (step2, "out2.json"), (step3, "out3.json")]:
            ok = pipeline.run_step(script, tmp_dir / out_name, "fake_paper")
            results.append(ok)
            if not ok:
                break

        assert results == [True, False]  # 第3步不该被跑到
        assert not step3_called.exists()
    finally:
        shutil.rmtree(tmp_dir)


def test_noninteractive_step_receives_eof_instead_of_waiting_for_user():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        script = _write_fake_step(root, "needs_input.py", "input('retry?')\n")
        output = root / "partial.json"
        output.write_text("{}", encoding="utf-8")
        assert pipeline.run_step(script, output, "test", non_interactive=True) is False


def test_deepseek_groups_run_three_at_a_time_and_keep_input_order():
    import os
    import threading
    import claude_api_call as api

    barrier = threading.Barrier(3, timeout=5)
    lock = threading.Lock()
    active = peak = 0

    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        barrier.wait()
        with lock:
            active -= 1
        return value

    with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test"}):
        assert list(api._ordered_parallel_map(work, range(6))) == list(range(6))
    assert peak == 3


def test_deepseek_retries_only_the_failed_transport_request():
    import json
    import os
    import claude_api_call as api

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": "valid result"}}]}).encode()

    with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test", "DEEPSEEK_API_KEY": "test"}), \
            patch.object(api, "urlopen", side_effect=[TimeoutError("read timeout"), Response()]) as request, \
            patch.object(api.time, "sleep"):
        assert api.call_claude("test") == "valid result"
        assert request.call_count == 2


def test_label_retry_reuses_only_matching_validated_conclusions():
    import json
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent / "step1_process_conclusions"))
    import step1a_pipeline as step

    conclusions = [{"id": f"conclusion_{n}", "content": f"Claim {n}."} for n in range(1, 4)]

    def result(item):
        return dict(item, labeling_status="ok", labeled_segments={"segments": [
            {"text": item["content"], "label": "assertion", "number": 1}]})

    calls = []

    def process(item):
        calls.append(item["id"])
        return result(item)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paper = root / "test"
        paper.mkdir()
        (paper / "graph.json").write_text("{}")
        previous = [result(conclusions[0]), result(conclusions[1])]
        previous[1]["labeled_segments"]["segments"][0]["text"] = "Wrong source text."
        output = paper / "conclusions_labeled.json"
        output.write_text(json.dumps(previous))
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test", "_ARIADNE_RETRY_STEP": Path(step.__file__).name}), \
                patch.object(step, "DATA_DIR", root), patch.object(step.sys, "argv", ["step", "test"]), \
                patch.object(step, "extract_conclusions", return_value=conclusions), \
                patch.object(step, "process_conclusion", side_effect=process):
            step.main()
        assert sorted(calls) == ["conclusion_2", "conclusion_3"]
        assert [item["id"] for item in json.loads(output.read_text())] == [item["id"] for item in conclusions]
        calls.clear()
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test", "_ARIADNE_RETRY_STEP": ""}), \
                patch.object(step, "DATA_DIR", root), patch.object(step.sys, "argv", ["step", "test"]), \
                patch.object(step, "extract_conclusions", return_value=conclusions), \
                patch.object(step, "process_conclusion", side_effect=process):
            step.main()
        assert sorted(calls) == [item["id"] for item in conclusions]


def test_parallel_group_failure_is_not_converted_to_a_result():
    import os
    import claude_api_call as api

    def work(value):
        if value == 1:
            raise RuntimeError("failed group")
        return value

    with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test"}):
        try:
            list(api._ordered_parallel_map(work, range(5)))
        except RuntimeError as exc:
            assert str(exc) == "failed group"
        else:
            raise AssertionError("worker failures must stop the stage")


def test_deepseek_honors_rate_limit_delay_and_does_not_retry_auth_errors():
    import os
    from urllib.error import HTTPError
    import claude_api_call as api

    rate_limit = HTTPError("https://example.invalid", 429, "rate limit", {"Retry-After": "7"}, None)
    unauthorized = HTTPError("https://example.invalid", 401, "unauthorized", {}, None)
    with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test", "DEEPSEEK_API_KEY": "test"}), \
            patch.object(api, "urlopen", side_effect=[rate_limit, unauthorized]) as request, \
            patch.object(api.time, "sleep") as sleep:
        try:
            api.call_claude("test")
        except RuntimeError as exc:
            assert "401" in str(exc)
        else:
            raise AssertionError("authentication errors must fail")
        assert request.call_count == 2
        sleep.assert_called_once_with(7.0)


def test_completion_retry_rejects_invalid_cached_references():
    import json
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parent / "step2_claim_completeness"))
    import step2b_complete_claims as step

    calls = []

    def process(parts, numbers):
        calls.extend(numbers)
        return {number: {"完整表述": f"Claim {number}.", "需要更多上下文": []} for number in numbers}, "ok"

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        paper = root / "test"
        paper.mkdir()
        organized = [{"id": f"paper::conclusion_{n}", "organized_parts": [
            {"number": n, "label": "assertion", "content": f"Claim {n}."}]} for n in range(1, 4)]
        records = [{"conclusion": f"conclusion_{n}", "number": n, "text": f"Claim {n}.",
                    "是纯实验数据": False, "完整表述": f"Claim {n}.", "需要更多上下文": [],
                    "completion_status": "ok" if n < 3 else "fallback_original"} for n in range(1, 4)]
        records[1]["完整表述"] = "Self-reference 【2】."
        (paper / "organized_content.json").write_text(json.dumps(organized))
        output = paper / "claim_completeness_analysis.json"
        output.write_text(json.dumps(records))
        with patch.dict(os.environ, {"DEEPSEEK_MODEL": "test", "_ARIADNE_RETRY_STEP": Path(step.__file__).name}), \
                patch.object(step, "DATA_DIR", root), patch.object(step.sys, "argv", ["step", "test"]), \
                patch.object(step, "process_conclusion", side_effect=process):
            step.main()
        assert sorted(calls) == [2, 3]
        result = json.loads(output.read_text())
        assert [item["number"] for item in result] == [1, 2, 3]
        assert all(item["completion_status"] == "ok" for item in result)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  OK: {t.__name__}")
    print(f"\n全部 {len(tests)} 个测试通过")
