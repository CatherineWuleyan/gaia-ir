from __future__ import annotations

import copy
import io
import json
import ssl
import tempfile
import unittest
from email.utils import formatdate
from http.client import IncompleteRead
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

from agent_pipeline_v2.compiler_projection import project_for_official_compiler
from agent_pipeline_v2.step2 import (
    DeepSeekV4FlashObservationTool, MODEL_NAME, _best_paragraph_anchor,
    _deduplicate_extractions, _experiment_candidates, _observation_binding,
    _request_budget, _request_json, _RequestBudgetExceeded,
    _judge_duplicate_observations, _tool_call,
)
from pipeline_harness.domain.tools import ToolCallRequest, ToolCallResponse
from pipeline_harness.runner import run_pipeline
from pipeline_harness.store import RunStore, atomic_write_json, read_json
from pipeline_harness.view.projector import project_run


class RequestJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.request = Request("https://example.invalid/chat/completions", data=b"{}")
        self.clock = self.enterContext(patch("agent_pipeline_v2.step2.time.monotonic", return_value=100.0))
        self.enterContext(patch("agent_pipeline_v2.step2.time.time", return_value=1700000000.0))
        self.sleep = self.enterContext(patch("agent_pipeline_v2.step2.time.sleep"))
        self.enterContext(patch("agent_pipeline_v2.step2.random.uniform", return_value=0.25))

    def test_rate_limit_respects_retry_after_seconds_and_http_date(self) -> None:
        for retry_after in ("7", formatdate(1700000007, usegmt=True)):
            with self.subTest(retry_after=retry_after):
                self.sleep.reset_mock()
                limited = HTTPError(self.request.full_url, 429, "rate limited", {"Retry-After": retry_after}, None)
                with patch("agent_pipeline_v2.step2.urlopen", side_effect=[limited, io.BytesIO(b'{"ok":true}')]) as send:
                    self.assertEqual({"ok": True}, _request_json(self.request))
                self.assertEqual(2, send.call_count)
                self.sleep.assert_called_once_with(7.0)

    def test_read_timeout_and_incomplete_body_retry_the_whole_request(self) -> None:
        class BrokenResponse(io.BytesIO):
            def __init__(self, error: Exception) -> None:
                super().__init__(b"")
                self.error = error

            def read(self, *args: object, **kwargs: object) -> bytes:
                raise self.error

        for failure in (TimeoutError("read timed out"), IncompleteRead(b'{"ok":', 5)):
            with self.subTest(failure=type(failure).__name__):
                self.sleep.reset_mock()
                broken = BrokenResponse(failure)
                with patch("agent_pipeline_v2.step2.urlopen", side_effect=[broken, io.BytesIO(b'{"ok":true}')]) as send:
                    self.assertEqual({"ok": True}, _request_json(self.request))
                self.assertTrue(broken.closed)
                self.assertEqual(2, send.call_count)
                self.sleep.assert_called_once_with(1.25)

    def test_permanent_http_and_certificate_failures_are_not_retried(self) -> None:
        failures = [HTTPError(self.request.full_url, code, "permanent", {}, None)
                    for code in (400, 401, 402, 403, 404, 501)]
        failures.append(URLError(ssl.SSLCertVerificationError("certificate verify failed")))
        for failure in failures:
            with self.subTest(failure=str(failure)):
                with patch("agent_pipeline_v2.step2.urlopen", side_effect=failure) as send:
                    with self.assertRaises(type(failure)) as raised:
                        _request_json(self.request)
                self.assertIs(failure, raised.exception)
                self.assertEqual(1, send.call_count)
        self.sleep.assert_not_called()

    def test_transient_transport_exhaustion_preserves_third_exception(self) -> None:
        failures = [URLError(TimeoutError(f"timeout {index}")) for index in range(3)]
        with patch("agent_pipeline_v2.step2.urlopen", side_effect=failures) as send:
            with self.assertRaises(URLError) as raised:
                _request_json(self.request)
        self.assertIs(failures[-1], raised.exception)
        self.assertEqual(3, send.call_count)
        self.assertEqual([1.25, 2.25], [call.args[0] for call in self.sleep.call_args_list])

    def test_recoverable_http_statuses_are_retried(self) -> None:
        for code in (408, 500, 502, 503, 504):
            with self.subTest(code=code):
                failure = HTTPError(self.request.full_url, code, "transient", {}, None)
                with patch("agent_pipeline_v2.step2.urlopen", side_effect=[failure, io.BytesIO(b'{}')]) as send:
                    self.assertEqual({}, _request_json(self.request))
                self.assertEqual(2, send.call_count)

    def test_initial_call_and_repair_share_four_attempt_limit(self) -> None:
        responses = [TimeoutError("first"), TimeoutError("second"), io.BytesIO(b'{}'), io.BytesIO(b'{}')]
        with patch("agent_pipeline_v2.step2.urlopen", side_effect=responses) as send:
            with _request_budget():
                self.assertEqual({}, _request_json(self.request))
                with _request_budget():
                    self.assertEqual({}, _request_json(self.request))
                with self.assertRaises(_RequestBudgetExceeded) as raised:
                    _request_json(self.request)
            self.assertNotIsInstance(raised.exception, ValueError)
            self.assertEqual(4, send.call_count)
        # A completed task does not consume the next task's budget.
        with patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(b'{}')) as send:
            self.assertEqual({}, _request_json(self.request))
            send.assert_called_once()

    def test_invalid_json_is_not_transport_retried_and_can_be_repaired(self) -> None:
        with patch("agent_pipeline_v2.step2.urlopen", side_effect=[io.BytesIO(b'{'), io.BytesIO(b'{"fixed":true}')]) as send:
            with _request_budget():
                with self.assertRaises(json.JSONDecodeError):
                    _request_json(self.request)
                self.assertEqual(1, send.call_count)
                self.assertEqual({"fixed": True}, _request_json(self.request))
        self.sleep.assert_not_called()

    def test_deadline_bounds_new_request_timeout_and_blocks_later_calls(self) -> None:
        with patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(b'{}')) as send:
            with _request_budget():
                self.clock.return_value = 699.5
                self.assertEqual({}, _request_json(self.request))
                send.assert_called_once_with(self.request, timeout=0.5)
                self.clock.return_value = 700.0
                with self.assertRaises(_RequestBudgetExceeded):
                    _request_json(self.request)
                self.assertEqual(1, send.call_count)

    def test_retry_after_beyond_deadline_fails_without_early_retry(self) -> None:
        failure = HTTPError(self.request.full_url, 429, "rate limited", {"Retry-After": "601"}, None)
        with patch("agent_pipeline_v2.step2.urlopen", side_effect=failure) as send:
            with self.assertRaises(_RequestBudgetExceeded) as raised:
                _request_json(self.request)
        self.assertIs(failure, raised.exception.__cause__)
        self.assertIn("HTTP 429", str(raised.exception))
        self.assertIn("601.0s", str(raised.exception))
        self.assertIn("600.0s remain", str(raised.exception))
        send.assert_called_once()
        self.sleep.assert_not_called()

    def test_step2_does_not_prompt_repair_auth_or_exhausted_transport(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        context = SimpleNamespace(options={"tool_plugin": "unused:tool"}, run_id="test", attempt=1,
                                  inputs=[], work_dir=root)
        self.enterContext(patch("agent_pipeline_v2.step2._load_deepseek_env"))
        self.enterContext(patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}))
        self.enterContext(patch("agent_pipeline_v2.step2.instantiate", return_value=DeepSeekV4FlashObservationTool()))
        failures = [(HTTPError(self.request.full_url, code, "denied", {}, None), 1) for code in (401, 402, 403)]
        failures.append((URLError(TimeoutError("read timed out")), 3))
        for failure, attempts in failures:
            with self.subTest(failure=str(failure)):
                with patch("agent_pipeline_v2.step2.urlopen", side_effect=failure) as send:
                    with self.assertRaises(ValueError):
                        _tool_call(context, [], call_number=1, operation="rewrite_observations",
                                   selected_claims=[{"id": "claim_1", "content": "Measured improvement."}])
                self.assertEqual(attempts, send.call_count)
                audit = read_json(root / "semantic_step_2_tool_response_1.json")
                self.assertEqual(type(failure).__name__, audit["response"]["error"]["type"])
                self.assertEqual(str(failure), audit["response"]["error"]["message"])

    def test_step2_still_repairs_invalid_json_once(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        context = SimpleNamespace(options={"tool_plugin": "unused:tool"}, run_id="test", attempt=1,
                                  inputs=[], work_dir=root)
        content = {"claims": [{"id": "claim_1", "content": "Measured improvement.",
                                "paragraph_anchor_ids": ["anchor_1"]}]}
        raw = {"choices": [{"message": {"content": json.dumps(content)}}]}
        with patch("agent_pipeline_v2.step2._load_deepseek_env"), \
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("agent_pipeline_v2.step2.instantiate", return_value=DeepSeekV4FlashObservationTool()), \
                patch("agent_pipeline_v2.step2.urlopen", side_effect=[io.BytesIO(b'{'), io.BytesIO(json.dumps(raw).encode())]) as send:
            normalized, _ = _tool_call(context, [], call_number=1, operation="rewrite_observations",
                                      selected_claims=[{"id": "claim_1", "content": "Measured improvement.",
                                                        "source_excerpts": ["[anchor_1] Measured improvement."]}])
        repair_prompt = json.loads(send.call_args_list[1].args[0].data)["messages"][0]["content"]
        self.assertIn("Correct the previous validation failure", repair_prompt)
        self.assertEqual(2, send.call_count)
        self.assertEqual("claim_1", normalized["claims"][0]["id"])
        self.sleep.assert_not_called()

    def test_optional_dedup_judge_keeps_observations_when_retry_budget_expires(self) -> None:
        observations = {key: {"content": {"canonical": "Measured improvement."}}
                        for key in ("claim_1", "claim_2")}
        failure = HTTPError(self.request.full_url, 429, "rate limited", {"Retry-After": "601"}, None)
        with patch("agent_pipeline_v2.step2._load_deepseek_env"), \
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("agent_pipeline_v2.step2.urlopen", side_effect=failure) as send:
            self.assertEqual([], _judge_duplicate_observations(None, observations, [("claim_1", "claim_2")]))
        send.assert_called_once()
        self.sleep.assert_not_called()


class FakeObservationTool:
    name = "fake-observation"
    version = "1"
    calls: list[ToolCallRequest] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        self.__class__.calls.append(request)
        operation = request.operation
        selected = request.parameters.get("selected_claims")
        if operation == "rewrite_observations":
            claims = [{
                "id": item["id"],
                "content": f"Revised: {item['content']}",
                "paragraph_anchor_ids": list(item.get("source_anchor_ids", [])),
            } for item in selected]
            normalized = {"status": "claims_extracted", "claims": claims, "relations": []}
            return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)
        candidate = request.parameters["experiment_candidates"][0]
        anchor_ids = [item["anchor_id"] for item in candidate["paragraphs"]]
        result_anchor = next(value for value in anchor_ids if value.endswith("result"))
        claims = [
            {
                "candidate_id": candidate["candidate_id"],
                "_extraction_key": f"{candidate['candidate_id']}:0",
                "S": "setting one",
                "A": "method",
                "B": "baseline",
                "M": "accuracy",
                "R": "improved by 5 points",
                "content": "In setting one, the method improved accuracy by 5 points over baseline.",
                "paragraph_anchor_ids": [result_anchor],
            },
            {
                "candidate_id": candidate["candidate_id"],
                "_extraction_key": f"{candidate['candidate_id']}:1",
                "S": "setting two",
                "A": "method",
                "B": "baseline",
                "M": "accuracy",
                "R": "decreased by 2 points",
                "content": "In setting two, the method decreased accuracy by 2 points relative to baseline.",
                "paragraph_anchor_ids": [result_anchor],
            },
        ]
        normalized = {"status": "claims_extracted", "claims": claims, "relations": []}
        return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)

    @classmethod
    def _extract_relations(cls, candidate: dict, observations: list[dict], base_url: str) -> tuple[list[dict], list]:
        del candidate, base_url
        keys = [item["observation_key"] for item in observations]
        if len(keys) < 2:
            return [], []
        return [{
            "observation_keys": keys,
            "claim_id": "claim_2",
            "expression": f"([O:{keys[0]}] 和 [O:{keys[1]}]) 是 [claim_2] 的例子或证据",
        }], []


class ExpandingObservationTool(FakeObservationTool):
    name = "expanding-observation"
    paragraph_counts: list[int] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        if request.operation not in ("extract_observation_claims", "revise_observation_claims"):
            return super().invoke(request)
        candidate = request.parameters["experiment_candidates"][0]
        self.__class__.paragraph_counts.append(len(candidate["paragraphs"]))
        if len(self.__class__.paragraph_counts) == 1:
            normalized = {"status": "insufficient_context", "needed_context": "Need a wider window."}
            return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)
        return super().invoke(request)


class AlwaysInsufficientObservationTool(FakeObservationTool):
    name = "always-insufficient-observation"
    paragraph_counts: list[int] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        if request.operation not in ("extract_observation_claims", "revise_observation_claims"):
            return super().invoke(request)
        candidate = request.parameters["experiment_candidates"][0]
        self.__class__.paragraph_counts.append(len(candidate["paragraphs"]))
        normalized = {"status": "insufficient_context", "needed_context": "No result is available."}
        return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)


class FailingObservationTool:
    name = "failing-observation"
    version = "1"

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        raise RuntimeError("synthetic endpoint failure")


class Step2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "paper_text.md").write_text(
            "[#intro] Background paragraph.\n\n"
            "[#fig2] ![Figure 2](fig2.png)\n\n"
            "[#result] Under both settings, Figure 2 reports the measured accuracy effects.\n\n"
            "[#after] Follow-up discussion.\n",
            encoding="utf-8",
        )
        atomic_write_json(
            self.root / "claims_final.json",
            {
                "claim": [
                    {
                        "number": 1,
                        "conclusion": "data",
                        "text": "Figure 2 reports the measured accuracy effects.",
                        "is_pure_data": True,
                    },
                    {"number": 2, "conclusion": "claim", "text": "The method is effective."},
                ],
                "note": [],
                "relation": [],
            },
        )
        atomic_write_json(
            self.root / "manifest.json",
            {
                "artifacts": [
                    {"path": "paper_text.md", "kind": "source.paper_text", "media_type": "text/markdown"},
                    {"path": "claims_final.json", "kind": "source.claims_final", "media_type": "application/json"},
                ]
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_relation_dedup_uses_endpoints_not_model_wording(self) -> None:
        claims = [
            {"_extraction_key": "o1", "content": "Observation one"},
            {"_extraction_key": "o2", "content": "Observation two"},
        ]
        relations = [
            {"observation_keys": ["o1", "o2"], "claim_id": "claim_2", "expression": "first wording"},
            {"observation_keys": ["o2", "o1"], "claim_id": "claim_2", "expression": "second wording"},
        ]
        _, unique = _deduplicate_extractions(claims, relations)
        self.assertEqual(1, len(unique))
        self.assertEqual(["o1", "o2"], unique[0]["observation_keys"])

    def test_observation_binding_writes_refined_components_and_merges_anchors(self) -> None:
        existing = {
            "claim_1": {
                "type": "observation_claim",
                "content": {"canonical": "Original sentence.", "S": "old scope"},
                "source_anchor_ids": ["anchor_claim_1"],
            },
        }
        refined = [{
            "observation_key": "claim_1",
            "content": "Refined observation sentence.",
            "paragraph_anchor_ids": ["anchor_paragraph_result"],
            "S": "new scope",
            "A": "method",
            "M": "accuracy",
            "R": "improved",
        }]
        bound_nodes, bound_ids = _observation_binding(refined, {"claim_1": "claim_1"}, existing)
        self.assertEqual({"claim_1": "claim_1"}, bound_ids)
        node = bound_nodes["claim_1"]
        self.assertEqual("Refined observation sentence.", node["content"]["canonical"])
        self.assertEqual("new scope", node["content"]["S"])
        self.assertEqual("improved", node["content"]["R"])
        self.assertEqual(["anchor_claim_1", "anchor_paragraph_result"], node["source_anchor_ids"])
        # Binding never creates a parallel phenomenon node or equivalence data.
        self.assertEqual({"claim_1"}, set(bound_nodes))

    def test_prompts_exclude_prior_work_and_refine_imported_observations(self) -> None:
        candidate = {
            "candidate_id": "image_1",
            "image_name": "figure1.png",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
            "focus_anchor_ids": ["anchor_result"],
        }
        extraction_prompt = DeepSeekV4FlashObservationTool._prompt([candidate], None, None, None)
        self.assertIn("Never extract a result attributed to cited studies or prior work", extraction_prompt)
        rewrite_prompt = DeepSeekV4FlashObservationTool._rewrite_observation_prompt(
            [{"id": "claim_1", "content": "The method improved accuracy."}],
        )
        self.assertIn("Retain each supplied id verbatim", rewrite_prompt)
        self.assertIn("\"S\"", rewrite_prompt)

    def test_blank_optional_baseline_and_uncertainty_are_normalized_to_omission(self) -> None:
        candidate = {
            "candidate_id": "image_1",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
            "focus_anchor_ids": ["anchor_result"],
        }
        content = json.dumps({
            "status": "claims_extracted",
            "claims": [{
                "candidate_id": "image_1",
                "S": "the stated setting",
                "A": "the method",
                "B": "   ",
                "M": "success rate",
                "R": "reached the reported level",
                "U": "   ",
                "content": "In the stated setting, the method reached the reported success rate.",
                "paragraph_anchor_ids": ["anchor_result"],
            }],
        })
        normalized = DeepSeekV4FlashObservationTool._normalize(content, [candidate])
        self.assertNotIn("B", normalized["claims"][0])
        self.assertNotIn("U", normalized["claims"][0])

    def test_absence_placeholders_are_rejected_instead_of_persisted(self) -> None:
        candidate = {
            "candidate_id": "image_1",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
            "focus_anchor_ids": ["anchor_result"],
        }
        for field, value in (("B", "baseline not reported"), ("U", "uncertainty not reported")):
            claim = {
                "candidate_id": "image_1", "S": "setting", "A": "method", "M": "accuracy", "R": "improved",
                "content": "The method improved accuracy in the setting.", "paragraph_anchor_ids": ["anchor_result"],
                field: value,
            }
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "omit"):
                DeepSeekV4FlashObservationTool._normalize(
                    json.dumps({"status": "claims_extracted", "claims": [claim]}), [candidate]
                )

    def test_mechanical_anchor_excludes_headings_and_requires_two_tokens_and_threshold(self) -> None:
        paragraphs = [
            {"anchor_id": "heading", "text": "# Measured accuracy effects"},
            {"anchor_id": "single", "text": "Accuracy only."},
            {"anchor_id": "result", "text": "The experiment reports measured accuracy effects for the method."},
        ]
        self.assertEqual(
            "result",
            _best_paragraph_anchor("The method has measured accuracy effects.", paragraphs)["anchor_id"],
        )
        self.assertIsNone(_best_paragraph_anchor("Completely unrelated statement.", paragraphs))

    def test_short_table_results_match_rows_and_preserve_numeric_values(self) -> None:
        paragraphs = [
            {"anchor_id": "wrong", "text": "| SciBERT | 0.74 |"},
            {"anchor_id": "table", "text": "\n".join([
                "| Model | F1 |", "| SciBERT | 0.75 |",
                *[f"| OtherModel{i} | 0.90 |" for i in range(60)],
            ])},
            {"anchor_id": "appendix", "text": "| SciBERT | 0.75 |"},
        ]
        self.assertEqual("table", _best_paragraph_anchor("SciBERT 0.75,", paragraphs)["anchor_id"])
        self.assertIsNone(_best_paragraph_anchor("SciBERT 0.76,", paragraphs))
        self.assertIsNone(_best_paragraph_anchor("SciBERT", paragraphs))

    def test_short_error_percentage_matches_source_row(self) -> None:
        paragraphs = [
            {"anchor_id": "wrong", "text": "Implicit stance is discussed in the guidelines."},
            {"anchor_id": "errors", "text": "| Implicit stance | indirect comparison | ~20% |\n"
             "| Evidence mismatch | wrong claim | ~15% |"},
        ]
        self.assertEqual("errors", _best_paragraph_anchor("implicit stance about 20%,", paragraphs)["anchor_id"])

    def test_anchor_matches_both_range_endpoints_without_treating_model_version_as_result(self) -> None:
        paragraphs = [
            {"anchor_id": "wrong", "text": "The confidence interval is 0.73–0.85."},
            {"anchor_id": "right", "text": "The confidence interval is 0.73–0.81."},
        ]
        self.assertEqual("right", _best_paragraph_anchor(
            "The confidence interval is 0.73-0.81.", paragraphs,
        )["anchor_id"])
        self.assertIsNone(_best_paragraph_anchor("GPT-4.1", [{"anchor_id": "model", "text": "GPT-4.1"}]))

    def test_missing_source_anchor_fails_before_sending_rewrite(self) -> None:
        (self.root / "paper_text.md").write_text("Completely unrelated source.\n", encoding="utf-8")
        with patch("agent_pipeline_v2.step2._tool_call") as call:
            store = RunStore.create(self.root / "runs", self.pipeline(), input_manifest=self.root / "manifest.json")
            run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("failed", run.status)
        self.assertEqual("step2_extract_observations", run.current_stage)
        self.assertTrue(any(f["code"] == "STEP2_EXTRACTION_FAILED" for f in run.findings))
        call.assert_not_called()

    def pipeline(self, tool: str = f"{__name__}:FakeObservationTool") -> dict:
        value = json.loads(
            (Path(__file__).parents[1] / "pipeline.step2.json").read_text(encoding="utf-8")
        )
        value["pipeline_id"] += "-test"
        value["stages"][2]["options"]["tool_plugin"] = tool
        return value

    def test_extracts_anchored_observations_into_graph_and_projects_as_claims(self) -> None:
        FakeObservationTool.calls = []
        store = RunStore.create(self.root / "runs", self.pipeline(), input_manifest=self.root / "manifest.json")
        run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("succeeded", run.status)
        formalization_ref = max(
            (ref for ref in store.load_artifacts() if ref.kind == "formalization"),
            key=lambda ref: int(ref.metadata["step"]),
        )
        formalization = read_json(store.artifact_path(formalization_ref))
        self.assertEqual([], formalization["workflow"]["revisions"])
        self.assertIn("anchor_paragraph_result", formalization["knowledges"]["claim_1"]["source_anchor_ids"])
        self.assertEqual("observation_claim", formalization["knowledges"]["claim_1"]["type"])
        self.assertEqual(
            "Revised: Figure 2 reports the measured accuracy effects.",
            formalization["knowledges"]["claim_1"]["content"]["canonical"],
        )
        self.assertEqual("observation_claim", formalization["knowledges"]["claim_O01"]["type"])
        self.assertEqual(
            "In setting one, the method improved accuracy by 5 points over baseline.",
            formalization["knowledges"]["claim_O01"]["content"]["canonical"],
        )
        self.assertEqual("setting one", formalization["knowledges"]["claim_O01"]["content"]["S"])
        self.assertEqual("improved by 5 points", formalization["knowledges"]["claim_O01"]["content"]["R"])
        self.assertIn("claim_O01", formalization["graph"]["nodes"])
        # Stage 1 deleted the phenomenon layer entirely: every experiment is one
        # observation node and there is no E claim or equivalence operator.
        self.assertFalse(any(key.startswith("claim_E") for key in formalization["knowledges"]))
        self.assertFalse(any(key.startswith("claim_E") for key in formalization["graph"]["nodes"]))
        self.assertFalse(any(operator.get("type") == "equivalence" for operator in formalization["graph"]["operators"]))
        relation = formalization["workflow"]["non_reasoning_links"][-1]
        self.assertEqual(["claim_O01", "claim_O02"], relation["sources"])
        self.assertEqual("claim_2", relation["target"])
        self.assertEqual("([claim_O01] 和 [claim_O02]) 是 [claim_2] 的例子或证据", relation["metadata"]["relation"]["expression"])
        self.assertEqual("image_01_fig2", relation["metadata"]["relation"]["relation_context_id"])
        candidates = FakeObservationTool.calls[0].parameters["experiment_candidates"]
        self.assertEqual(
            ["anchor_paragraph_intro", "anchor_paragraph_fig2", "anchor_paragraph_result", "anchor_paragraph_after"],
            [item["anchor_id"] for item in candidates[0]["paragraphs"]],
        )
        view = project_run(store.run_dir)
        self.assertFalse(any(str(node["entity_id"]).startswith("claim_E") for node in view.nodes))
        self.assertFalse(any(str(edge.get("entity_id", "")).startswith("operator_equivalence") for edge in view.edges))
        observation_node = next(node for node in view.nodes if node["entity_id"] == "claim_O01")
        self.assertEqual("observation_claim", observation_node["kind"])
        projected = project_for_official_compiler(formalization)
        observation = next(item for item in projected["graph"]["knowledges"] if item["id"] == "claim_O01")
        self.assertEqual("claim", observation["type"])
        self.assertNotIn("role" + "s", observation)

    def test_imported_relations_survive_relation_merge(self) -> None:
        claims_final = read_json(self.root / "claims_final.json")
        claims_final["relation"] = [{"connects": [1, 2], "expression": "[1] 是 [2] 的例子或证据"}]
        atomic_write_json(self.root / "claims_final.json", claims_final)
        FakeObservationTool.calls = []
        store = RunStore.create(self.root / "merge-runs", self.pipeline(), input_manifest=self.root / "manifest.json")
        self.assertEqual("succeeded", run_pipeline(store.run_dir, max_stages=3).status)
        formalization_ref = max(
            (ref for ref in store.load_artifacts() if ref.kind == "formalization"),
            key=lambda ref: int(ref.metadata["step"]),
        )
        formalization = read_json(store.artifact_path(formalization_ref))
        links = formalization["workflow"]["non_reasoning_links"]
        imported = next(link for link in links if link["id"] == "relation_1")
        self.assertEqual(["claim_1"], imported["sources"])
        self.assertEqual("claim_2", imported["target"])
        self.assertEqual("claims_final", imported["metadata"]["source"])
        # Step 2 appends its own discovered relation without replacing imports.
        self.assertTrue(any(link["id"].startswith("relation_step2_") for link in links))
        self.assertEqual(2, len(links))

    def test_insufficient_context_allows_two_mechanical_expansions(self) -> None:
        (self.root / "paper_text.md").write_text(
            "[#earlier] Earlier experimental context.\n\n"
            "[#intro] Background paragraph.\n\n"
            "[#fig2] ![Figure 2](fig2.png)\n\n"
            "[#result] Under both settings, Figure 2 reports the measured accuracy effects.\n\n"
            "[#after] Follow-up discussion.\n\n"
            "[#later] Later experimental context.\n",
            encoding="utf-8",
        )
        ExpandingObservationTool.calls = []
        ExpandingObservationTool.paragraph_counts = []
        tool = f"{__name__}:ExpandingObservationTool"
        store = RunStore.create(self.root / "runs", self.pipeline(tool), input_manifest=self.root / "manifest.json")
        self.assertEqual("succeeded", run_pipeline(store.run_dir, max_stages=3).status)
        self.assertEqual(2, len(ExpandingObservationTool.paragraph_counts))
        self.assertGreater(ExpandingObservationTool.paragraph_counts[1], ExpandingObservationTool.paragraph_counts[0])

    def test_insufficient_context_is_retained_without_terminating_step2(self) -> None:
        paragraphs = [f"[#p{index}] Context paragraph {index}." for index in range(1, 8)]
        paragraphs.insert(3, "[#fig2] ![Figure 2](fig2.png)")
        paragraphs.insert(4, "[#result] Figure 2 reports an experiment.")
        (self.root / "paper_text.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        AlwaysInsufficientObservationTool.paragraph_counts = []
        tool = f"{__name__}:AlwaysInsufficientObservationTool"
        store = RunStore.create(self.root / "runs", self.pipeline(tool), input_manifest=self.root / "manifest.json")
        run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("succeeded", run.status)
        self.assertEqual(3, len(AlwaysInsufficientObservationTool.paragraph_counts))
        self.assertLess(AlwaysInsufficientObservationTool.paragraph_counts[0], AlwaysInsufficientObservationTool.paragraph_counts[1])
        self.assertLess(AlwaysInsufficientObservationTool.paragraph_counts[1], AlwaysInsufficientObservationTool.paragraph_counts[2])

    def test_model_and_comparison_regime_prompt(self) -> None:
        self.assertEqual("deepseek-v4-flash", MODEL_NAME)
        self.assertIn("Return JSON only", DeepSeekV4FlashObservationTool._prompt([], None, None, None))
        self.assertIn("self-contained experimental finding", DeepSeekV4FlashObservationTool._prompt([], None, None, None))
        self.assertIn("do not omit a broad result", DeepSeekV4FlashObservationTool._prompt([], None, None, None))
        candidates = [{
            "candidate_id": "image_01_fig2",
            "image_name": "fig2.png",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
            "related_claims": [{"id": "claim_7", "source_anchor_ids": ["anchor_result"]}],
        }]
        claims = {
            "status": "claims_extracted",
            "claims": [
                {"candidate_id": "image_01_fig2", "S": "s", "A": "a", "B": "b", "M": "m", "R": "r", "U": "u1", "content": "claim one", "paragraph_anchor_ids": ["anchor_result"]},
                {"candidate_id": "image_01_fig2", "S": "s", "A": "a", "M": "m", "R": "r", "U": "u2", "content": "claim two", "paragraph_anchor_ids": ["anchor_result"]},
            ],
            "relations": [],
        }
        normalized = DeepSeekV4FlashObservationTool._normalize(json.dumps(claims), candidates)
        self.assertEqual(2, len(normalized["claims"]))
        self.assertTrue(any("B" not in claim for claim in normalized["claims"]))
        self.assertEqual([], normalized["relations"])

    def test_prompt_split_rule_is_finding_based_and_table_guidance_is_scoped(self) -> None:
        figure_candidate = {
            "candidate_id": "image_01_fig2", "kind": "figure", "image_name": "fig2.png",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
        }
        general = DeepSeekV4FlashObservationTool._prompt([figure_candidate], None, None, None)
        self.assertIn("self-contained experimental finding", general)
        self.assertNotIn("distinct evidence-supported experimental unit", general)
        # The cell/row/column enumeration ban belongs only in the table-scoped
        # guidance, never in the common prompt.
        self.assertNotIn("per cell, per row, per column", general)
        table_candidate = {
            "candidate_id": "table_01_6", "kind": "table", "table_label": "Table 6",
            "paragraphs": [{"anchor_id": "anchor_table", "text": "<table></table>"}],
        }
        table_prompt = DeepSeekV4FlashObservationTool._prompt([table_candidate], None, None, None)
        self.assertIn("This candidate is a TABLE", table_prompt)
        self.assertIn("per cell, per row, per column", table_prompt)
        self.assertIn("/ table Table 6", table_prompt)

    def test_experiment_candidates_include_tables_and_exclude_foreign_objects(self) -> None:
        from pipeline_harness.domain.stages import _paper_paragraphs
        paper = (
            "[#cap] Table 6: Summary of resource usage.\n\n"
            "[#table] <table><tr><td>BBBP</td><td>Fine-Grained</td><td>59.1</td></tr></table>\n\n"
            "[#fig3] ![Figure 3](fig3.png)\n\n"
            "[#fig4] ![Figure 4](fig4.png)\n"
        )
        paragraphs = _paper_paragraphs(paper.splitlines())
        candidates = _experiment_candidates(paragraphs, radius=1)
        by_kind: dict[str, list[dict]] = {}
        for candidate in candidates:
            by_kind.setdefault(candidate["kind"], []).append(candidate)
        self.assertEqual(2, len(by_kind["figure"]))
        self.assertEqual(1, len(by_kind["table"]))
        table = by_kind["table"][0]
        self.assertEqual("Table 6", table["table_label"])
        table_anchors = [item["anchor_id"] for item in table["paragraphs"]]
        self.assertIn("anchor_paragraph_cap", table_anchors)
        self.assertIn("anchor_paragraph_table", table_anchors)
        self.assertNotIn("anchor_paragraph_fig3", table_anchors)
        self.assertNotIn("anchor_paragraph_fig4", table_anchors)
        fig3 = next(item for item in by_kind["figure"] if item["image_name"] == "fig3.png")
        fig3_anchors = [item["anchor_id"] for item in fig3["paragraphs"]]
        self.assertNotIn("anchor_paragraph_table", fig3_anchors)
        self.assertNotIn("anchor_paragraph_fig4", fig3_anchors)

    def test_relation_classifier_groups_multiple_observations_for_one_target(self) -> None:
        groups = DeepSeekV4FlashObservationTool._relation_groups(
            [
                {"observation_key": "o1", "content": "e1", "paragraph_anchor_ids": ["anchor_1"]},
                {"observation_key": "o2", "content": "e2", "paragraph_anchor_ids": ["anchor_1"]},
            ],
            {"related_claims": [
                {"id": "claim_1", "type": "claim", "content": "c1", "source_anchor_ids": ["anchor_1"]},
                {"id": "note_1", "type": "note", "content": "n1", "source_anchor_ids": ["anchor_1"]},
            ]},
        )
        self.assertEqual(1, len(groups))
        self.assertEqual(["o1", "o2"], [item["observation_key"] for item in groups[0]["evidence_candidates"]])
        payload = {"classifications": [
            {"group_id": "group_1", "relations": [{
                "source_observation_keys": ["o1", "o2"], "relation_type": "evidence",
                "expression": "([O:o1] 和 [O:o2]) 是 [claim_1] 的例子或证据",
            }]},
        ]}
        relations = DeepSeekV4FlashObservationTool._normalize_group_relations(json.dumps(payload), groups)
        self.assertEqual([{"observation_keys": ["o1", "o2"], "claim_id": "claim_1", "expression": "([O:o1] 和 [O:o2]) 是 [claim_1] 的例子或证据"}], relations)

    def test_relation_normalization_failure_retains_the_received_raw_response(self) -> None:
        candidate = {
            "candidate_id": "image_one",
            "image_name": "one.png",
            "focus_anchor_ids": ["anchor_one"],
            "paragraphs": [{"anchor_id": "anchor_one", "text": "Measured result."}],
            "related_claims": [{
                "id": "claim_1",
                "type": "claim",
                "content": "The method is generally effective.",
                "source_anchor_ids": ["anchor_one"],
            }],
        }
        invalid_relation = {"choices": [{"message": {"content": json.dumps({
            "classifications": [{"group_id": "unknown_group", "relations": []}],
        })}}]}
        with patch("agent_pipeline_v2.step2._load_deepseek_env"), patch.dict(
            "os.environ", {"DEEPSEEK_API_KEY": "test-key"}
        ), patch(
            "agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(json.dumps(invalid_relation).encode("utf-8"))
        ):
            with self.assertRaisesRegex(ValueError, "group_id") as raised:
                DeepSeekV4FlashObservationTool._extract_relations(
                    candidate,
                    [{"observation_key": "image_one:0", "content": "Observed one."}],
                    "https://api.deepseek.com/v1",
                )
        self.assertEqual([invalid_relation], raised.exception.raw_responses)

    def test_cluster_normalizer_requires_complete_relation_coverage(self) -> None:
        cluster = {
            "cluster_id": "cluster_001", "relation_context_id": "experiment_1",
            "claims": [
                {"claim_id": "claim_1", "content": "Composite statement."},
                {"claim_id": "claim_O01", "content": "Atomic observation."},
            ],
            "candidate_relations": [{
                "relation_id": "relation_1", "sources": ["claim_O01"], "target": "claim_1",
                "expression": "[claim_O01] 是 [claim_1] 的证据",
            }],
            "source_excerpts": [{"anchor_id": "anchor_1", "text": "Source."}],
        }
        payload = {"clusters": [{
            "cluster_id": "cluster_001", "weakpoints": [{
                "member_relation_ids": ["relation_1"], "evidence_claim_ids": ["claim_1"],
                "target_claim_id": ["claim_O01"], "reasoning_type": "deduction",
                "expression": "[claim_1] 推出 [claim_O01]",
            }], "rejected_relation_ids": [],
        }]}
        raw = {"choices": [{"message": {"content": json.dumps(payload)}}]}
        request = ToolCallRequest(
            "cluster_call", "deepseek-v4-flash-observation", "7", "normalize_weakpoint_clusters", [],
            {"clusters": [cluster]},
        )
        with patch("agent_pipeline_v2.step2._load_deepseek_env"), \
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), \
                patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())):
            response = DeepSeekV4FlashObservationTool().invoke(request)
        self.assertEqual("succeeded", response.status)
        self.assertEqual(["claim_O01"], response.normalized["clusters"][0]["weakpoints"][0]["target_claim_id"])

    def test_rewritten_observations_normalize_refined_components(self) -> None:
        selected = [{"id": "claim_1", "content": "Observed one.",
                     "source_excerpts": ["[anchor_1] Observed one."]}]
        payload = {"claims": [{
            "id": "claim_1",
            "S": "the stated setting",
            "A": "the method",
            "M": "accuracy",
            "R": "improved",
            "content": "In the stated setting, the method improved accuracy.",
            "paragraph_anchor_ids": ["anchor_1"],
        }]}
        normalized = DeepSeekV4FlashObservationTool._normalize_rewritten_observations(
            json.dumps(payload), selected,
        )
        self.assertEqual("the stated setting", normalized[0]["S"])
        self.assertEqual("improved", normalized[0]["R"])
        self.assertEqual(["anchor_1"], normalized[0]["paragraph_anchor_ids"])
        for anchors in ([], ["anchor_claim_1"], ["anchor_other"]):
            with self.subTest(anchors=anchors), self.assertRaisesRegex(ValueError, "anchors"):
                invalid = copy.deepcopy(payload)
                invalid["claims"][0]["paragraph_anchor_ids"] = anchors
                DeepSeekV4FlashObservationTool._normalize_rewritten_observations(json.dumps(invalid), selected)
        payload["claims"].append(dict(payload["claims"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one rewritten claim per selected id"):
            DeepSeekV4FlashObservationTool._normalize_rewritten_observations(
                json.dumps(payload), selected,
            )
        payload["claims"] = [{"id": "claim_other", "content": "x", "paragraph_anchor_ids": ["anchor_1"]}]
        with self.assertRaisesRegex(ValueError, "unknown or duplicate id"):
            DeepSeekV4FlashObservationTool._normalize_rewritten_observations(
                json.dumps(payload), selected,
            )

    def test_expanded_context_claim_must_cite_a_focus_anchor(self) -> None:
        candidates = [{
            "candidate_id": "image_01_fig2",
            "image_name": "fig2.png",
            "focus_anchor_ids": ["anchor_fig2"],
            "paragraphs": [
                {"anchor_id": "anchor_fig2", "text": "Figure 2."},
                {"anchor_id": "anchor_elsewhere", "text": "An unrelated result."},
            ],
        }]
        claims = {
            "status": "claims_extracted",
            "claims": [{
                "candidate_id": "image_01_fig2", "S": "s", "A": "a", "M": "m", "R": "r", "U": "u",
                "content": "Under s, a on m showed r; uncertainty: u.",
                "paragraph_anchor_ids": ["anchor_elsewhere"],
            }],
        }
        with self.assertRaisesRegex(ValueError, "no focus paragraph anchor"):
            DeepSeekV4FlashObservationTool._normalize(json.dumps(claims), candidates)

    def test_deepseek_sends_each_candidate_in_a_separate_request(self) -> None:
        class FakeHTTPResponse:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeHTTPResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        candidates = [
            {
                "candidate_id": "image_one",
                "image_name": "one.png",
                "paragraphs": [{"anchor_id": "anchor_one", "text": "One."}],
            },
            {
                "candidate_id": "image_two",
                "image_name": "two.png",
                "paragraphs": [{"anchor_id": "anchor_two", "text": "Two."}],
            },
        ]
        prompts: list[str] = []

        def fake_urlopen(http_request: object, timeout: int) -> FakeHTTPResponse:
            del timeout
            prompt = json.loads(http_request.data.decode("utf-8"))["messages"][0]["content"]
            prompts.append(prompt)
            candidate_id = "image_one" if "Candidate image_one" in prompt else "image_two"
            anchor_id = "anchor_one" if candidate_id == "image_one" else "anchor_two"
            result = {
                "status": "claims_extracted",
                "claims": [{
                    "candidate_id": candidate_id,
                    "S": "setting",
                    "A": "method",
                    "M": "accuracy",
                    "R": "improved",
                    "content": "In the setting, the method improved accuracy.",
                    "paragraph_anchor_ids": [anchor_id],
                }],
            }
            return FakeHTTPResponse({"choices": [{"message": {"content": json.dumps(result)}}]})

        tool = DeepSeekV4FlashObservationTool()
        request = ToolCallRequest(
            "call",
            tool.name,
            tool.version,
            operation="extract_observation_claims",
            inputs=[],
            parameters={"experiment_candidates": candidates},
        )
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}, clear=False), patch(
            "agent_pipeline_v2.step2.urlopen", side_effect=fake_urlopen
        ):
            response = tool.invoke(request)
        self.assertEqual(2, len(prompts))
        self.assertIn("Candidate image_one", prompts[0])
        self.assertNotIn("Candidate image_two", prompts[0])
        self.assertIn('Allowed paragraph_anchor_ids: ["anchor_one"]', prompts[0])
        self.assertIn("Candidate image_two", prompts[1])
        self.assertNotIn("Candidate image_one", prompts[1])
        self.assertEqual(2, len(response.normalized["claims"]))
        self.assertNotIn("equivalent_claims", response.normalized)

    def test_tool_failure_is_preserved_as_existing_audit_artifact(self) -> None:
        tool = f"{__name__}:FailingObservationTool"
        store = RunStore.create(self.root / "runs", self.pipeline(tool), input_manifest=self.root / "manifest.json")
        run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("failed", run.status)
        self.assertEqual("step2_extract_observations", run.current_stage)
        self.assertEqual(2, run.next_stage_index)
        audits = [ref for ref in store.load_artifacts() if ref.kind == "tool.semantic_review.response"]
        self.assertGreaterEqual(len(audits), 1)
        audit = read_json(store.artifact_path(audits[0]))
        self.assertEqual("failed", audit["response"]["status"])
        self.assertEqual("RuntimeError", audit["response"]["error"]["type"])


if __name__ == "__main__":
    unittest.main()
