from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_pipeline_v2.compiler_projection import project_for_official_compiler
from agent_pipeline_v2.step2 import DeepSeekV4FlashObservationTool, MODEL_NAME, _best_paragraph_anchor
from pipeline_harness.domain.tools import ToolCallRequest, ToolCallResponse
from pipeline_harness.runner import run_pipeline
from pipeline_harness.store import RunStore, atomic_write_json, read_json
from pipeline_harness.view.projector import project_run


class FakeObservationTool:
    name = "fake-observation"
    version = "1"
    calls: list[ToolCallRequest] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        self.__class__.calls.append(request)
        candidate = request.parameters["experiment_candidates"][0]
        anchor_ids = [item["anchor_id"] for item in candidate["paragraphs"]]
        result_anchor = next(value for value in anchor_ids if value.endswith("result"))
        selected = request.parameters.get("selected_claims")
        if selected:
            claims = [{
                "id": selected[0]["id"],
                "candidate_id": candidate["candidate_id"],
                "S": "revised setting",
                "A": "revised method",
                "B": "baseline",
                "M": "accuracy",
                "R": "improved by 7 points",
                "content": "In the revised setting, the revised method improved accuracy by 7 points over baseline.",
                "paragraph_anchor_ids": [result_anchor],
            }]
        else:
            claims = [
                {
                    "candidate_id": candidate["candidate_id"],
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
                    "S": "setting two",
                    "A": "method",
                    "B": "baseline",
                    "M": "accuracy",
                    "R": "decreased by 2 points",
                    "content": "In setting two, the method decreased accuracy by 2 points relative to baseline.",
                    "paragraph_anchor_ids": [result_anchor],
                },
            ]
        normalized = {
            "status": "claims_extracted",
            "claims": claims,
            "equivalent_claims": [
                {
                    "observation_key": f"generated:{index}",
                    "content": f"Under {claim['S']}, {claim['A']} has the reported result on {claim['M']}.",
                    "paragraph_anchor_ids": list(claim["paragraph_anchor_ids"]),
                }
                for index, claim in enumerate(claims, 1)
            ],
            "relations": [] if selected else [{
                "phenomenon_keys": ["generated:1", "generated:2"],
                "claim_id": "claim_2",
                "expression": "([E:generated:1] 和 [E:generated:2]) 是 [claim_2] 的例子或证据",
            }],
        }
        return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)


class ExpandingObservationTool(FakeObservationTool):
    name = "expanding-observation"
    paragraph_counts: list[int] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        candidate = request.parameters["experiment_candidates"][0]
        self.__class__.paragraph_counts.append(len(candidate["paragraphs"]))
        if len(self.__class__.paragraph_counts) == 1:
            normalized = {"status": "insufficient_context", "needed_context": "Need a wider window."}
            return ToolCallResponse(request.call_id, "succeeded", normalized, normalized)
        return super().invoke(request)


class AlwaysInsufficientObservationTool:
    name = "always-insufficient-observation"
    version = "1"
    paragraph_counts: list[int] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
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

    def test_prompts_exclude_prior_work_and_require_nonverbatim_equivalent(self) -> None:
        candidate = {
            "candidate_id": "image_1",
            "image_name": "figure1.png",
            "paragraphs": [{"anchor_id": "anchor_result", "text": "Result."}],
            "focus_anchor_ids": ["anchor_result"],
        }
        extraction_prompt = DeepSeekV4FlashObservationTool._prompt([candidate], None, None, None)
        self.assertIn("Never extract a result attributed to cited studies or prior work", extraction_prompt)
        equivalent_prompt = DeepSeekV4FlashObservationTool._equivalent_claim_prompt(
            [{"_extraction_key": "image_1:0", "content": "The method improved accuracy."}],
            candidate,
        )
        self.assertIn("never copy O's content verbatim", equivalent_prompt)

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
        self.assertEqual("observation_claim", formalization["knowledges"]["claim_O01"]["type"])
        self.assertEqual(
            "In setting one, the method improved accuracy by 5 points over baseline.",
            formalization["knowledges"]["claim_O01"]["content"]["canonical"],
        )
        self.assertIn("claim_O01", formalization["graph"]["nodes"])
        self.assertEqual("claim", formalization["knowledges"]["claim_E01"]["type"])
        self.assertEqual(
            formalization["knowledges"]["claim_O01"]["source_anchor_ids"],
            formalization["knowledges"]["claim_E01"]["source_anchor_ids"],
        )
        self.assertIn("claim_E01", formalization["graph"]["nodes"])
        self.assertIn(
            {
                "id": "operator_equivalence_E01_O01",
                "type": "equivalence",
                "variables": ["claim_E01", "claim_O01"],
            },
            formalization["graph"]["operators"],
        )
        relation = formalization["workflow"]["non_reasoning_links"][-1]
        self.assertEqual(["claim_E01", "claim_E02"], relation["sources"])
        self.assertEqual("claim_2", relation["target"])
        self.assertEqual("([claim_E01] 和 [claim_E02]) 是 [claim_2] 的例子或证据", relation["metadata"]["relation"]["expression"])
        self.assertEqual("image_01_fig2", relation["metadata"]["relation"]["relation_context_id"])
        candidates = FakeObservationTool.calls[0].parameters["experiment_candidates"]
        self.assertEqual(
            ["anchor_paragraph_intro", "anchor_paragraph_fig2", "anchor_paragraph_result", "anchor_paragraph_after"],
            [item["anchor_id"] for item in candidates[0]["paragraphs"]],
        )
        view = project_run(store.run_dir)
        self.assertFalse(any(node["entity_id"] == "operator_equivalence_E01_O01" for node in view.nodes))
        equivalence = next(
            edge for edge in view.edges
            if edge.get("entity_id") == "operator_equivalence_E01_O01"
        )
        self.assertEqual("equivalence", equivalence["semantic_type"])
        self.assertEqual("step:2:knowledge:claim_E01", equivalence["source"])
        self.assertEqual("step:2:knowledge:claim_O01", equivalence["target"])
        self.assertEqual(
            {
                "id": "operator_equivalence_E01_O01",
                "type": "equivalence",
                "variables": ["claim_E01", "claim_O01"],
            },
            equivalence["details"],
        )
        with self.assertRaisesRegex(ValueError, "omit Gaia's required conclusion helper"):
            project_for_official_compiler(formalization)
        compiler_ready = copy.deepcopy(formalization)
        compiler_ready["graph"]["operators"] = []
        compiler_ready["revision"]["content_hash"] = ""
        from agent_pipeline_v2.authoring import content_hash
        compiler_ready["revision"]["content_hash"] = content_hash(compiler_ready)
        projected = project_for_official_compiler(compiler_ready)
        observation = next(item for item in projected["graph"]["knowledges"] if item["id"] == "claim_O01")
        self.assertEqual("claim", observation["type"])
        self.assertNotIn("role" + "s", observation)

    def test_insufficient_context_expands_mechanical_window(self) -> None:
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

    def test_three_expansions_mark_and_terminate_step2_when_context_remains_insufficient(self) -> None:
        paragraphs = [f"[#p{index}] Context paragraph {index}." for index in range(1, 8)]
        paragraphs.insert(3, "[#fig2] ![Figure 2](fig2.png)")
        paragraphs.insert(4, "[#result] Figure 2 reports an experiment.")
        (self.root / "paper_text.md").write_text("\n\n".join(paragraphs) + "\n", encoding="utf-8")
        AlwaysInsufficientObservationTool.paragraph_counts = []
        tool = f"{__name__}:AlwaysInsufficientObservationTool"
        store = RunStore.create(self.root / "runs", self.pipeline(tool), input_manifest=self.root / "manifest.json")
        run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("failed", run.status)
        self.assertTrue(
            any(
                "nine mechanical expansions (+1 through +9; 4 distinct windows)" in finding["message"]
                for finding in run.findings
            )
        )
        self.assertEqual(4, len(AlwaysInsufficientObservationTool.paragraph_counts))
        self.assertLess(AlwaysInsufficientObservationTool.paragraph_counts[0], AlwaysInsufficientObservationTool.paragraph_counts[1])
        self.assertLess(AlwaysInsufficientObservationTool.paragraph_counts[1], AlwaysInsufficientObservationTool.paragraph_counts[2])
        self.assertLess(AlwaysInsufficientObservationTool.paragraph_counts[2], AlwaysInsufficientObservationTool.paragraph_counts[3])

    def test_model_and_comparison_regime_prompt(self) -> None:
        self.assertEqual("deepseek-v4-flash", MODEL_NAME)
        self.assertIn("Return JSON only", DeepSeekV4FlashObservationTool._prompt([], None, None, None))
        self.assertIn("distinct evidence-supported experimental unit", DeepSeekV4FlashObservationTool._prompt([], None, None, None))
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

    def test_relation_classifier_groups_multiple_phenomena_for_one_target(self) -> None:
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
        self.assertEqual(["o1", "o2"], [item["phenomenon_key"] for item in groups[0]["evidence_candidates"]])
        payload = {"classifications": [
            {"group_id": "group_1", "relations": [{
                "source_phenomenon_keys": ["o1", "o2"], "relation_type": "evidence",
                "expression": "([E:o1] 和 [E:o2]) 是 [claim_1] 的例子或证据",
            }]},
        ]}
        relations = DeepSeekV4FlashObservationTool._normalize_group_relations(json.dumps(payload), groups)
        self.assertEqual([{"phenomenon_keys": ["o1", "o2"], "claim_id": "claim_1", "expression": "([E:o1] 和 [E:o2]) 是 [claim_1] 的例子或证据"}], relations)

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
        extraction = {"choices": [{"message": {"content": json.dumps({
            "status": "claims_extracted",
            "claims": [{
                "candidate_id": "image_one",
                "S": "the stated setting",
                "A": "the method",
                "M": "accuracy",
                "R": "improved",
                "content": "In the stated setting, the method improved accuracy.",
                "paragraph_anchor_ids": ["anchor_one"],
            }],
        })}}]}
        equivalent = {"choices": [{"message": {"content": json.dumps({
            "equivalent_claims": [{
                "observation_key": "image_one:0",
                "content": "Under the stated setting, the method improved accuracy; uncertainty is not reported.",
            }],
        })}}]}
        invalid_relation = {"choices": [{"message": {"content": json.dumps({
            "classifications": [{"group_id": "unknown_group", "relations": []}],
        })}}]}
        request = ToolCallRequest(
            "raw-retention",
            DeepSeekV4FlashObservationTool.name,
            DeepSeekV4FlashObservationTool.version,
            "extract_observation_claims",
            [],
            {"experiment_candidates": [candidate]},
        )
        responses = [
            io.BytesIO(json.dumps(item).encode("utf-8"))
            for item in (extraction, equivalent, invalid_relation)
        ]
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key"}), patch(
            "agent_pipeline_v2.step2.urlopen", side_effect=responses
        ):
            response = DeepSeekV4FlashObservationTool().invoke(request)
        self.assertEqual("failed", response.status)
        self.assertEqual(invalid_relation, response.raw["responses"][-1])
        self.assertEqual(3, response.metadata["request_count"])
        self.assertIn("group_id", response.error["message"])

    def test_cluster_normalizer_requires_complete_relation_coverage(self) -> None:
        cluster = {
            "cluster_id": "cluster_001", "relation_context_id": "experiment_1",
            "claims": [
                {"claim_id": "claim_1", "content": "Composite phenomenon."},
                {"claim_id": "claim_E01", "content": "Atomic phenomenon."},
            ],
            "candidate_relations": [{
                "relation_id": "relation_1", "sources": ["claim_E01"], "target": "claim_1",
                "expression": "[claim_E01] 是 [claim_1] 的证据",
            }],
            "source_excerpts": [{"anchor_id": "anchor_1", "text": "Source."}],
        }
        payload = {"clusters": [{
            "cluster_id": "cluster_001", "weakpoints": [{
                "member_relation_ids": ["relation_1"], "evidence_claim_ids": ["claim_1"],
                "target_claim_id": ["claim_E01"], "reasoning_type": "deduction",
                "expression": "[claim_1] 推出 [claim_E01]",
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
        self.assertEqual(["claim_E01"], response.normalized["clusters"][0]["weakpoints"][0]["target_claim_id"])

    def test_equivalent_claims_are_one_to_one_and_copy_anchors(self) -> None:
        observations = [
            {"_extraction_key": "o1", "content": "Observed one.", "paragraph_anchor_ids": ["anchor_1"]},
            {"_extraction_key": "o2", "content": "Observed two.", "paragraph_anchor_ids": ["anchor_2"]},
        ]
        payload = {"equivalent_claims": [
            {"observation_key": "o1", "content": "Phenomenon one."},
            {"observation_key": "o2", "content": "Phenomenon two."},
        ]}
        normalized = DeepSeekV4FlashObservationTool._normalize_equivalent_claims(json.dumps(payload), observations)
        self.assertEqual(["anchor_1"], normalized[0]["paragraph_anchor_ids"])
        self.assertEqual(["anchor_2"], normalized[1]["paragraph_anchor_ids"])
        payload["equivalent_claims"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one equivalent claim per observation"):
            DeepSeekV4FlashObservationTool._normalize_equivalent_claims(json.dumps(payload), observations)
        duplicate = {"equivalent_claims": [
            {"observation_key": "o1", "content": "Observed one."},
            {"observation_key": "o2", "content": "Phenomenon two."},
        ]}
        with self.assertRaisesRegex(ValueError, "must not copy"):
            DeepSeekV4FlashObservationTool._normalize_equivalent_claims(json.dumps(duplicate), observations)

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
            if "For every supplied experimental observation O" in prompt:
                candidate_id = "image_one" if '"candidate_id": "image_one"' in prompt else "image_two"
                result = {"equivalent_claims": [{
                    "observation_key": f"{candidate_id}:0",
                    "content": "Under setting, the method improves accuracy; uncertainty is not reported.",
                }]}
                return FakeHTTPResponse({"choices": [{"message": {"content": json.dumps(result)}}]})
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
        self.assertEqual(4, len(prompts))
        self.assertIn("Candidate image_one", prompts[0])
        self.assertNotIn("Candidate image_two", prompts[0])
        self.assertIn('Allowed paragraph_anchor_ids: ["anchor_one"]', prompts[0])
        self.assertIn("For every supplied experimental observation O", prompts[1])
        self.assertIn("Candidate image_two", prompts[2])
        self.assertNotIn("Candidate image_one", prompts[2])
        self.assertIn("For every supplied experimental observation O", prompts[3])
        self.assertEqual(2, len(response.normalized["claims"]))
        self.assertEqual(2, len(response.normalized["equivalent_claims"]))

    def test_tool_failure_is_preserved_as_existing_audit_artifact(self) -> None:
        tool = f"{__name__}:FailingObservationTool"
        store = RunStore.create(self.root / "runs", self.pipeline(tool), input_manifest=self.root / "manifest.json")
        run = run_pipeline(store.run_dir, max_stages=3)
        self.assertEqual("failed", run.status)
        audits = [ref for ref in store.load_artifacts() if ref.kind == "tool.semantic_review.response"]
        self.assertEqual(1, len(audits))
        audit = read_json(store.artifact_path(audits[0]))
        self.assertEqual("failed", audit["response"]["status"])
        self.assertEqual("RuntimeError", audit["response"]["error"]["type"])


if __name__ == "__main__":
    unittest.main()
