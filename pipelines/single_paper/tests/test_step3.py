from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from agent_pipeline_v2.authoring import emit_formalization
from agent_pipeline_v2.step2 import DeepSeekV4FlashObservationTool
from agent_pipeline_v2.step3 import (
    _relation_clusters, _screen_classifications,
    _transitive_shortcut_indexes, _validate_cluster_result,
)
from pipeline_harness.domain.tools import ToolCallRequest, ToolCallResponse
from pipeline_harness.runner import run_pipeline
from pipeline_harness.store import RunStore, atomic_write_json
from pipeline_harness.view.projector import project_run


class StripClaimsFinalSourcePlugin:
    """Downgrade claims_final relations so Step 3 exercises cluster+LLM.

    Production adopts claims_final relations directly; tests that target the
    LLM normalization path deliberately drop the high-confidence marker.
    """

    def run(self, context):
        reference = context.latest("formalization")
        document = json.loads(context.artifact_path(reference).read_text(encoding="utf-8"))
        for link in document["workflow"]["non_reasoning_links"]:
            metadata = link.get("metadata")
            if isinstance(metadata, dict):
                metadata.pop("source", None)
        prior = document["revision"]
        document["revision"] = {
            "revision_id": f"revision_{context.run_id}_strip",
            "supersedes": prior.get("revision_id"),
            "parent_hash": prior.get("content_hash"),
            "content_hash": "",
        }
        return emit_formalization(context, document, step=2, step_name="strip_claims_final")


class FakeWeakpointClassifier:
    name = "fake-weakpoint-classifier"
    version = "1"
    calls: list[ToolCallRequest] = []

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        self.__class__.calls.append(request)
        if request.operation == "classify_weakpoints":
            return ToolCallResponse(
                request.call_id, "succeeded", {},
                {"classifications": [
                    {"weakpoint_id": item["weakpoint_id"], "reasoning_type": "abduction"}
                    for item in request.parameters["weakpoints"]
                ]},
            )
        clusters = request.parameters["clusters"]
        normalized = []
        for cluster in clusters:
            weakpoints = []
            for relation in cluster["candidate_relations"]:
                endpoints = [*relation["sources"], relation["target"]]
                weakpoints.append({
                    "member_relation_ids": [relation["relation_id"]],
                    "evidence_claim_ids": list(relation["sources"]),
                    "target_claim_id": [relation["target"]],
                    "reasoning_type": "abduction",
                    "expression": " 是 ".join(f"[{key}]" for key in endpoints),
                })
            normalized.append({"cluster_id": cluster["cluster_id"], "weakpoints": weakpoints, "rejected_relation_ids": []})
        return ToolCallResponse(request.call_id, "succeeded", {"clusters": clusters}, {"clusters": normalized})


class ParallelWeakpointClassifier(FakeWeakpointClassifier):
    barrier = threading.Barrier(2)
    thread_ids: set[int] = set()

    def invoke(self, request: ToolCallRequest) -> ToolCallResponse:
        if request.operation == "normalize_weakpoint_clusters":
            self.__class__.thread_ids.add(threading.get_ident())
            self.__class__.barrier.wait(timeout=2)
        return super().invoke(request)


class Step3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "paper_text.md").write_text("[#result] The observation is reported here.\n", encoding="utf-8")
        atomic_write_json(self.root / "claims_final.json", {
            "claim": [
                {"number": 1, "conclusion": "c", "text": "Observed effect.", "is_pure_data": False},
                {"number": 2, "conclusion": "c", "text": "Second observed effect.", "is_pure_data": False},
                {"number": 3, "conclusion": "c", "text": "General mechanism.", "is_pure_data": False},
            ],
            "note": [],
            "relation": [
                {"connects": [1, 2, 3], "expression": "([1] 和 [2]) 是 [3] 的例子或证据"},
                {"connects": [1, 2], "expression": "[1] 与 [2] 矛盾"},
            ],
        })
        atomic_write_json(self.root / "manifest.json", {"artifacts": [
            {"path": "paper_text.md", "kind": "source.paper_text", "media_type": "text/markdown"},
            {"path": "claims_final.json", "kind": "source.claims_final", "media_type": "application/json"},
        ]})

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _recovery_store(self, name: str, *, compile_output: bool = False) -> RunStore:
        stages = [
            {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter", "options": {}},
            {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin", "options": {}},
            {"name": "strip", "plugin": f"{__name__}:StripClaimsFinalSourcePlugin", "options": {}},
            {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin",
             "options": {"tool_plugin": f"{__name__}:FakeWeakpointClassifier"}},
        ]
        if compile_output:
            stages.extend([
                {"name": "step4", "plugin": "agent_pipeline_v2.step4:Step4FormalizeReasoningPlugin", "options": {}},
                {"name": "step5", "plugin": "pipeline_harness.domain.compiler:Step5CompileGaiaIRPlugin",
                 "options": {"compiler_plugin": "pipeline_harness.domain.compiler:Gaia05OfficialCompilerTool"}},
            ])
        return RunStore.create(
            self.root / name, {"pipeline_id": "step3-recovery-test", "version": "1", "stages": stages},
            input_manifest=self.root / "manifest.json",
        )

    def _semantic_audits(self, store: RunStore) -> list[dict]:
        return [
            json.loads(store.artifact_path(ref).read_text(encoding="utf-8"))
            for ref in store.load_artifacts() if ref.kind == "tool.semantic_review.response"
        ]

    def test_cluster_validation_repair_keeps_frozen_input(self) -> None:
        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke

        def invoke(tool, request):
            response = original(tool, request)
            if request.operation == "normalize_weakpoint_clusters" and "repair_feedback" not in request.parameters:
                response.normalized["clusters"][0]["weakpoints"][0].pop("expression")
            return response

        store = self._recovery_store("cluster-repair")
        with patch.object(FakeWeakpointClassifier, "invoke", invoke):
            result = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", result.status)
        requests = [call for call in FakeWeakpointClassifier.calls if call.operation == "normalize_weakpoint_clusters"]
        self.assertEqual(2, len(requests))
        self.assertEqual(requests[0].inputs, requests[1].inputs)
        self.assertEqual(requests[0].parameters["clusters"], requests[1].parameters["clusters"])
        self.assertIn("fields", requests[1].parameters["repair_feedback"])
        audits = self._semantic_audits(store)
        self.assertTrue(all(audit["response"]["status"] == "succeeded" for audit in audits))
        cluster_audit = next(audit for audit in audits if audit["request"]["operation"] == "normalize_weakpoint_clusters")
        self.assertEqual(requests[1].to_dict(), cluster_audit["request"])
        self.assertEqual(cluster_audit["request"]["call_id"], cluster_audit["response"]["call_id"])

    def test_classification_json_and_missing_id_failures_are_repaired(self) -> None:
        original = FakeWeakpointClassifier.invoke
        for malformed in ("invalid_json", "missing_id"):
            with self.subTest(malformed=malformed):
                FakeWeakpointClassifier.calls = []
                classification_requests = []

                def invoke(tool, request):
                    response = original(tool, request)
                    if request.operation == "classify_weakpoints":
                        classification_requests.append(request)
                        return DeepSeekV4FlashObservationTool()._classify_weakpoints(request)
                    return response

                def transport(request, *, timeout):
                    if len(classification_requests) == 1:
                        content = "{invalid" if malformed == "invalid_json" else '{"classifications":[]}'
                    else:
                        content = json.dumps({"classifications": [
                            {"weakpoint_id": item["weakpoint_id"], "reasoning_type": None}
                            for item in classification_requests[-1].parameters["weakpoints"]
                        ]})
                    return io.BytesIO(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

                store = self._recovery_store(malformed)
                with (
                    patch.object(FakeWeakpointClassifier, "invoke", invoke),
                    patch("agent_pipeline_v2.step2._load_deepseek_env"),
                    patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://unused.invalid"}),
                    patch("agent_pipeline_v2.step2.urlopen", side_effect=transport) as http,
                ):
                    result = run_pipeline(store.run_dir)
                self.assertEqual("succeeded", result.status)
                self.assertEqual(2, http.call_count)
                self.assertEqual(classification_requests[0].inputs, classification_requests[1].inputs)
                self.assertEqual(classification_requests[0].parameters["weakpoints"], classification_requests[1].parameters["weakpoints"])
                self.assertTrue(classification_requests[1].parameters["repair_feedback"])
                self.assertEqual(1, sum(call.operation == "normalize_weakpoint_clusters" for call in FakeWeakpointClassifier.calls))

    def test_transport_and_auth_failures_are_not_semantically_retried(self) -> None:
        original = FakeWeakpointClassifier.invoke
        for operation in ("normalize_weakpoint_clusters", "classify_weakpoints"):
            for message in ("transport retries exhausted", "DeepSeek request failed with HTTP 401"):
                with self.subTest(operation=operation, message=message):
                    FakeWeakpointClassifier.calls = []

                    def invoke(tool, request):
                        response = original(tool, request)
                        if request.operation == operation:
                            raise RuntimeError(message)
                        return response

                    store = self._recovery_store(f"failure-{operation}-{message.split()[-1]}", compile_output=True)
                    with patch.object(FakeWeakpointClassifier, "invoke", invoke):
                        result = run_pipeline(store.run_dir)
                    self.assertEqual("failed", result.status)
                    failed_calls = [call for call in FakeWeakpointClassifier.calls if call.operation == operation]
                    self.assertEqual(1, len(failed_calls))
                    self.assertNotIn("repair_feedback", failed_calls[0].parameters)
                    refs = store.load_artifacts()
                    self.assertFalse(any(ref.kind == "formalization" and ref.metadata.get("step") == 3 for ref in refs))
                    self.assertFalse(any(ref.kind == "gaia.ir" for ref in refs))
                    audits = self._semantic_audits(store)
                    failed_audits = [audit for audit in audits if audit["response"]["status"] == "failed"]
                    self.assertEqual(1, len(failed_audits))
                    self.assertIn(message, failed_audits[0]["response"]["error"]["message"])
                    self.assertFalse(any(audit["response"].get("metadata", {}).get("fallback") for audit in audits))
                    self.assertTrue(any(item["severity"] == "error" and item["stage"] == "step3" for item in result.findings))

    def test_transport_and_semantic_repair_share_four_request_budget(self) -> None:
        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke

        def invoke(tool, request):
            response = original(tool, request)
            if request.operation == "classify_weakpoints":
                return DeepSeekV4FlashObservationTool()._classify_weakpoints(request)
            return response

        def unavailable():
            return HTTPError("https://unused.invalid", 503, "Unavailable", {}, io.BytesIO(b"temporary outage"))

        invalid = io.BytesIO(json.dumps({"choices": [{"message": {"content": "{invalid"}}]}).encode())
        store = self._recovery_store("shared-request-budget", compile_output=True)
        with (
            patch.object(FakeWeakpointClassifier, "invoke", invoke),
            patch("agent_pipeline_v2.step2._load_deepseek_env"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-key", "DEEPSEEK_BASE_URL": "https://unused.invalid"}),
            patch("agent_pipeline_v2.step2.urlopen", side_effect=[unavailable(), unavailable(), invalid, unavailable()]) as http,
            patch("agent_pipeline_v2.step2.time.sleep") as sleep,
        ):
            result = run_pipeline(store.run_dir)
        self.assertEqual("failed", result.status)
        self.assertEqual(4, http.call_count)
        self.assertEqual(2, sleep.call_count)
        classification_requests = [call for call in FakeWeakpointClassifier.calls if call.operation == "classify_weakpoints"]
        self.assertEqual(2, len(classification_requests))
        self.assertIn("JSON", classification_requests[1].parameters["repair_feedback"])
        failed_audit = next(audit for audit in self._semantic_audits(store) if audit["response"]["status"] == "failed")
        self.assertIn("budget exhausted", failed_audit["response"]["error"]["message"].lower())
        self.assertFalse(any(ref.kind == "gaia.ir" for ref in store.load_artifacts()))

    def test_one_malformed_cluster_does_not_retry_a_successful_cluster(self) -> None:
        claims = json.loads((self.root / "claims_final.json").read_text(encoding="utf-8"))
        claims["relation"].append({"connects": [2, 1], "expression": "[2] 推出 [1]"})
        atomic_write_json(self.root / "claims_final.json", claims)
        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke

        def invoke(tool, request):
            response = original(tool, request)
            if request.operation == "normalize_weakpoint_clusters":
                relations = request.parameters["clusters"][0]["candidate_relations"]
                if any(item["relation_id"] == "relation_3" for item in relations) and "repair_feedback" not in request.parameters:
                    response.normalized["clusters"][0]["weakpoints"][0].pop("expression")
            return response

        store = self._recovery_store("partial-cluster-repair")
        with patch.object(FakeWeakpointClassifier, "invoke", invoke):
            result = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", result.status)
        calls_by_cluster = {}
        for call in FakeWeakpointClassifier.calls:
            if call.operation == "normalize_weakpoint_clusters":
                cluster = call.parameters["clusters"][0]
                calls_by_cluster.setdefault(cluster["cluster_id"], []).append(call)
        self.assertEqual([1, 2], sorted(len(calls) for calls in calls_by_cluster.values()))
        self.assertEqual(1, sum(call.operation == "classify_weakpoints" for call in FakeWeakpointClassifier.calls))
        self.assertEqual(3, len(self._semantic_audits(store)))

    def test_failed_stage_resumes_from_prior_checkpoint_and_can_compile(self) -> None:
        from pipeline_harness.checks import check_run
        from weekly_reports.export_single_artifacts import export_run

        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke
        store = self._recovery_store("resume-after-outage", compile_output=True)
        with patch.object(FakeWeakpointClassifier, "invoke", side_effect=RuntimeError("transport retries exhausted")):
            failed = run_pipeline(store.run_dir)
        self.assertEqual("failed", failed.status)
        self.assertEqual(3, failed.next_stage_index)
        prior_checkpoint = failed.last_checkpoint_id
        prior_artifact_ids = {ref.artifact_id for ref in store.load_artifacts() if ref.producer_stage != "step3"}
        self.assertFalse(any(ref.kind == "gaia.ir" for ref in store.load_artifacts()))

        def valid_null_classification(tool, request):
            response = original(tool, request)
            if request.operation == "classify_weakpoints":
                for item in response.normalized["classifications"]:
                    item["reasoning_type"] = None
            return response

        with patch.object(FakeWeakpointClassifier, "invoke", valid_null_classification):
            recovered = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", recovered.status)
        self.assertEqual({"import": 1, "step1": 1, "strip": 1, "step3": 2, "step4": 1, "step5": 1}, recovered.attempts)
        self.assertNotEqual(prior_checkpoint, recovered.last_checkpoint_id)
        self.assertTrue(prior_artifact_ids <= {ref.artifact_id for ref in store.load_artifacts()})
        exported = export_run(store.run_dir, "recovered-paper", self.root / "export")
        self.assertEqual({"gaia.ir", "formalization", "knowledge.index"}, set(exported))
        self.assertTrue(all(path.is_file() for path in exported.values()))
        compile_ref = next(ref for ref in store.load_artifacts() if ref.kind == "tool.gaia_compile.response")
        compile_audit = json.loads(store.artifact_path(compile_ref).read_text(encoding="utf-8"))
        self.assertEqual("succeeded", compile_audit["response"]["status"])
        self.assertEqual("official", compile_audit["response"]["metadata"]["contract_status"])
        self.assertEqual([], [finding.to_dict() for finding in check_run(store.run_dir) if finding.severity == "error"])

    def test_failed_cluster_preserves_other_successful_cluster_audit(self) -> None:
        claims = json.loads((self.root / "claims_final.json").read_text(encoding="utf-8"))
        claims["relation"].append({"connects": [2, 1], "expression": "[2] 推出 [1]"})
        atomic_write_json(self.root / "claims_final.json", claims)
        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke

        def invoke(tool, request):
            response = original(tool, request)
            if request.operation == "normalize_weakpoint_clusters":
                relations = request.parameters["clusters"][0]["candidate_relations"]
                if any(item["relation_id"] == "relation_3" for item in relations):
                    raise RuntimeError("transport retries exhausted")
            return response

        store = self._recovery_store("partial-cluster-outage")
        with patch.object(FakeWeakpointClassifier, "invoke", invoke):
            result = run_pipeline(store.run_dir)
        self.assertEqual("failed", result.status)
        self.assertEqual(2, len(FakeWeakpointClassifier.calls))
        self.assertTrue(all(call.operation == "normalize_weakpoint_clusters" for call in FakeWeakpointClassifier.calls))
        audits = self._semantic_audits(store)
        self.assertEqual(["failed", "succeeded"], sorted(audit["response"]["status"] for audit in audits))
        self.assertFalse(any(ref.kind == "formalization" and ref.metadata.get("step") == 3 for ref in store.load_artifacts()))

    def test_cluster_requests_stay_within_four_concurrent_calls(self) -> None:
        atomic_write_json(self.root / "claims_final.json", {
            "claim": [
                {"number": index, "conclusion": "c", "text": f"Independent observation {index}.", "is_pure_data": False}
                for index in range(1, 10)
            ],
            "note": [],
            "relation": [
                {"connects": [index, 9], "expression": f"[{index}] 是 [9] 的例子或证据"}
                for index in range(1, 9)
            ],
        })
        FakeWeakpointClassifier.calls = []
        original = FakeWeakpointClassifier.invoke
        lock = threading.Lock()
        first_four_started = threading.Event()
        overflow = threading.Event()
        release = threading.Event()
        active = 0
        peak = 0

        def invoke(tool, request):
            nonlocal active, peak
            if request.operation != "normalize_weakpoint_clusters":
                return original(tool, request)
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 4:
                    first_four_started.set()
                if active > 4:
                    overflow.set()
            try:
                if not release.wait(timeout=5):
                    raise RuntimeError("test did not release request barrier")
                return original(tool, request)
            finally:
                with lock:
                    active -= 1

        store = self._recovery_store("bounded-cluster-concurrency")
        with patch.object(FakeWeakpointClassifier, "invoke", invoke), ThreadPoolExecutor(max_workers=1) as caller:
            future = caller.submit(run_pipeline, store.run_dir)
            try:
                self.assertTrue(first_four_started.wait(timeout=3))
                self.assertFalse(overflow.wait(timeout=0.1))
            finally:
                release.set()
            result = future.result(timeout=5)
        self.assertEqual("succeeded", result.status)
        self.assertEqual(4, peak)
        self.assertEqual(8, sum(call.operation == "normalize_weakpoint_clusters" for call in FakeWeakpointClassifier.calls))

    def test_transitive_shortcut_requires_full_source_coverage(self) -> None:
        # A chain that only covers a subset of a hyperedge's sources must not
        # remove the direct edge, or the uncovered observations are orphaned.
        partial = [
            (0, {"o1", "o2", "o3"}, "claim_27", "abduction"),
            (1, {"o1"}, "claim_65", "abduction"),
            (2, {"claim_65"}, "claim_27", "abduction"),
        ]
        self.assertEqual(set(), _transitive_shortcut_indexes(partial))
        # A chain whose leaves cover every direct source still replaces it.
        full = [
            (0, {"o1"}, "claim_27", "abduction"),
            (1, {"o1"}, "claim_65", "abduction"),
            (2, {"claim_65"}, "claim_27", "abduction"),
        ]
        self.assertEqual({0}, _transitive_shortcut_indexes(full))

    def test_screening_keeps_expandable_relation_but_rejects_duplicate_content(self) -> None:
        document = {
            "knowledges": {
                "claim_1": {"type": "claim", "content": {"canonical": "Observed effect"}, "source_anchor_ids": ["p1"]},
                "claim_2": {"type": "claim", "content": {"canonical": "Broader mechanism"}, "source_anchor_ids": ["p1"]},
                "claim_3": {"type": "claim", "content": {"canonical": "Observed effect"}, "source_anchor_ids": ["p1"]},
            },
            "workflow": {"source_anchors": [{"anchor_id": "p1", "source_kind": "source.paper_text"}]},
        }
        weakpoints = [
            {"id": "w1", "payload": {"evidence_claim_ids": ["claim_1"], "target_claim_id": ["claim_2"]}},
            {"id": "w2", "payload": {"evidence_claim_ids": ["claim_1"], "target_claim_id": ["claim_3"]}},
        ]
        result, rejected = _screen_classifications(document, weakpoints, {"w1": "abduction", "w2": "deduction"})
        self.assertEqual({"w1": "abduction", "w2": None}, result)
        self.assertEqual({"w2"}, rejected)

    def test_classifies_only_reasoning_type_and_records_audited_input(self) -> None:
        FakeWeakpointClassifier.calls = []
        pipeline = {
            "pipeline_id": "step3-classification-test", "version": "1",
            "stages": [
                {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter", "options": {}},
                {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin", "options": {}},
                {"name": "strip", "plugin": f"{__name__}:StripClaimsFinalSourcePlugin", "options": {}},
                {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin", "options": {"tool_plugin": f"{__name__}:FakeWeakpointClassifier"}},
            ],
        }
        store = RunStore.create(self.root / "runs", pipeline, input_manifest=self.root / "manifest.json")
        result = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", result.status)
        self.assertEqual(2, len(FakeWeakpointClassifier.calls))
        self.assertEqual("normalize_weakpoint_clusters", FakeWeakpointClassifier.calls[0].operation)
        record = FakeWeakpointClassifier.calls[0].parameters["clusters"][0]
        self.assertEqual({"cluster_id", "relation_context_id", "claims", "candidate_relations", "source_excerpts"}, set(record))
        claims = {item["claim_id"]: item["content"] for item in record["claims"]}
        self.assertEqual("Observed effect.", claims["claim_1"])
        self.assertEqual("General mechanism.", claims["claim_3"])
        formalizations = [item for item in store.load_artifacts() if item.kind == "formalization"]
        final = json.loads(store.artifact_path(formalizations[-1]).read_text(encoding="utf-8"))
        self.assertEqual([], final["workflow"]["revisions"])
        self.assertEqual([], final["workflow"]["weakpoints"])
        self.assertTrue(
            not final["workflow"]["weakpoints"]
            or any(item["code"] == "STEP3_REJECTED_WEAKPOINT" for item in result.findings)
        )
        self.assertEqual(
            {
                "id": "operator_relation_2",
                "type": "contradiction",
                "variables": ["claim_1", "claim_2"],
                "metadata": {"expression": "[1] 与 [2] 矛盾", "source_relation_id": "relation_2"},
            },
            final["graph"]["operators"][0],
        )
        self.assertEqual(2, len([item for item in store.load_artifacts() if item.kind == "tool.semantic_review.response"]))
        finding_codes = {item["code"] for item in result.findings}
        self.assertIn("STEP3_REJECTED_WEAKPOINT", finding_codes)
        self.assertIn("STEP3_NO_OBSERVATION_CLAIM", finding_codes)

    def test_claims_final_relations_are_adopted_without_llm_and_in_order(self) -> None:
        claims_final = json.loads((self.root / "claims_final.json").read_text(encoding="utf-8"))
        claims_final["relation"].append({"connects": [1, 2], "expression": "[1] 推出 [2]"})
        atomic_write_json(self.root / "claims_final.json", claims_final)
        FakeWeakpointClassifier.calls = []
        pipeline = {
            "pipeline_id": "step3-claims-final-test", "version": "1",
            "stages": [
                {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter", "options": {}},
                {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin", "options": {}},
                {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin", "options": {"tool_plugin": f"{__name__}:FakeWeakpointClassifier"}},
            ],
        }
        store = RunStore.create(self.root / "claims-final-runs", pipeline, input_manifest=self.root / "manifest.json")
        result = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", result.status)
        # High-confidence imported relations never reach the LLM judge.
        self.assertEqual([], FakeWeakpointClassifier.calls)
        final_ref = next(ref for ref in store.load_artifacts() if ref.kind == "formalization" and ref.metadata["step"] == 3)
        final = json.loads(store.artifact_path(final_ref).read_text(encoding="utf-8"))
        weakpoints = final["workflow"]["weakpoints"]
        self.assertEqual(
            ["weakpoint_relation_1", "weakpoint_relation_3"],
            [item["id"] for item in weakpoints],
        )
        payload = weakpoints[0]["payload"]
        self.assertEqual(["claim_1", "claim_2"], payload["evidence_claim_ids"])
        self.assertEqual(["claim_3"], payload["target_claim_id"])
        self.assertIsNone(payload["reasoning_type"])
        self.assertEqual("([claim_1] 和 [claim_2]) 是 [claim_3] 的例子或证据", payload["expression"])
        implied = weakpoints[1]["payload"]
        self.assertEqual(["claim_1"], implied["evidence_claim_ids"])
        self.assertEqual(["claim_2"], implied["target_claim_id"])
        self.assertEqual("[claim_1] 推出 [claim_2]", implied["expression"])
        # The fixed contradiction relation is still materialized mechanically.
        self.assertEqual(
            {
                "id": "operator_relation_2",
                "type": "contradiction",
                "variables": ["claim_1", "claim_2"],
                "metadata": {"expression": "[1] 与 [2] 矛盾", "source_relation_id": "relation_2"},
            },
            final["graph"]["operators"][0],
        )
        self.assertFalse(any(item["code"] == "STEP3_UNRESOLVED_IMPORTED_RELATION" for item in result.findings))

    def test_argument_cluster_reorients_and_merges_observations_without_expanding_two_hops(self) -> None:
        anchors = [
            {"anchor_id": "p1", "source_kind": "source.paper_text"},
            {"anchor_id": "p2", "source_kind": "source.paper_text"},
            {"anchor_id": "p3", "source_kind": "source.paper_text"},
        ]
        document = {
            "knowledges": {
                "claim_O15": {"type": "claim", "content": {"canonical": "Global permutation degraded performance."}, "source_anchor_ids": ["p1"]},
                "claim_O16": {"type": "claim", "content": {"canonical": "Local permutation moderately degraded performance."}, "source_anchor_ids": ["p1"]},
                "claim_21": {"type": "claim", "content": {"canonical": "Global and local permutation have the stated effects."}, "source_anchor_ids": ["p1"]},
                "claim_23": {"type": "claim", "content": {"canonical": "Layerwise mask statistics transfer ticket information."}, "source_anchor_ids": ["p1"]},
                "claim_20": {"type": "claim", "content": {"canonical": "Mask structure contains useful information."}, "source_anchor_ids": ["p2"]},
                "claim_99": {"type": "claim", "content": {"canonical": "A remote claim."}, "source_anchor_ids": ["p3"]},
            },
            "workflow": {"source_anchors": anchors},
        }

        def link(key, sources, target, expression, context=None):
            relation = {"expression": expression}
            if context is not None:
                relation["relation_context_id"] = context
            return {"id": key, "sources": sources, "target": target, "metadata": {"relation": relation}}

        links = [
            link("relation_27", ["claim_O15"], "claim_21", "[claim_O15] 是 [claim_21] 的证据", "experiment_mask"),
            link("relation_28", ["claim_O16"], "claim_21", "[claim_O16] 是 [claim_21] 的证据", "experiment_mask"),
            link("relation_29", ["claim_O15"], "claim_23", "[claim_O15] 是 [claim_23] 的证据", "experiment_mask"),
            link("relation_30", ["claim_O16"], "claim_23", "[claim_O16] 是 [claim_23] 的证据", "experiment_mask"),
            link("relation_8", ["claim_21"], "claim_20", "[claim_21] 推出 [claim_20]"),
            link("relation_remote", ["claim_20"], "claim_99", "[claim_20] 推出 [claim_99]"),
        ]
        with patch("agent_pipeline_v2.step3._anchor_excerpt", return_value="Source excerpt"):
            clusters = _relation_clusters(None, document, links)
        self.assertEqual(
            {"relation_27", "relation_28", "relation_29", "relation_30", "relation_8"},
            {item["relation_id"] for item in clusters[0]["candidate_relations"]},
        )
        self.assertEqual(["relation_remote"], [item["relation_id"] for item in clusters[1]["candidate_relations"]])
        normalized = {"clusters": [
            {"cluster_id": clusters[0]["cluster_id"], "weakpoints": [
                {"member_relation_ids": ["relation_27", "relation_28"], "evidence_claim_ids": ["claim_21"],
                 "target_claim_id": ["claim_O15", "claim_O16"], "reasoning_type": "deduction",
                 "expression": "[claim_21] 推出 [claim_O15] 和 [claim_O16]"},
                {"member_relation_ids": ["relation_29", "relation_30"], "evidence_claim_ids": ["claim_21"],
                 "target_claim_id": ["claim_23"], "reasoning_type": "abduction",
                 "expression": "[claim_21] 溯因支持 [claim_23]"},
                {"member_relation_ids": ["relation_8"], "evidence_claim_ids": ["claim_23"],
                 "target_claim_id": ["claim_20"], "reasoning_type": "deduction",
                 "expression": "[claim_23] 推出 [claim_20]"},
            ], "rejected_relation_ids": []},
            {"cluster_id": clusters[1]["cluster_id"], "weakpoints": [], "rejected_relation_ids": ["relation_remote"]},
        ]}
        weakpoints = _validate_cluster_result(document, clusters, normalized)
        self.assertEqual(
            [(["claim_21"], ["claim_O15", "claim_O16"]),
             (["claim_21"], ["claim_23"]),
             (["claim_23"], ["claim_20"])],
            [(item["payload"]["evidence_claim_ids"], item["payload"]["target_claim_id"]) for item in weakpoints],
        )

    def test_clusters_read_one_frozen_input_in_parallel_and_merge_once(self) -> None:
        claims = json.loads((self.root / "claims_final.json").read_text(encoding="utf-8"))
        claims["relation"].append({"connects": [2, 1], "expression": "[2] 推出 [1]"})
        atomic_write_json(self.root / "claims_final.json", claims)
        ParallelWeakpointClassifier.calls = []
        ParallelWeakpointClassifier.thread_ids = set()
        ParallelWeakpointClassifier.barrier = threading.Barrier(2)
        pipeline = {
            "pipeline_id": "step3-parallel-test", "version": "1",
            "stages": [
                {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter", "options": {}},
                {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin", "options": {}},
                {"name": "strip", "plugin": f"{__name__}:StripClaimsFinalSourcePlugin", "options": {}},
                {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin",
                 "options": {"tool_plugin": f"{__name__}:ParallelWeakpointClassifier"}},
            ],
        }
        store = RunStore.create(self.root / "parallel-runs", pipeline, input_manifest=self.root / "manifest.json")
        self.assertEqual("succeeded", run_pipeline(store.run_dir).status)
        self.assertEqual(3, len(ParallelWeakpointClassifier.calls))
        cluster_calls = [call for call in ParallelWeakpointClassifier.calls if call.operation == "normalize_weakpoint_clusters"]
        self.assertEqual(2, len(cluster_calls))
        self.assertTrue(all(len(call.parameters["clusters"]) == 1 for call in cluster_calls))
        self.assertEqual(2, len(ParallelWeakpointClassifier.thread_ids))
        step3_refs = [ref for ref in store.load_artifacts() if ref.kind == "formalization" and ref.metadata["step"] == 3]
        self.assertEqual(1, len(step3_refs))

    def test_nested_fixed_expression_is_materialized_as_an_ast(self) -> None:
        claims = json.loads((self.root / "claims_final.json").read_text(encoding="utf-8"))
        claims["relation"] = [{"connects": [1, 2, 3], "expression": "[1] 或 ([2] 且 [3])"}]
        atomic_write_json(self.root / "claims_final.json", claims)
        pipeline = {
            "pipeline_id": "step3-ast-test", "version": "1",
            "view_adapter": "agent_pipeline_v2.view:V2FormalizationViewAdapter",
            "stages": [
                {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter", "options": {}},
                {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin", "options": {}},
                {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin", "options": {}},
            ],
        }
        store = RunStore.create(self.root / "ast-runs", pipeline, input_manifest=self.root / "manifest.json")
        self.assertEqual("succeeded", run_pipeline(store.run_dir).status)
        final_ref = next(ref for ref in store.load_artifacts() if ref.kind == "formalization" and ref.metadata["step"] == 3)
        final = json.loads(store.artifact_path(final_ref).read_text(encoding="utf-8"))
        by_type = {item["type"]: item for item in final["graph"]["operators"]}
        self.assertEqual(["claim_2", "claim_3"], by_type["conjunction"]["variables"])
        helper = by_type["conjunction"]["conclusion"]
        self.assertEqual(["claim_1", helper], by_type["disjunction"]["variables"])
        self.assertIn(helper, final["graph"]["nodes"])
        view = project_run(store.run_dir)
        helper_node = next(node for node in view.nodes if node.get("step") == 3 and node.get("entity_id") == helper)
        self.assertEqual("formal_internal", helper_node["details"]["visibility"])
        self.assertEqual("helpers", helper_node["layer"])
        self.assertEqual(["standard"], helper_node["visible_at"])
        self.assertFalse(any(document["title"] == helper for document in view.search_documents))


    def _jev_store(self):
        (self.root / "paper_text.md").write_text(
            "[#result] Observed effect. Second observed effect. General mechanism.\n", encoding="utf-8")
        return RunStore.create(self.root / "jev-runs", {
            "pipeline_id": "jev-classification-test", "version": "1",
            "stages": [
                {"name": "import", "plugin": "agent_pipeline_v2.step1:ClaimsFinalInputImporter"},
                {"name": "step1", "plugin": "agent_pipeline_v2.step1:Step1ImportClaimsFinalPlugin"},
                {"name": "step2", "plugin": "agent_pipeline_v2.step2:Step2ExtractObservationsPlugin",
                 "options": {"tool_plugin": "agent_pipeline_v2.step2:DeepSeekV4FlashObservationTool"}},
                {"name": "step3", "plugin": "agent_pipeline_v2.step3:Step3AnalyzeReasoningPlugin"},
            ],
        }, input_manifest=self.root / "manifest.json")

    @staticmethod
    def _latest_formalization(store):
        ref = max((ref for ref in store.load_artifacts() if ref.kind == "formalization"),
                  key=lambda ref: int(ref.metadata.get("step", 0)))
        return json.loads(store.artifact_path(ref).read_text(encoding="utf-8"))

    @staticmethod
    def _jev_answer(choice="abduction"):
        return {"type": "choice", "choice": choice, "confidence": 1.0,
                "probabilities": {key: float(key == choice) for key in ("deduction", "abduction", "analogy", "null")}}

    def test_jev_classifies_trusted_import_without_changing_endpoints_and_preserves_raw_probabilities(self):
        store = self._jev_store()
        raw = {"model": "jev-1.14.0", "answers": {"w_0": self._jev_answer()}}
        with patch.dict(os.environ, {"GAIA_RELATION_CLASSIFIER": "jev", "TYPESAFE_API_KEY": "test-typesafe-key",
                                     "TYPESAFE_MODEL": "jev-1.14.0"}), \
                patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as send:
            run = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", run.status, run.findings)
        send.assert_called_once()
        request = send.call_args.args[0]
        self.assertEqual("https://api.typesafe.ai/v1/systemone", request.full_url)
        self.assertEqual("Bearer test-typesafe-key", request.get_header("Authorization"))
        body = json.loads(request.data)
        self.assertEqual("jev-1.14.0", body["model"])
        self.assertEqual({"model", "state", "questions"}, set(body))
        self.assertEqual({"deduction", "abduction", "analogy", "null"}, set(body["questions"]["w_0"]["criteria"]))
        self.assertEqual(["claim_1", "claim_2"], [c["claim_id"] for c in body["state"]["w_0"]["evidence_claims"]])
        final = self._latest_formalization(store)
        payload = final["workflow"]["weakpoints"][0]["payload"]
        self.assertEqual("abduction", payload["reasoning_type"])
        self.assertEqual(["claim_1", "claim_2"], payload["evidence_claim_ids"])
        self.assertEqual(["claim_3"], payload["target_claim_id"])
        audits = [a for a in self._semantic_audits(store) if a["request"]["operation"] == "classify_weakpoints"]
        self.assertEqual(1, len(audits))
        self.assertEqual("jev-weakpoint-classifier", audits[0]["request"]["tool_name"])
        self.assertEqual(raw, audits[0]["response"]["raw"]["response"])
        self.assertEqual("jev-1.14.0", audits[0]["response"]["metadata"]["model"])

    def test_jev_null_maps_to_existing_null_without_deepseek_fallback(self):
        store = self._jev_store()
        raw = {"model": "jev-1.14.0", "answers": {"w_0": self._jev_answer("null")}}
        with patch.dict(os.environ, {"GAIA_RELATION_CLASSIFIER": "jev", "TYPESAFE_API_KEY": "test-key"}), \
                patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as send:
            run = run_pipeline(store.run_dir)
        self.assertEqual("succeeded", run.status, run.findings)
        send.assert_called_once()
        self.assertIsNone(self._latest_formalization(store)["workflow"]["weakpoints"][0]["payload"]["reasoning_type"])

    def test_jev_auth_error_fails_stage_and_is_audited_without_fallback(self):
        store = self._jev_store()
        error = HTTPError("https://api.typesafe.ai/v1/systemone", 401, "Unauthorized", {}, None)
        with patch.dict(os.environ, {"GAIA_RELATION_CLASSIFIER": "jev", "TYPESAFE_API_KEY": "test-key"}), \
                patch("agent_pipeline_v2.step2.urlopen", side_effect=error) as send:
            run = run_pipeline(store.run_dir)
        self.assertEqual("failed", run.status)
        self.assertEqual("step3", run.current_stage)
        send.assert_called_once()
        audits = [a for a in self._semantic_audits(store) if a["request"]["operation"] == "classify_weakpoints"]
        self.assertEqual("failed", audits[0]["response"]["status"])
        self.assertEqual("HTTPError", audits[0]["response"]["error"]["type"])

    def test_jev_missing_key_fails_before_any_network_request(self):
        store = self._jev_store()
        with patch.dict(os.environ, {"GAIA_RELATION_CLASSIFIER": "jev", "TYPESAFE_API_KEY": ""}), \
                patch("agent_pipeline_v2.step2.urlopen") as send:
            run = run_pipeline(store.run_dir)
        self.assertEqual("failed", run.status)
        self.assertTrue(any("TYPESAFE_API_KEY" in f["message"] for f in run.findings))
        send.assert_not_called()

    def test_jev_rejects_malformed_or_missing_decisions_instead_of_guessing(self):
        from copy import deepcopy
        tool = DeepSeekV4FlashObservationTool()
        tool.name, tool.model = "jev-weakpoint-classifier", "jev-1.14.0"
        request = ToolCallRequest("jev-test", tool.name, tool.version, "classify_weakpoints", [],
                                  {"weakpoints": [{"weakpoint_id": "w1"}]})
        valid = {"model": "jev-1.14.0", "answers": {"w_0": self._jev_answer()}}
        variants = [dict(valid, answers={})]
        for key, value in [("choice", "entailment"), ("confidence", float("nan")),
                           ("probabilities", {"abduction": 1.0}),
                           ("probabilities", {"deduction": 0.8, "abduction": 0.1, "analogy": 0.05, "null": 0.05})]:
            raw = deepcopy(valid)
            raw["answers"]["w_0"][key] = value
            variants.append(raw)
        for raw in variants:
            with self.subTest(raw=raw), patch.dict(os.environ, {"TYPESAFE_API_KEY": "test-key"}), \
                    patch("agent_pipeline_v2.step2.urlopen", return_value=io.BytesIO(json.dumps(raw).encode())) as send:
                with self.assertRaises(ValueError):
                    tool.invoke(request)
                send.assert_called_once()


if __name__ == "__main__":
    unittest.main()
