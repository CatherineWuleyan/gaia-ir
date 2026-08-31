import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_harness.runner import run_pipeline
from pipeline_harness.store import RunStore, atomic_write_json, read_json


def _ir(name, cid, content, category=None):
    metadata = {"category": category} if category else {}
    return {"namespace":"papers","package_name":name,"scope":"local","ir_hash":"sha256:"+"0"*64,
            "knowledges":[{"id":f"papers:{name}::{cid}","content":content,"metadata":metadata,"source_anchor_ids":[f"a:{name}"]}],"operators":[],"strategies":[],"composes":[],"formula_graphs":[]}

def _config():
    stages=[
        {"name":"step0_select_scope","plugin":"pipeline_merge.step0:Step0SelectScopePlugin","options":{"mode":"incremental","scope":{"domain":"test","selection":["papers:new"]}}},
        {"name":"step1_freeze_inputs","plugin":"pipeline_merge.step1:Step1FreezeInputsPlugin","options":{}},
        {"name":"step2_retrieve_domain_local","plugin":"pipeline_merge.step2:Step2RetrieveDomainLocalPlugin","options":{}},
        {"name":"step3_identify_structures","plugin":"pipeline_merge.step3:Step3IdentifyStructuresPlugin","options":{}},]
    return {"pipeline_id":"pipeline-merge-step3-test","version":"0.4.0","stages":stages}

class Step3Tests(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def _run(self, old_content):
        atomic_write_json(self.root/"new.json", _ir("new","a","sparse networks transfer across dataset settings"))
        atomic_write_json(self.root/"old.json", _ir("old","b",old_content))
        atomic_write_json(self.root/"manifest.json", {"artifacts":[{"path":"new.json","kind":"gaia.ir"},{"path":"old.json","kind":"gaia.ir"}]})
        store=RunStore.create(self.root/"runs",_config(),input_manifest=self.root/"manifest.json")
        with patch("pipeline_merge.step1._validate_with_official_gaia",side_effect=lambda payload:(payload,"test")), \
             patch("pipeline_merge.step3._post_json", side_effect=self._llm):
            run=run_pipeline(store.run_dir)
        ref=next(r for r in store.load_artifacts() if r.kind=="integration.step3_proposals")
        return run,read_json(store.artifact_path(ref))

    @staticmethod
    def _llm(prompt):
        import json
        candidates = json.loads(prompt.split("CANDIDATES=", 1)[1])
        if "operator_candidate|not_operator" in prompt:
            decision = "not_operator" if "model settings" in prompt else "operator_candidate"
            return {"decisions": [{"candidate_id": item["candidate_id"], "decision": decision} for item in candidates]}
        if "deterministic operator candidates" in prompt:
            return {"decisions": [{"candidate_id": item["candidate_id"], "type": "equivalence", "variables": item["source_qids"], "expression": "equivalent"} for item in candidates]}
        return {"weakpoints": []}

    def test_exact_cross_package_hit_becomes_operator(self):
        run, proposal = self._run("sparse networks transfer across dataset settings")
        self.assertEqual(run.status, "succeeded")
        self.assertEqual(proposal["operators"][0]["type"], "equivalence")

    def test_non_exact_hit_is_rejected_not_invented(self):
        run, proposal = self._run("sparse networks transfer across model settings")
        self.assertEqual(run.status, "succeeded")
        self.assertEqual(proposal["operators"], [])
        self.assertEqual(proposal["weakpoints"], [])

    def test_same_category_equivalence_merges_nodes_and_keeps_anchors(self):
        atomic_write_json(self.root/"new.json", _ir("new","a","sparse networks transfer across dataset settings", "O"))
        atomic_write_json(self.root/"old.json", _ir("old","b","sparse networks transfer across dataset settings", "O"))
        atomic_write_json(self.root/"manifest.json", {"artifacts":[{"path":"new.json","kind":"gaia.ir"},{"path":"old.json","kind":"gaia.ir"}]})
        store=RunStore.create(self.root/"runs",_config(),input_manifest=self.root/"manifest.json")
        with patch("pipeline_merge.step1._validate_with_official_gaia",side_effect=lambda payload:(payload,"test")), \
             patch("pipeline_merge.step3._post_json", side_effect=self._llm):
            run=run_pipeline(store.run_dir)
        ref=next(r for r in store.load_artifacts() if r.kind=="integration.step3_proposals")
        proposal=read_json(store.artifact_path(ref))
        self.assertEqual(run.status, "succeeded")
        self.assertEqual(len(proposal["equivalence_merges"]), 1)
        self.assertEqual(set(proposal["equivalence_merges"][0]["anchor_ids"]), {"a:new", "a:old"})


if __name__ == "__main__":
    unittest.main()
