from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipeline_harness.checks import check_run
from pipeline_harness.runner import run_pipeline
from pipeline_harness.store import RunStore, read_json
from pipeline_harness.synthetic import SYNTHETIC_PIPELINE
from pipeline_harness.view.model import ViewDocument, search_view
from pipeline_harness.view.projector import project_run


class ViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = RunStore.create(self.root, SYNTHETIC_PIPELINE)
        run_pipeline(self.store.run_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_projects_standalone_view_and_searches_unicode(self) -> None:
        view = project_run(self.store.run_dir)
        viewer = self.store.run_dir / "views/viewer.html"
        self.assertTrue(viewer.is_file())
        html = viewer.read_text(encoding="utf-8")
        self.assertNotIn("__VIEW_DATA__", html)
        self.assertIn("SYNTH_NODE_001", html)
        self.assertEqual(3, len(search_view(view, "合成条目")))
        exact = search_view(view, "ＳＹＮＴＨ＿ＮＯＤＥ＿００１")
        self.assertEqual("SYNTH_NODE_001", exact[0]["id"])

    def test_unfilled_template_explains_how_to_generate_a_view(self) -> None:
        template = (
            Path(__file__).parents[1]
            / "pipeline_harness"
            / "view"
            / "viewer.html"
        ).read_text(encoding="utf-8")
        self.assertIn("这是 Viewer 模板", template)
        self.assertIn("pipeline_harness view", template)
        self.assertIn('<option value="standard" selected>Standard</option>', template)
        self.assertNotIn('<option value="audit">', template)
        self.assertIn("get('dev') === '1'", template)
        self.assertIn("'3':new Set(['claims','weakpoints','operators'])", template)
        self.assertIn("const operators=view.nodes.filter(node => String(node.step) === step && node.kind === 'operator');", template)
        self.assertIn("equivalence:'≡'", template)
        self.assertIn(".edge.equivalence { marker-start", template)
        self.assertIn("function directOperatorEdges", template)
        self.assertNotIn("visual_reorientation:true", template)
        self.assertIn("function layoutConstraintEdges", template)
        self.assertIn("function connectedComponents", template)
        self.assertIn("function componentLayout", template)
        self.assertIn("engine:'dagre-components'", template)
        self.assertIn("function isAlternativePlaceholder", template)
        self.assertIn("eoPair?14:groupsFor(edge).length?8:1", template)
        self.assertNotIn("function relationalLayoutGroups", template)
        self.assertIn("item.kind==='weakpoint'", template)
        self.assertIn("toggleWeakpoint(item)", template)
        self.assertIn('id="detail"', template)
        self.assertIn("propertyRow('reasoning type',reasoningType)", template)

    def test_clean_projection_passes_check(self) -> None:
        project_run(self.store.run_dir)
        self.assertEqual([], check_run(self.store.run_dir))

    def test_check_detects_stale_view_hash(self) -> None:
        project_run(self.store.run_dir)
        view_path = self.store.run_dir / "views/view_model.json"
        payload = read_json(view_path)
        payload["title"] = "tampered"
        view_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
        codes = {finding.code for finding in check_run(self.store.run_dir)}
        self.assertIn("VIEW_HASH_MISMATCH", codes)


if __name__ == "__main__":
    unittest.main()
