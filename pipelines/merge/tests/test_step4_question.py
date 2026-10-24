"""Regression tests for the question-led merge pipeline."""
from pathlib import Path
import shutil
import subprocess
import unittest

from pipeline_merge_question.step4 import _package_name
from pipeline_merge_question.question_layer import build_answer_layer, evaluate, papers_of_node


class QuestionStep4Tests(unittest.TestCase):
    def test_package_name_is_qid_safe(self):
        self.assertEqual(
            _package_name("Knowledge Graphs for AI for Science, LLM Agents"),
            "knowledge_graphs_for_ai_for_science_llm_agents",
        )
        self.assertRegex(_package_name("AI/LLM + 中文"), r"^[a-z0-9_]+$")
        self.assertEqual(_package_name("+++"), "domain")


    @unittest.skipUnless(shutil.which("node"), "Node.js is required for viewer regression tests")
    def test_viewer_hides_components_without_visible_path_to_root(self):
        template = (Path(__file__).parents[1] / "pipeline_merge_question" / "viewer_question.html").read_text()
        start = template.index("  function applyFilters(){")
        end = template.index("  function updateSearch(){", start)
        script = r"""
const assert = require('node:assert/strict');
function element() {
  const classes = new Set();
  return {style: {}, setAttribute() {}, textContent: '', classList: {
    toggle(k, on) { if (on) classes.add(k); else classes.delete(k); },
    contains(k) { return classes.has(k); }, add(k) { classes.add(k); }
  }};
}
const controls = {'toggle-isolated': {checked: true}, 'toggle-delta': {checked: false},
  'min-degree': {value: '1'}, 'toggle-full-domain': {checked: false}};
const document = {getElementById(id) { return controls[id]; }};
const nodes = ['R1', 'bridge', 'leaf', 'island1', 'island2', 'isolated'].map(id =>
  ({id, isRoot: id === 'R1', connected: id !== 'isolated', degree: id === 'leaf' ? 3 : 1, el: element()}));
const nodeIndex = new Map(nodes.map((n, i) => [n.id, i]));
const answerRoots = [{id: 'R1'}];
const answerChainNodes = new Set(['bridge', 'leaf', 'isolated']);
const crossPaperNodes = new Set(['leaf', 'island1', 'island2']);
const exploratoryNodes = new Set();
const exploratoryEdges = new Set();
const answerChainEdges = new Set();
let selected = 'island1';
const edgeEls = [['bridge', 'R1'], ['leaf', 'bridge'], ['island1', 'island2']].map(([source, target], i) =>
  ({edge: {id: 'e' + i, source, target, answer: i === 0, layer: i === 0 ? 'answer' : '',
    cross_paper: i > 0, reasoning_type: 'infer'}, path: element(), text: element()}));
function paint() {}
""" + template[start:end] + r"""
function visible() { return nodes.filter(n => !n.el.classList.contains('hidden')).map(n => n.id); }
applyFilters();
assert.deepEqual(visible(), ['R1', 'bridge', 'leaf']);
assert.equal(edgeEls[2].edge.hidden, true);
// Filtering out the bridge must also hide its otherwise eligible leaf.
controls['min-degree'].value = '2';
applyFilters();
assert.deepEqual(visible(), ['R1']);
controls['min-degree'].value = '1';
// The explicit full-domain mode can still reveal disconnected components.
controls['toggle-full-domain'].checked = true;
applyFilters();
assert.deepEqual(visible(), nodes.map(n => n.id));
controls['toggle-full-domain'].checked = false;
applyFilters();
assert.deepEqual(visible(), ['R1', 'bridge', 'leaf']);
// Generic graphs without an answer layer retain their original visibility.
answerRoots.length = 0;
applyFilters();
assert.deepEqual(visible(), nodes.map(n => n.id));
"""
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)



    @unittest.skipUnless(shutil.which("node"), "Node.js is required for viewer regression tests")
    def test_viewer_centers_root_graph_in_available_canvas(self):
        template = (Path(__file__).parents[1] / "pipeline_merge_question" / "viewer_question.html").read_text()
        start = template.index("  function viewportRect(){")
        end = template.index("  function runLayout(", start)
        script = r"""
const assert = require('node:assert/strict');
const W = () => 1600, H = () => 1048;
const document = {getElementById() { return {classList: {contains() { return false; }}}; }};
const nodes = [
  {index: 0, r: 40, isRoot: true, el: {classList: {contains() {return false;}}}},
  {index: 1, r: 10, isRoot: false, el: {classList: {contains() {return false;}}}},
  {index: 2, r: 50, isRoot: false, el: {classList: {contains() {return true;}}}}
];
const state = {positions: [{x: 100, y: 100}, {x: 1000, y: 300}, {x: -9000, y: -9000}]};
let transform;
function applyTransform() {}
function paint() {}
""" + template[start:end] + r"""
fit();
const rect = viewportRect();
// Bounds include the viewer's node and text padding; hidden nodes do not affect fit.
const centerX = ((100 - 40 - 10) + (1000 + 10 + 10)) / 2;
const centerY = ((100 - 40 - 4) + (300 + 10 + 24)) / 2;
assert.ok(Math.abs(transform.x + centerX * transform.k - (rect.left + rect.width / 2)) < 1e-8);
assert.ok(Math.abs(transform.y + centerY * transform.k - (rect.top + rect.height / 2)) < 1e-8);
"""
        result = subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class QuestionPaperProvenanceTests(unittest.TestCase):
    @staticmethod
    def node(node_id, provenance=None):
        metadata = {} if provenance is None else {"package_provenance": provenance}
        return {"id": node_id, "kind": "claim", "details": {"metadata": metadata}}

    def layer(self, nodes):
        view = {"nodes": nodes, "edges": []}
        roots = [{"id": "answer:r1", "code": "R1", "statement": "candidate"}]
        links = {"R1": [{"claim_id": n["id"], "relation": "support", "reason": "linked"}
                        for n in nodes]}
        return view, build_answer_layer(view, {"zh": "问题", "en": "question"}, roots, links)

    def test_unknown_k_cannot_supply_a_second_paper(self):
        nodes = [self.node("knowledge:papers:paper_a::claim_1"),
                 self.node("knowledge:papers:paper_a::claim_2"),
                 self.node("knowledge:candidate_K_1")]
        view, layer = self.layer(nodes)
        root = layer["roots"][0]
        self.assertEqual(root["stats"]["support_claim_count"], 3)
        self.assertEqual(root["stats"]["support_papers"], ["paper_a"])
        self.assertEqual(root["stats"]["support_paper_count"], 1)
        self.assertEqual(root["verdict_class"], "partial")
        result = evaluate(view, layer)
        self.assertEqual(result["base"]["paper_count"], 1)
        self.assertEqual(result["answer_layer"]["papers_touched"], 1)
        self.assertEqual(result["answer_layer"]["claims_per_paper"], {"paper_a": 2})

    def test_unknown_only_is_exploratory_and_zero_papers(self):
        view, layer = self.layer([self.node("knowledge:candidate_K_1")])
        root = layer["roots"][0]
        self.assertEqual(root["support_ids"], ["knowledge:candidate_K_1"])
        self.assertEqual(root["stats"]["support_papers"], [])
        self.assertEqual(root["verdict_class"], "exploratory")
        result = evaluate(view, layer)
        self.assertEqual(result["base"]["paper_count"], 0)
        self.assertEqual(result["answer_layer"]["papers_touched"], 0)
        self.assertEqual(result["answer_layer"]["claims_per_paper"], {})

    def test_known_k_provenance_counts_real_papers_once(self):
        k = self.node("knowledge:candidate_K_1", ["papers:paper_b", "papers:paper_a", "papers:paper_a"])
        self.assertEqual(papers_of_node(k), ["paper_a", "paper_b"])
        nodes = [k, self.node("knowledge:papers:paper_a::claim_1"),
                 self.node("knowledge:papers:paper_b::claim_1")]
        view, layer = self.layer(nodes)
        self.assertEqual(layer["roots"][0]["stats"]["support_papers"], ["paper_a", "paper_b"])
        self.assertEqual(layer["roots"][0]["verdict_class"], "primary")
        result = evaluate(view, layer)
        self.assertEqual(result["base"]["paper_count"], 2)
        self.assertEqual(result["answer_layer"]["claims_per_paper"], {"paper_a": 2, "paper_b": 2})

    def test_unknown_or_malformed_provenance_is_not_a_paper(self):
        for provenance in (None, [], ["other", "papers:other", "papers:", "candidate_K_1", 4],
                           "papers:paper_a", {"papers:paper_a": True}):
            with self.subTest(provenance=provenance):
                self.assertEqual(papers_of_node(self.node("knowledge:candidate_K_1", provenance)), [])
        for node_id in ("knowledge:other::claim_1", "knowledge:papers:other::claim_1",
                        "candidate_K_1::claim_1", "claim_1"):
            with self.subTest(node_id=node_id):
                self.assertEqual(papers_of_node(self.node(node_id)), [])

    def test_direct_paper_namespace_still_resolves_without_metadata(self):
        for node_id in ("knowledge:papers:paper_a::claim_1", "papers:paper_a::claim_1"):
            with self.subTest(node_id=node_id):
                self.assertEqual(papers_of_node(self.node(node_id)), ["paper_a"])

if __name__ == "__main__":
    unittest.main()
