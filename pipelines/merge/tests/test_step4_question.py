"""Regression tests for the question-led merge pipeline."""
from pathlib import Path
import shutil
import subprocess
import unittest

from pipeline_merge_question.step4 import _package_name


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


if __name__ == "__main__":
    unittest.main()
