"""Regression tests for the question-led merge pipeline."""
from pathlib import Path
import json
import shutil
import subprocess
import unittest

from pipeline_merge_question.step4 import _package_name
from pipeline_merge_question.question_layer import (
    _normalize_links, build_answer_layer, candidate_pool, evaluate, link_roots, papers_of_node,
)


class QuestionEvidenceCoverageTests(unittest.TestCase):
    @staticmethod
    def root():
        return {"code": "R1", "title_zh": "模型", "title_en": "Nonlinear model",
                "statement": "A nonlinear model", "criteria": "Nonlinear equations",
                "scope": "The specified system", "answer_type": "modeling_choice",
                "seed_claim_ids": []}

    @classmethod
    def workflow_root(cls):
        root = cls.root()
        root["answer_type"] = "method_route"
        return root

    @staticmethod
    def view(count=145):
        return {"nodes": [{"id": f"knowledge:papers:paper_a::claim_{i}",
                           "display_label": f"P{i}", "kind": "claim", "layer": "claims",
                           "summary": "完整证据，不依赖英文字面重合。"}
                          for i in range(count)], "edges": []}

    def test_every_claim_is_visited_and_more_than_twelve_links_survive(self):
        view = self.view()
        visited = []

        def call(prompt):
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            visited.extend(item["id"] for item in batch)
            return {"links": [{"claim_id": item["id"], "relation": "support",
                               "reason": "States the specified model."} for item in batch]}

        result, raw = link_roots(view, {"en": "Model?"}, [self.root()], call_json=call)
        expected = {item["id"] for item in view["nodes"]}
        self.assertEqual(set(visited), expected)
        self.assertEqual(len(visited), 2 * len(expected))
        self.assertEqual({item["claim_id"] for item in result["R1"]}, expected)
        self.assertGreater(len(raw), 1)
        self.assertTrue(all(len(item["candidates"]) <= 64 for item in raw))

    def test_support_must_survive_the_separate_evidence_check(self):
        view = self.view(2)
        calls = []

        def call(prompt):
            calls.append(prompt)
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            selected = batch if len(calls) == 1 else batch[:1] if len(calls) == 2 else []
            return {"links": [{"claim_id": item["id"], "relation": "support", "reason": "Evidence."}
                               for item in selected]}

        result, _ = link_roots(view, {"en": "Model?"}, [self.workflow_root()], call_json=call)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(result["R1"]), 1)

    def test_sparse_verification_is_resolved_with_individual_decisions(self):
        calls = []

        def call(prompt):
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            calls.append(batch)
            if len(calls) == 1:
                return {"links": [{"claim_id": item["id"], "relation": "support", "reason": "Proposed."}
                                   for item in batch]}
            if len(calls) == 2:
                return {"links": []}
            return {"links": [{"claim_id": batch[0]["id"],
                               "relation": "support" if len(calls) == 3 else "bounds",
                               "reason": "Reports the implemented component under the stated condition."}]}

        result, raw = link_roots(self.view(2), {"en": "Model?"}, [self.workflow_root()], call_json=call)
        self.assertEqual([len(batch) for batch in calls], [2, 2, 1, 1])
        self.assertEqual(len(raw), 4)
        self.assertEqual([link["claim_id"] for link in result["R1"]], [calls[2][0]["id"]])
        self.assertIn("stated condition", result["R1"][0]["reason"])

    def test_a_single_claim_rejection_is_not_retried_or_implicitly_retained(self):
        calls = []

        def call(prompt):
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            calls.append(batch)
            return {"links": [{"claim_id": batch[0]["id"], "relation": "support", "reason": "Proposed."}]
                    if len(calls) == 1 else []}

        result, _ = link_roots(self.view(1), {"en": "Model?"}, [self.root()], call_json=call)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["R1"], [])

    def test_model_choices_do_not_promote_unconfirmed_claims_through_isolated_votes(self):
        calls = []

        def call(prompt):
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            calls.append(batch)
            if len(calls) == 1:
                return {"links": [{"claim_id": item["id"], "relation": "support", "reason": "Proposed."}
                                   for item in batch]}
            if len(calls) == 2:
                return {"links": []}
            self.fail("Model choices must not seek another isolated vote after batch rejection")

        result, _ = link_roots(self.view(2), {"en": "Model?"}, [self.root()], call_json=call)
        self.assertEqual(len(calls), 2)
        self.assertEqual(result["R1"], [])

    def test_malformed_judge_output_is_not_treated_as_semantic_rejection(self):
        for response in ({}, {"links": None}, []):
            with self.subTest(response=response):
                with self.assertRaisesRegex(ValueError, "links array"):
                    link_roots(self.view(1), {"en": "Model?"}, [self.root()], call_json=lambda _: response)

    def test_individual_review_failure_does_not_return_partial_success(self):
        calls = []

        def call(prompt):
            calls.append(prompt)
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            if len(calls) == 1:
                return {"links": [{"claim_id": item["id"], "relation": "support", "reason": "Proposed."}
                                   for item in batch]}
            if len(calls) == 2:
                return {"links": []}
            raise RuntimeError("API unavailable")

        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            link_roots(self.view(2), {"en": "Model?"}, [self.workflow_root()], call_json=call)

    def test_expanding_a_pool_preserves_its_prefix_and_full_qualifications(self):
        view = self.view(80)
        view["nodes"][0]["summary"] = "Background " * 50 + "Only valid for a different material."
        view["nodes"][0]["details"] = {"relevance_role": "TEST_ANSWER_DO_NOT_SEND"}
        narrow = candidate_pool(view, self.root(), limit=44)
        wider = candidate_pool(view, self.root(), limit=60, broaden=True)
        self.assertEqual(narrow, wider[:len(narrow)])
        complete = candidate_pool(view, self.root(), limit=80)
        claim = next(item for item in complete if item["id"] == view["nodes"][0]["id"])
        self.assertTrue(claim["content"].endswith("Only valid for a different material."))
        self.assertNotIn("TEST_ANSWER_DO_NOT_SEND", json.dumps(claim))

    def test_a_unique_display_code_is_resolved_but_ambiguous_codes_are_not(self):
        candidates = [{"id": "a", "code": "P1"}, {"id": "b", "code": "duplicate"},
                      {"id": "c", "code": "duplicate"}]
        response = {"links": [{"claim_id": code, "relation": "support", "reason": "Evidence."}
                              for code in ("P1", "duplicate", "unknown")]}
        links = _normalize_links(response, self.root(), candidates)
        self.assertEqual([item["claim_id"] for item in links], ["a"])

    def test_conflicting_relations_are_dropped_before_deduplication(self):
        response = {"links": [
            {"claim_id": "a", "relation": relation, "reason": "Evidence."}
            for relation in ("support", "oppose")] + [
            {"claim_id": "b", "relation": "support", "reason": "Evidence."}] * 2}
        links = _normalize_links(response, self.root(), [{"id": "a"}, {"id": "b"}])
        self.assertEqual([(item["claim_id"], item["relation"]) for item in links], [("b", "support")])

    def test_batch_failure_does_not_return_a_partial_success(self):
        calls = []

        def call(prompt):
            calls.append(prompt)
            if len(calls) > 1:
                raise RuntimeError("API unavailable")
            batch = json.loads(prompt.split("\nCANDIDATES=", 1)[1])
            return {"links": [{"claim_id": batch[0]["id"], "relation": "support", "reason": "Evidence."}]}

        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            link_roots(self.view(), {"en": "Model?"}, [self.root()], call_json=call)


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

    def test_a_single_marginal_second_paper_is_not_multi_paper_support(self):
        # Nine claims from one paper plus one from another is a single-paper
        # result with a marginal second citation, not cross-paper support.
        nodes = [self.node(f"knowledge:papers:paper_a::claim_{i}") for i in range(1, 10)]
        nodes.append(self.node("knowledge:papers:paper_b::claim_1"))
        view, layer = self.layer(nodes)
        root = layer["roots"][0]
        self.assertEqual(root["stats"]["support_paper_count"], 2)
        self.assertEqual(root["stats"]["second_paper_support"], 1)
        self.assertEqual(root["stats"]["support_paper_counts"], {"paper_a": 9, "paper_b": 1})
        self.assertFalse(root["stats"]["multi_paper_corroborated"])
        self.assertEqual(root["verdict_class"], "partial")
        self.assertEqual(root["verdict"], "single-paper core")

    def test_two_real_second_paper_claims_restore_primary(self):
        nodes = [self.node(f"knowledge:papers:paper_a::claim_{i}") for i in range(1, 4)]
        nodes += [self.node("knowledge:papers:paper_b::claim_1"),
                  self.node("knowledge:papers:paper_b::claim_2")]
        view, layer = self.layer(nodes)
        root = layer["roots"][0]
        self.assertEqual(root["stats"]["second_paper_support"], 2)
        self.assertTrue(root["stats"]["multi_paper_corroborated"])
        self.assertEqual(root["verdict_class"], "primary")

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
