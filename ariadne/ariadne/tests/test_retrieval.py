"""Tests for retrieval parsing and offline materialisation (no network)."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ariadne import retrieval
from ariadne.config import Config, Runtime
from ariadne.util import AriadneError


def _runtime(tmp: Path, **overrides) -> Runtime:
    config = Config.load(tmp / "config.json")
    values = dict(
        config=config,
        model_api_key="test-key",
        model_base_url="https://api.example.com/v1",
        model="test-model",
        merge_model="test-model",
        bohr_profile="ariadne-test",
        bohr_access_key="",
        bohr_host="https://open.bohrium.com",
        auto_approve_spend=True,
        payment_source="photon",
        python=sys.executable,
        bohr_bin="/usr/local/bin/bohr",
    )
    values.update(overrides)
    return Runtime(**values)


class SearchCommandTest(unittest.TestCase):
    def test_search_passes_profile_billing_and_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            payload = {
                "ok": True,
                "data": {
                    "papers": {
                        "papers:paper_1": {
                            "paper_id": "123456789",
                            "en_title": "Aircraft simulation",
                            "doi": "10.48550/arxiv.2501.00001",
                            "authors": ["A", "B"],
                        }
                    }
                },
            }
            captured = {}

            def fake_run(cmd, **kwargs):
                captured["cmd"] = list(cmd)
                return subprocess.CompletedProcess(
                    cmd, 0, stdout=json.dumps(payload), stderr=""
                )

            with mock.patch("ariadne.retrieval.run", side_effect=fake_run):
                records = retrieval.search("aircraft", 2, root, runtime)

            command = captured["cmd"]
            self.assertIn("--profile", command)
            self.assertEqual(command[command.index("--profile") + 1], "ariadne-test")
            self.assertIn("--yes", command)
            self.assertIn("--payment-source", command)
            self.assertEqual(command[-2:], ["-o", "json"])
            self.assertEqual(records[0]["lkm_id"], "123456789")
            self.assertEqual(records[0]["arxiv"], "2501.00001")
            self.assertEqual(records[0]["authors"], "A, B")
            self.assertTrue((root / "01_retrieval" / "search_raw.json").is_file())

    def test_variables_fallback(self) -> None:
        payload = {
            "ok": True,
            "data": {
                "papers": {},
                "variables": [
                    {"paper_id": "123456789", "title": "First paper"},
                    {"paper_id": "987654321", "title": "Second paper"},
                    {"paper_id": "123456789", "title": "duplicate"},
                ],
            },
        }
        records = retrieval._records_from_search(payload)
        self.assertEqual([item["lkm_id"] for item in records],
                         ["123456789", "987654321"])


class GraphNormalisationTest(unittest.TestCase):
    def test_bare_graph_is_wrapped(self) -> None:
        raw = {"nodes": [{"id": "n1"}], "edges": []}
        graph = retrieval._normalise_graph_payload(raw)
        self.assertEqual(graph["code"], 0)
        self.assertEqual(graph["data"]["papers"][0]["graph"]["nodes"][0]["id"], "n1")

    def test_items_envelope_is_converted(self) -> None:
        raw = {"ok": True, "data": {"items": [{"graph": {"nodes": []}}]}}
        graph = retrieval._normalise_graph_payload(raw)
        self.assertEqual(graph["data"]["papers"], [{"graph": {"nodes": []}}])

    def test_papers_envelope_is_preserved(self) -> None:
        raw = {"code": 0, "data": {"papers": [{"graph": {"nodes": []}}]}}
        graph = retrieval._normalise_graph_payload(raw)
        self.assertEqual(graph, raw)


class OfflineMaterialiseTest(unittest.TestCase):
    def test_spec_without_lkm_id_gets_stable_local_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            graph_path = root / "graph.json"
            text_path = root / "paper.md"
            graph_path.write_text(
                json.dumps({
                    "data": {"papers": [{
                        "paper": {"en_title": "Local paper"},
                        "graph": {"nodes": [{"kind": "conclusion", "content": "C"}],
                                  "edges": []},
                    }]}
                }),
                encoding="utf-8",
            )
            text_path.write_text("# Local paper\n", encoding="utf-8")
            spec = root / "papers.json"
            spec.write_text(json.dumps({"papers": [{
                "title": "Local paper",
                "graph_path": str(graph_path),
                "ocr_path": str(text_path),
            }]}), encoding="utf-8")
            runtime = _runtime(root)
            run_dir = root / "run"
            papers = retrieval.retrieve(
                "q", run_dir, paper_spec=spec, runtime=runtime, fulltext=False
            )
            self.assertEqual(len(papers), 1)
            self.assertTrue(papers[0]["lkm_id"].startswith("local_"))
            self.assertEqual(papers[0]["title"], "Local paper")
            self.assertTrue((run_dir / papers[0]["graph_path"]).is_file())
            self.assertTrue((run_dir / papers[0]["ocr_path"]).is_file())


class QueryPlanningTest(unittest.TestCase):
    def test_deterministic_fallback_queries(self) -> None:
        queries, planner = retrieval.plan_queries(
            "如何利用知识图谱辅助实验设计？",
            question_en="How can knowledge graphs support experiment design?",
            domain="knowledge graphs, experiment design",
            max_queries=4,
            use_query_planner=False,
        )
        self.assertEqual(planner, "fallback")
        self.assertLessEqual(len(queries), 4)
        sources = [item["source"] for item in queries]
        self.assertEqual(sources[0], "english")
        self.assertIn("question", sources)
        keyword = next(item["query"] for item in queries if item["source"] == "keywords")
        self.assertIn("知识图谱", keyword)
        self.assertIn("实验设计", keyword)

    def test_llm_queries_are_added_and_limited(self) -> None:
        def fake_call(prompt):
            return {"queries": [
                "knowledge graph experiment design",
                "scientific decision support knowledge graph",
                "automated experiment planning",
            ]}

        queries, planner = retrieval.plan_queries(
            "如何利用知识图谱辅助实验设计？",
            question_en="How can knowledge graphs support experiment design?",
            max_queries=4,
            call_json=fake_call,
        )
        self.assertEqual(planner, "llm")
        self.assertLessEqual(len(queries), 4)
        self.assertEqual(
            [item["query"] for item in queries if item["source"] == "llm"],
            ["knowledge graph experiment design", "scientific decision support knowledge graph"],
        )

    def test_llm_failure_is_non_fatal(self) -> None:
        def broken(prompt):
            raise RuntimeError("planner offline")

        queries, planner = retrieval.plan_queries(
            "知识图谱 实验设计", use_query_planner=True, call_json=broken
        )
        self.assertEqual(planner, "fallback")
        self.assertTrue(queries)
        self.assertTrue(all(item["source"] != "llm" for item in queries))


class CandidateRankingTest(unittest.TestCase):
    def test_multi_query_hit_outranks_single_query_rich_graph(self) -> None:
        two_hits = {"query_positions": [0, 1], "search_score": 0.5,
                    "node_count": 10, "conclusion_count": 2}
        one_hit = {"query_positions": [0], "search_score": 0.9,
                   "node_count": 30, "conclusion_count": 5}
        self.assertGreater(retrieval._candidate_rank(two_hits),
                           retrieval._candidate_rank(one_hit))

    def test_search_many_merges_and_ranks(self) -> None:
        def fake_search(query, index, top_k, run_dir, runtime, **kwargs):
            if index == 0:
                return ([{"lkm_id": "1", "title": "A", "search_score": 0.2},
                         {"lkm_id": "2", "title": "B", "search_score": 0.1}], "")
            return ([{"lkm_id": "1", "title": "A", "search_score": 0.8},
                     {"lkm_id": "3", "title": "C", "search_score": 0.2}], "")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            with mock.patch("ariadne.retrieval._search_one", side_effect=fake_search):
                candidates, audits = retrieval.search_many(
                    [{"query": "q1", "source": "english"},
                     {"query": "q2", "source": "llm"}],
                    5, root, runtime,
                )
        self.assertEqual(len(audits), 2)
        self.assertEqual(candidates[0]["record"]["lkm_id"], "1")
        self.assertEqual(candidates[0]["query_positions"], [0, 1])
        self.assertAlmostEqual(candidates[0]["search_score"], 0.8)

    def test_search_scores_come_from_variables(self) -> None:
        payload = {
            "ok": True,
            "data": {
                "papers": {"paper:1": {"en_title": "A"}},
                "variables": [{"paper_id": "1", "rerank_score": 0.7}],
            },
        }
        records = retrieval._records_from_search(payload)
        self.assertAlmostEqual(records[0]["search_score"], 0.7)


def _graph(nodes: int, conclusions: int = 1) -> dict:
    graph_nodes = [
        {"kind": "conclusion", "content": f"conclusion {index}"}
        for index in range(conclusions)
    ]
    graph_nodes += [
        {"kind": "highlight", "content": f"highlight {index}"}
        for index in range(max(0, nodes - conclusions))
    ]
    return {"code": 0, "data": {"papers": [{
        "graph": {"nodes": graph_nodes, "edges": []},
        "paper": {"en_title": "Paper"},
    }]}}


def _candidate(pid: str) -> dict:
    return {"record": {"lkm_id": pid, "title": pid}, "query_positions": [0],
            "search_score": 0.5}


class RelevanceTest(unittest.TestCase):
    def test_topic_phrases_beat_generic_hits(self) -> None:
        queries = [{"query": "knowledge graph experiment design decision support"}]
        relevant = retrieval._relevance(
            {"title": "Automating discovery through Knowledge Graphs and LLM Agents",
             "abstract": "scientific experiment design and decision support"}, queries)
        generic = retrieval._relevance(
            {"title": "Towards Continuous Experiment-driven MLOps",
             "abstract": "deployment pipelines and monitoring"}, queries)
        self.assertGreater(relevant, generic)
        self.assertGreaterEqual(relevant, 0.2)


class ProbeSelectionTest(unittest.TestCase):
    def test_skips_empty_and_thin_graphs(self) -> None:
        graphs = {
            "thin": _graph(2, 0),
            "rich1": _graph(6, 2),
            "rich2": _graph(8, 3),
        }

        def fake_fetch(record, dest, run_dir, runtime):
            pid = record["lkm_id"]
            if pid == "empty":
                raise retrieval.GraphUnavailable("empty", reason="empty_graph")
            graph = graphs[pid]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(graph), encoding="utf-8")
            return dest, graph

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            with mock.patch("ariadne.retrieval._fetch_graph", side_effect=fake_fetch):
                selected, audit = retrieval.probe_and_select(
                    [_candidate("empty"), _candidate("thin"),
                     _candidate("rich1"), _candidate("rich2")],
                    root, runtime, top_k=2, candidate_limit=4,
                    min_papers=2, min_graph_nodes=3,
                )
            self.assertEqual({item["record"]["lkm_id"] for item in selected},
                             {"rich1", "rich2"})
            reasons = {item["lkm_id"]: item["skip_reason"]
                       for item in audit["candidates"]}
            self.assertEqual(reasons["empty"], "empty_graph")
            self.assertEqual(reasons["thin"], "thin_graph")
            self.assertTrue((root / "01_retrieval" / "selection.json").is_file())

    def test_raises_when_too_few_usable_papers(self) -> None:
        def fake_fetch(record, dest, run_dir, runtime):
            graph = _graph(5, 1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(graph), encoding="utf-8")
            return dest, graph

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            with mock.patch("ariadne.retrieval._fetch_graph", side_effect=fake_fetch):
                with self.assertRaises(AriadneError):
                    retrieval.probe_and_select(
                        [_candidate("only")], root, runtime, top_k=2,
                        candidate_limit=2, min_papers=2, min_graph_nodes=3,
                    )

    def test_extends_probe_when_first_pass_has_too_few_graphs(self) -> None:
        def fake_fetch(record, dest, run_dir, runtime):
            pid = record["lkm_id"]
            if pid in {"e1", "e2", "e3", "e4"}:
                raise retrieval.GraphUnavailable("empty", reason="empty_graph")
            graph = _graph(5, 1)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(graph), encoding="utf-8")
            return dest, graph

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            with mock.patch("ariadne.retrieval._fetch_graph", side_effect=fake_fetch):
                selected, audit = retrieval.probe_and_select(
                    [_candidate(pid) for pid in ("e1", "e2", "e3", "e4", "g1", "g2")],
                    root, runtime, top_k=2, candidate_limit=2,
                    min_papers=2, min_graph_nodes=3,
                )
        self.assertEqual(len(selected), 2)
        self.assertEqual(audit["probed"], 6)
        self.assertGreater(audit["max_probe"], audit["first_pass"])

    def test_materialise_selected_writes_text_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = _runtime(root)
            probe = root / "01_retrieval" / "probe" / "1.json"
            probe.parent.mkdir(parents=True)
            probe.write_text(json.dumps(_graph(5, 2)), encoding="utf-8")
            selected = [{
                "record": {"lkm_id": "1", "title": "Paper", "abstract": ""},
                "probe_path": "01_retrieval/probe/1.json",
                "query_positions": [0], "rank_score": 4.2,
                "node_count": 5, "conclusion_count": 2,
            }]
            papers = retrieval.materialise_selected(
                selected, root, runtime, fulltext=False
            )
            self.assertEqual(papers[0]["lkm_id"], "1")
            self.assertTrue((root / papers[0]["graph_path"]).is_file())
            text = (root / papers[0]["ocr_path"]).read_text(encoding="utf-8")
            self.assertIn("conclusion 0", text)


class BohrCommandTest(unittest.TestCase):
    def test_command_without_profile_or_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _runtime(
                Path(tmp), bohr_profile="", auto_approve_spend=False,
                payment_source="",
            )
            from ariadne.config import bohr_command

            command = bohr_command(runtime, "lkm", "search", "q")
            self.assertEqual(command[0], "/usr/local/bin/bohr")
            self.assertNotIn("--profile", command)
            self.assertNotIn("--yes", command)
            self.assertNotIn("--payment-source", command)


if __name__ == "__main__":
    unittest.main()
