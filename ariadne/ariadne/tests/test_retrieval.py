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
