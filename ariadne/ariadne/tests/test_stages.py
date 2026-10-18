"""The public checkout must contain every file Ariadne's full mode needs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ariadne import paper2ir
from ariadne.full import REQUIRED_FILES
from ariadne.util import REPO


class PublicDependenciesTest(unittest.TestCase):
    def test_required_files_exist(self) -> None:
        missing = sorted(rel for rel in REQUIRED_FILES if not (REPO / rel).is_file())
        self.assertEqual(missing, [])

    def test_clean_claims_copy_keeps_code_and_drops_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = paper2ir._copy_pipeline(Path(tmp) / "pipeline")
            self.assertTrue((destination / "run_full_pipeline.py").is_file())
            self.assertTrue((destination / "step1_process_conclusions").is_dir())
            self.assertFalse((destination / "data").exists())
            self.assertFalse((destination / "test_run_full_pipeline.py").exists())
            self.assertFalse((destination / "__pycache__").exists())


if __name__ == "__main__":
    unittest.main()
