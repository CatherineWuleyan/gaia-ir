"""Regression tests for the question-led merge pipeline."""
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


if __name__ == "__main__":
    unittest.main()
