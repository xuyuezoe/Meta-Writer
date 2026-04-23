from __future__ import annotations

import re
import unittest

from examples.benchmark_template import (
    DOCUMENT_LEVEL_CONSTRAINT_PREFIX,
    _extract_paragraph_blocks,
    build_benchmark_task_config,
)


class BenchmarkTemplateTests(unittest.TestCase):
    def test_benchmark_constraints_are_marked_as_document_level(self) -> None:
        config = build_benchmark_task_config("med_s001")
        constraints = config["constraints"]

        self.assertIsInstance(constraints, list)
        self.assertTrue(constraints)
        self.assertTrue(
            all(
                isinstance(item, str)
                and item.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX)
                for item in constraints
            )
        )
        self.assertIn("language", config)
        self.assertEqual(config["language"], "en")

    def test_benchmark_task_is_english(self) -> None:
        config = build_benchmark_task_config("med_s001")
        task = config["task"]
        reference = config["reference"]

        self.assertIsInstance(task, str)
        self.assertNotRegex(task, r"[\u4e00-\u9fff]")
        self.assertIn("English", task)
        self.assertIsInstance(reference, dict)
        self.assertEqual(reference["language"], "en")
        self.assertTrue(
            all(
                not re.search(r"[\u4e00-\u9fff]", item)
                for item in reference["constraints"]["must_include"]
            )
        )

    def test_extract_paragraph_blocks_skips_markdown_wrappers(self) -> None:
        text = (
            "## 第一节\n\n"
            "第一段正文。\n\n"
            "---\n\n"
            "## 第二节\n\n"
            "第二段正文。\n\n"
            "第三段正文。"
        )

        self.assertEqual(
            _extract_paragraph_blocks(text),
            ["第一段正文。", "第二段正文。", "第三段正文。"],
        )


if __name__ == "__main__":
    unittest.main()
