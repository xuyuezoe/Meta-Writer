from __future__ import annotations

import unittest

from src.orchestrator_v2 import SelfCorrectingOrchestrator


class OrchestratorOutputTests(unittest.TestCase):
    def test_assemble_text_does_not_inject_missing_content_placeholder(self) -> None:
        text = SelfCorrectingOrchestrator._assemble_text(
            object(),
            {"sec1": "Intro", "sec2": "Body"},
            {"sec1": "First paragraph."},
        )

        self.assertIn("## Body", text)
        self.assertNotIn("[sec2", text)
        self.assertNotIn("content missing", text)

    def test_coerce_degraded_section_content_returns_blank_for_none(self) -> None:
        normalized = SelfCorrectingOrchestrator._coerce_degraded_section_content(
            object(),
            None,
        )

        self.assertEqual(normalized, "")


if __name__ == "__main__":
    unittest.main()
