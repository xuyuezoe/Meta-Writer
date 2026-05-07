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

    def test_assemble_text_collapses_internal_paragraph_breaks(self) -> None:
        """节内的 \\n\\n 必须被折叠为单个换行，保证每节产生恰好 1 个内容块。"""
        import sys
        sys.path.insert(0, "metabench/src")
        from metabench.local_metrics import split_blocks

        text = SelfCorrectingOrchestrator._assemble_text(
            object(),
            {"sec1": "Intro", "sec2": "Body"},
            {
                "sec1": "Para one.\n\nPara two.\n\nPara three.",
                "sec2": "Single block here.",
            },
        )

        blocks = split_blocks(text, drop_markdown_wrappers=True)
        self.assertEqual(len(blocks), 2)
        self.assertIn("Para one.", blocks[0])
        self.assertIn("Para two.", blocks[0])
        self.assertIn("Para three.", blocks[0])
        self.assertIn("Single block here.", blocks[1])


class CitationFailureGuardTests(unittest.TestCase):
    """验证引用密度失败不会触发 partial_rollback（Fix 2）。"""

    def _make_citation_only_report(self, valid_markers: int = 0):
        """构造仅含引用密度失败（type=reference, severity=major）的验证报告。"""
        from src.core.validation import Issue, ValidationReport
        report = ValidationReport(
            passed=False,
            issues=[
                Issue(
                    type="reference",
                    severity="major",
                    description=f"引用密度不足：本节仅有 {valid_markers} 个合法引用标记，要求至少 1 个。",
                    location="sec5",
                )
            ],
            dcas_score=0.85,
        )
        # 构造一个通过的 ReferenceReport
        from src.references.types import SectionReferenceReport
        ref_rpt = SectionReferenceReport(
            passed=False,
            valid_marker_count=valid_markers,
            invalid_marker_count=0,
            invalid_r_indices=set(),
            issues=report.issues,
        )
        report.reference_report = ref_rpt
        return report

    def _make_mixed_report(self):
        """构造包含引用失败 + 内容一致性失败的混合报告。"""
        from src.core.validation import Issue, ValidationReport
        return ValidationReport(
            passed=False,
            issues=[
                Issue(type="reference", severity="major",
                      description="引用密度不足：0个", location="sec4"),
                Issue(type="alignment", severity="major",
                      description="内容覆盖度不足", location="sec4"),
            ],
            dcas_score=0.4,
        )

    def test_citation_only_failure_is_detected(self) -> None:
        """纯引用失败（无其他 major issues）应被识别为 citation_only_failure。"""
        report = self._make_citation_only_report(valid_markers=0)
        non_ref_major = [
            i for i in report.issues
            if i.type != "reference" and i.severity in ("critical", "major")
        ]
        self.assertEqual(len(non_ref_major), 0)

    def test_mixed_failure_is_not_citation_only(self) -> None:
        """引用失败 + 内容失败的混合报告不应被识别为 citation_only_failure。"""
        report = self._make_mixed_report()
        non_ref_major = [
            i for i in report.issues
            if i.type != "reference" and i.severity in ("critical", "major")
        ]
        self.assertGreater(len(non_ref_major), 0)


class ReferenceValidatorMinCitationsTests(unittest.TestCase):
    """验证 Fix 1：min_citations=1 已应用。"""

    def test_reference_validator_requires_only_one_citation(self) -> None:
        from src.validators.reference_validator import ReferenceValidator
        validator = ReferenceValidator(min_citations=1)
        # 有1个合法引用时应通过
        report = validator.validate(
            content="This is text with [R1] marker.",
            valid_r_set={1, 2, 3},
            section_id="sec5",
        )
        self.assertTrue(report.passed)

    def test_reference_validator_fails_with_zero_citations(self) -> None:
        from src.validators.reference_validator import ReferenceValidator
        validator = ReferenceValidator(min_citations=1)
        # 零引用时应失败
        report = validator.validate(
            content="This section has no citation markers at all.",
            valid_r_set={1, 2, 3},
            section_id="sec5",
        )
        self.assertFalse(report.passed)

    def test_online_validator_uses_min_citations_one(self) -> None:
        """OnlineValidator 实例化时必须使用 min_citations=1。"""
        from src.validators.online_validator import OnlineValidator
        import inspect
        source = inspect.getsource(OnlineValidator.__init__)
        self.assertIn("min_citations=1", source,
                      "OnlineValidator must set min_citations=1 on ReferenceValidator")


if __name__ == "__main__":
    unittest.main()
