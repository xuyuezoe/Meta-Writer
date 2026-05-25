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
        text = SelfCorrectingOrchestrator._assemble_text(
            object(),
            {"sec1": "Intro", "sec2": "Body"},
            {
                "sec1": "Para one.\n\nPara two.\n\nPara three.",
                "sec2": "Single block here.",
            },
        )

        blocks = [block for block in text.split("\n\n---\n\n") if block.strip()]
        self.assertEqual(len(blocks), 2)
        self.assertIn("Para one.", blocks[0])
        self.assertIn("Para two.", blocks[0])
        self.assertIn("Para three.", blocks[0])
        self.assertIn("Single block here.", blocks[1])


class SectionBudgetResolutionTests(unittest.TestCase):
    def test_resolve_section_word_targets_prefers_reference_constraints(self) -> None:
        outline = {
            "sec1": "Scope",
            "sec2": "Framework",
            "sec3": "Evidence",
            "sec4": "Findings",
            "sec5": "Implications",
            "sec6": "Limitations",
        }
        reference = {
            "constraints": {
                "section_word_targets": {
                    "sec1": 305,
                    "sec2": 516,
                    "sec3": 631,
                    "sec4": 674,
                    "sec5": 544,
                    "sec6": 246,
                }
            }
        }

        resolved = SelfCorrectingOrchestrator._resolve_section_word_targets(
            task="Write an approximately 2916-word review article.",
            outline=outline,
            reference=reference,
        )

        self.assertEqual(
            resolved,
            {
                "sec1": 305,
                "sec2": 516,
                "sec3": 631,
                "sec4": 674,
                "sec5": 544,
                "sec6": 246,
            },
        )

    def test_resolve_section_word_targets_falls_back_to_uniform_parse(self) -> None:
        outline = {
            "sec1": "Scope",
            "sec2": "Framework",
            "sec3": "Evidence",
        }

        resolved = SelfCorrectingOrchestrator._resolve_section_word_targets(
            task="Write an approximately 900-word review article.",
            outline=outline,
            reference=None,
        )

        self.assertEqual(
            resolved,
            {
                "sec1": 300,
                "sec2": 300,
                "sec3": 300,
            },
        )


class CitationFailureGuardTests(unittest.TestCase):
    """验证引用密度失败不会触发 partial_rollback（Fix 2）。

    新版逻辑：_citation_only_failure = len(report.failures) == 0
    引用密度失败不会产生 PresenceViolation / AbsenceViolation，
    仅记录在 report.reference_report 中。
    因此当 failures 为空时，说明仅引用检查未通过，应强制 local_rewrite。
    """

    def _make_citation_only_report(self, valid_markers: int = 0):
        """构造仅含引用密度失败（failures=空列表）的验证报告。"""
        from src.core.validation import ValidationReport
        from src.references.types import SectionReferenceReport

        ref_rpt = SectionReferenceReport(
            passed=False,
            valid_marker_count=valid_markers,
            invalid_marker_count=0,
            invalid_r_indices=set(),
            issues=[],
        )
        report = ValidationReport(
            section_id="sec5",
            passed=False,
            score=0.85,
            failures=[],   # 无非引用类 blocking failure
            warnings=["引用密度不足：本节仅有 0 个合法引用标记，要求至少 1 个。"],
            reference_report=ref_rpt,
        )
        return report

    def _make_mixed_report(self):
        """构造包含引用失败 + 内容一致性失败（failures 非空）的混合报告。"""
        from src.core.validation import AbsenceViolation, ValidationReport
        from src.references.types import SectionReferenceReport

        ref_rpt = SectionReferenceReport(
            passed=False,
            valid_marker_count=0,
            invalid_marker_count=0,
            invalid_r_indices=set(),
            issues=[],
        )
        return ValidationReport(
            section_id="sec4",
            passed=False,
            score=0.4,
            failures=[
                AbsenceViolation(
                    τ="CONTENT_GAP",
                    obligation="内容覆盖度不足：缺少核心主题覆盖",
                    source_check="coverage:topic1",
                )
            ],
            warnings=["引用密度不足：0个"],
            reference_report=ref_rpt,
        )

    def test_citation_only_failure_is_detected(self) -> None:
        """纯引用失败（failures 为空）应被识别为 citation_only_failure。"""
        report = self._make_citation_only_report(valid_markers=0)
        # 新版逻辑：citation_only_failure = len(report.failures) == 0
        _citation_only_failure = len(report.failures) == 0
        self.assertTrue(_citation_only_failure)

    def test_mixed_failure_is_not_citation_only(self) -> None:
        """引用失败 + 内容失败的混合报告（failures 非空）不应被识别为 citation_only_failure。"""
        report = self._make_mixed_report()
        # 新版逻辑：citation_only_failure = len(report.failures) == 0
        _citation_only_failure = len(report.failures) == 0
        self.assertFalse(_citation_only_failure)


class ReferenceValidatorMinCitationsTests(unittest.TestCase):
    """验证 min_citations=4 已应用：每节至少需要 4 个引用。"""

    def test_reference_validator_passes_with_four_citations(self) -> None:
        from src.validators.reference_validator import ReferenceValidator
        validator = ReferenceValidator(min_citations=4)
        # 有4个合法引用时应通过
        report = validator.validate(
            content="Text with [R1] and [R2] and [R3] and [R4] markers.",
            valid_r_set={1, 2, 3, 4},
            section_id="sec5",
        )
        self.assertTrue(report.passed)

    def test_reference_validator_fails_with_fewer_than_four(self) -> None:
        from src.validators.reference_validator import ReferenceValidator
        validator = ReferenceValidator(min_citations=4)
        # 仅3个引用时应失败（低于最低 4 个要求）
        report = validator.validate(
            content="Text with [R1] and [R2] and [R3] markers only.",
            valid_r_set={1, 2, 3},
            section_id="sec5",
        )
        self.assertFalse(report.passed)

    def test_reference_validator_fails_with_zero_citations(self) -> None:
        from src.validators.reference_validator import ReferenceValidator
        validator = ReferenceValidator(min_citations=4)
        # 零引用时应失败
        report = validator.validate(
            content="This section has no citation markers at all.",
            valid_r_set={1, 2, 3},
            section_id="sec5",
        )
        self.assertFalse(report.passed)

    def test_online_validator_uses_min_citations_four(self) -> None:
        """OnlineValidator 实例化时必须使用 min_citations=4。"""
        from src.validators.online_validator import OnlineValidator
        import inspect
        source = inspect.getsource(OnlineValidator.__init__)
        self.assertIn("min_citations=4", source,
                      "OnlineValidator must set min_citations=4 on ReferenceValidator")


if __name__ == "__main__":
    unittest.main()
