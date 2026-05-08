from __future__ import annotations

from pathlib import Path
import re
import unittest

from examples.benchmark_template import (
    DOCUMENT_LEVEL_CONSTRAINT_PREFIX,
    _build_outline,
    _extract_paragraph_blocks,
    build_benchmark_task_config,
    evaluate_output,
    load_benchmark_task,
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

    def test_benchmark_source_assets_are_english(self) -> None:
        samples_path = Path("metabench/examples/samples.jsonl")
        samples_text = samples_path.read_text(encoding="utf-8")

        self.assertNotRegex(samples_text, r"[\u4e00-\u9fff]")

    def test_extract_paragraph_blocks_skips_markdown_wrappers(self) -> None:
        text = (
            "## Section One\n\n"
            "First paragraph.\n\n"
            "---\n\n"
            "## Section Two\n\n"
            "Second paragraph.\n\n"
            "Third paragraph."
        )

        self.assertEqual(
            _extract_paragraph_blocks(text),
            ["First paragraph.", "Second paragraph.", "Third paragraph."],
        )

    def test_benchmark_reference_keeps_once_keywords(self) -> None:
        config = build_benchmark_task_config("med_s001")
        reference = config["reference"]

        self.assertIsInstance(reference, dict)
        self.assertIn("once_keywords", reference["constraints"])
        self.assertTrue(reference["constraints"]["once_keywords"])

    def test_evaluate_output_penalizes_large_length_shortfall(self) -> None:
        config = build_benchmark_task_config("med_s001")
        reference = config["reference"]
        constraints = reference["constraints"]
        must_include = constraints["must_include"]
        expected_blocks = constraints["expected_blocks"]

        paragraph = (
            "This review covers "
            + ", ".join(must_include)
            + ". "
            + "analysis " * 40
        ).strip()
        generated_text = "\n\n".join(paragraph for _ in range(expected_blocks))

        evaluation = evaluate_output(generated_text, reference)
        diagnostics = evaluation["diagnostics"]

        self.assertEqual(evaluation["entity_consistency_score"], 1.0)
        self.assertLess(evaluation["length_score"], 0.5)
        self.assertGreater(evaluation["constraint_violation_rate"], 0.5)
        self.assertFalse(diagnostics["length_within_tolerance"])
        self.assertLess(
            diagnostics["response_word_count"],
            diagnostics["required_length_words"],
        )


class OutlineStructureTests(unittest.TestCase):
    """验证 _build_outline 生成 7 节大纲并与 range_keyword 位置对齐。"""

    def _analysis_outline(self) -> dict:
        return _build_outline(
            "analysis",
            ["scope", "classification framework", "hemodynamics", "limitations"],
        )

    def test_outline_has_seven_sections(self) -> None:
        outline = self._analysis_outline()
        self.assertEqual(len(outline), 7)
        for i in range(1, 8):
            self.assertIn(f"sec{i}", outline)

    def test_sec1_contains_scope_keyword(self) -> None:
        outline = self._analysis_outline()
        self.assertIn("scope", outline["sec1"].lower())

    def test_sec2_contains_classification_keyword(self) -> None:
        outline = self._analysis_outline()
        self.assertIn("classification framework", outline["sec2"].lower())

    def test_sec_penultimate_contains_limitations_keyword(self) -> None:
        outline = self._analysis_outline()
        N = len(outline)
        penultimate = f"sec{N - 1}"
        self.assertIn("limitation", outline[penultimate].lower())

    def test_sec7_contains_future_work_keyword(self) -> None:
        outline = self._analysis_outline()
        self.assertIn("future work", outline["sec7"].lower())

    def test_writing_task_also_produces_seven_sections(self) -> None:
        outline = _build_outline("writing", ["classification framework", "future work"])
        self.assertEqual(len(outline), 7)


class VerbatimConstraintTests(unittest.TestCase):
    """验证 load_benchmark_task 对 must_include 注入逐字精确约束。"""

    def setUp(self) -> None:
        self.task = load_benchmark_task("med_s001")
        self.constraints: list = self.task["constraints"]  # type: ignore[assignment]

    def _constraint_text(self) -> str:
        return "\n".join(self.constraints)

    def test_verbatim_hemodynamics_constraint_present(self) -> None:
        # 确保精确词形 "hemodynamics" 被注入，而非仅 "hemodynamic"
        self.assertIn("'hemodynamics'", self._constraint_text())

    def test_verbatim_scope_constraint_present(self) -> None:
        self.assertIn("'scope'", self._constraint_text())

    def test_verbatim_limitations_constraint_present(self) -> None:
        self.assertIn("'limitations'", self._constraint_text())

    def test_single_paragraph_constraint_present(self) -> None:
        text = self._constraint_text()
        self.assertIn("single continuous prose paragraph", text)
        self.assertIn("exactly 7 major paragraphs", text)

    def test_comparison_periodic_verbatim_constraint_present(self) -> None:
        text = self._constraint_text()
        self.assertIn("'comparison'", text)
        self.assertIn("every 2 paragraphs", text)


class PostProcessingRangeKeywordTests(unittest.TestCase):
    """验证评估器在 7 块文章上对所有 range_keyword 均通过。"""

    def _make_7block_article(self) -> str:
        """
        构造一篇包含全部必须关键词且每节为单段的 7 块文章，
        模拟 orchestrator_v2 post-processing 后的输出。
        """
        sections = [
            ("Scope, terminology, and practice boundaries",
             "This review defines the scope of acute coronary syndrome (ACS) within "
             "cardiovascular medicine, focusing on adult inpatient management across STEMI, "
             "NSTEMI, and unstable angina subtypes. The mechanism driving each subtype is "
             "distinct, and hemodynamics deteriorate at different rates. Limitations of "
             "outpatient extrapolation are noted. The classification framework used here "
             "stratifies by ECG and troponin criteria."),
            ("Classification framework — organize and compare",
             "The classification framework organizes ACS into three subtypes. Detailed "
             "comparison of STEMI, NSTEMI, and UA reveals distinct hemodynamics and urgency. "
             "The mechanism of complete versus partial coronary occlusion determines subtype. "
             "A further comparison of risk scores such as GRACE and TIMI shows how adult "
             "inpatient pathways diverge. This systematic comparison structures subsequent "
             "evidence synthesis."),
            ("Mechanisms and pathophysiology",
             "The mechanism of atherosclerotic plaque rupture exposes thrombogenic material, "
             "triggering acute coronary occlusion. A comparison of plaque erosion versus "
             "rupture reveals different inflammatory signatures and hemodynamics profiles. "
             "Smooth muscle cell phenotypic modulation drives plaque vulnerability. This "
             "comparison between subtypes underlies risk stratification in cardiovascular "
             "medicine for adult inpatient populations."),
            ("Hemodynamics and clinical evidence",
             "Hemodynamics in ACS range from preserved function to cardiogenic shock. "
             "Comparison of invasive versus conservative strategies guides adult inpatient "
             "triage. Invasive hemodynamics monitoring reveals real-time cardiac output and "
             "filling pressures. Evidence integration via a comparison of multiple RCTs shows "
             "early revascularization improves outcomes in adult inpatient care."),
            ("Limitations of current evidence",
             "Limitations of current evidence include underrepresentation of elderly adult "
             "inpatient populations in landmark trials. Methodological limitations include "
             "variable troponin assay thresholds. Measurement limitations arise from "
             "non-standardized hemodynamics monitoring protocols. These limitations impede "
             "direct comparison across trial cohorts and guideline development."),
            ("Evidence gaps and ongoing controversies",
             "Evidence gaps persist around optimal antiplatelet duration in complex "
             "hemodynamics scenarios. Ongoing controversies surround complete versus "
             "culprit-only revascularization. Comparison of conflicting guidelines reveals "
             "inconsistent recommendations for adult inpatient care. Future work must address "
             "these comparison-relevant evidence deficits within cardiovascular medicine."),
            ("Future work and research agenda",
             "Future work should prioritize individualized hemodynamics-based risk models "
             "for adult inpatient populations. Research agenda items include translational "
             "studies on mechanism-specific cardioprotection and scope-broadening to diverse "
             "ethnic cohorts. Limitations identified above define future investigation "
             "priorities. Comparison of emerging biomarkers will refine the classification "
             "framework for acute coronary syndrome stratification."),
        ]
        parts = [f"## {title}\n\n{content}" for title, content in sections]
        return "\n\n---\n\n".join(parts)

    def setUp(self) -> None:
        self.task = load_benchmark_task("med_s001")
        self.reference = self.task["reference"]
        self.article = self._make_7block_article()
        # 模拟 orchestrator post-processing（单节单块折叠）
        import re
        sections_raw = re.split(r"\n\n---\n\n", self.article)
        collapsed = []
        for sec in sections_raw:
            head_end = sec.find("\n\n")
            if head_end == -1:
                collapsed.append(sec)
                continue
            heading = sec[:head_end]
            body = re.sub(r"\n\s*\n", "\n", sec[head_end:]).strip()
            collapsed.append(f"{heading}\n\n{body}")
        self.article = "\n\n---\n\n".join(collapsed)

    def test_exactly_7_blocks(self) -> None:
        blocks = _extract_paragraph_blocks(self.article)
        self.assertEqual(len(blocks), 7)

    def test_all_range_keywords_hit(self) -> None:
        result = evaluate_output(self.article, self.reference)
        diag = result["diagnostics"]
        self.assertEqual(diag["missing_range_keywords"], [],
                         f"Missing range keywords: {diag['missing_range_keywords']}")

    def test_all_once_keywords_matched(self) -> None:
        result = evaluate_output(self.article, self.reference)
        diag = result["diagnostics"]
        self.assertEqual(diag["missing_keywords"], [],
                         f"Missing keywords: {diag['missing_keywords']}")

    def test_periodic_comparison_hit(self) -> None:
        result = evaluate_output(self.article, self.reference)
        diag = result["diagnostics"]
        self.assertIn("comparison", diag["periodic_keyword_hits"],
                      f"Periodic hits: {diag['periodic_keyword_hits']}")

    def test_structure_signal_is_perfect(self) -> None:
        result = evaluate_output(self.article, self.reference)
        diag = result["diagnostics"]
        self.assertAlmostEqual(diag["structure_signal"], 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
