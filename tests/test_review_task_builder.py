import tempfile
import textwrap
import unittest
from pathlib import Path

from meta_bench.experiments.review_task_builder import derive_review_task_from_article
from meta_bench.six_slot_priors import SIX_SLOT_ORDER, classify_outline_to_six_slots


class ReviewTaskBuilderTests(unittest.TestCase):
    def _write_article(self, raw_text: str) -> Path:
        handle = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
        handle.write(textwrap.dedent(raw_text).strip() + "\n")
        handle.close()
        return Path(handle.name)

    def test_real_review_builder_aligns_sparse_raw_outline_to_six_slots(self):
        article_path = self._write_article(
            """
            ---
            paper_id: PMCFAKE0001
            title: Dupilumab outcomes in refractory bullous pemphigoid
            abstract: Review article abstract.
            stats:
              word_count: 1200
            ---

            # Dupilumab outcomes in refractory bullous pemphigoid

            ## Introduction
            This review defines the disease context, scope, and terminology for bullous pemphigoid.

            A second introductory paragraph extends the framing and clinical context before the evidence synthesis.

            ## Methods
            We summarize the search strategy, eligibility criteria, and measurement choices used across the reviewed studies.

            The methods discussion also covers evidence selection and assay considerations across cohorts.

            ## Discussion
            The discussion synthesizes response rates, adverse events, subgroup patterns, comparative findings, and interpretation across studies.

            A second discussion paragraph expands the findings, compares heterogeneity, and connects the evidence to treatment decisions.

            A third discussion paragraph addresses practice implications, uncertainty, conflicting signals, and therapeutic interpretation.

            ## Conclusion
            Taken together, the evidence remains limited and future multicenter studies are needed to resolve the main evidence gaps.

            ## References
            1. Example reference.
            """
        )
        self.addCleanup(article_path.unlink)

        derived = derive_review_task_from_article(article_path=article_path)

        self.assertEqual(derived.spec.expected_sections, 6)
        self.assertEqual(list(derived.outline.keys()), [f"sec{i}" for i in range(1, 7)])
        self.assertEqual(sum(derived.section_word_targets.values()), derived.spec.body_target_words)
        self.assertTrue(all(derived.section_word_targets[f"sec{i}"] > 0 for i in range(1, 7)))

        slot_mapping = classify_outline_to_six_slots(list(derived.outline.items()))
        self.assertEqual(
            [slot_mapping[f"sec{i}"] for i in range(1, 7)],
            list(SIX_SLOT_ORDER),
        )

        reference_constraints = derived.config["reference"]["constraints"]
        self.assertEqual(
            reference_constraints["section_prior_scheme"],
            "real_article_six_slot_aligned",
        )
        self.assertTrue(reference_constraints["six_slot_prior_version"])
        trace = reference_constraints["section_budget_trace"]
        self.assertEqual(trace["source"], "real_article_six_slot_alignment")
        self.assertEqual(trace["raw_article_section_count"], 4)
        self.assertEqual(trace["scaled_to_body_target_words"], derived.spec.body_target_words)

    def test_real_review_builder_filters_correction_note_as_non_body(self):
        article_path = self._write_article(
            """
            ---
            paper_id: PMCFAKE0002
            title: Diagnostic accuracy review with correction note
            abstract: Review article abstract.
            stats:
              word_count: 900
            ---

            # Diagnostic accuracy review with correction note

            ## Introduction
            This review defines the scope and background.

            ## Conclusions
            The main conclusions are summarized here.

            ## Correction note
            This section only corrects metadata and should not be treated as body content.

            ## References
            1. Example reference.
            """
        )
        self.addCleanup(article_path.unlink)

        with self.assertRaisesRegex(ValueError, "at least 3 content sections"):
            derive_review_task_from_article(article_path=article_path)


if __name__ == "__main__":
    unittest.main()
