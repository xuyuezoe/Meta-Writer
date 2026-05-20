import json
import tempfile
import unittest
from pathlib import Path

from meta_bench.six_slot_priors import (
    SIX_SLOT_ORDER,
    derive_six_slot_priors_from_corpus,
)


class SixSlotPriorsTests(unittest.TestCase):
    def test_derived_priors_do_not_evenly_split_middle_slots(self):
        article_one = """# Review A

## Introduction

Scope paragraph with background context and terminology.

Framework paragraph with mechanism framing and pathway context.

## Methods

Methods paragraph with assay details and measurement strategy.

## Results

Results paragraph with findings and profile comparisons.

## Discussion

Discussion paragraph with clinical implications and therapeutic interpretation.

Discussion closing paragraph with future work and limitations.
"""
        article_two = """# Review B

## Introduction

Opening context paragraph.

Second introduction paragraph extending the scope.

## Discussion

Long discussion paragraph with interpretation and clinical implications.

Another discussion paragraph with management implications.

Final discussion paragraph with limitations and future research priorities.
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir)
            (corpus_dir / "PMC00000001.md").write_text(article_one, encoding="utf-8")
            (corpus_dir / "PMC00000002.md").write_text(article_two, encoding="utf-8")

            payload = derive_six_slot_priors_from_corpus(corpus_dir)

        shares = payload["normalized_mean_shares"]
        self.assertEqual(payload["mapped_article_count"], 2)
        self.assertEqual(list(shares.keys()), list(SIX_SLOT_ORDER))
        self.assertAlmostEqual(sum(shares.values()), 1.0, places=5)
        middle_values = [
            shares["framework_mechanism"],
            shares["evidence_methods"],
            shares["findings_synthesis"],
            shares["implications_discussion"],
        ]
        self.assertGreater(max(middle_values) - min(middle_values), 0.01)
        self.assertGreater(shares["implications_discussion"], shares["framework_mechanism"])


if __name__ == "__main__":
    unittest.main()
