from __future__ import annotations

import unittest

from meta_bench.evaluator import evaluate_meta_bench


class StaticProxyJudge:
    def judge_proxy_question(self, *, final_text, question):
        del final_text
        return type(
            "Decision",
            (),
            {
                "qid": question.qid,
                "question": question.question,
                "answered": True,
                "covered_points": list(question.required_points),
                "missing_points": [],
                "rationale": "covered",
            },
        )()


class MetaBenchEvaluatorTests(unittest.TestCase):
    def test_evaluate_meta_bench_collects_three_dimensions(self) -> None:
        final_text = """# Acute coronary syndrome review

## Introduction

This review defines the scope, terminology, and background for acute coronary syndrome in cardiovascular medicine and adult inpatient care.

## Evidence synthesis

The classification framework compares pathophysiology, hemodynamics, and evidence integration across studies and trials.

## Conclusion

Overall, the article discusses limitations and future work for acute coronary syndrome in adult inpatient care.
"""
        reference = {
            "task_id": "med_s001",
            "constraints": {
                "must_include": [
                    "scope",
                    "classification framework",
                    "acute coronary syndrome",
                    "cardiovascular medicine",
                    "adult inpatient care",
                    "limitations",
                    "future work",
                ],
                "required_length_words": 40,
                "expected_sections": 7,
            },
            "proxy_questions": [
                {
                    "qid": "q1",
                    "question": "Does it define scope?",
                    "required_points": ["defines scope"],
                }
            ],
        }

        result = evaluate_meta_bench(
            final_text,
            reference,
            proxy_judge=StaticProxyJudge(),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["framework"], "meta_bench")
        self.assertIn("entity_consistency_score", result["metric_scores"])
        self.assertIn("proxy_hit_rate", result["metric_scores"])
        self.assertIn("length_score", result["metric_scores"])
        self.assertIn("completion_rate", result["metric_scores"])
        self.assertIn("citation_count", result["metric_scores"])
        self.assertIn("source_balance", result["metric_scores"])

    def test_evaluate_meta_bench_marks_unscored_metrics(self) -> None:
        final_text = """## Introduction

Scope and background.

## Main body

Evidence and comparison.

## Conclusion

Overall limitations and future work.
"""
        reference = {
            "task_id": "med_s001",
            "constraints": {
                "must_include": ["scope"],
                "required_length_words": 10,
                "expected_sections": 3,
            },
            "proxy_questions": [
                {
                    "qid": "q1",
                    "question": "Does it define scope?",
                    "required_points": ["scope"],
                }
            ],
        }

        result = evaluate_meta_bench(final_text, reference, proxy_judge=None)

        self.assertIn("proxy_hit_rate", result["score_coverage"]["unscored_metrics"])
        self.assertNotIn("proxy_hit_rate", result["metric_scores"])


if __name__ == "__main__":
    unittest.main()
