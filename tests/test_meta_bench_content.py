import unittest

from meta_bench.content import (
    LLMProxyQuestionJudge,
    ProxyQuestionDecision,
    ProxyQuestionSpec,
    create_default_proxy_question_judge,
    evaluate_content_dimension,
    keyword_in_text,
    score_entity_consistency,
    score_proxy_hit_rate,
)


class StaticProxyJudge:
    def __init__(self, answered_by_qid):
        self.answered_by_qid = answered_by_qid

    def judge_proxy_question(self, *, final_text, question):
        answered = self.answered_by_qid[question.qid]
        return ProxyQuestionDecision(
            qid=question.qid,
            question=question.question,
            answered=answered,
            covered_points=question.required_points if answered else [],
            missing_points=[] if answered else question.required_points,
            rationale="static test decision",
        )


class EntityConsistencyTests(unittest.TestCase):
    def test_keyword_in_text_matches_case_insensitive_phrases(self):
        text = "This review focuses on Acute Coronary Syndrome in adult care."

        self.assertTrue(keyword_in_text(text, "acute coronary syndrome"))

    def test_score_entity_consistency_returns_ratio_and_details(self):
        result = score_entity_consistency(
            "Scope and acute coronary syndrome are discussed.",
            ["scope", "acute coronary syndrome", "hemodynamics"],
        )

        self.assertAlmostEqual(result.score, 2 / 3)
        self.assertEqual(result.keyword_hit_count, 2)
        self.assertEqual(result.keyword_total, 3)
        self.assertEqual(result.matched_keywords, ["scope", "acute coronary syndrome"])
        self.assertEqual(result.missing_keywords, ["hemodynamics"])

    def test_evaluate_content_dimension_extracts_must_include_from_reference(self):
        result = evaluate_content_dimension(
            "The paper defines scope and limitations.",
            {
                "constraints": {
                    "must_include": ["scope", "limitations"],
                }
            },
            proxy_judge=None,
        )

        self.assertEqual(result["dimension"], "content")
        self.assertEqual(result["scores"]["entity_consistency_score"], 1.0)

    def test_empty_keywords_are_rejected(self):
        with self.assertRaises(ValueError):
            score_entity_consistency("valid text", [])


class ProxyHitRateTests(unittest.TestCase):
    def test_score_proxy_hit_rate_uses_judge_decisions(self):
        questions = [
            ProxyQuestionSpec("q1", "Does it define the scope?", ["scope defined"]),
            ProxyQuestionSpec("q2", "Does it discuss gaps?", ["gaps discussed"]),
        ]
        judge = StaticProxyJudge({"q1": True, "q2": False})

        result = score_proxy_hit_rate("Generated article text.", questions, judge)

        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.answered_count, 1)
        self.assertEqual(result.question_total, 2)
        self.assertEqual(result.covered_point_count, 1)
        self.assertEqual(result.required_point_total, 2)
        self.assertEqual(result.decisions[0].covered_points, ["scope defined"])
        self.assertEqual(result.decisions[1].missing_points, ["gaps discussed"])

    def test_score_proxy_hit_rate_uses_required_point_coverage(self):
        class IncompleteProxyJudge:
            def judge_proxy_question(self, *, final_text, question):
                return ProxyQuestionDecision(
                    qid=question.qid,
                    question=question.question,
                    answered=True,
                    covered_points=["scope"],
                    missing_points=["limitations"],
                    rationale="partially covered",
                )

        questions = [
            ProxyQuestionSpec(
                "q1",
                "Does it cover scope and limitations?",
                ["scope", "limitations"],
            ),
        ]

        result = score_proxy_hit_rate(
            "Generated article text.",
            questions,
            IncompleteProxyJudge(),
        )

        self.assertEqual(result.score, 0.5)
        self.assertEqual(result.answered_count, 0)
        self.assertEqual(result.covered_point_count, 1)
        self.assertEqual(result.required_point_total, 2)
        self.assertTrue(result.decisions[0].answered)
        self.assertEqual(result.decisions[0].missing_points, ["limitations"])

    def test_evaluate_content_dimension_adds_proxy_score_when_judge_is_provided(self):
        reference = {
            "constraints": {"must_include": ["scope"]},
            "proxy_questions": [
                {
                    "qid": "q1",
                    "question": "Does it define the scope?",
                    "required_points": ["scope defined"],
                }
            ],
        }
        judge = StaticProxyJudge({"q1": True})

        result = evaluate_content_dimension(
            "The article defines scope.",
            reference,
            proxy_judge=judge,
        )

        self.assertEqual(result["scores"]["entity_consistency_score"], 1.0)
        self.assertEqual(result["scores"]["proxy_hit_rate"], 1.0)

    def test_evaluate_content_dimension_marks_proxy_unscored_without_judge(self):
        reference = {
            "constraints": {"must_include": ["scope"]},
            "proxy_questions": [
                {
                    "qid": "q1",
                    "question": "Does it define the scope?",
                    "required_points": ["scope defined"],
                }
            ],
        }

        result = evaluate_content_dimension(
            "The article defines scope.",
            reference,
            proxy_judge=None,
        )

        self.assertNotIn("proxy_hit_rate", result["scores"])
        self.assertEqual(
            result["diagnostics"]["proxy_hit_rate"]["reason"],
            "proxy_judge_not_provided",
        )

    def test_llm_proxy_judge_parses_strict_json_response(self):
        judge = LLMProxyQuestionJudge(
            lambda _prompt: (
                '{"answered": true, "covered_points": ["scope"], '
                '"missing_points": [], "rationale": "covered"}'
            )
        )
        question = ProxyQuestionSpec("q1", "Does it define scope?", ["scope"])

        decision = judge.judge_proxy_question(
            final_text="The article defines scope.",
            question=question,
        )

        self.assertTrue(decision.answered)
        self.assertEqual(decision.covered_points, ["scope"])
        self.assertEqual(decision.rationale, "covered")

    def test_llm_proxy_judge_parses_fenced_json_response(self):
        judge = LLMProxyQuestionJudge(
            lambda _prompt: (
                "```json\n"
                '{"answered": false, "covered_points": [], '
                '"missing_points": ["scope"], "rationale": "missing"}'
                "\n```"
            )
        )
        question = ProxyQuestionSpec("q1", "Does it define scope?", ["scope"])

        decision = judge.judge_proxy_question(
            final_text="No relevant content.",
            question=question,
        )

        self.assertFalse(decision.answered)
        self.assertEqual(decision.missing_points, ["scope"])

    def test_create_default_proxy_question_judge_uses_project_llm_client(self):
        judge = create_default_proxy_question_judge(
            api_key="test-key",
            model="test-model",
            base_url="http://example.test/v1",
            load_env=False,
        )

        self.assertIsInstance(judge, LLMProxyQuestionJudge)
        self.assertEqual(getattr(judge, "model"), "test-model")


if __name__ == "__main__":
    unittest.main()
