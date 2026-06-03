"""Content-dimension scores for the clean MetaBench evaluator.

The first content score is ``entity_consistency_score``: the fraction of
required keywords that appear in the generated text.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Mapping, Protocol, Sequence, cast


@dataclass(frozen=True)
class EntityConsistencyResult:
    """Keyword-match result for entity consistency."""

    score: float
    keyword_hit_count: int
    keyword_total: int
    matched_keywords: list[str]
    missing_keywords: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "entity_consistency_score": self.score,
            "keyword_hit_count": self.keyword_hit_count,
            "keyword_total": self.keyword_total,
            "matched_keywords": self.matched_keywords,
            "missing_keywords": self.missing_keywords,
        }


@dataclass(frozen=True)
class ProxyQuestionSpec:
    """A topic-derived question that the output should answer."""

    qid: str
    question: str
    required_points: list[str]


@dataclass(frozen=True)
class ProxyQuestionDecision:
    """LLM-judge decision for one proxy question."""

    qid: str
    question: str
    answered: bool
    covered_points: list[str]
    missing_points: list[str]
    rationale: str = ""


@dataclass(frozen=True)
class ProxyHitRateResult:
    """Aggregated LLM-judge coverage over proxy questions."""

    score: float
    answered_count: int
    question_total: int
    covered_point_count: int
    required_point_total: int
    decisions: list[ProxyQuestionDecision]

    def to_dict(self) -> dict[str, object]:
        return {
            "proxy_hit_rate": self.score,
            "answered_count": self.answered_count,
            "question_total": self.question_total,
            "covered_point_count": self.covered_point_count,
            "required_point_total": self.required_point_total,
            "decisions": [
                {
                    "qid": decision.qid,
                    "question": decision.question,
                    "answered": decision.answered,
                    "covered_points": decision.covered_points,
                    "missing_points": decision.missing_points,
                    "rationale": decision.rationale,
                }
                for decision in self.decisions
            ],
        }


class ProxyQuestionJudge(Protocol):
    """Judge interface for topic-derived proxy questions."""

    def judge_proxy_question(
        self,
        *,
        final_text: str,
        question: ProxyQuestionSpec,
    ) -> ProxyQuestionDecision:
        """Return whether the final text answers one proxy question."""


class LLMProxyQuestionJudge:
    """API-agnostic LLM judge wrapper.

    The benchmark core does not know about a concrete model provider. Callers
    pass a ``complete(prompt) -> str`` function, and this wrapper handles the
    prompt contract plus JSON parsing.
    """

    def __init__(self, complete: Callable[[str], str]):
        self._complete = complete

    def judge_proxy_question(
        self,
        *,
        final_text: str,
        question: ProxyQuestionSpec,
    ) -> ProxyQuestionDecision:
        prompt = build_proxy_question_judge_prompt(final_text, question)
        raw_response = self._complete(prompt)
        return parse_proxy_question_judge_response(raw_response, question)


def create_default_proxy_question_judge(
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 512,
    load_env: bool = True,
) -> LLMProxyQuestionJudge:
    """Create a real project LLM judge from the current API configuration.

    Environment variables follow the rest of the project:
        API_KEY, BASE_URL, MODEL

    ``META_BENCH_PROXY_JUDGE_MODEL`` can override ``MODEL`` for this judge.
    """

    if load_env:
        from dotenv import load_dotenv

        load_dotenv(override=True)

    resolved_api_key = api_key or os.getenv("API_KEY")
    if not resolved_api_key:
        raise EnvironmentError("API_KEY is required for proxy question judging")

    resolved_model = (
        model
        or os.getenv("META_BENCH_PROXY_JUDGE_MODEL")
        or os.getenv("MODEL")
        or "glm-5"
    )
    resolved_base_url = base_url if base_url is not None else os.getenv("BASE_URL")

    from src.utils.llm_client import LLMClient

    client = LLMClient(
        api_key=resolved_api_key,
        model=resolved_model,
        base_url=resolved_base_url,
    )

    def complete(prompt: str) -> str:
        return client.generate(
            prompt,
            temperature=0.0,
            max_tokens=max_tokens,
            strip_think=True,
        )

    judge = LLMProxyQuestionJudge(complete)
    setattr(judge, "llm_client", client)
    setattr(judge, "model", resolved_model)
    return judge


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def _normalize_keywords(keywords: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for keyword in keywords:
        value = str(keyword).strip()
        if value:
            normalized.append(value)
    return normalized


def keyword_in_text(text: str, keyword: str) -> bool:
    """Return whether ``keyword`` appears in ``text``.

    Matching is deterministic and local: case-insensitive substring matching
    after whitespace normalization. Phrases such as "acute coronary syndrome"
    are therefore matched as phrases, while Chinese keywords work unchanged.
    """

    if keyword.strip() == "":
        return False
    return _normalize_text(keyword) in _normalize_text(text)


def score_entity_consistency(
    final_text: str,
    keywords: Sequence[object],
) -> EntityConsistencyResult:
    """Score required-keyword coverage.

    Formula:
        entity_consistency_score = matched_keyword_count / keyword_total

    Args:
        final_text: Generated answer text.
        keywords: Required keywords or phrases. Each list item counts once,
            matching the original benchmark behavior.
    """

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")

    required_keywords = _normalize_keywords(keywords)
    if not required_keywords:
        raise ValueError("keywords must contain at least one non-empty keyword")

    matched_keywords = [
        keyword for keyword in required_keywords if keyword_in_text(final_text, keyword)
    ]
    missing_keywords = [
        keyword for keyword in required_keywords if keyword not in matched_keywords
    ]
    score = len(matched_keywords) / len(required_keywords)
    return EntityConsistencyResult(
        score=score,
        keyword_hit_count=len(matched_keywords),
        keyword_total=len(required_keywords),
        matched_keywords=matched_keywords,
        missing_keywords=missing_keywords,
    )


def build_proxy_question_judge_prompt(
    final_text: str,
    question: ProxyQuestionSpec,
) -> str:
    """Build the deterministic prompt contract for proxy-question judging."""

    required_points_text = "\n".join(
        f"- {point}" for point in question.required_points
    )
    if required_points_text == "":
        required_points_text = "- Judge from the question itself; no explicit points were provided."
    return (
        "You are a benchmark judge. Decide whether the generated article answers "
        "the topic-derived proxy question.\n\n"
        "Use only the generated article. Do not reward unsupported assumptions.\n"
        "Classify every required answer point independently. Copy each required "
        "answer point into exactly one of covered_points or missing_points. "
        "Set answered=true only when every required answer point is directly "
        "covered. If any required point is missing, vague, or only implied, "
        "set answered=false.\n"
        "Return strict JSON with this schema:\n"
        '{\n'
        '  "answered": true,\n'
        '  "covered_points": ["..."],\n'
        '  "missing_points": ["..."],\n'
        '  "rationale": "short explanation"\n'
        '}\n\n'
        f"Question id: {question.qid}\n"
        f"Proxy question: {question.question}\n\n"
        "Required answer points:\n"
        f"{required_points_text}\n\n"
        "Generated article:\n"
        f"{final_text}"
    )


def parse_proxy_question_judge_response(
    raw_response: str,
    question: ProxyQuestionSpec,
) -> ProxyQuestionDecision:
    """Parse one LLM-judge response into a typed decision."""

    json_text = _extract_json_object(raw_response)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError("proxy question judge response must be valid JSON") from exc

    if not isinstance(parsed, dict):
        raise ValueError("proxy question judge response must decode to an object")

    covered_points = parsed.get("covered_points", [])
    missing_points = parsed.get("missing_points", [])
    if not isinstance(covered_points, list):
        covered_points = []
    if not isinstance(missing_points, list):
        missing_points = []

    return ProxyQuestionDecision(
        qid=question.qid,
        question=question.question,
        answered=bool(parsed.get("answered", False)),
        covered_points=[
            str(point).strip() for point in covered_points if str(point).strip()
        ],
        missing_points=[
            str(point).strip() for point in missing_points if str(point).strip()
        ],
        rationale=str(parsed.get("rationale", "")),
    )


def _extract_json_object(raw_response: str) -> str:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def score_proxy_hit_rate(
    final_text: str,
    proxy_questions: Sequence[ProxyQuestionSpec],
    judge: ProxyQuestionJudge,
) -> ProxyHitRateResult:
    """Score how many required proxy-question points are covered.

    Formula:
        proxy_hit_rate = covered_required_point_count / required_point_total

    ``answered_count`` remains a whole-question diagnostic: a question is fully
    answered only when the judge marks it answered and reports no missing
    required points.
    """

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if not proxy_questions:
        raise ValueError("proxy_questions must contain at least one question")

    decisions = [
        judge.judge_proxy_question(final_text=final_text, question=question)
        for question in proxy_questions
    ]
    answered_count = sum(
        1 for decision in decisions if decision.answered and not decision.missing_points
    )
    covered_point_count = 0
    required_point_total = 0
    for question, decision in zip(proxy_questions, decisions):
        required_count = len(question.required_points)
        if required_count == 0:
            required_point_total += 1
            if decision.answered and not decision.missing_points:
                covered_point_count += 1
            continue

        required_point_total += required_count
        covered_count = len({point.casefold() for point in decision.covered_points})
        if covered_count == 0 and decision.answered and not decision.missing_points:
            covered_count = required_count
        covered_point_count += min(required_count, covered_count)

    score = covered_point_count / required_point_total
    return ProxyHitRateResult(
        score=score,
        answered_count=answered_count,
        question_total=len(proxy_questions),
        covered_point_count=covered_point_count,
        required_point_total=required_point_total,
        decisions=decisions,
    )


def _extract_must_include(reference: Mapping[str, object]) -> list[object]:
    constraints = reference.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TypeError("reference.constraints must be a mapping")
    must_include = constraints.get("must_include")
    if not isinstance(must_include, list):
        raise TypeError("reference.constraints.must_include must be a list")
    return list(must_include)


def _normalize_required_points(item: Mapping[str, object]) -> list[str]:
    for key in ("required_points", "answer_points", "expected_points"):
        raw_points = item.get(key)
        if isinstance(raw_points, list):
            return [str(point).strip() for point in raw_points if str(point).strip()]
        if isinstance(raw_points, str) and raw_points.strip():
            return [raw_points.strip()]

    legacy_answer = item.get("answer")
    if isinstance(legacy_answer, str) and legacy_answer.strip():
        return [legacy_answer.strip()]
    return []


def _extract_proxy_questions(reference: Mapping[str, object]) -> list[ProxyQuestionSpec]:
    raw_questions = reference.get("proxy_questions", [])
    if raw_questions is None:
        return []
    if not isinstance(raw_questions, list):
        raise TypeError("reference.proxy_questions must be a list")

    questions: list[ProxyQuestionSpec] = []
    for index, raw_item in enumerate(raw_questions, start=1):
        if not isinstance(raw_item, Mapping):
            raise TypeError("each proxy question must be a mapping")
        item = cast(Mapping[str, object], raw_item)
        question_text = str(item.get("question", "")).strip()
        if not question_text:
            raise ValueError("each proxy question must contain a non-empty question")
        qid = str(item.get("qid", f"q{index}")).strip() or f"q{index}"
        questions.append(
            ProxyQuestionSpec(
                qid=qid,
                question=question_text,
                required_points=_normalize_required_points(item),
            )
        )
    return questions


def evaluate_content_dimension(
    final_text: str,
    reference: Mapping[str, object],
    *,
    proxy_judge: ProxyQuestionJudge | Literal["default"] | None = "default",
) -> dict[str, object]:
    """Evaluate the content dimension currently implemented in ``meta_bench``."""

    entity_result = score_entity_consistency(
        final_text=final_text,
        keywords=_extract_must_include(reference),
    )
    scores: dict[str, object] = {
        "entity_consistency_score": entity_result.score,
    }
    diagnostics: dict[str, object] = {
        "entity_consistency": entity_result.to_dict(),
    }

    proxy_questions = _extract_proxy_questions(reference)
    if proxy_questions:
        active_proxy_judge: ProxyQuestionJudge | None
        if proxy_judge == "default":
            active_proxy_judge = create_default_proxy_question_judge()
        else:
            active_proxy_judge = proxy_judge

        if active_proxy_judge is None:
            diagnostics["proxy_hit_rate"] = {
                "status": "not_evaluated",
                "reason": "proxy_judge_not_provided",
                "question_total": len(proxy_questions),
            }
        else:
            proxy_result = score_proxy_hit_rate(
                final_text=final_text,
                proxy_questions=proxy_questions,
                judge=active_proxy_judge,
            )
            scores["proxy_hit_rate"] = proxy_result.score
            diagnostics["proxy_hit_rate"] = proxy_result.to_dict()

    return {
        "dimension": "content",
        "scores": scores,
        "diagnostics": diagnostics,
    }
