"""Unified evaluation entry point for the clean MetaBench package."""

from __future__ import annotations

from typing import Literal, Mapping

from .citation import evaluate_citation_dimension
from .content import ProxyQuestionJudge, evaluate_content_dimension
from .structure import evaluate_structure_dimension

META_BENCH_METRIC_ORDER = [
    "entity_consistency_score",
    "proxy_hit_rate",
    "length_score",
    "completion_rate",
    "source_fidelity",
    "section_distribution",
    "citation_count",
    "source_balance",
]
def _safe_content_evaluation(
    final_text: str,
    reference: Mapping[str, object],
    *,
    proxy_judge: ProxyQuestionJudge | Literal["default"] | None,
) -> dict[str, object]:
    if proxy_judge != "default":
        return evaluate_content_dimension(
            final_text,
            reference,
            proxy_judge=proxy_judge,
        )

    try:
        return evaluate_content_dimension(
            final_text,
            reference,
            proxy_judge="default",
        )
    except Exception as exc:
        result = evaluate_content_dimension(
            final_text,
            reference,
            proxy_judge=None,
        )
        diagnostics = result.setdefault("diagnostics", {})
        proxy_diagnostics = diagnostics.setdefault("proxy_hit_rate", {})
        if isinstance(proxy_diagnostics, dict):
            proxy_diagnostics["status"] = "not_evaluated"
            proxy_diagnostics["reason"] = "proxy_judge_error"
            proxy_diagnostics["error"] = str(exc)
        return result


def evaluate_meta_bench(
    final_text: str,
    reference: Mapping[str, object],
    *,
    proxy_judge: ProxyQuestionJudge | Literal["default"] | None = "default",
) -> dict[str, object]:
    """Evaluate a generated article with the three-dimension MetaBench scheme."""

    content_result = _safe_content_evaluation(
        final_text,
        reference,
        proxy_judge=proxy_judge,
    )
    structure_result = evaluate_structure_dimension(final_text, reference)
    citation_result = evaluate_citation_dimension(final_text, reference)

    dimensions = {
        "content": content_result,
        "structure": structure_result,
        "citation": citation_result,
    }

    metric_scores: dict[str, float] = {}
    for result in dimensions.values():
        scores = result.get("scores", {})
        if not isinstance(scores, dict):
            continue

        numeric_scores = {
            str(metric_name): float(metric_value)
            for metric_name, metric_value in scores.items()
            if isinstance(metric_value, (int, float))
        }
        metric_scores.update(numeric_scores)
    scored_metric_count = len(metric_scores)
    expected_metric_count = len(META_BENCH_METRIC_ORDER)
    unscored_metrics = [
        metric_name
        for metric_name in META_BENCH_METRIC_ORDER
        if metric_name not in metric_scores
    ]

    return {
        "status": "completed",
        "framework": "meta_bench",
        "task_id": reference.get("task_id"),
        "metric_scores": metric_scores,
        "score_coverage": {
            "scored_metric_count": scored_metric_count,
            "expected_metric_count": expected_metric_count,
            "unscored_metrics": unscored_metrics,
        },
        "dimensions": dimensions,
    }


__all__ = [
    "META_BENCH_METRIC_ORDER",
    "evaluate_meta_bench",
]
