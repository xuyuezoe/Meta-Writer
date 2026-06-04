"""Deterministic task generator for the clean MetaBench package."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .content import ProxyQuestionSpec
from .section_budget import build_section_budget
from .schemas import GeneratedTask, TaskSpec


def _clean_items(items: list[str]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        value = str(item).strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        cleaned.append(value)
        seen.add(key)
    return cleaned


def _join_human(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _clean_text(text: str) -> str:
    """Normalize display text for prompts and outlines."""

    value = str(text)
    value = value.replace("閳?", " - ")
    value = value.replace("閳ワ拷", " - ")
    value = value.replace("鈥?", " - ")
    value = value.replace("斜泻", " - ")
    return " ".join(value.split())


def _resolve_body_target_words(spec: TaskSpec) -> int:
    """Resolve the body-only target word count used by generation and scoring."""

    resolved = spec.body_target_words if spec.body_target_words is not None else spec.target_words
    body_target_words = int(resolved)
    if body_target_words <= 0:
        raise ValueError("body_target_words must be positive")
    return body_target_words


def build_must_include(spec: TaskSpec) -> list[str]:
    """Build required keyword list for entity-consistency scoring."""

    return _clean_items(
        [
            "scope",
            spec.organizer,
            spec.domain,
            spec.topic,
            *spec.focus_points,
            spec.practice_context,
            *spec.closing_requirements,
            *spec.extra_must_include,
        ]
    )


def generate_prompt(spec: TaskSpec) -> str:
    """Generate the user-facing writing prompt for a benchmark task."""

    body_target_words = _resolve_body_target_words(spec)
    focus_text = _join_human(spec.focus_points) or "evidence integration"
    closing_text = _join_human(spec.closing_requirements) or "limitations and future work"
    return _clean_text(
        f"Write an approximately {body_target_words}-word {spec.paper_type.replace('_', ' ')} "
        f"article in {spec.language} on {spec.topic} within {spec.domain}. "
        "Keep the piece as a long-form scholarly review rather than a case report, "
        "a popular-science summary, or an outline. "
        f"Open by defining the scope, discussion boundaries, and terminology, while keeping "
        f"{spec.practice_context} as an explicit practice context. "
        f"Organize the main body around a {spec.organizer} and keep "
        f"{spec.practice_context} on the main line of discussion. "
        f"Compare and synthesize the literature around {focus_text}. "
        f"Develop at least {spec.expected_sections} natural sections and close with {closing_text}."
    )


def generate_outline(spec: TaskSpec) -> dict[str, str]:
    """Generate an outline with exactly ``expected_sections`` sections."""

    section_count = max(3, int(spec.expected_sections))
    focus_points = spec.focus_points or ["mechanism", "evidence integration"]
    outline: "OrderedDict[str, str]" = OrderedDict()

    outline["sec1"] = _clean_text(
        "Scope, terminology, and practice context - define the scope and situate "
        f"{spec.topic} within {spec.domain}"
    )

    outline[f"sec{section_count}"] = _clean_text(
        "Future work and research agenda - outline priorities for future investigation "
        f"and {spec.closing_requirements[-1] if spec.closing_requirements else 'future work'}"
    )
    if section_count >= 2:
        outline[f"sec{section_count - 1}"] = _clean_text(
            "Limitations and evidence gaps - assess methodological constraints, "
            "uncertainty, and open questions"
        )

    body_templates = [
        _clean_text(
            f"Organizing framework - structure the discussion around {spec.organizer}"
        ),
        _clean_text(
            f"Mechanisms and explanatory models - synthesize evidence on {focus_points[0]}"
        ),
        _clean_text(
            "Evidence synthesis - compare findings across populations, methods, and care settings"
        ),
        _clean_text(
            f"Practice implications - connect evidence to {spec.practice_context}"
        ),
        _clean_text(
            "Heterogeneity and subgroup considerations - compare variation across patient groups"
        ),
        _clean_text(
            "Implementation considerations - discuss feasibility, trade-offs, and system constraints"
        ),
        _clean_text(
            "Controversies and unresolved questions - compare conflicting findings"
        ),
    ]

    body_count = section_count - 3
    for index in range(body_count):
        section_id = f"sec{index + 2}"
        outline[section_id] = body_templates[index % len(body_templates)]

    if section_count == 3:
        outline["sec2"] = _clean_text(
            f"Evidence synthesis and limitations - compare the literature around "
            f"{spec.organizer} and identify evidence gaps"
        )

    return dict(sorted(outline.items(), key=lambda item: int(item[0][3:])))


def generate_proxy_questions(spec: TaskSpec) -> list[ProxyQuestionSpec]:
    """Generate deterministic topic-derived proxy questions."""

    focus_text = _join_human(spec.focus_points) or "the main evidence base"
    closing_text = _join_human(spec.closing_requirements) or "limitations and future work"
    article_specific_points = [
        f"covers article-specific detail: {item}"
        for item in _clean_items(spec.extra_must_include)[:4]
    ]

    return [
        ProxyQuestionSpec(
            qid="q_scope",
            question=(
                f"Does the article define the scope, boundaries, and key terminology for "
                f"{spec.topic} within {spec.domain}?"
            ),
            required_points=[
                f"defines the scope of {spec.topic}",
                "states discussion boundaries or terminology",
                f"situates the topic within {spec.domain}",
            ],
        ),
        ProxyQuestionSpec(
            qid="q_organizer",
            question=(
                f"Does the article organize the discussion around a {spec.organizer}?"
            ),
            required_points=[
                f"uses {spec.organizer} as an organizing axis",
                "separates or compares major categories rather than listing facts only",
            ],
        ),
        ProxyQuestionSpec(
            qid="q_evidence",
            question=(
                f"Does the article compare and synthesize evidence about {focus_text} "
                f"for {spec.topic}?"
            ),
            required_points=[
                *(f"discusses {point}" for point in spec.focus_points),
                *article_specific_points,
                "compares or synthesizes evidence rather than only mentioning isolated facts",
            ],
        ),
        ProxyQuestionSpec(
            qid="q_context",
            question=(
                f"Does the article connect {spec.topic} to the {spec.practice_context} context?"
            ),
            required_points=[
                f"explicitly discusses {spec.practice_context}",
                "connects the topic to practical decisions, workflows, or constraints",
            ],
        ),
        ProxyQuestionSpec(
            qid="q_closing",
            question=(
                f"Does the article close by addressing {closing_text} for {spec.topic}?"
            ),
            required_points=[
                *(f"discusses {item}" for item in spec.closing_requirements),
                "identifies uncertainty, open questions, or next steps",
            ],
        ),
    ]


def _proxy_question_to_dict(question: ProxyQuestionSpec) -> dict[str, Any]:
    return {
        "qid": question.qid,
        "question": question.question,
        "required_points": list(question.required_points),
    }


def generate_constraints(
    spec: TaskSpec,
    *,
    outline: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Generate scoring constraints for the task reference."""

    must_include = build_must_include(spec)
    expected_sections = max(3, int(spec.expected_sections))
    resolved_outline = outline or generate_outline(spec)
    body_target_words = _resolve_body_target_words(spec)
    section_budget = build_section_budget(
        task_id=spec.task_id,
        body_target_words=body_target_words,
        outline=resolved_outline,
    )

    constraints = {
        "required_length_words": body_target_words,
        "body_target_words": body_target_words,
        "total_target_words": int(spec.target_words),
        "expected_sections": expected_sections,
        "must_include": must_include,
        "section_word_targets": dict(section_budget.section_word_targets),
        "section_roles": dict(section_budget.section_roles),
        "required_sections": _build_required_completion_slots(
            dict(section_budget.section_roles)
        ),
        "periodic_requirements": [
            "The main body should include explicit comparison or synthesis at regular intervals.",
            "The second half should address limitations, evidence gaps, or future work.",
        ],
    }
    constraints.update(section_budget.to_reference_payload())
    return constraints


def _build_required_completion_slots(section_roles: dict[str, str]) -> list[str]:
    """Build completion slots from the task's section-role layout."""

    closing_role_aliases = {"conclusion", "limitations_gaps"}
    required_slots = ["introduction", "main_body"]

    roles_in_use = {
        str(role).strip()
        for role in section_roles.values()
        if str(role).strip()
    }
    if roles_in_use & closing_role_aliases:
        if "limitations_gaps" in roles_in_use:
            required_slots.append("limitations_gaps")
        else:
            required_slots.append("conclusion")
    else:
        required_slots.append("conclusion")

    return required_slots


def generate_review_task(
    spec: TaskSpec,
    *,
    outline_override: dict[str, str] | None = None,
) -> GeneratedTask:
    """Generate a complete review-style benchmark task."""

    prompt = generate_prompt(spec)
    outline = dict(outline_override) if outline_override is not None else generate_outline(spec)
    constraints = generate_constraints(spec, outline=outline)
    proxy_questions = generate_proxy_questions(spec)
    reference = {
        "task_id": spec.task_id,
        "paper_type": spec.paper_type,
        "topic": spec.topic,
        "domain": spec.domain,
        "language": spec.language,
        "constraints": constraints,
        "outline": outline,
        "proxy_questions": [_proxy_question_to_dict(item) for item in proxy_questions],
    }
    return GeneratedTask(
        task_id=spec.task_id,
        prompt=prompt,
        constraints=constraints,
        outline=outline,
        reference=reference,
    )


def build_generation_constraints(spec: TaskSpec) -> list[str]:
    """Build runtime constraints for the main generation loop."""

    body_target_words = _resolve_body_target_words(spec)
    focus_text = _join_human(spec.focus_points) or "the main evidence base"
    closing_text = _join_human(spec.closing_requirements) or "limitations and future work"
    return [
        f"Write the article in {spec.language}.",
        f"Target body length is about {body_target_words} words overall, excluding the final references list.",
        f"Develop approximately {max(3, int(spec.expected_sections))} major sections following the outline.",
        f"Keep the discussion centered on {spec.topic} within {spec.domain}.",
        f"Use {spec.organizer} as a visible organizing logic in the main body.",
        f"Compare and synthesize evidence about {focus_text}.",
        f"Explicitly connect the discussion to {spec.practice_context}.",
        f"Close by addressing {closing_text}.",
    ]


def build_main_task_config(
    spec: TaskSpec,
    *,
    session_name: str | None = None,
    corpus_dir: str | None = None,
    outline_override: dict[str, str] | None = None,
) -> dict[str, object]:
    """Adapt a clean MetaBench task into the main runtime config shape."""

    generated = generate_review_task(spec, outline_override=outline_override)
    resolved_session_name = session_name or f"meta_bench_{spec.task_id}"
    config = {
        "task": generated.prompt,
        "constraints": build_generation_constraints(spec),
        "outline": generated.outline,
        "session_name": resolved_session_name,
        "reference": generated.reference,
    }
    if isinstance(corpus_dir, str) and corpus_dir.strip():
        config["corpus_dir"] = corpus_dir
    return config
