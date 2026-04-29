"""English benchmark adapter for MetaWriter.

This module loads benchmark samples from the local `metabench/examples/samples.jsonl`
bundle, converts them into task configs that MetaWriter can run directly, and
provides a lightweight local evaluator that does not depend on an external judge.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Union, cast


BENCHMARK_ROOT: Path = Path(__file__).resolve().parent.parent / "metabench"
METABENCH_SRC_ROOT: Path = BENCHMARK_ROOT / "src"
SAMPLES_PATH: Path = BENCHMARK_ROOT / "examples" / "samples.jsonl"
DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "Document-level requirement: "

if str(METABENCH_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(METABENCH_SRC_ROOT))

from metabench.local_metrics import (
    compute_acc_once,
    compute_acc_periodic,
    compute_acc_range,
    compute_completion_rate,
    compute_instruction_hits,
    compute_proxy_qa,
    compute_soft_instruction_hits,
    compute_structure_scores,
    contains_keyword,
    count_length_units,
    evaluate_periodic_keywords,
    evaluate_range_keywords,
    split_blocks,
)
from metabench.scoring import compute_s_length

ORGANIZER_CANDIDATES_EN = [
    "classification framework",
    "clinical pathway",
    "controversy focus",
    "subgroup stratification",
    "implementation barriers",
    "evidence map",
]

CLOSING_CANDIDATES_EN = [
    "evidence gaps",
    "open questions",
    "research agenda",
    "future work",
]


def _contains_cjk(text: str) -> bool:
    """Return True when the text still contains Chinese characters."""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def is_document_level_constraint(requirement: str) -> bool:
    """Return whether the constraint is tagged as document-level."""
    return requirement.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX)


def _mark_document_level_constraint(requirement: str) -> str:
    """Prefix a document-level benchmark requirement for the validator layer."""
    return f"{DOCUMENT_LEVEL_CONSTRAINT_PREFIX}{requirement}"


def _extract_paragraph_blocks(text: str) -> List[str]:
    """Extract body paragraphs while skipping MetaWriter assembly wrappers."""
    normalized_text = text.replace("\r\n", "\n").strip()
    if normalized_text == "":
        return []
    return split_blocks(normalized_text, drop_markdown_wrappers=True)


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """Read a JSONL file into a list of dictionaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark sample file not found: {file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        rows.append(json.loads(stripped_line))
    return rows


def list_benchmark_task_ids() -> List[str]:
    """Return benchmark task IDs in file order."""
    task_ids: List[str] = []
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        sample_id = str(sample_row["sample_id"])
        if sample_id == "":
            raise ValueError("benchmark sample_id must not be empty")
        task_ids.append(sample_id)
    return task_ids


def _target_section_count(required_length_words: int) -> int:
    """Choose a review-like section count from the target length."""

    if required_length_words <= 3000:
        return 4
    if required_length_words <= 7000:
        return 5
    if required_length_words <= 12000:
        return 6
    return 7


def _section_word_budgets(required_length_words: int, section_count: int) -> List[int]:
    """Distribute an approximate word budget across sections."""

    if section_count <= 0:
        raise ValueError("section_count must be positive")

    base_budget = required_length_words // section_count
    remainder = required_length_words % section_count
    return [
        base_budget + (1 if section_index < remainder else 0)
        for section_index in range(section_count)
    ]


def _with_budget(title: str, word_budget: int) -> str:
    return f"{title} (target about {word_budget} words)"


def _build_outline(
    task_type: str,
    must_include: List[str],
    required_length_words: int,
) -> Dict[str, str]:
    """Build a review-like outline with explicit section word budgets."""
    organizer_candidates = [
        item for item in ORGANIZER_CANDIDATES_EN if item in must_include
    ]
    closing_candidates = [
        item for item in CLOSING_CANDIDATES_EN if item in must_include
    ]
    organizer_label = (
        organizer_candidates[0] if organizer_candidates else "core organizing axis"
    )
    closing_label = closing_candidates[0] if closing_candidates else "future work"
    focus_label = must_include[4] if len(must_include) > 4 else "core evidence"
    context_label = must_include[5] if len(must_include) > 5 else "practice context"
    evidence_label = must_include[6] if len(must_include) > 6 else "evidence integration"

    if task_type == "analysis":
        section_titles = [
            "Scope, terminology, and review boundaries",
            f"Background and organizing framework around the {organizer_label}",
            f"Core evidence on {focus_label} and {evidence_label}",
            f"Comparative synthesis for the {context_label} context",
            f"Limitations, {closing_label}, and closing judgment",
        ]
    else:
        section_titles = [
            "Background, problem framing, and review scope",
            f"Organizing framework and evidence base for the {organizer_label}",
            f"Comparative discussion of {focus_label} and {evidence_label}",
            f"Implementation and practice implications for {context_label}",
            f"Limitations, {closing_label}, and forward-looking discussion",
        ]

    section_count = _target_section_count(required_length_words)
    if section_count >= 6:
        section_titles.insert(
            3,
            f"Methodological contrasts and evidence conflicts around the {organizer_label}",
        )
    if section_count >= 7:
        section_titles.insert(
            -1,
            f"Open questions and research agenda for {context_label}",
        )

    if section_count < len(section_titles):
        section_titles = section_titles[: section_count - 1] + [section_titles[-1]]
    else:
        section_titles = section_titles[:section_count]
    budgets = _section_word_budgets(required_length_words, len(section_titles))
    return {
        f"sec{index + 1}": _with_budget(title, budgets[index])
        for index, title in enumerate(section_titles)
    }


def _normalize_reference(reference: Union[str, Dict[str, object]]) -> Dict[str, object]:
    """Normalize a structured reference payload or its JSON serialization."""
    if isinstance(reference, dict):
        return reference
    if isinstance(reference, str):
        parsed_reference = json.loads(reference)
        if not isinstance(parsed_reference, dict):
            raise ValueError("reference JSON must decode to a dictionary")
        return parsed_reference
    raise TypeError("reference must be a dict or a JSON string")


def _contains_keyword(text: str, keyword: str) -> bool:
    """Return whether a keyword appears in a case-insensitive match."""
    return contains_keyword(text, keyword)


def _parse_int_field(raw_value: object, field_name: str) -> int:
    """Parse an integer field and reject non-integral types early."""
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} must be an integer or an integer string")
    return int(raw_value)


def _normalize_keyword_specs(raw_items: object, field_name: str) -> List[Dict[str, object]]:
    """Normalize keyword constraint specs into dictionaries."""
    if not isinstance(raw_items, list):
        raise TypeError(f"{field_name} must be a list")
    return [
        dict(item) if isinstance(item, dict) else {"keyword": str(item)}
        for item in raw_items
    ]


def _validate_english_sample(sample_row: Dict[str, object]) -> None:
    """Guard against accidentally reintroducing Chinese benchmark assets."""
    serialized = json.dumps(sample_row, ensure_ascii=False)
    if _contains_cjk(serialized):
        sample_id = str(sample_row.get("sample_id", "unknown"))
        raise ValueError(
            f"Benchmark sample {sample_id} still contains Chinese text. "
            "Regenerate the benchmark assets in English before running."
        )


def load_benchmark_task(task_id: str) -> Dict[str, object]:
    """Load one benchmark sample and adapt it into a MetaWriter task config."""
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        if str(sample_row.get("sample_id")) != task_id:
            continue

        _validate_english_sample(sample_row)

        prompt = str(sample_row["prompt"]).strip()
        task_type = str(sample_row["task_type"])

        raw_constraints_object = sample_row["constraints"]
        if not isinstance(raw_constraints_object, dict):
            raise TypeError("constraints must be a dictionary")
        raw_constraints = cast(Dict[str, object], raw_constraints_object)

        must_include_object = raw_constraints["must_include"]
        if not isinstance(must_include_object, list):
            raise TypeError("constraints.must_include must be a list")
        must_include = [str(item) for item in must_include_object]

        periodic_requirements_object = raw_constraints["periodic_requirements"]
        if not isinstance(periodic_requirements_object, list):
            raise TypeError("constraints.periodic_requirements must be a list")
        periodic_requirements = [str(item) for item in periodic_requirements_object]

        required_length_words = _parse_int_field(
            raw_constraints["required_length_words"],
            "constraints.required_length_words",
        )
        expected_blocks = _parse_int_field(
            raw_constraints["expected_blocks"],
            "constraints.expected_blocks",
        )
        range_keywords = _normalize_keyword_specs(
            raw_constraints["range_keywords"],
            "constraints.range_keywords",
        )
        periodic_keywords = _normalize_keyword_specs(
            raw_constraints["periodic_keywords"],
            "constraints.periodic_keywords",
        )
        once_keywords = [str(item) for item in must_include]
        if "once_keywords" in raw_constraints:
            once_keywords_object = raw_constraints["once_keywords"]
            if not isinstance(once_keywords_object, list):
                raise TypeError("constraints.once_keywords must be a list")
            once_keywords = [str(item) for item in once_keywords_object]

        proxy_questions_object = sample_row["proxy_questions"]
        if not isinstance(proxy_questions_object, list):
            raise TypeError("proxy_questions must be a list")
        checklist_object = sample_row["checklist"]
        if not isinstance(checklist_object, list):
            raise TypeError("checklist must be a list")
        proxy_questions_list = cast(List[object], proxy_questions_object)
        checklist_list = cast(List[object], checklist_object)

        constraints: List[str] = [
            _mark_document_level_constraint("Write the entire article in English."),
            _mark_document_level_constraint(
                f"Target length is about {required_length_words} words."
            ),
            _mark_document_level_constraint(
                f"The body should contain at least {expected_blocks} natural paragraphs and read like a long-form review rather than a brief or outline."
            ),
            _mark_document_level_constraint(
                "Use the approximate word budget in each section title to distribute the target length across the article."
            ),
            _mark_document_level_constraint(
                "Use a survey-paper style: define the scope first, then organize, compare, and synthesize the evidence before closing with limitations and future work."
            ),
            *[
                _mark_document_level_constraint(f"Must explicitly cover: {item}.")
                for item in must_include
            ],
            *[
                _mark_document_level_constraint(f"Periodic requirement: {item}")
                for item in periodic_requirements
            ],
        ]

        reference: Dict[str, object] = {
            "sample_id": str(sample_row["sample_id"]),
            "task_type": task_type,
            "language": "en",
            "prompt": prompt,
            "constraints": {
                "required_length_words": required_length_words,
                "must_include": must_include,
                "once_keywords": once_keywords,
                "periodic_requirements": periodic_requirements,
                "expected_blocks": expected_blocks,
                "range_keywords": range_keywords,
                "periodic_keywords": periodic_keywords,
            },
            "proxy_questions": [
                {
                    "qid": str(question["qid"]),
                    "question": str(question["question"]),
                    "answer": str(question["answer"]),
                }
                for question in proxy_questions_list
                if isinstance(question, dict)
            ],
            "checklist": [str(item) for item in checklist_list],
        }

        return {
            "task": prompt,
            "constraints": constraints,
            "outline": _build_outline(task_type, must_include, required_length_words),
            "reference": reference,
        }

    raise ValueError(f"Unknown benchmark task: {task_id}")


def evaluate_output(
    generated_text: str, reference: Union[str, Dict[str, object]]
) -> Dict[str, object]:
    """Evaluate a generated benchmark answer with lightweight local heuristics."""
    normalized_reference = _normalize_reference(reference)
    normalized_text = generated_text.strip()
    if normalized_text == "":
        raise ValueError("generated_text must not be empty")

    raw_constraints_object = normalized_reference["constraints"]
    if not isinstance(raw_constraints_object, dict):
        raise TypeError("reference.constraints must be a dictionary")
    raw_constraints = cast(Dict[str, object], raw_constraints_object)

    must_include_object = raw_constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("reference.constraints.must_include must be a list")
    must_include = [str(item) for item in must_include_object]
    once_keywords_object = raw_constraints.get("once_keywords", must_include)
    if not isinstance(once_keywords_object, list):
        raise TypeError("reference.constraints.once_keywords must be a list")
    once_keywords = [str(item) for item in once_keywords_object]

    required_length_words = _parse_int_field(
        raw_constraints["required_length_words"],
        "reference.constraints.required_length_words",
    )

    expected_blocks = _parse_int_field(
        raw_constraints["expected_blocks"],
        "reference.constraints.expected_blocks",
    )

    range_keywords = _normalize_keyword_specs(
        raw_constraints["range_keywords"],
        "reference.constraints.range_keywords",
    )
    periodic_keywords = _normalize_keyword_specs(
        raw_constraints["periodic_keywords"],
        "reference.constraints.periodic_keywords",
    )

    checklist_object = normalized_reference["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("reference.checklist must be a list")
    checklist_list = cast(List[object], checklist_object)
    checklist = [str(item) for item in checklist_list]

    proxy_questions_object = normalized_reference["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("reference.proxy_questions must be a list")
    proxy_questions_list = cast(List[object], proxy_questions_object)
    proxy_questions = [
        dict(item) for item in proxy_questions_list if isinstance(item, dict)
    ]

    if len(must_include) == 0:
        raise ValueError("reference.constraints.must_include must not be empty")
    if len(proxy_questions) == 0:
        raise ValueError("reference.proxy_questions must not be empty")

    paragraph_blocks = _extract_paragraph_blocks(normalized_text)
    body_text = "\n\n".join(paragraph_blocks)
    sentence_parts = re.split(r"[.!?]+", body_text)
    sentence_count = sum(1 for part in sentence_parts if part.strip() != "")

    matched_keywords = [
        item for item in must_include if _contains_keyword(normalized_text, item)
    ]
    matched_proxy_answers = [
        str(item["qid"])
        for item in proxy_questions
        if _contains_keyword(normalized_text, str(item["answer"]))
    ]

    range_keyword_hits, range_keyword_global_fallback_hits, missing_range_keywords = (
        evaluate_range_keywords(paragraph_blocks, range_keywords)
    )
    periodic_keyword_hits, periodic_keyword_partial_hits, missing_periodic_keywords = (
        evaluate_periodic_keywords(paragraph_blocks, periodic_keywords)
    )

    response_word_count = count_length_units(normalized_text)
    length_ratio = response_word_count / required_length_words
    length_score = compute_s_length(required_length_words, normalized_text)
    completion_rate = compute_completion_rate(paragraph_blocks, expected_blocks)
    once_signal = compute_acc_once(normalized_text, once_keywords)
    range_signal = compute_acc_range(paragraph_blocks, range_keywords)
    periodic_signal = compute_acc_periodic(paragraph_blocks, periodic_keywords)
    proxy_hit_count, proxy_total = compute_proxy_qa(normalized_text, proxy_questions)
    instruction_hits, instruction_total = compute_instruction_hits(
        normalized_text,
        must_include,
    )
    checklist_hit_count, checklist_total = compute_soft_instruction_hits(
        normalized_text,
        checklist,
    )
    syntax_pass_rate, schema_pass_rate = compute_structure_scores(
        normalized_text,
        drop_markdown_wrappers=True,
    )

    raw_keyword_coverage = instruction_hits / instruction_total
    proxy_hit_rate = proxy_hit_count / proxy_total
    checklist_signal = (
        checklist_hit_count / checklist_total if checklist_total > 0 else raw_keyword_coverage
    )
    semantic_coverage_score = min(
        1.0,
        0.2 * raw_keyword_coverage
        + 0.3 * proxy_hit_rate
        + 0.25 * checklist_signal
        + 0.15 * range_signal
        + 0.1 * periodic_signal,
    )
    entity_consistency_score = max(raw_keyword_coverage, semantic_coverage_score)
    structure_signal = min(
        1.0,
        0.2 * completion_rate
        + 0.15 * once_signal
        + 0.15 * range_signal
        + 0.15 * periodic_signal
        + 0.15 * syntax_pass_rate
        + 0.2 * schema_pass_rate,
    )
    logical_coherence = min(
        1.0,
        0.3 * structure_signal
        + 0.25 * proxy_hit_rate
        + 0.2 * semantic_coverage_score
        + 0.25 * length_score,
    )
    semantic_violation_rate = (1.0 - semantic_coverage_score) ** 1.25
    length_violation_rate = 1.0 - length_score
    constraint_violation_rate = max(semantic_violation_rate, length_violation_rate)

    return {
        "constraint_violation_rate": constraint_violation_rate,
        "entity_consistency_score": entity_consistency_score,
        "logical_coherence": logical_coherence,
        "length_score": length_score,
        "diagnostics": {
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                item for item in must_include if item not in matched_keywords
            ],
            "matched_proxy_question_ids": matched_proxy_answers,
            "response_word_count": response_word_count,
            "required_length_words": required_length_words,
            "length_ratio": length_ratio,
            "length_within_tolerance": 0.8 <= length_ratio <= 1.2,
            "paragraph_count": len(paragraph_blocks),
            "sentence_count": sentence_count,
            "expected_blocks": expected_blocks,
            "completion_rate": completion_rate,
            "once_signal": once_signal,
            "raw_keyword_coverage": raw_keyword_coverage,
            "proxy_hit_rate": proxy_hit_rate,
            "semantic_coverage_score": semantic_coverage_score,
            "semantic_violation_rate": semantic_violation_rate,
            "length_violation_rate": length_violation_rate,
            "checklist_hit_count": checklist_hit_count,
            "checklist_total": checklist_total,
            "checklist_signal": checklist_signal,
            "range_signal": range_signal,
            "range_keyword_hits": range_keyword_hits,
            "range_keyword_global_fallback_hits": range_keyword_global_fallback_hits,
            "missing_range_keywords": missing_range_keywords,
            "periodic_signal": periodic_signal,
            "periodic_keyword_hits": periodic_keyword_hits,
            "periodic_keyword_partial_hits": periodic_keyword_partial_hits,
            "missing_periodic_keywords": missing_periodic_keywords,
            "proxy_hit_count": proxy_hit_count,
            "proxy_total": proxy_total,
            "syntax_pass_rate": syntax_pass_rate,
            "schema_pass_rate": schema_pass_rate,
            "structure_signal": structure_signal,
            "checklist": checklist,
        },
    }


def build_benchmark_task_config(task_id: str) -> Dict[str, object]:
    """Build the benchmark task config registered into the main task registry."""
    benchmark_task = load_benchmark_task(task_id)
    constraints_object = benchmark_task["constraints"]
    if not isinstance(constraints_object, list):
        raise TypeError("benchmark_task.constraints must be a list")
    outline_object = benchmark_task["outline"]
    if not isinstance(outline_object, dict):
        raise TypeError("benchmark_task.outline must be a dictionary")
    reference_object = benchmark_task["reference"]
    if not isinstance(reference_object, dict):
        raise TypeError("benchmark_task.reference must be a dictionary")

    constraints_list = cast(List[object], constraints_object)
    outline_dict = cast(Dict[str, object], outline_object)
    reference_dict = cast(Dict[str, object], reference_object)

    return {
        "task": str(benchmark_task["task"]),
        "constraints": list(constraints_list),
        "outline": dict(outline_dict),
        "reference": dict(reference_dict),
        "language": "en",
        "session_name": f"metabench_{task_id}",
    }
