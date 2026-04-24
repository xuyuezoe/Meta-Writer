"""English benchmark adapter for MetaWriter.

This module loads benchmark samples from the local `metabench/examples/samples.jsonl`
bundle, converts them into task configs that MetaWriter can run directly, and
provides a lightweight local evaluator that does not depend on an external judge.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Union, cast


BENCHMARK_ROOT: Path = Path(__file__).resolve().parent.parent / "metabench"
SAMPLES_PATH: Path = BENCHMARK_ROOT / "examples" / "samples.jsonl"
DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "Document-level requirement: "

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

    paragraph_blocks: List[str] = []
    for raw_block in normalized_text.split("\n\n"):
        block = raw_block.strip()
        if block == "" or block == "---":
            continue
        if re.match(r"^#{1,6}\s+", block):
            continue
        paragraph_blocks.append(block)
    return paragraph_blocks


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


def _build_outline(task_type: str, must_include: List[str]) -> Dict[str, str]:
    """Build a compact three-section outline for the benchmark sample."""
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

    if task_type == "analysis":
        return {
            "sec1": "Scope, core concepts, and analytical perspective",
            "sec2": f"Comparative and synthetic analysis organized around the {organizer_label}",
            "sec3": f"Limitations, {closing_label}, and closing judgment",
        }

    return {
        "sec1": "Background, problem framing, and review scope",
        "sec2": f"Comparison, integration, and discussion under the {organizer_label}",
        "sec3": f"Reflections on limitations, {closing_label}, and forward-looking discussion",
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
    return keyword.lower() in text.lower()


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
            "outline": _build_outline(task_type, must_include),
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

    matched_keywords = [
        item for item in must_include if _contains_keyword(normalized_text, item)
    ]
    matched_proxy_answers = [
        str(item["qid"])
        for item in proxy_questions
        if _contains_keyword(normalized_text, str(item["answer"]))
    ]

    paragraph_blocks = _extract_paragraph_blocks(normalized_text)
    body_text = "\n\n".join(paragraph_blocks)
    sentence_parts = re.split(r"[.!?]+", body_text)
    sentence_count = sum(1 for part in sentence_parts if part.strip() != "")

    range_keyword_hits: List[str] = []
    missing_range_keywords: List[str] = []
    for item in range_keywords:
        keyword = str(item["keyword"])
        start_index = max(1, int(item["start"])) - 1
        end_index = min(len(paragraph_blocks), int(item["end"]))
        candidate_blocks = paragraph_blocks[start_index:end_index]
        if any(_contains_keyword(block, keyword) for block in candidate_blocks):
            range_keyword_hits.append(keyword)
        else:
            missing_range_keywords.append(keyword)

    periodic_keyword_hits: List[str] = []
    missing_periodic_keywords: List[str] = []
    for item in periodic_keywords:
        keyword = str(item["keyword"])
        every_value = int(item["every"])
        start_paragraph = max(1, int(item["start"]))
        if every_value <= 0:
            raise ValueError("periodic_keywords.every must be positive")

        target_hit_count = 0
        current_paragraph = start_paragraph
        while current_paragraph <= len(paragraph_blocks):
            target_hit_count += 1
            current_paragraph += every_value

        actual_hit_count = sum(
            1
            for block in paragraph_blocks[start_paragraph - 1 :]
            if _contains_keyword(block, keyword)
        )
        if actual_hit_count >= target_hit_count and target_hit_count > 0:
            periodic_keyword_hits.append(keyword)
        else:
            missing_periodic_keywords.append(keyword)

    entity_consistency_score = len(matched_keywords) / len(must_include)
    proxy_hit_rate = len(matched_proxy_answers) / len(proxy_questions)
    paragraph_signal = min(1.0, len(paragraph_blocks) / expected_blocks)
    sentence_signal = 1.0 if sentence_count >= expected_blocks * 2 else 0.5
    range_signal = (
        len(range_keyword_hits) / len(range_keywords) if len(range_keywords) > 0 else 1.0
    )
    periodic_signal = (
        len(periodic_keyword_hits) / len(periodic_keywords)
        if len(periodic_keywords) > 0
        else 1.0
    )
    structure_signal = min(
        1.0,
        0.35 * paragraph_signal
        + 0.2 * sentence_signal
        + 0.25 * range_signal
        + 0.2 * periodic_signal,
    )
    checklist_signal = len(matched_keywords) / max(len(checklist), len(must_include))
    logical_coherence = min(
        1.0, 0.5 * structure_signal + 0.3 * proxy_hit_rate + 0.2 * checklist_signal
    )
    constraint_violation_rate = 1.0 - entity_consistency_score

    return {
        "constraint_violation_rate": constraint_violation_rate,
        "entity_consistency_score": entity_consistency_score,
        "logical_coherence": logical_coherence,
        "diagnostics": {
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                item for item in must_include if item not in matched_keywords
            ],
            "matched_proxy_question_ids": matched_proxy_answers,
            "paragraph_count": len(paragraph_blocks),
            "sentence_count": sentence_count,
            "expected_blocks": expected_blocks,
            "range_keyword_hits": range_keyword_hits,
            "missing_range_keywords": missing_range_keywords,
            "periodic_keyword_hits": periodic_keyword_hits,
            "missing_periodic_keywords": missing_periodic_keywords,
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
