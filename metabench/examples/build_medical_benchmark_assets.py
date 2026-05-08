"""Rebuild the local English benchmark assets.

The benchmark tasks in `samples.jsonl` are the canonical source. This script
validates that the task bundle is English-only, then deterministically rebuilds
the aligned demo outputs and metrics files so the local examples stay in sync.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

CURRENT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = CURRENT_DIR.parent.parent
SAMPLES_PATH: Path = CURRENT_DIR / "samples.jsonl"
OUTPUTS_PATH: Path = CURRENT_DIR / "outputs.jsonl"
METRICS_PATH: Path = CURRENT_DIR / "metrics.jsonl"
TARGET_SAMPLE_COUNT = 400

sys.path.insert(0, str(REPO_ROOT))

from examples.benchmark_template import evaluate_output

ORGANIZER_ANCHORS = [
    "classification framework",
    "clinical pathway",
    "controversy focus",
    "subgroup stratification",
    "implementation barriers",
    "evidence map",
]

CLOSING_ANCHORS = [
    "future work",
    "open questions",
    "evidence gaps",
    "research agenda",
]


def _write_jsonl(file_path: Path, rows: List[Dict[str, object]]) -> None:
    """Write UTF-8 JSONL rows back to disk."""
    file_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """Load JSONL rows from disk."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing benchmark asset file: {file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        rows.append(json.loads(stripped_line))
    return rows


def _contains_cjk(text: str) -> bool:
    """Return True when the text still contains Chinese characters."""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def _parse_int_value(raw_value: object, field_name: str) -> int:
    """Parse an integer-like value."""
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} must be an integer or an integer string")
    return int(raw_value)


def _extract_constraints(sample_row: Dict[str, object]) -> Dict[str, object]:
    """Return the validated constraint dictionary from a sample row."""
    constraints_object = sample_row["constraints"]
    if not isinstance(constraints_object, dict):
        raise TypeError("constraints must be a dictionary")
    return constraints_object


def _contains_keyword(text: str, keyword: str) -> bool:
    """Return whether a keyword appears in a case-insensitive match."""
    return keyword.lower() in text.lower()


def _choose_organizer_anchor(must_include: List[str]) -> str:
    """Return the organizer anchor from the required keyword list."""
    for organizer_anchor in ORGANIZER_ANCHORS:
        if organizer_anchor in must_include:
            return organizer_anchor
    return "core organizing axis"


def _choose_closing_anchor(must_include: List[str]) -> str:
    """Return a closing anchor that should appear near the end."""
    for closing_anchor in CLOSING_ANCHORS:
        if closing_anchor in must_include:
            return closing_anchor
    return "future work"


def _ensure_keywords(blocks: List[str], keywords: List[str]) -> List[str]:
    """Inject missing keywords across different paragraphs when needed."""
    updated_blocks = list(blocks)
    if not updated_blocks:
        return updated_blocks

    for keyword_index, keyword in enumerate(keywords):
        joined_text = "\n\n".join(updated_blocks)
        if _contains_keyword(joined_text, keyword):
            continue
        block_position = keyword_index % len(updated_blocks)
        updated_blocks[block_position] += (
            f" This paragraph explicitly revisits {keyword} as a required benchmark anchor."
        )
    return updated_blocks


def _ensure_range_keywords(
    blocks: List[str], range_specs: List[Dict[str, int | str]]
) -> List[str]:
    """Inject range-constrained keywords into allowed paragraph windows."""
    updated_blocks = list(blocks)
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = max(1, int(spec["start"])) - 1
        end_index = min(len(updated_blocks), int(spec["end"]))
        if end_index <= start_index:
            continue
        target_slice = updated_blocks[start_index:end_index]
        if any(_contains_keyword(block, keyword) for block in target_slice):
            continue
        updated_blocks[start_index] += (
            f" The discussion here intentionally places {keyword} in the expected section window."
        )
    return updated_blocks


def _ensure_periodic_keywords(
    blocks: List[str], periodic_specs: List[Dict[str, int | str]]
) -> List[str]:
    """Inject periodic keywords at their expected cadence."""
    updated_blocks = list(blocks)
    for spec in periodic_specs:
        keyword = str(spec["keyword"])
        every_value = int(spec["every"])
        start_index = max(1, int(spec["start"])) - 1
        if every_value <= 0:
            raise ValueError("periodic keyword every must be greater than zero")

        current_index = start_index
        while current_index < len(updated_blocks):
            if not _contains_keyword(updated_blocks[current_index], keyword):
                updated_blocks[current_index] += (
                    f" The analysis also returns to {keyword} at this scheduled interval."
                )
            current_index += every_value
    return updated_blocks


def _build_block_text(block_index: int, sample_row: Dict[str, object]) -> str:
    """Build one English paragraph for the demo output bundle."""
    constraints = _extract_constraints(sample_row)

    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include must be a list")
    must_include = [str(item) for item in must_include_object]

    expected_blocks = _parse_int_value(
        constraints["expected_blocks"], "constraints.expected_blocks"
    )
    organizer_anchor = _choose_organizer_anchor(must_include)
    closing_anchor = _choose_closing_anchor(must_include)

    domain_name = must_include[2] if len(must_include) > 2 else "medicine"
    subtopic = must_include[3] if len(must_include) > 3 else "the topic"
    focus_name = must_include[4] if len(must_include) > 4 else "the central focus"
    context_keyword = must_include[5] if len(must_include) > 5 else "the practice setting"
    evidence_keyword = must_include[6] if len(must_include) > 6 else "the evidence base"
    optional_items = [
        item
        for item in must_include
        if item
        not in {
            "scope",
            "limitations",
            "future work",
            domain_name,
            subtopic,
            focus_name,
            context_keyword,
            evidence_keyword,
            organizer_anchor,
        }
    ]

    style_index = (
        sum(ord(character) for character in str(sample_row["sample_id"]))
        + len(optional_items)
    ) % 4
    opening_variants = [
        "The opening paragraph",
        "The first section",
        "This introduction",
        "The review begins by",
    ]
    organizer_variants = [
        "The next paragraph establishes",
        "The structural section clarifies",
        "The article then adopts",
        "The organizational paragraph centers on",
    ]
    evidence_variants = [
        "It explicitly compares competing studies and synthesis pathways.",
        "It links multiple evidence streams rather than listing findings in isolation.",
        "It keeps trade-offs and synthesis judgments visible throughout the discussion.",
        "It contrasts alternative interpretations before integrating them into one argument.",
    ]
    limitation_variants = [
        "The penultimate section concentrates on limitations.",
        "The section before the ending turns to limitations.",
        "The closing analysis first addresses limitations.",
        "The review reserves a late section for limitations.",
    ]
    future_variants = [
        "The final section outlines future work.",
        "The ending paragraph turns to future work.",
        "The conclusion closes with future work.",
        "The review finishes with future work.",
    ]

    if block_index == 0:
        return (
            f"{opening_variants[style_index]} defines the scope of {subtopic} within {domain_name}, "
            f"sets the terminology, and limits the review to evidence directly relevant to {focus_name}, "
            f"{context_keyword}, and {evidence_keyword}."
        )
    if block_index == 1:
        return (
            f"{organizer_variants[style_index]} {organizer_anchor} as the main organizing axis, "
            f"so the literature is synthesized rather than presented as a chronological inventory."
        )
    if block_index < expected_blocks - 2:
        extra_anchor = (
            optional_items[(block_index + style_index) % len(optional_items)]
            if optional_items
            else organizer_anchor
        )
        return (
            f"Paragraph {block_index + 1} compares alternative lines of evidence around {focus_name}. "
            f"{evidence_variants[(block_index + style_index) % len(evidence_variants)]} "
            f"It connects {evidence_keyword}, {context_keyword}, and {extra_anchor} within the same analytical frame."
        )
    if block_index == expected_blocks - 2:
        return (
            f"{limitation_variants[style_index]} Current findings on {subtopic} remain constrained by heterogeneous cohorts, "
            f"uneven endpoint definitions, incomplete transferability to {context_keyword}, and unresolved evidence gaps, "
            f"so any strong conclusion still needs context-sensitive calibration."
        )
    return (
        f"{future_variants[style_index]} The next research steps should keep examining {focus_name}, "
        f"{evidence_keyword}, and cross-setting implementation in {context_keyword}, while also clarifying how "
        f"{closing_anchor} should shape the long-range agenda."
    )


def _build_output_row(sample_row: Dict[str, object], sample_index: int) -> Dict[str, object]:
    """Build one deterministic English demo output row."""
    constraints = _extract_constraints(sample_row)

    expected_blocks = _parse_int_value(
        constraints["expected_blocks"], "constraints.expected_blocks"
    )
    range_keywords_object = constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("range_keywords must be a list")
    periodic_keywords_object = constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("periodic_keywords must be a list")
    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include must be a list")

    response_blocks = [
        _build_block_text(block_index, sample_row)
        for block_index in range(expected_blocks)
    ]
    response_blocks = _ensure_keywords(
        response_blocks, [str(item) for item in must_include_object]
    )
    response_blocks = _ensure_range_keywords(
        response_blocks,
        [dict(item) for item in range_keywords_object if isinstance(item, dict)],
    )
    response_blocks = _ensure_periodic_keywords(
        response_blocks,
        [dict(item) for item in periodic_keywords_object if isinstance(item, dict)],
    )
    response_text = "\n\n".join(response_blocks)

    if _contains_cjk(response_text):
        raise ValueError(f"Generated demo output is not fully English for {sample_row['sample_id']}")

    usage_total_tokens = 900 + sample_index * 17
    latency_seconds = round(16.0 + (sample_index % 23) * 0.73, 2)
    return {
        "sample_id": str(sample_row["sample_id"]),
        "model_name": "demo-model",
        "response": response_text,
        "latency_seconds": latency_seconds,
        "usage_total_tokens": usage_total_tokens,
    }


def _compute_range_hit_rate(
    response_text: str, range_specs: List[Dict[str, int | str]]
) -> float:
    """Compute the range-keyword hit rate."""
    blocks = [block.strip() for block in response_text.split("\n\n") if block.strip() != ""]
    if len(range_specs) == 0:
        return 1.0

    hit_count = 0
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = max(1, int(spec["start"])) - 1
        end_index = min(len(blocks), int(spec["end"]))
        if any(_contains_keyword(block, keyword) for block in blocks[start_index:end_index]):
            hit_count += 1
    return hit_count / len(range_specs)


def _compute_periodic_hit_rate(
    response_text: str, periodic_specs: List[Dict[str, int | str]]
) -> float:
    """Compute the periodic-keyword hit rate."""
    blocks = [block.strip() for block in response_text.split("\n\n") if block.strip() != ""]
    if len(periodic_specs) == 0:
        return 1.0

    per_spec_scores: List[float] = []
    for spec in periodic_specs:
        keyword = str(spec["keyword"])
        every_value = int(spec["every"])
        start_index = max(1, int(spec["start"]))
        target_positions: List[int] = []
        current_position = start_index
        while current_position <= len(blocks):
            target_positions.append(current_position)
            current_position += every_value
        if len(target_positions) == 0:
            per_spec_scores.append(1.0)
            continue
        hit_count = 0
        for position in target_positions:
            if _contains_keyword(blocks[position - 1], keyword):
                hit_count += 1
        per_spec_scores.append(hit_count / len(target_positions))
    return sum(per_spec_scores) / len(per_spec_scores)


def _estimate_soft_hits(
    checklist: List[str], must_include: List[str], response_text: str
) -> int:
    """Estimate soft instruction hits by matching checklist items to present anchors."""
    hit_count = 0
    for item in checklist:
        if any(_contains_keyword(item, keyword) and _contains_keyword(response_text, keyword) for keyword in must_include):
            hit_count += 1
    return hit_count


def _build_metric_row(
    sample_row: Dict[str, object],
    output_row: Dict[str, object],
    sample_index: int,
) -> Dict[str, object]:
    """Build one deterministic metrics row aligned with the English assets."""
    constraints = _extract_constraints(sample_row)

    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include must be a list")
    checklist_object = sample_row["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("checklist must be a list")
    proxy_questions_object = sample_row["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("proxy_questions must be a list")
    range_keywords_object = constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("range_keywords must be a list")
    periodic_keywords_object = constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("periodic_keywords must be a list")

    must_include = [str(item) for item in must_include_object]
    checklist = [str(item) for item in checklist_object]
    response_text = str(output_row["response"])
    reference = {
        "constraints": dict(constraints),
        "proxy_questions": list(proxy_questions_object),
        "checklist": checklist,
    }
    evaluation = evaluate_output(response_text, reference)
    diagnostics = evaluation["diagnostics"]

    if not isinstance(diagnostics, dict):
        raise TypeError("evaluate_output diagnostics must be a dictionary")

    matched_keywords_object = diagnostics["matched_keywords"]
    if not isinstance(matched_keywords_object, list):
        raise TypeError("diagnostics.matched_keywords must be a list")
    matched_proxy_question_ids_object = diagnostics["matched_proxy_question_ids"]
    if not isinstance(matched_proxy_question_ids_object, list):
        raise TypeError("diagnostics.matched_proxy_question_ids must be a list")

    hard_hits = len(matched_keywords_object)
    soft_hits = _estimate_soft_hits(checklist, must_include, response_text)
    proxy_hits = len(matched_proxy_question_ids_object)

    quality_base = 4.05 + (sample_index % 7) * 0.09
    return {
        "sample_id": str(sample_row["sample_id"]),
        "completion_rate": 1.0,
        "acc_once": round(hard_hits / len(must_include), 4),
        "acc_range": round(
            _compute_range_hit_rate(
                response_text,
                [dict(item) for item in range_keywords_object if isinstance(item, dict)],
            ),
            4,
        ),
        "acc_periodic": round(
            _compute_periodic_hit_rate(
                response_text,
                [dict(item) for item in periodic_keywords_object if isinstance(item, dict)],
            ),
            4,
        ),
        "quality_scores": {
            "Accuracy": round(min(4.9, quality_base), 2),
            "Coherence": round(min(4.9, quality_base + 0.08), 2),
            "Clarity": round(min(4.9, quality_base + 0.04), 2),
            "ReadingExperience": round(min(4.9, quality_base + 0.1), 2),
        },
        "instruction_hits": hard_hits + soft_hits,
        "instruction_total": len(must_include) + len(checklist),
        "instruction_hard_hits": hard_hits,
        "instruction_hard_total": len(must_include),
        "instruction_soft_hits": soft_hits,
        "instruction_soft_total": len(checklist),
        "syntax_pass_rate": 1.0,
        "schema_pass_rate": 1.0,
        "proxy_qa_correct": proxy_hits,
        "proxy_qa_total": len(proxy_questions_object),
    }


def main() -> None:
    """Validate the English task bundle and rebuild aligned outputs and metrics."""
    sample_rows = _read_jsonl_rows(SAMPLES_PATH)
    if len(sample_rows) != TARGET_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_COUNT} benchmark samples, found {len(sample_rows)}"
        )

    for sample_row in sample_rows:
        serialized = json.dumps(sample_row, ensure_ascii=False)
        if _contains_cjk(serialized):
            raise ValueError(
                f"Benchmark sample {sample_row.get('sample_id')} still contains Chinese text"
            )

    output_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    for sample_index, sample_row in enumerate(sample_rows):
        output_row = _build_output_row(sample_row, sample_index)
        output_rows.append(output_row)
        metric_rows.append(_build_metric_row(sample_row, output_row, sample_index))

    if not (
        len(sample_rows) == len(output_rows) == len(metric_rows) == TARGET_SAMPLE_COUNT
    ):
        raise ValueError("Benchmark asset counts are inconsistent")

    _write_jsonl(SAMPLES_PATH, sample_rows)
    _write_jsonl(OUTPUTS_PATH, output_rows)
    _write_jsonl(METRICS_PATH, metric_rows)


if __name__ == "__main__":
    main()
