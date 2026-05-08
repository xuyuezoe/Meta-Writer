"""Shared local metrics for MetaBench-style long-form evaluation."""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

_CN_IGNORED_TOKENS = {
    "是否",
    "覆盖",
    "包含",
    "给出",
    "有",
    "方案",
    "方面",
    "内容",
}

_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "article",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "have",
    "in",
    "into",
    "is",
    "it",
    "must",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "their",
    "this",
    "to",
    "use",
    "with",
}


def split_blocks(
    response_text: str,
    *,
    drop_markdown_wrappers: bool = False,
) -> List[str]:
    """Split text into paragraph-like blocks."""

    normalized_text = response_text.replace("\r\n", "\n").strip()
    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", normalized_text)
        if block.strip() != ""
    ]

    if drop_markdown_wrappers:
        blocks = [
            block
            for block in blocks
            if block != "---" and re.match(r"^#{1,6}\s+", block) is None
        ]

    if len(blocks) == 0:
        return [normalized_text]
    return blocks


def contains_keyword(text: str, keyword: str) -> bool:
    """Return whether keyword appears in text with a case-insensitive match."""

    return keyword.lower() in text.lower()


def count_length_units(text: str) -> int:
    """Count CJK characters plus Latin tokens for mixed-language length checks."""

    stripped_text = text.strip()
    if stripped_text == "":
        return 0

    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", stripped_text))
    latin_token_count = len(
        re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*", stripped_text)
    )
    return cjk_count + latin_token_count


def compute_completion_rate(blocks: List[str], expected_blocks: int) -> float:
    """Compute paragraph completion rate against expected block count."""

    if expected_blocks <= 0:
        raise ValueError("expected_blocks must be positive")
    return min(1.0, len(blocks) / expected_blocks)


def compute_acc_once(response_text: str, once_keywords: List[str]) -> float:
    """Compute once-keyword hit rate."""

    if len(once_keywords) == 0:
        return 1.0

    hits = sum(1 for keyword in once_keywords if contains_keyword(response_text, keyword))
    return hits / len(once_keywords)


def compute_acc_range(blocks: List[str], range_specs: List[Dict[str, object]]) -> float:
    """Compute ranged keyword hit rate."""

    if len(range_specs) == 0:
        return 1.0

    hits = 0
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = int(spec["start"])
        end_index = int(spec["end"])
        if start_index <= 0 or end_index <= 0 or end_index < start_index:
            raise ValueError(f"Invalid range spec: {spec}")

        target_blocks = blocks[start_index - 1 : end_index]
        if contains_keyword("\n".join(target_blocks), keyword):
            hits += 1
    return hits / len(range_specs)


def compute_acc_periodic(
    blocks: List[str],
    periodic_specs: List[Dict[str, object]],
) -> float:
    """Compute periodic keyword hit rate."""

    if len(periodic_specs) == 0:
        return 1.0

    per_spec_scores: List[float] = []
    for spec in periodic_specs:
        keyword = str(spec["keyword"])
        every = int(spec["every"])
        start_index = int(spec["start"])
        if every <= 0 or start_index <= 0:
            raise ValueError(f"Invalid periodic spec: {spec}")

        expected_positions: List[int] = []
        cursor = start_index
        while cursor <= len(blocks):
            expected_positions.append(cursor)
            cursor += every

        if len(expected_positions) == 0:
            per_spec_scores.append(1.0)
            continue

        hit_count = sum(
            1
            for position in expected_positions
            if contains_keyword(blocks[position - 1], keyword)
        )
        per_spec_scores.append(hit_count / len(expected_positions))

    return sum(per_spec_scores) / len(per_spec_scores)


def compute_proxy_qa(
    response_text: str,
    proxy_questions: List[Dict[str, object]],
) -> Tuple[int, int]:
    """Compute proxy-QA hits from answer keyword matches."""

    total_count = len(proxy_questions)
    if total_count == 0:
        raise ValueError("proxy_questions must not be empty")

    hit_count = 0
    for question in proxy_questions:
        answer = str(question["answer"]).strip()
        if answer != "" and contains_keyword(response_text, answer):
            hit_count += 1
    return hit_count, total_count


def compute_instruction_hits(
    response_text: str,
    must_include: List[str],
) -> Tuple[int, int]:
    """Compute hard instruction hits from must-include items."""

    instruction_total = len(must_include)
    if instruction_total == 0:
        raise ValueError("must_include must not be empty")

    instruction_hits = sum(
        1 for item in must_include if contains_keyword(response_text, item)
    )
    return instruction_hits, instruction_total


def extract_soft_keywords(check_item: str) -> List[str]:
    """Extract fallback soft keywords from a checklist item."""

    cjk_tokens = re.findall(r"[\u4e00-\u9fff]{2,}", check_item)
    cjk_tokens = [token for token in cjk_tokens if token not in _CN_IGNORED_TOKENS]
    if len(cjk_tokens) > 0:
        return cjk_tokens

    english_tokens = re.findall(
        r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*",
        check_item.lower(),
    )
    content_tokens = [
        token for token in english_tokens if token not in _EN_STOPWORDS and len(token) > 1
    ]
    if len(content_tokens) == 0 and check_item.strip() != "":
        return [check_item.strip()]

    keywords: List[str] = []
    seen: set[str] = set()
    max_width = min(3, len(content_tokens))
    for width in range(max_width, 0, -1):
        for index in range(len(content_tokens) - width + 1):
            phrase = " ".join(content_tokens[index : index + width])
            if phrase not in seen:
                keywords.append(phrase)
                seen.add(phrase)

    if len(keywords) == 0 and check_item.strip() != "":
        return [check_item.strip()]
    return keywords


def compute_soft_instruction_hits(
    response_text: str,
    checklist: List[str],
) -> Tuple[int, int]:
    """Compute soft instruction hits from the checklist."""

    soft_total = len(checklist)
    if soft_total == 0:
        return 0, 0

    soft_hits = 0
    for check_item in checklist:
        item_text = str(check_item).strip()
        if item_text == "":
            continue

        if contains_keyword(response_text, item_text):
            soft_hits += 1
            continue

        candidate_keywords = extract_soft_keywords(item_text)
        if any(contains_keyword(response_text, keyword) for keyword in candidate_keywords):
            soft_hits += 1

    return soft_hits, soft_total


def compute_structure_scores(
    response_text: str,
    *,
    drop_markdown_wrappers: bool = False,
) -> Tuple[float, float]:
    """Compute syntax and schema structure scores from generated text."""

    left_count = response_text.count("(") + response_text.count("[") + response_text.count("{")
    right_count = response_text.count(")") + response_text.count("]") + response_text.count("}")
    bracket_ok = 1.0 if left_count == right_count else 0.0

    json_extract_ok = 1.0
    json_candidates = re.findall(r"\{.*?\}", response_text, flags=re.DOTALL)
    if len(json_candidates) > 0:
        valid_json_count = 0
        for candidate in json_candidates:
            try:
                json.loads(candidate)
                valid_json_count += 1
            except json.JSONDecodeError:
                continue
        json_extract_ok = valid_json_count / len(json_candidates)

    syntax_pass_rate = 0.5 * bracket_ok + 0.5 * json_extract_ok

    blocks = split_blocks(
        response_text,
        drop_markdown_wrappers=drop_markdown_wrappers,
    )
    block_score = 1.0 if len(blocks) >= 3 else len(blocks) / 3.0
    avg_len = sum(len(block) for block in blocks) / len(blocks)
    avg_len_score = 1.0 if avg_len >= 60 else avg_len / 60.0
    schema_pass_rate = 0.5 * block_score + 0.5 * avg_len_score

    return syntax_pass_rate, schema_pass_rate
