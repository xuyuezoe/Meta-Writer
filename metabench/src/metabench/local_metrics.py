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

_KEYWORD_ALIASES: Dict[str, List[str]] = {
    "scope": [
        "scope",
        "scoping",
        "discussion boundary",
        "discussion boundaries",
        "problem framing",
        "review boundary",
        "review boundaries",
    ],
    "future work": [
        "future work",
        "future research",
        "future directions",
        "future studies",
        "research agenda",
        "next steps",
    ],
    "limitations": [
        "limitations",
        "limits",
        "evidence gaps",
        "knowledge gaps",
        "data gaps",
        "uncertainties",
    ],
    "open questions": [
        "open questions",
        "unresolved questions",
        "unanswered questions",
        "research questions",
    ],
    "evidence gaps": [
        "evidence gaps",
        "knowledge gaps",
        "data gaps",
        "research gaps",
        "uncertainties",
    ],
    "comparison": [
        "comparison",
        "compare",
        "compared",
        "comparative",
        "contrasts",
        "trade-offs",
        "differences",
    ],
    "synthesis": [
        "synthesis",
        "synthesize",
        "synthesizes",
        "integrate",
        "integrates",
        "integration",
    ],
    "classification framework": [
        "classification framework",
        "classification scheme",
        "organizing framework",
        "taxonomy",
        "stratification framework",
    ],
    "clinical pathway": [
        "clinical pathway",
        "clinical pathways",
        "care pathway",
        "care pathways",
        "diagnostic pathway",
        "treatment pathway",
    ],
    "controversy focus": [
        "controversy focus",
        "controversy",
        "debate",
        "evidence conflict",
        "evidence conflicts",
        "unresolved question",
        "contested issue",
    ],
    "subgroup stratification": [
        "subgroup stratification",
        "stratification",
        "subgroup analysis",
        "subgroups",
    ],
    "implementation barriers": [
        "implementation barriers",
        "implementation barrier",
        "implementation challenges",
        "implementation challenge",
        "feasibility",
        "adoption barriers",
    ],
    "evidence map": [
        "evidence map",
        "evidence landscape",
        "body of evidence",
        "evidence base",
    ],
    "guideline differences": [
        "guideline differences",
        "guideline variation",
        "guideline variations",
        "guideline disagreement",
        "divergent guidelines",
    ],
    "risk benefit balance": [
        "risk-benefit balance",
        "risk benefit balance",
        "benefit-risk balance",
        "trade-off",
        "trade-offs",
    ],
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


def _normalize_keyword_text(text: str) -> str:
    """Normalize text for local keyword checks."""

    lowered = text.lower().replace("-", " ")
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", lowered).strip()


def _english_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", _normalize_keyword_text(text))


def _light_stem(token: str) -> str:
    """Use a tiny stemmer for common English inflection variants."""

    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ive"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _alias_candidates(keyword: str) -> List[str]:
    normalized_keyword = _normalize_keyword_text(keyword)
    aliases = _KEYWORD_ALIASES.get(normalized_keyword, [])
    candidates = [keyword, normalized_keyword, *aliases]

    seen: set[str] = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        normalized_candidate = _normalize_keyword_text(candidate)
        if normalized_candidate != "" and normalized_candidate not in seen:
            unique_candidates.append(normalized_candidate)
            seen.add(normalized_candidate)
    return unique_candidates


def _contains_stem_sequence(text_stems: List[str], alias_stems: List[str]) -> bool:
    if len(alias_stems) == 0:
        return False
    if len(alias_stems) == 1:
        return alias_stems[0] in set(text_stems)

    start_positions = [
        index for index, stem in enumerate(text_stems) if stem == alias_stems[0]
    ]
    for start in start_positions:
        cursor = start
        matched = True
        for expected_stem in alias_stems[1:]:
            next_cursor = None
            search_stop = min(len(text_stems), cursor + 5)
            for probe in range(cursor + 1, search_stop):
                if text_stems[probe] == expected_stem:
                    next_cursor = probe
                    break
            if next_cursor is None:
                matched = False
                break
            cursor = next_cursor
        if matched:
            return True
    return False


def contains_keyword(text: str, keyword: str) -> bool:
    """Return whether a keyword or close benchmark anchor appears in text."""

    normalized_text = _normalize_keyword_text(text)
    if normalized_text == "":
        return False

    for alias in _alias_candidates(keyword):
        if alias in normalized_text:
            return True

    text_stems = [_light_stem(token) for token in _english_tokens(text)]
    for alias in _alias_candidates(keyword):
        alias_stems = [
            _light_stem(token)
            for token in _english_tokens(alias)
            if token not in _EN_STOPWORDS
        ]
        if _contains_stem_sequence(text_stems, alias_stems):
            return True
        if len(alias_stems) >= 3:
            text_stem_set = set(text_stems)
            overlap = sum(1 for stem in set(alias_stems) if stem in text_stem_set)
            if overlap / len(set(alias_stems)) >= 0.66:
                return True
    return False


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
    """Compute ranged keyword hit rate with a global fallback."""

    if len(range_specs) == 0:
        return 1.0

    score = 0.0
    full_text = "\n".join(blocks)
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = int(spec["start"])
        end_index = int(spec["end"])
        if start_index <= 0 or end_index <= 0 or end_index < start_index:
            raise ValueError(f"Invalid range spec: {spec}")

        target_blocks = blocks[start_index - 1 : end_index]
        if contains_keyword("\n".join(target_blocks), keyword):
            score += 1.0
        elif contains_keyword(full_text, keyword):
            score += 0.6
    return score / len(range_specs)


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
        periodic_score = hit_count / len(expected_positions)
        tail_text = "\n".join(blocks[start_index - 1 :])
        if periodic_score == 0.0 and contains_keyword(tail_text, keyword):
            periodic_score = 0.5
        per_spec_scores.append(periodic_score)

    return sum(per_spec_scores) / len(per_spec_scores)


def evaluate_range_keywords(
    blocks: List[str],
    range_specs: List[Dict[str, object]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return window hits, global fallback hits, and missing ranged keywords."""

    window_hits: List[str] = []
    global_fallback_hits: List[str] = []
    missing: List[str] = []
    full_text = "\n".join(blocks)

    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = int(spec["start"])
        end_index = int(spec["end"])
        if start_index <= 0 or end_index <= 0 or end_index < start_index:
            raise ValueError(f"Invalid range spec: {spec}")

        target_blocks = blocks[start_index - 1 : end_index]
        if contains_keyword("\n".join(target_blocks), keyword):
            window_hits.append(keyword)
        elif contains_keyword(full_text, keyword):
            global_fallback_hits.append(keyword)
        else:
            missing.append(keyword)

    return window_hits, global_fallback_hits, missing


def evaluate_periodic_keywords(
    blocks: List[str],
    periodic_specs: List[Dict[str, object]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return strong hits, partial fallback hits, and missing periodic keywords."""

    strong_hits: List[str] = []
    partial_hits: List[str] = []
    missing: List[str] = []

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
            strong_hits.append(keyword)
            continue

        hit_count = sum(
            1
            for position in expected_positions
            if contains_keyword(blocks[position - 1], keyword)
        )
        if hit_count / len(expected_positions) >= 0.5:
            strong_hits.append(keyword)
            continue

        tail_text = "\n".join(blocks[start_index - 1 :])
        if contains_keyword(tail_text, keyword):
            partial_hits.append(keyword)
        else:
            missing.append(keyword)

    return strong_hits, partial_hits, missing


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
