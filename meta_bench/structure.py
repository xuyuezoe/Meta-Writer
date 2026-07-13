"""Structure-dimension scores for the clean MetaBench evaluator."""

from __future__ import annotations

import re
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$", re.MULTILINE)
DEFAULT_REQUIRED_COMPLETION_SLOTS = ["introduction", "main_body", "conclusion"]
SMART_APOSTROPHES = "\u2018\u2019\u2032"

TITLE_HEADINGS = {"title"}
ABSTRACT_HEADINGS = {"abstract"}
KEYWORD_HEADINGS = {"keywords", "keyword"}
INTRODUCTION_HEADINGS = {"introduction", "background"}
REFERENCE_HEADINGS = {
    "reference",
    "references",
    "bibliography",
    "literature cited",
    "works cited",
}
CONCLUSION_HEADINGS = {
    "conclusion",
    "conclusions",
    "summary",
    "key messages",
    "implications",
    "interpretation",
    "conclusions and implications",
    "clinical implications",
    "future directions",
    "future direction",
    "future work",
    "research agenda",
    "future work and research agenda",
    "concluding remarks",
    "perspective",
    "perspectives",
}
LIMITATIONS_HEADINGS = {
    "limitations",
    "limitation",
    "limitations and evidence gaps",
    "limitations and future work",
    "limitations and future directions",
    "limitations and future research priorities",
    "limitations, heterogeneity, and future research priorities",
    "evidence gaps",
    "limitations and research agenda",
}
NON_BODY_HEADINGS = {
    *REFERENCE_HEADINGS,
    *ABSTRACT_HEADINGS,
    *KEYWORD_HEADINGS,
    "acknowledgement",
    "acknowledgements",
    "acknowledgment",
    "acknowledgments",
    "funding",
    "funding statement",
    "disclosure",
    "disclosures",
    "author contributions",
    "authors' contributions",
    "contributor information",
    "author information",
    "data availability",
    "data availability statement",
    "availability of data and materials",
    "conflict of interest",
    "conflicts of interest",
    "competing interests",
    "ethics statement",
    "institutional review board statement",
    "informed consent",
    "informed consent statement",
    "consent",
    "consent for publication",
    "supporting information",
    "supplementary material",
    "supplementary materials",
    "supplementary information",
    "generative ai statement",
    "ai statement",
    "publisher's note",
    "correction",
    "correction note",
    "author correction",
    "publisher correction",
    "erratum",
    "errata",
    "corrigendum",
    "corrigenda",
    "author response",
    "response to reviewers",
    "section",
}
FALLBACK_SIGNAL_TERMS = {
    "introduction": {
        "scope",
        "background",
        "define",
        "defined",
        "definition",
        "terminology",
        "context",
        "overview",
        "introduce",
        "introduction",
    },
    "main_body": {
        "evidence",
        "finding",
        "findings",
        "mechanism",
        "mechanisms",
        "compare",
        "comparison",
        "synthesize",
        "synthesis",
        "studies",
        "trial",
        "trials",
        "cohort",
        "analysis",
        "method",
        "methods",
        "results",
    },
    "conclusion": {
        "conclusion",
        "conclusions",
        "summary",
        "summarize",
        "overall",
        "taken together",
        "in summary",
        "future work",
        "future studies",
        "future research",
        "limitations",
        "implications",
        "further studies",
        "research agenda",
    },
    "limitations_gaps": {
        "limitations",
        "limitation",
        "future work",
        "future studies",
        "future research",
        "future directions",
        "research agenda",
        "research priorities",
        "evidence gaps",
        "gap",
        "gaps",
        "heterogeneity",
        "uncertainty",
        "open questions",
    },
}


@dataclass(frozen=True)
class LengthScoreResult:
    """Length-control score and diagnostics."""

    score: float
    response_word_count: int
    required_length_words: int
    length_ratio: float
    length_within_tolerance: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "length_adherence": self.score,
            "length_score": self.score,
            "response_word_count": self.response_word_count,
            "required_length_words": self.required_length_words,
            "length_ratio": self.length_ratio,
            "length_within_tolerance": self.length_within_tolerance,
        }


@dataclass(frozen=True)
class SectionPresence:
    """Parsed section presence record."""

    section_index: int
    heading: str
    word_count: int
    text_present: bool


@dataclass(frozen=True)
class MarkdownSection:
    """Raw Markdown section before functional classification."""

    heading: str
    normalized_heading: str
    body: str
    level: int
    is_title: bool = False


@dataclass(frozen=True)
class FunctionalSlot:
    """One functional section slot used by completion scoring."""

    slot: str
    filled: bool
    source: str
    heading: str
    word_count: int
    matched_signals: list[str]


@dataclass(frozen=True)
class CompletionRateResult:
    """Section-completion score and diagnostics."""

    score: float
    complete_section_count: int
    expected_section_count: int
    parsed_section_count: int
    sections: list[SectionPresence]
    required_slots: list[str]
    slots: list[FunctionalSlot]

    def to_dict(self) -> dict[str, object]:
        return {
            "completion_rate": self.score,
            "complete_section_count": self.complete_section_count,
            "expected_section_count": self.expected_section_count,
            "parsed_section_count": self.parsed_section_count,
            "required_slots": self.required_slots,
            "slots": [
                {
                    "slot": slot.slot,
                    "filled": slot.filled,
                    "source": slot.source,
                    "heading": slot.heading,
                    "word_count": slot.word_count,
                    "matched_signals": slot.matched_signals,
                }
                for slot in self.slots
            ],
            "sections": [
                {
                    "section_index": section.section_index,
                    "heading": section.heading,
                    "word_count": section.word_count,
                    "text_present": section.text_present,
                }
                for section in self.sections
            ],
        }


@dataclass(frozen=True)
class HeadingSoftRecallResult:
    """LiRA-style heading soft recall over generated and gold outlines."""

    score: float
    gold_headings: list[str]
    predicted_headings: list[str]
    gold_cardinality: float
    predicted_cardinality: float
    union_cardinality: float
    intersection_cardinality: float
    similarity_backend: str

    def to_dict(self) -> dict[str, object]:
        return {
            "heading_soft_recall": self.score,
            "outline_alignment_score": self.score,
            "gold_headings": self.gold_headings,
            "predicted_headings": self.predicted_headings,
            "gold_cardinality": self.gold_cardinality,
            "predicted_cardinality": self.predicted_cardinality,
            "union_cardinality": self.union_cardinality,
            "intersection_cardinality": self.intersection_cardinality,
            "similarity_backend": self.similarity_backend,
        }


def count_length_units(text: str) -> int:
    """Count English length units.

    Clean MetaBench tasks are English-only, so the length baseline counts
    English and numeric tokens only. Non-English characters are ignored.
    """

    stripped = text.strip()
    if not stripped:
        return 0

    return len(
        re.findall(r"[A-Za-z0-9]+(?:[-_/][A-Za-z0-9]+)*", stripped)
    )


def compute_length_score_from_ratio(length_ratio: float) -> float:
    """Map length ratio to a [0, 1] threshold-free adherence score.

    The new rule keeps length as a comparable metric without arbitrary bands:

        min(W / B, B / W)

    where W is generated body length and B is the task target.
    """

    if length_ratio < 0:
        raise ValueError("length_ratio must be non-negative")

    if length_ratio == 0:
        score = 0.0
    else:
        score = min(length_ratio, 1.0 / length_ratio)
    return max(0.0, min(1.0, score))


def score_length(final_text: str, required_length_words: int) -> LengthScoreResult:
    """Score whether ``final_text`` satisfies the required length."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if required_length_words <= 0:
        raise ValueError("required_length_words must be positive")

    response_word_count = count_length_units(final_text)
    if response_word_count <= 0:
        raise ValueError("final_text contains no countable length units")

    length_ratio = response_word_count / required_length_words
    score = compute_length_score_from_ratio(length_ratio)
    return LengthScoreResult(
        score=score,
        response_word_count=response_word_count,
        required_length_words=required_length_words,
        length_ratio=length_ratio,
        length_within_tolerance=score >= 0.8,
    )


def _token_counter(text: str) -> Counter[str]:
    tokens = re.findall(r"[a-z0-9]+", text.casefold())
    return Counter(tokens)


def _token_cosine_similarity(a: str, b: str) -> float:
    left = _token_counter(a)
    right = _token_counter(b)
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[token] * right[token] for token in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return max(0.0, min(1.0, numerator / (left_norm * right_norm)))


def _soft_cardinality(
    headings: Sequence[str],
    similarity_fn: Callable[[str, str], float],
) -> float:
    if not headings:
        return 0.0
    total = 0.0
    for heading in headings:
        similarity_sum = sum(similarity_fn(heading, other) for other in headings)
        if similarity_sum <= 0:
            similarity_sum = 1.0
        total += 1.0 / similarity_sum
    return total


def score_heading_soft_recall(
    predicted_headings: Sequence[str],
    gold_headings: Sequence[str],
    *,
    similarity_fn: Callable[[str, str], float] | None = None,
    similarity_backend: str = "token_cosine",
) -> HeadingSoftRecallResult:
    """Compute LiRA's heading soft recall formula.

    LiRA defines soft cardinality from pairwise heading similarities and scores
    recall as card(G intersection P) / card(G).
    """

    clean_gold = [str(heading).strip() for heading in gold_headings if str(heading).strip()]
    clean_predicted = [
        str(heading).strip() for heading in predicted_headings if str(heading).strip()
    ]
    if not clean_gold:
        raise ValueError("gold_headings must contain at least one heading")
    sim = similarity_fn or _token_cosine_similarity
    gold_cardinality = _soft_cardinality(clean_gold, sim)
    predicted_cardinality = _soft_cardinality(clean_predicted, sim)
    union_headings = list(dict.fromkeys([*clean_gold, *clean_predicted]))
    union_cardinality = _soft_cardinality(union_headings, sim)
    intersection_cardinality = max(
        0.0,
        min(gold_cardinality, gold_cardinality + predicted_cardinality - union_cardinality),
    )
    score = 0.0 if gold_cardinality <= 0 else intersection_cardinality / gold_cardinality
    return HeadingSoftRecallResult(
        score=round(max(0.0, min(1.0, score)), 4),
        gold_headings=clean_gold,
        predicted_headings=clean_predicted,
        gold_cardinality=round(gold_cardinality, 4),
        predicted_cardinality=round(predicted_cardinality, 4),
        union_cardinality=round(union_cardinality, 4),
        intersection_cardinality=round(intersection_cardinality, 4),
        similarity_backend=similarity_backend,
    )


def normalize_heading(heading: str) -> str:
    """Normalize a Markdown heading for structural classification."""

    value = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", heading.strip())
    for char in SMART_APOSTROPHES:
        value = value.replace(char, "'")
    return re.sub(r"\s+", " ", value).casefold()


def parse_markdown_sections(final_text: str) -> list[MarkdownSection]:
    """Parse raw Markdown sections, preserving title and metadata sections."""

    normalized_text = final_text.replace("\r\n", "\n")
    matches = list(HEADING_RE.finditer(normalized_text))
    if not matches:
        return []

    has_subheadings = any(len(match.group(1)) >= 2 for match in matches)
    sections: list[MarkdownSection] = []
    for index, match in enumerate(matches):
        marker = match.group(1)
        heading = match.group(2).strip()
        normalized_heading = normalize_heading(heading)
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_text)
        body = normalized_text[body_start:body_end].strip()

        is_title = has_subheadings and len(marker) == 1 and index == 0
        sections.append(
            MarkdownSection(
                heading=heading,
                normalized_heading=normalized_heading,
                body=body,
                level=len(marker),
                is_title=is_title,
            )
        )
    return sections


def parse_content_sections(final_text: str) -> list[SectionPresence]:
    """Parse non-metadata content sections from Markdown headings."""

    parsed_sections = parse_markdown_sections(final_text)
    sections: list[SectionPresence] = []
    for section in parsed_sections:
        if section.is_title:
            continue
        if _slot_for_heading(section.normalized_heading) in {
            "abstract",
            "keywords",
            "references",
            "metadata",
        }:
            continue
        word_count = count_length_units(section.body)
        sections.append(
            SectionPresence(
                section_index=len(sections) + 1,
                heading=section.heading,
                word_count=word_count,
                text_present=word_count > 0,
            )
        )
    return sections


def _slot_for_heading(normalized_heading: str) -> str:
    if normalized_heading in ABSTRACT_HEADINGS:
        return "abstract"
    if normalized_heading in KEYWORD_HEADINGS:
        return "keywords"
    if normalized_heading in INTRODUCTION_HEADINGS:
        return "introduction"
    if normalized_heading in LIMITATIONS_HEADINGS:
        return "limitations_gaps"
    if normalized_heading in CONCLUSION_HEADINGS:
        return "conclusion"
    if any(token in normalized_heading for token in ("introduction", "background", "scope", "terminology", "context")):
        return "introduction"
    if any(
        token in normalized_heading
        for token in (
            "limitations",
            "evidence gap",
            "evidence gaps",
            "future research priorities",
            "heterogeneity",
        )
    ):
        return "limitations_gaps"
    if any(token in normalized_heading for token in ("conclusion", "summary", "future work", "future direction", "research agenda", "concluding")):
        return "conclusion"
    if normalized_heading in REFERENCE_HEADINGS:
        return "references"
    if normalized_heading in NON_BODY_HEADINGS:
        return "metadata"
    return "main_body"


def _join_text(existing: str, new_text: str) -> str:
    if not new_text.strip():
        return existing
    if not existing.strip():
        return new_text.strip()
    return f"{existing.strip()}\n\n{new_text.strip()}"


def _paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]


def _matched_signals(text: str, slot: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).casefold()
    return sorted(
        signal
        for signal in FALLBACK_SIGNAL_TERMS.get(slot, set())
        if signal in normalized
    )


def _make_slot(
    slot: str,
    text: str,
    *,
    source: str,
    heading: str,
    matched_signals: list[str] | None = None,
) -> FunctionalSlot:
    word_count = count_length_units(text)
    return FunctionalSlot(
        slot=slot,
        filled=word_count > 0,
        source=source,
        heading=heading,
        word_count=word_count,
        matched_signals=list(matched_signals or []),
    )


def _empty_slot(slot: str) -> FunctionalSlot:
    return FunctionalSlot(
        slot=slot,
        filled=False,
        source="missing",
        heading="",
        word_count=0,
        matched_signals=[],
    )


def _candidate_slot(
    slot: str,
    text: str,
    *,
    source: str,
    heading: str,
    min_words: int,
) -> FunctionalSlot | None:
    signals = _matched_signals(text, slot)
    word_count = count_length_units(text)
    if word_count >= min_words and signals:
        return FunctionalSlot(
            slot=slot,
            filled=True,
            source=source,
            heading=heading,
            word_count=word_count,
            matched_signals=signals,
        )
    return None


def classify_functional_sections(final_text: str) -> dict[str, FunctionalSlot]:
    """Classify text into functional article slots with limited fallbacks."""

    parsed_sections = parse_markdown_sections(final_text)
    slot_texts: dict[str, str] = {
        "title": "",
        "abstract": "",
        "keywords": "",
        "introduction": "",
        "main_body": "",
        "conclusion": "",
        "limitations_gaps": "",
        "references": "",
    }
    slot_sources: dict[str, str] = {}
    slot_headings: dict[str, str] = {}

    content_sections: list[MarkdownSection] = []
    for section in parsed_sections:
        if section.is_title:
            slot_texts["title"] = section.heading
            slot_sources["title"] = "heading"
            slot_headings["title"] = section.heading
            continue

        slot = _slot_for_heading(section.normalized_heading)
        if slot == "metadata":
            continue
        slot_texts[slot] = _join_text(slot_texts.get(slot, ""), section.body)
        slot_sources.setdefault(slot, "explicit_heading")
        slot_headings.setdefault(slot, section.heading)
        if slot not in {"abstract", "keywords", "references"}:
            content_sections.append(section)

    if not slot_texts["introduction"].strip() and content_sections:
        first_section = content_sections[0]
        first_paragraphs = _paragraphs(first_section.body)
        candidate_text = first_paragraphs[0] if first_paragraphs else first_section.body
        candidate = _candidate_slot(
            "introduction",
            candidate_text,
            source="fallback_first_paragraph",
            heading=first_section.heading,
            min_words=20,
        )
        if candidate is not None:
            slot_texts["introduction"] = candidate_text
            slot_sources["introduction"] = candidate.source
            slot_headings["introduction"] = candidate.heading

    if not slot_texts["main_body"].strip():
        candidate_parts: list[str] = []
        for section in content_sections:
            if _slot_for_heading(section.normalized_heading) in {"introduction", "conclusion", "limitations_gaps"}:
                paragraphs = _paragraphs(section.body)
                if len(paragraphs) > 2:
                    candidate_parts.extend(paragraphs[1:-1])
                continue
            candidate_parts.append(section.body)
        candidate_text = "\n\n".join(part for part in candidate_parts if part.strip())
        candidate = _candidate_slot(
            "main_body",
            candidate_text,
            source="fallback_middle_paragraphs",
            heading="",
            min_words=40,
        )
        if candidate is not None:
            slot_texts["main_body"] = candidate_text
            slot_sources["main_body"] = candidate.source
            slot_headings["main_body"] = candidate.heading

    if not slot_texts["conclusion"].strip() and content_sections:
        tail_section = content_sections[-1]
        tail_paragraphs = _paragraphs(tail_section.body)
        candidate_text = tail_paragraphs[-1] if tail_paragraphs else tail_section.body
        candidate = _candidate_slot(
            "conclusion",
            candidate_text,
            source="fallback_tail_paragraph",
            heading=tail_section.heading,
            min_words=20,
        )
        if candidate is not None:
            slot_texts["conclusion"] = candidate_text
            slot_sources["conclusion"] = candidate.source
            slot_headings["conclusion"] = candidate.heading

    if not slot_texts["limitations_gaps"].strip():
        for section in reversed(content_sections):
            if _slot_for_heading(section.normalized_heading) == "limitations_gaps":
                slot_texts["limitations_gaps"] = section.body
                slot_sources["limitations_gaps"] = "explicit_heading"
                slot_headings["limitations_gaps"] = section.heading
                break

    if not slot_texts["limitations_gaps"].strip() and content_sections:
        tail_section = content_sections[-1]
        tail_paragraphs = _paragraphs(tail_section.body)
        candidate_text = tail_paragraphs[-1] if tail_paragraphs else tail_section.body
        candidate = _candidate_slot(
            "limitations_gaps",
            candidate_text,
            source="fallback_tail_paragraph",
            heading=tail_section.heading,
            min_words=20,
        )
        if candidate is not None:
            slot_texts["limitations_gaps"] = candidate_text
            slot_sources["limitations_gaps"] = candidate.source
            slot_headings["limitations_gaps"] = candidate.heading

    slots: dict[str, FunctionalSlot] = {}
    for slot, text in slot_texts.items():
        if text.strip():
            explicit_signals = _matched_signals(text, slot)
            slots[slot] = _make_slot(
                slot,
                text,
                source=slot_sources.get(slot, "explicit_heading"),
                heading=slot_headings.get(slot, ""),
                matched_signals=explicit_signals,
            )
        else:
            slots[slot] = _empty_slot(slot)
    return slots


def score_completion_rate(
    final_text: str,
    expected_section_count: int,
    required_slots: list[str] | None = None,
) -> CompletionRateResult:
    """Score whether required functional sections are present and non-empty."""

    if not isinstance(final_text, str) or final_text.strip() == "":
        raise ValueError("final_text must be a non-empty string")
    if expected_section_count <= 0:
        raise ValueError("expected_section_count must be positive")

    required_slot_names = required_slots or DEFAULT_REQUIRED_COMPLETION_SLOTS
    if not required_slot_names:
        raise ValueError("required_slots must not be empty")

    sections = parse_content_sections(final_text)
    slots_by_name = classify_functional_sections(final_text)
    scored_slots = [
        slots_by_name.get(slot_name, _empty_slot(slot_name))
        for slot_name in required_slot_names
    ]
    complete_section_count = sum(1 for slot in scored_slots if slot.filled)
    denominator = len(required_slot_names)
    score = min(1.0, complete_section_count / denominator)
    return CompletionRateResult(
        score=score,
        complete_section_count=complete_section_count,
        expected_section_count=denominator,
        parsed_section_count=len(sections),
        sections=sections,
        required_slots=required_slot_names,
        slots=scored_slots,
    )


def _extract_required_length_words(reference: Mapping[str, object]) -> int:
    constraints = reference.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TypeError("reference.constraints must be a mapping")
    raw_value = constraints.get("required_length_words")
    if not isinstance(raw_value, (int, str)):
        raise TypeError("reference.constraints.required_length_words must be an int or string")
    return int(raw_value)


def _extract_expected_section_count(reference: Mapping[str, object]) -> int:
    constraints = reference.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TypeError("reference.constraints must be a mapping")
    raw_value = constraints.get("expected_sections", constraints.get("expected_blocks"))
    if not isinstance(raw_value, (int, str)):
        raise TypeError(
            "reference.constraints.expected_sections must be an int or string"
        )
    return int(raw_value)


def _extract_required_completion_slots(reference: Mapping[str, object]) -> list[str]:
    constraints = reference.get("constraints")
    if not isinstance(constraints, Mapping):
        raise TypeError("reference.constraints must be a mapping")
    raw_slots = constraints.get("required_sections", constraints.get("required_slots"))
    if raw_slots is None:
        return list(DEFAULT_REQUIRED_COMPLETION_SLOTS)
    if not isinstance(raw_slots, list):
        raise TypeError("reference.constraints.required_sections must be a list")
    slots = [str(slot).strip() for slot in raw_slots if str(slot).strip()]
    if not slots:
        raise ValueError("reference.constraints.required_sections must not be empty")
    return slots


def _extract_gold_outline_headings(reference: Mapping[str, object]) -> list[str]:
    raw_outline = reference.get("outline")
    if isinstance(raw_outline, Mapping):
        headings = [str(value).strip() for value in raw_outline.values() if str(value).strip()]
        if headings:
            return headings
    constraints = reference.get("constraints")
    if isinstance(constraints, Mapping):
        raw_sections = constraints.get("required_section_headings")
        if isinstance(raw_sections, list):
            headings = [str(value).strip() for value in raw_sections if str(value).strip()]
            if headings:
                return headings
        raw_roles = constraints.get("section_roles")
        if isinstance(raw_roles, Mapping):
            headings = [str(value).strip() for value in raw_roles.values() if str(value).strip()]
            if headings:
                return headings
    return _extract_required_completion_slots(reference)


def _extract_predicted_headings(final_text: str) -> list[str]:
    headings = [section.heading for section in parse_content_sections(final_text)]
    if headings:
        return headings
    slots = classify_functional_sections(final_text)
    return [
        slot.heading or slot.slot.replace("_", " ")
        for slot in slots.values()
        if slot.filled and slot.slot not in {"title", "abstract", "keywords", "references"}
    ]


def evaluate_structure_dimension(
    final_text: str,
    reference: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the structure dimension currently implemented in ``meta_bench``."""

    length_result = score_length(
        final_text=final_text,
        required_length_words=_extract_required_length_words(reference),
    )
    completion_result = score_completion_rate(
        final_text=final_text,
        expected_section_count=_extract_expected_section_count(reference),
        required_slots=_extract_required_completion_slots(reference),
    )
    heading_result = score_heading_soft_recall(
        predicted_headings=_extract_predicted_headings(final_text),
        gold_headings=_extract_gold_outline_headings(reference),
    )
    return {
        "dimension": "structure",
        "scores": {
            "length_adherence": length_result.score,
            "heading_soft_recall": heading_result.score,
        },
        "diagnostics": {
            "length": length_result.to_dict(),
            "heading_soft_recall": heading_result.to_dict(),
            "completion_legacy": completion_result.to_dict(),
        },
    }
