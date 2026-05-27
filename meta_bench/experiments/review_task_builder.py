"""Build MetaBench task configs directly from real review articles."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from examples.tasks import TaskFactory
from meta_bench.schemas import TaskSpec
from meta_bench.six_slot_priors import derive_article_six_slot_word_counts
from meta_bench.structure import (
    ABSTRACT_HEADINGS,
    KEYWORD_HEADINGS,
    NON_BODY_HEADINGS,
    REFERENCE_HEADINGS,
    count_length_units,
    normalize_heading,
    parse_markdown_sections,
)
from meta_bench.task_generator import build_main_task_config
from src.references.loaders.markdown_loader import _parse_frontmatter


DEFAULT_CORPUS_DIR = "./data_sample/med_papers_review_augmented_strict"
EXPANDED_LENGTH_MODE = "expanded_random_long_form"
EXPANDED_BODY_TARGET_MIN_WORDS = 3500
EXPANDED_BODY_TARGET_MAX_WORDS = 8000
EXPANDED_BODY_TARGET_STEP_WORDS = 100
PUBLIC_REVIEW_CONSTRAINT_DROP_KEYS = (
    "length_mode",
    "source_body_target_words",
    "source_total_target_words",
    "sampled_body_target_words",
    "reference_target_words",
    "section_budget_trace",
    "six_slot_prior_version",
)
SMART_APOSTROPHES = "\u2018\u2019\u2032"
GENERIC_OUTLINE_HEADINGS = {
    "introduction",
    "background",
    "methods",
    "methodology",
    "materials and methods",
    "discussion",
    "results",
    "conclusion",
    "conclusions",
    "summary",
}
TITLE_PREFIX_PATTERNS = (
    r"^a\s+meta-analysis\s+of\s+",
    r"^an\s+meta-analysis\s+of\s+",
    r"^systematic\s+review\s+and\s+meta-analysis\s+of\s+",
    r"^a\s+systematic\s+review\s+and\s+meta-analysis\s+of\s+",
    r"^a\s+systematic\s+review\s+of\s+",
    r"^an\s+umbrella\s+review\s+of\s+",
    r"^a\s+scoping\s+review\s+of\s+",
)
TITLE_SUFFIX_PATTERNS = (
    r":\s*a\s+systematic\s+review\s+and\s+meta-analysis\s*$",
    r":\s*systematic\s+review\s+and\s+meta-analysis\s*$",
    r":\s*a\s+meta-analysis\s*$",
    r":\s*meta-analysis\s*$",
    r":\s*a\s+systematic\s+review\s*$",
    r":\s*systematic\s+review\s*$",
    r":\s*scoping\s+review\s*$",
)
GENERIC_FOCUS_POINT_PATTERNS = (
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "review",
    "discussion",
    "methods",
    "methodology",
    "results",
    "conclusion",
)
CANONICAL_SIX_SLOT_OUTLINE = {
    "sec1": "Scope, terminology, and practice context",
    "sec2": "Mechanistic and organizing framework",
    "sec3": "Evidence base, methods, and measurement strategy",
    "sec4": "Findings and cross-study synthesis",
    "sec5": "Clinical implications and interpretive discussion",
    "sec6": "Limitations, heterogeneity, and future research priorities",
}
SECTION_ID_TO_SLOT = {
    "sec1": "scope_context",
    "sec2": "framework_mechanism",
    "sec3": "evidence_methods",
    "sec4": "findings_synthesis",
    "sec5": "implications_discussion",
    "sec6": "limitations_future",
}


@dataclass(frozen=True)
class DerivedReviewTask:
    """Structured task artifacts derived from one review article."""

    paper_id: str
    source_article_path: str
    spec: TaskSpec
    outline: dict[str, str]
    section_word_targets: dict[str, int]
    config: dict[str, object]

    @property
    def task_name(self) -> str:
        return f"metabench_{self.paper_id.lower()}"

    @property
    def module_name(self) -> str:
        return self.task_name


def _normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_heading_key(text: str) -> str:
    value = normalize_heading(str(text))
    for char in SMART_APOSTROPHES:
        value = value.replace(char, "'")
    return _normalize_space(value)


def _frontmatter_keywords(raw_keywords: object) -> list[str]:
    if raw_keywords is None:
        return []
    if isinstance(raw_keywords, list):
        values = raw_keywords
    else:
        values = re.split(r"[;,]", str(raw_keywords))
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_space(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def _read_article_summary_row(
    *,
    summary_csv_path: Path,
    paper_id: str,
) -> dict[str, Any] | None:
    if not summary_csv_path.exists():
        return None
    with summary_csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        paper_id_key = next(
            (
                name for name in fieldnames
                if str(name).strip() in {"文章编号", "paper_id"}
                or "文章编号" in str(name)
            ),
            None,
        )
        total_without_refs_key = next(
            (
                name for name in fieldnames
                if str(name).strip() in {"总字数(不含参考文献)", "word_count_without_references"}
                or "总字数(不含参考文献)" in str(name)
            ),
            None,
        )
        total_with_refs_key = next(
            (
                name for name in fieldnames
                if str(name).strip() in {"总字数(含参考文献)", "word_count_with_references"}
                or "总字数(含参考文献)" in str(name)
            ),
            None,
        )
        if not paper_id_key or not total_without_refs_key:
            return None
        for row in reader:
            if _normalize_space(row.get(paper_id_key, "")) != paper_id:
                continue
            body_target_words = int(_normalize_space(row.get(total_without_refs_key, "0")) or 0)
            total_target_words = body_target_words
            if total_with_refs_key:
                total_target_words = int(_normalize_space(row.get(total_with_refs_key, "0")) or body_target_words)
            return {
                "paper_id": paper_id,
                "body_target_words": body_target_words,
                "total_target_words": total_target_words,
            }
    return None


def _article_outline_from_body(body: str) -> dict[str, str]:
    outline: dict[str, str] = {}
    for section in parse_markdown_sections(body):
        if section.is_title:
            continue
        normalized = _normalize_heading_key(section.heading)
        if normalized in ABSTRACT_HEADINGS:
            continue
        if normalized in KEYWORD_HEADINGS:
            continue
        if normalized in REFERENCE_HEADINGS:
            continue
        if normalized in NON_BODY_HEADINGS:
            continue
        if count_length_units(section.body) <= 0:
            continue
        outline[f"sec{len(outline) + 1}"] = _normalize_space(section.heading)
    if len(outline) < 2:
        raise ValueError("derived article outline must contain at least 2 content sections")
    return outline


def _article_raw_section_word_targets(
    body: str,
    *,
    outline: Mapping[str, str],
) -> dict[str, int]:
    targets: dict[str, int] = {}
    section_index = 0
    for section in parse_markdown_sections(body):
        if section.is_title:
            continue
        normalized = _normalize_heading_key(section.heading)
        if normalized in ABSTRACT_HEADINGS:
            continue
        if normalized in KEYWORD_HEADINGS:
            continue
        if normalized in REFERENCE_HEADINGS:
            continue
        if normalized in NON_BODY_HEADINGS:
            continue
        word_count = count_length_units(section.body)
        if word_count <= 0:
            continue
        section_index += 1
        sec_id = f"sec{section_index}"
        if sec_id not in outline:
            continue
        targets[sec_id] = int(word_count)
    if list(targets.keys()) != list(outline.keys()):
        raise ValueError("real article section_word_targets do not align with derived outline")
    return targets


def _scale_section_targets(
    actual_targets: Mapping[str, float | int],
    *,
    total_words: int,
) -> dict[str, int]:
    if total_words <= 0:
        raise ValueError("total_words must be positive")
    raw_total = sum(float(value) for value in actual_targets.values())
    if raw_total <= 0:
        raise ValueError("actual section targets must sum to a positive value")
    raw_values = {
        section_id: total_words * float(actual_targets[section_id]) / raw_total
        for section_id in actual_targets
    }
    floors = {
        section_id: int(raw_values[section_id])
        for section_id in actual_targets
    }
    remainder = total_words - sum(floors.values())
    ranked = sorted(
        actual_targets.keys(),
        key=lambda section_id: (raw_values[section_id] - floors[section_id], section_id),
        reverse=True,
    )
    scaled = dict(floors)
    for section_id in ranked[:remainder]:
        scaled[section_id] += 1
    return scaled


def _build_canonical_six_slot_outline() -> dict[str, str]:
    return dict(CANONICAL_SIX_SLOT_OUTLINE)


def _article_six_slot_word_targets(body: str) -> dict[str, float]:
    slot_word_counts = derive_article_six_slot_word_counts(body)
    return {
        section_id: float(slot_word_counts[slot])
        for section_id, slot in SECTION_ID_TO_SLOT.items()
    }


def _strip_title_review_markers(title: str) -> str:
    value = _normalize_space(title)
    for pattern in TITLE_SUFFIX_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    for pattern in TITLE_PREFIX_PATTERNS:
        value = re.sub(pattern, "", value, flags=re.IGNORECASE)
    return _normalize_space(value.strip(" -:;,."))


def _derive_topic(title: str) -> str:
    stripped = _strip_title_review_markers(title)
    if stripped:
        return stripped
    return _normalize_space(title)


def _derive_domain(title: str, abstract: str, keywords: Sequence[str]) -> str:
    combined = " ".join([title, abstract, *keywords]).casefold()
    if any(token in combined for token in ("patient", "patients", "therapy", "treatment", "risk", "disease")):
        return "clinical evidence synthesis"
    return "biomedical evidence synthesis"


def _derive_practice_context(title: str, abstract: str) -> str:
    combined = f"{title}\n{abstract}".casefold()
    if any(token in combined for token in ("patient", "patients", "treatment", "therapy", "risk", "management", "clinical")):
        return "evidence-based clinical decision-making"
    return "biomedical research interpretation"


def _derive_organizer(title: str, outline: Mapping[str, str]) -> str:
    combined = " ".join([title, *outline.values()]).casefold()
    if any(token in combined for token in ("meta-analysis", "meta analysis", "systematic review")):
        return "systematic evidence synthesis framework"
    if any(token in combined for token in ("mechanism", "pathway", "classification", "framework")):
        return "mechanism and evidence framework"
    return "evidence comparison framework"


def _extract_title_chunks(title: str) -> list[str]:
    clean_title = _strip_title_review_markers(title)
    chunks: list[str] = []
    splitters = [
        r"\s+versus\s+",
        r"\s+vs\.?\s+",
        r"\s+in patients with\s+",
        r"\s+among patients with\s+",
        r"\s+for\s+",
        r":\s+",
    ]
    candidates = [clean_title]
    for splitter in splitters:
        next_candidates: list[str] = []
        for candidate in candidates:
            parts = [part.strip(" -:;,.") for part in re.split(splitter, candidate, flags=re.IGNORECASE) if part.strip(" -:;,.")]
            if len(parts) > 1:
                next_candidates.extend(parts)
            else:
                next_candidates.append(candidate)
        candidates = next_candidates
    for candidate in candidates:
        normalized = _normalize_space(candidate)
        if not normalized:
            continue
        lower = normalized.casefold()
        if any(pattern == lower for pattern in GENERIC_FOCUS_POINT_PATTERNS):
            continue
        if len(re.findall(r"[A-Za-z0-9]+", normalized)) < 2:
            continue
        chunks.append(normalized)
    return chunks


def _derive_focus_points(
    *,
    title: str,
    keywords: Sequence[str],
    outline: Mapping[str, str],
) -> list[str]:
    focus_points: list[str] = []
    seen: set[str] = set()

    for keyword in keywords:
        item = _normalize_space(keyword)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        focus_points.append(item)

    for heading in outline.values():
        normalized_heading = normalize_heading(heading)
        if normalized_heading in GENERIC_OUTLINE_HEADINGS:
            continue
        item = _normalize_space(heading)
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        focus_points.append(item)

    for chunk in _extract_title_chunks(title):
        key = chunk.casefold()
        if key in seen:
            continue
        seen.add(key)
        focus_points.append(chunk)

    if not focus_points:
        focus_points.append(_derive_topic(title))
    return focus_points[:4]


def _derive_extra_must_include(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}"
    candidates = re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        item = _normalize_space(candidate)
        if len(item) < 2:
            continue
        if item.casefold() in {"and", "the"}:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned[:4]


def _article_word_targets(
    frontmatter: Mapping[str, object],
    body: str,
) -> tuple[int, int]:
    stats = frontmatter.get("stats")
    if isinstance(stats, Mapping):
        raw_total = stats.get("word_count")
        if isinstance(raw_total, int) and raw_total > 0:
            total_words = int(raw_total)
        else:
            total_words = 0
    else:
        total_words = 0

    references_match = re.search(r"(?ms)^##\s+References\s*$", body)
    if references_match is not None:
        body_text = body[: references_match.start()]
    else:
        body_text = body
    body_words = count_length_units(body_text)
    if total_words <= 0:
        total_words = body_words
    return total_words, body_words


def _stable_seed_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _expanded_body_target_profile(
    *,
    task_id: str,
    source_body_words: int,
) -> dict[str, int | str]:
    if EXPANDED_BODY_TARGET_MIN_WORDS <= 0:
        raise ValueError("expanded body target minimum must be positive")
    if EXPANDED_BODY_TARGET_MAX_WORDS < EXPANDED_BODY_TARGET_MIN_WORDS:
        raise ValueError("expanded body target maximum must be >= minimum")
    if EXPANDED_BODY_TARGET_STEP_WORDS <= 0:
        raise ValueError("expanded body target step must be positive")

    seed = f"{task_id}:{EXPANDED_LENGTH_MODE}"
    rng = random.Random(_stable_seed_int(seed))
    bucket_count = (
        (EXPANDED_BODY_TARGET_MAX_WORDS - EXPANDED_BODY_TARGET_MIN_WORDS)
        // EXPANDED_BODY_TARGET_STEP_WORDS
    )
    sampled_body_target_words = (
        EXPANDED_BODY_TARGET_MIN_WORDS
        + rng.randrange(bucket_count + 1) * EXPANDED_BODY_TARGET_STEP_WORDS
    )
    runtime_body_target_words = max(int(source_body_words), sampled_body_target_words)
    return {
        "length_mode": EXPANDED_LENGTH_MODE,
        "length_seed": seed,
        "expanded_body_target_min_words": EXPANDED_BODY_TARGET_MIN_WORDS,
        "expanded_body_target_max_words": EXPANDED_BODY_TARGET_MAX_WORDS,
        "expanded_body_target_step_words": EXPANDED_BODY_TARGET_STEP_WORDS,
        "sampled_body_target_words": sampled_body_target_words,
        "runtime_body_target_words": runtime_body_target_words,
    }


def _clean_public_review_constraints(constraints: dict[str, Any]) -> None:
    for key in PUBLIC_REVIEW_CONSTRAINT_DROP_KEYS:
        constraints.pop(key, None)


def derive_review_task_from_article(
    *,
    article_path: Path,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    summary_csv_path: Path | None = None,
) -> DerivedReviewTask:
    raw_text = article_path.read_text(encoding="utf-8", errors="replace")
    frontmatter, body = _parse_frontmatter(raw_text)

    if not isinstance(frontmatter, Mapping):
        raise ValueError("article frontmatter is missing or invalid")

    paper_id = _normalize_space(frontmatter.get("paper_id") or article_path.stem)
    title = _normalize_space(frontmatter.get("title") or article_path.stem)
    abstract = _normalize_space(frontmatter.get("abstract") or "")
    keywords = _frontmatter_keywords(frontmatter.get("keywords"))
    raw_outline = _article_outline_from_body(body)
    outline = _build_canonical_six_slot_outline()
    source_total_words, source_body_words = _article_word_targets(frontmatter, body)
    article_summary = None
    if summary_csv_path is not None:
        article_summary = _read_article_summary_row(
            summary_csv_path=summary_csv_path,
            paper_id=paper_id,
        )
    if article_summary is not None:
        source_body_words = int(article_summary["body_target_words"])
        source_total_words = int(article_summary["total_target_words"])

    task_id = f"med_{paper_id.lower()}"
    length_profile = _expanded_body_target_profile(
        task_id=task_id,
        source_body_words=int(source_body_words),
    )
    reference_target_words = max(int(source_total_words) - int(source_body_words), 0)
    body_words = int(length_profile["runtime_body_target_words"])
    total_words = body_words + reference_target_words

    spec = TaskSpec(
        task_id=task_id,
        topic=_derive_topic(title),
        domain=_derive_domain(title, abstract, keywords),
        paper_type="medical_review",
        target_words=total_words,
        body_target_words=body_words,
        expected_sections=len(CANONICAL_SIX_SLOT_OUTLINE),
        language="English",
        practice_context=_derive_practice_context(title, abstract),
        organizer=_derive_organizer(title, raw_outline),
        focus_points=_derive_focus_points(
            title=title,
            keywords=keywords,
            outline=raw_outline,
        ),
        closing_requirements=["limitations", "future work"],
        extra_must_include=_derive_extra_must_include(title, abstract),
    )
    config = build_main_task_config(
        spec,
        session_name=f"metabench_{paper_id.lower()}",
        corpus_dir=corpus_dir,
        outline_override=dict(outline),
    )
    reference = config.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("generated task config is missing reference payload")
    constraints = reference.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("generated task config is missing constraints payload")
    section_word_targets = dict(constraints["section_word_targets"])
    _clean_public_review_constraints(constraints)

    return DerivedReviewTask(
        paper_id=paper_id,
        source_article_path=str(article_path.resolve()),
        spec=spec,
        outline=dict(outline),
        section_word_targets=dict(section_word_targets),
        config=config,
    )


def build_task_snapshot_from_article(
    *,
    article_path: Path,
    corpus_dir: str = DEFAULT_CORPUS_DIR,
    summary_csv_path: Path | None = None,
) -> dict[str, Any]:
    derived = derive_review_task_from_article(
        article_path=article_path,
        corpus_dir=corpus_dir,
        summary_csv_path=summary_csv_path,
    )
    snapshot = {
        "task_name": derived.task_name,
        "task": derived.config.get("task"),
        "constraints": derived.config.get("constraints"),
        "outline": derived.config.get("outline"),
        "section_word_targets": derived.section_word_targets,
        "session_name": derived.config.get("session_name"),
        "reference": derived.config.get("reference"),
        "corpus_dir": derived.config.get("corpus_dir"),
    }
    return snapshot


def render_task_module(derived: DerivedReviewTask) -> str:
    """Render a Python task module for a derived review task."""

    focus_points_repr = json.dumps(list(derived.spec.focus_points), ensure_ascii=False, indent=8)
    extra_must_include_repr = json.dumps(list(derived.spec.extra_must_include), ensure_ascii=False, indent=8)
    outline_repr = json.dumps(derived.outline, ensure_ascii=False, indent=4)
    runtime_drop_keys_repr = repr(("section_budget_trace", "six_slot_prior_version"))

    return (
        f'"""MetaBench task derived from review article {derived.paper_id}."""\n\n'
        "from __future__ import annotations\n\n"
        "from typing import Dict\n\n"
        "from meta_bench import TaskSpec, build_main_task_config\n\n\n"
        f"CUSTOM_OUTLINE = {outline_repr}\n\n\n"
        "def get_task_config() -> Dict[str, object]:\n"
        f'    """Return the {derived.paper_id}-derived MetaBench task configuration."""\n\n'
        "    spec = TaskSpec(\n"
        f'        task_id="{derived.spec.task_id}",\n'
        f'        topic="{derived.spec.topic}",\n'
        f'        domain="{derived.spec.domain}",\n'
        f"        target_words={int(derived.spec.target_words)},\n"
        f"        body_target_words={int(derived.spec.body_target_words or derived.spec.target_words)},\n"
        f"        expected_sections={int(derived.spec.expected_sections)},\n"
        f'        practice_context="{derived.spec.practice_context}",\n'
        f'        organizer="{derived.spec.organizer}",\n'
        f"        focus_points={focus_points_repr},\n"
        f"        extra_must_include={extra_must_include_repr},\n"
        "    )\n"
        "    config = build_main_task_config(\n"
        "        spec,\n"
        f'        session_name="metabench_{derived.paper_id.lower()}",\n'
        f'        corpus_dir="{DEFAULT_CORPUS_DIR}",\n'
        "        outline_override=dict(CUSTOM_OUTLINE),\n"
        "    )\n\n"
        '    reference = config["reference"]\n'
        '    constraints = reference["constraints"]\n'
        f"    for key in {runtime_drop_keys_repr}:\n"
        "        constraints.pop(key, None)\n\n"
        "    return config\n"
    )


def write_task_module(
    derived: DerivedReviewTask,
    *,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_task_module(derived), encoding="utf-8")
    return output_path


__all__ = [
    "DEFAULT_CORPUS_DIR",
    "DerivedReviewTask",
    "build_task_snapshot_from_article",
    "derive_review_task_from_article",
    "render_task_module",
    "write_task_module",
]
