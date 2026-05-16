"""Deterministic section-word budgeting for MetaBench tasks."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Mapping, Sequence

from .citation import CitationChunk, classify_chunk_roles


SECTION_SHARE_PRIORS: dict[str, float] = {
    "introduction": 0.3748,
    "taxonomy": 0.0052,
    "evidence_synthesis": 0.3182,
    "limitations_gaps": 0.0382,
    "conclusion": 0.0612,
    "methods": 0.0381,
    "results": 0.0308,
    "discussion": 0.1334,
}

DEFAULT_ROLE_FOR_OUTLINE = "evidence_synthesis"
RELATIVE_JITTER = 0.15
EMPIRICAL_SHARE_WEIGHT = 0.7
UNIFORM_SHARE_WEIGHT = 0.3


@dataclass(frozen=True)
class SectionBudget:
    """Per-section word budget plus trace metadata."""

    body_target_words: int
    section_word_targets: dict[str, int]
    section_roles: dict[str, str]
    role_base_shares: dict[str, float]
    role_jitter_factors: dict[str, float]
    role_adjusted_shares: dict[str, float]
    seed: str
    relative_jitter: float

    def to_reference_payload(self) -> dict[str, object]:
        return {
            "body_target_words": self.body_target_words,
            "section_word_targets": dict(self.section_word_targets),
            "section_roles": dict(self.section_roles),
            "section_budget_trace": {
                "seed": self.seed,
                "relative_jitter": self.relative_jitter,
                "role_base_shares": dict(self.role_base_shares),
                "role_jitter_factors": dict(self.role_jitter_factors),
                "role_adjusted_shares": dict(self.role_adjusted_shares),
            },
        }


def _stable_seed_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _normalize_outline(outline: Mapping[str, object]) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for section_id, title in outline.items():
        sid = str(section_id).strip()
        if not sid:
            continue
        normalized.append((sid, str(title).strip()))
    return normalized


def _classify_outline_roles(
    outline_items: Sequence[tuple[str, str]],
) -> dict[str, str]:
    if not outline_items:
        return {}

    chunks = [
        CitationChunk(
            chunk_id=section_id,
            index=index,
            title=title,
            body=title,
            word_count=max(1, len(title.split())),
        )
        for index, (section_id, title) in enumerate(outline_items, start=1)
    ]
    classifications = classify_chunk_roles(
        chunks,
        reference={"outline": dict(outline_items)},
    )
    roles: dict[str, str] = {}
    for chunk, classification in zip(chunks, classifications):
        role = classification.canonical_type
        if role == "other":
            role = DEFAULT_ROLE_FOR_OUTLINE
        roles[chunk.chunk_id] = role
    return roles


def _build_role_shares(
    *,
    roles_in_use: set[str],
    seed: str,
    relative_jitter: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    rng = random.Random(_stable_seed_int(seed))

    base_shares: dict[str, float] = {}
    jitter_factors: dict[str, float] = {}
    raw_adjusted: dict[str, float] = {}
    for role in sorted(roles_in_use):
        base = float(SECTION_SHARE_PRIORS.get(role, 0.0))
        if base <= 0:
            base = 1e-4
        jitter = rng.uniform(1.0 - relative_jitter, 1.0 + relative_jitter)
        base_shares[role] = base
        jitter_factors[role] = jitter
        raw_adjusted[role] = base * jitter

    total = sum(raw_adjusted.values())
    if total <= 0:
        even_share = 1.0 / max(len(roles_in_use), 1)
        adjusted = {role: even_share for role in roles_in_use}
    else:
        adjusted = {role: raw_adjusted[role] / total for role in roles_in_use}
    return base_shares, jitter_factors, adjusted


def _round_targets_from_shares(
    *,
    total_words: int,
    per_section_share: Mapping[str, float],
) -> dict[str, int]:
    ordered_ids = list(per_section_share.keys())
    raw_values = {
        section_id: total_words * float(per_section_share[section_id])
        for section_id in ordered_ids
    }
    floors = {section_id: int(raw_values[section_id]) for section_id in ordered_ids}
    allocated = sum(floors.values())
    remainder = max(total_words - allocated, 0)
    ranked_by_fraction = sorted(
        ordered_ids,
        key=lambda section_id: (
            raw_values[section_id] - floors[section_id],
            section_id,
        ),
        reverse=True,
    )
    targets = dict(floors)
    for section_id in ranked_by_fraction[:remainder]:
        targets[section_id] += 1
    return targets


def _blend_with_uniform_section_prior(
    per_section_share: Mapping[str, float],
) -> dict[str, float]:
    section_ids = list(per_section_share.keys())
    if not section_ids:
        return {}

    uniform_share = 1.0 / len(section_ids)
    blended = {
        section_id: (
            EMPIRICAL_SHARE_WEIGHT * float(per_section_share[section_id])
            + UNIFORM_SHARE_WEIGHT * uniform_share
        )
        for section_id in section_ids
    }
    total = sum(blended.values())
    if total <= 0:
        return {section_id: uniform_share for section_id in section_ids}
    return {section_id: blended[section_id] / total for section_id in section_ids}


def build_section_budget(
    *,
    task_id: str,
    body_target_words: int,
    outline: Mapping[str, object],
    relative_jitter: float = RELATIVE_JITTER,
) -> SectionBudget:
    """Build deterministic per-section targets from empirical role shares."""

    if body_target_words <= 0:
        raise ValueError("body_target_words must be positive")

    outline_items = _normalize_outline(outline)
    if not outline_items:
        raise ValueError("outline must not be empty")

    section_roles = _classify_outline_roles(outline_items)
    role_to_sections: dict[str, list[str]] = {}
    for section_id, _title in outline_items:
        role = section_roles.get(section_id, DEFAULT_ROLE_FOR_OUTLINE)
        role_to_sections.setdefault(role, []).append(section_id)

    roles_in_use = set(role_to_sections)
    base_shares, jitter_factors, adjusted_role_shares = _build_role_shares(
        roles_in_use=roles_in_use,
        seed=task_id,
        relative_jitter=relative_jitter,
    )

    per_section_share: dict[str, float] = {}
    for role, section_ids in role_to_sections.items():
        role_share = adjusted_role_shares[role]
        split_share = role_share / max(len(section_ids), 1)
        for section_id in section_ids:
            per_section_share[section_id] = split_share

    per_section_share = _blend_with_uniform_section_prior(per_section_share)
    section_word_targets = _round_targets_from_shares(
        total_words=body_target_words,
        per_section_share=per_section_share,
    )

    return SectionBudget(
        body_target_words=body_target_words,
        section_word_targets=section_word_targets,
        section_roles=section_roles,
        role_base_shares={key: round(value, 6) for key, value in base_shares.items()},
        role_jitter_factors={
            key: round(value, 6) for key, value in jitter_factors.items()
        },
        role_adjusted_shares={
            key: round(value, 6) for key, value in adjusted_role_shares.items()
        },
        seed=task_id,
        relative_jitter=relative_jitter,
    )
