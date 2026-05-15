"""Typed data structures for the clean MetaBench implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TaskSpec:
    """Structured input used to generate a benchmark task."""

    task_id: str
    topic: str
    domain: str
    paper_type: str = "medical_review"
    target_words: int = 4200
    body_target_words: int | None = None
    expected_sections: int = 7
    language: str = "English"
    practice_context: str = "clinical practice"
    organizer: str = "classification framework"
    focus_points: list[str] = field(default_factory=list)
    closing_requirements: list[str] = field(
        default_factory=lambda: ["limitations", "future work"]
    )
    extra_must_include: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GeneratedTask:
    """Generated benchmark task payload."""

    task_id: str
    prompt: str
    constraints: dict[str, Any]
    outline: dict[str, str]
    reference: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
