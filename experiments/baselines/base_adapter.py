"""Shared data structures for baseline system adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class BaselineAdapterError(RuntimeError):
    """Raised when a baseline adapter cannot produce a valid run."""


@dataclass(frozen=True)
class BaselineTask:
    """Task payload passed to baseline systems."""

    task_id: str
    topic: str
    task_description: str
    constraints: list[str]
    outline: dict[str, str]
    corpus_dir: str
    target_words: int
    reference: Mapping[str, Any] | None = None

    @classmethod
    def from_meta_bench_config(
        cls,
        task_id: str,
        config: Mapping[str, Any],
    ) -> "BaselineTask":
        """Build a baseline task from a MetaBench task config."""

        try:
            task_description = str(config["task"])
            raw_constraints = config["constraints"]
            raw_outline = config["outline"]
        except KeyError as exc:
            raise BaselineAdapterError(
                f"missing required MetaBench task field: {exc.args[0]}"
            ) from exc

        if not isinstance(raw_constraints, list):
            raise BaselineAdapterError("MetaBench task constraints must be a list")
        if not isinstance(raw_outline, Mapping):
            raise BaselineAdapterError("MetaBench task outline must be a mapping")

        reference = config.get("reference")
        reference_map = reference if isinstance(reference, Mapping) else {}
        constraints_map = reference_map.get("constraints")
        constraints_ref = constraints_map if isinstance(constraints_map, Mapping) else {}

        topic = str(
            reference_map.get("topic")
            or config.get("topic")
            or task_id
        )
        target_words = _first_int(
            reference_map.get("total_target_words"),
            constraints_ref.get("total_target_words"),
            constraints_ref.get("body_target_words"),
            constraints_ref.get("required_length_words"),
            config.get("target_words"),
            default=4200,
        )
        corpus_dir = str(
            config.get("corpus_dir")
            or "./data_sample/med_papers_review_augmented_strict"
        )

        return cls(
            task_id=task_id,
            topic=topic,
            task_description=task_description,
            constraints=[str(item) for item in raw_constraints],
            outline={str(key): str(value) for key, value in raw_outline.items()},
            corpus_dir=corpus_dir,
            target_words=target_words,
            reference=reference if isinstance(reference, Mapping) else None,
        )

    def to_bridge_payload(self) -> dict[str, Any]:
        """Return the public task payload for external bridge scripts.

        The evaluation reference is intentionally omitted to avoid leaking gold
        scoring metadata to comparison systems.
        """

        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "task_description": self.task_description,
            "constraints": list(self.constraints),
            "outline": dict(self.outline),
            "corpus_dir": self.corpus_dir,
            "target_words": self.target_words,
        }


@dataclass(frozen=True)
class BaselineResult:
    """Normalized output from one baseline adapter run."""

    system: str
    final_text: str
    status: str = "completed"
    total_tokens: int = 0
    request_count: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def llm_stats(self) -> dict[str, int]:
        return {
            "total_tokens": int(self.total_tokens),
            "request_count": int(self.request_count),
        }


def _first_int(*values: Any, default: int) -> int:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip():
            try:
                return int(value)
            except ValueError:
                continue
    return default
