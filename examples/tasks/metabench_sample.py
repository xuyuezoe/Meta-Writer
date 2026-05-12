"""Local MetaBench sample task configuration."""

from __future__ import annotations

from typing import Dict

from meta_bench import TaskSpec, build_main_task_config


def get_task_config() -> Dict[str, object]:
    """Return the default MetaBench sample task configuration."""
    spec = TaskSpec(
        task_id="med_s001",
        topic="acute coronary syndrome",
        domain="cardiovascular medicine",
        target_words=4200,
        expected_sections=7,
        practice_context="adult inpatient care",
        organizer="classification framework",
        focus_points=[
            "pathophysiology",
            "hemodynamics",
            "evidence integration",
        ],
    )
    return build_main_task_config(
        spec,
        session_name="metabench_sample_med_s001",
        corpus_dir="./data_sample/med_papers",
    )
