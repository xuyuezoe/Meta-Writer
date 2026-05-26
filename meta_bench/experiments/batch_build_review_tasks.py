"""Batch-build MetaBench review tasks and experiment assets from real articles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .build_review_experiment_assets import build_experiment_assets
from .review_task_builder import derive_review_task_from_article, write_task_module


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTICLE_DIR: Path | None = None
DEFAULT_SUMMARY_CSV: Path | None = None
DEFAULT_TASK_OUTPUT_DIR = REPO_ROOT / "examples" / "tasks"
DEFAULT_EXPERIMENTS_DIR = REPO_ROOT / "data_sample" / "experiments"


def _normalize_paper_id(value: str) -> str:
    paper_id = str(value or "").strip()
    if not paper_id:
        raise ValueError("paper_id must not be empty")
    return paper_id


def _article_path_for_paper(paper_id: str, article_dir: Path) -> Path:
    article_path = article_dir / f"{paper_id}.md"
    if not article_path.exists():
        raise FileNotFoundError(f"Article markdown not found: {article_path}")
    return article_path


def build_single_review_task_bundle(
    *,
    paper_id: str,
    article_dir: Path,
    summary_csv_path: Path | None = DEFAULT_SUMMARY_CSV,
    task_output_dir: Path = DEFAULT_TASK_OUTPUT_DIR,
    experiments_dir: Path = DEFAULT_EXPERIMENTS_DIR,
) -> dict[str, Any]:
    normalized_paper_id = _normalize_paper_id(paper_id)
    article_path = _article_path_for_paper(normalized_paper_id, article_dir)

    derived = derive_review_task_from_article(
        article_path=article_path,
        summary_csv_path=summary_csv_path,
    )
    task_module_path = task_output_dir / f"{derived.module_name}.py"
    write_task_module(derived, output_path=task_module_path)

    experiment_dir = experiments_dir / normalized_paper_id.lower()
    experiment_assets = build_experiment_assets(
        article_path=article_path,
        output_dir=experiment_dir,
        summary_csv_path=summary_csv_path,
        derive_task_snapshot=True,
    )

    return {
        "paper_id": normalized_paper_id,
        "task_name": derived.task_name,
        "task_id": derived.spec.task_id,
        "task_module": str(task_module_path),
        "experiment_dir": str(experiment_dir),
        "body_target_words": int(derived.spec.body_target_words or 0),
        "total_target_words": int(derived.spec.target_words),
        "expected_sections": int(derived.spec.expected_sections),
        "section_word_targets": dict(derived.section_word_targets),
        "experiment_assets": experiment_assets,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch build review-derived MetaBench tasks")
    parser.add_argument("--paper-id", action="append", required=True, help="PMC paper id, repeatable")
    parser.add_argument("--article-dir", type=Path, default=DEFAULT_ARTICLE_DIR, required=DEFAULT_ARTICLE_DIR is None)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--task-output-dir", type=Path, default=DEFAULT_TASK_OUTPUT_DIR)
    parser.add_argument("--experiments-dir", type=Path, default=DEFAULT_EXPERIMENTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    results = []
    for paper_id in args.paper_id:
        results.append(
            build_single_review_task_bundle(
                paper_id=paper_id,
                article_dir=args.article_dir,
                summary_csv_path=args.summary_csv,
                task_output_dir=args.task_output_dir,
                experiments_dir=args.experiments_dir,
            )
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
