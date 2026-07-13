from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import main
from examples.tasks import META_BENCH_TASK_TIERS, TASK_REGISTRY


class MainCliTests(unittest.TestCase):
    def _make_args(
        self,
        *,
        task_name: str | None = None,
        task_id: str | None = None,
        all_tasks: bool = False,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            task_name=task_name,
            task_id=task_id,
            all=all_tasks,
            list_tasks=False,
            print_response=False,
        )

    def test_resolve_default_tasks_runs_first_registered_meta_bench_task(self) -> None:
        args = self._make_args()
        self.assertEqual(
            main._resolve_requested_task_names(args),
            [main.META_BENCH_TASK_NAMES[0]],
        )

    def test_resolve_single_task_id(self) -> None:
        args = self._make_args(task_id="med_s001")
        self.assertEqual(
            main._resolve_requested_task_names(args),
            ["metabench_sample"],
        )

    def test_resolve_all_meta_bench_tasks(self) -> None:
        args = self._make_args(all_tasks=True)
        resolved = main._resolve_requested_task_names(args)
        self.assertEqual(resolved, main.META_BENCH_TASK_NAMES)

    def test_default_meta_bench_tasks_are_medium_or_long_reviews(self) -> None:
        self.assertEqual(len(META_BENCH_TASK_TIERS["benchmark"]), 50)
        self.assertEqual(len(META_BENCH_TASK_TIERS["medium"]), 16)
        self.assertEqual(len(META_BENCH_TASK_TIERS["long"]), 34)
        self.assertEqual(len(META_BENCH_TASK_TIERS["all"]), 50)

        for task_name in main.META_BENCH_TASK_NAMES:
            config = TASK_REGISTRY[task_name]()
            constraints = config["reference"]["constraints"]
            self.assertGreaterEqual(constraints["body_target_words"], 3500)
            self.assertNotIn("length_mode", constraints)
            self.assertNotIn("section_budget_trace", constraints)

        for task_name in META_BENCH_TASK_TIERS["long"]:
            config = TASK_REGISTRY[task_name]()
            constraints = config["reference"]["constraints"]
            self.assertGreaterEqual(constraints["body_target_words"], 5000)

    def test_build_batch_summary(self) -> None:
        summary = main._build_batch_summary(
            [
                {
                    "status": "completed",
                    "meta_bench_evaluation": {
                        "metric_scores": {
                            "article_entity_recall": 0.9,
                            "coverage_score": 0.6,
                            "length_adherence": 0.8,
                            "heading_soft_recall": 0.9,
                            "citation_quality_f1": 0.7,
                            "section_distribution": 0.8,
                            "citation_count": 0.85,
                            "source_balance": 0.85,
                        },
                    },
                },
                {
                    "status": "completed",
                    "meta_bench_evaluation": {
                        "metric_scores": {
                            "article_entity_recall": 0.7,
                            "coverage_score": 0.3,
                            "length_adherence": 0.7,
                            "heading_soft_recall": 0.7,
                            "citation_quality_f1": 0.5,
                            "section_distribution": 0.6,
                            "citation_count": 0.65,
                            "source_balance": 0.65,
                        },
                    },
                },
                {
                    "status": "failed",
                    "error": "boom",
                },
            ]
        )

        self.assertEqual(summary["task_count"], 3)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(
            summary["average_metric_scores"]["article_entity_recall"],
            0.8,
        )
        self.assertEqual(
            summary["average_metric_scores"]["coverage_score"],
            0.45,
        )

    def test_write_run_bundle_copies_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            session_dir = root / "sessions"
            output_dir.mkdir()
            session_dir.mkdir()

            task_input_file = output_dir / "demo_task_input.json"
            text_file = output_dir / "demo_text.txt"
            correction_log_file = output_dir / "demo_correction_log.json"
            dtg_file = output_dir / "demo_dtg.json"
            summary_file = output_dir / "demo_summary.json"
            run_log_file = output_dir / "demo_run.log"
            eval_file = output_dir / "demo_meta_bench_eval.json"
            citation_manifest_file = output_dir / "demo_citation_manifest.json"
            chunk_map_file = output_dir / "demo_chunk_map.json"
            session_file = session_dir / "demo.json"

            for path in (
                task_input_file,
                text_file,
                correction_log_file,
                dtg_file,
                summary_file,
                run_log_file,
                eval_file,
                citation_manifest_file,
                chunk_map_file,
                session_file,
            ):
                path.write_text("{}", encoding="utf-8")

            bundle_dir = main._write_run_bundle(
                output_dir=output_dir,
                session_name="demo",
                task_input_file=task_input_file,
                text_file=text_file,
                correction_log_file=correction_log_file,
                dtg_file=dtg_file,
                summary_file=summary_file,
                session_file=session_file,
                run_log_file=run_log_file,
                meta_bench_eval_file=eval_file,
                citation_manifest_file=citation_manifest_file,
                chunk_map_file=chunk_map_file,
            )

            self.assertTrue((bundle_dir / "task_input.json").exists())
            self.assertTrue((bundle_dir / "text.txt").exists())
            self.assertTrue((bundle_dir / "correction_log.json").exists())
            self.assertTrue((bundle_dir / "dtg.json").exists())
            self.assertTrue((bundle_dir / "summary.json").exists())
            self.assertTrue((bundle_dir / "session.json").exists())
            self.assertTrue((bundle_dir / "run.log").exists())
            self.assertTrue((bundle_dir / "meta_bench_eval.json").exists())
            self.assertTrue((bundle_dir / "citation_manifest.json").exists())
            self.assertTrue((bundle_dir / "chunk_map.json").exists())
            self.assertTrue((bundle_dir / "artifact_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
