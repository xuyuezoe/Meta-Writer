from __future__ import annotations

import argparse
import unittest

import main


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

    def test_resolve_default_task(self) -> None:
        args = self._make_args()
        self.assertEqual(main._resolve_requested_task_names(args), [main.TASK_NAME])

    def test_resolve_single_benchmark_task_id(self) -> None:
        args = self._make_args(task_id="med_s010")
        self.assertEqual(
            main._resolve_requested_task_names(args),
            [f"{main.BENCHMARK_TASK_PREFIX}med_s010"],
        )

    def test_resolve_all_benchmark_tasks(self) -> None:
        args = self._make_args(all_tasks=True)
        resolved = main._resolve_requested_task_names(args)
        expected = [
            f"{main.BENCHMARK_TASK_PREFIX}{task_id}"
            for task_id in main.list_benchmark_task_ids()
        ]
        self.assertEqual(resolved, expected)

    def test_build_batch_summary(self) -> None:
        summary = main._build_batch_summary(
            [
                {
                    "status": "completed",
                    "benchmark_evaluation": {
                        "constraint_violation_rate": 0.1,
                        "entity_consistency_score": 0.8,
                        "logical_coherence": 0.9,
                    },
                },
                {
                    "status": "completed",
                    "benchmark_evaluation": {
                        "constraint_violation_rate": 0.3,
                        "entity_consistency_score": 0.6,
                        "logical_coherence": 0.7,
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
            summary["average_benchmark_scores"],
            {
                "constraint_violation_rate": 0.2,
                "entity_consistency_score": 0.7,
                "logical_coherence": 0.8,
            },
        )


if __name__ == "__main__":
    unittest.main()
