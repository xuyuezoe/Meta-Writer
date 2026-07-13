"""In-process Direct-LLM baseline adapter."""

from __future__ import annotations

from .base_adapter import BaselineAdapterError, BaselineResult, BaselineTask


class DirectLLMAdapter:
    """Run the task with a plain LLM prompt and no MetaWriter machinery."""

    system_name = "direct-llm"

    def __init__(self, llm_client):
        if llm_client is None:
            raise BaselineAdapterError("direct-llm requires an llm_client")
        self._client = llm_client

    def run(self, task: BaselineTask, *, work_dir: str) -> BaselineResult:
        del work_dir
        prompt = _build_direct_prompt(task)
        max_tokens = max(1024, int(task.target_words * 2.2))
        text = self._client.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            log_meta={"system": self.system_name, "task_id": task.task_id},
        )
        final_text = str(text).strip()
        if not final_text:
            raise BaselineAdapterError("direct-llm produced empty output")

        stats = {}
        if hasattr(self._client, "get_statistics"):
            raw_stats = self._client.get_statistics()
            if isinstance(raw_stats, dict):
                stats = raw_stats

        return BaselineResult(
            system=self.system_name,
            final_text=final_text,
            total_tokens=int(stats.get("total_tokens", 0) or 0),
            request_count=int(stats.get("request_count", 0) or 0),
        )


def _build_direct_prompt(task: BaselineTask) -> str:
    constraints = "\n".join(f"- {item}" for item in task.constraints)
    outline = "\n".join(f"- {key}: {value}" for key, value in task.outline.items())
    return (
        "Write a scholarly literature review for the following benchmark task.\n\n"
        f"Task id: {task.task_id}\n"
        f"Topic: {task.topic}\n"
        f"Target length: about {task.target_words} words.\n\n"
        f"Task description:\n{task.task_description}\n\n"
        f"Constraints:\n{constraints}\n\n"
        f"Outline:\n{outline}\n\n"
        "Return only the final review text."
    )
