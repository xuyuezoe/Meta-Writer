"""
Direct-LLM 基线适配器（S0）

功能：
    最弱基线：把完整任务描述（主题 + 约束 + 大纲 + 目标长度）一次性传给 LLM，
    单次调用生成全文。使用与 MetaWriter 相同的骨干模型与语料访问口径之外的
    "无系统框架"对照。

设计动机（第一性原理）：
    Direct-LLM 量化"有系统框架 vs. 无系统框架"的总增益。它必须满足：
        1. 单次生成：不做任何在线验证、修复、记忆，纯 prompt → 文本。
        2. 同骨干：使用与 MetaWriter 相同的 LLMClient，保证差异只来自系统设计。
        3. 进程内：无外部依赖，直接调用 LLMClient.generate()。

注意：
    Direct-LLM 不接入检索语料（无 RAG），这是其作为"最弱基线"的定义之一；
    若需"单次生成 + 检索"的更强基线，应另立适配器，不在此混入（保持基线纯净）。
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from .base_adapter import (
    BaselineAdapter,
    BaselineAdapterError,
    BaselineResult,
    BaselineTask,
)

# 单次生成的 token 上限：按目标词数放宽（英文约 1.4 token/词，留足余量）。
# 设上界防止异常长任务导致请求超限。
_TOKENS_PER_WORD = 2
_MAX_OUTPUT_TOKENS_CAP = 32768
_MIN_OUTPUT_TOKENS = 2048


class DirectLLMAdapter(BaselineAdapter):
    """
    Direct-LLM 单次生成适配器

    功能：
        构造统一 prompt，单次调用 LLMClient 生成全文。

    参数：
        llm_client: LLM 客户端实例（需提供 generate() 与 get_statistics()）。
        temperature: 生成温度（默认 0.7，与 MetaWriter 首轮生成一致）。
    """

    system_name = "direct-llm"

    def __init__(self, llm_client: Any, *, temperature: float = 0.7) -> None:
        self._llm = llm_client
        self._temperature = temperature

    def run(self, task: BaselineTask, *, work_dir: str) -> BaselineResult:
        """
        单次生成全文

        参数：
            task:     规范化对比任务
            work_dir: 工作目录（Direct-LLM 不产出中间文件，仅占位以满足契约）

        返回值：
            BaselineResult：含最终文本与 LLM 统计

        异常：
            BaselineAdapterError：当 LLM 调用失败或产出空文本时抛出。
        """
        prompt = self._build_prompt(task)
        max_tokens = self._resolve_max_tokens(task.target_words)

        start = time.time()
        try:
            text = self._llm.generate(
                prompt=prompt,
                temperature=self._temperature,
                max_tokens=max_tokens,
                log_meta={"component": "DirectLLMBaseline", "task_id": task.task_id},
            )
        except Exception as exc:  # 显式包装为适配器异常，保留原始堆栈语义
            raise BaselineAdapterError(
                f"[Direct-LLM 调用失败] task={task.task_id}: {exc}"
            ) from exc
        wall_time = time.time() - start

        text = (text or "").strip()
        if not text:
            raise BaselineAdapterError(
                f"[Direct-LLM 产出为空] task={task.task_id}：模型返回空文本"
            )

        stats = self._safe_statistics()
        return BaselineResult(
            system=self.system_name,
            final_text=text,
            total_tokens=stats.get("total_tokens"),
            request_count=stats.get("request_count"),
            wall_time_seconds=round(wall_time, 3),
            status="completed",
            extra={"prompt_chars": len(prompt), "max_tokens": max_tokens},
        )

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _build_prompt(self, task: BaselineTask) -> str:
        """
        构造单次生成 prompt

        功能：
            将主题、约束、大纲、目标长度拼为一个完整指令，要求 LLM 一次写完全文。

        参数：
            task: 规范化对比任务

        返回值：
            str：完整 prompt
        """
        constraint_block = self._format_constraints(task.constraints)
        outline_block = self._format_outline(task.outline)

        return (
            f"{task.task_description}\n\n"
            f"Target length: approximately {task.target_words} words.\n\n"
            f"{constraint_block}"
            f"{outline_block}"
            "Write the complete article in a single response. "
            "Follow the outline section by section, satisfy every constraint, "
            "and reach the target length. Output only the article text."
        )

    @staticmethod
    def _format_constraints(constraints: List[str]) -> str:
        """格式化约束块；无约束时返回空串"""
        if not constraints:
            return ""
        lines = "\n".join(f"- {c}" for c in constraints)
        return f"Constraints to satisfy:\n{lines}\n\n"

    @staticmethod
    def _format_outline(outline: Dict[str, str]) -> str:
        """格式化大纲块；无大纲时返回空串"""
        if not outline:
            return ""
        lines = "\n".join(f"[{sid}] {title}" for sid, title in outline.items())
        return f"Outline ({len(outline)} sections):\n{lines}\n\n"

    @staticmethod
    def _resolve_max_tokens(target_words: int) -> int:
        """
        按目标词数解析输出 token 上限

        参数：
            target_words: 目标总词数

        返回值：
            int：约束在 [_MIN_OUTPUT_TOKENS, _MAX_OUTPUT_TOKENS_CAP] 内的 token 上限
        """
        estimated = max(target_words, 0) * _TOKENS_PER_WORD
        return max(_MIN_OUTPUT_TOKENS, min(_MAX_OUTPUT_TOKENS_CAP, estimated))

    def _safe_statistics(self) -> Dict[str, Any]:
        """
        读取 LLM 统计

        返回值：
            Dict：含 total_tokens / request_count；客户端未提供该接口时返回空字典。

        关键实现细节：
            统计缺失不视为错误（部分网关不返回 usage），但必须区分"没有统计"
            与"统计为 0"，故缺失时返回空字典（上层将其映射为 None）。
        """
        getter = getattr(self._llm, "get_statistics", None)
        if not callable(getter):
            return {}
        stats = getter()
        if not isinstance(stats, dict):
            return {}
        return stats
