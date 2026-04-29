"""
章节规划器：SectionPlanner

功能：
    在每节生成前，根据 Task Plan、当前 DSL 状态和已完成内容摘要，
    为当前节制定局部计划（SectionIntent）。
    生成的 SectionIntent 同时注册为 DTG 中的 intent_node，支持 plan_level 诊断。

依赖：LLMClient、SectionIntent（core/plan.py）、DTGStore（memory/dtg_store.py）
被依赖：Orchestrator（每节生成前调用）
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Optional

from ..core.plan import SectionIntent
from ..memory.dtg_store import DTGStore
from ..utils.llm_client import LLMClient


class SectionPlanner:
    """
    章节规划器

    功能：
        调用 LLM 生成当前节的 SectionIntent，并将其注册为 DTG 的 intent_node。
        实现"决定做什么"与"执行"的职责分离。

    参数：
        llm_client: LLM 客户端实例
        dtg_store: DTGStore 实例（用于 add_intent_node）

    关键实现细节：
        温度固定 0.3（规划阶段需要保守确定性）。
        max_tokens 32768（放宽上限，防止长上下文导致截断）。
        LLM 输出 JSON 结构描述局部计划，解析失败时返回保守默认值。
    """

    _TEMPERATURE = 0.3
    _MAX_TOKENS = 32768

    def __init__(self, llm_client: LLMClient, dtg_store: DTGStore):
        self._llm = llm_client
        self._dtg = dtg_store

    def plan_section(
        self,
        section_id: str,
        section_title: str,
        task_description: str,
        dsl_context: str,
        section_summaries: str,
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
        word_target: Optional[int] = None,
    ) -> SectionIntent:
        """
        为指定节生成 SectionIntent，并注册到 DTG

        功能：
            1. 构建规划 prompt，调用 LLM 生成结构化局部计划
            2. 解析 JSON 格式输出为 SectionIntent
            3. 将 SectionIntent 注册为 DTG 的 intent_node（建立 DERIVED_FROM 边）

        参数：
            section_id: 当前节 ID
            section_title: 当前节大纲标题
            task_description: 全局任务描述
            dsl_context: 按 salience 排序后注入的 DSL 上下文字符串
            section_summaries: 已完成章节的内容摘要
            source_dsl_entry_ids: 生成此规划时引用的 DSL 条目 ID 列表
            dsl_trust_at_generation: 生成时 DSL 的整体可信度（memory_trust_level）

        返回值：
            SectionIntent：生成的局部计划对象

        关键实现细节：
            intent_node 的置信度由 dsl_trust_at_generation 决定。
            解析失败时返回保守默认 SectionIntent（local_goal 填充 section_title）。
        """
        prompt = self._build_prompt(
            section_title=section_title,
            task_description=task_description,
            dsl_context=dsl_context,
            section_summaries=section_summaries,
            word_target=word_target,
        )

        raw = self._llm.generate(
            prompt,
            temperature=self._TEMPERATURE,
            max_tokens=self._MAX_TOKENS,
            strip_think=True,
            allow_think_only_fallback=False,
            log_meta={
                "component": "SectionPlanner",
                "section_id": section_id,
                "intent_title": section_title,
            },
        )

        intent = self._parse_intent(
            raw=raw,
            section_id=section_id,
            source_dsl_entry_ids=source_dsl_entry_ids,
            dsl_trust_at_generation=dsl_trust_at_generation,
            word_target=word_target,
        )

        # 注册到 DTG 为 intent_node
        self._dtg.add_intent_node(
            section_id=section_id,
            intent_content=intent.to_prompt_text(),
            source_dsl_entry_ids=source_dsl_entry_ids,
            confidence=dsl_trust_at_generation,
        )

        return intent

    def _build_prompt(
        self,
        section_title: str,
        task_description: str,
        dsl_context: str,
        section_summaries: str,
        word_target: Optional[int] = None,
    ) -> str:
        """
        构建 SectionIntent 生成 prompt

        参数：
            section_title: 当前节大纲标题
            task_description: 全局任务描述
            dsl_context: DSL 条目注入文本
            section_summaries: 已完成章节摘要

        返回值：
            str：完整 prompt 字符串
        """
        summaries_block = (
            f"Completed section summaries:\n{section_summaries}\n\n"
            if section_summaries
            else ""
        )
        dsl_block = (
            f"Current DSL state (top 8 entries by salience):\n{dsl_context}\n\n"
            if dsl_context
            else ""
        )

        word_target_line = (
            f"Word count target for this section: approximately {word_target} words.\n"
            if word_target is not None else ""
        )
        return (
            "Create a local plan for the section that will be written next. "
            "Output only one JSON object. Do not use markdown code fences, XML, or extra explanation.\n\n"
            f"Task goal: {task_description}\n"
            f"Section responsibility: {section_title}\n"
            f"{word_target_line}"
            f"\n{dsl_block}"
            f"{summaries_block}"
            "[Important principles]\n"
            "1. Keep the local plan strictly inside this section's responsibility; do not plan material that belongs to later sections.\n"
            "2. If there is a main conflict, this section may advance it but must not fully resolve it unless the outline explicitly marks this section as the ending.\n"
            "3. Do not repeat content that has already been handled in completed sections.\n"
            "4. The success_criteria must include an explicit word count requirement matching the word count target above.\n\n"
            "Output format (strict JSON):\n"
            "{\n"
            "  \"local_goal\": \"...\",\n"
            "  \"scope_boundary\": \"...\",\n"
            "  \"open_loops_to_advance\": [\"...\"],\n"
            "  \"commitments_to_maintain\": [\"...\"],\n"
            "  \"risks_to_avoid\": [\"...\"],\n"
            "  \"success_criteria\": [\"...\"]\n"
            "}\n"
            "If a list field is empty, return []. Do not write \"none\". Do not output markdown code fences."
        )

    def _parse_intent(
        self,
        raw: str,
        section_id: str,
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
        word_target: Optional[int] = None,
    ) -> SectionIntent:
        """
        解析 LLM 输出的 XML 格式 SectionIntent

        功能：
            提取五个 XML 字段，分号分隔的列表项转为 Python 列表。
            解析失败时返回保守默认值（使用 section_id 作为 local_goal）。

        参数：
            raw: LLM 原始输出
            section_id: 当前节 ID
            source_dsl_entry_ids: DSL 来源条目 ID 列表
            dsl_trust_at_generation: 生成时 DSL 可信度

        返回值：
            SectionIntent 实例
        """
        payload = self._extract_json_object(raw)
        if not payload:
            return self._build_default_intent(section_id, source_dsl_entry_ids, dsl_trust_at_generation, word_target)

        data = self._safe_parse_intent_json(payload)
        if data is None:
            return self._build_default_intent(section_id, source_dsl_entry_ids, dsl_trust_at_generation, word_target)

        def _norm_list(value) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        def _first_value(*keys: str):
            for key in keys:
                if key in data:
                    return data.get(key)
            return None

        local_goal = str(_first_value("local_goal", "goal") or "").strip() or f"Complete the content for section {section_id}"
        scope_boundary = str(_first_value("scope_boundary", "scope") or "").strip()
        open_loops = _norm_list(_first_value("open_loops_to_advance", "open_loops"))
        commitments = _norm_list(_first_value("commitments_to_maintain", "commitments_to_preserve"))
        risks = _norm_list(_first_value("risks_to_avoid", "risks"))
        success_criteria = _norm_list(_first_value("success_criteria", "criteria")) or ["The content matches the section goal and does not violate major constraints."]

        return SectionIntent.create(
            section_id=section_id,
            local_goal=local_goal,
            scope_boundary=scope_boundary,
            open_loops_to_advance=open_loops,
            commitments_to_maintain=commitments,
            risks_to_avoid=risks,
            success_criteria=success_criteria,
            source_dsl_entry_ids=source_dsl_entry_ids,
            dsl_trust_at_generation=dsl_trust_at_generation,
            word_target=word_target,
        )

    def _extract_json_object(self, raw: str) -> Optional[str]:
        """提取首个合法 JSON 对象文本，失败返回 None"""
        if not raw:
            return None
        raw = raw.strip()
        # 先尝试整体解析
        try:
            json.loads(raw)
            return raw
        except Exception:
            pass

        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        for idx in range(start, len(raw)):
            ch = raw[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start:idx + 1]
        return None

    def _safe_parse_intent_json(self, payload: str) -> Optional[Dict]:
        """解析 JSON，失败返回 None"""
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            repaired = self._repair_common_json_escapes(payload)
            if repaired == payload:
                return None
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    def _repair_common_json_escapes(self, payload: str) -> str:
        """
        Repair common near-JSON escape mistakes emitted by models.

        The main observed failure mode is backslash-escaped apostrophes inside
        double-quoted JSON strings, such as Alzheimer\'s, which is invalid JSON.
        """
        return re.sub(r"\\'", "'", payload)

    def _build_default_intent(
        self,
        section_id: str,
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
        word_target: Optional[int] = None,
    ) -> SectionIntent:
        """构造保守默认 SectionIntent"""
        return SectionIntent.create(
            section_id=section_id,
            local_goal=f"Complete the content for section {section_id}",
            scope_boundary="",
            open_loops_to_advance=[],
            commitments_to_maintain=[],
            risks_to_avoid=[],
            success_criteria=["The content matches the section goal and does not violate major constraints."],
            source_dsl_entry_ids=source_dsl_entry_ids,
            dsl_trust_at_generation=dsl_trust_at_generation,
            word_target=word_target,
        )
