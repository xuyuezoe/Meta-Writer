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
            f"已完成章节摘要：\n{section_summaries}\n\n"
            if section_summaries
            else ""
        )
        dsl_block = (
            f"当前话语状态（按显著性筛选的前 8 条）：\n{dsl_context}\n\n"
            if dsl_context
            else ""
        )

        return (
            "根据以下信息，为即将生成的章节制定局部计划。"
            "请只输出一个 JSON 对象，不要 markdown 代码块、不要 XML、不要额外解释。\n\n"
            f"任务目标：{task_description}\n"
            f"本节大纲职责：{section_title}\n\n"
            f"{dsl_block}"
            f"{summaries_block}"
            "【重要原则】\n"
            "1. 本节局部计划必须严格限定在本节大纲职责范围内，不得规划属于后续章节的内容。\n"
            "2. 如果故事中存在主要冲突，本节应推进该冲突，但不得在本节内完整解决；"
            "除非本节大纲明确标注为结局节。\n"
            "3. 已完成章节中已处理的内容不应在本节重复叙述。\n\n"
            "输出格式（严格 JSON）：\n"
            "{\n"
            "  \"local_goal\": \"...\",\n"
            "  \"scope_boundary\": \"...\",\n"
            "  \"open_loops_to_advance\": [\"...\"],\n"
            "  \"commitments_to_maintain\": [\"...\"],\n"
            "  \"risks_to_avoid\": [\"...\"],\n"
            "  \"success_criteria\": [\"...\"]\n"
            "}\n"
            "若无内容，数组字段必须返回 []，不要写 \"无\"；不要输出 markdown 代码块。"
        )

    def _parse_intent(
        self,
        raw: str,
        section_id: str,
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
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
            return self._build_default_intent(section_id, source_dsl_entry_ids, dsl_trust_at_generation)

        data = self._safe_parse_intent_json(payload)
        if data is None:
            return self._build_default_intent(section_id, source_dsl_entry_ids, dsl_trust_at_generation)

        def _norm_list(value) -> List[str]:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            return []

        local_goal = str(data.get("local_goal", "")).strip() or f"完成 {section_id} 节的内容生成"
        scope_boundary = str(data.get("scope_boundary", "")).strip()
        open_loops = _norm_list(data.get("open_loops_to_advance"))
        commitments = _norm_list(data.get("commitments_to_maintain"))
        risks = _norm_list(data.get("risks_to_avoid"))
        success_criteria = _norm_list(data.get("success_criteria")) or ["内容符合节目标，无严重约束违反"]

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
            return None

    def _build_default_intent(
        self,
        section_id: str,
        source_dsl_entry_ids: List[str],
        dsl_trust_at_generation: float,
    ) -> SectionIntent:
        """构造保守默认 SectionIntent"""
        return SectionIntent.create(
            section_id=section_id,
            local_goal=f"完成 {section_id} 节的内容生成",
            scope_boundary="",
            open_loops_to_advance=[],
            commitments_to_maintain=[],
            risks_to_avoid=[],
            success_criteria=["内容符合节目标，无严重约束违反"],
            source_dsl_entry_ids=source_dsl_entry_ids,
            dsl_trust_at_generation=dsl_trust_at_generation,
        )
