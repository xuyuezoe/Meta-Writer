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
        max_tokens 512（局部计划不需要长输出）。
        LLM 以 XML 格式输出 SectionIntent 字段，解析失败时返回保守默认值。
    """

    _TEMPERATURE = 0.3
    _MAX_TOKENS = 512

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
            2. 解析 XML 格式输出为 SectionIntent
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
            "请严格按照 XML 格式输出，不要任何额外文字。\n\n"
            f"任务目标：{task_description}\n"
            f"本节大纲职责：{section_title}\n\n"
            f"{dsl_block}"
            f"{summaries_block}"
            "【重要原则】\n"
            "1. 本节局部计划必须严格限定在本节大纲职责范围内，不得规划属于后续章节的内容。\n"
            "2. 如果故事中存在主要冲突，本节应推进该冲突，但不得在本节内完整解决；"
            "除非本节大纲明确标注为结局节。\n"
            "3. 已完成章节中已处理的内容不应在本节重复叙述。\n\n"
            "输出格式：\n"
            "<local_goal>本节需要实现的具体目标，不超过 50 字</local_goal>\n"
            "<scope_boundary>本节明确不应涉及的内容（避免越界），不超过 30 字</scope_boundary>\n"
            "<open_loops_to_advance>本节应推进的未闭合线索，用分号分隔，无则填「无」</open_loops_to_advance>\n"
            "<commitments_to_maintain>本节必须维护的承诺，用分号分隔，无则填「无」</commitments_to_maintain>\n"
            "<risks_to_avoid>本节需避免的高风险冲突，用分号分隔，无则填「无」</risks_to_avoid>\n"
            "<success_criteria>本节通过验证的最低标准（1-2 条），用分号分隔</success_criteria>"
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
        def extract(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
            return m.group(1).strip() if m else ""

        def split_list(text: str) -> List[str]:
            if not text or text == "无":
                return []
            return [item.strip() for item in text.split("；") if item.strip() and item.strip() != "无"]

        local_goal = extract("local_goal") or f"完成 {section_id} 节的内容生成"
        scope_boundary = extract("scope_boundary") or ""
        open_loops = split_list(extract("open_loops_to_advance"))
        commitments = split_list(extract("commitments_to_maintain"))
        risks = split_list(extract("risks_to_avoid"))
        criteria_raw = extract("success_criteria")
        success_criteria = split_list(criteria_raw) or ["内容符合节目标，无严重约束违反"]

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
