"""
承诺提取器：CommitmentExtractor

功能：
    从已生成的节内容中，通过 LLM 提取话语承诺对象，并分类为 LedgerEntry。
    是 DSL（Discourse State Ledger）的写入入口。

依赖：LLMClient、LedgerEntry、CommitmentType、ConstraintType（来自 core/ledger.py）
被依赖：DiscourseLedger（通过 Orchestrator 调用后写入）
"""
from __future__ import annotations

import json
import re
from typing import List

from ..core.ledger import CommitmentType, ConstraintType, LedgerEntry
from ..utils.llm_client import LLMClient


class CommitmentExtractor:
    """
    承诺提取器

    功能：
        调用 LLM 从节内容中提取带类型和约束强度的承诺对象，
        输出可直接写入 DiscourseLedger 的 LedgerEntry 列表。

    参数：
        llm_client: LLM 客户端实例

    关键实现细节：
        使用结构化 prompt 要求 LLM 以 JSON 数组格式输出承诺列表，
        每项包含 content / commitment_type / constraint_type 三个字段。
        对无法解析的输出进行降级处理（跳过该条目，不抛出异常）。
    """

    # 合法枚举值映射（LLM 输出 → 枚举）
    _COMMITMENT_TYPE_MAP = {
        "fact":         CommitmentType.FACT,
        "commitment":   CommitmentType.COMMITMENT,
        "open_loop":    CommitmentType.OPEN_LOOP,
        "hypothesis":   CommitmentType.HYPOTHESIS,
        "style_policy": CommitmentType.STYLE_POLICY,
    }

    _CONSTRAINT_TYPE_MAP = {
        "immutable": ConstraintType.IMMUTABLE,
        "stateful":  ConstraintType.STATEFUL,
        "soft":      ConstraintType.SOFT,
    }

    def __init__(self, llm_client: LLMClient):
        self._llm_client = llm_client

    def extract(
        self,
        section_content: str,
        section_id: str,
        decision_id: str,
        existing_summary: str = "",
    ) -> List[LedgerEntry]:
        """
        从节内容中提取承诺对象列表

        功能：
            1. 构建结构化 prompt，要求 LLM 识别五类承诺
            2. 解析 LLM 输出的 JSON 数组
            3. 将每项转换为 LedgerEntry，过滤无法解析的项

        参数：
            section_content: 当前节已生成的文本内容
            section_id: 当前节 ID（写入 LedgerEntry.source_section）
            decision_id: 生成此节的决策 ID（写入 LedgerEntry.source_decision_id）
            existing_summary: 已有内容摘要（用于上下文，提升提取准确性）

        返回值：
            List[LedgerEntry]：提取的账本条目列表（未写入 DiscourseLedger，由调用方写入）

        关键实现细节：
            trust_level 初始值由 constraint_type 决定：
                IMMUTABLE → 1.0（叙事上不可逆的设定，完全可信）
                STATEFUL  → 0.8（可合法演进的状态，中等可信）
                SOFT      → 0.6（风格偏好或非核心细节，较低初始可信度）
        """
        prompt = self._build_prompt(section_content, existing_summary)
        raw = self._llm_client.generate(prompt, temperature=0.0, max_tokens=32768)
        items = self._parse_output(raw)

        entries: List[LedgerEntry] = []
        for item in items:
            entry = self._build_entry(item, section_id, decision_id)
            if entry is not None:
                entries.append(entry)
        return entries

    def _build_prompt(self, section_content: str, existing_summary: str) -> str:
        """
        构建用于承诺提取的结构化 prompt

        参数：
            section_content: 当前节生成内容
            existing_summary: 已有章节摘要（可为空）

        返回值：
            str：完整 prompt 字符串
        """
        context_block = f"已有内容摘要：\n{existing_summary}\n\n" if existing_summary else ""
        return (
            "你是一个叙事分析助手。请从以下文本中提取所有话语承诺对象，"
            "以 JSON 数组格式输出，不要添加任何解释。\n\n"
            f"{context_block}"
            f"当前节内容：\n{section_content}\n\n"
            "每个承诺对象格式如下：\n"
            "{\n"
            '  "content": "承诺的简洁描述（不超过60字）",\n'
            '  "commitment_type": "fact"|"commitment"|"open_loop"|"hypothesis"|"style_policy",\n'
            '  "constraint_type": "immutable"|"stateful"|"soft"\n'
            "}\n\n"
            "类型说明：\n"
            "  fact         — 已确立的客观事实（人物特征、世界设定等）\n"
            "  commitment   — 对后文的明确承诺（角色行动、情节走向等）\n"
            "  open_loop    — 未解决的悬念或未完成的线索\n"
            "  hypothesis   — 角色的猜测或推断（不约束后文，不注入 prompt）\n"
            "  style_policy — 全局风格要求（叙事视角、语气、格式等）\n\n"
            "约束强度说明：\n"
            "  immutable — 叙事上不可逆的设定（角色身份、世界规则、已发生事件等），后续节不得矛盾\n"
            "  stateful  — 可随情节推进合法演变的状态（位置、关系、情绪等），更新须前后连贯\n"
            "  soft      — 风格偏好或非核心细节，可酌情调整\n\n"
            "输出 JSON 数组，无其他内容："
        )

    def _parse_output(self, raw: str) -> List[dict]:
        """
        解析 LLM 输出，提取承诺对象字典列表

        功能：
            优先尝试整体 JSON 解析，失败后用正则提取 JSON 数组块。

        参数：
            raw: LLM 原始输出字符串

        返回值：
            List[dict]：解析成功的字典列表（可能为空列表）
        """
        text = raw.strip()

        # 尝试整体解析
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (json.JSONDecodeError, ValueError):
            pass

        # 正则提取第一个 JSON 数组块
        m = re.search(r"\[.*?\]", text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, list):
                    return [item for item in parsed if isinstance(item, dict)]
            except (json.JSONDecodeError, ValueError):
                pass

        return []

    def _build_entry(
        self,
        item: dict,
        section_id: str,
        decision_id: str,
    ) -> LedgerEntry | None:
        """
        将 LLM 输出的单个字典转换为 LedgerEntry

        功能：
            校验 content / commitment_type / constraint_type 字段合法性，
            构建 LedgerEntry，初始 trust_level 由 constraint_type 决定。

        参数：
            item: LLM 输出的单个承诺字典
            section_id: 当前节 ID
            decision_id: 当前节对应的决策 ID

        返回值：
            LedgerEntry 或 None（字段非法时返回 None）
        """
        content = item.get("content", "").strip()
        if not content:
            return None

        raw_ct = item.get("commitment_type", "").lower()
        commitment_type = self._COMMITMENT_TYPE_MAP.get(raw_ct)
        if commitment_type is None:
            return None

        raw_const = item.get("constraint_type", "soft").lower()
        constraint_type = self._CONSTRAINT_TYPE_MAP.get(raw_const, ConstraintType.SOFT)

        # 初始 trust_level 由约束强度决定
        trust_level_map = {
            ConstraintType.IMMUTABLE: 1.0,
            ConstraintType.STATEFUL:  0.8,
            ConstraintType.SOFT:      0.6,
        }
        initial_trust = trust_level_map[constraint_type]

        return LedgerEntry.create(
            commitment_type=commitment_type,
            content=content,
            constraint_type=constraint_type,
            source_section=section_id,
            source_decision_id=decision_id,
            trust_level=initial_trust,
        )
