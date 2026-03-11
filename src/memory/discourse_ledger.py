"""
话语状态账本：DiscourseLedger

功能：
    管理长文本生成过程中所有已提取的话语承诺对象（LedgerEntry）及其关系（EntryRelation）。
    提供关系层候选剪枝、显著性评分（salience_score）、回退/清除、冲突检测等核心能力。
    DiscourseLedger 是 DSL 主贡献的核心实现载体。

依赖：LedgerEntry、EntryRelation、CommitmentType、ConstraintType（来自 core/ledger.py）
被依赖：CommitmentExtractor（写入条目）、OnlineValidator（读取注入）、
        Orchestrator（触发回退/清除）、MetaState（读取 memory_trust_level）
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional, Set, Tuple

from ..core.ledger import (
    CommitmentType,
    ConstraintType,
    EntryRelation,
    LedgerEntry,
)
from ..utils.llm_client import LLMClient


class DiscourseLedger:
    """
    话语状态账本

    功能：
        1. 管理 LedgerEntry 对象的生命周期（写入、撤销、解析、更新稳定性）
        2. 维护 EntryRelation 图，支持级联撤销、冲突传播、OPEN_LOOP 自动闭合
        3. 在入账前用候选剪枝规则（R1-R4）过滤候选对，再触发 LLM 关系判断
        4. 运行时计算每条可注入条目的 salience_score，返回前 K 条用于 prompt 注入
        5. 支持整节回退（rollback）和精确记忆清除（purge）

    参数：
        llm_client: 用于关系提取的 LLM 客户端（可为 None，关系提取退化为规则模式）
        max_inject_entries: 每次注入的最大条目数（默认 8）

    关键实现细节：
        关系提取只对 COMMITMENT 和 OPEN_LOOP 类型执行。
        HYPOTHESIS 类型不注入 prompt。
        salience_score 运行时计算，不静态存储在 LedgerEntry 中。
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_inject_entries: int = 8,
    ):
        self._llm_client = llm_client
        self.max_inject_entries = max_inject_entries

        # 主存储：entry_id → LedgerEntry
        self._entries: Dict[str, LedgerEntry] = {}

        # 关系图：entry_id → List[EntryRelation]（按 source_id 索引）
        self._relations: Dict[str, List[EntryRelation]] = {}

        # 反向索引：target_id → List[EntryRelation]（用于级联查找）
        self._relations_by_target: Dict[str, List[EntryRelation]] = {}

        # 历史失败关联：entry_id → 失败次数（用于 salience failure_history 因子）
        self._failure_associations: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 第一阶段：条目写入
    # ------------------------------------------------------------------

    def add_entry(self, entry: LedgerEntry) -> List[str]:
        """
        写入新账本条目，并触发关系提取（如适用）

        功能：
            1. 写入前检测与现有条目的 conflicts 关系
            2. 写入条目
            3. 若类型为 COMMITMENT 或 OPEN_LOOP，对候选对执行关系提取
            4. 若新条目与某 OPEN_LOOP 存在 resolves 关系，自动闭合

        参数：
            entry: 待写入的账本条目

        返回值：
            List[str]：写入过程中检测到的冲突描述列表（空列表表示无冲突）

        关键实现细节：
            冲突检测使用已有 conflicts 关系扫描，不额外触发 LLM。
        """
        # 第一步：冲突检测（扫描现有 conflicts 关系）
        conflict_warnings: List[str] = []
        existing_conflicts = self._find_existing_conflicts(entry)
        for conflicting_id in existing_conflicts:
            conflicting = self._entries.get(conflicting_id)
            if conflicting and conflicting.is_active():
                conflict_warnings.append(
                    f"新条目与 [{conflicting_id}] 存在冲突：{conflicting.content[:60]}"
                )

        # 第二步：写入
        self._entries[entry.entry_id] = entry
        self._relations.setdefault(entry.entry_id, [])
        self._relations_by_target.setdefault(entry.entry_id, [])

        # 第三步：关系提取（仅 COMMITMENT 和 OPEN_LOOP）
        if entry.commitment_type in (CommitmentType.COMMITMENT, CommitmentType.OPEN_LOOP):
            candidates = self._get_relation_candidates(entry)
            for candidate_id in candidates:
                self._extract_relation(entry.entry_id, candidate_id)

        # 第四步：检查此条目是否自动闭合某个 OPEN_LOOP
        self._check_resolves_open_loops(entry)

        return conflict_warnings

    def _find_existing_conflicts(self, new_entry: LedgerEntry) -> List[str]:
        """
        扫描现有关系图，找到与 new_entry 内容可能冲突的条目 ID

        功能：
            使用 R1（关键词重叠）快速筛选候选，再检查已有 conflicts 边。

        参数：
            new_entry: 待入账的新条目

        返回值：
            List[str]：与新条目存在 conflicts 关系的现有条目 ID 列表
        """
        new_keywords = self._extract_keywords(new_entry.content)
        conflicting: List[str] = []

        for eid, relations in self._relations.items():
            entry = self._entries.get(eid)
            if not entry or not entry.is_active():
                continue
            existing_keywords = self._extract_keywords(entry.content)
            if self._jaccard(new_keywords, existing_keywords) > 0.2:
                for rel in relations:
                    if rel.relation_type == "conflicts" and rel.target_id not in conflicting:
                        conflicting.append(rel.target_id)
        return conflicting

    # ------------------------------------------------------------------
    # 第二阶段：候选剪枝与关系提取
    # ------------------------------------------------------------------

    def _get_relation_candidates(self, entry: LedgerEntry) -> List[str]:
        """
        使用 R1-R4 规则筛选与 entry 可能存在关系的候选条目 ID

        功能：
            在触发 LLM 判断之前，用纯规则方法降低候选集规模，节省 LLM 开销。

        候选剪枝规则（满足任一即进入候选集）：
            R1 关键词重叠：Jaccard > 0.2
            R2 实体重叠：共享至少一个大写词（简单命名实体近似）
            R3 时间临近：introduced_at 相差不超过 2 节（以秒估算，每节约 60s）
            R4 同线索关联：两条目均引用同一 OPEN_LOOP 的 entry_id

        参数：
            entry: 当前待判断的新条目

        返回值：
            List[str]：进入 LLM 关系判断的候选条目 ID 列表（不包含 entry 自身）
        """
        entry_keywords = self._extract_keywords(entry.content)
        entry_entities = self._extract_entities(entry.content)
        candidates: Set[str] = set()

        for eid, existing in self._entries.items():
            if eid == entry.entry_id or not existing.is_active():
                continue

            # R1：关键词重叠
            existing_keywords = self._extract_keywords(existing.content)
            if self._jaccard(entry_keywords, existing_keywords) > 0.2:
                candidates.add(eid)
                continue

            # R2：实体重叠
            existing_entities = self._extract_entities(existing.content)
            if entry_entities & existing_entities:
                candidates.add(eid)
                continue

            # R3：时间临近（2节 ≈ 120秒）
            if abs(entry.introduced_at - existing.introduced_at) <= 120:
                candidates.add(eid)
                continue

            # R4：同线索关联（共享同一 OPEN_LOOP 引用）
            if (
                entry.commitment_type == CommitmentType.OPEN_LOOP
                and existing.commitment_type == CommitmentType.OPEN_LOOP
            ):
                candidates.add(eid)

        return list(candidates)

    def _extract_relation(self, source_id: str, target_id: str) -> None:
        """
        对两个候选条目执行关系判断（LLM 或规则降级）

        功能：
            若 LLM 客户端可用，调用 LLM 判断 supports/conflicts/resolves 关系；
            否则跳过（无关系写入）。

        参数：
            source_id: 来源条目 ID（新条目）
            target_id: 目标条目 ID（现有候选条目）

        关键实现细节：
            LLM prompt 要求以 JSON 格式返回 relation_type 和 confidence。
            仅接受 confidence >= 0.5 的关系，低置信度输出直接丢弃。
        """
        if self._llm_client is None:
            return

        source = self._entries.get(source_id)
        target = self._entries.get(target_id)
        if not source or not target:
            return

        prompt = (
            "判断以下两条叙事承诺之间的关系，只能输出 JSON，不要任何解释。\n\n"
            f'条目A："{source.content}"\n'
            f'条目B："{target.content}"\n\n'
            "输出格式：\n"
            '{"relation_type": "supports"|"conflicts"|"resolves"|"none", "confidence": 0.0~1.0}\n\n'
            "语义说明：\n"
            "  supports：A 为 B 提供依据或强化 B\n"
            "  conflicts：A 与 B 相互矛盾\n"
            "  resolves：A 闭合了 B 的悬念或承诺（仅当 B 是 open_loop 时有效）\n"
            "  none：无明显关系"
        )

        try:
            raw = self._llm_client.generate(prompt, temperature=0.0, max_tokens=64)
            parsed = self._parse_relation_json(raw)
            if not parsed:
                return
            relation_type, confidence = parsed
            if relation_type == "none" or confidence < 0.5:
                return

            relation = EntryRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
            )
            self._relations[source_id].append(relation)
            self._relations_by_target[target_id].append(relation)

            # resolves 关系：自动闭合目标 OPEN_LOOP
            if relation_type == "resolves":
                target_entry = self._entries.get(target_id)
                if target_entry and target_entry.commitment_type == CommitmentType.OPEN_LOOP:
                    target_entry.is_resolved = True
                    target_entry.resolved_in = source.source_section

        except Exception:
            # LLM 调用失败时静默跳过，不影响主流程
            pass

    def _parse_relation_json(self, raw: str) -> Optional[Tuple[str, float]]:
        """
        从 LLM 输出中解析关系类型和置信度

        参数：
            raw: LLM 原始输出字符串

        返回值：
            Tuple[str, float] 或 None（解析失败时）
        """
        import json as _json
        try:
            obj = _json.loads(raw.strip())
            relation_type = obj.get("relation_type", "none")
            confidence = float(obj.get("confidence", 0.0))
            if relation_type in ("supports", "conflicts", "resolves", "none"):
                return relation_type, confidence
        except Exception:
            pass
        # 降级：正则提取
        m = re.search(r'"relation_type"\s*:\s*"(\w+)"', raw)
        m2 = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
        if m and m2:
            return m.group(1), float(m2.group(1))
        return None

    def _check_resolves_open_loops(self, new_entry: LedgerEntry) -> None:
        """
        检查 new_entry 是否通过现有 resolves 关系自动闭合某个 OPEN_LOOP

        参数：
            new_entry: 刚写入的新条目
        """
        for rel in self._relations.get(new_entry.entry_id, []):
            if rel.relation_type == "resolves":
                target = self._entries.get(rel.target_id)
                if target and target.commitment_type == CommitmentType.OPEN_LOOP:
                    target.is_resolved = True
                    target.resolved_in = new_entry.source_section

    # ------------------------------------------------------------------
    # 第三阶段：生命周期管理
    # ------------------------------------------------------------------

    def revoke_entry(self, entry_id: str, revoked_by_section: str) -> None:
        """
        撤销指定条目，并触发级联 trust_level 衰减

        功能：
            1. 标记条目为 revoked
            2. 对所有 supports(entry_id, B) 的 B 降低 trust_level（衰减 0.2）

        参数：
            entry_id: 被撤销的条目 ID
            revoked_by_section: 触发撤销的节 ID
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return
        entry.revoke(revoked_by_section)

        # 级联：所有 supports(entry_id, B) 的 B 降低 trust_level
        for rel in self._relations.get(entry_id, []):
            if rel.relation_type == "supports":
                downstream = self._entries.get(rel.target_id)
                if downstream and downstream.is_active():
                    downstream.trust_level = max(0.0, downstream.trust_level - 0.2 * rel.confidence)

    def update_entry_stability(self, section_id: str, all_sections_so_far: List[str]) -> None:
        """
        在每节生成后更新所有活跃条目的稳定性分数

        功能：
            遍历所有活跃条目，计算自引入以来经过的无冲突节数，调用 update_stability()。

        参数：
            section_id: 刚完成生成的节 ID
            all_sections_so_far: 已完成生成的所有节 ID 列表（按顺序）
        """
        for entry in self._entries.values():
            if not entry.is_active():
                continue
            if entry.source_section not in all_sections_so_far:
                continue
            src_idx = all_sections_so_far.index(entry.source_section)
            cur_idx = all_sections_so_far.index(section_id) if section_id in all_sections_so_far else src_idx
            passed_sections = max(0, cur_idx - src_idx)
            entry.update_stability(passed_sections)
            # 将当前节加入 support_span
            if section_id not in entry.support_span:
                entry.support_span.append(section_id)

    def record_failure_association(self, entry_ids: List[str]) -> None:
        """
        记录一批条目与当前验证失败的关联（用于 salience 的 failure_history 因子）

        参数：
            entry_ids: 与当前失败关联的条目 ID 列表
        """
        for eid in entry_ids:
            self._failure_associations[eid] = self._failure_associations.get(eid, 0) + 1

    # ------------------------------------------------------------------
    # 第四阶段：回退与清除
    # ------------------------------------------------------------------

    def rollback_to_section(
        self,
        cutoff_section_id: str,
        section_queue: List[str],
    ) -> List[str]:
        """
        整节回退：清除在 cutoff_section_id 之后引入的所有条目

        功能：
            删除 source_section 在 section_queue 中位于 cutoff_section_id 之后的条目。
            对 stability_score > 0.7 的被清除条目发出警告。

        参数：
            cutoff_section_id: 保留至此节（含）的最后一节 ID
            section_queue: 全局节顺序列表

        返回值：
            List[str]：被清除的高稳定性条目内容警告列表
        """
        if cutoff_section_id not in section_queue:
            return []

        cutoff_idx = section_queue.index(cutoff_section_id)
        sections_to_remove: Set[str] = set(section_queue[cutoff_idx + 1:])

        warnings: List[str] = []
        to_remove: List[str] = []

        for eid, entry in self._entries.items():
            if entry.source_section in sections_to_remove:
                if entry.stability_score > 0.7:
                    warnings.append(
                        f"高稳定性条目被清除（stability={entry.stability_score:.2f}）：{entry.content[:60]}"
                    )
                to_remove.append(eid)

        for eid in to_remove:
            del self._entries[eid]
            self._relations.pop(eid, None)
            self._relations_by_target.pop(eid, None)
            self._failure_associations.pop(eid, None)

        # 清理关系图中指向被删除条目的引用
        for eid in list(self._relations.keys()):
            self._relations[eid] = [
                r for r in self._relations[eid]
                if r.target_id not in to_remove
            ]
        for eid in list(self._relations_by_target.keys()):
            self._relations_by_target[eid] = [
                r for r in self._relations_by_target[eid]
                if r.source_id not in to_remove
            ]

        return warnings

    def purge_contaminated_entries(
        self,
        contaminated_section: str,
        conflict_description: str,
    ) -> List[str]:
        """
        精确记忆清除（对应 state_level 错误）

        功能：
            找到 source_section == contaminated_section 且 commitment_type == FACT
            且内容与 conflict_description 语义相关（关键词重叠）的条目，
            撤销这些条目并触发级联 trust_level 衰减。

        参数：
            contaminated_section: 疑似被污染的节 ID
            conflict_description: 矛盾描述（用于关键词匹配）

        返回值：
            List[str]：被撤销的条目 ID 列表
        """
        conflict_keywords = self._extract_keywords(conflict_description)
        purged: List[str] = []

        for eid, entry in self._entries.items():
            if (
                entry.source_section == contaminated_section
                and entry.commitment_type == CommitmentType.FACT
                and entry.is_active()
            ):
                entry_keywords = self._extract_keywords(entry.content)
                if self._jaccard(conflict_keywords, entry_keywords) > 0.15:
                    self.revoke_entry(eid, contaminated_section)
                    purged.append(eid)

        return purged

    # ------------------------------------------------------------------
    # 第五阶段：显著性评分与注入
    # ------------------------------------------------------------------

    def get_injectable_entries(
        self,
        target_section_idx: int,
        total_sections: int,
        recent_decision_ids: List[str],
        historical_failure_entry_ids: List[str],
        outline: Dict[str, str],
        target_section_id: str,
    ) -> List[LedgerEntry]:
        """
        获取用于 prompt 注入的账本条目（按 salience 排序取前 K 条）

        功能：
            1. 筛选可注入类型（FACT / COMMITMENT / OPEN_LOOP / STYLE_POLICY）
            2. 过滤非活跃条目
            3. 计算每条的 salience_score
            4. 返回前 max_inject_entries 条

        参数：
            target_section_idx: 目标节在全局大纲中的序号（0-based）
            total_sections: 全局大纲总节数
            recent_decision_ids: 最近 2 个决策引用的条目 ID 集合
            historical_failure_entry_ids: 历史上关联过验证失败的条目 ID 集合
            outline: 全局大纲 {section_id: section_title}
            target_section_id: 目标节 ID

        返回值：
            List[LedgerEntry]：按 salience_score 降序排列的可注入条目列表
        """
        injectable_types = {
            CommitmentType.FACT,
            CommitmentType.COMMITMENT,
            CommitmentType.OPEN_LOOP,
            CommitmentType.STYLE_POLICY,
        }
        target_title = outline.get(target_section_id, "")
        target_keywords = self._extract_keywords(target_title)
        recent_ids_set = set(recent_decision_ids)
        failure_ids_set = set(historical_failure_entry_ids)

        scored: List[Tuple[float, LedgerEntry]] = []

        for entry in self._entries.values():
            if entry.commitment_type not in injectable_types:
                continue
            if not entry.is_active():
                continue

            salience = self._compute_salience(
                entry=entry,
                target_section_idx=target_section_idx,
                total_sections=total_sections,
                recent_ids_set=recent_ids_set,
                failure_ids_set=failure_ids_set,
                target_keywords=target_keywords,
            )
            scored.append((salience, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[: self.max_inject_entries]]

    def _compute_salience(
        self,
        entry: LedgerEntry,
        target_section_idx: int,
        total_sections: int,
        recent_ids_set: Set[str],
        failure_ids_set: Set[str],
        target_keywords: Set[str],
    ) -> float:
        """
        计算单条账本条目的 salience_score（运行时计算，不静态存储）

        因子：
            1. time_relevance    ：OPEN_LOOP 随时间指数增加（越久未闭合越显著）；
                                  其他类型随时间线性衰减（用目标节相对位置估计）
            2. reference_recency ：最近 2 个决策引用过此条目 → +0.3
            3. failure_history   ：此条目曾关联过验证失败 → +0.3
            4. commitment_urgency：COMMITMENT 类型在最后 2 节时权重急剧升高
            5. outline_match     ：内容关键词与目标节大纲标题的 Jaccard 相似度

        参数：
            entry: 待评分条目
            target_section_idx: 目标节序号（0-based）
            total_sections: 总节数
            recent_ids_set: 最近 2 决策引用的条目 ID 集合
            failure_ids_set: 历史失败关联的条目 ID 集合
            target_keywords: 目标节标题关键词集合

        返回值：
            float：综合显著性分数 [0.0, 2.0+]（绝对值无意义，用于相对排序）
        """
        # 因子1：time_relevance
        if total_sections > 1:
            relative_pos = target_section_idx / (total_sections - 1)
        else:
            relative_pos = 1.0

        if entry.commitment_type == CommitmentType.OPEN_LOOP and not entry.is_resolved:
            time_relevance = 1.0 + relative_pos * 2.0   # 越晚未闭合越显著
        else:
            time_relevance = max(0.1, 1.0 - relative_pos * 0.5)  # 线性衰减

        # 因子2：reference_recency
        reference_recency = 0.3 if entry.entry_id in recent_ids_set else 0.0

        # 因子3：failure_history
        failure_count = self._failure_associations.get(entry.entry_id, 0)
        failure_history = 0.3 if failure_count > 0 else 0.0

        # 因子4：commitment_urgency（最后2节时，COMMITMENT 急剧增加）
        commitment_urgency = 0.0
        if entry.commitment_type == CommitmentType.COMMITMENT:
            remaining = total_sections - 1 - target_section_idx
            if remaining <= 2:
                commitment_urgency = 1.0 - remaining * 0.4  # 最后1节=0.6，最后2节=0.2

        # 因子5：outline_match
        entry_keywords = self._extract_keywords(entry.content)
        outline_match = self._jaccard(entry_keywords, target_keywords) if target_keywords else 0.0

        salience = (
            time_relevance
            + reference_recency
            + failure_history
            + commitment_urgency
            + outline_match
        )
        return salience

    # ------------------------------------------------------------------
    # 第六阶段：统计查询
    # ------------------------------------------------------------------

    def compute_memory_trust_level(self) -> float:
        """
        计算当前 DSL 整体可信度（所有活跃条目 trust_level 的均值）

        返回值：
            float：[0.0, 1.0]，活跃条目为空时返回 1.0
        """
        active_entries = [e for e in self._entries.values() if e.is_active()]
        if not active_entries:
            return 1.0
        return sum(e.trust_level for e in active_entries) / len(active_entries)

    def get_active_entries(self) -> List[LedgerEntry]:
        """返回所有活跃条目"""
        return [e for e in self._entries.values() if e.is_active()]

    def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        """按 ID 查询条目"""
        return self._entries.get(entry_id)

    def get_open_loops(self) -> List[LedgerEntry]:
        """返回所有未闭合的 OPEN_LOOP 条目"""
        return [
            e for e in self._entries.values()
            if e.commitment_type == CommitmentType.OPEN_LOOP
            and not e.is_resolved
            and e.is_active()
        ]

    def get_low_trust_entry_ids(self, threshold: float = 0.5) -> List[str]:
        """返回 trust_level 低于阈值的活跃条目 ID 列表"""
        return [
            eid for eid, entry in self._entries.items()
            if entry.is_active() and entry.trust_level < threshold
        ]

    def to_dict(self) -> Dict:
        """序列化为字典（用于持久化和日志）"""
        return {
            "entries": {eid: e.to_dict() for eid, e in self._entries.items()},
            "relations": {
                eid: [r.to_dict() for r in rels]
                for eid, rels in self._relations.items()
                if rels
            },
            "memory_trust_level": round(self.compute_memory_trust_level(), 3),
        }

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        """
        从文本中提取关键词（过滤停用词和短词）

        参数：
            text: 输入文本

        返回值：
            Set[str]：关键词集合（小写）
        """
        stop_words = {
            "的", "了", "是", "在", "和", "与", "或", "有", "一", "不",
            "也", "都", "但", "他", "她", "它", "我", "你", "们", "对",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
        }
        words = re.findall(r"[\u4e00-\u9fa5]+|[a-zA-Z]{3,}", text.lower())
        return {w for w in words if w not in stop_words and len(w) >= 2}

    @staticmethod
    def _extract_entities(text: str) -> Set[str]:
        """
        从文本中提取简单命名实体（大写词或中文专有词近似）

        参数：
            text: 输入文本

        返回值：
            Set[str]：命名实体集合
        """
        # 英文：连续大写开头的词
        english_entities = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text))
        # 中文：引号内的词组（简单近似）
        chinese_entities = set(re.findall('\u300c([^\u300d]{2,8})\u300d|\u201c([^\u201d]{2,8})\u201d', text))
        flat_chinese: Set[str] = set()
        for match in chinese_entities:
            for part in match:
                if part:
                    flat_chinese.add(part)
        return english_entities | flat_chinese

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        """
        计算两个集合的 Jaccard 相似度

        参数：
            set_a: 集合 A
            set_b: 集合 B

        返回值：
            float：Jaccard 相似度 [0.0, 1.0]
        """
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
