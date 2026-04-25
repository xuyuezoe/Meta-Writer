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

import json
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from ..core.ledger import (
    CommitmentType,
    ConstraintType,
    EntryRelation,
    LedgerEntry,
)
if TYPE_CHECKING:
    from ..utils.llm_client import LLMClient


class DiscourseLedger:
    """
    话语状态账本

    功能：
        1. 管理 LedgerEntry 对象的生命周期（写入、撤销、解析、更新稳定性）
        2. 维护 EntryRelation 图，支持级联撤销、冲突传播、OPEN_LOOP 自动闭合
        3. 在 section 级别批量处理关系判断，避免逐 pair 同步阻塞
        4. 运行时计算每条可注入条目的 salience_score，返回前 K 条用于 prompt 注入
        5. 支持整节回退（rollback）和精确记忆清除（purge）
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_inject_entries: int = 8,
        candidate_score_threshold: float = 1.0,
        run_logger=None,
    ):
        self._llm_client = llm_client
        self.max_inject_entries = max_inject_entries
        self.candidate_score_threshold = candidate_score_threshold
        self._run_logger = run_logger
        self.logger = logging.getLogger(__name__)

        # 主存储：entry_id → LedgerEntry
        self._entries: Dict[str, LedgerEntry] = {}

        # 关系图：entry_id → List[EntryRelation]（按 source_id 索引）
        self._relations: Dict[str, List[EntryRelation]] = {}

        # 反向索引：target_id → List[EntryRelation]（用于级联查找）
        self._relations_by_target: Dict[str, List[EntryRelation]] = {}

        # 历史失败关联：entry_id → 失败次数（用于 salience failure_history 因子）
        self._failure_associations: Dict[str, int] = {}

        # section 内待处理关系队列
        self._pending_relation_pairs: List[Tuple[str, str]] = []
        self._pending_relation_keys: Set[Tuple[str, str]] = set()
        self._relation_result_cache: Dict[Tuple[str, str], Tuple[str, float]] = {}

        self._relation_stats_total = self._new_relation_stats()
        self._relation_stats_window = self._new_relation_stats()

    # ------------------------------------------------------------------
    # 第一阶段：条目写入
    # ------------------------------------------------------------------

    def add_entry(self, entry: LedgerEntry) -> List[str]:
        """
        写入新账本条目，并为关系层生成候选与入队。

        功能：
            1. 写入前检测与现有条目的 conflicts 关系
            2. 写入条目
            3. 若类型为 COMMITMENT 或 OPEN_LOOP，对候选对执行预筛与入队
            4. 不在 add_entry() 中触发 LLM 关系判断
        """
        conflict_warnings: List[str] = []
        existing_conflicts = self._find_existing_conflicts(entry)
        for conflicting_id in existing_conflicts:
            conflicting = self._entries.get(conflicting_id)
            if conflicting and conflicting.is_active():
                conflict_warnings.append(
                    f"新条目与 [{conflicting_id}] 存在冲突：{conflicting.content[:60]}"
                )

        self._entries[entry.entry_id] = entry
        self._relations.setdefault(entry.entry_id, [])
        self._relations_by_target.setdefault(entry.entry_id, [])

        if entry.commitment_type in (CommitmentType.COMMITMENT, CommitmentType.OPEN_LOOP):
            for eid, existing in self._entries.items():
                if eid == entry.entry_id or not existing.is_active():
                    continue
                self._bump_relation_stat("raw_pairs_checked")
                keep, gate_score, signals = self._gate_relation_pair(entry, existing)
                source_id, target_id = self._canonicalize_pair_direction(entry.entry_id, eid)
                pair_key = self._make_pair_key(source_id, target_id)

                if pair_key in self._pending_relation_keys:
                    self._bump_relation_stat("pairs_dedup_skipped")
                    self._log_gate_decision(
                        section_id=entry.source_section,
                        source_id=source_id,
                        target_id=target_id,
                        keep=False,
                        gate_score=gate_score,
                        signals=signals,
                        note="dedup_skipped",
                    )
                    continue

                if pair_key in self._relation_result_cache:
                    self._bump_relation_stat("pairs_cache_hit")
                    cached = self._relation_result_cache[pair_key]
                    self._log_gate_decision(
                        section_id=entry.source_section,
                        source_id=source_id,
                        target_id=target_id,
                        keep=False,
                        gate_score=gate_score,
                        signals=signals,
                        note=f"cache_hit:{cached[0]}@{cached[1]:.2f}",
                    )
                    continue

                if keep:
                    self._pending_relation_pairs.append((source_id, target_id))
                    self._pending_relation_keys.add(pair_key)
                    self._bump_relation_stat("pairs_enqueued")
                    self._log_gate_decision(
                        section_id=entry.source_section,
                        source_id=source_id,
                        target_id=target_id,
                        keep=True,
                        gate_score=gate_score,
                        signals=signals,
                        note="enqueued",
                    )
                else:
                    self._relation_result_cache[pair_key] = ("none_prefilter", 1.0)
                    self._bump_relation_stat("pairs_prefilter_none")
                    self._log_gate_decision(
                        section_id=entry.source_section,
                        source_id=source_id,
                        target_id=target_id,
                        keep=False,
                        gate_score=gate_score,
                        signals=signals,
                        note="none_prefilter",
                    )

        return conflict_warnings

    def _find_existing_conflicts(self, new_entry: LedgerEntry) -> List[str]:
        """
        扫描现有关系图，找到与 new_entry 内容可能冲突的条目 ID。
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
    # 第二阶段：候选召回、预筛与批处理
    # ------------------------------------------------------------------

    def _get_type_pair_bonus(self, a: LedgerEntry, b: LedgerEntry) -> float:
        if a.commitment_type == CommitmentType.COMMITMENT and b.commitment_type == CommitmentType.OPEN_LOOP:
            return 1.0
        if a.commitment_type == CommitmentType.OPEN_LOOP and b.commitment_type == CommitmentType.COMMITMENT:
            return 1.0
        if a.commitment_type == CommitmentType.COMMITMENT and b.commitment_type == CommitmentType.COMMITMENT:
            return 0.5
        return 0.0

    def _make_pair_key(self, a: str, b: str) -> Tuple[str, str]:
        return tuple(sorted((a, b)))

    def _canonicalize_pair_direction(self, a: str, b: str) -> Tuple[str, str]:
        """
        统一关系判定方向。

        说明：
            该方向不是纯语义方向，而是语义规则与流水线时序规则的组合。
            当前主要目标是稳定可计算，而不是构建绝对语义真图。
        """
        entry_a = self._entries.get(a)
        entry_b = self._entries.get(b)
        if not entry_a or not entry_b:
            return a, b

        if (
            entry_a.commitment_type == CommitmentType.COMMITMENT
            and entry_b.commitment_type == CommitmentType.OPEN_LOOP
        ):
            return a, b
        if (
            entry_a.commitment_type == CommitmentType.OPEN_LOOP
            and entry_b.commitment_type == CommitmentType.COMMITMENT
        ):
            return b, a
        return a, b

    def _gate_relation_pair(
        self,
        new_entry: LedgerEntry,
        existing_entry: LedgerEntry,
    ) -> Tuple[bool, float, Dict[str, float]]:
        new_text = new_entry.content.strip()
        existing_text = existing_entry.content.strip()
        if self._is_too_short_noise_pair(new_text, existing_text):
            return False, 0.0, {
                "term_overlap_score": 0.0,
                "term_score": 0.0,
                "type_bonus": 0.0,
                "hard_drop": 1.0,
                "hard_drop_short": 1.0,
                "hard_drop_template": 0.0,
            }
        if self._is_template_noise_pair(new_text, existing_text):
            return False, 0.0, {
                "term_overlap_score": 0.0,
                "term_score": 0.0,
                "type_bonus": 0.0,
                "hard_drop": 1.0,
                "hard_drop_short": 0.0,
                "hard_drop_template": 1.0,
            }

        term_overlap_score = self._compute_term_overlap_score(new_text, existing_text)
        if term_overlap_score >= 0.5:
            term_score = 2.0
        elif term_overlap_score >= 0.25:
            term_score = 1.0
        elif term_overlap_score >= 0.1:
            term_score = 0.5
        else:
            term_score = 0.0
        type_bonus = self._get_type_pair_bonus(new_entry, existing_entry)
        gate_score = term_score + type_bonus
        keep = gate_score >= self.candidate_score_threshold
        if (
            keep
            and new_entry.commitment_type == CommitmentType.COMMITMENT
            and existing_entry.commitment_type == CommitmentType.COMMITMENT
            and term_score < 1.0
        ):
            keep = False
        return keep, gate_score, {
            "term_overlap_score": round(term_overlap_score, 3),
            "term_score": term_score,
            "type_bonus": type_bonus,
            "hard_drop": 0.0,
            "hard_drop_short": 0.0,
            "hard_drop_template": 0.0,
        }

    def _score_pending_pair_priority(self, source_id: str, target_id: str) -> float:
        source = self._entries.get(source_id)
        target = self._entries.get(target_id)
        if not source or not target:
            return float("-inf")

        priority = 0.0
        if (
            source.commitment_type == CommitmentType.COMMITMENT
            and target.commitment_type == CommitmentType.OPEN_LOOP
        ):
            priority += 2.0

        priority += self._compute_term_overlap_score(source.content, target.content) * 2.0
        priority -= self._generic_phrase_penalty(source.content)
        priority -= self._generic_phrase_penalty(target.content)

        if abs(source.introduced_at - target.introduced_at) <= 120:
            priority += 0.3

        return priority

    def _build_batch_relation_prompt(self, pairs: List[Tuple[str, str]]) -> str:
        lines = [
            "Determine the relation for each DSL entry pair below.",
            "Judge each pair independently.",
            "Allowed relation_type values are supports, conflicts, resolves, and none.",
            "Use resolves only when the target is an open_loop.",
            "Base the answer only on the two DSL entries provided. Do not add background knowledge.",
            "If the relation is unclear, return none.",
            "Return a JSON array only. Do not include explanations or markdown code fences.",
            "Each object must include source_id, target_id, relation_type, and confidence.",
            "",
            "Pairs to judge:",
        ]

        for source_id, target_id in pairs:
            source = self._entries.get(source_id)
            target = self._entries.get(target_id)
            if not source or not target:
                continue
            lines.extend(
                [
                    f"- source_id: {source_id}",
                    f"  source_type: {source.commitment_type.value}",
                    f'  source_content: "{source.content}"',
                    f"- target_id: {target_id}",
                    f"  target_type: {target.commitment_type.value}",
                    f'  target_content: "{target.content}"',
                    "",
                ]
            )

        lines.append(
            'Example output: [{"source_id":"...","target_id":"...","relation_type":"none","confidence":0.0}]'
        )
        return "\n".join(lines)

    def _parse_batch_relation_json(
        self,
        raw: str,
    ) -> List[Tuple[str, str, str, float]]:
        payload = raw.strip()
        if payload.startswith("```"):
            payload = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", payload)
            payload = re.sub(r"\n?```$", "", payload).strip()

        parsed = json.loads(payload)
        if not isinstance(parsed, list):
            raise ValueError("batch relation output is not a list")

        results: List[Tuple[str, str, str, float]] = []
        for item in parsed:
            if not isinstance(item, dict):
                raise ValueError("batch relation item is not an object")
            source_id = str(item["source_id"])
            target_id = str(item["target_id"])
            relation_type = str(item["relation_type"])
            confidence = float(item["confidence"])
            results.append((source_id, target_id, relation_type, confidence))
        return results

    def _apply_relation_result(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        confidence_threshold: float,
    ) -> None:
        if relation_type not in {"supports", "conflicts", "resolves"}:
            return

        if confidence < confidence_threshold:
            return

        relation = EntryRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
        )
        self._relations.setdefault(source_id, []).append(relation)
        self._relations_by_target.setdefault(target_id, []).append(relation)

        if relation_type == "resolves":
            target = self._entries.get(target_id)
            source = self._entries.get(source_id)
            if (
                source is not None
                and target is not None
                and target.commitment_type == CommitmentType.OPEN_LOOP
            ):
                target.is_resolved = True
                target.resolved_in = source.source_section

    def process_pending_relations(
        self,
        section_id: Optional[str] = None,
        max_pairs: int = 8,
        batch_size: int = 4,
        confidence_threshold: float = 0.5,
    ) -> Dict[str, int | float]:
        """
        批量处理待判定关系，返回本节统计。
        """
        start_time = time.time()
        section_stats = dict(self._relation_stats_window)
        section_stats.setdefault("processed", 0)
        section_stats.setdefault("skipped", 0)

        valid_pairs: List[Tuple[str, str]] = []
        for source_id, target_id in self._pending_relation_pairs:
            source = self._entries.get(source_id)
            target = self._entries.get(target_id)
            if not source or not target or not source.is_active() or not target.is_active():
                section_stats["skipped"] += 1
                continue
            valid_pairs.append((source_id, target_id))

        if self._llm_client is None:
            section_stats["processed"] = 0
            section_stats["skipped"] += len(valid_pairs)
            self._pending_relation_pairs = []
            self._pending_relation_keys.clear()
            section_stats["remaining_queue"] = 0
            section_stats["time_cost_ms"] = int((time.time() - start_time) * 1000)
            self._reset_relation_stats_window()
            return section_stats

        ranked_pairs = sorted(
            valid_pairs,
            key=lambda pair: self._score_pending_pair_priority(pair[0], pair[1]),
            reverse=True,
        )
        selected_pairs = ranked_pairs[:max_pairs]
        remaining_pairs = ranked_pairs[max_pairs:]

        self._pending_relation_pairs = remaining_pairs
        self._pending_relation_keys = {
            self._make_pair_key(source_id, target_id)
            for source_id, target_id in remaining_pairs
        }

        processed_keys = {self._make_pair_key(source_id, target_id) for source_id, target_id in selected_pairs}

        for batch_start in range(0, len(selected_pairs), max(1, batch_size)):
            batch_pairs = selected_pairs[batch_start: batch_start + max(1, batch_size)]
            if not batch_pairs:
                continue

            prompt = self._build_batch_relation_prompt(batch_pairs)
            section_stats["pairs_sent_to_llm"] += len(batch_pairs)

            try:
                raw = self._llm_client.generate(
                    prompt,
                    temperature=0.0,
                    max_tokens=32768,
                    log_meta={
                        "component": "DSLRelation",
                        "section_id": section_id,
                        "batch_size": len(batch_pairs),
                    },
                )
                batch_results = self._parse_batch_relation_json(raw)
            except Exception:
                for source_id, target_id in batch_pairs:
                    self._relation_result_cache[self._make_pair_key(source_id, target_id)] = ("none_llm", 0.0)
                    section_stats["pairs_none_llm"] += 1
                continue

            expected_pairs = {
                (source_id, target_id): self._make_pair_key(source_id, target_id)
                for source_id, target_id in batch_pairs
            }
            seen_pairs: Set[Tuple[str, str]] = set()

            for source_id, target_id, relation_type, confidence in batch_results:
                pair = (source_id, target_id)
                if pair not in expected_pairs:
                    continue

                seen_pairs.add(pair)
                pair_key = expected_pairs[pair]
                if relation_type not in {"supports", "conflicts", "resolves", "none"}:
                    relation_type = "none"
                    confidence = 0.0

                if relation_type == "none":
                    self._relation_result_cache[pair_key] = ("none_llm", confidence)
                    section_stats["pairs_none_llm"] += 1
                    self._log_relation_result(section_id, source_id, target_id, relation_type, confidence, False)
                    continue

                self._relation_result_cache[pair_key] = (relation_type, confidence)
                applied = confidence >= confidence_threshold
                self._log_relation_result(section_id, source_id, target_id, relation_type, confidence, applied)
                if confidence >= confidence_threshold:
                    self._apply_relation_result(
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation_type,
                        confidence=confidence,
                        confidence_threshold=confidence_threshold,
                    )
                    stat_key = f"pairs_{relation_type}"
                    if stat_key in section_stats:
                        section_stats[stat_key] += 1

            missing_pairs = set(expected_pairs.keys()) - seen_pairs
            for source_id, target_id in missing_pairs:
                self._relation_result_cache[self._make_pair_key(source_id, target_id)] = ("none_llm", 0.0)
                section_stats["pairs_none_llm"] += 1

        section_stats["processed"] = len(selected_pairs)
        section_stats["remaining_queue"] = len(self._pending_relation_pairs)
        section_stats["time_cost_ms"] = int((time.time() - start_time) * 1000)

        for key in self._relation_stats_total:
            self._relation_stats_total[key] += int(section_stats.get(key, 0))

        for pair_key in processed_keys:
            self._pending_relation_keys.discard(pair_key)

        self._reset_relation_stats_window()
        return section_stats

    # ------------------------------------------------------------------
    # 第三阶段：生命周期管理
    # ------------------------------------------------------------------

    def revoke_entry(self, entry_id: str, revoked_by_section: str) -> None:
        """
        撤销指定条目，并触发级联 trust_level 衰减。
        """
        entry = self._entries.get(entry_id)
        if not entry:
            return
        entry.revoke(revoked_by_section)

        for rel in self._relations.get(entry_id, []):
            if rel.relation_type == "supports":
                downstream = self._entries.get(rel.target_id)
                if downstream and downstream.is_active():
                    downstream.trust_level = max(0.0, downstream.trust_level - 0.2 * rel.confidence)

    def update_entry_stability(self, section_id: str, all_sections_so_far: List[str]) -> None:
        """
        在每节生成后更新所有活跃条目的稳定性分数。
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
            if section_id not in entry.support_span:
                entry.support_span.append(section_id)

    def record_failure_association(self, entry_ids: List[str]) -> None:
        """
        记录一批条目与当前验证失败的关联（用于 salience 的 failure_history 因子）。
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
        整节回退：清除在 cutoff_section_id 之后引入的所有条目。
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

        for eid in list(self._relations.keys()):
            self._relations[eid] = [r for r in self._relations[eid] if r.target_id not in to_remove]
        for eid in list(self._relations_by_target.keys()):
            self._relations_by_target[eid] = [r for r in self._relations_by_target[eid] if r.source_id not in to_remove]

        if to_remove:
            removed_ids = set(to_remove)
            self._pending_relation_pairs = [
                (source_id, target_id)
                for source_id, target_id in self._pending_relation_pairs
                if source_id not in removed_ids and target_id not in removed_ids
            ]
            self._pending_relation_keys = {
                self._make_pair_key(source_id, target_id)
                for source_id, target_id in self._pending_relation_pairs
            }
            self._relation_result_cache = {
                pair_key: result
                for pair_key, result in self._relation_result_cache.items()
                if pair_key[0] not in removed_ids and pair_key[1] not in removed_ids
            }

        return warnings

    def purge_contaminated_entries(
        self,
        contaminated_section: str,
        conflict_description: str,
    ) -> List[str]:
        """
        精确记忆清除（对应 state_level 错误）。
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
        获取用于 prompt 注入的账本条目（按 salience 排序取前 K 条）。
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
        计算单条账本条目的 salience_score（运行时计算，不静态存储）。
        """
        if total_sections > 1:
            relative_pos = target_section_idx / (total_sections - 1)
        else:
            relative_pos = 1.0

        if entry.commitment_type == CommitmentType.OPEN_LOOP and not entry.is_resolved:
            time_relevance = 1.0 + relative_pos * 2.0
        else:
            time_relevance = max(0.1, 1.0 - relative_pos * 0.5)

        reference_recency = 0.3 if entry.entry_id in recent_ids_set else 0.0
        failure_history = 0.3 if entry.entry_id in failure_ids_set else 0.0

        commitment_urgency = 0.0
        if entry.commitment_type == CommitmentType.COMMITMENT:
            remaining = total_sections - 1 - target_section_idx
            if remaining <= 2:
                commitment_urgency = 1.0 - remaining * 0.4

        entry_keywords = self._extract_keywords(entry.content)
        outline_match = self._jaccard(entry_keywords, target_keywords) if target_keywords else 0.0

        return (
            time_relevance
            + reference_recency
            + failure_history
            + commitment_urgency
            + outline_match
        )

    # ------------------------------------------------------------------
    # 第六阶段：统计查询
    # ------------------------------------------------------------------

    def compute_memory_trust_level(self) -> float:
        """
        计算当前 DSL 整体可信度（所有活跃条目 trust_level 的均值）。
        """
        active_entries = [e for e in self._entries.values() if e.is_active()]
        if not active_entries:
            return 1.0
        return sum(e.trust_level for e in active_entries) / len(active_entries)

    def get_active_entries(self) -> List[LedgerEntry]:
        return [e for e in self._entries.values() if e.is_active()]

    def get_entry(self, entry_id: str) -> Optional[LedgerEntry]:
        return self._entries.get(entry_id)

    def get_open_loops(self) -> List[LedgerEntry]:
        return [
            e for e in self._entries.values()
            if e.commitment_type == CommitmentType.OPEN_LOOP
            and not e.is_resolved
            and e.is_active()
        ]

    def get_low_trust_entry_ids(self, threshold: float = 0.5) -> List[str]:
        return [
            eid for eid, entry in self._entries.items()
            if entry.is_active() and entry.trust_level < threshold
        ]

    def to_dict(self) -> Dict:
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

    def _new_relation_stats(self) -> Dict[str, int]:
        return {
            "raw_pairs_checked": 0,
            "pairs_enqueued": 0,
            "pairs_dedup_skipped": 0,
            "pairs_prefilter_none": 0,
            "pairs_cache_hit": 0,
            "pairs_sent_to_llm": 0,
            "pairs_none_llm": 0,
            "pairs_supports": 0,
            "pairs_conflicts": 0,
            "pairs_resolves": 0,
        }

    def _reset_relation_stats_window(self) -> None:
        self._relation_stats_window = self._new_relation_stats()

    def _bump_relation_stat(self, key: str, amount: int = 1) -> None:
        if key in self._relation_stats_window:
            self._relation_stats_window[key] += amount

    def _log_gate_decision(
        self,
        section_id: str,
        source_id: str,
        target_id: str,
        keep: bool,
        gate_score: float,
        signals: Dict[str, float],
        note: str,
    ) -> None:
        if self._run_logger is not None:
            source = self._entries.get(source_id)
            target = self._entries.get(target_id)
            self._run_logger.log_dsl_gate_pair(
                section_id=section_id,
                source_id=source_id,
                target_id=target_id,
                keep=keep,
                gate_score=gate_score,
                signals=signals,
                note=note,
                source_type=source.commitment_type.value if source is not None else "-",
                target_type=target.commitment_type.value if target is not None else "-",
                source_content=(source.content[:80] if source is not None else ""),
                target_content=(target.content[:80] if target is not None else ""),
            )

    def _log_relation_result(
        self,
        section_id: Optional[str],
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        applied: bool,
    ) -> None:
        if self._run_logger is not None:
            self._run_logger.log_dsl_relation_result(
                section_id=section_id or "-",
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                confidence=confidence,
                applied=applied,
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip().lower())

    @classmethod
    def _extract_keyword_tokens(cls, text: str) -> Set[str]:
        normalized = cls._normalize_text(text)
        tokens: Set[str] = set()
        stop_tokens = {
            "the", "and", "for", "with", "from", "that", "this", "into", "onto", "over",
            "under", "within", "across", "about", "after", "before", "through", "during",
            "provide", "provides", "systematic", "systematically", "review", "analysis",
            "future", "section", "sections", "detailed", "comparison", "important", "issue",
            "framework", "discussion", "approach", "result", "results", "content",
        }

        english_terms = re.findall(r"\b[a-z][a-z0-9]*(?:-[a-z0-9]+)*\b", normalized)
        for token in english_terms:
            if len(token) >= 3 and token not in stop_tokens:
                tokens.add(token)

        title_acronyms = re.findall(r"\b[A-Z]{2,8}\b", text)
        tokens.update(token.lower() for token in title_acronyms)

        mixed_tokens = re.findall(r"\b[a-zA-Z]*\d+[a-zA-Z0-9-]*\b", text)
        tokens.update(token.lower() for token in mixed_tokens if len(token) >= 2)

        numeric_spans = re.findall(r"\b\d{1,4}(?:-\d{1,4})?(?:st|nd|rd|th)?\b", normalized)
        tokens.update(numeric_spans)
        return {token for token in tokens if token}

    @classmethod
    def _extract_keywords(cls, text: str) -> Set[str]:
        return cls._extract_keyword_tokens(text)

    @staticmethod
    def _overlap_ratio(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        denominator = min(len(set_a), len(set_b))
        return intersection / denominator if denominator > 0 else 0.0

    @classmethod
    def _compute_term_overlap_score(cls, source_text: str, target_text: str) -> float:
        keyword_overlap = cls._overlap_ratio(
            cls._extract_keyword_tokens(source_text),
            cls._extract_keyword_tokens(target_text),
        )
        return min(1.0, keyword_overlap)

    @classmethod
    def _is_too_short_noise_pair(cls, source_text: str, target_text: str) -> bool:
        return len(source_text.strip()) < 6 and len(target_text.strip()) < 6

    @classmethod
    def _is_template_noise_text(cls, text: str) -> bool:
        normalized = cls._normalize_text(text)
        if not normalized:
            return True
        if re.fullmatch(r"[\W_]+", normalized):
            return True
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", normalized):
            return True
        return normalized in {
            "see above",
            "same as above",
            "to be added",
            "to be completed",
            "placeholder",
            "tbd",
            "n/a",
        }

    @classmethod
    def _is_template_noise_pair(cls, source_text: str, target_text: str) -> bool:
        return cls._is_template_noise_text(source_text) and cls._is_template_noise_text(target_text)

    @classmethod
    def _generic_phrase_penalty(cls, text: str) -> float:
        normalized = cls._normalize_text(text)
        generic_phrases = {
            "provide a framework",
            "systematically review",
            "future sections",
            "detailed comparison",
            "important issue",
            "further analysis",
            "practical pathway",
            "ongoing attention",
        }
        penalty = 0.0
        for phrase in generic_phrases:
            if phrase in normalized:
                penalty += 0.4
        return penalty

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0
