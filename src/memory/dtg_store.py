"""
决策追溯图存储：DTGStore

功能：
    存储和管理 Decision Trace Graph（DTG）。
    支持决策节点、意图节点（intent_node）的存储与多类型边（GENERATES /
    REFERENCES / GUIDES / DERIVED_FROM）的追踪。
    提供决策链追溯、结构性回退和图导出。

依赖：Decision（core/decision.py）
被依赖：Orchestrator、MRSD（路径检测）、SectionPlanner（写入 intent_node）
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from ..core.decision import Decision


class DTGStore:
    """
    决策追溯图存储

    核心功能：
        1. 存储和索引决策节点（decision_node）
        2. 存储和索引意图节点（intent_node）及其边（GUIDES / DERIVED_FROM）
        3. 追溯决策链（trace_decision_chain）— DFS 算法
        4. 结构性回退（rollback_to_section）— 时间戳驱动
        5. 导出 DTG 图结构（export_dtg）— 含四类边

    关键设计：
        多维索引（section_to_decision / decision_by_id / intent_by_section）。
        intent_node 是 plan_level 诊断的可追溯对象，进入 DTG 追踪。
        GUIDES 边：intent_node → decision_node（Section Intent 指导了该 Decision）。
        DERIVED_FROM 边：intent_node → DSL entry_id（Section Intent 从 DSL 派生）。
    """

    def __init__(self, storage_path: str = "./sessions", session_name: str = "session"):
        """
        初始化 DTGStore

        参数：
            storage_path: 存储目录路径
            session_name: 当前会话名称，用于文件命名和清理

        关键行为：
            启动时自动清理当前 session 的旧文件，确保每次运行干净
        """
        self.storage_path = Path(storage_path)
        self.session_name = session_name

        # 启动时清理旧文件
        self._clean_session_files()

        # 确保目录存在
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 核心存储：决策节点
        self.decision_log: List[Decision] = []
        self.section_to_decision: Dict[str, str] = {}   # section_id → decision_id
        self.decision_by_id: Dict[str, Decision] = {}   # decision_id → Decision

        # 意图节点存储：section_id → intent_node 字典
        self.intent_by_section: Dict[str, Dict] = {}

        # GUIDES 边：intent_node_id → List[decision_id]
        self.guides_edges: Dict[str, List[str]] = {}

        # DERIVED_FROM 边：intent_node_id → List[dsl_entry_id]
        self.derived_from_edges: Dict[str, List[str]] = {}

        # 回退历史
        self.rollback_history: List[Dict] = []

        self.logger = logging.getLogger(__name__)

    def _clean_session_files(self) -> None:
        """
        清理当前 session 的所有旧文件

        删除：
            - {session_name}.json
            - {session_name}_dtg.json
            - {session_name}_metadata.json

        保留其他 session 的文件（如果有）
        """
        if not self.storage_path.exists():
            return

        patterns = [
            f"{self.session_name}.json",
            f"{self.session_name}_dtg.json",
            f"{self.session_name}_metadata.json",
        ]

        for pattern in patterns:
            filepath = self.storage_path / pattern
            if filepath.exists():
                filepath.unlink()
                print(f"已清理旧文件: {filepath}")

    # ------------------------------------------------------------------
    # 第一阶段：写入
    # ------------------------------------------------------------------

    def add_decision(self, decision: Decision) -> None:
        """
        添加决策节点并建立索引

        功能：
            写入决策到 decision_log，建立 section → decision 和 id → decision 索引。
            若当前节已有 intent_node，自动建立 GUIDES 边。

        参数：
            decision: 待写入的 Decision 对象
        """
        self.decision_log.append(decision)
        self.decision_by_id[decision.decision_id] = decision
        self.section_to_decision[decision.target_section] = decision.decision_id

        # 自动建立 GUIDES 边（若当前节已有 intent_node）
        intent_node_id = f"intent:{decision.target_section}"
        if intent_node_id in {f"intent:{sid}" for sid in self.intent_by_section}:
            self.guides_edges.setdefault(intent_node_id, [])
            if decision.decision_id not in self.guides_edges[intent_node_id]:
                self.guides_edges[intent_node_id].append(decision.decision_id)

        self.logger.debug(
            "添加决策 id=%s target=%s refs=%d",
            decision.decision_id,
            decision.target_section,
            decision.get_reference_count(),
        )

    def add_intent_node(
        self,
        section_id: str,
        intent_content: str,
        source_dsl_entry_ids: List[str],
        confidence: float,
    ) -> str:
        """
        添加 Section Intent 作为 intent_node 进入 DTG 追踪

        功能：
            创建 intent_node 并存储，建立 DERIVED_FROM 边（intent → DSL 条目）。
            intent_node_id 格式为 "intent:{section_id}"。

        参数：
            section_id: 对应的节 ID
            intent_content: Section Intent 的自然语言描述（from to_prompt_text()）
            source_dsl_entry_ids: 生成此 Section Intent 时引用的 DSL 条目 ID 列表
            confidence: intent_node 的初始置信度（由 DSL trust_level 均值决定）

        返回值：
            str：intent_node_id（"intent:{section_id}"）

        关键实现细节：
            DERIVED_FROM 边记录 intent_node 与 DSL 条目的派生关系，
            用于 plan_level 诊断时判断是否被低 trust DSL 污染。
        """
        intent_node_id = f"intent:{section_id}"

        node = {
            "id": intent_node_id,
            "type": "intent_node",
            "section_id": section_id,
            "content": intent_content,
            "source_dsl_entries": source_dsl_entry_ids,
            "confidence": confidence,
            "timestamp": int(time.time()),
        }
        self.intent_by_section[section_id] = node

        # 建立 DERIVED_FROM 边
        self.derived_from_edges[intent_node_id] = list(source_dsl_entry_ids)

        # 初始化 GUIDES 边（后续 add_decision 时填充）
        self.guides_edges.setdefault(intent_node_id, [])

        self.logger.debug(
            "添加 intent_node id=%s dsl_refs=%d conf=%.2f",
            intent_node_id,
            len(source_dsl_entry_ids),
            confidence,
        )

        return intent_node_id

    def update_intent_confidence(self, section_id: str, new_confidence: float) -> None:
        """
        更新 intent_node 的置信度（在 plan_level 诊断后调用）

        参数：
            section_id: 对应节 ID
            new_confidence: 更新后的置信度
        """
        node = self.intent_by_section.get(section_id)
        if node:
            node["confidence"] = new_confidence

    # ------------------------------------------------------------------
    # 第二阶段：查询
    # ------------------------------------------------------------------

    def find_decisions_referencing(self, section_id: str) -> List[Decision]:
        """
        找到所有引用某节的决策

        参数：
            section_id: 被引用的节 ID

        返回值：
            List[Decision]：引用了该节的决策列表
        """
        return [
            d for d in self.decision_log
            if any(ref_id == section_id for ref_id, _ in d.referenced_sections)
        ]

    def trace_decision_chain(
        self,
        section_id: str,
        max_depth: int = 10,
    ) -> List[Decision]:
        """
        追溯决策链（核心算法）

        算法：深度优先搜索（DFS）
            1. 找到生成 section_id 的决策 D
            2. 递归追溯 D 引用的所有节
            3. 去重并按时间戳升序排序

        参数：
            section_id: 起始节 ID
            max_depth: 最大追溯深度

        返回值：
            List[Decision]：按时间戳排序的决策链

        关键实现细节：
            时间复杂度 O(V+E)，visited 集合防止环路。
        """
        chain: List[Decision] = []
        visited: set = set()

        def dfs(sid: str, depth: int) -> None:
            if depth > max_depth or sid in visited:
                return
            visited.add(sid)

            decision_id = self.section_to_decision.get(sid)
            if not decision_id:
                return

            decision = self.decision_by_id.get(decision_id)
            if not decision:
                return

            chain.append(decision)
            for ref_section_id, _ in decision.referenced_sections:
                dfs(ref_section_id, depth + 1)

        dfs(section_id, 0)
        chain.sort(key=lambda d: d.timestamp)
        return chain

    def get_intent_node(self, section_id: str) -> Optional[Dict]:
        """
        获取指定节的 intent_node

        参数：
            section_id: 节 ID

        返回值：
            Dict 或 None（未找到时）
        """
        return self.intent_by_section.get(section_id)

    def get_intent_source_dsl_entries(self, section_id: str) -> List[str]:
        """
        获取指定节 intent_node 的 DSL 来源条目 ID 列表

        功能：
            用于 plan_level 诊断时判断 Section Intent 是否从可信 DSL 生成。

        参数：
            section_id: 节 ID

        返回值：
            List[str]：DSL 条目 ID 列表（未找到时返回空列表）
        """
        node = self.intent_by_section.get(section_id)
        if not node:
            return []
        return node.get("source_dsl_entries", [])

    # ------------------------------------------------------------------
    # 第三阶段：回退
    # ------------------------------------------------------------------

    def rollback_to_section(self, last_valid_section: Optional[str]) -> None:
        """
        回退到指定节之后的所有决策和意图节点

        操作：
            1. 找到 cutoff 时间戳（last_valid_section 对应决策的时间戳）
            2. 删除所有晚于 cutoff 的决策
            3. 删除对应的 intent_node 及其边
            4. 重建索引，记录回退历史

        参数：
            last_valid_section: 保留的最后一节 ID（None 表示回退到初始状态）
        """
        # 确定截断时间戳
        if last_valid_section is None:
            cutoff_ts = -1
        else:
            decision_id = self.section_to_decision.get(last_valid_section)
            if not decision_id:
                self.logger.warning("rollback: section '%s' 未找到对应决策，跳过", last_valid_section)
                return
            cutoff_ts = self.decision_by_id[decision_id].timestamp

        # 分离保留 / 丢弃
        kept = [d for d in self.decision_log if d.timestamp <= cutoff_ts]
        removed = [d for d in self.decision_log if d.timestamp > cutoff_ts]

        if not removed:
            self.logger.info("rollback: 无需删除任何决策")
            return

        removed_section_ids = {d.target_section for d in removed}

        # 记录回退历史
        self.rollback_history.append({
            "rollback_at": int(time.time()),
            "last_valid_section": last_valid_section,
            "cutoff_timestamp": cutoff_ts,
            "removed_count": len(removed),
            "removed_ids": [d.decision_id for d in removed],
        })

        # 重建决策存储和索引
        self.decision_log = kept
        self.decision_by_id = {d.decision_id: d for d in kept}
        self.section_to_decision = {d.target_section: d.decision_id for d in kept}

        # 清除被回退节的 intent_node 及其边
        for section_id in removed_section_ids:
            intent_node_id = f"intent:{section_id}"
            self.intent_by_section.pop(section_id, None)
            self.guides_edges.pop(intent_node_id, None)
            self.derived_from_edges.pop(intent_node_id, None)

        self.logger.info(
            "rollback 完成：保留 %d 条，删除 %d 条决策（cutoff_ts=%d）",
            len(kept),
            len(removed),
            cutoff_ts,
        )

    # ------------------------------------------------------------------
    # 第四阶段：导出
    # ------------------------------------------------------------------

    def export_dtg(self) -> Dict:
        """
        导出 DTG 为图结构

        返回格式：
        {
            "nodes": [decision_node + content_node + intent_node],
            "edges": [GENERATES + REFERENCES + GUIDES + DERIVED_FROM],
            "metadata": {...}
        }

        关键实现细节：
            intent_node 包含 source_dsl_entries 字段，用于 plan_level 诊断追溯。
        """
        nodes: List[Dict] = []
        edges: List[Dict] = []

        # 决策节点和内容节点
        for decision in self.decision_log:
            nodes.append({
                "id": decision.decision_id,
                "type": "decision",
                "label": decision.decision[:60],
                "confidence": decision.confidence,
                "phase": decision.phase,
                "timestamp": decision.timestamp,
            })

            content_node_id = f"content:{decision.target_section}"
            nodes.append({
                "id": content_node_id,
                "type": "content",
                "label": decision.target_section,
            })

            # GENERATES 边：Decision → Content
            edges.append({
                "source": decision.decision_id,
                "target": content_node_id,
                "type": "GENERATES",
            })

            # REFERENCES 边：Content(ref) → Decision
            for ref_section_id, snippet in decision.referenced_sections:
                edges.append({
                    "source": f"content:{ref_section_id}",
                    "target": decision.decision_id,
                    "type": "REFERENCES",
                    "snippet": snippet[:100],
                })

        # intent_node 及其边
        for section_id, node in self.intent_by_section.items():
            nodes.append(node)
            intent_node_id = node["id"]

            # GUIDES 边：intent_node → decision_node
            for decision_id in self.guides_edges.get(intent_node_id, []):
                edges.append({
                    "source": intent_node_id,
                    "target": decision_id,
                    "type": "GUIDES",
                })

            # DERIVED_FROM 边：intent_node → DSL entry_id
            for dsl_entry_id in self.derived_from_edges.get(intent_node_id, []):
                edges.append({
                    "source": intent_node_id,
                    "target": dsl_entry_id,
                    "type": "DERIVED_FROM",
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_decisions": len(self.decision_log),
                "total_intent_nodes": len(self.intent_by_section),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "rollback_count": len(self.rollback_history),
                "exported_at": int(time.time()),
            },
        }

    # ------------------------------------------------------------------
    # 第五阶段：持久化
    # ------------------------------------------------------------------

    def save_to_disk(self, filename: str = "session") -> None:
        """
        保存到磁盘（JSON 格式）

        参数：
            filename: 文件名（不含扩展名），默认 "session"
        """
        data = {
            "decision_log": [d.to_dict() for d in self.decision_log],
            "intent_nodes": {
                sid: node for sid, node in self.intent_by_section.items()
            },
            "rollback_history": self.rollback_history,
            "dtg": self.export_dtg(),
        }
        path = self.storage_path / f"{filename}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("已保存到 %s", path)

    # ------------------------------------------------------------------
    # 第六阶段：统计
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """
        获取 DTG 统计信息

        返回值：
            Dict：包含决策数、节数、回退数、intent 节点数等统计指标
        """
        if not self.decision_log:
            avg_confidence = 0.0
            avg_refs = 0.0
        else:
            avg_confidence = sum(d.confidence for d in self.decision_log) / len(self.decision_log)
            avg_refs = sum(d.get_reference_count() for d in self.decision_log) / len(self.decision_log)

        return {
            "total_decisions": len(self.decision_log),
            "total_sections": len(self.section_to_decision),
            "total_intent_nodes": len(self.intent_by_section),
            "rollback_count": len(self.rollback_history),
            "avg_confidence": round(avg_confidence, 3),
            "avg_references_per_decision": round(avg_refs, 2),
        }
