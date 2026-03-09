from typing import List, Dict, Optional
from pathlib import Path
import json
import logging
import time

from ..core.decision import Decision


class DTGStore:
    """
    存储和管理Decision Trace Graph 

    核心功能：
    1. 存储和索引决策
    2. 追溯决策链（trace_decision_chain）
    3. 导出DTG结构（export_dtg）
    4. 智能回退（rollback_to_section）

    关键设计：
    1. 多维索引（支持快速查询）
    2. trace_decision_chain是DTG的核心价值
    3. rollback_to_section支持自我修正
    """

    def __init__(self, storage_path: str = "./sessions", session_name: str = "session"):
        """
        初始化DTGStore

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

        # 核心存储
        self.decision_log: List[Decision] = []

        # 索引
        self.section_to_decision: Dict[str, str] = {}   # section_id -> decision_id
        self.decision_by_id: Dict[str, Decision] = {}   # decision_id -> Decision

        # 回退历史
        self.rollback_history: List[Dict] = []

        self.logger = logging.getLogger(__name__)

    def _clean_session_files(self):
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
    # 写入
    # ------------------------------------------------------------------

    def add_decision(self, decision: Decision):
        """添加决策并建立索引"""
        self.decision_log.append(decision)
        self.decision_by_id[decision.decision_id] = decision
        self.section_to_decision[decision.target_section] = decision.decision_id
        self.logger.debug(
            "添加决策 id=%s target=%s refs=%d",
            decision.decision_id,
            decision.target_section,
            decision.get_reference_count(),
        )

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def find_decisions_referencing(self, section_id: str) -> List[Decision]:
        """找到所有引用某section的决策"""
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
        1. 找到生成section_id的决策D
        2. 递归追溯D引用的所有section
        3. 去重并按时间戳升序排序

        时间复杂度：O(V+E)
        """
        chain: List[Decision] = []
        visited: set = set()

        def dfs(sid: str, depth: int):
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

    # ------------------------------------------------------------------
    # 回退
    # ------------------------------------------------------------------

    def rollback_to_section(self, last_valid_section: Optional[str]):
        """
        回退到指定section之后的所有决策

        操作：
        1. 找到cutoff时间戳（last_valid_section对应决策的时间戳）
        2. 删除所有晚于cutoff的决策
        3. 重建索引
        4. 记录回退历史
        """
        # 确定截断时间戳
        if last_valid_section is None:
            # 回退到初始状态
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

        # 记录回退历史
        self.rollback_history.append({
            "rollback_at": int(time.time()),
            "last_valid_section": last_valid_section,
            "cutoff_timestamp": cutoff_ts,
            "removed_count": len(removed),
            "removed_ids": [d.decision_id for d in removed],
        })

        # 重建存储和索引
        self.decision_log = kept
        self.decision_by_id = {d.decision_id: d for d in kept}
        self.section_to_decision = {d.target_section: d.decision_id for d in kept}

        self.logger.info(
            "rollback完成：保留%d条，删除%d条决策（cutoff_ts=%d）",
            len(kept),
            len(removed),
            cutoff_ts,
        )

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    def export_dtg(self) -> Dict:
        """
        导出DTG为图结构

        返回格式：
        {
            "nodes": [Decision节点 + Content节点],
            "edges": [GENERATES边 + REFERENCES边],
            "metadata": {...}
        }
        """
        nodes = []
        edges = []

        for decision in self.decision_log:
            # Decision节点
            nodes.append({
                "id": decision.decision_id,
                "type": "decision",
                "label": decision.decision[:60],
                "confidence": decision.confidence,
                "phase": decision.phase,
                "timestamp": decision.timestamp,
            })

            # Content节点（每个target_section一个）
            content_node_id = f"content:{decision.target_section}"
            nodes.append({
                "id": content_node_id,
                "type": "content",
                "label": decision.target_section,
            })

            # GENERATES边：Decision -> Content
            edges.append({
                "source": decision.decision_id,
                "target": content_node_id,
                "type": "GENERATES",
            })

            # REFERENCES边：Content(ref) -> Decision
            for ref_section_id, snippet in decision.referenced_sections:
                edges.append({
                    "source": f"content:{ref_section_id}",
                    "target": decision.decision_id,
                    "type": "REFERENCES",
                    "snippet": snippet[:100],
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "total_decisions": len(self.decision_log),
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "rollback_count": len(self.rollback_history),
                "exported_at": int(time.time()),
            },
        }

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save_to_disk(self, filename: str = "session"):
        """保存到磁盘（JSON格式）"""
        data = {
            "decision_log": [d.to_dict() for d in self.decision_log],
            "rollback_history": self.rollback_history,
            "dtg": self.export_dtg(),
        }
        path = self.storage_path / f"{filename}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("已保存到 %s", path)

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.decision_log:
            avg_confidence = 0.0
            avg_refs = 0.0
        else:
            avg_confidence = sum(d.confidence for d in self.decision_log) / len(self.decision_log)
            avg_refs = sum(d.get_reference_count() for d in self.decision_log) / len(self.decision_log)

        return {
            "total_decisions": len(self.decision_log),
            "total_sections": len(self.section_to_decision),
            "rollback_count": len(self.rollback_history),
            "avg_confidence": round(avg_confidence, 3),
            "avg_references_per_decision": round(avg_refs, 2),
        }
