"""
支持子集构建器：SupportSubsetBuilder

功能：
    根据 SectionIntent，从 DTG / DSL / Neocortex 中召回可用于生成 prompt 的支持子集。
    正式流程：
        Step1: 根据 SectionIntent 构造 query，并在 DTG 中召回语义相关锚点节点
        Step2: 从锚点沿高权边扩展
        Step3: 结构补全 + DSL 回溯筛选
        Step4: 检索新皮层知识
        Step5: 返回可用于 prompt 的支持子集结构

被依赖：Orchestrator
"""

import logging
from typing import Any, Dict, List, Set, Tuple


class SupportSubsetBuilder:
    """
    支持子集构建器

    依赖：
        dtg_store: DTG 存储，提供加权边读取能力
        discourse_ledger: DSL 账本，提供 get_entry()
        neocortex_store: 新皮层知识存储，提供 retrieve_items()
        similarity_service: 语义相似度服务，提供 compute_similarity()
    """

    def __init__(self, dtg_store, discourse_ledger, neocortex_store, similarity_service):
        self.dtg_store = dtg_store
        self.discourse_ledger = discourse_ledger
        self.neocortex_store = neocortex_store
        self.similarity_service = similarity_service
        self.logger = logging.getLogger(__name__)

    def build_query_from_intent(self, section_intent) -> str:
        """根据 SectionIntent 拼接语义检索 query。"""
        parts: List[str] = []

        local_goal = getattr(section_intent, "local_goal", "")
        if local_goal:
            parts.append(str(local_goal))

        open_loops = getattr(section_intent, "open_loops_to_advance", [])
        if open_loops:
            if isinstance(open_loops, (list, tuple, set)):
                parts.extend(str(item) for item in open_loops if item)
            else:
                parts.append(str(open_loops))

        commitments = getattr(section_intent, "commitments_to_maintain", [])
        if commitments:
            if isinstance(commitments, (list, tuple, set)):
                parts.extend(str(item) for item in commitments if item)
            else:
                parts.append(str(commitments))

        return "\n".join(parts)

    def compute_similarity(self, query: str, text: str) -> float:
        """调用外部相似度服务，不在这里实现替代算法。"""
        if self.similarity_service is None:
            raise ValueError("similarity_service is not configured")
        return self.similarity_service.compute_similarity(query, text)

    def collect_retrievable_dtg_nodes(self) -> List[Dict]:
        """收集适合作为语义锚点的 DTG 节点。"""
        nodes: List[Dict] = []

        # intent_node 有自然语言 content，适合作为 anchor。
        for raw_node in self.dtg_store.intent_by_section.values():
            if isinstance(raw_node, dict) and "id" in raw_node:
                nodes.append({
                    "id": raw_node["id"],
                    "type": "intent_node",
                    "raw": raw_node,
                })

        # decision_node 有 decision / reasoning / expected_effect，适合作为 anchor。
        for decision in self.dtg_store.decision_log:
            nodes.append({
                "id": decision.decision_id,
                "type": "decision_node",
                "raw": decision,
            })

        return nodes

    def stringify_node(self, node: Dict) -> str:
        """把可检索节点转换为语义相似度输入文本。"""
        node_type = node.get("type")
        raw = node.get("raw")

        if node_type == "intent_node" and isinstance(raw, dict):
            return str(raw.get("content", ""))

        if node_type == "decision_node":
            return "\n".join([
                str(getattr(raw, "decision", "")),
                str(getattr(raw, "reasoning", "")),
                str(getattr(raw, "expected_effect", "")),
            ]).strip()

        return ""

    def retrieve_anchor_nodes(self, section_intent, top_k: int = 2) -> List[Dict]:
        """Step1: 在 DTG 中召回语义相关锚点节点。"""
        query_text = self.build_query_from_intent(section_intent)
        candidate_nodes = self.collect_retrievable_dtg_nodes()
        scored_nodes: List[Dict] = []

        for node in candidate_nodes:
            node_text = self.stringify_node(node)
            if not node_text.strip():
                continue

            score = self.compute_similarity(query_text, node_text)
            scored_nodes.append({
                "node_id": node["id"],
                "node_type": node["type"],
                "score": score,
                "text": node_text,
            })

        scored_nodes.sort(key=lambda item: item["score"], reverse=True)
        return scored_nodes[:top_k]

    def expand_from_anchor_nodes(
        self,
        anchor_nodes,
        node_budget: int = 6,
    ) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str, str]]]:
        """Step2: 从锚点沿高权重边扩展 DTG 支持节点。"""
        max_hops = 2
        top_b = 2
        hop_thresholds = {1: 0.40, 2: 0.55}

        selected_nodes: Set[Tuple[str, str]] = set()
        selected_edges: Set[Tuple[str, str, str]] = set()
        frontier: List[str] = []

        for anchor in anchor_nodes:
            node_id = anchor.get("node_id")
            if not node_id:
                continue
            node_type = anchor.get("node_type") or self.infer_node_type(node_id)
            selected_nodes.add((node_id, node_type))
            frontier.append(node_id)

        if len(selected_nodes) >= node_budget:
            return selected_nodes, selected_edges

        for hop in range(1, max_hops + 1):
            threshold = hop_thresholds[hop]
            next_frontier: List[str] = []

            for current_node_id in frontier:
                weighted_edges = self.dtg_store.get_adjacent_weighted_edges(
                    current_node_id,
                    dsl_store=self.discourse_ledger,
                )
                candidate_edges = [
                    edge for edge in weighted_edges
                    if edge.get("weight", 0.0) >= threshold
                ]
                candidate_edges.sort(key=lambda edge: edge.get("weight", 0.0), reverse=True)

                for edge in candidate_edges[:top_b]:
                    source = edge["source"]
                    target = edge["target"]
                    edge_type = edge["type"]
                    neighbor_id = self.get_neighbor_node(current_node_id, edge)
                    neighbor_type = self.infer_node_type(neighbor_id)

                    selected_edges.add((source, target, edge_type))
                    selected_nodes.add((neighbor_id, neighbor_type))

                    if neighbor_id not in next_frontier:
                        next_frontier.append(neighbor_id)

                    if len(selected_nodes) >= node_budget:
                        return selected_nodes, selected_edges

            frontier = next_frontier
            if not frontier:
                break

        return selected_nodes, selected_edges

    def infer_node_type(self, node_id: str) -> str:
        """根据 node_id 和现有存储推断节点类型。"""
        if node_id.startswith("intent:"):
            return "intent_node"
        if node_id.startswith("content:"):
            return "content_node"
        if node_id in self.dtg_store.decision_by_id:
            return "decision_node"

        get_entry = getattr(self.discourse_ledger, "get_entry", None)
        if callable(get_entry) and get_entry(node_id) is not None:
            return "dsl_entry"

        items_obj = getattr(self.neocortex_store, "items", {})
        if isinstance(items_obj, dict):
            if node_id in items_obj:
                return "neocortex_item"
        else:
            for item in items_obj:
                if getattr(item, "item_id", None) == node_id:
                    return "neocortex_item"

        return "unknown"

    def get_neighbor_node(self, current_node_id: str, edge: Dict) -> str:
        """给定当前节点和边，返回边另一端的节点。"""
        if edge["source"] == current_node_id:
            return edge["target"]
        return edge["source"]

    def structural_completion_and_dsl_backtracking(
        self,
        selected_nodes,
        selected_edges,
        query_text: str,
        dsl_budget: int = 4,
    ) -> Tuple[Set[Tuple[str, str]], List[Dict]]:
        """Step3: 结构补全，并沿 intent_node 回溯筛选 DSL。"""
        completed_nodes: Set[Tuple[str, str]] = set(selected_nodes)

        # 结构补全：content_node 补 parent decision / intent。
        for node_id, node_type in list(completed_nodes):
            if node_type == "content_node":
                section_id = node_id.replace("content:", "", 1)

                parent_decision = self.dtg_store.get_decision_for_section(section_id)
                if parent_decision is not None:
                    completed_nodes.add((parent_decision.decision_id, "decision_node"))

                parent_intent = self.dtg_store.get_intent_node(section_id)
                if parent_intent is not None and "id" in parent_intent:
                    completed_nodes.add((parent_intent["id"], "intent_node"))

            elif node_type == "decision_node":
                decision = self.dtg_store.decision_by_id.get(node_id)
                if decision is not None:
                    intent = self.dtg_store.get_intent_node(decision.target_section)
                    if intent is not None and "id" in intent:
                        completed_nodes.add((intent["id"], "intent_node"))

        intent_node_ids = {
            node_id
            for node_id, node_type in completed_nodes
            if node_type == "intent_node"
        }

        dsl_support_count: Dict[str, int] = {}
        for intent_node_id in intent_node_ids:
            for dsl_id in self.dtg_store.derived_from_edges.get(intent_node_id, []):
                dsl_support_count[dsl_id] = dsl_support_count.get(dsl_id, 0) + 1

        selected_dsl: List[Dict] = []
        for dsl_id, support_count in dsl_support_count.items():
            entry = self.discourse_ledger.get_entry(dsl_id)
            if entry is None or not entry.is_active():
                continue

            dsl_text = entry.content
            sim_score = self.compute_similarity(query_text, dsl_text)
            support_score = 0.5 if support_count == 1 else 1.0
            final_score = 0.8 * sim_score + 0.2 * support_score

            selected_dsl.append({
                "dsl_id": entry.entry_id,
                "score": final_score,
                "text": entry.content,
                "entry": entry,
            })

        selected_dsl.sort(key=lambda item: item["score"], reverse=True)
        return completed_nodes, selected_dsl[:dsl_budget]

    def retrieve_neocortex_items(self, query_text: str) -> Tuple[List[Dict], List[str]]:
        """Step4: 调用新皮层检索接口。"""
        retrieve_items = getattr(self.neocortex_store, "retrieve_items", None)
        if not callable(retrieve_items):
            return [], []

        result = retrieve_items(
            section_query=query_text,
            active_budget=2,
            dormant_budget=1,
            archived_budget=1,
            archived_sim_gate=0.85,
        )

        if isinstance(result, tuple) and len(result) == 2:
            return result

        neo_items = result
        neo_ids = [
            item["item_id"]
            for item in neo_items
            if isinstance(item, dict) and "item_id" in item
        ]
        return neo_items, neo_ids

    def _preview(self, text: str, limit: int = 120) -> str:
        return str(text).replace("\n", " ")[:limit]

    def _compact_anchor(self, anchor: Dict) -> Dict:
        return {
            "node_id": anchor.get("node_id"),
            "node_type": anchor.get("node_type"),
            "score": round(float(anchor.get("score", 0.0)), 4),
            "text_preview": self._preview(anchor.get("text", "")),
        }

    def _compact_edge(self, edge_tuple) -> Dict:
        source, target, edge_type = edge_tuple
        return {
            "source": source,
            "target": target,
            "type": edge_type,
        }

    def _compact_dsl(self, dsl_item: Dict) -> Dict:
        return {
            "dsl_id": dsl_item.get("dsl_id"),
            "score": round(float(dsl_item.get("score", 0.0)), 4),
            "text_preview": self._preview(dsl_item.get("text", "")),
        }

    def build(self, section_intent, node_budget: int = 6, dsl_budget: int = 4) -> Dict:
        """Step5: 构建可用于 prompt 的支持子集结构。"""
        query_text = self.build_query_from_intent(section_intent)
        anchor_nodes = self.retrieve_anchor_nodes(section_intent, top_k=2)
        selected_nodes, selected_edges = self.expand_from_anchor_nodes(
            anchor_nodes,
            node_budget=node_budget,
        )
        completed_nodes, selected_dsl = self.structural_completion_and_dsl_backtracking(
            selected_nodes,
            selected_edges,
            query_text,
            dsl_budget=dsl_budget,
        )
        neo_items, neo_ids = self.retrieve_neocortex_items(query_text)

        node_ids = [node_id for node_id, _ in completed_nodes]
        node_ids.extend(dsl_item["dsl_id"] for dsl_item in selected_dsl)
        node_ids.extend(neo_ids)
        unique_node_ids = list(dict.fromkeys(node_ids))

        return {
            "query_text": query_text,
            "anchor_nodes": anchor_nodes,
            "dtg_nodes": completed_nodes,
            "selected_edges": selected_edges,
            "dsl_entries": selected_dsl,
            "neocortex_items": neo_items,
            "node_ids": unique_node_ids,
            "debug_trace": {
                "query_text": query_text,
                "anchor_nodes": [
                    self._compact_anchor(anchor) for anchor in anchor_nodes
                ],
                "dtg_nodes": [
                    {"node_id": node_id, "node_type": node_type}
                    for node_id, node_type in completed_nodes
                ],
                "selected_edges": [
                    self._compact_edge(edge) for edge in selected_edges
                ],
                "dsl_entries": [
                    self._compact_dsl(item) for item in selected_dsl
                ],
                "neocortex_debug": getattr(
                    self.neocortex_store,
                    "last_retrieval_debug",
                    {},
                ),
                "final_node_ids": list(dict.fromkeys(node_ids)),
            },
        }
