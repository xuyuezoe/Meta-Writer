"""
新皮层知识存储模块：NeocortexStore

功能：
    实现新皮层知识存储模块的基础版本，提供存储、生命周期管理、激活更新和检索接口。

依赖：
    从 ..core.neocortex 导入 NeocortexItem

注意：
    实现 consolidate_from_section_graph()。从DTG和DSL中提取知识写入新皮层
    实现基础存储、生命周期、激活更新和检索接口。
"""

from typing import Dict, List, Optional, Tuple
from ..core.neocortex import NeocortexItem


class NeocortexStore:
    """
    新皮层知识存储模块

    功能：
        管理新皮层知识的存储、生命周期、激活更新和检索。

    依赖：
        similarity_service: 相似度计算服务（外部传入）
    """

    def __init__(self, similarity_service=None):
        """
        初始化 NeocortexStore

        参数：
            similarity_service: 相似度计算服务（可选）
        """
        self.items: Dict[str, NeocortexItem] = {}
        self.similarity_service = similarity_service
        self.last_retrieval_debug = {}

    def add_item(self, item: NeocortexItem) -> None:
        """
        添加新皮层知识项

        参数：
            item: NeocortexItem 对象
        """
        self.items[item.item_id] = item

    def get_item(self, item_id: str) -> Optional[NeocortexItem]:
        """
        根据ID获取知识项

        参数：
            item_id: 知识项ID

        返回值：
            NeocortexItem 或 None（未找到时）
        """
        return self.items.get(item_id)

    def get_all_items(self) -> List[NeocortexItem]:
        """
        获取所有知识项

        返回值：
            NeocortexItem 列表
        """
        return list(self.items.values())

    def to_dict(self) -> Dict:
        """
        序列化为字典

        返回值：
            包含所有知识项的字典
        """
        return {
            "items": {item_id: item.to_dict() for item_id, item in self.items.items()},
            "total_items": len(self.items),
        }

    def update_activation_levels(
        self,
        support_item_ids: List[str],
        decay: float = 0.8,
        retrieve_gain: float = 0.2,
    ) -> None:
        """
        更新激活水平

        公式：
        N_j(s+1) = decay * N_j(s) + retrieve_gain * R_j(s)

        参数：
            N_j(s)：新皮层知识 j 的激活水平
            Rj(s)：该知识是否进入支持子集
            decay: 衰减系数（默认0.8）
            retrieve_gain: 检索增益系数（默认0.2）
        """
        for item in self.items.values():
            if item.is_static:
                item.activation_level = 1.0
                item.memory_state = "active"
                continue

            current_activation = item.activation_level
            retrieve_factor = 1.0 if item.item_id in support_item_ids else 0.0

            new_activation = decay * current_activation + retrieve_gain * retrieve_factor

            # 确保激活水平在合理范围内
            item.activation_level = max(0.0, min(1.0, new_activation))

    def update_memory_states(self) -> None:
        """
        更新记忆状态

        逻辑：
            - 按 activation_level 从高到低排序
            - 前 20% 标记为 active
            - 20% 到 60% 标记为 dormant
            - 后 40% 标记为 archived
            - is_static=True 的 item 永远 active
            - 小样本情况下至少要合理
        """
        if not self.items:
            return

        # 先处理静态项
        for item in self.items.values():
            if item.is_static:
                item.memory_state = "active"
                item.activation_level = 1.0

        # 获取所有非静态项并按激活水平排序
        non_static_items = [
            item for item in self.items.values()
            if not item.is_static
        ]

        if not non_static_items:
            return

        # 按激活水平降序排序
        sorted_items = sorted(non_static_items, key=lambda x: x.activation_level, reverse=True)
        total_count = len(sorted_items)

        # 计算各状态的边界
        active_count = max(1, int(total_count * 0.2))  # 至少1个
        dormant_count = max(1, int(total_count * 0.4))  # 至少1个
        archived_count = total_count - active_count - dormant_count

        # 分配状态
        for i, item in enumerate(sorted_items):
            if i < active_count:
                item.memory_state = "active"
            elif i < active_count + dormant_count:
                item.memory_state = "dormant"
            else:
                item.memory_state = "archived"

    def _compact_item_info(self, item_info: Dict) -> Dict:
        return {
            "item_id": item_info.get("item_id"),
            "source_id": item_info.get("source_id"),
            "memory_state": item_info.get("memory_state"),
            "sim_score": round(float(item_info.get("sim_score", 0.0)), 4),
            "content_preview": str(item_info.get("content", ""))[:120],
        }

    def retrieve_items(
        self,
        section_query: str,
        active_budget: int = 2,
        dormant_budget: int = 1,
        archived_budget: int = 1,
        archived_sim_gate: float = 0.85,
    ) -> Tuple[List[Dict], List[str]]:
        """
        检索知识项

        参数：
            section_query: 节查询文本
            active_budget: 活跃状态预算
            dormant_budget: 休眠状态预算
            archived_budget: 归档状态预算
            archived_sim_gate: 归档项相似度阈值

        返回值：
            Tuple[List[Dict], List[str]]: (检索到的项目列表, 项目ID列表)
        """
        if self.similarity_service is None:
            raise ValueError("similarity_service is not configured")

        # 更新记忆状态
        self.update_memory_states()

        # 初始化候选池
        active_candidates = []
        dormant_candidates = []
        archived_candidates = []

        # 计算相似度并分类
        for item in self.items.values():
            sim_score = self.similarity_service.compute_similarity(section_query, item.content)

            item_info = {
                "item_id": item.item_id,
                "content": item.content,
                "source_id": item.source_id,
                "sim_score": sim_score,
                "memory_state": item.memory_state,
            }

            if item.memory_state == "active":
                active_candidates.append(item_info)
            elif item.memory_state == "dormant":
                dormant_candidates.append(item_info)
            elif item.memory_state == "archived" and sim_score >= archived_sim_gate:
                archived_candidates.append(item_info)

        # 按相似度降序排序
        active_candidates.sort(key=lambda x: x["sim_score"], reverse=True)
        dormant_candidates.sort(key=lambda x: x["sim_score"], reverse=True)
        archived_candidates.sort(key=lambda x: x["sim_score"], reverse=True)

        # 按预算选取
        selected_items = (
            active_candidates[:active_budget] +
            dormant_candidates[:dormant_budget] +
            archived_candidates[:archived_budget]
        )

        # 再次按相似度降序排序
        selected_items.sort(key=lambda x: x["sim_score"], reverse=True)

        # 提取选中的项目ID
        selected_item_ids = [item["item_id"] for item in selected_items]

        self.last_retrieval_debug = {
            "active_candidates": [
                self._compact_item_info(item) for item in active_candidates
            ],
            "dormant_candidates": [
                self._compact_item_info(item) for item in dormant_candidates
            ],
            "archived_candidates": [
                self._compact_item_info(item) for item in archived_candidates
            ],
            "selected_items": [
                self._compact_item_info(item) for item in selected_items
            ],
            "selected_item_ids": selected_item_ids,
            "budgets": {
                "active_budget": active_budget,
                "dormant_budget": dormant_budget,
                "archived_budget": archived_budget,
                "archived_sim_gate": archived_sim_gate,
            },
        }

        return selected_items, selected_item_ids

    def _is_duplicate_safe(self, content: str) -> bool:
        """
        安全的重复检测：
        如果 similarity_service 未配置，则默认不判重（返回 False）
        """
        if self.similarity_service is None:
            return False
        return self.has_duplicate(content)

    def has_duplicate(self, content: str, threshold: float = 0.95) -> bool:
        """
        检查是否存在重复内容

        参数：
            content: 要检查的内容
         //   threshold: 相似度阈值（默认0.95）

        返回值：
            bool: 如果存在重复返回True，否则返回False
        """
        if self.similarity_service is None:
            raise ValueError("similarity_service is not configured")

        for item in self.items.values():
            if self.similarity_service.is_duplicate(content, item.content, threshold):
                return True
        return False

    def consolidate_from_section_graph(
        self,
        section_id: str,
        dtg_store,
        discourse_ledger,
        generated_content: Dict[str, str],
        llm_client,
    ) -> List[NeocortexItem]:
        """
        在每节验证通过后，将当前 section 的知识写入新皮层

        参数：
            section_id: 当前节ID
            dtg_store: DTG存储实例
            discourse_ledger: 话语状态账本实例
            generated_content: 已生成内容字典
            llm_client: LLM客户端实例

        返回值：
            List[NeocortexItem]: 新添加的知识项列表
        """
        added_items = []

        # Step 2: 处理 intent_node
        intent_node = dtg_store.get_intent_node(section_id)
        if intent_node and intent_node.get("confidence", 0) >= 0.7:
            content = intent_node.get("content", "")
            source_id = intent_node.get("id", "")

            if content:
                if not self._is_duplicate_safe(content):
                    item = NeocortexItem.create(
                        content=content,
                        source_id=source_id
                    )
                    self.add_item(item)
                    added_items.append(item)

        # Step 3: 处理 decision_node
        decision = dtg_store.get_decision_for_section(section_id)
        if decision and decision.confidence >= 0.7:
            content = f"{decision.decision} {decision.reasoning} {getattr(decision, 'expected_effect', '')}"
            source_id = decision.decision_id

            if content:
                if not self._is_duplicate_safe(content):
                    item = NeocortexItem.create(
                        content=content,
                        source_id=source_id
                    )
                    self.add_item(item)
                    added_items.append(item)

        # Step 4: 处理 content_node（需要 LLM 抽象）
        section_text = generated_content.get(section_id)
        if section_text:
            # 调用 LLM 抽象为一条长期知识（只生成一句话）
            prompt = (
                "You are a knowledge abstraction assistant. "
                "Extract one key long-term knowledge point from the following text. "
                "Return only one sentence without explanation.\n\n"
                f"Text:\n{section_text}\n\n"
                "Abstracted knowledge point:"
            )
            abstracted_content = llm_client.generate(prompt, temperature=0.0, max_tokens=100).strip()

            source_id = dtg_store.get_content_node_id(section_id)

            if abstracted_content:
                if not self._is_duplicate_safe(abstracted_content):
                    item = NeocortexItem.create(
                        content=abstracted_content,
                        source_id=source_id
                    )
                    self.add_item(item)
                    added_items.append(item)

        # Step 5: 处理 DSL（图结构 + 门控）
        intent_node = dtg_store.get_intent_node(section_id)
        if intent_node:
            intent_id = intent_node["id"]

            # 获取图结构中的DSL ID
            graph_dsl_ids = set(
                dtg_store.derived_from_edges.get(intent_id, [])
            )

            # 获取候选条目
            gated_entries = discourse_ledger.get_neocortex_candidate_entries()

            # 筛选属于当前图结构的条目
            selected_entries = [
                entry for entry in gated_entries
                if entry.entry_id in graph_dsl_ids
            ]

            # 处理每个选中的条目
            for entry in selected_entries:
                content = entry.content
                source_id = entry.entry_id

                if content:
                    if not self._is_duplicate_safe(content):
                        item = NeocortexItem.create(
                            content=content,
                            source_id=source_id
                        )
                        self.add_item(item)
                        added_items.append(item)

                # 标记为已整合
                entry.is_consolidated = True

        # Step 6: 返回
        return added_items
