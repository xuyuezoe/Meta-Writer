"""
Human-readable memory report writer.

This module writes a Markdown report for inspecting support subsets,
neocortex retrieval, and neocortex state across sections. It does not
replace run.log or memory_trace.jsonl.
"""

from pathlib import Path
from typing import Any, Iterable


class MemoryReporter:
    def __init__(self, output_dir: str, session_name: str):
        self.report_path = Path(output_dir) / f"{session_name}_memory_report.md"
        if self.report_path.exists():
            self.report_path.unlink()
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self._write("# Memory Report\n")

    def write_section_support_subset(self, section_id: str, support_subset: dict) -> None:
        """Write a readable support-subset report for one section."""
        dtg_nodes = list(support_subset.get("dtg_nodes", []) or [])
        dsl_entries = list(support_subset.get("dsl_entries", []) or [])
        neocortex_items = list(support_subset.get("neocortex_items", []) or [])

        self._write(f"\n## Section {section_id}\n")

        self._write("### Support Subset Summary\n")
        self._write(f"- node_count: {len(support_subset.get('node_ids', []) or dtg_nodes)}")
        self._write(f"- dsl_count: {len(dsl_entries)}")
        self._write(f"- neocortex_count: {len(neocortex_items)}\n")

        self._write("### Query\n")
        self._write(self._preview(support_subset.get("query_text", ""), limit=1000))
        self._write("")

        self._write("### DTG Nodes\n")
        if not dtg_nodes:
            self._write("- None")
        for node in dtg_nodes:
            node_info = self._normalize_dtg_node(node)
            self._write(f"- node_id: {node_info.get('node_id')}")
            self._write(f"  - node_type: {node_info.get('node_type')}")
            if node_info.get("text_preview"):
                self._write(f"  - text_preview: {node_info.get('text_preview')}")
        self._write("")

        self._write("### Selected Edges\n")
        selected_edges = list(support_subset.get("selected_edges", []) or [])
        if not selected_edges:
            self._write("- None")
        for edge in selected_edges:
            edge_info = self._normalize_edge(edge)
            self._write(
                f"- {edge_info.get('source')} --{edge_info.get('type')}"
                f"(weight={edge_info.get('weight')})--> {edge_info.get('target')}"
            )
        self._write("")

        self._write("### DSL Entries\n")
        if not dsl_entries:
            self._write("- None")
        for item in dsl_entries:
            self._write(f"- dsl_id: {item.get('dsl_id')}")
            score_parts = []
            for key in ("score", "sim_score", "support_count"):
                if key in item:
                    score_parts.append(f"{key}={item.get(key)}")
            if score_parts:
                self._write(f"  - scores: {', '.join(score_parts)}")
            type_parts = []
            for key in ("commitment_type", "constraint_type"):
                if key in item:
                    type_parts.append(f"{key}={item.get(key)}")
            if type_parts:
                self._write(f"  - types: {', '.join(type_parts)}")
            trust_parts = []
            for key in ("trust_level", "stability_score"):
                if key in item:
                    trust_parts.append(f"{key}={item.get(key)}")
            if trust_parts:
                self._write(f"  - trust: {', '.join(trust_parts)}")
            text = item.get("text", item.get("text_preview", ""))
            if text:
                self._write(f"  - text: {self._preview(text)}")
        self._write("")

        self._write("### Neocortex Items Retrieved\n")
        if not neocortex_items:
            self._write("- None")
        for item in neocortex_items:
            self._write(f"- item_id: {item.get('item_id')}")
            self._write(f"  - source_id: {item.get('source_id')}")
            self._write(f"  - source_kind: {item.get('source_kind')}")
            self._write(f"  - memory_state: {item.get('memory_state')}")
            self._write(f"  - activation_level: {item.get('activation_level')}")
            if "sim_score" in item:
                self._write(f"  - sim_score: {item.get('sim_score')}")
            content = item.get("content", item.get("content_preview", ""))
            if content:
                self._write(f"  - content: {self._preview(content)}")
        self._write("")

        # 归档池全量候选（不管是否过门限），按 sim_score 降序排列
        archived_all = (
            support_subset.get("debug_trace", {})
            .get("neocortex_debug", {})
            .get("archived_all_candidates", [])
        )
        self._write("### Archived Candidates (all, sorted by sim_score desc)\n")
        if not archived_all:
            self._write("- None")
        for item in archived_all:
            self._write(f"- item_id: {item.get('item_id')}")
            self._write(f"  - source_id: {item.get('source_id')}")
            self._write(f"  - activation_level: {item.get('activation_level')}")
            self._write(f"  - sim_score: {item.get('sim_score')}")
        self._write("")

    def write_neocortex_after_section(
        self,
        section_id: str,
        neocortex_store,
        dtg_store=None,
        discourse_ledger=None,
    ) -> None:
        """Write the complete neocortex state after a section completes."""
        self._write(f"\n### Neocortex State After Section {section_id}\n")

        items = list(self._iter_neocortex_items(neocortex_store))
        if not items:
            self._write("- None\n")
            return

        for item in items:
            source_id = getattr(item, "source_id", None)
            self._write(f"- item_id: {getattr(item, 'item_id', None)}")
            self._write(f"  - source_id: {source_id}")
            self._write(
                f"  - source_kind: "
                f"{self._infer_source_kind(source_id, dtg_store, discourse_ledger)}"
            )
            self._write(
                f"  - source_label: "
                f"{self._source_label(source_id, dtg_store, discourse_ledger)}"
            )
            self._write(f"  - memory_state: {getattr(item, 'memory_state', None)}")
            self._write(f"  - activation_level: {getattr(item, 'activation_level', None)}")
            self._write(f"  - is_static: {getattr(item, 'is_static', None)}")
            self._write(f"  - created_section: {getattr(item, 'created_section', None)}")
            self._write(f"  - content: {self._preview(getattr(item, 'content', ''))}")
        self._write("")

    def _write(self, text) -> None:
        with open(self.report_path, "a", encoding="utf-8") as file:
            file.write(str(text) + "\n")

    def _preview(self, text, limit: int = 500) -> str:
        return str(text).replace("\n", " ")[:limit]

    def _infer_source_kind(self, source_id: str, dtg_store=None, discourse_ledger=None) -> str:
        """
        Infer the source type for a neocortex item from its source_id.
        """
        if not source_id:
            return "unknown"

        source_id = str(source_id)
        if source_id.startswith("intent:"):
            return "intent"
        if source_id.startswith("content:"):
            return "content"

        decision_by_id = getattr(dtg_store, "decision_by_id", None)
        if decision_by_id is not None:
            try:
                if source_id in decision_by_id:
                    return "decision"
            except Exception:
                pass

        get_entry = getattr(discourse_ledger, "get_entry", None)
        if callable(get_entry):
            try:
                if get_entry(source_id) is not None:
                    return "dsl"
            except Exception:
                pass

        return "unknown"

    def _source_label(self, source_id: str, dtg_store=None, discourse_ledger=None) -> str:
        """
        Return a readable source label for a neocortex item.
        """
        kind = self._infer_source_kind(source_id, dtg_store, discourse_ledger)
        return f"{kind}:{source_id}"

    def _iter_neocortex_items(self, neocortex_store) -> Iterable[Any]:
        items = getattr(neocortex_store, "items", [])
        if isinstance(items, dict):
            return items.values()
        if isinstance(items, list):
            return items
        return []

    def _normalize_dtg_node(self, node: Any) -> dict:
        if isinstance(node, dict):
            return {
                "node_id": node.get("node_id", node.get("id")),
                "node_type": node.get("node_type", node.get("type")),
                "text_preview": node.get("text_preview", ""),
            }
        if isinstance(node, (tuple, list)) and len(node) >= 2:
            return {
                "node_id": node[0],
                "node_type": node[1],
                "text_preview": "",
            }
        return {
            "node_id": str(node),
            "node_type": "",
            "text_preview": "",
        }

    def _normalize_edge(self, edge: Any) -> dict:
        if isinstance(edge, dict):
            return {
                "source": edge.get("source"),
                "target": edge.get("target"),
                "type": edge.get("type"),
                "weight": edge.get("weight"),
            }
        if isinstance(edge, (tuple, list)) and len(edge) >= 3:
            return {
                "source": edge[0],
                "target": edge[1],
                "type": edge[2],
                "weight": None,
            }
        return {
            "source": None,
            "target": None,
            "type": None,
            "weight": None,
        }
