"""
Reference integration data structures.

ReferenceItem      — 语料库中一个检索到的 chunk
ReferenceBundle    — 某节 per-section re-ranking 时使用的候选集（内部）
GlobalPaperIndex   — 全局论文索引：R1…RN → paper_id + 元数据，贯穿整个文档生命周期
SectionReferenceReport — ReferenceValidator 对某节的验证报告
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..core.validation import Issue


@dataclass
class ReferenceItem:
    """语料库中一个检索到的 chunk（用于内部排序和展示）。"""
    paper_id: str
    title: str
    chunk_id: str
    text: str
    rank: int
    retrieval_score: float


@dataclass
class GlobalPaperEntry:
    """
    全局索引中一篇论文的完整信息。

    r_index:       全局编号（1-based，对应正文中的 [R1]、[R2] 等标记）
    paper_id:      语料库内部 UUID
    title:         论文标题
    abstract:      论文摘要全文
    top_chunk_text: BM25 检索得分最高的 chunk 完整文本（不截断）
    authors:       作者列表
    doi:           DOI 字符串（可能为空）
    retrieval_score: 全局检索得分（用于排序）
    """
    r_index: int
    paper_id: str
    title: str
    abstract: str
    top_chunk_text: str
    authors: List[str]
    doi: str
    retrieval_score: float


@dataclass
class GlobalPaperIndex:
    """
    文档级全局论文索引，在生成主循环开始前由 Orchestrator 构建，
    贯穿所有章节的整个生命周期保持不变。

    设计原则：
    - LLM 只看到简单整数索引 R1…RN，永不接触 UUID
    - valid_r_set 是代码做后处理验证的权威来源
    - entries 按 r_index 升序排列
    """
    entries: List[GlobalPaperEntry] = field(default_factory=list)

    # --- 派生属性（懒加载） ---
    _by_r: Dict[int, GlobalPaperEntry] = field(default_factory=dict, repr=False)
    _by_pid: Dict[str, GlobalPaperEntry] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._by_r = {e.r_index: e for e in self.entries}
        self._by_pid = {e.paper_id: e for e in self.entries}

    # --- 只读访问 ---

    def is_empty(self) -> bool:
        return len(self.entries) == 0

    @property
    def valid_r_set(self) -> Set[int]:
        """所有合法的 r_index 集合，用于后处理验证。"""
        return set(self._by_r.keys())

    @property
    def n(self) -> int:
        return len(self.entries)

    def get_by_r(self, r_index: int) -> Optional[GlobalPaperEntry]:
        return self._by_r.get(r_index)

    def get_by_paper_id(self, paper_id: str) -> Optional[GlobalPaperEntry]:
        return self._by_pid.get(paper_id)

    def r_label(self, r_index: int) -> str:
        """返回人类可读的引用标记，如 '[R3]'。"""
        return f"[R{r_index}]"


@dataclass
class ReferenceBundle:
    """
    per-section 候选 chunk 集合，仅用于 Retriever 内部 re-ranking。
    不再传递给 Generator 或 Validator；只用于决定向 LLM 展示哪些论文。
    """
    section_id: str
    query: str
    items: List[ReferenceItem] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.items) == 0

    def paper_ids(self) -> List[str]:
        return list(dict.fromkeys(item.paper_id for item in self.items))

    def get_paper_chunks(self, paper_id: str) -> List[ReferenceItem]:
        return [item for item in self.items if item.paper_id == paper_id]


@dataclass
class SectionReferenceReport:
    """
    ReferenceValidator 对某节的验证报告。

    新架构下验证全部由代码完成（无 LLM），只检查 [Rx] 标记的合法性。
    invalid_r_indices: 正文中出现但不在全局索引中的越界 r_index 集合。
    """
    passed: bool
    issues: List["Issue"] = field(default_factory=list)
    invalid_r_indices: Set[int] = field(default_factory=set)
    valid_marker_count: int = 0
    invalid_marker_count: int = 0

    def metrics_dict(self) -> dict:
        return {
            "valid_marker_count": self.valid_marker_count,
            "invalid_marker_count": self.invalid_marker_count,
            "invalid_r_indices": sorted(self.invalid_r_indices),
        }
