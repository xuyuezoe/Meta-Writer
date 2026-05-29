"""
评估子集构建：eval_subset

功能：
    以固定随机种子（seed=42）从 MetaBench 任务池中分层抽样，构建可复现的
    评估子集（40 任务主集、20 任务子集、Scaling 6 档位）。

设计动机（第一性原理）：
    实验结论的可复现性要求"任何人重跑都得到同一批任务 ID"。这意味着：
        1. 抽样必须确定性：固定种子 + 确定性排序，杜绝集合迭代顺序带来的随机性。
        2. 分层必须显式：按文档长度分层（Short/Medium/Long），保证子集覆盖
           不同长度区间，否则主实验会偏向某一长度档。
        3. 容量不足必须报错：当某一层的可用任务少于请求数时，显式抛错而非
           静默欠采样——欠采样会悄悄削弱统计效力（禁止兜底）。

与 ablation_study_plan.md §2.2 / §6.1 的对应：
    EVAL_40_COUNTS：Short 15 / Medium 10 / Long 15（主评估集）
    EVAL_20：从 40 集再确定性抽半数（EXP-III 用）
    SCALING_BINS：6 个词数档位（EXP-V.1 Scaling 分析）
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# 固定随机种子：贯穿所有抽样操作，保证可复现
DEFAULT_SEED: int = 42

# 长度分层边界（词数），对应方案 §2.2
SHORT_MAX_WORDS: int = 5000
MEDIUM_MAX_WORDS: int = 8000

# 三层的名称常量
STRATUM_SHORT = "short"
STRATUM_MEDIUM = "medium"
STRATUM_LONG = "long"

# 40 任务主评估集的分层配额
EVAL_40_COUNTS: Dict[str, int] = {
    STRATUM_SHORT: 15,
    STRATUM_MEDIUM: 10,
    STRATUM_LONG: 15,
}

# Scaling 分析的 6 个词数档位（目标词数中心值）
SCALING_BINS: Tuple[int, ...] = (3000, 4500, 6000, 8000, 10000, 12800)


class EvalSubsetError(ValueError):
    """评估子集构建非法异常（如任务池容量不足）"""


@dataclass(frozen=True)
class TaskPoolEntry:
    """
    任务池条目

    功能：
        承载抽样所需的最小信息：任务 ID 与目标词数。

    参数：
        task_id:      任务 ID（如 "med_s001"）
        target_words: 该任务的目标总词数（用于长度分层）
    """

    task_id: str
    target_words: int

    @property
    def stratum(self) -> str:
        """根据目标词数返回所属长度分层名称"""
        return classify_length(self.target_words)


def classify_length(target_words: int) -> str:
    """
    按目标词数判定长度分层

    参数：
        target_words: 目标总词数

    返回值：
        str：STRATUM_SHORT / STRATUM_MEDIUM / STRATUM_LONG 之一

    关键实现细节：
        边界采用方案 §2.2 定义：≤5000 为 Short，5001–8000 为 Medium，>8000 为 Long。
    """
    if target_words <= SHORT_MAX_WORDS:
        return STRATUM_SHORT
    if target_words <= MEDIUM_MAX_WORDS:
        return STRATUM_MEDIUM
    return STRATUM_LONG


def _group_by_stratum(pool: Sequence[TaskPoolEntry]) -> Dict[str, List[TaskPoolEntry]]:
    """
    按长度分层分组，组内按 task_id 确定性排序

    参数：
        pool: 任务池条目序列

    返回值：
        Dict[str, List[TaskPoolEntry]]：分层名到该层任务列表的映射，
            每层内部按 task_id 字典序排序（保证抽样前顺序确定）。
    """
    grouped: Dict[str, List[TaskPoolEntry]] = {
        STRATUM_SHORT: [],
        STRATUM_MEDIUM: [],
        STRATUM_LONG: [],
    }
    for entry in pool:
        grouped[entry.stratum].append(entry)
    for stratum in grouped:
        grouped[stratum].sort(key=lambda e: e.task_id)
    return grouped


def stratified_sample(
    pool: Sequence[TaskPoolEntry],
    counts: Dict[str, int],
    *,
    seed: int = DEFAULT_SEED,
) -> List[str]:
    """
    分层随机抽样（确定性）

    功能：
        在每个长度分层内独立做无放回随机抽样，取出指定数量的任务 ID。

    参数：
        pool:   任务池条目序列
        counts: 各分层的抽样配额（如 EVAL_40_COUNTS）
        seed:   随机种子（默认 42）

    返回值：
        List[str]：抽中的任务 ID 列表，按 (分层顺序, task_id) 确定性排序，
            保证同一输入恒得同一输出。

    异常：
        EvalSubsetError：当某分层可用任务数少于请求配额时抛出，
            明确指出缺口，避免静默欠采样。

    关键实现细节：
        每个分层使用独立的 random.Random(seed) 实例，避免跨层抽样相互干扰，
        且不污染全局随机状态。
    """
    grouped = _group_by_stratum(pool)
    selected: List[str] = []

    for stratum in (STRATUM_SHORT, STRATUM_MEDIUM, STRATUM_LONG):
        requested = counts.get(stratum, 0)
        if requested <= 0:
            continue
        available = grouped[stratum]
        if len(available) < requested:
            raise EvalSubsetError(
                f"[评估子集容量不足] 分层 '{stratum}' 仅有 {len(available)} 个可用任务，"
                f"但请求抽取 {requested} 个。请扩充 MetaBench 任务池，"
                f"或调整该分层的抽样配额。"
            )
        rng = random.Random(seed)
        chosen = rng.sample(available, requested)
        selected.extend(entry.task_id for entry in chosen)

    selected.sort()
    return selected


def deterministic_halve(task_ids: Sequence[str], *, seed: int = DEFAULT_SEED) -> List[str]:
    """
    确定性取半（用于从 40 任务集构建 20 任务子集）

    功能：
        对给定任务 ID 列表做确定性随机抽样，取一半（向下取整）。

    参数：
        task_ids: 源任务 ID 列表
        seed:     随机种子（默认 42）

    返回值：
        List[str]：抽中的任务 ID 列表（已排序）

    异常：
        EvalSubsetError：当输入为空时抛出。
    """
    unique_sorted = sorted(set(task_ids))
    if not unique_sorted:
        raise EvalSubsetError("[评估子集非法] 输入任务列表为空，无法取半")
    half = max(1, len(unique_sorted) // 2)
    rng = random.Random(seed)
    chosen = rng.sample(unique_sorted, half)
    return sorted(chosen)


def assign_scaling_bins(
    pool: Sequence[TaskPoolEntry],
    *,
    bins: Sequence[int] = SCALING_BINS,
    per_bin: int = 5,
    seed: int = DEFAULT_SEED,
) -> Dict[int, List[str]]:
    """
    为 Scaling 分析分配各词数档位的任务

    功能：
        将任务按目标词数就近归入最接近的档位中心，再在每档内确定性抽样 per_bin 个。

    参数：
        pool:    任务池条目序列
        bins:    档位中心词数（默认 SCALING_BINS）
        per_bin: 每档抽样数（默认 5）
        seed:    随机种子（默认 42）

    返回值：
        Dict[int, List[str]]：档位中心词数到任务 ID 列表的映射。

    异常：
        EvalSubsetError：当某档可用任务不足 per_bin 时抛出。

    关键实现细节：
        就近归档使用绝对词数差最小原则；每档内按 task_id 排序后用固定种子抽样。
    """
    bucket: Dict[int, List[TaskPoolEntry]] = {center: [] for center in bins}
    for entry in pool:
        nearest = min(bins, key=lambda center: abs(entry.target_words - center))
        bucket[nearest].append(entry)

    result: Dict[int, List[str]] = {}
    for center in bins:
        available = sorted(bucket[center], key=lambda e: e.task_id)
        if len(available) < per_bin:
            raise EvalSubsetError(
                f"[Scaling 档位容量不足] 档位 {center} 词仅有 {len(available)} 个任务，"
                f"但每档需要 {per_bin} 个。请扩充该词数区间的任务。"
            )
        rng = random.Random(seed)
        chosen = rng.sample(available, per_bin)
        result[center] = sorted(entry.task_id for entry in chosen)
    return result


def _extract_target_words(reference: Dict[str, Any]) -> Optional[int]:
    """
    从 reference 提取目标总词数

    功能：
        词数字段既可能在 reference 顶层，也可能嵌套在 reference["constraints"] 下
        （meta_bench.task_generator 当前写入后者）。本函数兼顾两处。

    参数：
        reference: MetaBench reference 字典

    返回值：
        Optional[int]：目标总词数；均未找到时返回 None。

    关键实现细节：
        优先级 total_target_words > body_target_words > required_length_words；
        顶层与 constraints 子字典都查找。
    """
    candidates: List[Dict[str, Any]] = [reference]
    constraints = reference.get("constraints")
    if isinstance(constraints, dict):
        candidates.append(constraints)

    for source in candidates:
        for key in ("total_target_words", "body_target_words", "required_length_words"):
            raw_value = source.get(key)
            if isinstance(raw_value, (int, float)) and int(raw_value) > 0:
                return int(raw_value)
    return None


def build_pool_from_registry() -> List[TaskPoolEntry]:
    """
    从 MetaBench 任务注册表构建任务池

    功能：
        遍历已注册的 MetaBench 任务，从其 reference 中提取目标总词数，
        组装为可供抽样的任务池。

    返回值：
        List[TaskPoolEntry]：任务池条目列表（按 task_id 排序）

    异常：
        EvalSubsetError：当某任务 reference 缺失可用的词数字段时抛出，
            避免用错误的词数误判分层。

    关键实现细节：
        词数优先级：total_target_words > body_target_words > required_length_words。
        这三个字段由 meta_bench.task_generator 在构建 reference 时写入。
    """
    # 延迟导入，避免配置层在无运行环境时被强依赖
    from examples.tasks import TASK_ID_REGISTRY, TASK_REGISTRY

    pool: List[TaskPoolEntry] = []
    for task_id, task_name in TASK_ID_REGISTRY.items():
        config = TASK_REGISTRY[task_name]()
        reference = config.get("reference")
        if not isinstance(reference, dict):
            raise EvalSubsetError(
                f"[任务池构建失败] 任务 '{task_id}' 缺少 reference 字段，无法判定长度"
            )
        target_words = _extract_target_words(reference)
        if target_words is None:
            raise EvalSubsetError(
                f"[任务池构建失败] 任务 '{task_id}' 的 reference 缺少有效词数字段"
                f"（total_target_words / body_target_words / required_length_words）"
            )
        pool.append(TaskPoolEntry(task_id=task_id, target_words=target_words))

    pool.sort(key=lambda e: e.task_id)
    return pool


def build_eval_40(
    pool: Optional[Sequence[TaskPoolEntry]] = None,
    *,
    seed: int = DEFAULT_SEED,
) -> List[str]:
    """
    构建 40 任务主评估集

    参数：
        pool: 任务池；为 None 时从注册表构建。
        seed: 随机种子（默认 42）

    返回值：
        List[str]：40 个任务 ID（Short 15 / Medium 10 / Long 15）
    """
    source = list(pool) if pool is not None else build_pool_from_registry()
    return stratified_sample(source, EVAL_40_COUNTS, seed=seed)


def build_eval_20(
    pool: Optional[Sequence[TaskPoolEntry]] = None,
    *,
    seed: int = DEFAULT_SEED,
) -> List[str]:
    """
    构建 20 任务子集（从 40 集确定性取半，EXP-III 用）

    参数：
        pool: 任务池；为 None 时从注册表构建。
        seed: 随机种子（默认 42）

    返回值：
        List[str]：20 个任务 ID
    """
    eval_40 = build_eval_40(pool, seed=seed)
    return deterministic_halve(eval_40, seed=seed)
