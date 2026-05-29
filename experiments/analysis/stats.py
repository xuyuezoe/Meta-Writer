"""
统计检验：stats

功能：
    提供配对 Wilcoxon 符号秩检验、Bonferroni 多重比较修正、Cohen's d 效果量，
    用于"MetaWriter Full vs. 对比系统/消融变体"的显著性判定。

设计动机（第一性原理）：
    - 样本量约 40、指标非正态，故选非参数的配对 Wilcoxon 符号秩检验（比 t 检验保守）。
    - 多个变体同时与 Full 比较会抬高假阳性率，故用 Bonferroni 修正显著性阈值。
    - p 值只说"有无差异"，不说"差异多大"，故并报 Cohen's d 效果量。

实现选择：
    全部以纯标准库实现，不依赖 scipy/numpy，保证分析可在任意环境复现。
    Wilcoxon 采用正态近似（含并列秩校正与连续性校正），适用于样本量 n≥~10 的场景，
    与方案的 40/20 任务规模相符。该近似为统计学标准做法，已在 docstring 显式标注。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


class StatsError(ValueError):
    """统计输入非法异常（如配对长度不一致）"""


def bonferroni_alpha(num_comparisons: int, alpha: float = 0.05) -> float:
    """
    计算 Bonferroni 修正后的显著性阈值

    参数：
        num_comparisons: 同时进行的比较次数（如消融 7 个变体 vs Full → 7）
        alpha:           家族错误率（默认 0.05）

    返回值：
        float：修正后的单次比较显著性阈值 alpha / num_comparisons

    异常：
        StatsError：num_comparisons < 1 时抛出。
    """
    if num_comparisons < 1:
        raise StatsError(f"[统计输入非法] 比较次数须 ≥ 1，实际 {num_comparisons}")
    return alpha / num_comparisons


def _standard_normal_cdf(z: float) -> float:
    """标准正态分布累积分布函数 Φ(z)，由误差函数 erf 实现"""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _average_ranks(values: Sequence[float]) -> List[float]:
    """
    计算平均秩（并列值取平均秩）

    参数：
        values: 待排秩的数值序列

    返回值：
        List[float]：与输入等长的秩序列（1-based，平均秩处理并列）
    """
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        # 找出并列区间 [i, j)
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        # 该区间的平均秩（1-based）
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


@dataclass(frozen=True)
class WilcoxonResult:
    """
    Wilcoxon 符号秩检验结果

    参数：
        statistic:    检验统计量 W（W+ 与 W- 中的较小者）
        z:            正态近似的标准化统计量
        p_value:      双侧 p 值
        n_effective:  有效样本量（剔除差值为 0 的配对后）
        method:       使用的方法标注（normal_approximation）
    """

    statistic: float
    z: float
    p_value: float
    n_effective: int
    method: str = "normal_approximation"


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> WilcoxonResult:
    """
    配对 Wilcoxon 符号秩检验（正态近似）

    功能：
        检验配对样本 x、y 的差值中位数是否显著偏离 0。

    参数：
        x: 配对的第一组观测（如 Full 的各任务指标）
        y: 配对的第二组观测（如某变体的各任务指标），须与 x 等长且一一对应

    返回值：
        WilcoxonResult：含统计量、z、双侧 p、有效样本量

    异常：
        StatsError：x、y 长度不一致或为空时抛出。

    关键实现细节：
        1. 计算差值 d=x-y，剔除 d=0 的配对（零差不携带方向信息）。
        2. 对 |d| 求平均秩（并列取平均秩）。
        3. W+ = 正差秩和，W- = 负差秩和，统计量 W=min(W+,W-)。
        4. 正态近似：均值 μ=n(n+1)/4，方差含并列校正
           σ²=[n(n+1)(2n+1) - Σ(t³-t)/2]/24；连续性校正后求 z 与双侧 p。
        全部配对零差时返回 p=1.0（无可检验信息）。
    """
    if len(x) != len(y):
        raise StatsError(f"[统计输入非法] 配对长度不一致: {len(x)} vs {len(y)}")
    if len(x) == 0:
        raise StatsError("[统计输入非法] 配对样本为空")

    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    nonzero = [d for d in diffs if d != 0.0]
    n = len(nonzero)
    if n == 0:
        return WilcoxonResult(statistic=0.0, z=0.0, p_value=1.0, n_effective=0)

    abs_diffs = [abs(d) for d in nonzero]
    ranks = _average_ranks(abs_diffs)

    w_plus = sum(r for r, d in zip(ranks, nonzero) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, nonzero) if d < 0)
    statistic = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0

    # 并列校正项：对每组并列大小 t，累加 (t³ - t)
    tie_correction = 0.0
    sorted_abs = sorted(abs_diffs)
    i = 0
    while i < len(sorted_abs):
        j = i
        while j + 1 < len(sorted_abs) and sorted_abs[j + 1] == sorted_abs[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_correction += t ** 3 - t
        i = j + 1

    var_w = (n * (n + 1) * (2 * n + 1) - tie_correction / 2.0) / 24.0
    if var_w <= 0:
        return WilcoxonResult(statistic=statistic, z=0.0, p_value=1.0, n_effective=n)

    # 连续性校正
    z = (statistic - mean_w + 0.5) / math.sqrt(var_w)
    p_value = 2.0 * _standard_normal_cdf(z)
    p_value = max(0.0, min(1.0, p_value))
    return WilcoxonResult(statistic=statistic, z=z, p_value=p_value, n_effective=n)


def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    """
    配对样本的 Cohen's d 效果量

    功能：
        以配对差值的均值除以差值标准差，量化差异大小（与样本量无关）。

    参数：
        x: 配对第一组
        y: 配对第二组（与 x 等长）

    返回值：
        float：Cohen's d（差值标准差为 0 时返回 0.0）

    异常：
        StatsError：长度不一致或样本数 < 2 时抛出。

    关键实现细节：
        采用差值的样本标准差（ddof=1）。|d|>0.5 视为中等效果，>0.8 为大效果。
    """
    if len(x) != len(y):
        raise StatsError(f"[统计输入非法] 配对长度不一致: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise StatsError("[统计输入非法] Cohen's d 需要至少 2 个配对样本")

    diffs = [float(a) - float(b) for a, b in zip(x, y)]
    mean_diff = sum(diffs) / len(diffs)
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
    std_diff = math.sqrt(variance)
    if std_diff == 0.0:
        return 0.0
    return mean_diff / std_diff


@dataclass(frozen=True)
class PairedComparison:
    """
    一次配对比较的完整结论

    参数：
        n:             配对样本数
        mean_a:        第一组均值
        mean_b:        第二组均值
        mean_diff:     均值差（a - b）
        wilcoxon:      Wilcoxon 检验结果
        cohens_d:      Cohen's d 效果量
        alpha_corrected: 修正后显著性阈值
        significant:   是否显著（p < alpha_corrected）
    """

    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    wilcoxon: WilcoxonResult
    cohens_d: float
    alpha_corrected: float
    significant: bool


def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    num_comparisons: int,
    alpha: float = 0.05,
) -> PairedComparison:
    """
    执行一次完整的配对比较（Wilcoxon + Bonferroni + Cohen's d）

    参数：
        values_a:        第一组配对观测（如 Full）
        values_b:        第二组配对观测（如某变体），与 values_a 一一对应
        num_comparisons: Bonferroni 修正的比较总数
        alpha:           家族错误率（默认 0.05）

    返回值：
        PairedComparison：含均值、检验、效果量与显著性判定
    """
    wilcoxon = wilcoxon_signed_rank(values_a, values_b)
    d = cohens_d_paired(values_a, values_b)
    alpha_corrected = bonferroni_alpha(num_comparisons, alpha)
    mean_a = sum(map(float, values_a)) / len(values_a)
    mean_b = sum(map(float, values_b)) / len(values_b)
    return PairedComparison(
        n=len(values_a),
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_a - mean_b,
        wilcoxon=wilcoxon,
        cohens_d=d,
        alpha_corrected=alpha_corrected,
        significant=wilcoxon.p_value < alpha_corrected,
    )


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    """
    计算均值与样本标准差（ddof=1）

    参数：
        values: 数值序列

    返回值：
        Tuple[float, float]：(均值, 样本标准差)；样本数 < 2 时标准差为 0.0

    异常：
        StatsError：序列为空时抛出。
    """
    if not values:
        raise StatsError("[统计输入非法] 空序列无法计算均值")
    floats = [float(v) for v in values]
    mean = sum(floats) / len(floats)
    if len(floats) < 2:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in floats) / (len(floats) - 1)
    return mean, math.sqrt(variance)
