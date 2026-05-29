"""
LaTeX 表格生成：tables

功能：
    把聚合结果（均值±标准差）与显著性判定渲染为论文可直接使用的 LaTeX 表格，
    对接 aaai2027 模版。

设计动机：
    手工把数百个数字填进 LaTeX 表既易错又不可复现。本模块从聚合数据确定性地
    生成表格源码，并以 `*`/`**` 标注显著性，保证表格与数据一致。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def significance_marker(p_value: float, *, alpha: float = 0.05, alpha_strict: float = 0.01) -> str:
    """
    由 p 值生成显著性标记

    参数：
        p_value:      检验 p 值
        alpha:        弱显著阈值（默认 0.05 → 单星）
        alpha_strict: 强显著阈值（默认 0.01 → 双星）

    返回值：
        str："**"（p<alpha_strict）/ "*"（p<alpha）/ ""（不显著）
    """
    if p_value < alpha_strict:
        return "**"
    if p_value < alpha:
        return "*"
    return ""


def mean_std_cell(mean: float, std: float, *, precision: int = 3, marker: str = "") -> str:
    """
    渲染单个 "均值 ± 标准差" 单元格

    参数：
        mean:      均值
        std:       标准差
        precision: 小数位数
        marker:    追加的显著性标记（如 "*"）

    返回值：
        str：形如 "0.032 ± 0.007*" 的单元格文本
    """
    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}{marker}"


def _escape_latex(text: str) -> str:
    """转义 LaTeX 特殊字符（下划线等），避免方法名 no_dsl 破坏编译"""
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def render_results_table(
    aggregated: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    metrics: Sequence[str],
    model: Optional[str] = None,
    comparisons: Optional[Mapping[Tuple[str, str], float]] = None,
    alpha: float = 0.05,
    alpha_strict: float = 0.01,
    precision: int = 3,
    caption: str = "Main results.",
    label: str = "tab:main",
) -> str:
    """
    渲染主结果 LaTeX 表（行=方法，列=指标）

    参数：
        aggregated:  aggregate_mean_std 的输出（含 method/model/metric/mean/std/n）
        methods:     行顺序（方法标签序列）
        metrics:     列顺序（指标名序列）
        model:       仅选取该骨干模型的行（None 表示聚合数据已限定单模型）
        comparisons: 可选 {(method, metric): p_value}，用于在单元格追加显著性标记
        alpha/alpha_strict: 显著性阈值
        precision:   小数位数
        caption/label: 表标题与标签

    返回值：
        str：完整的 LaTeX table 环境源码

    关键实现细节：
        以 (method, metric) 为键建索引；缺失组合渲染为 "--"，绝不静默填 0。
    """
    index: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for record in aggregated:
        if model is not None and record.get("model") != model:
            continue
        key = (str(record.get("method")), str(record.get("metric")))
        index[key] = record

    col_spec = "l" + "c" * len(metrics)
    lines: List[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    header = "Method & " + " & ".join(_escape_latex(m) for m in metrics) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods:
        cells: List[str] = [_escape_latex(method)]
        for metric in metrics:
            record = index.get((method, metric))
            if record is None:
                cells.append("--")
                continue
            marker = ""
            if comparisons is not None:
                p_value = comparisons.get((method, metric))
                if isinstance(p_value, (int, float)):
                    marker = significance_marker(
                        float(p_value), alpha=alpha, alpha_strict=alpha_strict
                    )
            cells.append(
                mean_std_cell(
                    float(record["mean"]),
                    float(record["std"]),
                    precision=precision,
                    marker=marker,
                )
            )
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)
