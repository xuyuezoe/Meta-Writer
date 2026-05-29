"""
实验分析层（L3 analysis）

功能：
    把分散的运行摘要（summary.json）聚合为可发表的表格与统计结论。

子模块：
    aggregate — 扫描运行产物 → 长表/宽表/配对序列 → CSV
    stats     — 配对 Wilcoxon 符号秩检验 + Bonferroni 修正 + Cohen's d
    scaling   — EXP-V.1：DSL 增益随文档长度的 scaling 数据
    tables    — 聚合结果 → LaTeX 表（±std 与显著性标记）

设计原则：
    本层不依赖 scipy/numpy，全部统计以纯标准库实现（确定性、可复现、零额外依赖），
    数值方法（如 Wilcoxon 正态近似）在对应函数中显式说明。
"""
from __future__ import annotations

__all__: list[str] = []
