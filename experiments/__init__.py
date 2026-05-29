"""
MetaWriter 实验基础设施包（experiments）

功能：
    承载对比实验（EXP-II）、消融实验（EXP-IV）、骨干模型泛化（EXP-III）、
    深度分析（EXP-V）所需的全部编排、配置与统计工具。

设计原则（第一性原理）：
    1. 业务逻辑（src/）只做最小侵入式改造（仅"加开关"），
       所有实验编排、对比系统接入、结果聚合都隔离在本包内。
    2. 配置层（config/）为纯数据结构，无副作用，是一切执行与分析的基础。
    3. 全部产物落盘到 experiments_out/，与单跑调试目录 outputs/ 物理隔离，
       避免污染日常调试。

子模块：
    config/   — L0 配置层：AblationConfig / BackboneConfig / RunContext / eval_subset
    runners/  — L2 执行层：run_metawriter / run_baseline / batch_driver
    baselines/— 对比系统真实接入适配器（子进程 + 环境隔离）
    analysis/ — L3 分析层：aggregate / stats / scaling / tables
"""
from __future__ import annotations

__all__: list[str] = []
