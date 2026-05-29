# MetaWriter 实验基础设施（experiments/）

本目录承载 AAAI 投稿所需的全部对比与消融实验：主实验（EXP-I）、系统横向对比（EXP-II）、
骨干模型泛化（EXP-III）、消融实验（EXP-IV）、深度分析（EXP-V）。

> 设计原则：业务逻辑（`src/`）只做最小侵入式改造（仅"加开关"），所有实验编排、对比系统接入、
> 统计分析都隔离在本包内；全部产物落盘到 `experiments_out/`，与日常调试目录 `outputs/` 物理隔离。

## 目录结构

```
experiments/
├── config/      L0 配置层（纯数据，无副作用）
│   ├── ablation.py     消融预设注册表（PRESETS + from_preset）；核心类型在 src/core/ablation.py
│   ├── backbone.py     五骨干模型模版（环境变量解析，密钥不入库）
│   ├── run_context.py  四元组命名 {task}__{method}__{model}__{run} 与 provenance
│   └── eval_subset.py  seed=42 分层抽样（40/20 任务集、Scaling 6 档）
├── runners/     L2 执行层
│   ├── run_metawriter.py  跑 MetaWriter（Full 或消融变体）单任务
│   ├── run_baseline.py    跑对比系统（Direct-LLM 或外部系统）单任务
│   └── batch_driver.py    矩阵驱动（任务×方法×模型×重复）+ 断点续跑 + token 预算
├── baselines/   对比系统真实接入（子进程 + 环境隔离）
│   ├── direct_llm.py            S0 Direct-LLM（进程内单次生成）
│   ├── subprocess_adapter.py    外部系统通用子进程驱动
│   ├── autosurvey/lira/surveyforge/paperorchestra_adapter.py
│   └── envs/                    各系统接入配置 + baseline_env + 桥接模版（见 envs/README.md）
└── analysis/    L3 分析层
    ├── aggregate.py  summary.json → 长表/分组均值±std/配对序列 → CSV
    ├── stats.py      配对 Wilcoxon + Bonferroni(α=0.05/7) + Cohen's d（纯标准库）
    ├── scaling.py    EXP-V.1：DSL 增益随文档长度的 scaling
    └── tables.py     聚合结果 → LaTeX 表（±std 与 */** 显著性）
```

## 准备工作

### 1. 配置骨干模型密钥（baseline_env）

```bash
cp experiments/baselines/envs/baseline_env.env.template experiments/baselines/envs/baseline_env.env
# 编辑 baseline_env.env，填入五个骨干模型的 MODEL / BASE_URL / API_KEY
```

环境变量名与 `config/backbone.py` 对齐：`MINIMAX_*` / `GPT4O_*` / `CLAUDE_SONNET_*` /
`DEEPSEEK_V3_*` / `QWEN25_72B_*`。

### 2. 配置外部对比系统（仅 EXP-II 需要）

见 `baselines/envs/README.md`：为每个系统克隆 repo、建独立 venv、编写桥接脚本、
填写 `envs/<system>.json`。

## 快速运行（单次）

通过扩展后的 `main.py` 进入实验模式（产物写入 `experiments_out/runs/`）：

```bash
# MetaWriter 完整系统（EXP-I）
python main.py --task-id med_s001 --model minimax --run-id r1

# 某个消融变体（EXP-IV）
python main.py --task-id med_s001 --ablation a1_no_dsl --model minimax --run-id r1

# Direct-LLM 基线（EXP-II）
python main.py --task-id med_s001 --baseline direct-llm --model minimax --run-id r1

# 外部对比系统（EXP-II，需先配好 envs/）
python main.py --task-id med_s001 --baseline autosurvey --model minimax --run-id r1
```

合法消融预设：`full`, `a0_no_correction`, `a1_no_dsl`, `a2_no_mrsd`, `a3_no_metastate`,
`a4_no_planner`, `a6_no_dsl_relations`, `a7_no_memory_purge`（A5 No-HyDE 已随 HyDE 移除而删除）。

## 批量运行（矩阵）

```python
from experiments.config.eval_subset import build_eval_40
from experiments.runners.batch_driver import build_ablation_matrix, build_comparison_matrix, run_matrix

tasks = build_eval_40()  # seed=42 可复现 40 任务集

# EXP-IV 消融矩阵：40 任务 × 8 方法 × 3 次
ablation_specs = build_ablation_matrix(
    tasks,
    ["full", "a0_no_correction", "a1_no_dsl", "a2_no_mrsd",
     "a3_no_metastate", "a4_no_planner", "a6_no_dsl_relations", "a7_no_memory_purge"],
    model="minimax", runs=3,
)
report = run_matrix(ablation_specs, token_budget=60_000_000)  # 超预算自动停机
```

`run_matrix` 支持断点续跑（已完成单元自动跳过）、故障隔离（单元失败不中断整批）、
进度清单落盘（`experiments_out/manifests/matrix_progress.json`）。

## 分析与制表

```python
from experiments.analysis import aggregate, stats, tables

summaries = aggregate.load_run_summaries()
rows = aggregate.to_long_rows(summaries)
agg = aggregate.aggregate_mean_std(rows)            # 方法×模型×指标 的均值±std
aggregate.write_csv(agg, "experiments_out/tables/aggregate.csv")

# Full vs no_dsl 在 CVR 上的配对显著性（Bonferroni 修正 7 个消融比较）
a, b, _ = aggregate.paired_series_by_task(summaries, metric="constraint_violation_rate",
                                          method_a="full", method_b="no_dsl", model="minimax")
result = stats.paired_comparison(a, b, num_comparisons=7)
print(result.wilcoxon.p_value, result.cohens_d, result.significant)

# 生成论文主表 LaTeX
latex = tables.render_results_table(agg, methods=["full", "no_dsl"], metrics=["constraint_violation_rate"], model="minimax")
```

## 测试

```bash
python -m pytest tests/test_ablation_config.py tests/test_run_context.py \
  tests/test_ablation_switches.py tests/test_baseline_adapters.py \
  tests/test_runners.py tests/test_analysis_stats.py -q
```
