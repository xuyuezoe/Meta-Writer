# Meta-Writer

Meta-Writer 是一个面向长文本生成的自我修正写作系统。它不把长文生成看作一次性输出，而是拆成“规划、生成、验证、诊断、修复”的连续过程，在生成中间就发现问题并做局部修正，从而提升长文本的连贯性、约束遵守能力与可追溯性。

项目当前已经具备可运行的工程形态，既可以执行示例写作任务，也可以接入仓库内置的 benchmark 样本进行批量运行和评估。

## 核心特性

- 自我修正生成：每个 section 生成后立即验证，失败时自动重试或局部修复
- 决策追溯：用 DTG（Decision Trace Graph）记录生成决策及依赖关系
- 动态约束记忆：用 DSL（Discourse Ledger）维护承诺、开放问题和跨段约束
- 多任务支持：支持创意写作、议论文、综述论文和 benchmark 样本
- 结果可落盘：自动保存文本、运行日志、修正日志、DTG 和运行摘要

## 项目结构

```text
Meta-Writer/
├── main.py
├── examples/
│   ├── benchmark_template.py
│   └── tasks/
├── metabench/
│   ├── examples/
│   ├── config/
│   └── src/metabench/
├── src/
│   ├── agents/
│   ├── algorithms/
│   ├── core/
│   ├── evaluation/
│   ├── logging/
│   ├── memory/
│   ├── metrics/
│   ├── utils/
│   ├── validators/
│   └── orchestrator_v2.py
├── outputs/
├── sessions/
└── tests/
```

## 运行环境

本项目默认在 `conda` 环境 `metawriter` 中运行。

启动前请先激活环境：

```bash
conda activate metawriter
```

然后安装依赖：

```bash
pip install -r requirements.txt
```

## 配置说明

项目使用 `.env` 管理模型调用配置。可先复制模板文件：

```bash
cp .env.example .env
```

最小配置如下：

```bash
API_KEY=your_api_key_here
BASE_URL=https://api.example.com/v1
MODEL=your-model-name
```

说明：

- `API_KEY` 必填
- `BASE_URL` 为模型服务地址
- `MODEL` 为所使用的模型名称

## 快速开始

### 查看可用任务

```bash
python3 main.py --list-tasks
```

### 运行普通任务

```bash
python3 main.py --task survey_paper
python3 main.py --task argumentative_essay
python3 main.py --task scifi_story
```

### 运行单个 benchmark 样本

```bash
python3 main.py --task-id med_s010
```

### 批量运行全部 benchmark 样本

```bash
python3 main.py --all
```

### 打印完整生成结果

```bash
python3 main.py --task survey_paper --print-response
```

默认情况下，直接执行：

```bash
python3 main.py
```

会运行一个内置 benchmark 样本，便于快速验证主流程是否可用。

## 输出结果

运行完成后，项目会在 `outputs/` 和 `sessions/` 下保存相关产物。常见文件包括：

- `outputs/{session_name}_text.txt`：最终生成文本
- `outputs/{session_name}_run.log`：完整运行日志
- `outputs/{session_name}_correction_log.json`：修正记录
- `outputs/{session_name}_dtg.json`：决策追溯图
- `outputs/{session_name}_summary.json`：单次运行摘要
- `outputs/{session_name}_benchmark_eval.json`：benchmark 任务评估结果
- `sessions/{session_name}.json`：会话持久化结果

## Benchmark 说明

项目内置了最小 benchmark 子模块，样本位于 `metabench/examples/samples.jsonl`。

当使用 `--task-id` 或 `--all` 时，系统会：

1. 读取本地 benchmark 样本
2. 将样本转换为 Meta-Writer 可执行任务
3. 走真实生成主流程
4. 对生成结果执行本地评估

因此，benchmark 运行结果基于真实生成输出，而不是预置答案回放。

## 适用场景

Meta-Writer 适合用于：

- 长篇综述或分析类文本生成
- 对结构、约束和一致性要求较高的写作任务
- 需要研究“生成中自我修正”机制的实验型项目
- 需要将生成过程记录为可分析工件的场景

## License

MIT License
