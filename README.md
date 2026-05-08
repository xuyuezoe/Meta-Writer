# MetaWriter
Self-Correcting Long-Form Generation via Decision Trace Graphs

## 核心特性

MetaWriter 是一个长文本自我修正生成系统，核心创新在于引入 **决策追溯图（DTG）** 追踪生成过程中的决策依赖，并在验证失败时执行有针对性的修正策略。

- **在线自我修正**：生成过程中实时验证，发现问题立即修正
- **决策追溯图（DTG）**：记录每次生成决策及其依赖关系
- **分层修复策略**：根据错误类型自动选择重试、加强约束或局部回退
- **完整可追溯**：运行日志、修正日志、DTG 和 session 都会落盘

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo>
cd metawriter
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置 API

```bash
cp .env.example .env
# 编辑 .env，填入 API_KEY、MODEL、BASE_URL
```

### 4. 运行

普通任务：

```bash
python main.py --task survey_paper
python main.py --task argumentative_essay
python -m main --task scifi_story
```

benchmark：

```bash
python main.py --task-id med_s010
python main.py --all
python -m main --task-id med_s010
```

默认情况下，`python main.py` / `python -m main` 会直接运行完整的 metabench 批量任务，也就是与 `--all` 相同的正式 benchmark 主链路。

### 5. 查看结果

每次运行会自动清理对应 `session_name` 下的旧输出，避免历史结果混入当前实验。

常见产物如下：

| 文件 | 内容 |
|---|---|
| `outputs/{session_name}_text.txt` | 生成的完整文本 |
| `outputs/{session_name}_correction_log.json` | 修正行为日志 |
| `outputs/{session_name}_dtg.json` | 决策追溯图 |
| `outputs/{session_name}_summary.json` | 单次运行摘要 |
| `outputs/{session_name}_benchmark_eval.json` | benchmark 评估结果 |
| `outputs/{session_name}_run.log` | 完整运行日志 |
| `sessions/{session_name}.json` | 完整 session |

## Benchmark

benchmark 的正式入口已经接入 `main.py`，不再依赖单独的 demo 脚本。

设计上分两层：

- `examples/benchmark_template.py`：负责加载本地 benchmark 样本，并提供本地评估函数
- `main.py`：负责把 benchmark 样本接进 MetaWriter 的真实生成主循环

运行 `--task-id` 或 `--all` 时，流程会：

1. 从本地 `metabench/examples/samples.jsonl` 加载样本
2. 通过 `TASK_REGISTRY` 动态注册为正式任务
3. 走 MetaWriter 的真实生成、验证、修正主循环
4. 对真实生成结果调用 `evaluate_output()` 做本地评估

## 项目结构

```text
metawriter/
├── main.py
├── examples/
│   ├── benchmark_template.py
│   └── tasks/
│       ├── argumentative_essay.py
│       ├── metabench_sample.py
│       ├── scifi_story.py
│       └── survey_paper.py
├── metabench/
│   ├── examples/
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
└── README.md
```

## 配置说明

在 `.env` 中配置：

```bash
API_KEY=your_api_key_here
MODEL=your-model-name
BASE_URL=https://...
```

系统默认优先兼容 OpenAI Chat Completions，也支持通过同一个 `BASE_URL` 自动探测兼容 Anthropic Messages 的网关。

如果需要核对请求是否真的打到了服务端，可以查看：

```text
outputs/llm_api_trace.jsonl
```

这个 trace 会记录每次请求的时间、协议、endpoint 和状态。

## 故障排查

**Q: 报错说没有 API_KEY**

A: 检查 `.env` 中是否正确设置了 `API_KEY`。

**Q: benchmark 跑的是不是 demo 结果而不是真实生成？**

A: 不是。`main.py --task-id ...` / `main.py --all` 会走 MetaWriter 的真实生成主循环，评估也基于真实生成文本而不是预存输出。

**Q: 如何确认请求是否真的发出？**

A: 除了服务端面板，还可以直接检查 `outputs/llm_api_trace.jsonl` 与 `outputs/{session_name}_run.log`。

## 共创原则

- 禁止直接提交到 `main`
- 每个任务创建独立分支
- 分支命名：类型/简短描述
- 使用中文描述
- 注释除了说明功能，也尽量说明设计目的
- 提交前先自测，合并前先沟通

## 许可

MIT License
