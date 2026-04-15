# MetaWriter
Self-Correcting Long-Form Generation via Decision Trace Graphs

## 核心特性

MetaWriter 是一个长文本自我修正生成系统，核心创新在于引入 **决策追溯图（DTG）** 追踪生成过程中的所有决策依赖，并在验证失败时精准定位错误根源，执行有针对性的修正策略（重试、加强约束或回退）。

- **在线自我修正**：生成过程中实时验证，发现问题立即修正，无需后处理
- **决策追溯图（DTG）**：记录每个生成决策的依赖关系，支持跨节点回溯错误根源
- **智能修正策略**：根据问题类型自动选择 RETRY / STRENGTHEN_CONSTRAINT / ROLLBACK
- **完整可溯源**：每个生成决策都有明确的 reasoning 和对历史内容的引用

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

### 3. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入 API_KEY、MODEL、BASE_URL
```

### 4. 运行

```bash
python main.py
```

默认运行 `scifi_story` 任务。切换任务只需修改 `main.py` 第一行参数：

```python
TASK_NAME = "argumentative_essay"   # 改为其他任务名即可
```

每次运行会自动清理 `sessions/` 和 `outputs/` 中的旧文件，确保结果干净可复现。

运行后生成以下输出文件：

| 文件 | 内容 |
|---|---|
| `outputs/demo_text.txt` | 生成的完整文本 |
| `outputs/demo_correction_log.json` | 修正行为日志 + 统计数据 |
| `outputs/demo_dtg.json` | 决策追溯图（可视化用） |
| `sessions/demo_session.json` | 完整 session（决策链） |

### 5. 查看结果

```bash
# 查看生成的文本
cat outputs/demo_text.txt

# 查看修正日志（分析自我修正行为）
cat outputs/demo_correction_log.json

# 查看决策日志（分析 DTG）
cat sessions/demo_session.json
```

## 系统架构

```
SelfCorrectingOrchestrator（主循环）
    ↓
┌───────────┬───────────┬───────────┬───────────┐
│ Generator │ DTGStore  │ Validator │ Debugger  │
│  (生成)   │  (决策图) │  (验证)   │  (定位)   │
└───────────┴───────────┴───────────┴───────────┘
```

主循环逐节生成，每节通过四层验证（格式 → 约束 → 对齐度 → 一致性），验证失败时根据诊断结果执行修正策略。

## 修正策略说明

| 问题类型 | 触发条件 | 修正策略 |
|---|---|---|
| 对齐度极低 | DCAS < 0.5 | RETRY_SIMPLE |
| 约束违反（历史根源） | violations + suspected_source | ROLLBACK |
| 约束违反（当前问题） | violations only | STRENGTHEN_CONSTRAINT |
| 一致性问题（历史根源） | consistency + suspected_source | ROLLBACK |
| 其他 | — | RETRY_WITH_STRONGER_PROMPT |

## 项目结构

```
metawriter/
├── main.py                 # 项目入口（修改 TASK_NAME 切换任务）
├── src/
│   ├── core/               # 核心数据结构（泛化）
│   ├── agents/             # 生成器（泛化）
│   ├── memory/             # DTG存储模块（泛化）
│   ├── algorithms/         # 调试器（泛化）
│   ├── metrics/            # 评分器（泛化）
│   ├── validators/         # 验证器（泛化）
│   ├── logging/            # 日志（泛化）
│   └── orchestrator_v2.py  # 协调器（泛化）
├── examples/
│   ├── tasks/
│   │   ├── scifi_story.py          # 任务：科幻短篇故事
│   │   ├── argumentative_essay.py  # 任务：议论文（大数据与隐私）
│   │   └── survey_paper.py         # 任务：综述论文（LLM 与软件工程）
├── sessions/               # 动态生成（.gitignore）
├── outputs/                # 动态生成（.gitignore）
├── .env.example            # API 配置模板
└── README.md
```

`src/` 目录下所有模块保持泛化，接受任意任务输入。任务定义集中在 `examples/tasks/`，可通过 `python main.py --task <task_name>` 或环境变量 `TASK_NAME` 统一调度。

## 配置

在 `.env` 文件中配置：

```bash
API_KEY=your_api_key_here
MODEL=your-model-name          # 如 gpt-4o、deepseek-chat、MiniMax-M2.5
BASE_URL=https://...           # API 端点，使用 OpenAI 官方可省略
```

系统兼容所有遵循 OpenAI Chat Completions API 格式的服务（OpenAI、DeepSeek、MiniMax、智谱、本地 vLLM/Ollama 等），切换模型只需修改 `.env`，无需改动任何代码。

## 实验数据

运行后，从 `outputs/demo_correction_log.json` 可以提取：

- 首次成功率 (First-Attempt Success Rate)
- 平均尝试次数 (Average Attempts)
- 回退频率和距离 (Rollback Frequency & Distance)
- 策略使用分布 (Strategy Distribution)

## 故障排查

**Q: 生成失败，报"API key 错误"**
A: 检查 `.env` 文件中的 `API_KEY` 是否正确设置。

**Q: 重试次数过多**
A: 可能 constraints 不够明确，或 LLM 理解偏差。检查约束描述是否清晰具体。

**Q: 回退导致无限循环**
A: 系统设有最大回退次数限制（`MAX_ROLLBACKS=5`），超过后自动改为重试。3次重试失败后降级继续。

## 共创原则

为了共建更好的协作共创环境，要求各位共创作者遵循以下原则：

- 禁止直接提交到 main
- 每个任务创建独立分支
- 分支命名：类型/简短描述
- 使用中文描述
- 代码撰写勤写注释，方便其他贡献者快速理解
- 一次提交要求至少完成一份完整功能
- 尽量做到每次提交的内容小而完整
- 描述清晰具体
- pr要求描述：改动内容，测试情况，相关问题（如果有）

记住：提交前检查，合并前测试，有疑问先沟通

## 许可

MIT License
