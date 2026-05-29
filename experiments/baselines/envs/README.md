# 对比系统接入配置（envs/）

本目录是四个外部对比系统（AutoSurvey / LiRA / SurveyForge / PaperOrchestra）的接入配置区。
MetaWriter 通过"子进程 + 独立解释器 + 标准化桥接"的方式真实驱动这些系统，
以保证环境隔离（各系统依赖互不污染）与对比公平（同输入、同评估）。

## 一、接入原理

```
MetaWriter 适配器                          外部系统 repo（独立 venv）
─────────────────                          ────────────────────────
写出 input.json   ──(--input)──►  bridge 脚本读取 input.json
                                   └─► 调用该系统自身管线生成综述
读取 output.txt   ◄─(--output)──  bridge 脚本写出最终文本
读取 output_stats.json (可选) ◄─(--stats)── bridge 写出 token 统计
```

适配器与外部系统之间**只通过文件契约通信**，不共享 Python 进程，因此各系统可使用
自己的虚拟环境与依赖版本。

## 二、你需要做的三件事

### 1. 克隆并准备每个系统的运行环境

为每个系统建立独立虚拟环境并安装其依赖（按各 repo 的官方说明）：

- AutoSurvey: https://github.com/AutoSurveys/AutoSurvey
- LiRA: https://github.com/lira-workflow/auto-review-writing
- SurveyForge: https://github.com/Alpha-Innovator/SurveyForge
- PaperOrchestra: https://github.com/google-research/paper-orchestra

### 2. 为每个系统编写桥接脚本（bridge）

参照 `bridge_template.py`，在每个系统的 repo 内放置一个桥接脚本。它的职责：
读取标准 `input.json` → 调用该系统真实管线 → 把最终综述写入 `--output` 指定路径，
并可选地把 `{"total_tokens":int,"request_count":int}` 写入 `--stats` 路径。

桥接脚本是接入工作的核心——它是"标准契约"与"各系统私有 API"之间唯一的翻译层。

### 3. 填写每个系统的 `<system>.json` 配置

把 `repo_path` / `python_executable` / `bridge_script` 填为真实路径：

```json
{
  "repo_path": "/abs/path/to/AutoSurvey",
  "python_executable": "/abs/path/to/AutoSurvey/.venv/bin/python",
  "bridge_script": "/abs/path/to/AutoSurvey/metawriter_bridge.py",
  "extra_args": [],
  "timeout_seconds": 3600
}
```

### 4. 填写统一 API 配置 `baseline_env.env`

复制 `baseline_env.env.template` 为 `baseline_env.env`，填入各系统所需的 API 密钥
与端点。该文件会被注入所有外部系统子进程的环境变量中。

## 三、标准 input.json 字段

| 字段 | 含义 |
|------|------|
| `task_id` | 任务 ID（如 med_s001） |
| `topic` | 综述主题 |
| `task_description` | 完整任务描述（自然语言 prompt） |
| `constraints` | 约束关键词列表 |
| `outline` | 章节大纲 {section_id: title} |
| `corpus_dir` | 检索语料库目录（所有系统共享同一语料） |
| `target_words` | 目标总词数 |

> 注意：`input.json` **刻意不含评估真值（reference）**，以保证对比系统在信息对称
> 前提下只看到任务输入，杜绝评估泄漏。

## 四、契约校验

未填写配置时，运行会显式报错（指明缺失项），绝不会静默产出空文本——
这是实验可信度的底线。
