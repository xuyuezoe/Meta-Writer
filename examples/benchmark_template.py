"""Benchmark 任务适配器。

功能：
    将仓库内 `metabench/examples/samples.jsonl` 中的样本转换为 Meta-Writer
    可直接消费的任务配置，并提供一个无需外部裁判模型的本地评估函数。

关键实现细节：
    1. `load_benchmark_task` 直接读取本地 JSONL 样本。
    2. `evaluate_output` 使用规则化启发式检查约束满足、信息命中与结构完整性。
    3. `reference` 返回结构化字典，保留评估所需最小信息，不依赖外部文档。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union


BENCHMARK_ROOT: Path = Path(__file__).resolve().parent.parent / "metabench"
SAMPLES_PATH: Path = BENCHMARK_ROOT / "examples" / "samples.jsonl"


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """读取 JSONL 文件。

    参数：
        file_path: JSONL 文件路径。

    返回值：
        List[Dict[str, object]]：逐行解析后的字典列表。

    关键实现细节：
        遇到空行时跳过，避免文稿整理后出现格式噪声导致加载失败。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"未找到 benchmark 样本文件：{file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        rows.append(json.loads(stripped_line))
    return rows


def list_benchmark_task_ids() -> List[str]:
    """列出当前可用的 benchmark 任务 ID。

    参数：
        无。

    返回值：
        List[str]：按文件顺序返回样本 ID 列表。

    关键实现细节：
        直接复用本地 JSONL 解析结果，确保注册表与样本源始终一致。
    """
    task_ids: List[str] = []
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        sample_id = str(sample_row["sample_id"])
        if sample_id == "":
            raise ValueError("benchmark sample_id 不能为空")
        task_ids.append(sample_id)
    return task_ids


def _build_outline(task_type: str) -> Dict[str, str]:
    """根据任务类型生成最小可用大纲。

    参数：
        task_type: benchmark 样本中的任务类型。

    返回值：
        Dict[str, str]：适配 Meta-Writer 的 section 大纲。

    关键实现细节：
        采用固定三段式结构，避免为 benchmark 样本额外维护独立大纲文档。
    """
    if task_type == "analysis":
        return {
            "sec1": "问题背景与核心目标",
            "sec2": "关键分析与约束展开",
            "sec3": "结论、风险与建议",
        }
    return {
        "sec1": "任务开篇与目标界定",
        "sec2": "主体展开与约束覆盖",
        "sec3": "总结与收束",
    }


def _normalize_reference(reference: Union[str, Dict[str, object]]) -> Dict[str, object]:
    """规范化评估参考信息。

    参数：
        reference: 结构化参考字典，或 JSON 字符串形式的参考信息。

    返回值：
        Dict[str, object]：统一后的参考信息字典。

    关键实现细节：
        若传入字符串，则必须是合法 JSON，避免默认值掩盖字段缺失问题。
    """
    if isinstance(reference, dict):
        return reference
    if isinstance(reference, str):
        parsed_reference = json.loads(reference)
        if not isinstance(parsed_reference, dict):
            raise ValueError("reference JSON 必须解析为字典")
        return parsed_reference
    raise TypeError("reference 必须是 Dict[str, object] 或 JSON 字符串")


def _contains_keyword(text: str, keyword: str) -> bool:
    """判断文本中是否包含关键字。

    参数：
        text: 待检查文本。
        keyword: 关键字。

    返回值：
        bool：是否命中。

    关键实现细节：
        统一转为小写，兼容中英文字段混写场景。
    """
    return keyword.lower() in text.lower()


def load_benchmark_task(task_id: str) -> Dict[str, object]:
    """从本地 benchmark 样本加载任务。

    参数：
        task_id: benchmark 中的任务标识符。

    返回值：
        Dict[str, object]：
            - task: 任务描述
            - constraints: 约束列表
            - outline: 章节大纲
            - reference: 评估参考信息

    关键实现细节：
        直接从 `metabench/examples/samples.jsonl` 检索匹配样本，并将约束压平为
        Meta-Writer 能直接消费的文本约束列表。
    """
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        if str(sample_row.get("sample_id")) != task_id:
            continue

        raw_constraints = sample_row["constraints"]
        must_include = [str(item) for item in raw_constraints["must_include"]]
        periodic_requirements = [
            str(item) for item in raw_constraints["periodic_requirements"]
        ]
        required_length_words = int(raw_constraints["required_length_words"])
        task_type = str(sample_row["task_type"])

        constraints: List[str] = [
            f"目标长度约 {required_length_words} 词",
            *[f"必须覆盖：{item}" for item in must_include],
            *[f"周期性要求：{item}" for item in periodic_requirements],
        ]

        reference: Dict[str, object] = {
            "sample_id": str(sample_row["sample_id"]),
            "task_type": task_type,
            "prompt": str(sample_row["prompt"]),
            "constraints": {
                "required_length_words": required_length_words,
                "must_include": must_include,
                "periodic_requirements": periodic_requirements,
            },
            "proxy_questions": [
                {
                    "qid": str(question["qid"]),
                    "question": str(question["question"]),
                    "answer": str(question["answer"]),
                }
                for question in sample_row["proxy_questions"]
            ],
            "checklist": [str(item) for item in sample_row["checklist"]],
        }

        return {
            "task": str(sample_row["prompt"]),
            "constraints": constraints,
            "outline": _build_outline(task_type),
            "reference": reference,
        }

    raise ValueError(f"未找到 benchmark 任务：{task_id}")


def evaluate_output(
    generated_text: str, reference: Union[str, Dict[str, object]]
) -> Dict[str, object]:
    """评估生成结果。

    参数：
        generated_text: Meta-Writer 生成的文本。
        reference: benchmark 任务参考信息。

    返回值：
        Dict[str, object]：
            - constraint_violation_rate: 约束违反率
            - entity_consistency_score: 关键实体一致性评分
            - logical_coherence: 逻辑连贯性评分
            - diagnostics: 详细诊断信息

    关键实现细节：
        采用本地可复现的启发式规则，不依赖外部 judge 模型，便于 example 接口直连跑通。
    """
    normalized_reference = _normalize_reference(reference)
    normalized_text = generated_text.strip()
    if normalized_text == "":
        raise ValueError("generated_text 不能为空")

    raw_constraints = normalized_reference["constraints"]
    must_include = [str(item) for item in raw_constraints["must_include"]]
    checklist = [str(item) for item in normalized_reference["checklist"]]
    proxy_questions = [dict(item) for item in normalized_reference["proxy_questions"]]

    matched_keywords = [
        item for item in must_include if _contains_keyword(normalized_text, item)
    ]
    matched_proxy_answers = [
        str(item["qid"])
        for item in proxy_questions
        if _contains_keyword(normalized_text, str(item["answer"]))
    ]
    paragraph_blocks = [
        block.strip() for block in normalized_text.splitlines() if block.strip() != ""
    ]
    sentence_count = sum(
        1
        for part in normalized_text.replace("！", "。").replace("？", "。").split("。")
        if part.strip() != ""
    )

    if len(must_include) == 0:
        raise ValueError("reference.constraints.must_include 不能为空")
    if len(proxy_questions) == 0:
        raise ValueError("reference.proxy_questions 不能为空")

    entity_consistency_score = len(matched_keywords) / len(must_include)
    proxy_hit_rate = len(matched_proxy_answers) / len(proxy_questions)
    structure_signal = 1.0 if len(paragraph_blocks) >= 3 or sentence_count >= 3 else 0.5
    checklist_signal = len(matched_keywords) / max(len(checklist), len(must_include))
    logical_coherence = min(
        1.0, 0.5 * structure_signal + 0.3 * proxy_hit_rate + 0.2 * checklist_signal
    )
    constraint_violation_rate = 1.0 - entity_consistency_score

    return {
        "constraint_violation_rate": constraint_violation_rate,
        "entity_consistency_score": entity_consistency_score,
        "logical_coherence": logical_coherence,
        "diagnostics": {
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                item for item in must_include if item not in matched_keywords
            ],
            "matched_proxy_question_ids": matched_proxy_answers,
            "paragraph_count": len(paragraph_blocks),
            "sentence_count": sentence_count,
            "checklist": checklist,
        },
    }


def build_benchmark_task_config(task_id: str) -> Dict[str, object]:
    """构造可注册到 Meta-Writer 的 benchmark 任务配置。

    参数：
        task_id: benchmark 样本 ID。

    返回值：
        Dict[str, object]：包含任务、约束、大纲、参考信息与会话名。

    关键实现细节：
        统一在这里生成 `session_name`，避免不同 example 文件重复拼接命名逻辑。
    """
    benchmark_task = load_benchmark_task(task_id)
    return {
        "task": str(benchmark_task["task"]),
        "constraints": list(benchmark_task["constraints"]),
        "outline": dict(benchmark_task["outline"]),
        "reference": dict(benchmark_task["reference"]),
        "session_name": f"metabench_{task_id}",
    }
