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
from typing import Dict, List, Union, cast


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


def _build_outline(task_type: str, must_include: List[str]) -> Dict[str, str]:
    """根据任务类型与锚点生成最小可用大纲。

    参数：
        task_type: benchmark 样本中的任务类型。
        must_include: 当前样本的核心锚点列表。

    返回值：
        Dict[str, str]：适配 Meta-Writer 的 section 大纲。

    关键实现细节：
        保持固定三段式接口，但根据样本中的主组织锚点动态改变二级结构语义，
        避免所有医学任务暴露完全相同的 outline。
    """
    organizer_candidates = [
        item
        for item in ["分类框架", "临床路径", "争议焦点", "子群分层", "实施障碍", "证据地图"]
        if item in must_include
    ]
    closing_candidates = [
        item
        for item in ["证据缺口", "开放问题", "研究议程", "未来工作"]
        if item in must_include
    ]
    organizer_label = organizer_candidates[0] if len(organizer_candidates) > 0 else "核心组织轴"
    closing_label = closing_candidates[0] if len(closing_candidates) > 0 else "未来工作"
    if task_type == "analysis":
        return {
            "sec1": "研究范围、核心概念与分析视角",
            "sec2": f"{organizer_label}驱动的代表路径比较与综合分析",
            "sec3": f"局限性、{closing_label}与收束判断",
        }
    return {
        "sec1": "研究背景、问题界定与综述范围",
        "sec2": f"{organizer_label}组织下的比较、整合与讨论",
        "sec3": f"局限性反思、{closing_label}与未来工作",
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


def _parse_int_field(raw_value: object, field_name: str) -> int:
    """解析并校验整数字段。

    参数：
        raw_value: 原始字段值。
        field_name: 字段名称。

    返回值：
        int：解析后的整数值。

    关键实现细节：
        仅接受整数或整数字符串，避免在 benchmark 配置加载时吞掉结构错误。
    """
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} 必须是整数或整数字符串")
    return int(raw_value)


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
        Meta-Writer 能直接消费的文本约束列表，同时显式加入综述型长文风格提示。
    """
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        if str(sample_row.get("sample_id")) != task_id:
            continue

        raw_constraints_object = sample_row["constraints"]
        if not isinstance(raw_constraints_object, dict):
            raise TypeError("constraints 必须是字典")
        raw_constraints = cast(Dict[str, object], raw_constraints_object)

        must_include_object = raw_constraints["must_include"]
        if not isinstance(must_include_object, list):
            raise TypeError("constraints.must_include 必须是列表")
        must_include = [str(item) for item in must_include_object]

        periodic_requirements_object = raw_constraints["periodic_requirements"]
        if not isinstance(periodic_requirements_object, list):
            raise TypeError("constraints.periodic_requirements 必须是列表")
        periodic_requirements = [str(item) for item in periodic_requirements_object]

        required_length_words = _parse_int_field(
            raw_constraints["required_length_words"],
            "constraints.required_length_words",
        )
        expected_blocks = _parse_int_field(
            raw_constraints["expected_blocks"],
            "constraints.expected_blocks",
        )

        range_keywords_object = raw_constraints["range_keywords"]
        if not isinstance(range_keywords_object, list):
            raise TypeError("constraints.range_keywords 必须是列表")
        range_keywords = [
            dict(item) if isinstance(item, dict) else {"keyword": str(item)}
            for item in range_keywords_object
        ]

        periodic_keywords_object = raw_constraints["periodic_keywords"]
        if not isinstance(periodic_keywords_object, list):
            raise TypeError("constraints.periodic_keywords 必须是列表")
        periodic_keywords = [
            dict(item) if isinstance(item, dict) else {"keyword": str(item)}
            for item in periodic_keywords_object
        ]
        task_type = str(sample_row["task_type"])

        proxy_questions_object = sample_row["proxy_questions"]
        if not isinstance(proxy_questions_object, list):
            raise TypeError("proxy_questions 必须是列表")
        checklist_object = sample_row["checklist"]
        if not isinstance(checklist_object, list):
            raise TypeError("checklist 必须是列表")
        proxy_questions_list = cast(List[object], proxy_questions_object)
        checklist_list = cast(List[object], checklist_object)

        constraints: List[str] = [
            f"目标长度约 {required_length_words} 字",
            f"正文至少形成 {expected_blocks} 个自然段，整体写成综述型长文而非简报或提纲",
            "写作风格应接近 survey paper：先界定范围，再做分类、比较、综合，最后讨论局限性与未来工作",
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
                "expected_blocks": expected_blocks,
                "range_keywords": range_keywords,
                "periodic_keywords": periodic_keywords,
            },
            "proxy_questions": [
                {
                    "qid": str(question["qid"]),
                    "question": str(question["question"]),
                    "answer": str(question["answer"]),
                }
                for question in proxy_questions_list
                if isinstance(question, dict)
            ],
            "checklist": [str(item) for item in checklist_list],
        }

        return {
            "task": str(sample_row["prompt"]),
            "constraints": constraints,
            "outline": _build_outline(task_type, must_include),
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
        在原有关键词命中基础上，额外利用段落数量、位置约束与周期性关键词，
        对综述型长文的结构信号进行软性评分。
    """
    normalized_reference = _normalize_reference(reference)
    normalized_text = generated_text.strip()
    if normalized_text == "":
        raise ValueError("generated_text 不能为空")

    raw_constraints_object = normalized_reference["constraints"]
    if not isinstance(raw_constraints_object, dict):
        raise TypeError("reference.constraints 必须是字典")
    raw_constraints = cast(Dict[str, object], raw_constraints_object)

    must_include_object = raw_constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("reference.constraints.must_include 必须是列表")
    must_include = [str(item) for item in must_include_object]

    expected_blocks = _parse_int_field(
        raw_constraints["expected_blocks"],
        "reference.constraints.expected_blocks",
    )

    range_keywords_object = raw_constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("reference.constraints.range_keywords 必须是列表")
    range_keywords = [
        dict(item) for item in range_keywords_object if isinstance(item, dict)
    ]

    periodic_keywords_object = raw_constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("reference.constraints.periodic_keywords 必须是列表")
    periodic_keywords = [
        dict(item) for item in periodic_keywords_object if isinstance(item, dict)
    ]

    checklist_object = normalized_reference["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("reference.checklist 必须是列表")
    checklist_list = cast(List[object], checklist_object)
    checklist = [str(item) for item in checklist_list]

    proxy_questions_object = normalized_reference["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("reference.proxy_questions 必须是列表")
    proxy_questions_list = cast(List[object], proxy_questions_object)
    proxy_questions = [
        dict(item) for item in proxy_questions_list if isinstance(item, dict)
    ]

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

    range_keyword_hits: List[str] = []
    missing_range_keywords: List[str] = []
    for item in range_keywords:
        keyword = str(item["keyword"])
        start_index = max(1, int(item["start"])) - 1
        end_index = min(len(paragraph_blocks), int(item["end"]))
        candidate_blocks = paragraph_blocks[start_index:end_index]
        if any(_contains_keyword(block, keyword) for block in candidate_blocks):
            range_keyword_hits.append(keyword)
        else:
            missing_range_keywords.append(keyword)

    periodic_keyword_hits: List[str] = []
    missing_periodic_keywords: List[str] = []
    for item in periodic_keywords:
        keyword = str(item["keyword"])
        every_value = int(item["every"])
        start_paragraph = max(1, int(item["start"]))
        if every_value <= 0:
            raise ValueError("periodic_keywords.every 必须为正整数")

        target_hit_count = 0
        current_paragraph = start_paragraph
        while current_paragraph <= len(paragraph_blocks):
            target_hit_count += 1
            current_paragraph += every_value

        actual_hit_count = sum(
            1
            for block in paragraph_blocks[start_paragraph - 1 :]
            if _contains_keyword(block, keyword)
        )
        if actual_hit_count >= target_hit_count and target_hit_count > 0:
            periodic_keyword_hits.append(keyword)
        else:
            missing_periodic_keywords.append(keyword)

    entity_consistency_score = len(matched_keywords) / len(must_include)
    proxy_hit_rate = len(matched_proxy_answers) / len(proxy_questions)
    paragraph_signal = min(1.0, len(paragraph_blocks) / expected_blocks)
    sentence_signal = 1.0 if sentence_count >= expected_blocks * 2 else 0.5
    range_signal = (
        len(range_keyword_hits) / len(range_keywords) if len(range_keywords) > 0 else 1.0
    )
    periodic_signal = (
        len(periodic_keyword_hits) / len(periodic_keywords)
        if len(periodic_keywords) > 0
        else 1.0
    )
    structure_signal = min(
        1.0,
        0.35 * paragraph_signal
        + 0.2 * sentence_signal
        + 0.25 * range_signal
        + 0.2 * periodic_signal,
    )
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
            "expected_blocks": expected_blocks,
            "range_keyword_hits": range_keyword_hits,
            "missing_range_keywords": missing_range_keywords,
            "periodic_keyword_hits": periodic_keyword_hits,
            "missing_periodic_keywords": missing_periodic_keywords,
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
    constraints_object = benchmark_task["constraints"]
    if not isinstance(constraints_object, list):
        raise TypeError("benchmark_task.constraints 必须是列表")
    outline_object = benchmark_task["outline"]
    if not isinstance(outline_object, dict):
        raise TypeError("benchmark_task.outline 必须是字典")
    reference_object = benchmark_task["reference"]
    if not isinstance(reference_object, dict):
        raise TypeError("benchmark_task.reference 必须是字典")
    constraints_list = cast(List[object], constraints_object)
    outline_dict = cast(Dict[str, object], outline_object)
    reference_dict = cast(Dict[str, object], reference_object)

    return {
        "task": str(benchmark_task["task"]),
        "constraints": list(constraints_list),
        "outline": dict(outline_dict),
        "reference": dict(reference_dict),
        "session_name": f"metabench_{task_id}",
    }
