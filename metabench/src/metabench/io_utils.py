"""MetaBench 输入输出工具模块。

该模块负责 JSON/JSONL 读写、数据对象构建与运行产物导出。
"""

from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

from metabench.schemas import (
    ConstraintSpec,
    EvalMetrics,
    InputSample,
    ModelOutput,
    ProxyQuestion,
    SampleResult,
    TokenUsage,
)


def read_json(path: Path) -> Dict[str, object]:
    """读取 JSON 文件。"""

    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    """读取 JSONL 文件。"""

    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line_data = line.strip()
            if line_data == "":
                continue
            rows.append(json.loads(line_data))
    return rows


def write_json(path: Path, payload: Dict[str, object]) -> None:
    """写入 JSON 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, payload_list: Iterable[Dict[str, object]]) -> None:
    """写入 JSONL 文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for payload in payload_list:
            file_obj.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_input_sample(raw_item: Dict[str, object]) -> InputSample:
    """从原始字典构建 InputSample。"""

    raw_constraints = raw_item["constraints"]
    constraints = ConstraintSpec(
        required_length_words=int(raw_constraints["required_length_words"]),
        must_include=list(raw_constraints["must_include"]),
        periodic_requirements=list(raw_constraints["periodic_requirements"]),
    )

    proxy_questions: List[ProxyQuestion] = []
    for question_item in raw_item["proxy_questions"]:
        proxy_questions.append(
            ProxyQuestion(
                qid=str(question_item["qid"]),
                question=str(question_item["question"]),
                answer=str(question_item["answer"]),
            )
        )

    return InputSample(
        sample_id=str(raw_item["sample_id"]),
        task_type=str(raw_item["task_type"]),
        prompt=str(raw_item["prompt"]),
        constraints=constraints,
        proxy_questions=proxy_questions,
        checklist=[str(item) for item in raw_item["checklist"]],
    )


def build_model_output(raw_item: Dict[str, object]) -> ModelOutput:
    """从原始字典构建 ModelOutput。"""

    token_usage_data = raw_item["token_usage"]
    token_usage = TokenUsage(
        prompt_tokens=int(token_usage_data["prompt_tokens"]),
        completion_tokens=int(token_usage_data["completion_tokens"]),
        total_tokens=int(token_usage_data["total_tokens"]),
        usage_total_tokens=(
            None
            if token_usage_data.get("usage_total_tokens") is None
            else int(token_usage_data["usage_total_tokens"])
        ),
    )

    return ModelOutput(
        sample_id=str(raw_item["sample_id"]),
        model_name=str(raw_item["model_name"]),
        response=str(raw_item["response"]),
        latency_seconds=float(raw_item["latency_seconds"]),
        token_usage=token_usage,
    )


def build_eval_metrics(raw_item: Dict[str, object]) -> EvalMetrics:
    """从原始字典构建 EvalMetrics。"""

    quality_scores: Dict[str, float] = {}
    for dim_name, dim_score in raw_item["quality_scores"].items():
        quality_scores[str(dim_name)] = float(dim_score)

    instruction_hits = int(raw_item["instruction_hits"])
    instruction_total = int(raw_item["instruction_total"])

    # 兼容旧 metrics：若缺失硬/软字段，则默认全部归为硬约束
    instruction_hard_hits = int(raw_item.get("instruction_hard_hits", instruction_hits))
    instruction_hard_total = int(raw_item.get("instruction_hard_total", instruction_total))
    instruction_soft_hits = int(raw_item.get("instruction_soft_hits", 0))
    instruction_soft_total = int(raw_item.get("instruction_soft_total", 0))

    return EvalMetrics(
        completion_rate=float(raw_item["completion_rate"]),
        acc_once=float(raw_item["acc_once"]),
        acc_range=float(raw_item["acc_range"]),
        acc_periodic=float(raw_item["acc_periodic"]),
        quality_scores=quality_scores,
        instruction_hits=instruction_hits,
        instruction_total=instruction_total,
        instruction_hard_hits=instruction_hard_hits,
        instruction_hard_total=instruction_hard_total,
        instruction_soft_hits=instruction_soft_hits,
        instruction_soft_total=instruction_soft_total,
        syntax_pass_rate=float(raw_item["syntax_pass_rate"]),
        schema_pass_rate=float(raw_item["schema_pass_rate"]),
        proxy_qa_correct=int(raw_item["proxy_qa_correct"]),
        proxy_qa_total=int(raw_item["proxy_qa_total"]),
    )


def sample_result_to_dict(sample_result: SampleResult) -> Dict[str, object]:
    """将 SampleResult 序列化为字典。"""

    return asdict(sample_result)


def write_radar_csv(path: Path, sample_results: List[SampleResult]) -> None:
    """导出七维雷达图 CSV 数据。"""

    rows: List[Dict[str, object]] = []
    for result in sample_results:
        rows.append(
            {
                "sample_id": result.sample_id,
                "s_stability": result.scores.s_stability,
                "s_quality": result.scores.s_quality,
                "s_length": result.scores.s_length,
                "s_info": result.scores.s_info,
                "s_instruction": result.scores.s_instruction,
                "s_structure": result.scores.s_structure,
                "s_cost": result.scores.s_cost,
                "overall_score": result.overall_score,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    field_names = [
        "sample_id",
        "s_stability",
        "s_quality",
        "s_length",
        "s_info",
        "s_instruction",
        "s_structure",
        "s_cost",
        "overall_score",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=field_names)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
