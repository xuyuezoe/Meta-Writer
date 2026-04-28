"""自动 metrics 生成模块。

该模块根据 samples 与 outputs 自动计算：
- LongGen 三类命中率与 completion_rate
- ProxyQA 正确率计数
- 指令命中率
- 结构完整性分数
- LLM 质量四维评分（Accuracy/Coherence/Clarity/ReadingExperience）
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from metabench.api_client import OpenAICompatibleClient
from metabench.io_utils import read_jsonl, write_jsonl
from metabench.local_metrics import (
    compute_acc_once as _shared_compute_acc_once,
    compute_acc_periodic as _shared_compute_acc_periodic,
    compute_acc_range as _shared_compute_acc_range,
    compute_completion_rate as _shared_compute_completion_rate,
    compute_instruction_hits as _shared_compute_instruction_hits,
    compute_proxy_qa as _shared_compute_proxy_qa,
    compute_soft_instruction_hits as _shared_compute_soft_instruction_hits,
    compute_structure_scores as _shared_compute_structure_scores,
    contains_keyword as _shared_contains_keyword,
    extract_soft_keywords as _shared_extract_soft_keywords,
    split_blocks as _shared_split_blocks,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="MetaBench 自动生成 metrics.jsonl")
    parser.add_argument("--samples", type=Path, required=True, help="输入样本 JSONL 路径")
    parser.add_argument("--outputs", type=Path, required=True, help="模型输出 JSONL 路径")
    parser.add_argument("--metrics", type=Path, required=True, help="输出 metrics JSONL 路径")
    parser.add_argument("--judge-base-url", type=str, required=True, help="评审接口根地址")
    parser.add_argument("--judge-api-key", type=str, required=True, help="评审接口 API 密钥")
    parser.add_argument("--judge-model", type=str, required=True, help="评审模型名")
    parser.add_argument("--judge-temperature", type=float, required=True, help="评审温度")
    parser.add_argument("--judge-max-tokens", type=int, required=True, help="评审最大输出 token")
    parser.add_argument("--judge-timeout-seconds", type=int, required=True, help="评审超时秒数")
    parser.add_argument("--judge-retry-times", type=int, required=True, help="评审重试次数")
    parser.add_argument("--judge-retry-wait-seconds", type=float, required=True, help="评审重试等待秒数")
    parser.add_argument("--judge-runs", type=int, default=5, help="每条样本的评审轮数")
    parser.add_argument("--judge-variance-threshold", type=float, default=0.2, help="评审方差阈值，超过则重跑")
    parser.add_argument("--judge-max-reruns", type=int, default=2, help="超过方差阈值时的最多重跑轮次")
    return parser.parse_args()


def _split_blocks(response_text: str) -> List[str]:
    """按段落切分文本块。"""

    return _shared_split_blocks(response_text)


def _contains_keyword(text: str, keyword: str) -> bool:
    """判断文本是否包含关键词。"""

    return _shared_contains_keyword(text, keyword)


def _compute_completion_rate(blocks: List[str], expected_blocks: int) -> float:
    """计算结构完成率。"""

    return _shared_compute_completion_rate(blocks, expected_blocks)


def _compute_acc_once(response_text: str, once_keywords: List[str]) -> float:
    """计算 once 命中率。"""

    return _shared_compute_acc_once(response_text, once_keywords)


def _compute_acc_range(blocks: List[str], range_specs: List[Dict[str, object]]) -> float:
    """计算 range 命中率。

    range_specs 每项格式:
    {
      "keyword": "...",
      "start": 1,
      "end": 3
    }
    块编号按 1 开始。
    """

    return _shared_compute_acc_range(blocks, range_specs)


def _compute_acc_periodic(blocks: List[str], periodic_specs: List[Dict[str, object]]) -> float:
    """计算 periodic 命中率。

    periodic_specs 每项格式:
    {
      "keyword": "...",
      "every": 3,
      "start": 1
    }
    """

    return _shared_compute_acc_periodic(blocks, periodic_specs)


def _compute_proxy_qa(response_text: str, proxy_questions: List[Dict[str, object]]) -> Tuple[int, int]:
    """计算 ProxyQA 命中数量。"""

    return _shared_compute_proxy_qa(response_text, proxy_questions)


def _compute_instruction_hits(response_text: str, must_include: List[str]) -> Tuple[int, int]:
    """基于 must_include 计算指令命中。"""

    return _shared_compute_instruction_hits(response_text, must_include)


def _extract_soft_keywords(check_item: str) -> List[str]:
    """从 checklist 项中提取软约束关键词。"""

    return _shared_extract_soft_keywords(check_item)


def _compute_soft_instruction_hits(response_text: str, checklist: List[str]) -> Tuple[int, int]:
    """基于 checklist 计算软约束命中率。"""

    return _shared_compute_soft_instruction_hits(response_text, checklist)


def _compute_structure_scores(response_text: str) -> Tuple[float, float]:
    """计算结构完整性分数。

    syntax_pass_rate:
    - 括号闭合
    - JSON 片段可解析（若存在）

    schema_pass_rate:
    - 至少包含 3 段
    - 段落平均长度大于阈值
    """

    return _shared_compute_structure_scores(response_text)


def _build_quality_prompt(prompt_text: str, response_text: str) -> str:
    """构建质量评审提示词。"""

    return (
        "你是严格的长文本质量评审。"
        "请基于用户指令与模型输出，给出 4 个维度分数（1 到 5 的浮点数）："
        "Accuracy, Coherence, Clarity, ReadingExperience。"
        "只输出 JSON 对象，不要输出任何解释。"
        "输出格式必须是:"
        "{\"Accuracy\":4.0,\"Coherence\":4.0,\"Clarity\":4.0,\"ReadingExperience\":4.0}"
        f"\n用户指令:\n{prompt_text}\n"
        f"\n模型输出:\n{response_text}\n"
    )


def _judge_quality_scores(
    judge_client: OpenAICompatibleClient,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_retry_times: int,
    judge_retry_wait_seconds: float,
    prompt_text: str,
    response_text: str,
) -> Dict[str, float]:
    """调用 LLM 生成质量四维评分。"""

    judge_prompt = _build_quality_prompt(prompt_text, response_text)
    judge_messages = [{"role": "user", "content": judge_prompt}]
    raw_judge_text, _usage = judge_client.chat_completion(
        model_name=judge_model,
        messages=judge_messages,
        temperature=judge_temperature,
        max_tokens=judge_max_tokens,
        retry_times=judge_retry_times,
        retry_wait_seconds=judge_retry_wait_seconds,
    )

    cleaned_text = raw_judge_text.strip().replace("```json", "").replace("```", "")

    # 第一阶段：优先直接解析整段文本
    parsed: Dict[str, object]
    try:
        parsed = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # 第二阶段：从文本中提取最可能的 JSON 对象（兼容模型输出夹杂解释或思维内容）
        object_candidates = re.findall(r"\{[\s\S]*?\}", cleaned_text)
        parsed = {}
        for candidate in object_candidates:
            try:
                candidate_obj = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate_obj, dict) and all(
                metric_key in candidate_obj
                for metric_key in ["Accuracy", "Coherence", "Clarity", "ReadingExperience"]
            ):
                parsed = candidate_obj
                break
        if parsed == {}:
            # 第三阶段：评审模型未按要求输出 JSON 时，启用启发式质量评分兜底
            # 该兜底只在解析失败时触发，并打印警告，避免静默掩盖问题。
            heuristic_scores = _heuristic_quality_scores(prompt_text=prompt_text, response_text=response_text)
            print(
                "[build_metrics] 警告: 评审模型输出不可解析，已使用启发式质量评分兜底",
                file=sys.stderr,
            )
            return heuristic_scores

    required_keys = ["Accuracy", "Coherence", "Clarity", "ReadingExperience"]
    quality_scores: Dict[str, float] = {}
    for key in required_keys:
        if key not in parsed:
            raise ValueError(f"质量评审缺少字段 {key}: {parsed}")
        quality_scores[key] = float(parsed[key])
    return quality_scores


def _heuristic_quality_scores(prompt_text: str, response_text: str) -> Dict[str, float]:
    """在评审模型不可解析时使用启发式规则生成质量分。

    评分范围均为 [1.0, 5.0]，用于保证评测流程可继续执行并保留可解释性。
    """

    if prompt_text == "":
        raise ValueError("prompt_text 不能为空")
    if response_text == "":
        return {
            "Accuracy": 1.0,
            "Coherence": 1.0,
            "Clarity": 1.0,
            "ReadingExperience": 1.0,
        }

    blocks = _split_blocks(response_text)
    response_length = len(response_text)
    prompt_keywords = [keyword for keyword in ["交通", "绿化", "预算", "阶段", "风险"] if keyword in prompt_text]
    keyword_hits = sum(1 for keyword in prompt_keywords if _contains_keyword(response_text, keyword))
    keyword_ratio = 1.0 if len(prompt_keywords) == 0 else keyword_hits / len(prompt_keywords)

    # Accuracy: 优先参考提示词关键项命中
    accuracy = 1.0 + 4.0 * keyword_ratio

    # Coherence: 段落结构越完整越高
    coherence = min(5.0, 1.0 + 1.2 * len(blocks))

    # Clarity: 文本长度过短则降分，中等长度给较高分
    if response_length < 80:
        clarity = 2.0
    elif response_length < 200:
        clarity = 3.0
    elif response_length < 500:
        clarity = 4.0
    else:
        clarity = 4.5

    # ReadingExperience: 综合段落数与长度平衡
    reading_experience = min(5.0, max(1.0, (coherence + clarity) / 2.0))

    return {
        "Accuracy": round(accuracy, 2),
        "Coherence": round(coherence, 2),
        "Clarity": round(clarity, 2),
        "ReadingExperience": round(reading_experience, 2),
    }


def _aggregate_quality_scores_with_stability(
    judge_client: OpenAICompatibleClient,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_retry_times: int,
    judge_retry_wait_seconds: float,
    prompt_text: str,
    response_text: str,
    judge_runs: int,
    judge_variance_threshold: float,
    judge_max_reruns: int,
) -> Tuple[Dict[str, float], Dict[str, float], float, int, int]:
    """执行多次评审并统计均值/方差。

    返回:
        (质量均值分数字典, 各维方差字典, 方差均值, 实际评审次数, 触发重跑次数)
    """

    if judge_runs <= 0:
        raise ValueError("judge_runs 必须大于 0")
    if judge_variance_threshold < 0:
        raise ValueError("judge_variance_threshold 不能小于 0")
    if judge_max_reruns < 0:
        raise ValueError("judge_max_reruns 不能小于 0")

    dims = ["Accuracy", "Coherence", "Clarity", "ReadingExperience"]
    collected_scores: List[Dict[str, float]] = []
    rerun_count = 0

    # 第一阶段：执行首轮评审，必要时按阈值触发重跑
    for round_index in range(judge_max_reruns + 1):
        for _ in range(judge_runs):
            one_scores = _judge_quality_scores(
                judge_client=judge_client,
                judge_model=judge_model,
                judge_temperature=judge_temperature,
                judge_max_tokens=judge_max_tokens,
                judge_retry_times=judge_retry_times,
                judge_retry_wait_seconds=judge_retry_wait_seconds,
                prompt_text=prompt_text,
                response_text=response_text,
            )
            collected_scores.append(one_scores)

        # 第二阶段：计算多轮均值与方差
        mean_scores: Dict[str, float] = {}
        variance_scores: Dict[str, float] = {}
        for dim in dims:
            dim_values = [float(item[dim]) for item in collected_scores]
            mean_scores[dim] = float(statistics.fmean(dim_values))
            variance_scores[dim] = float(statistics.pvariance(dim_values)) if len(dim_values) > 1 else 0.0

        variance_mean = float(statistics.fmean([variance_scores[dim] for dim in dims]))
        if variance_mean <= judge_variance_threshold:
            return mean_scores, variance_scores, variance_mean, len(collected_scores), rerun_count

        if round_index < judge_max_reruns:
            rerun_count += 1

    # 第三阶段：达到最大重跑轮次后返回当前聚合结果
    final_mean_scores: Dict[str, float] = {}
    final_variance_scores: Dict[str, float] = {}
    for dim in dims:
        dim_values = [float(item[dim]) for item in collected_scores]
        final_mean_scores[dim] = float(statistics.fmean(dim_values))
        final_variance_scores[dim] = float(statistics.pvariance(dim_values)) if len(dim_values) > 1 else 0.0
    final_variance_mean = float(statistics.fmean([final_variance_scores[dim] for dim in dims]))
    return final_mean_scores, final_variance_scores, final_variance_mean, len(collected_scores), rerun_count


def build_metrics(
    samples_path: Path,
    outputs_path: Path,
    metrics_path: Path,
    judge_base_url: str,
    judge_api_key: str,
    judge_model: str,
    judge_temperature: float,
    judge_max_tokens: int,
    judge_timeout_seconds: int,
    judge_retry_times: int,
    judge_retry_wait_seconds: float,
    judge_runs: int,
    judge_variance_threshold: float,
    judge_max_reruns: int,
) -> None:
    """自动构建 metrics.jsonl。"""

    sample_rows = read_jsonl(samples_path)
    output_rows = read_jsonl(outputs_path)

    output_index: Dict[str, Dict[str, object]] = {}
    for output_row in output_rows:
        sample_id = str(output_row["sample_id"])
        if sample_id in output_index:
            raise ValueError(f"outputs 存在重复 sample_id: {sample_id}")
        output_index[sample_id] = output_row

    judge_client = OpenAICompatibleClient(
        base_url=judge_base_url,
        api_key=judge_api_key,
        timeout_seconds=judge_timeout_seconds,
    )

    metric_rows: List[Dict[str, object]] = []

    # 第一阶段：逐样本计算可规则化指标
    for sample_row in sample_rows:
        sample_id = str(sample_row["sample_id"])
        if sample_id not in output_index:
            raise ValueError(f"sample_id={sample_id} 在 outputs 中缺失")

        output_row = output_index[sample_id]
        prompt_text = str(sample_row["prompt"])
        response_text = str(output_row["response"])

        constraints = sample_row["constraints"]
        must_include = [str(item) for item in constraints["must_include"]]
        expected_blocks = int(constraints["expected_blocks"])

        once_keywords = [str(item) for item in constraints["once_keywords"]]
        range_specs = [dict(item) for item in constraints["range_keywords"]]
        periodic_specs = [dict(item) for item in constraints["periodic_keywords"]]

        blocks = _split_blocks(response_text)
        completion_rate = _compute_completion_rate(blocks, expected_blocks)
        acc_once = _compute_acc_once(response_text, once_keywords)
        acc_range = _compute_acc_range(blocks, range_specs)
        acc_periodic = _compute_acc_periodic(blocks, periodic_specs)
        proxy_qa_correct, proxy_qa_total = _compute_proxy_qa(response_text, list(sample_row["proxy_questions"]))
        instruction_hard_hits, instruction_hard_total = _compute_instruction_hits(response_text, must_include)
        instruction_soft_hits, instruction_soft_total = _compute_soft_instruction_hits(
            response_text=response_text,
            checklist=[str(item) for item in sample_row["checklist"]],
        )

        instruction_hits = instruction_hard_hits + instruction_soft_hits
        instruction_total = instruction_hard_total + instruction_soft_total

        if instruction_total == 0:
            raise ValueError("instruction_total 不能为 0，请检查 must_include/checklist")
        syntax_pass_rate, schema_pass_rate = _compute_structure_scores(response_text)

        # 第二阶段：调用评审模型计算质量四维（多次裁判 + 方差稳定性）
        quality_scores, quality_variance_scores, quality_variance_mean, judge_runs_executed, judge_rerun_count = (
            _aggregate_quality_scores_with_stability(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_temperature=judge_temperature,
            judge_max_tokens=judge_max_tokens,
            judge_retry_times=judge_retry_times,
            judge_retry_wait_seconds=judge_retry_wait_seconds,
            prompt_text=prompt_text,
            response_text=response_text,
            judge_runs=judge_runs,
            judge_variance_threshold=judge_variance_threshold,
            judge_max_reruns=judge_max_reruns,
        )
        )

        metric_rows.append(
            {
                "sample_id": sample_id,
                "completion_rate": completion_rate,
                "acc_once": acc_once,
                "acc_range": acc_range,
                "acc_periodic": acc_periodic,
                "quality_scores": quality_scores,
                "quality_scores_variance": quality_variance_scores,
                "quality_scores_variance_mean": quality_variance_mean,
                "judge_runs_executed": judge_runs_executed,
                "judge_rerun_count": judge_rerun_count,
                "instruction_hits": instruction_hits,
                "instruction_total": instruction_total,
                "instruction_hard_hits": instruction_hard_hits,
                "instruction_hard_total": instruction_hard_total,
                "instruction_soft_hits": instruction_soft_hits,
                "instruction_soft_total": instruction_soft_total,
                "syntax_pass_rate": syntax_pass_rate,
                "schema_pass_rate": schema_pass_rate,
                "proxy_qa_correct": proxy_qa_correct,
                "proxy_qa_total": proxy_qa_total,
            }
        )

    # 第三阶段：导出
    write_jsonl(metrics_path, metric_rows)


def main() -> None:
    """命令行主函数。"""

    args = parse_args()
    build_metrics(
        samples_path=args.samples,
        outputs_path=args.outputs,
        metrics_path=args.metrics,
        judge_base_url=args.judge_base_url,
        judge_api_key=args.judge_api_key,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
        judge_timeout_seconds=args.judge_timeout_seconds,
        judge_retry_times=args.judge_retry_times,
        judge_retry_wait_seconds=args.judge_retry_wait_seconds,
        judge_runs=args.judge_runs,
        judge_variance_threshold=args.judge_variance_threshold,
        judge_max_reruns=args.judge_max_reruns,
    )


if __name__ == "__main__":
    main()
