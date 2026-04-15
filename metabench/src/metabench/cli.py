"""MetaBench 命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from metabench.pipeline import run_pipeline
from metabench.tokenization import TokenizerConfig


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="MetaBench 七维评测命令行工具")
    parser.add_argument("--samples", type=Path, required=True, help="输入样本 JSONL 路径")
    parser.add_argument("--outputs", type=Path, required=True, help="模型输出 JSONL 路径")
    parser.add_argument("--metrics", type=Path, required=True, help="中间指标 JSONL 路径")
    parser.add_argument("--baseline", type=Path, required=True, help="成本基线 JSON 路径")
    parser.add_argument("--run-output-dir", type=Path, required=True, help="运行输出目录")
    parser.add_argument("--run-id", type=str, required=True, help="运行标识")
    parser.add_argument("--model-name", type=str, required=True, help="被评测模型名")
    parser.add_argument("--judge-model", type=str, required=True, help="裁判模型名")
    parser.add_argument("--data-version", type=str, required=True, help="数据版本")
    parser.add_argument("--tokenizer-backend", type=str, required=True, help="tokenizer 后端")
    parser.add_argument("--tokenizer-name", type=str, required=True, help="tokenizer 名称")
    parser.add_argument("--power-p", type=float, required=True, help="幂平均参数")
    return parser.parse_args()


def main() -> None:
    """命令行主函数。"""

    args = parse_args()
    tokenizer_config = TokenizerConfig(
        backend=args.tokenizer_backend,
        tokenizer_name=args.tokenizer_name,
    )

    summary = run_pipeline(
        samples_path=args.samples,
        outputs_path=args.outputs,
        metrics_path=args.metrics,
        baseline_path=args.baseline,
        run_output_dir=args.run_output_dir,
        run_id=args.run_id,
        model_name=args.model_name,
        judge_model=args.judge_model,
        data_version=args.data_version,
        tokenizer_config=tokenizer_config,
        power_p=args.power_p,
    )
    print(summary)


if __name__ == "__main__":
    main()
