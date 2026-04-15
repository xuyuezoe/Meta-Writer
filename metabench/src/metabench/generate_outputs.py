"""Baseline 输出生成器。

该模块读取 samples.jsonl，使用 OpenAI 兼容 API 逐条生成响应，
输出为 pipeline 可直接消费的 outputs.jsonl。
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List

from metabench.api_client import OpenAICompatibleClient
from metabench.io_utils import read_jsonl, write_jsonl


def _curl_messages_once(
    base_url: str,
    api_key: str,
    model_name: str,
    request_messages: List[Dict[str, str]],
    temperature: float,
    timeout_seconds: int,
) -> str:
    """通过 curl 调用 /v1/messages 并提取文本。

    返回值:
        解析出的文本，失败时返回空字符串。
    """

    try:
        import subprocess

        url = base_url.rstrip("/") + "/v1/messages"
        payload = json.dumps(
            {
                "model": model_name,
                "messages": request_messages,
                "temperature": temperature,
            },
            ensure_ascii=False,
        )
        cmd = [
            "curl",
            "-s",
            "--max-time",
            str(timeout_seconds),
            "-X",
            "POST",
            url,
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "-H",
            "anthropic-version: 2023-06-01",
            "-d",
            payload,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds + 3)
        if proc.returncode != 0 or not proc.stdout:
            return ""

        js = json.loads(proc.stdout)
        if not isinstance(js, dict):
            return ""

        if isinstance(js.get("content"), list):
            parts: List[str] = []
            thinking_parts: List[str] = []
            for seg in js.get("content"):
                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                    parts.append(seg.get("text"))
                elif isinstance(seg, dict) and isinstance(seg.get("thinking"), str):
                    thinking_parts.append(seg.get("thinking"))
                elif isinstance(seg, str):
                    parts.append(seg)
            if len(parts) > 0:
                return "\n".join(parts)
            if len(thinking_parts) > 0:
                return "\n".join(thinking_parts)
        if isinstance(js.get("text"), str):
            return js.get("text")
        return ""
    except Exception:
        return ""


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="MetaBench Baseline 输出生成工具")
    parser.add_argument("--samples", type=Path, required=True, help="输入样本 JSONL 路径")
    parser.add_argument("--outputs", type=Path, required=True, help="输出结果 JSONL 路径")
    parser.add_argument("--base-url", type=str, required=True, help="OpenAI 兼容接口根地址")
    parser.add_argument("--api-key", type=str, required=True, help="API 密钥")
    parser.add_argument("--model-name", type=str, required=True, help="生成模型名")
    parser.add_argument("--temperature", type=float, required=True, help="生成温度")
    parser.add_argument("--max-tokens", type=int, required=True, help="最大输出 token")
    parser.add_argument("--timeout-seconds", type=int, required=True, help="请求超时秒数")
    parser.add_argument("--retry-times", type=int, required=True, help="重试次数")
    parser.add_argument("--retry-wait-seconds", type=float, required=True, help="重试等待秒数")
    parser.add_argument("--force-curl-fallback", action="store_true", help="强制启用 curl 回退以尝试 /v1/messages（当 client 解析失败或返回空时）")
    parser.add_argument("--use-curl-only", action="store_true", help="仅使用 curl 调用 /v1/messages，绕过 urllib client")
    return parser.parse_args()


def run_generation(
    samples_path: Path,
    outputs_path: Path,
    base_url: str,
    api_key: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    retry_times: int,
    retry_wait_seconds: float,
    force_curl_fallback: bool,
    use_curl_only: bool,
) -> None:
    """执行 baseline 输出生成。"""

    client = OpenAICompatibleClient(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    sample_rows = read_jsonl(samples_path)
    output_rows: List[Dict[str, object]] = []

    # 第一阶段：逐条推理
    for sample_row in sample_rows:
        sample_id = str(sample_row["sample_id"])
        prompt = str(sample_row["prompt"])
        request_messages = [{"role": "user", "content": prompt}]

        start_time = time.time()
        try:
            if use_curl_only:
                response_text = _curl_messages_once(
                    base_url=base_url,
                    api_key=api_key,
                    model_name=model_name,
                    request_messages=request_messages,
                    temperature=temperature,
                    timeout_seconds=timeout_seconds,
                )
                usage_total_tokens = None
                latency_seconds = time.time() - start_time
                if response_text == "":
                    raise RuntimeError("curl-only 模式未返回可解析文本")
            else:
                response_text, usage_total_tokens = client.chat_completion(
                    model_name=model_name,
                    messages=request_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    retry_times=retry_times,
                    retry_wait_seconds=retry_wait_seconds,
                )
                latency_seconds = time.time() - start_time
        except Exception as e:
            # 单条样本失败时记录占位输出并继续，避免整个 run 被中断
            print(f"[generate_outputs] sample_id={sample_id} 推理失败: {e}", file=sys.stderr)
            # 尝试直接通过 /v1/messages 快速回退（使用 curl），以便在 client 解析失败时仍能获取文本
            response_text = _curl_messages_once(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                request_messages=request_messages,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
            # 若回退也失败，则保留空字符串占位
            usage_total_tokens = None
            latency_seconds = -1.0

        # 当启用了强制 curl 回退且 client 返回空字符串时，也尝试一次 curl 回退（避免 client 解析失败导致空响应）
        if force_curl_fallback and not response_text:
            response_text = _curl_messages_once(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                request_messages=request_messages,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            )
        # 第二阶段：写入标准输出结构
        output_rows.append(
            {
                "sample_id": sample_id,
                "model_name": model_name,
                "response": response_text,
                "latency_seconds": latency_seconds,
                "usage_total_tokens": usage_total_tokens,
            }
        )

    write_jsonl(outputs_path, output_rows)
    print(f"[generate_outputs] 写入输出文件 {outputs_path}，共 {len(output_rows)} 条记录")


def main() -> None:
    """命令行主函数。"""

    args = parse_args()
    run_generation(
        samples_path=args.samples,
        outputs_path=args.outputs,
        base_url=args.base_url,
        api_key=args.api_key,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
        retry_times=args.retry_times,
        retry_wait_seconds=args.retry_wait_seconds,
        force_curl_fallback=args.force_curl_fallback,
        use_curl_only=args.use_curl_only,
    )


if __name__ == "__main__":
    main()
