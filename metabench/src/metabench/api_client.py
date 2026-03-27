"""OpenAI 兼容接口客户端。

该模块使用标准库实现对 Chat Completions 接口的调用，
用于 baseline 生成与质量评审两类任务。
"""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional, Tuple
from urllib import error, request
import sys


class OpenAICompatibleClient:
    """OpenAI 兼容接口客户端。

    参数:
        base_url: 接口根地址，例如 https://api.openai.com
        api_key: API 密钥。
        timeout_seconds: 单次请求超时秒数。
    """

    def __init__(self, base_url: str, api_key: str, timeout_seconds: int) -> None:
        if base_url == "":
            raise ValueError("base_url 不能为空")
        if api_key == "":
            raise ValueError("api_key 不能为空")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须大于 0")
        self.base_url: str = base_url.rstrip("/")
        self.api_key: str = api_key
        self.timeout_seconds: int = timeout_seconds
        # 是否同时发送 X-Api-Key / Api-Key 等冗余头，默认关闭以提高兼容性
        self.send_key_headers: bool = False

    def chat_completion(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        retry_times: int,
        retry_wait_seconds: float,
    ) -> Tuple[str, Optional[int]]:
        """调用 Chat Completions 接口。

        参数:
            model_name: 模型名。
            messages: 对话消息。
            temperature: 采样温度。
            max_tokens: 最大输出 token。
            retry_times: 重试次数。
            retry_wait_seconds: 重试等待时间。

        返回:
            元组 (文本响应, usage_total_tokens)。
        """

        if model_name == "":
            raise ValueError("model_name 不能为空")
        if len(messages) == 0:
            raise ValueError("messages 不能为空")
        # 允许传入 0 表示不设置 max_tokens（交由后端决定或视为无限）
        if max_tokens < 0:
            raise ValueError("max_tokens 必须为非负整数；0 表示不限制")
        if retry_times <= 0:
            raise ValueError("retry_times 必须大于 0")
        if retry_wait_seconds <= 0:
            raise ValueError("retry_wait_seconds 必须大于 0")

        # 尝试多种常见 endpoint 以提高兼容性（包含 Sub2API / proxy / api 前缀）
        prefixes = [
            self.base_url,
            f"{self.base_url}/v1",
            f"{self.base_url}/api",
            f"{self.base_url}/api/v1",
            f"{self.base_url}/proxy",
            f"{self.base_url}/proxy/v1",
        ]

        endpoints_suffix = [
            "/openai/chat/completions",
            "/openai/completions",
            "/chat/completions",
            "/completions",
            "/messages",
            "/responses",
            "/generate",
            "/openai/complete",
            "/complete",
            "/proxy/openai/completions",
            "/proxy/completions",
            "/proxy/chat/completions",
        ]

        # 对于已知使用 messages 协议的模型（例如 MiniMax），优先直接尝试 /v1/messages 路径以降低失败率
        if model_name.lower().startswith("minimax"):
            # 优先尝试明确的 /v1/messages 路径，再尝试其他 messages 变体，最后才是其它 endpoint
            messages_first = []
            # 明确把 base_url/v1/messages 放在首位
            messages_first.append(f"{self.base_url.rstrip('/')}/v1/messages")
            # 其次尝试 api/v1/messages、api/messages、proxy/v1/messages 等
            messages_first.extend([
                f"{self.base_url.rstrip('/')}/api/v1/messages",
                f"{self.base_url.rstrip('/')}/api/messages",
                f"{self.base_url.rstrip('/')}/proxy/v1/messages",
                f"{self.base_url.rstrip('/')}/proxy/messages",
                f"{self.base_url.rstrip('/')}/messages",
            ])
            # 去重并保持顺序
            seen = set()
            messages_first_unique = []
            for e in messages_first:
                if e not in seen:
                    seen.add(e)
                    messages_first_unique.append(e)
            # MiniMax 场景仅尝试 messages 协议，避免在大量无关 endpoint 上消耗重试时间
            endpoints = messages_first_unique
        else:
            endpoints = [p.rstrip("/") + s for p in prefixes for s in endpoints_suffix]

        # 构造 payload：当 max_tokens == 0 时统一省略该字段以避免后端将 0 视为非法
        def maybe_add_max(payload: Dict[str, object]) -> Dict[str, object]:
            if max_tokens > 0:
                payload["max_tokens"] = max_tokens
            return payload

        payload_chat = maybe_add_max({
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        })

        payload_completions = maybe_add_max({
            "model": model_name,
            "prompt": messages[-1]["content"] if len(messages) > 0 else "",
            "temperature": temperature,
        })

        # Responses API 风格的请求体（很多实现接受 input 或 messages）
        payload_responses = maybe_add_max({
            "model": model_name,
            "input": messages[-1]["content"] if len(messages) > 0 else "",
            "messages": messages,
            "temperature": temperature,
        })

        # Messages API 风格（Anthropic/Claude 前端示例）
        payload_messages = maybe_add_max({"model": model_name, "messages": messages, "temperature": temperature})

        # 第一阶段：按 endpoint 尝试
        for endpoint in endpoints:
            for attempt_index in range(retry_times):
                try:
                    # 选择合适的 payload：优先 chat/completions，再尝试 completions、responses、generate 等
                    if endpoint.rstrip("/").endswith("chat/completions"):
                        payload = payload_chat
                    elif endpoint.rstrip("/").endswith("/completions"):
                        payload = payload_completions
                    elif endpoint.rstrip("/").endswith("responses") or endpoint.rstrip("/").endswith("/generate"):
                        payload = payload_responses
                    elif endpoint.rstrip("/").endswith("/messages"):
                        # 明确为 messages API 使用 messages 风格的请求体（Anthropic/Minimax）
                        payload = payload_messages
                    else:
                        payload = {"model": model_name, "input": messages[-1]["content"]}
                        if max_tokens > 0:
                            payload["max_tokens"] = max_tokens

                    # 根据 endpoint / payload 动态构造 headers，减少对非目标后端的干扰
                    headers: Dict[str, str] = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}",
                        "User-Agent": "metabench/1.0",
                    }
                    if self.send_key_headers:
                        headers["X-Api-Key"] = self.api_key
                        headers["Api-Key"] = self.api_key

                    # 仅当请求体包含 messages（或 endpoint 表明是 messages API）时，添加 anthropic-version
                    if (isinstance(payload, dict) and "messages" in payload) or endpoint.rstrip("/").endswith("/messages"):
                        headers["anthropic-version"] = "2023-06-01"

                    raw_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = request.Request(url=endpoint, data=raw_bytes, headers=headers, method="POST")
                    with request.urlopen(req, timeout=self.timeout_seconds) as response_obj:
                        raw = response_obj.read()
                        # 若返回非 JSON（例如 HTML），尝试跳过
                        try:
                            response_text = raw.decode("utf-8")
                        except Exception:
                            response_text = ""
                    try:
                        response_json = json.loads(response_text)
                    except Exception:
                        # 非 JSON 响应，打印原始文本以便诊断并跳过此 endpoint
                        print(f"[api_client] 非 JSON 响应 ({endpoint}) 快照:\n{response_text[:800]}", file=sys.stderr)
                        raise ValueError("非 JSON 响应，可能为网关页面或错误 HTML")

                    # 第二阶段：弹性解析响应结构，兼容 Responses API、Chat/Completions、以及其他变体
                    content = None

                    # 1) 常见的 choices -> message/text/content
                    choices = response_json.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        first = choices[0]
                        # 有些实现的 message.content 是字符串，也有可能是类似 Claude 的 list-of-blocks
                        if isinstance(first.get("message"), dict):
                            msg = first.get("message")
                            mc = msg.get("content")
                            if isinstance(mc, str):
                                content = mc
                            elif isinstance(mc, list):
                                # 例如 [ {"type":"output_text","text":"..."}, ... ]
                                parts: List[str] = []
                                thinking_parts: List[str] = []
                                for seg in mc:
                                    if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                                        parts.append(seg.get("text"))
                                    elif isinstance(seg, dict) and isinstance(seg.get("thinking"), str):
                                        # 当后端只返回 thinking 块时，回退使用其内容，避免整条样本丢失
                                        thinking_parts.append(seg.get("thinking"))
                                    elif isinstance(seg, str):
                                        parts.append(seg)
                                if parts:
                                    content = "\n".join(parts)
                                elif thinking_parts:
                                    content = "\n".join(thinking_parts)
                        elif isinstance(first.get("text"), str):
                            content = first["text"]
                        elif isinstance(first.get("content"), str):
                            content = first["content"]

                    # 2) Responses API 常见结构：output 或 data
                    if content is None:
                        output = response_json.get("output") or response_json.get("outputs") or response_json.get("data")
                        if isinstance(output, str):
                            content = output
                        elif isinstance(output, list) and len(output) > 0:
                            parts: List[str] = []
                            for item in output:
                                if isinstance(item, str):
                                    parts.append(item)
                                elif isinstance(item, dict):
                                    # 许多实现使用 item["content"] 为 list
                                    c = item.get("content") or item.get("text")
                                    if isinstance(c, str):
                                        parts.append(c)
                                    elif isinstance(c, list):
                                        for sub in c:
                                            if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                                parts.append(sub.get("text"))
                                            elif isinstance(sub, str):
                                                parts.append(sub)
                            if len(parts) > 0:
                                content = "\n".join(parts)

                    # 3) 有些实现将主输出放在 top-level 的 "response" 字段或嵌套结构
                    if content is None:
                        resp_field = response_json.get("response")
                        if isinstance(resp_field, dict):
                            # 尝试解析 resp_field.output 或 content
                            rf_output = resp_field.get("output") or resp_field.get("outputs")
                            if isinstance(rf_output, str):
                                content = rf_output
                            elif isinstance(rf_output, list) and len(rf_output) > 0:
                                first = rf_output[0]
                                if isinstance(first, dict):
                                    c = first.get("content")
                                    if isinstance(c, str):
                                        content = c
                                    elif isinstance(c, list):
                                        texts = [seg.get("text") for seg in c if isinstance(seg, dict) and isinstance(seg.get("text"), str)]
                                        if texts:
                                            content = "\n".join(texts)

                    # 4) 某些实现直接在顶层返回 message-like 对象，包含 content/text（独立检查）
                    if content is None:
                        if isinstance(response_json.get("content"), str):
                            content = response_json.get("content")
                        elif isinstance(response_json.get("content"), list) and len(response_json.get("content")) > 0:
                            parts: List[str] = []
                            thinking_parts: List[str] = []
                            for seg in response_json.get("content"):
                                if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                                    parts.append(seg.get("text"))
                                elif isinstance(seg, dict) and isinstance(seg.get("thinking"), str):
                                    thinking_parts.append(seg.get("thinking"))
                                elif isinstance(seg, str):
                                    parts.append(seg)
                            if parts:
                                content = "\n".join(parts)
                            elif thinking_parts:
                                content = "\n".join(thinking_parts)
                        elif isinstance(response_json.get("text"), str):
                            content = response_json.get("text")

                    if not isinstance(content, str):
                        raise ValueError(f"接口返回无法解析的内容结构: {response_json}")

                    usage_total_tokens: Optional[int] = None
                    usage = response_json.get("usage")
                    if isinstance(usage, dict) and isinstance(usage.get("total_tokens"), int):
                        usage_total_tokens = int(usage["total_tokens"])

                    return content, usage_total_tokens
                except error.HTTPError as http_err:
                    # 读取返回体以便调试（可能是 JSON 或文本）
                    body_text = ""
                    try:
                        if hasattr(http_err, 'read'):
                            body_text = http_err.read().decode('utf-8', errors='ignore')
                        elif getattr(http_err, 'fp', None) is not None:
                            body_text = http_err.fp.read().decode('utf-8', errors='ignore')
                    except Exception:
                        body_text = ""
                    print(f"[api_client] HTTPError {getattr(http_err,'code',None)} from {endpoint}: {body_text[:400]}", file=sys.stderr)
                    # 若为 404，直接跳到下一个 endpoint
                    if getattr(http_err, 'code', None) == 404:
                        break
                    # 其他 HTTP 错误不立即终止，等待并重试或继续下一个 endpoint
                    time.sleep(retry_wait_seconds)
                except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as request_error:
                    # 非致命错误：等待重试或继续尝试其它 endpoint
                    print(f"[api_client] request error from {endpoint}: {request_error}", file=sys.stderr)
                    time.sleep(retry_wait_seconds)

        raise RuntimeError("接口调用失败：所有候选 endpoint 均无法调用")
