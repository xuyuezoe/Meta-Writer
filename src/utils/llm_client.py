"""
通用 LLM 客户端

默认优先兼容 OpenAI Chat Completions，也支持兼容 Anthropic Messages 的网关。
对于像 Sub2API 这类会按分组把不同模型挂到不同协议入口的服务，
客户端会自动选择成功的协议，并在服务端暂时无可用账号时做有限重试。
"""
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai
import requests


class _ProtocolCallError(RuntimeError):
    """
    单次协议调用失败，用于区分是否可重试。

    目的：
        让 generate() 能把“协议本身不兼容”和“服务端只是暂时拥塞”分开处理，
        避免把所有失败都当成永久错误直接中断主流程。
    """

    def __init__(
        self,
        protocol: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.protocol = protocol
        self.retryable = retryable
        self.status_code = status_code


class LLMClient:
    """
    通用 LLM 客户端

    参数：
        api_key:  服务商提供的 API 密钥
        model:    模型名称（如 gpt-4o、deepseek-chat、MiniMax-M2.5）
        base_url: API 端点；如果是自定义网关，会自动尝试兼容 Messages 协议

    关键设计：
        - 优先兼容 OpenAI Chat Completions API
        - 对自定义网关自动回退到 Anthropic Messages API
        - 在客户端层统一处理推理模型的 <think> 标签剥离
        - 对服务端暂时过载/无可用账号做有限重试
        - 统计 token 用量和请求次数，供调用方分析
    """

    _MAX_RETRY_ROUNDS = 8
    _MAX_RETRY_DELAY_SECONDS = 8.0

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._http = requests.Session()
        self._preferred_protocol: Optional[str] = None
        self._trace_file = self._resolve_trace_file()
        self.total_tokens = 0
        self.request_count = 0

    def _resolve_trace_file(self) -> Optional[Path]:
        """
        解析 API 调用 trace 文件路径。

        目的：
            当用户怀疑“没有真实打到 API”时，除了看服务端面板，还能直接在本地
            拿到逐次请求的时间、模型、协议与状态，降低排障时的信息不对称。
        """
        raw_path = os.getenv("LLM_API_TRACE_FILE", "").strip()
        trace_path = Path(raw_path) if raw_path else Path.cwd() / "outputs" / "llm_api_trace.jsonl"

        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return trace_path

    def _trace_api_event(self, event: Dict[str, Any]) -> None:
        """
        将单次 API 事件追加写入本地 JSONL trace。

        目的：
            让“请求是否真的发出”“命中了哪个协议”“返回了什么状态”都有落盘证据，
            避免排查时只能依赖控制台输出或外部仪表盘。
        """
        if self._trace_file is None:
            return

        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "model": self.model,
            "base_url": self.base_url,
            **event,
        }
        try:
            with self._trace_file.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _build_messages_url(self) -> str:
        """
        根据 base_url 推导兼容 Anthropic Messages 的地址。

        目的：
            主流程仍然只暴露一个 base_url 配置项，不要求上层知道
            OpenAI Chat 和 Anthropic Messages 的具体子路径差异。
        """
        if not self.base_url:
            raise RuntimeError("未配置 base_url，无法使用 Messages 协议")

        if self.base_url.endswith("/messages"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/messages"
        return f"{self.base_url}/v1/messages"

    def _looks_like_openai_model(self) -> bool:
        """
        粗略判断当前模型是否更像 OpenAI 原生模型。

        目的：
            在首次请求时尽量把探测请求打到更可能成功的协议，减少无意义重试。
        """
        lowered = self.model.strip().lower()
        openai_prefixes = (
            "gpt-",
            "chatgpt",
            "codex",
            "o1",
            "o3",
            "o4",
            "omni",
        )
        return lowered.startswith(openai_prefixes)

    def _candidate_protocols(self) -> List[str]:
        """
        给当前请求生成候选协议顺序。

        规则：
        - 一旦某个协议成功过，本轮后续都固定使用它，避免双倍请求
        - 自定义网关 + 非 OpenAI 原生模型：优先探测 Messages
        - 其他情况默认先尝试 OpenAI Chat
        """
        if self._preferred_protocol is not None:
            return [self._preferred_protocol]

        if not self.base_url:
            return ["openai_chat"]

        if self._looks_like_openai_model():
            return ["openai_chat", "anthropic_messages"]
        return ["anthropic_messages", "openai_chat"]

    def _retry_delay(self, round_index: int) -> float:
        """
        指数退避，避免服务端拥塞时持续猛打。

        目的：
            服务端返回“无可用账号/过载”时，给账号池恢复时间，
            否则本地重试越快，成功率往往越低。
        """
        return min(1.0 * (2**round_index), self._MAX_RETRY_DELAY_SECONDS)

    def _is_retryable_message(self, message: str) -> bool:
        lowered = message.strip().lower()
        if not lowered:
            return False

        retryable_markers = (
            "temporarily unavailable",
            "service temporarily unavailable",
            "please retry later",
            "upstream service overloaded",
            "overloaded",
            "no available accounts",
            "try again later",
            "timed out",
            "timeout",
            "connection error",
            "connection reset",
            "connection aborted",
            "rate limit",
            "server error",
            "bad gateway",
            "gateway timeout",
        )
        return any(marker in lowered for marker in retryable_markers)

    def _is_retryable_status(self, status_code: Optional[int]) -> bool:
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    def _extract_http_error(self, response: requests.Response) -> tuple[str, Optional[str]]:
        """
        尽可能从网关响应里抽取错误消息和类型。

        目的：
            不把错误处理绑死在单一厂商格式上，尽量从代理/网关的 JSON 或纯文本里
            还原出“是否值得重试”的信号。
        """
        body_text = response.text.strip()
        try:
            payload = response.json()
        except ValueError:
            preview = body_text[:240].replace("\n", " ")
            return preview or f"HTTP {response.status_code}", None

        if isinstance(payload, Mapping):
            error_obj = payload.get("error")
            if isinstance(error_obj, Mapping):
                message = str(error_obj.get("message", "")).strip()
                error_type = error_obj.get("type")
                if isinstance(error_type, str):
                    return message or f"HTTP {response.status_code}", error_type
                return message or f"HTTP {response.status_code}", None

            message = payload.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip(), None

        preview = body_text[:240].replace("\n", " ")
        return preview or f"HTTP {response.status_code}", None

    def _collect_usage(self, response: Any) -> None:
        """
        尽可能从响应中提取 token 使用量。

        目的：
            不管底层走的是 OpenAI 还是 Messages，都统一累计到同一份统计里，
            这样 benchmark 汇总时不需要关心具体协议。
        """
        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", None)
            if isinstance(total_tokens, int):
                self.total_tokens += total_tokens
                return

        if isinstance(response, Mapping):
            usage_object = response.get("usage")
            if isinstance(usage_object, Mapping):
                total_tokens = usage_object.get("total_tokens")
                if isinstance(total_tokens, int):
                    self.total_tokens += total_tokens
                    return

                anthropic_total = 0
                for field_name in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ):
                    value = usage_object.get(field_name)
                    if isinstance(value, int):
                        anthropic_total += value
                if anthropic_total > 0:
                    self.total_tokens += anthropic_total

    def _extract_text_from_response(self, response: Any) -> str:
        """
        从不同形态的响应对象中提取文本内容。

        兼容：
        - OpenAI Chat Completions
        - Anthropic Messages
        - 代理网关异常返回的纯文本 / HTML

        目的：
            上层 orchestrator 只接收“文本”这个抽象，不需要感知不同协议
            在响应结构上是 choices 还是 content blocks。
        """
        if isinstance(response, str):
            lowered = response.lower()
            if "<html" in lowered or "<!doctype html" in lowered:
                preview = response[:160].replace("\n", " ")
                raise RuntimeError(f"LLM API 返回了 HTML 页面而不是文本响应：{preview}")
            return response

        choices = getattr(response, "choices", None)
        if isinstance(choices, list) and choices:
            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None)
            if isinstance(content, str):
                return content

        if isinstance(response, Mapping):
            choices_object = response.get("choices")
            if isinstance(choices_object, list) and len(choices_object) > 0:
                first_choice = choices_object[0]
                if isinstance(first_choice, Mapping):
                    message = first_choice.get("message")
                    if isinstance(message, Mapping):
                        content = message.get("content")
                        if isinstance(content, str):
                            return content

            content_blocks = response.get("content")
            if isinstance(content_blocks, list):
                thinking_parts: List[str] = []
                visible_parts: List[str] = []
                for block in content_blocks:
                    if not isinstance(block, Mapping):
                        continue
                    block_type = block.get("type")
                    if block_type == "thinking" and isinstance(block.get("thinking"), str):
                        thinking_parts.append(block["thinking"])
                    elif block_type == "text" and isinstance(block.get("text"), str):
                        visible_parts.append(block["text"])

                assembled_parts: List[str] = []
                if thinking_parts:
                    # 统一包成 <think>...</think>，让上层沿用现有的剥离逻辑，
                    # 不需要为不同协议单独写一套“思维链”处理分支。
                    assembled_parts.append("<think>\n" + "\n\n".join(thinking_parts) + "\n</think>")
                if visible_parts:
                    assembled_parts.append("\n\n".join(visible_parts))
                if assembled_parts:
                    return "\n\n".join(assembled_parts).strip()

            completion = response.get("completion")
            if isinstance(completion, str):
                return completion

        raise RuntimeError(f"LLM API 返回了无法识别的响应类型：{type(response).__name__}")

    def _call_openai_chat(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Any:
        """
        调用 OpenAI Chat Completions。

        目的：
            保持对现有 OpenAI 兼容服务的直接支持，避免为了兼容特殊网关而破坏
            原本正常工作的主路径。
        """
        self.request_count += 1
        started_at = time.time()
        self._trace_api_event(
            {
                "phase": "request",
                "protocol": "openai_chat",
                "endpoint": f"{self.base_url or 'https://api.openai.com'}/chat/completions",
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences or None,
            )
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "openai_chat",
                    "endpoint": f"{self.base_url or 'https://api.openai.com'}/chat/completions",
                    "status": "ok",
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                }
            )
            return response
        except openai.OpenAIError as e:
            status_code = getattr(e, "status_code", None)
            message = str(e)
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "openai_chat",
                    "endpoint": f"{self.base_url or 'https://api.openai.com'}/chat/completions",
                    "status": "error",
                    "status_code": status_code,
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                    "error": message,
                }
            )
            raise _ProtocolCallError(
                "openai_chat",
                f"OpenAI Chat 调用失败：{message}",
                retryable=self._is_retryable_status(status_code) or self._is_retryable_message(message),
                status_code=status_code,
            ) from e

    def _call_anthropic_messages(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Mapping[str, Any]:
        """
        调用 Anthropic Messages 兼容接口。

        目的：
            一些网关会把非 OpenAI 模型只挂在 /v1/messages，而不会暴露成
            chat.completions；这里提供协议级回退，避免上层任务逻辑改写。
        """
        self.request_count += 1
        endpoint = self._build_messages_url()
        started_at = time.time()
        self._trace_api_event(
            {
                "phase": "request",
                "protocol": "anthropic_messages",
                "endpoint": endpoint,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if stop_sequences:
            payload["stop_sequences"] = stop_sequences

        try:
            response = self._http.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01",
                },
                json=payload,
                timeout=90,
            )
        except requests.RequestException as e:
            message = f"Anthropic Messages 调用失败：{e}"
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "anthropic_messages",
                    "endpoint": endpoint,
                    "status": "error",
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                    "error": message,
                }
            )
            raise _ProtocolCallError(
                "anthropic_messages",
                message,
                retryable=True,
            ) from e

        if not response.ok:
            message, error_type = self._extract_http_error(response)
            retryable = self._is_retryable_status(response.status_code) or self._is_retryable_message(message)
            if error_type in {"api_error", "overloaded_error"}:
                retryable = True
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "anthropic_messages",
                    "endpoint": endpoint,
                    "status": "error",
                    "status_code": response.status_code,
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                    "error": message,
                }
            )
            raise _ProtocolCallError(
                "anthropic_messages",
                f"Anthropic Messages 调用失败：HTTP {response.status_code} - {message}",
                retryable=retryable,
                status_code=response.status_code,
            )

        try:
            payload = response.json()
        except ValueError as e:
            preview = response.text[:240].replace("\n", " ")
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "anthropic_messages",
                    "endpoint": endpoint,
                    "status": "error",
                    "status_code": response.status_code,
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                    "error": f"non_json:{preview}",
                }
            )
            raise _ProtocolCallError(
                "anthropic_messages",
                f"Anthropic Messages 返回了非 JSON 响应：{preview}",
                retryable=False,
                status_code=response.status_code,
            ) from e

        if not isinstance(payload, Mapping):
            raise _ProtocolCallError(
                "anthropic_messages",
                f"Anthropic Messages 返回了无法识别的响应类型：{type(payload).__name__}",
                retryable=False,
                status_code=response.status_code,
            )

        if payload.get("type") == "error":
            error_obj = payload.get("error")
            message = ""
            error_type = None
            if isinstance(error_obj, Mapping):
                message = str(error_obj.get("message", "")).strip()
                maybe_type = error_obj.get("type")
                if isinstance(maybe_type, str):
                    error_type = maybe_type
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "anthropic_messages",
                    "endpoint": endpoint,
                    "status": "error",
                    "status_code": response.status_code,
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                    "error": message or "unknown error",
                }
            )
            raise _ProtocolCallError(
                "anthropic_messages",
                f"Anthropic Messages 返回错误：{message or 'unknown error'}",
                retryable=self._is_retryable_message(message) or error_type in {"api_error", "overloaded_error"},
            )

        self._trace_api_event(
            {
                "phase": "response",
                "protocol": "anthropic_messages",
                "endpoint": endpoint,
                "status": "ok",
                "status_code": response.status_code,
                "duration_ms": round((time.time() - started_at) * 1000, 2),
            }
        )
        return payload

    def _invoke_protocol(
        self,
        protocol: str,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Any:
        """按协议名分发到底层实现，目的是把 generate() 保持在“编排层”而不是“细节层”."""
        if protocol == "openai_chat":
            return self._call_openai_chat(prompt, temperature, max_tokens, stop_sequences)
        if protocol == "anthropic_messages":
            return self._call_anthropic_messages(prompt, temperature, max_tokens, stop_sequences)
        raise RuntimeError(f"未知协议：{protocol}")

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]] = None,
        strip_think: bool = True,
    ) -> str:
        """
        调用 LLM 生成文本。

        对于支持多协议的网关，会在首次调用时自动选择可用协议。
        对服务端暂时过载、无可用账号、网关 503 等情况会做有限重试。

        目的：
            把“协议探测”“传输层重试”“文本提取”都封装在客户端内部，
            让调用方继续以一个简单的 generate(prompt, ...) 接口工作。
        """
        protocol_errors: List[str] = []

        for round_index in range(self._MAX_RETRY_ROUNDS):
            protocols = self._candidate_protocols()
            round_retryable = False

            for protocol_index, protocol in enumerate(protocols):
                try:
                    response = self._invoke_protocol(
                        protocol,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop_sequences=stop_sequences,
                    )
                except _ProtocolCallError as error:
                    protocol_errors.append(f"[{error.protocol}] {error}")
                    round_retryable = round_retryable or error.retryable

                    # 非 OpenAI 原生模型优先探测 Messages。
                    # 如果 Messages 只是临时 503/过载，就直接等待后重试，
                    # 不再同一轮里额外打一遍必然无效的 chat.completions。
                    if (
                        protocol_index == 0
                        and protocol == "anthropic_messages"
                        and error.retryable
                        and len(protocols) > 1
                    ):
                        break
                    continue

                self._preferred_protocol = protocol
                self._collect_usage(response)
                text = self._extract_text_from_response(response)

                # 推理模型兼容（DeepSeek-R1、MiniMax M2.5/M2.7 等输出 <think> 块的模型）
                # strip_think=True（默认）：优先取可见输出；可见输出为空则回退到全文
                # strip_think=False：返回原始全文（含 <think> 块），供评估调用自行解析
                #
                # 目的：
                #   保持现有调用方的语义不变。
                #   生成正文时尽量隐藏思维链；做 judge/评估时又能保留完整响应。
                if strip_think and "<think>" in text:
                    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    text = visible if visible else text

                return text

            if not round_retryable or round_index == self._MAX_RETRY_ROUNDS - 1:
                break
            time.sleep(self._retry_delay(round_index))

        error_tail = "; ".join(protocol_errors[-6:]) if protocol_errors else "unknown error"
        raise RuntimeError(f"LLM API 调用失败：{error_tail}")

    def get_statistics(self) -> Dict:
        """
        返回调用统计。

        返回：
            {
                'total_tokens':  int,  # 累计消耗 token 数
                'request_count': int,  # 累计请求次数（含协议回退与重试）
            }
        """
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }
