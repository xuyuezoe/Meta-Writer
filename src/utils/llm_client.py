"""
通用 LLM 客户端。

功能：
    统一封装 MetaWriter 对外部模型服务的调用方式，默认兼容 OpenAI Chat
    Completions，也支持兼容 Anthropic Messages 的网关。

设计目的：
    1. 让上层生成主循环只依赖一个 `generate()` / `generate_structured()` 接口。
    2. 在客户端层吸收协议探测、有限重试、<think> 清洗和日志落盘这些脏活，
       避免把网络侧兼容逻辑扩散到 orchestrator、generator 等核心模块。
    3. 当用户怀疑“请求没有真的打到 API”时，提供可核对的本地 trace 文件。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import openai
import requests

T = TypeVar("T")


class _ProtocolCallError(RuntimeError):
    """
    单次协议调用失败，用于区分“可重试”与“应立刻终止”的错误。

    设计目的：
        让客户端在协议探测和重试时能精确判断失败语义。
        例如：
        - 协议不兼容：应尝试另一种协议
        - 服务端过载 / 无可用账号：应等待后重试
        - 明确参数错误：应立刻终止并把错误抛给上层
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
    通用 LLM 客户端。

    参数：
        api_key: 服务商提供的 API 密钥
        model: 模型名称（如 gpt-4o、MiniMax-M2.5、glm-4.6）
        base_url: API 端点；自定义网关时可只配置这一项给主流程使用

    关键设计：
        - 优先兼容 OpenAI Chat Completions
        - 自定义网关可自动探测并回退到 Anthropic Messages
        - 统一处理推理模型的 <think> 块
        - 统一统计 token、请求次数和本地 trace
        - 结构化输出优先使用 response_format，失败时自动降级
    """

    _MAX_RETRY_ROUNDS = 8
    _MAX_RETRY_DELAY_SECONDS = 8.0

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/") if base_url else None
        self._openai_base_url = self._normalize_openai_base_url(self.base_url)
        self._client = openai.OpenAI(api_key=api_key, base_url=self._openai_base_url)
        self._http = requests.Session()
        self._preferred_protocol: Optional[str] = None
        self._trace_file = self._resolve_trace_file()
        self._run_logger = None
        self.total_tokens = 0
        self.request_count = 0
        self.logger = logging.getLogger(__name__)

    def _normalize_openai_base_url(self, base_url: Optional[str]) -> Optional[str]:
        """
        规范化传给 OpenAI SDK 的 base_url。

        设计目的：
            很多兼容服务要求 SDK 的根路径以 `/v1` 结尾，但实际配置里常常只给裸 host。
            这里统一补全，避免“base_url 已配置但 SDK 实际打错路径”的隐性问题。
        """
        if not base_url:
            return None

        normalized = base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
        elif normalized.endswith("/messages"):
            normalized = normalized[: -len("/messages")]

        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1"

    def attach_run_logger(self, run_logger) -> None:
        """
        在运行日志器创建后再挂接进客户端。

        设计目的：
            避免 `LLMClient` 与 `RunLogger` 相互构造造成初始化顺序耦合，
            同时保证每次调用的 prompt / response 都能在统一日志里落盘。
        """
        self._run_logger = run_logger

    def _resolve_trace_file(self) -> Optional[Path]:
        """
        解析 API 调用 trace 文件路径。

        设计目的：
            当用户怀疑“没有真实打到 API”时，可以直接看本地 JSONL trace，
            不必只依赖服务端控制台或外部计费面板。
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

        设计目的：
            把“请求是否发出”“使用了哪个协议”“返回了什么状态”变成可审计证据，
            降低真实联调时的信息不对称。
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

    def _build_openai_chat_endpoint(self) -> str:
        """
        根据 base_url 生成 OpenAI Chat Completions 的展示用 endpoint。

        设计目的：
            trace 日志里要明确记录实际命中的接口地址，便于排查代理或网关路由问题。
            这里的地址主要用于日志展示，不改变 openai SDK 的真实请求行为。
        """
        if not self.base_url:
            return "https://api.openai.com/v1/chat/completions"

        if self._openai_base_url and self._openai_base_url.endswith("/v1"):
            return f"{self._openai_base_url}/chat/completions"
        if self._openai_base_url:
            return f"{self._openai_base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    def _build_messages_url(self) -> str:
        """
        根据 base_url 推导 Anthropic Messages 接口地址。

        设计目的：
            主流程只需要关心一个 `BASE_URL` 配置项，不需要知道底层是
            `/v1/chat/completions` 还是 `/v1/messages`。
        """
        if not self.base_url:
            raise RuntimeError("未配置 base_url，无法使用 Messages 协议")

        normalized = self.base_url.rstrip("/")
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
        if normalized.endswith("/messages"):
            return normalized
        if normalized.endswith("/v1"):
            return f"{normalized}/messages"
        return f"{normalized}/v1/messages"

    def _looks_like_openai_model(self) -> bool:
        """
        粗略判断当前模型是否更像 OpenAI 原生模型。

        设计目的：
            首次探测时优先把请求打到更可能成功的协议，减少无意义重试和额外时延。
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
        生成当前请求的候选协议顺序。

        设计目的：
            一旦某个协议成功过，后续请求尽量固定走它，避免每次都做双路探测。
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
        指数退避，给网关恢复时间。

        设计目的：
            服务端返回 503、无可用账号、限流等错误时，如果本地持续秒级猛打，
            成功率通常只会更低。
        """
        return min(1.0 * (2**round_index), self._MAX_RETRY_DELAY_SECONDS)

    def _is_retryable_message(self, message: str) -> bool:
        """根据错误消息文本判断是否值得重试。"""
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
        """根据 HTTP 状态码判断是否值得重试。"""
        return status_code in {408, 409, 429, 500, 502, 503, 504}

    def _extract_http_error(self, response: requests.Response) -> tuple[str, Optional[str]]:
        """
        尽可能从网关响应里抽取错误消息和错误类型。

        设计目的：
            不把错误处理绑死在某一个厂商格式上，尽量从代理 / 网关返回中
            还原出“是否可重试”的信号。
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
        尽可能从不同响应形态中提取 token 用量。

        设计目的：
            不论底层走的是 OpenAI 还是 Messages 协议，上层看到的统计都应保持统一。
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
        从不同协议和不同 SDK 返回对象中提取文本内容。

        设计目的：
            上层 orchestrator 只关心“文本输出”这个抽象，不应感知底层响应结构差异。
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
        *,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        调用 OpenAI Chat Completions。

        设计目的：
            保持对 OpenAI 兼容服务的直接支持，同时把 trace、错误分类和统计逻辑
            统一收敛在这一层。
        """
        self.request_count += 1
        endpoint = self._build_openai_chat_endpoint()
        started_at = time.time()
        self._trace_api_event(
            {
                "phase": "request",
                "protocol": "openai_chat",
                "endpoint": endpoint,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "response_format": response_format["type"] if response_format else None,
            }
        )

        request_kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop_sequences or None,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**request_kwargs)
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "openai_chat",
                    "endpoint": endpoint,
                    "status": "ok",
                    "duration_ms": round((time.time() - started_at) * 1000, 2),
                }
            )
            return response
        except openai.OpenAIError as error:
            status_code = getattr(error, "status_code", None)
            message = str(error)
            self._trace_api_event(
                {
                    "phase": "response",
                    "protocol": "openai_chat",
                    "endpoint": endpoint,
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
            ) from error

    def _call_anthropic_messages(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Mapping[str, Any]:
        """
        调用 Anthropic Messages 兼容接口。

        设计目的：
            某些网关只把非 OpenAI 模型挂在 `/v1/messages`，这里提供协议级回退，
            避免上层任务逻辑为了兼容网关而改写。
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
        except requests.RequestException as error:
            message = f"Anthropic Messages 调用失败：{error}"
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
            ) from error

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
            payload_object = response.json()
        except ValueError as error:
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
            ) from error

        if not isinstance(payload_object, Mapping):
            raise _ProtocolCallError(
                "anthropic_messages",
                f"Anthropic Messages 返回了无法识别的响应类型：{type(payload_object).__name__}",
                retryable=False,
                status_code=response.status_code,
            )

        if payload_object.get("type") == "error":
            error_obj = payload_object.get("error")
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
                status_code=response.status_code,
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
        return payload_object

    def _invoke_protocol(
        self,
        protocol: str,
        *,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Any:
        """
        按协议名分发到底层实现。

        设计目的：
            让 `generate()` 保持在编排层，而不是把具体网络细节散落到重试循环里。
        """
        if protocol == "openai_chat":
            return self._call_openai_chat(
                prompt,
                temperature,
                max_tokens,
                stop_sequences,
            )
        if protocol == "anthropic_messages":
            return self._call_anthropic_messages(
                prompt,
                temperature,
                max_tokens,
                stop_sequences,
            )
        raise RuntimeError(f"未知协议：{protocol}")

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]] = None,
        strip_think: bool = True,
        allow_think_only_fallback: bool = False,
        log_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        调用 LLM 生成文本。

        设计目的：
            把“协议探测”“传输层重试”“文本提取”“思维链清洗”“运行日志记录”
            都收敛到客户端内部，让调用方继续保持一个简单稳定的文本生成接口。
        """
        log_meta = dict(log_meta or {})
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
                original_text = text

                think_only = False
                if strip_think and "<think>" in text:
                    visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    if visible:
                        text = visible
                    else:
                        text = original_text if allow_think_only_fallback else ""
                        think_only = True

                if think_only:
                    log_meta["think_only_output"] = True

                log_meta["protocol"] = protocol
                self._log_llm_call(
                    prompt=prompt,
                    response=text,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    log_meta=log_meta,
                )
                return text

            if not round_retryable or round_index == self._MAX_RETRY_ROUNDS - 1:
                break
            time.sleep(self._retry_delay(round_index))

        error_tail = "; ".join(protocol_errors[-6:]) if protocol_errors else "unknown error"
        error_text = f"LLM API 调用失败：{error_tail}"
        self._log_llm_call(
            prompt=prompt,
            response=f"[ERROR] {error_text}",
            temperature=temperature,
            max_tokens=max_tokens,
            log_meta=log_meta,
        )
        raise RuntimeError(error_text)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 32768,
        log_meta: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        调用 LLM 生成结构化输出（Pydantic）。

        设计目的：
            优先走 response_format 获取稳定 JSON；如果当前网关不支持、协议不兼容
            或返回内容不可解析，再自动降级到普通文本生成 + 后处理 JSON。
        """
        import json as stdlib_json

        log_meta = dict(log_meta or {})

        try:
            response = self._call_openai_chat(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                stop_sequences=None,
                response_format={"type": "json_object"},
            )
            self._preferred_protocol = "openai_chat"
            self._collect_usage(response)

            raw_text = self._extract_text_from_response(response)
            cleaned_text = self._clean_thinking_model_output(raw_text)
            json_obj = stdlib_json.loads(cleaned_text)
            result = schema(**json_obj)

            log_meta["structured_output"] = True
            log_meta["schema"] = schema.__name__ if hasattr(schema, "__name__") else str(schema)
            log_meta["method"] = "response_format"
            log_meta["protocol"] = "openai_chat"
            self._log_llm_call(
                prompt=prompt,
                response=cleaned_text,
                temperature=temperature,
                max_tokens=max_tokens,
                log_meta=log_meta,
            )
            return result

        except _ProtocolCallError as api_error:
            self.logger.warning(
                "response_format 调用失败 ('%s' 可能不支持或协议不兼容): %s\n尝试降级使用普通生成 + 后处理...",
                self.model,
                api_error,
            )
            return self._generate_structured_fallback(prompt, schema, temperature, max_tokens, log_meta)

        except (stdlib_json.JSONDecodeError, TypeError, ValueError) as parse_error:
            self.logger.warning(
                "response_format 返回内容无法解析为 %s，尝试降级到普通生成：%s",
                schema.__name__ if hasattr(schema, "__name__") else str(schema),
                parse_error,
            )
            return self._generate_structured_fallback(prompt, schema, temperature, max_tokens, log_meta)

    def _generate_structured_fallback(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float,
        max_tokens: int,
        log_meta: Optional[Dict[str, Any]],
    ) -> T:
        """
        结构化输出的降级方案：普通生成 + JSON 后处理。

        设计目的：
            当 response_format 在某些网关、代理或模型分组上不可用时，
            主流程仍然可以继续运行，而不是直接中断整个 benchmark。
        """
        import json as stdlib_json

        if schema is None:
            raise RuntimeError("未提供可用的结构化 schema")

        fallback_meta = dict(log_meta or {})
        fallback_meta["structured_output"] = True
        fallback_meta["schema"] = schema.__name__ if hasattr(schema, "__name__") else str(schema)
        fallback_meta["method"] = "fallback_generate"

        json_prompt = (
            prompt
            + "\n\n【重要】请直接返回一个有效的 JSON 对象，不要添加任何说明、代码块或额外文本。"
            + "\n开始返回 JSON："
        )

        response_text = self.generate(
            json_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            strip_think=True,
            allow_think_only_fallback=False,
            log_meta=fallback_meta,
        )

        try:
            cleaned_text = self._clean_thinking_model_output(response_text)
            json_obj = stdlib_json.loads(cleaned_text)
            return schema(**json_obj)
        except (stdlib_json.JSONDecodeError, TypeError, ValueError) as error:
            raise RuntimeError(f"结构化输出调用失败: {error}") from error

    def _clean_thinking_model_output(self, text: str) -> str:
        """
        清洗推理模型输出中的 <think> 块和 Markdown 包装。

        设计目的：
            不同推理模型喜欢把最终 JSON 包在思维链或 ```json 代码块里，
            这里统一归一化，减少上层结构化解析失败率。
        """
        text = text.strip()
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?[\s\n]*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()

        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            text = text[json_start : json_end + 1]

        return text

    def _log_llm_call(
        self,
        prompt: str,
        response: str,
        temperature: float,
        max_tokens: int,
        log_meta: Dict[str, Any],
    ) -> None:
        """
        将一次 LLM 调用写入 RunLogger。

        设计目的：
            把协议、温度、token 上限、结构化模式等上下文一并记录下来，
            便于之后复盘 benchmark 运行过程。
        """
        if not self._run_logger:
            return

        component = str(log_meta.get("component", "LLMClient"))
        section_id = log_meta.get("section_id")
        attempt = log_meta.get("attempt")

        extra = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for key, value in log_meta.items():
            if key in {"component", "section_id", "attempt"}:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                extra[key] = value
            else:
                extra[key] = str(value)

        self._run_logger.log_llm_call(
            component=component,
            section_id=section_id,
            attempt=attempt,
            prompt_text=prompt,
            response_text=response,
            extra=extra,
        )

    def get_statistics(self) -> Dict[str, int]:
        """
        返回调用统计。

        设计目的：
            给 benchmark 汇总阶段提供统一的 token / request 计数口径，
            无需关心底层究竟走了哪种协议。
        """
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }
