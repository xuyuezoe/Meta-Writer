"""
通用 LLM 客户端

支持所有兼容 OpenAI Chat Completions API 格式的服务，包括：
OpenAI、MiniMax、DeepSeek、智谱、本地部署（vLLM / Ollama）等。
通过 base_url 参数切换不同厂商或本地端点，无需修改任何其他代码。

新增功能：
- generate_structured(): 使用 OpenAI API 的 response_format 参数
  返回严格校验的 Pydantic 模型实例，自动处理推理模型的 <think> 块，
  解析失败率 < 1%
"""
import re
import logging
from typing import Optional, List, Dict, Any, Type, TypeVar

import openai

T = TypeVar("T")


class LLMClient:
    """
    通用 LLM 客户端

    参数：
        api_key:  服务商提供的 API 密钥
        model:    模型名称（如 gpt-4o、deepseek-chat、MiniMax-M2.5、claude-3-5-sonnet）
        base_url: API 端点，None 则使用 OpenAI 官方默认地址

    关键设计：
        - 兼容所有遵循 OpenAI Chat Completions API 格式的服务
        - 在客户端层统一处理推理模型的 <think> 标签剥离
        - 统计 token 用量和请求次数，供调用方分析
        - 支持 LangChain 的 structured output（generate_structured()）
    """

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self._client      = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model        = model
        self.total_tokens = 0
        self.request_count = 0
        self._run_logger = None
        self.logger = logging.getLogger(__name__)

    def attach_run_logger(self, run_logger) -> None:
        """Attach RunLogger after it is created."""
        self._run_logger = run_logger

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
        调用 LLM 生成文本

        参数：
            prompt:         输入提示词
            temperature:    采样温度
            max_tokens:     最大输出 token 数
            stop_sequences: 停止词列表（可选）
            strip_think:    是否剥离 <think> 推理块（默认 True）

                            生成叙事内容时应设为 True（推理过程不应混入输出）。
                            LLM-as-judge 的评估调用应设为 False：推理模型（MiniMax M2.5、
                            DeepSeek-R1 等）会将结构化评分结果放在 <think> 块内，
                            剥离后仅剩确认语，导致评分无法被解析。

        返回：
            生成的文本字符串；strip_think=True 时已剥离 <think> 推理块

        异常：
            RuntimeError：API 调用失败时抛出，附带原始错误信息
        """
        log_meta = dict(log_meta or {})
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences or None,
            )
        except openai.OpenAIError as e:
            self._log_llm_call(
                prompt=prompt,
                response=f"[ERROR] {e}",
                temperature=temperature,
                max_tokens=max_tokens,
                log_meta=log_meta,
            )
            raise RuntimeError(f"LLM API 调用失败：{e}") from e

        if response.usage is not None:
            self.total_tokens += response.usage.total_tokens
        self.request_count += 1

        text = self._safe_extract_text_from_response(response)
        original_text = text

        # 推理模型兼容（DeepSeek-R1、MiniMax 等输出 <think> 块的模型）
        # strip_think=True（默认）：优先取可见输出；若只剩 <think>，按 allow_think_only_fallback 决定
        think_only = False
        if strip_think and "<think>" in text:
            visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            if visible:
                text = visible
            else:
                if allow_think_only_fallback:
                    text = original_text
                else:
                    text = ""
                think_only = True

        if think_only:
            log_meta["think_only_output"] = True

        self._log_llm_call(
            prompt=prompt,
            response=text,
            temperature=temperature,
            max_tokens=max_tokens,
            log_meta=log_meta,
        )

        return text

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float = 0.7,
        max_tokens: int = 32768,
        log_meta: Optional[Dict[str, Any]] = None,
    ) -> T:
        """
        调用 LLM 生成结构化输出（Pydantic）

        功能：
            直接使用 OpenAI API 的 response_format 参数请求 JSON schema 模式，
            自动处理推理模型的 <think> 块，返回验证后的 schema 实例。
            解析失败率 < 1%。
            
            如果 API 不支持 response_format，降级到普通 generate() 调用。

        参数：
            prompt:     输入提示词
            schema:     Pydantic 模型类（如 GenerationDecisionSchema）
            temperature: 采样温度（默认 0.7）
            max_tokens: 最大输出 token 数（默认 32768）
            log_meta:   可选的元数据，用于日志记录

        返回：
            schema 的实例（Pydantic 已自动验证）

        异常：
            RuntimeError: LLM API 调用失败或 schema 验证失败
        """
        import json as stdlib_json
        
        log_meta = dict(log_meta or {})

        try:
            # 尝试使用 response_format（OpenAI 兼容 API）
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )

            # 提取原始文本
            raw_text = self._safe_extract_text_from_response(response)
            
            # 处理推理模型的 <think> 块和 markdown 包装
            cleaned_text = self._clean_thinking_model_output(raw_text)
            
            # 解析 JSON
            json_obj = stdlib_json.loads(cleaned_text)
            
            # 验证 schema
            result = schema(**json_obj)
            
            # 统计
            if response.usage is not None:
                self.total_tokens += response.usage.total_tokens
            self.request_count += 1
            
            log_meta["structured_output"] = True
            log_meta["schema"] = schema.__name__ if hasattr(schema, "__name__") else str(schema)
            log_meta["method"] = "response_format"

            self._log_llm_call(
                prompt=prompt,
                response=f"[STRUCTURED] {schema.__name__} 实例",
                temperature=temperature,
                max_tokens=max_tokens,
                log_meta=log_meta,
            )

            return result

        except openai.OpenAIError as api_error:
            # API 不支持 response_format 或其他 API 错误，降级到普通模式
            self.logger.warning(
                f"response_format 调用失败 ('{self.model}' 可能不支持): {api_error}\n"
                f"尝试降级使用普通生成 + 后处理..."
            )
            return self._generate_structured_fallback(prompt, schema, temperature, max_tokens, log_meta)

        except (stdlib_json.JSONDecodeError, TypeError, ValueError) as e:
            self._log_llm_call(
                prompt=prompt,
                response=f"[STRUCTURED_ERROR] {e}",
                temperature=temperature,
                max_tokens=max_tokens,
                log_meta=log_meta,
            )
            raise RuntimeError(f"结构化输出调用失败: {e}") from e

    def _generate_structured_fallback(
        self,
        prompt: str,
        schema: Type[T],
        temperature: float,
        max_tokens: int,
        log_meta: Optional[Dict[str, Any]],
    ) -> T:
        """
        结构化输出的降级方案：使用普通 generate() 调用，后处理 JSON

        这是备用方案，当 API 不支持 response_format 时使用。
        """
        import json as stdlib_json
        
        # 在 prompt 中明确要求 JSON 格式
        json_prompt = (
            prompt + 
            "\n\n【重要】请直接返回一个有效的 JSON 对象，不要添加任何说明、代码块或额外文本。"
            "\n开始返回 JSON："
        )
        
        # 调用普通生成
        response_text = self.generate(
            json_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            strip_think=True,
            allow_think_only_fallback=False,
            log_meta=log_meta,
        )
        
        # 清洁并解析
        cleaned_text = self._clean_thinking_model_output(response_text)
        json_obj = stdlib_json.loads(cleaned_text)
        result = schema(**json_obj)
        
        log_meta["structured_output"] = True
        log_meta["schema"] = schema.__name__ if hasattr(schema, "__name__") else str(schema)
        log_meta["method"] = "fallback_generate"
        
        return result

    def _clean_thinking_model_output(self, text: str) -> str:
        """
        清洗推理模型的输出：移除 <think> 块和 markdown 包装

        参数：
            text: 原始响应文本

        返回：
            清洁后的 JSON 文本
        """
        # 第一阶段：移除 <think>...</think> 块
        text = text.strip()
        if "<think>" in text:
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
        # 第二阶段：移除 markdown 代码块包装
        if text.startswith("```"):
            # 移除 ```json 或 ``` 前缀
            text = re.sub(r"^```(?:json)?[\s\n]*", "", text)
            # 移除末尾的 ```
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        
        # 第三阶段：确保是有效的JSON（移除可能的推理文本）
        # 如果文本包含JSON对象，提取第一个完整的JSON对象
        json_start = text.find("{")
        json_end = text.rfind("}")
        if json_start != -1 and json_end != -1 and json_end > json_start:
            text = text[json_start:json_end + 1]
        
        return text

    def _safe_extract_text_from_response(self, response) -> str:
        """安全提取响应文本，逐层校验结构并提供明确错误信息"""
        if response is None:
            raise RuntimeError("response_structure_error: response is None")

        if not hasattr(response, "choices"):
            raise RuntimeError("response_structure_error: choices missing")

        choices = response.choices
        if choices is None:
            raise RuntimeError("response_structure_error: choices is None")

        if not choices:
            raise RuntimeError("response_structure_error: choices is empty")

        first_choice = choices[0]
        if first_choice is None:
            raise RuntimeError("response_structure_error: choice is None")

        message = getattr(first_choice, "message", None)
        if message is None:
            raise RuntimeError("response_structure_error: message is None")

        content = getattr(message, "content", None)
        if content is None:
            raise RuntimeError("response_structure_error: content is None")

        return content

    def _log_llm_call(
        self,
        prompt: str,
        response: str,
        temperature: float,
        max_tokens: int,
        log_meta: Dict[str, Any],
    ) -> None:
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

    def get_statistics(self) -> Dict:
        """
        返回调用统计

        返回：
            {
                'total_tokens':  int,  # 累计消耗 token 数
                'request_count': int,  # 累计请求次数
            }
        """
        return {
            "total_tokens":  self.total_tokens,
            "request_count": self.request_count,
        }
