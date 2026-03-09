"""
通用 LLM 客户端

支持所有兼容 OpenAI Chat Completions API 格式的服务，包括：
OpenAI、MiniMax、DeepSeek、智谱、本地部署（vLLM / Ollama）等。
通过 base_url 参数切换不同厂商或本地端点，无需修改任何其他代码。
"""
import re
import openai
from typing import Optional, List, Dict


class LLMClient:
    """
    通用 LLM 客户端

    参数：
        api_key:  服务商提供的 API 密钥
        model:    模型名称（如 gpt-4o、deepseek-chat、MiniMax-M2.5）
        base_url: API 端点，None 则使用 OpenAI 官方默认地址

    关键设计：
        - 兼容所有遵循 OpenAI Chat Completions API 格式的服务
        - 在客户端层统一处理推理模型的 <think> 标签剥离
        - 统计 token 用量和请求次数，供调用方分析
    """

    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None):
        self._client      = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model        = model
        self.total_tokens = 0
        self.request_count = 0

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        调用 LLM 生成文本

        参数：
            prompt:         输入提示词
            temperature:    采样温度
            max_tokens:     最大输出 token 数
            stop_sequences: 停止词列表（可选）

        返回：
            生成的文本字符串（已剥离 <think> 推理块）

        异常：
            RuntimeError：API 调用失败时抛出，附带原始错误信息
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop_sequences or None,
            )
        except openai.OpenAIError as e:
            raise RuntimeError(f"LLM API 调用失败：{e}") from e

        self.total_tokens  += response.usage.total_tokens
        self.request_count += 1

        text = response.choices[0].message.content or ""

        # 推理模型兼容（DeepSeek-R1、MiniMax M2.5 等输出 <think> 块的模型）
        # 优先取思考块之后的可见输出；若可见输出为空则回退到全文
        if "<think>" in text:
            visible = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            text = visible if visible else text

        return text

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
