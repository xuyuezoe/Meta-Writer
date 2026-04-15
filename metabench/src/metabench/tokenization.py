"""MetaBench tokenizer 计数模块。

该模块统一 token 统计口径，优先使用 tokenizer 计数，
并允许同时记录接口 usage 字段用于对账。
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Optional


@dataclass
class TokenizerConfig:
    """Tokenizer 配置。

    参数:
        backend: 后端类型，取值为 transformers 或 tiktoken。
        tokenizer_name: tokenizer 名称或编码名称。
    """

    backend: str
    tokenizer_name: str


def count_tokens_with_transformers(tokenizer_name: str, text: str) -> int:
    """使用 transformers 统计 token 数。

    参数:
        tokenizer_name: tokenizer 名称。
        text: 输入文本。

    返回:
        token 数。
    """

    try:
        transformers_module = importlib.import_module("transformers")
    except ImportError as import_error:
        raise ImportError("未安装 transformers，请安装 optional 依赖 tokenizer") from import_error

    auto_tokenizer = transformers_module.AutoTokenizer
    tokenizer = auto_tokenizer.from_pretrained(tokenizer_name, use_fast=True, trust_remote_code=False)
    encoded_ids = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded_ids)


def count_tokens_with_tiktoken(encoding_name: str, text: str) -> int:
    """使用 tiktoken 统计 token 数。

    参数:
        encoding_name: tiktoken 编码名称。
        text: 输入文本。

    返回:
        token 数。
    """

    try:
        tiktoken_module = importlib.import_module("tiktoken")
    except ImportError as import_error:
        raise ImportError("未安装 tiktoken，请安装 optional 依赖 tokenizer") from import_error

    encoding = tiktoken_module.get_encoding(encoding_name)
    token_ids = encoding.encode(text)
    return len(token_ids)


def count_tokens(tokenizer_config: TokenizerConfig, text: str) -> int:
    """统一 token 统计入口。

    参数:
        tokenizer_config: tokenizer 配置。
        text: 待统计文本。

    返回:
        token 数。
    """

    if tokenizer_config.backend == "transformers":
        return count_tokens_with_transformers(tokenizer_config.tokenizer_name, text)
    if tokenizer_config.backend == "tiktoken":
        return count_tokens_with_tiktoken(tokenizer_config.tokenizer_name, text)
    raise ValueError(f"不支持的 tokenizer backend: {tokenizer_config.backend}")


def resolve_usage_total_tokens(raw_usage_total_tokens: Optional[int]) -> Optional[int]:
    """规范化 usage.total_tokens。

    参数:
        raw_usage_total_tokens: 原始 usage total_tokens。

    返回:
        规范化后的 total_tokens；若值不合法则返回 None。
    """

    if raw_usage_total_tokens is None:
        return None
    if raw_usage_total_tokens < 0:
        return None
    return raw_usage_total_tokens
