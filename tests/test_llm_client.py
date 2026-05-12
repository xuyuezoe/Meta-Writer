from __future__ import annotations

import os
import sys
import types
import unittest

sys.modules.setdefault("openai", types.ModuleType("openai"))

from src.utils.llm_client import LLMClient


class _DummyOpenAIClient:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class LLMClientProxyEnvTests(unittest.TestCase):
    def test_normalize_proxy_env_vars_rewrites_socks_scheme_for_httpx(self) -> None:
        original_all_proxy = os.environ.get("ALL_PROXY")
        original_http_proxy = os.environ.get("HTTP_PROXY")
        try:
            os.environ["ALL_PROXY"] = "socks://127.0.0.1:7891/"
            os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890/"

            LLMClient._normalize_proxy_env_vars()

            self.assertEqual(os.environ["ALL_PROXY"], "socks5://127.0.0.1:7891/")
            self.assertEqual(os.environ["HTTP_PROXY"], "http://127.0.0.1:7890/")
        finally:
            if original_all_proxy is None:
                os.environ.pop("ALL_PROXY", None)
            else:
                os.environ["ALL_PROXY"] = original_all_proxy

            if original_http_proxy is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = original_http_proxy

    def test_normalize_openai_base_url_adds_v1_suffix_when_missing(self) -> None:
        client = object.__new__(LLMClient)

        self.assertEqual(
            client._normalize_openai_base_url("https://api.minimaxi.com"),
            "https://api.minimaxi.com/v1",
        )

    def test_normalize_openai_base_url_trims_endpoint_suffixes(self) -> None:
        client = object.__new__(LLMClient)

        self.assertEqual(
            client._normalize_openai_base_url("https://api.minimaxi.com/v1/chat/completions"),
            "https://api.minimaxi.com/v1",
        )
        self.assertEqual(
            client._normalize_openai_base_url("https://api.minimaxi.com/v1/messages"),
            "https://api.minimaxi.com/v1",
        )


class LLMClientProtocolSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._openai_module = sys.modules["openai"]
        self._original_openai_ctor = getattr(self._openai_module, "OpenAI", None)
        self._openai_module.OpenAI = _DummyOpenAIClient

    def tearDown(self) -> None:
        if self._original_openai_ctor is None:
            delattr(self._openai_module, "OpenAI")
        else:
            self._openai_module.OpenAI = self._original_openai_ctor

    def test_minimax_model_prefers_openai_chat_even_with_custom_gateway(self) -> None:
        client = LLMClient(
            api_key="test-key",
            model="MiniMax-M2.5",
            base_url="https://api.minimaxi.com/v1",
        )

        self.assertEqual(
            client._candidate_protocols(),
            ["openai_chat", "anthropic_messages"],
        )

    def test_minimaxi_gateway_prefers_openai_chat_even_for_non_openai_style_model_name(self) -> None:
        client = LLMClient(
            api_key="test-key",
            model="some-custom-model",
            base_url="https://api.minimaxi.com/v1",
        )

        self.assertEqual(
            client._candidate_protocols(),
            ["openai_chat", "anthropic_messages"],
        )


if __name__ == "__main__":
    unittest.main()
