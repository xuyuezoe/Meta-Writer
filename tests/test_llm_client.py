from __future__ import annotations

import os
import sys
import types
import unittest

sys.modules.setdefault("openai", types.ModuleType("openai"))

from src.utils.llm_client import LLMClient


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


if __name__ == "__main__":
    unittest.main()
