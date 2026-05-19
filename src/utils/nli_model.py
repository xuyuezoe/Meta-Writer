"""
多语言 NLI 模型封装：NLIModel

功能：
    使用 MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 进行自然语言推断（NLI）。
    用于 OnlineValidator._check_dsl_compliance() 中的 CONTRADICTION 检测，
    替代原 LLM-as-judge 路径，实现零 LLM 调用的 DSL 合规性验证。

标签约定（xnli 顺序）：
    logits[0] = CONTRADICTION（前提与假设矛盾）
    logits[1] = NEUTRAL
    logits[2] = ENTAILMENT（前提蕴含假设）

设计原则：
    predict() 用于单对调用（测试/调试），predict_batch() 用于批量推理（减少 kernel 调用开销）。

依赖：transformers、torch
被依赖：OnlineValidator
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NLIModel:
    """
    多语言 NLI 模型（mDeBERTa-v3）

    参数：
        model_name: HuggingFace 模型名称

    关键实现细节：
        xnli 标签顺序：0=CONTRADICTION, 1=NEUTRAL, 2=ENTAILMENT（非标准顺序）。
        批处理默认 batch_size=32，在 GPU 上速度显著优于逐对推理。
    """

    _LABEL_MAP = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}

    def __init__(
        self,
        model_name: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("正在加载 NLI 模型：%s", model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device)
        self._torch = torch
        self.model_name = model_name
        logger.info("NLI 模型加载完成：%s（device=%s）", model_name, self._device)

    def predict(self, premise: str, hypothesis: str) -> str:
        """
        单对 NLI 推断

        参数：
            premise:    前提文本（DSL 条目内容）
            hypothesis: 假设文本（生成内容窗口）

        返回：
            str: "CONTRADICTION" | "NEUTRAL" | "ENTAILMENT"
        """
        results = self.predict_batch([(premise, hypothesis)])
        return results[0]

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[str]:
        """
        批量 NLI 推断

        参数：
            pairs:      (premise, hypothesis) 列表
            batch_size: 每批处理的样本数（默认 32）

        返回：
            List[str]: 每对的标签，顺序与输入一致
        """
        if not pairs:
            return []

        results: List[str] = []
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start: start + batch_size]
            premises    = [p for p, _ in batch]
            hypotheses  = [h for _, h in batch]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self._device)

            with self._torch.no_grad():
                logits = self._model(**inputs).logits  # shape (B, 3)

            # argmax 得到每个样本的标签索引
            label_indices = logits.argmax(dim=-1).cpu().tolist()
            results.extend(self._LABEL_MAP[idx] for idx in label_indices)

        return results
