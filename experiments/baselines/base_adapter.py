"""
对比系统适配器基类：BaselineAdapter

功能：
    定义所有对比系统必须遵守的统一 I/O 契约：
        输入 BaselineTask（规范化任务）→ 输出 BaselineResult（文本 + 运行统计）。

设计动机（第一性原理）：
    对比实验要求"输入相同、评估相同，只有系统不同"。把这一契约固化为抽象基类，
    使新增对比系统只需实现 run()，而执行层（runners）与分析层（analysis）
    无需关心系统差异。
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


class BaselineAdapterError(RuntimeError):
    """
    对比系统适配器异常

    设计目的：
        把"环境未配置""子进程失败""输出缺失/为空"等错误显式抛出，
        避免对比系统静默产出空文本而污染对比结论。
    """


@dataclass(frozen=True)
class BaselineTask:
    """
    规范化对比任务（与具体系统无关）

    功能：
        把 MetaBench 任务统一抽象为对比系统的输入，剥离 MetaWriter 内部概念。

    参数：
        task_id:          任务 ID（如 "med_s001"）
        topic:            综述主题
        task_description: 完整任务描述（自然语言 prompt）
        constraints:      约束关键词列表
        outline:          章节大纲 {section_id: title}
        corpus_dir:       检索语料库目录（所有系统共享同一语料）
        target_words:     目标总词数
        reference:        MetaBench 评估所需的 reference（透传给 evaluate_meta_bench）
    """

    task_id: str
    topic: str
    task_description: str
    constraints: List[str]
    outline: Dict[str, str]
    corpus_dir: str
    target_words: int
    reference: Dict[str, Any]

    def to_bridge_payload(self) -> Dict[str, Any]:
        """
        序列化为外部系统桥接脚本的标准化输入

        功能：
            产出一个与具体系统无关的标准 input.json 负载，
            外部系统的桥接脚本据此读取任务并驱动其自身管线。

        返回值：
            Dict：标准化输入负载（不含 reference，避免评估信息泄漏给生成系统）。

        关键实现细节：
            刻意不包含 reference（评估真值），保证对比系统在"信息对称"前提下
            只看到任务输入，杜绝评估泄漏导致的不公平。
        """
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "task_description": self.task_description,
            "constraints": list(self.constraints),
            "outline": dict(self.outline),
            "corpus_dir": self.corpus_dir,
            "target_words": self.target_words,
        }

    @classmethod
    def from_meta_bench_config(
        cls,
        task_id: str,
        config: Mapping[str, Any],
    ) -> "BaselineTask":
        """
        从 MetaBench 任务配置构造规范化任务

        参数：
            task_id: 任务 ID
            config:  meta_bench 任务配置（含 task/constraints/outline/reference/corpus_dir）

        返回值：
            BaselineTask 实例

        异常：
            BaselineAdapterError：当配置缺少必要字段时抛出。
        """
        for field_name in ("task", "constraints", "outline"):
            if field_name not in config:
                raise BaselineAdapterError(
                    f"[对比任务构造失败] 任务 '{task_id}' 配置缺少字段: {field_name}"
                )

        reference = config.get("reference")
        reference_dict: Dict[str, Any] = dict(reference) if isinstance(reference, Mapping) else {}

        topic = ""
        if isinstance(reference, Mapping):
            raw_topic = reference.get("topic")
            if isinstance(raw_topic, str):
                topic = raw_topic

        target_words = 0
        if isinstance(reference, Mapping):
            # 词数字段可能在 reference 顶层或嵌套于 reference["constraints"]
            sources: List[Mapping[str, Any]] = [reference]
            nested_constraints = reference.get("constraints")
            if isinstance(nested_constraints, Mapping):
                sources.append(nested_constraints)
            # 取值优先 body 长度：MetaBench 的 length_score 按"正文词数（不含参考列表）"
            # 评分，对应 required_length_words / body_target_words；而 total_target_words
            # 含参考列表，约为 body 的两倍。若优先取 total，会让基线 prompt 的目标长度
            # （如 6083）与任务描述及评测口径（如 2916）相互矛盾，系统性诱导基线超长、
            # 不公平地压低其 length_score。故优先取 body 口径，total 仅作兜底。
            for source in sources:
                for key in ("body_target_words", "required_length_words", "total_target_words"):
                    raw_value = source.get(key)
                    if isinstance(raw_value, (int, float)) and int(raw_value) > 0:
                        target_words = int(raw_value)
                        break
                if target_words > 0:
                    break

        corpus_dir_value = config.get("corpus_dir")
        corpus_dir = (
            str(corpus_dir_value)
            if isinstance(corpus_dir_value, str) and corpus_dir_value.strip()
            else "./data_sample/med_papers_review_augmented_strict"
        )

        constraints_value = config["constraints"]
        outline_value = config["outline"]
        return cls(
            task_id=task_id,
            topic=topic,
            task_description=str(config["task"]),
            constraints=[str(c) for c in constraints_value] if isinstance(constraints_value, list) else [],
            outline={str(k): str(v) for k, v in outline_value.items()} if isinstance(outline_value, dict) else {},
            corpus_dir=corpus_dir,
            target_words=target_words,
            reference=reference_dict,
        )


@dataclass
class BaselineResult:
    """
    对比系统运行结果（统一输出）

    功能：
        承载对比系统产出的最终文本与运行统计，供 runners 落盘、analysis 聚合。

    参数：
        system:              系统标识（如 "autosurvey"、"direct-llm"）
        final_text:          生成的最终综述文本
        total_tokens:        LLM token 消耗（外部系统无法统计时为 None）
        request_count:       LLM 请求次数（无法统计时为 None）
        wall_time_seconds:   端到端墙钟耗时
        status:              运行状态（"completed" / "failed"）
        extra:               系统专属的附加信息（原始产物路径、版本号等）
    """

    system: str
    final_text: str
    total_tokens: Optional[int] = None
    request_count: Optional[int] = None
    wall_time_seconds: float = 0.0
    status: str = "completed"
    extra: Dict[str, Any] = field(default_factory=dict)

    def llm_stats(self) -> Dict[str, Optional[int]]:
        """返回与 MetaWriter summary 对齐的 LLM 统计字典"""
        return {
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


class BaselineAdapter(abc.ABC):
    """
    对比系统适配器抽象基类

    功能：
        统一对比系统的运行入口。子类实现 run()，把 BaselineTask 转化为
        BaselineResult。

    参数：
        system_name: 系统标识（用于命名与日志），由子类提供。
    """

    #: 系统标识（子类覆盖）
    system_name: str = "baseline"

    @abc.abstractmethod
    def run(self, task: BaselineTask, *, work_dir: str) -> BaselineResult:
        """
        运行对比系统并返回结果

        参数：
            task:     规范化对比任务
            work_dir: 该次运行的工作目录（存放 input.json / output.txt 等中间产物）

        返回值：
            BaselineResult：统一的运行结果

        异常：
            BaselineAdapterError：运行失败时抛出，附带可诊断信息。
        """
        raise NotImplementedError
