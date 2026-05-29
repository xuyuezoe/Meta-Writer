"""
运行上下文：RunContext

功能：
    把单次实验运行的四元组 (task_id, method, model, run_id) 收敛为一个对象，
    统一负责"文件命名"与"provenance（运行溯源）"两件事。

设计动机（第一性原理）：
    一次实验运行的身份由四个维度唯一确定：
        - 跑哪个任务（task_id，如 med_s001）
        - 用哪种方法（method，如 full / no_dsl / autosurvey）
        - 在哪个底座上（model，如 minimax / gpt-4o）
        - 第几次重复（run_id，如 r1 / r2 / r3，用于均值±标准差）
    所有产物文件名、所有结果聚合的分组键，都应从这一身份派生，
    而不应散落在各处临时拼接。集中到 RunContext 可保证命名规范全局一致，
    且每个产物都能反向解析回它的实验身份。

命名规范（对应 ablation_study_plan.md §9.3，并修正其分隔符歧义）：
    {task_id}__{method}__{model}__{run_id}
    例：med_s017__full__minimax__r1 / med_s001__no_dsl__gpt-4o__r2

    关键修正：方案原文用单下划线 "_" 作分隔符，但 task_id（med_s017）与
    method（no_dsl）内部本身就含单下划线，单下划线分隔无法被无歧义解析。
    故改用双下划线 "__" 作字段分隔符，单下划线保留为分量内部合法字符。
    这样既保持可读性，又保证 run_prefix 可被确定性反解析。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

# 四元组字段之间的分隔符（双下划线，分量内部不得出现）
FIELD_SEPARATOR = "__"


class RunContextError(ValueError):
    """运行上下文非法异常（如四元组字段含非法字符）"""


# 文件名安全字符集：字母、数字、单下划线、连字符、点。
# 但禁止出现双下划线（它是字段分隔符），以保证 run_prefix 可被无歧义解析。
_SAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9._\-]+$")


def _validate_token(field_name: str, value: str) -> str:
    """
    校验单个命名分量是否安全

    参数：
        field_name: 分量名称（用于报错定位）
        value:      分量取值

    返回值：
        str：去除首尾空白后的合法取值

    异常：
        RunContextError：当取值为空或含非法字符时抛出。
    """
    stripped = value.strip()
    if not stripped:
        raise RunContextError(f"[运行上下文非法] 字段 '{field_name}' 不能为空")
    if FIELD_SEPARATOR in stripped:
        raise RunContextError(
            f"[运行上下文非法] 字段 '{field_name}' 不得含字段分隔符 "
            f"'{FIELD_SEPARATOR}': '{value}'"
        )
    if not _SAFE_TOKEN_PATTERN.match(stripped):
        raise RunContextError(
            f"[运行上下文非法] 字段 '{field_name}' 含非法字符: '{value}'。"
            f"仅允许字母、数字、单下划线、'-'、'.'"
        )
    return stripped


@dataclass(frozen=True)
class RunContext:
    """
    单次运行的身份上下文

    功能：
        由四元组唯一确定一次运行，派生文件名前缀、产物目录与 provenance。

    参数：
        task_id:  任务 ID（如 "med_s001"）
        method:   方法标识（如 "full"、"no_dsl"、"autosurvey"）
        model:    骨干模型别名（如 "minimax"、"gpt-4o"）
        run_id:   重复运行编号（如 "r1"、"r2"、"r3"）
        root_dir: 实验产物根目录（默认 experiments_out/runs）

    关键实现细节：
        - 下划线 "_" 是命名分隔符，故四个分量内部禁止出现下划线（构造时校验）。
        - run_prefix 是所有产物文件名的统一前缀，bundle_dir 是该运行的独立目录。
    """

    task_id: str
    method: str
    model: str
    run_id: str
    root_dir: str = "./experiments_out/runs"

    def __post_init__(self) -> None:
        """构造后立即校验四个命名分量的合法性（fail-fast）"""
        # frozen dataclass 中需用 object.__setattr__ 写回规范化后的值
        object.__setattr__(self, "task_id", _validate_token("task_id", self.task_id))
        object.__setattr__(self, "method", _validate_token("method", self.method))
        object.__setattr__(self, "model", _validate_token("model", self.model))
        object.__setattr__(self, "run_id", _validate_token("run_id", self.run_id))

    @property
    def run_prefix(self) -> str:
        """
        运行文件名前缀

        返回值：
            str：形如 "med_s017__full__minimax__r1" 的稳定前缀
        """
        return FIELD_SEPARATOR.join(
            (self.task_id, self.method, self.model, self.run_id)
        )

    @property
    def bundle_dir(self) -> Path:
        """
        本次运行的独立产物目录

        返回值：
            Path：root_dir / run_prefix
        """
        return Path(self.root_dir) / self.run_prefix

    def artifact_path(self, artifact_name: str) -> Path:
        """
        派生某类产物的文件路径

        参数：
            artifact_name: 产物文件名（如 "summary.json"、"text.txt"）

        返回值：
            Path：bundle_dir / artifact_name
        """
        return self.bundle_dir / artifact_name

    def provenance(self, *, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
        """
        生成运行溯源字典

        功能：
            把运行身份 + 时间戳 + 可选附加信息（如消融配置、骨干 provenance）
            组装为可落盘的溯源记录。

        参数：
            extra: 附加溯源信息（如 AblationConfig.as_dict()、骨干 as_provenance()）

        返回值：
            Dict：完整的运行溯源记录
        """
        record: Dict[str, object] = {
            "task_id": self.task_id,
            "method": self.method,
            "model": self.model,
            "run_id": self.run_id,
            "run_prefix": self.run_prefix,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if extra:
            record["details"] = dict(extra)
        return record

    @classmethod
    def parse_prefix(cls, run_prefix: str, root_dir: str = "./experiments_out/runs") -> "RunContext":
        """
        从文件名前缀反解析出 RunContext

        功能：
            供分析层从产物目录名还原运行身份，是命名规范的逆操作。

        参数：
            run_prefix: 形如 "med_s017__full__minimax__r1" 的前缀
            root_dir:   产物根目录

        返回值：
            RunContext：解析出的运行上下文

        异常：
            RunContextError：当前缀分量数不等于 4 时抛出。

        关键实现细节：
            按双下划线分隔符切分。分量内部不含双下划线（由构造校验保证），
            故切分无歧义，得到 [task_id, method, model, run_id]。
        """
        parts = run_prefix.split(FIELD_SEPARATOR)
        if len(parts) != 4:
            raise RunContextError(
                f"[运行上下文非法] 无法从前缀解析四元组: '{run_prefix}'。"
                f"期望形如 'task__method__model__run' 的四段结构。"
            )
        task_id, method, model, run_id = parts
        return cls(
            task_id=task_id,
            method=method,
            model=model,
            run_id=run_id,
            root_dir=root_dir,
        )
