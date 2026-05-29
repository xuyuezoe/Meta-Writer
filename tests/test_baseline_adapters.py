"""
对比系统适配器单元测试（L2 baselines）

验证目标：
    1. BaselineTask 从 MetaBench 配置正确构造，且 to_bridge_payload 不泄漏 reference
    2. Direct-LLM 进程内单次生成返回文本与统计；空输出显式抛错
    3. SubprocessBaselineAdapter 端到端文件契约可用（用 fake bridge 真实跑子进程）
    4. 配置缺失 / 不完整 / 子进程失败 显式抛错（禁止静默空文本）
    5. registry 未知系统名、direct-llm 缺 client 显式抛错
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from experiments.baselines.base_adapter import (
    BaselineAdapterError,
    BaselineTask,
)
from experiments.baselines.direct_llm import DirectLLMAdapter
from experiments.baselines.registry import get_baseline_adapter
from experiments.baselines.subprocess_adapter import SubprocessBaselineAdapter


# ── 测试夹具 ──────────────────────────────────────────────────────────

def _sample_task() -> BaselineTask:
    return BaselineTask(
        task_id="med_s001",
        topic="acute coronary syndrome",
        task_description="Write a review about acute coronary syndrome.",
        constraints=["pathophysiology", "hemodynamics"],
        outline={"s1": "Introduction", "s2": "Methods"},
        corpus_dir="./data_sample/med_papers_review_augmented_strict",
        target_words=4200,
        reference={"task_id": "med_s001", "total_target_words": 4200, "secret": "answer"},
    )


class _FakeLLM:
    """假 LLM 客户端：记录调用并返回固定文本"""

    def __init__(self, text: str) -> None:
        self._text = text
        self.last_prompt = ""
        self.last_max_tokens = 0

    def generate(self, *, prompt: str, temperature: float, max_tokens: int, log_meta=None) -> str:
        self.last_prompt = prompt
        self.last_max_tokens = max_tokens
        return self._text

    def get_statistics(self):
        return {"total_tokens": 1234, "request_count": 1}


# ── BaselineTask ──────────────────────────────────────────────────────

def test_from_meta_bench_config_builds_task() -> None:
    config = {
        "task": "Write a review.",
        "constraints": ["a", "b"],
        "outline": {"s1": "Intro"},
        "corpus_dir": "./data_sample/med_papers_review_augmented_strict",
        "reference": {"topic": "x", "total_target_words": 6000, "task_id": "med_s002"},
    }
    task = BaselineTask.from_meta_bench_config("med_s002", config)
    assert task.topic == "x"
    assert task.target_words == 6000
    assert task.constraints == ["a", "b"]


def test_from_meta_bench_config_nested_word_target() -> None:
    """词数字段嵌套在 reference['constraints'] 下时也应正确提取（贴合真实 MetaBench 结构）"""
    config = {
        "task": "Write a review.",
        "constraints": ["a"],
        "outline": {"s1": "Intro"},
        "reference": {
            "topic": "acs",
            "constraints": {"total_target_words": 4200, "body_target_words": 4200},
        },
    }
    task = BaselineTask.from_meta_bench_config("med_s001", config)
    assert task.target_words == 4200
    assert task.topic == "acs"


def test_from_meta_bench_config_missing_field_raises() -> None:
    with pytest.raises(BaselineAdapterError):
        BaselineTask.from_meta_bench_config("med_s002", {"task": "x", "constraints": []})


def test_bridge_payload_excludes_reference() -> None:
    """桥接负载不得包含评估真值（防泄漏）"""
    payload = _sample_task().to_bridge_payload()
    assert "reference" not in payload
    assert "secret" not in json.dumps(payload)
    assert payload["task_id"] == "med_s001"
    assert payload["target_words"] == 4200


# ── Direct-LLM ────────────────────────────────────────────────────────

def test_direct_llm_generates(tmp_path: Path) -> None:
    client = _FakeLLM("This is a generated review about ACS.")
    adapter = DirectLLMAdapter(client)
    result = adapter.run(_sample_task(), work_dir=str(tmp_path))

    assert result.system == "direct-llm"
    assert "generated review" in result.final_text
    assert result.total_tokens == 1234
    assert result.request_count == 1
    # prompt 应包含约束与大纲
    assert "pathophysiology" in client.last_prompt
    assert "Introduction" in client.last_prompt


def test_direct_llm_empty_output_raises(tmp_path: Path) -> None:
    adapter = DirectLLMAdapter(_FakeLLM("   "))
    with pytest.raises(BaselineAdapterError):
        adapter.run(_sample_task(), work_dir=str(tmp_path))


# ── Subprocess 端到端（fake bridge）──────────────────────────────────

_FAKE_BRIDGE = '''\
import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--stats", required=False)
a = p.parse_args()
task = json.loads(Path(a.input).read_text(encoding="utf-8"))
text = "REVIEW for topic: " + task["topic"] + " (" + str(task["target_words"]) + " words target)"
Path(a.output).write_text(text, encoding="utf-8")
if a.stats:
    Path(a.stats).write_text(json.dumps({"total_tokens": 999, "request_count": 7}), encoding="utf-8")
'''


def _make_configured_envs(tmp_path: Path, *, bridge_body: str, system: str = "faketest") -> Path:
    """在临时 envs 目录写出 fake bridge 与已填好的系统配置"""
    envs_dir = tmp_path / "envs"
    envs_dir.mkdir()
    bridge = tmp_path / "bridge.py"
    bridge.write_text(bridge_body, encoding="utf-8")
    config = {
        "system": system,
        "repo_url": "https://example.com/repo",
        "repo_path": str(tmp_path),
        "python_executable": sys.executable,
        "bridge_script": str(bridge),
        "extra_args": [],
        "timeout_seconds": 60,
    }
    (envs_dir / f"{system}.json").write_text(json.dumps(config), encoding="utf-8")
    return envs_dir


def test_subprocess_adapter_end_to_end(tmp_path: Path) -> None:
    """通过 fake bridge 验证完整文件契约（写 input、跑子进程、读 output/stats）"""
    envs_dir = _make_configured_envs(tmp_path, bridge_body=_FAKE_BRIDGE)
    adapter = SubprocessBaselineAdapter(
        system_name="faketest",
        repo_url="https://example.com/repo",
        envs_dir=envs_dir,
    )
    work_dir = tmp_path / "work"
    result = adapter.run(_sample_task(), work_dir=str(work_dir))

    assert "REVIEW for topic: acute coronary syndrome" in result.final_text
    assert result.total_tokens == 999
    assert result.request_count == 7
    # input.json 应已写出且不含 reference
    written = json.loads((work_dir / "input.json").read_text(encoding="utf-8"))
    assert "reference" not in written


def test_subprocess_missing_config_raises(tmp_path: Path) -> None:
    """配置文件缺失应显式抛错"""
    empty_envs = tmp_path / "envs"
    empty_envs.mkdir()
    adapter = SubprocessBaselineAdapter("autosurvey", "url", envs_dir=empty_envs)
    with pytest.raises(BaselineAdapterError) as exc:
        adapter.run(_sample_task(), work_dir=str(tmp_path / "w"))
    assert "未配置" in str(exc.value) or "缺少配置" in str(exc.value)


def test_subprocess_incomplete_config_raises(tmp_path: Path) -> None:
    """配置缺必填项（空 repo_path）应显式抛错"""
    envs_dir = tmp_path / "envs"
    envs_dir.mkdir()
    (envs_dir / "autosurvey.json").write_text(
        json.dumps({"repo_path": "", "python_executable": "", "bridge_script": ""}),
        encoding="utf-8",
    )
    adapter = SubprocessBaselineAdapter("autosurvey", "url", envs_dir=envs_dir)
    with pytest.raises(BaselineAdapterError):
        adapter.run(_sample_task(), work_dir=str(tmp_path / "w"))


def test_subprocess_nonzero_exit_raises(tmp_path: Path) -> None:
    """子进程非零退出（bridge 抛异常）应显式抛错"""
    crashing_bridge = "import sys\nsys.exit(3)\n"
    envs_dir = _make_configured_envs(tmp_path, bridge_body=crashing_bridge)
    adapter = SubprocessBaselineAdapter("faketest", "url", envs_dir=envs_dir)
    with pytest.raises(BaselineAdapterError) as exc:
        adapter.run(_sample_task(), work_dir=str(tmp_path / "work"))
    assert "非零退出" in str(exc.value)


# ── registry ──────────────────────────────────────────────────────────

def test_registry_direct_llm_requires_client() -> None:
    with pytest.raises(BaselineAdapterError):
        get_baseline_adapter("direct-llm")


def test_registry_unknown_system_raises() -> None:
    with pytest.raises(BaselineAdapterError):
        get_baseline_adapter("not-a-system")


def test_registry_builds_external_adapters() -> None:
    for name in ("autosurvey", "lira", "surveyforge", "paperorchestra"):
        adapter = get_baseline_adapter(name)
        assert adapter.system_name == name
