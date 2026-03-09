"""
MetaWriter v4.0 — 项目入口

用法：
    python main.py

切换任务：
    修改下方 TASK_NAME 变量，然后直接运行即可。

可用任务：
    "scifi_story"         科幻短篇故事（创意写作）
    "argumentative_essay" 议论文——大数据时代的个人隐私保护（非虚构写作）
"""
import os
import sys
import json
import traceback
from pathlib import Path

# ================================================================
# 修改此处选择运行的任务
# ================================================================
TASK_NAME = "argumentative_essay"
# ================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

from src.orchestrator_v2 import SelfCorrectingOrchestrator
from src.utils.llm_client import LLMClient
from examples.tasks import TASK_REGISTRY


def main() -> None:
    """
    主入口函数

    流程：
        第一阶段：环境准备（清理旧文件、加载配置）
        第二阶段：任务加载（从注册表获取任务配置）
        第三阶段：生成（带自我修正的主循环）
        第四阶段：结果保存与统计输出
    """
    # ── 第一阶段：环境准备 ─────────────────────────────────────
    if TASK_NAME not in TASK_REGISTRY:
        print(f"错误：未知任务 '{TASK_NAME}'")
        print(f"可用任务：{list(TASK_REGISTRY.keys())}")
        return

    load_dotenv(override=True)
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("错误：请在 .env 文件中设置 API_KEY")
        return

    model    = os.getenv("MODEL", "MiniMax-M2.5")
    base_url = os.getenv("BASE_URL")

    # ── 第二阶段：任务加载 ─────────────────────────────────────
    config       = TASK_REGISTRY[TASK_NAME]()
    task         = config["task"]
    constraints  = config["constraints"]
    outline      = config["outline"]
    session_name = config["session_name"]

    # 清理本次任务的旧输出文件
    output_dir = Path("./outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_file in output_dir.glob(f"{session_name}_*"):
        old_file.unlink()

    print("=" * 60)
    print(f"MetaWriter  |  任务：{TASK_NAME}")
    print("=" * 60)
    print(f"\n任务描述：{task}")
    print(f"约束数量：{len(constraints)}")
    for c in constraints:
        print(f"  · {c}")
    print(f"\n大纲（{len(outline)} 个 section）：")
    for sid, title in outline.items():
        print(f"  [{sid}] {title}")
    print()

    # ── 第三阶段：生成 ─────────────────────────────────────────
    client       = LLMClient(api_key=api_key, model=model, base_url=base_url)
    orchestrator = SelfCorrectingOrchestrator(
        client, memory_path="./sessions", session_name=session_name
    )

    try:
        final_text, decisions, correction_log = orchestrator.generate_with_self_correction(
            task=task,
            constraints=constraints,
            outline=outline,
        )

        # ── 第四阶段：结果保存与统计输出 ──────────────────────
        print("\n" + "=" * 60)
        print("修正统计")
        print("=" * 60)

        stats = correction_log.get_statistics()
        print(f"  总 section 数:   {stats['total_sections']}")
        print(f"  首次成功:        {stats['success_first_try']} ({stats['success_rate_first_try']:.1%})")
        print(f"  总重试次数:      {stats['total_retries']}")
        print(f"  总回退次数:      {stats['total_rollbacks']}")
        print(f"  彻底失败节数:    {stats['total_failures']}")
        print(f"  平均尝试次数:    {stats['avg_attempts']:.2f}")

        if stats["total_rollbacks"] > 0:
            print(f"  平均回退距离:    {stats['avg_rollback_distance']:.1f} sections")

        if stats["retry_by_action"]:
            print("\n策略使用分布:")
            for action, count in sorted(stats["retry_by_action"].items(), key=lambda x: -x[1]):
                print(f"    {action}: {count} 次")

        dtg_stats = orchestrator.dtg.get_statistics()
        print(f"\nDTG 统计:")
        print(f"  决策总数:        {dtg_stats['total_decisions']}")
        print(f"  平均置信度:      {dtg_stats['avg_confidence']:.3f}")
        print(f"  平均引用数/决策: {dtg_stats['avg_references_per_decision']:.2f}")
        print(f"  累计回退次数:    {dtg_stats['rollback_count']}")

        llm_stats = client.get_statistics()
        print(f"\nLLM 统计:")
        print(f"  总 token 数:     {llm_stats['total_tokens']:,}")
        print(f"  请求次数:        {llm_stats['request_count']}")

        print("\n修正时间线:")
        print(correction_log.visualize_timeline())

        # 保存输出文件（以 session_name 为前缀）
        text_file = output_dir / f"{session_name}_text.txt"
        text_file.write_text(final_text, encoding="utf-8")
        print(f"\n生成文本:     {text_file}")

        log_file = str(output_dir / f"{session_name}_correction_log.json")
        correction_log.save(log_file)
        print(f"修正日志:     {log_file}")

        dtg_file = output_dir / f"{session_name}_dtg.json"
        dtg_file.write_text(
            json.dumps(orchestrator.dtg.export_dtg(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"DTG 图:       {dtg_file}")

        orchestrator.dtg.save_to_disk(session_name)
        print(f"完整 session: ./sessions/{session_name}.json")

        print("\n" + "=" * 60)
        print("生成文本预览（前500字符）")
        print("=" * 60)
        print(final_text[:500] + ("..." if len(final_text) > 500 else ""))

        print("\n" + "=" * 60)
        print("完成")
        print("=" * 60)

    except Exception as e:
        print(f"\n生成失败: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
