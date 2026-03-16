"""
运行日志器：RunLogger

功能：
    为每次 MetaWriter 运行生成结构化可读的文本日志文件，
    记录每节生成的完整过程：规划、DSL注入、每次尝试的 prompt/响应/验证/诊断/修复。

输出文件：
    outputs/{session_name}_run.log

格式：
    分节块（SECTION 分隔线）+ 分尝试块（ATTEMPT 分隔线），
    每块有清晰分隔线和层级缩进。

关键实现细节：
    文件句柄在初始化时打开，每次写入立即 flush，
    防止系统崩溃时丢失日志。Prompt 和 LLM 原始响应完整记录，不截断。
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from ..core.decision import Decision
    from ..core.meta_state import MetaState
    from ..core.plan import SectionIntent
    from ..core.state import GenerationState
    from ..core.validation import ValidationReport


class RunLogger:
    """
    运行日志器

    功能：
        以结构化可读格式记录 MetaWriter 完整运行过程。
        包含每节每次尝试的 prompt、LLM 原始响应、验证四层逐层结果、
        MRSD 诊断五步过程、MetaState 门控决策和修复动作。

    参数：
        output_dir: 输出目录路径
        session_name: 会话名称（用于文件命名）

    关键实现细节：
        每次调用立即写入并 flush（不缓存），确保崩溃不丢失日志。
        日志格式使用分隔线区分节块和尝试块，使用两空格缩进表示层级。
    """

    _RUN_SEPARATOR     = "=" * 80
    _SECTION_SEPARATOR = "#" * 80
    _ATTEMPT_PREFIX    = "─"

    def __init__(self, output_dir: str, session_name: str):
        """
        初始化运行日志器，打开文件句柄

        参数：
            output_dir: 输出目录（若不存在则自动创建）
            session_name: 会话名称，文件命名为 {session_name}_run.log
        """
        log_path = Path(output_dir) / f"{session_name}_run.log"
        if log_path.exists():
            log_path.unlink()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8")
        self._log_path = log_path

    def close(self) -> None:
        """
        关闭文件句柄

        功能：
            flush 后关闭，确保所有内容落盘。
        """
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()

    # ------------------------------------------------------------------
    # 内部写入工具
    # ------------------------------------------------------------------

    def _write(self, text: str) -> None:
        """
        立即写入一行（带换行和 flush）

        参数：
            text: 写入内容（不含末尾换行）
        """
        self._file.write(text + "\n")
        self._file.flush()

    def _write_block(self, lines: List[str], indent: str = "  ") -> None:
        """
        写入多行文本块，每行添加缩进

        参数：
            lines: 行列表
            indent: 缩进字符串（默认两空格）
        """
        for line in lines:
            self._write(indent + line)

    # ------------------------------------------------------------------
    # 运行级别日志
    # ------------------------------------------------------------------

    def log_run_start(
        self,
        task: str,
        constraints: List[str],
        outline: Dict[str, str],
    ) -> None:
        """
        记录运行开始（文件头部）

        参数：
            task: 任务描述
            constraints: 全局约束列表
            outline: 章节大纲 {section_id: title}
        """
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(self._RUN_SEPARATOR)
        self._write(f"RUN  任务：[{task[:60]}] | 章节数：{len(outline)} | 时间：{now}")
        self._write(self._RUN_SEPARATOR)
        self._write("")
        self._write("[全局约束]")
        if constraints:
            for c in constraints:
                self._write(f"  · {c}")
        else:
            self._write("  （无约束）")
        self._write("")
        self._write("[大纲]")
        for sid, title in outline.items():
            self._write(f"  [{sid}] {title}")
        self._write("")

    def log_run_summary(
        self,
        correction_stats: Dict[str, Any],
        meta_state: "MetaState",
    ) -> None:
        """
        记录运行汇总（文件末尾）

        参数：
            correction_stats: CorrectionLog.get_statistics() 的返回值
            meta_state: MetaState 对象（读取信任度和污染风险）
        """
        self._write("")
        self._write(self._RUN_SEPARATOR)
        self._write("RUN SUMMARY")
        self._write(self._RUN_SEPARATOR)
        self._write(f"  总节数:          {correction_stats.get('total_sections', 0)}")
        self._write(
            f"  首次成功:        {correction_stats.get('success_first_try', 0)} "
            f"({correction_stats.get('success_rate_first_try', 0):.1%})"
        )
        self._write(f"  总重试次数:      {correction_stats.get('total_retries', 0)}")
        self._write(f"  总回退次数:      {correction_stats.get('total_rollbacks', 0)}")
        self._write(f"  彻底失败节数:    {correction_stats.get('total_failures', 0)}")
        self._write(f"  平均尝试次数:    {correction_stats.get('avg_attempts', 0.0):.2f}")
        self._write(f"  DSL 信任度:      {meta_state.memory_trust_level:.3f}")
        self._write(f"  污染风险:        {meta_state.contamination_risk_score:.3f}")
        self._write("")

    # ------------------------------------------------------------------
    # 节级别日志
    # ------------------------------------------------------------------

    def log_section_start(
        self,
        section_id: str,
        title: str,
        idx: int,
        total: int,
        state: "GenerationState",
        dsl_active_count: int,
    ) -> None:
        """
        记录节开始（SECTION 块头）

        参数：
            section_id: 节 ID
            title: 节标题
            idx: 当前节序号（0-based）
            total: 总节数
            state: 当前生成状态
            dsl_active_count: DSL 活跃条目数
        """
        progress_pct = int(state.progress * 100)
        completed = len(state.generated_sections)
        self._write("")
        self._write(self._SECTION_SEPARATOR)
        self._write(f'SECTION [{section_id}] "{title}"  ({idx + 1}/{total})')
        self._write(self._SECTION_SEPARATOR)
        self._write("")
        self._write("[STATE]")
        self._write(
            f"  进度：{progress_pct}%（{completed}/{total} 节已完成）  "
            f"全局约束：{len(state.global_constraints)} 条  "
            f"DSL 活跃条目：{dsl_active_count} 条"
        )
        self._write("")

    def log_planning(self, section_id: str, intent: "SectionIntent") -> None:
        """
        记录规划结果（SectionIntent 内容）

        参数：
            section_id: 节 ID
            intent: SectionPlanner 生成的 SectionIntent
        """
        self._write("[PLAN] SectionIntent")
        self._write(f"  局部目标：{intent.local_goal}")
        self._write(
            f"  开放线索推进：{intent.open_loops_to_advance if intent.open_loops_to_advance else '[]'}"
        )
        self._write(
            f"  待维护承诺：{intent.commitments_to_maintain if intent.commitments_to_maintain else '[]'}"
        )
        self._write(
            f"  风险规避：{intent.risks_to_avoid if intent.risks_to_avoid else '[]'}"
        )
        self._write(
            f"  成功标准：{intent.success_criteria if intent.success_criteria else '[]'}"
        )
        self._write(f"  DSL 信任度：{intent.dsl_trust_at_generation:.3f}")
        self._write("")

    def log_dsl_injection(self, section_id: str, entries: List[Any]) -> None:
        """
        记录 DSL 注入条目详情

        参数：
            section_id: 节 ID
            entries: 可注入的 LedgerEntry 列表（来自 DiscourseLedger.get_injectable_entries）
        """
        self._write(f"[DSL 注入]  {len(entries)} 条")
        if entries:
            for e in entries:
                ct  = e.commitment_type.value  if hasattr(e, "commitment_type")  else "?"
                cst = e.constraint_type.value  if hasattr(e, "constraint_type")  else "?"
                content = e.content            if hasattr(e, "content")          else str(e)
                self._write(f"  * [{ct}/{cst}]  {content}")
        else:
            self._write("  （无条目）")
        self._write("")

    def log_section_success(
        self,
        section_id: str,
        total_attempts: int,
        dcas: float,
        new_entries: List[Any],
        total_active_entries: int,
        memory_trust: float,
    ) -> None:
        """
        记录节生成成功及新增 DSL 条目

        参数：
            section_id: 节 ID
            total_attempts: 本节实际总尝试次数（1-based）
            dcas: 最终 DCAS 分数
            new_entries: 本节新提取的 LedgerEntry 列表
            total_active_entries: 提取后 DSL 总活跃条目数
            memory_trust: 当前记忆信任度
        """
        self._write(f"[SUCCESS] ✓  DCAS={dcas:.3f}  尝试次数={total_attempts}")
        self._write(f"  新增 DSL 条目：{len(new_entries)} 条")
        for e in new_entries:
            ct  = e.commitment_type.value  if hasattr(e, "commitment_type")  else "?"
            cst = e.constraint_type.value  if hasattr(e, "constraint_type")  else "?"
            content = e.content            if hasattr(e, "content")          else str(e)
            self._write(f"    * [{ct}/{cst}]  {content}")
        self._write(f"  DSL 总活跃条目：{total_active_entries}  信任度：{memory_trust:.3f}")
        self._write("")

    def log_section_degraded(self, section_id: str, total_attempts: int) -> None:
        """
        记录节降级（超过最大重试次数，以最后一次内容继续）

        参数：
            section_id: 节 ID
            total_attempts: 本节总尝试次数
        """
        self._write(
            f"[DEGRADED] ✗  节 {section_id} 超过最大重试次数（{total_attempts} 次），以降级内容继续"
        )
        self._write("")

    # ------------------------------------------------------------------
    # 尝试级别日志
    # ------------------------------------------------------------------

    def log_attempt_start(
        self,
        section_id: str,
        attempt: int,
        temperature: float,
    ) -> None:
        """
        记录单次尝试开始（ATTEMPT 分隔线）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based，最大 MAX_RETRIES_PER_SECTION）
            temperature: 本次尝试的生成温度
        """
        label = f" ATTEMPT {attempt}/3  temp={temperature:.2f} "
        total_len = 80
        label_len = len(label)
        left_len  = (total_len - label_len) // 2
        right_len = total_len - left_len - label_len
        line = self._ATTEMPT_PREFIX * left_len + label + self._ATTEMPT_PREFIX * right_len
        self._write(line)

    def log_prompt(self, section_id: str, attempt: int, prompt_text: str) -> None:
        """
        记录完整 prompt（不截断）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            prompt_text: 完整 prompt 文本
        """
        self._write("[PROMPT]")
        for line in prompt_text.split("\n"):
            self._write("  " + line)
        self._write("[/PROMPT]")
        self._write("")

    def log_llm_raw_response(
        self,
        section_id: str,
        attempt: int,
        raw_text: str,
    ) -> None:
        """
        记录 LLM 原始响应（不截断）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            raw_text: LLM 返回的原始文本
        """
        self._write("[LLM 原始响应]")
        for line in raw_text.split("\n"):
            self._write("  " + line)
        self._write("[/LLM 原始响应]")
        self._write("")

    def log_parsed_decision(
        self,
        section_id: str,
        attempt: int,
        content: str,
        decision: "Decision",
    ) -> None:
        """
        记录解析后的决策对象和生成内容（不截断）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            content: 清洗后的生成内容
            decision: 解析完成的 Decision 对象
        """
        self._write("[PARSED DECISION]")
        self._write(f"  决策：{decision.decision}")
        self._write(f"  推理：{decision.reasoning[:300]}")
        self._write(f"  预期效果：{decision.expected_effect}")
        self._write(f"  置信度：{decision.confidence:.2f}")
        refs = [r[0] for r in decision.referenced_sections] if decision.referenced_sections else []
        self._write(f"  引用节：{refs}")
        self._write("")
        self._write("[GENERATED CONTENT]")
        for line in content.split("\n"):
            self._write("  " + line)
        self._write("[/GENERATED CONTENT]")
        self._write("")

    # ------------------------------------------------------------------
    # 验证层日志
    # ------------------------------------------------------------------

    def log_validation_start(self, section_id: str, attempt: int) -> None:
        """
        记录验证开始（[VALIDATION] 块头）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
        """
        self._write("[VALIDATION]")

    def log_validation_result(
        self,
        section_id: str,
        attempt: int,
        layer: str,
        passed: bool,
        details: str,
    ) -> None:
        """
        记录单层验证结果

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            layer: 验证层名称（格式检查/约束检查/对齐度(DCAS)/一致性检查）
            passed: 是否通过
            details: 验证细节描述
        """
        status = "PASS" if passed else "FAIL"
        self._write(f"  {layer:<16}: {status}  {details}")

    def log_validation_summary(
        self,
        section_id: str,
        attempt: int,
        report: "ValidationReport",
    ) -> None:
        """
        记录验证汇总结果（VALIDATION 块尾）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            report: 完整的 ValidationReport
        """
        self._write("  " + "─" * 41)
        if report.passed:
            self._write("  总计: PASS  阻断问题 0 个")
        else:
            blocking = report.issues
            self._write(f"  总计: FAIL  阻断问题 {len(blocking)} 个")
            for issue in blocking:
                self._write(f"    ! [{issue.severity.upper()}] {issue.description}")
        self._write("")

    # ------------------------------------------------------------------
    # 诊断与修复日志
    # ------------------------------------------------------------------

    def log_diagnosis(
        self,
        section_id: str,
        attempt: int,
        diagnosis: Any,
    ) -> None:
        """
        记录 MRSD 诊断结果（五步 BCP 输出）

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            diagnosis: DiagnosisResult 对象
        """
        self._write("[MRSD 诊断]")
        self._write(f"  错误层级    : {diagnosis.error_tier.value}")
        self._write(f"  错误来源    : {diagnosis.error_source.value}")
        self._write(f"  修复范围    : {diagnosis.repair_scope}")
        self._write(f"  置信度      : {diagnosis.confidence:.2f}")
        subgraph = diagnosis.causal_subgraph if diagnosis.causal_subgraph else []
        self._write(f"  因果子图    : {subgraph}")
        dc = diagnosis.decoding_config
        self._write(
            f"  解码配置    : strengthen_dsl_injection={dc.strengthen_dsl_injection}, "
            f"temperature→{dc.temperature:.2f}"
        )
        self._write("")

    def log_repair_action(
        self,
        section_id: str,
        attempt: int,
        repair_scope: str,
        details_dict: Dict[str, Any],
    ) -> None:
        """
        记录修复动作执行详情

        参数：
            section_id: 节 ID
            attempt: 尝试次数（1-based）
            repair_scope: 修复范围（local_rewrite / partial_rollback / memory_purge）
            details_dict: 修复细节键值对
        """
        self._write(f"[REPAIR] {repair_scope}")
        for key, val in details_dict.items():
            self._write(f"  {key}: {val}")
        self._write("")

    def log_meta_state_gate(
        self,
        section_id: str,
        action_name: str,
        granted: bool,
        reason: str,
    ) -> None:
        """
        记录 MetaState 门控决策

        参数：
            section_id: 节 ID
            action_name: 门控动作名称（如 allow_rollback / trust_validator_major）
            granted: 是否准许
            reason: 门控原因说明
        """
        status = "GRANTED" if granted else "DENIED"
        self._write(f"[META_STATE GATE] {action_name} → {status}  ({reason})")
        self._write("")
