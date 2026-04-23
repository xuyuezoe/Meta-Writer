"""
自我修正协调器：SelfCorrectingOrchestrator

功能：
    整合所有新组件，执行"规划 → 注入 → 生成 → 验证 → 诊断 → 修复"的完整主循环。
    三层修复策略（local_rewrite / partial_rollback / memory_purge）由 MRSD 诊断结果驱动。
    MetaState 提供元认知门控，MetricCollector 收集内部机制指标。

依赖：Generator、SectionPlanner、DiscourseLedger、CommitmentExtractor、DTGStore、
      OnlineValidator、MRSD、AlignmentScorer、MetaState、PlanState、MetricCollector、
      CorrectionLog
被依赖：main.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel

from .agents.generator import Generator
from .agents.section_planner import SectionPlanner
from .algorithms.mrsd import MRSD
from .core.decision import Decision
from .core.meta_state import MetaState
from .core.plan import PlanState, SectionIntent
from .core.state import GenerationState
from .evaluation.metric_collector import MetricCollector
from .logging.correction_log import CorrectionLog
from .logging.run_logger import RunLogger
from .memory.commitment_extractor import CommitmentExtractor
from .memory.discourse_ledger import DiscourseLedger
from .memory.dtg_store import DTGStore
from .metrics.alignment import AlignmentScorer
from .validators.online_validator import OnlineValidator


class SelfCorrectingOrchestrator:
    """
    自我修正协调器

    功能：
        1. 主循环：逐节规划 → DSL 注入 → 生成 → 验证 → MRSD 诊断 → 三层修复
        2. MetaState 元认知门控：gate_action() 控制 trust_validator_major / allow_rollback / strengthen_dsl_injection 三类动作
        3. DiscourseLedger 动态约束管理：salience 显著性驱动的条目注入
        4. MetricCollector 指标收集：不改变系统行为，仅记录

    参数：
        llm_client: LLM 客户端实例
        memory_path: DTG 存储路径（默认 ./sessions）
        session_name: 会话名称，用于文件命名和自动清理

    关键常数：
        MAX_RETRIES_PER_SECTION = 3
        MAX_ROLLBACKS = 5
        DCAS_THRESHOLD = 0.6
    """

    MAX_RETRIES_PER_SECTION = 3
    MAX_ROLLBACKS = 5
    DCAS_THRESHOLD = 0.6

    def __init__(
        self,
        llm_client,
        memory_path: str = "./sessions",
        session_name: str = "session",
        output_dir: str = "./outputs",
    ):
        """
        初始化自我修正协调器

        功能：
            创建并连接所有子组件，MetaState 初始化为信任状态。
            同时把 output_dir 传给 RunLogger，保证主入口与运行日志落盘目录一致。

        参数：
            llm_client: LLM 客户端实例
            memory_path: DTG 存储路径
            session_name: 会话名称
            output_dir: 输出目录（运行日志和相关工件的统一落盘位置）
        """
        self.dtg              = DTGStore(memory_path, session_name=session_name)
        self.meta_state       = MetaState()
        self.dsl              = DiscourseLedger(llm_client=llm_client)
        self.console          = Console()
        self.logger           = logging.getLogger(__name__)
        self.run_logger       = RunLogger(output_dir=output_dir, session_name=session_name)

        llm_client.attach_run_logger(self.run_logger)

        self.generator        = Generator(llm_client, run_logger=self.run_logger)
        self.section_planner  = SectionPlanner(llm_client, self.dtg)
        self.commitment_extractor = CommitmentExtractor(llm_client)
        self.alignment_scorer = AlignmentScorer(llm_client)
        self.online_validator = OnlineValidator(
            llm_client=llm_client,
            dtg_store=self.dtg,
            alignment_scorer=self.alignment_scorer,
            meta_state=self.meta_state,
            run_logger=self.run_logger,
        )
        self.mrsd = MRSD(dtg_store=self.dtg, llm_client=llm_client)
        self.correction_log   = CorrectionLog()
        self.metric_collector = MetricCollector()

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def generate_with_self_correction(
        self,
        task: str,
        constraints: List[str],
        outline: Dict[str, str],
    ) -> Tuple[str, List[Decision], CorrectionLog]:
        """
        自我修正生成（主方法）

        功能：
            主循环：对每节执行"规划→注入→生成→验证→诊断→修复"。
            失败时调用 MRSD 获取诊断结果，根据 repair_scope 选择修复策略。

        参数：
            task: 生成任务描述
            constraints: 全局约束列表（IMMUTABLE 类型）
            outline: 有序章节大纲 {section_id: section_title}

        返回值：
            Tuple[str, List[Decision], CorrectionLog]：
                (最终文本, 决策列表, 修正日志)
        """
        self._print_header(task, outline)
        self.run_logger.log_run_start(task, constraints, outline)

        try:
            state = self._initialize_state(constraints, outline)
            plan_state = PlanState(global_outline=outline)
            generated_content: Dict[str, str] = {}
            section_queue = list(outline.keys())

            rollback_count = 0
            current_idx = 0

            # 节级别的诊断跟踪
            consecutive_failures_this_section = 0
            last_purge_succeeded = False
            last_diagnosis_event_id: Optional[str] = None

            while current_idx < len(section_queue):
                section_id = section_queue[current_idx]
                state.current_section = section_id
                section_title = outline[section_id]

                self._print_section_start(section_id, section_title, current_idx, len(section_queue))
                if self.run_logger is not None:
                    self.run_logger.log_section_start(
                        section_id,
                        section_title,
                        current_idx,
                        len(section_queue),
                        state,
                        len(self.dsl.get_active_entries()),
                    )

                # 第一阶段：规划当前节（SectionPlanner）
                section_intent = self._plan_section(
                    section_id=section_id,
                    section_title=section_title,
                    task=task,
                    plan_state=plan_state,
                )
                plan_state.add_intent(section_intent)
                if self.run_logger is not None:
                    self.run_logger.log_planning(section_id, section_intent)

                # 准备 DSL 注入
                self._update_dsl_injection(state, section_id, section_queue, current_idx)

                rolled_back = False
                report = None
                content: Optional[str] = None
                decision: Optional[Decision] = None
                last_failure_reason: Optional[str] = None

                for attempt in range(self.MAX_RETRIES_PER_SECTION):
                    # 第二阶段：生成
                    temperature = self._get_temperature(consecutive_failures_this_section)
                    if self.run_logger is not None:
                        self.run_logger.log_attempt_start(section_id, attempt + 1, temperature)
                    try:
                        content, decision = self.generator.generate_with_decision(
                            state=state,
                            task=task,
                            recent_content=self._get_recent_content(),
                            section_intent=section_intent,
                            temperature=temperature,
                        )
                    except Exception as e:
                        self.logger.error("生成异常（section=%s attempt=%d）: %s", section_id, attempt + 1, e)
                        consecutive_failures_this_section += 1
                        last_failure_reason = "generator_parse_failure"
                        self.correction_log.add_retry(section_id, attempt + 1, "RETRY_SIMPLE", [str(e)])
                        continue

                    # 第三阶段：验证
                    try:
                        report = self.online_validator.validate_and_diagnose(decision, content, state)
                        self.meta_state.update_validator_stability(report.dcas_score)
                    except Exception as e:
                        self.logger.error("验证异常（section=%s）: %s", section_id, e)
                        report = None

                    if report is None:
                        last_failure_reason = "validator_exception"
                        consecutive_failures_this_section += 1
                        self.correction_log.add_retry(
                            section_id,
                            attempt + 1,
                            "VALIDATOR_EXCEPTION",
                            ["validator_exception"],
                        )
                        continue

                    # 第四阶段：处理验证结果
                    if report.passed:
                        # 验证通过
                        self._on_section_success(
                            section_id=section_id,
                            content=content,
                            decision=decision,
                            state=state,
                            generated_content=generated_content,
                            section_queue=section_queue,
                            plan_state=plan_state,
                            attempt=attempt,
                            dcas=report.dcas_score,
                        )
                        # 记录诊断结果（若上一次有诊断）
                        if last_diagnosis_event_id:
                            self.metric_collector.record_diagnosis_outcome(
                                last_diagnosis_event_id, succeeded=True
                            )
                            last_diagnosis_event_id = None
                        self.metric_collector.record_section_first_pass(
                            section_id, passed_on_first_try=(attempt == 0)
                        )
                        consecutive_failures_this_section = 0
                        last_purge_succeeded = False
                        current_idx += 1
                        break

                    # 验证失败：MRSD 诊断
                    last_failure_reason = "validator_unknown"
                    consecutive_failures_this_section += 1
                    diagnosis = self.mrsd.diagnose(
                        report=report,
                        current_section_id=section_id,
                        section_queue=section_queue,
                        contamination_risk_score=self.meta_state.contamination_risk_score,
                        consecutive_failures_this_section=consecutive_failures_this_section,
                        low_trust_dsl_ref_ratio=self._compute_low_trust_ratio(section_id),
                        last_purge_succeeded=last_purge_succeeded,
                        intent_from_trusted_dsl=(
                            section_intent.dsl_trust_at_generation > 0.6
                        ),
                        recent_section_failure_tiers=[],
                    )

                    # 记录诊断事件
                    diag_event_id = self.metric_collector.record_diagnosis(
                        section_id=section_id,
                        predicted_tier=diagnosis.error_tier.value,
                        predicted_source=diagnosis.error_source.value,
                        confidence=diagnosis.confidence,
                        repair_scope=diagnosis.repair_scope,
                        causal_subgraph_size=len(diagnosis.causal_subgraph),
                        llm_calls_used=min(self.mrsd._max_llm_budget, len(diagnosis.causal_subgraph) + 1),
                    )
                    if last_diagnosis_event_id:
                        self.metric_collector.record_diagnosis_outcome(
                            last_diagnosis_event_id, succeeded=False
                        )
                    last_diagnosis_event_id = diag_event_id

                    # 更新 MetaState
                    self.meta_state.record_failure(
                        diagnosis.error_tier.value, diagnosis.error_source.value
                    )
                    self.meta_state.update_contamination_risk(
                        low_trust_ref_ratio=self._compute_low_trust_ratio(section_id),
                        recent_same_tier_failure_rate=0.0,
                    )

                    self._print_failure(section_id, attempt + 1, diagnosis, report)
                    self.correction_log.add_retry(
                        section_id, attempt + 1,
                        f"{diagnosis.repair_scope}({diagnosis.error_tier.value})",
                        report.issues,
                    )

                    # 执行修复
                    if diagnosis.repair_scope == "partial_rollback":
                        # 回退策略：需要 MetaState 门控
                        if (
                            not self.meta_state.gate_action("allow_rollback")
                            or rollback_count >= self.MAX_ROLLBACKS
                            or not diagnosis.should_rollback()
                        ):
                            self.logger.info("回退被门控或超限，降级为 local_rewrite")
                            self._update_dsl_injection_strengthen(state)
                            continue

                        target = diagnosis.target_section
                        if target and self._execute_rollback(
                            target_section=target,
                            current_section=section_id,
                            reason="; ".join(i.description for i in report.issues),
                            state=state,
                            generated_content=generated_content,
                            section_queue=section_queue,
                            plan_state=plan_state,
                        ):
                            rollback_count += 1
                            self.meta_state.remaining_rollback_budget = max(
                                0, self.meta_state.remaining_rollback_budget - 1
                            )
                            self.correction_log.add_rollback(
                                from_section=section_id,
                                to_section=target,
                                reason=f"MRSD:{diagnosis.error_tier.value}",
                            )
                            self.metric_collector.record_repair(
                                section_id=section_id,
                                repair_scope="partial_rollback",
                                triggered_by_tier=diagnosis.error_tier.value,
                                triggered_by_confidence=diagnosis.confidence,
                                succeeded=True,
                                extra_llm_calls=self.mrsd._max_llm_budget,
                                rollback_distance=abs(
                                    section_queue.index(section_id)
                                    - (section_queue.index(target) if target in section_queue else 0)
                                ),
                            )
                            consecutive_failures_this_section = 0
                            current_idx = section_queue.index(target)
                            rolled_back = True
                            break
                        else:
                            self.logger.warning("回退执行失败，降级为 local_rewrite")
                            continue

                    elif diagnosis.repair_scope == "memory_purge":
                        # 精确记忆清除
                        purged = self.dsl.purge_contaminated_entries(
                            contaminated_section=section_id,
                            conflict_description="; ".join(
                                i.description for i in report.issues[:2]
                            ),
                        )
                        last_purge_succeeded = len(purged) > 0
                        self.logger.info("memory_purge：清除 %d 条条目", len(purged))
                        # 更新 DSL 注入
                        self._update_dsl_injection(state, section_id, section_queue, current_idx)
                        self.metric_collector.record_repair(
                            section_id=section_id,
                            repair_scope="memory_purge",
                            triggered_by_tier=diagnosis.error_tier.value,
                            triggered_by_confidence=diagnosis.confidence,
                            succeeded=last_purge_succeeded,
                            extra_llm_calls=0,
                        )
                        continue

                    else:
                        # local_rewrite：更新 DSL 注入（若需要强化）
                        if diagnosis.decoding_config.strengthen_dsl_injection:
                            self._update_dsl_injection_strengthen(state)
                        # plan_level：重规划 SectionIntent
                        if diagnosis.decoding_config.trigger_section_intent_revision:
                            section_intent = self._plan_section(
                                section_id=section_id,
                                section_title=section_title,
                                task=task,
                                plan_state=plan_state,
                                is_revision=True,
                                revision_reason="; ".join(i.description for i in report.issues),
                            )
                            plan_state.revise_intent(
                                section_id=section_id,
                                new_intent=section_intent,
                                reason="plan_level repair",
                            )
                        self.metric_collector.record_repair(
                            section_id=section_id,
                            repair_scope="local_rewrite",
                            triggered_by_tier=diagnosis.error_tier.value,
                            triggered_by_confidence=diagnosis.confidence,
                            succeeded=False,  # 后续验证后更新
                            extra_llm_calls=1,
                        )
                        continue

                else:
                    # 超过最大重试次数：降级接受最后一次的版本
                    if not rolled_back:
                        reason_code = last_failure_reason or "validator_unknown"
                        self.logger.warning(
                            "degraded_acceptance: section=%s attempts=%d reason=%s",
                            section_id,
                            self.MAX_RETRIES_PER_SECTION,
                            reason_code,
                        )
                        if self.run_logger is not None:
                            self.run_logger.log_section_degraded(
                                section_id,
                                self.MAX_RETRIES_PER_SECTION,
                                reason_code,
                            )
                        fallback_content = self._coerce_degraded_section_content(content)
                        fallback_decision = decision

                        generated_content[section_id] = fallback_content
                        state.generated_sections.append(section_id)
                        state.section_snippets[section_id] = fallback_content[:300]
                        state.section_summaries[section_id] = fallback_content[:500]
                        state.update_progress()
                        if fallback_decision:
                            self.dtg.add_decision(fallback_decision)

                        last_issues = report.issues if report else []
                        self.correction_log.add_failure(section_id, last_issues)
                        state.flagged_issues.append(
                            f"{section_id}: degraded content accepted after validation failure"
                        )
                        self.metric_collector.record_section_first_pass(section_id, False)
                        consecutive_failures_this_section = 0
                        current_idx += 1

            # 组装最终文本
            final_text = self._assemble_text(outline, generated_content)
            self._print_summary()
            self.run_logger.log_run_summary(
                self.correction_log.get_statistics(),
                self.meta_state,
            )

            return final_text, self.dtg.decision_log, self.correction_log
        finally:
            self.run_logger.close()

    # ------------------------------------------------------------------
    # 第一阶段：规划
    # ------------------------------------------------------------------

    def _plan_section(
        self,
        section_id: str,
        section_title: str,
        task: str,
        plan_state: PlanState,
        is_revision: bool = False,
        revision_reason: str = "",
    ) -> SectionIntent:
        """
        调用 SectionPlanner 生成当前节的 SectionIntent

        功能：
            获取 DSL 上下文、已完成章节摘要，调用 SectionPlanner.plan_section()。

        参数：
            section_id: 当前节 ID
            section_title: 当前节标题
            task: 全局任务描述
            plan_state: 当前规划状态
            is_revision: 是否为修订（plan_level repair 时为 True）
            revision_reason: 修订原因（plan_level repair 时提供）

        返回值：
            SectionIntent：生成的局部计划
        """
        memory_trust = self.dsl.compute_memory_trust_level()
        low_trust_ids = self.dsl.get_low_trust_entry_ids(threshold=0.5)
        dsl_entry_ids = [e.entry_id for e in self.dsl.get_active_entries()]

        injectable = self.dsl.get_injectable_entries(
            target_section_idx=list(plan_state.global_outline.keys()).index(section_id)
            if section_id in plan_state.global_outline else 0,
            total_sections=len(plan_state.global_outline),
            recent_decision_ids=[],
            historical_failure_entry_ids=[],
            outline=plan_state.global_outline,
            target_section_id=section_id,
        )
        dsl_context = "\n".join(f"- [{e.commitment_type.value}] {e.content}" for e in injectable)

        section_summaries_str = "\n".join(
            f"[{sid}] {intent.local_goal}"
            for sid, intent in list(plan_state.section_intents.items())[:5]
        )

        if is_revision and revision_reason:
            task_with_reason = f"{task}\n\n(Revision reason: {revision_reason})"
        else:
            task_with_reason = task

        try:
            return self.section_planner.plan_section(
                section_id=section_id,
                section_title=section_title,
                task_description=task_with_reason,
                dsl_context=dsl_context,
                section_summaries=section_summaries_str,
                source_dsl_entry_ids=dsl_entry_ids,
                dsl_trust_at_generation=memory_trust,
            )
        except Exception as e:
            self.logger.warning("SectionPlanner 异常，使用默认 intent：%s", e)
            return SectionIntent.create(
                section_id=section_id,
                local_goal=f"Complete the content for {section_title}",
                scope_boundary=f"This section must stay within {section_title} and must not cover later sections",
                open_loops_to_advance=[],
                commitments_to_maintain=[],
                risks_to_avoid=[],
                success_criteria=["The content matches the section goal and does not violate major constraints."],
                source_dsl_entry_ids=dsl_entry_ids,
                dsl_trust_at_generation=memory_trust,
            )

    # ------------------------------------------------------------------
    # 第二阶段：DSL 注入
    # ------------------------------------------------------------------

    def _update_dsl_injection(
        self,
        state: GenerationState,
        section_id: str,
        section_queue: List[str],
        current_idx: int,
    ) -> None:
        """
        更新 GenerationState 的 DSL 注入文本

        功能：
            计算当前节的 injectable entries，格式化为注入字符串，
            写入 state.dsl_injection。

        参数：
            state: 当前生成状态
            section_id: 当前节 ID
            section_queue: 全局节列表
            current_idx: 当前节序号（0-based）
        """
        injectable = self.dsl.get_injectable_entries(
            target_section_idx=current_idx,
            total_sections=len(section_queue),
            recent_decision_ids=[
                d.decision_id for d in self.dtg.decision_log[-2:]
            ],
            historical_failure_entry_ids=[],
            outline={sid: sid for sid in section_queue},
            target_section_id=section_id,
        )
        if injectable:
            state.dsl_injection = "\n".join(
                f"- [{e.commitment_type.value}/{e.constraint_type.value}] {e.content}"
                for e in injectable
            )
        else:
            state.dsl_injection = ""

        if getattr(self, "run_logger", None) is not None:
            self.run_logger.log_dsl_injection(section_id, injectable)

    def _update_dsl_injection_strengthen(self, state: GenerationState) -> None:
        """
        强化 DSL 注入（strengthen_dsl_injection=True 时调用）

        功能：
            在当前 dsl_injection 基础上追加强调说明。
        """
        if state.dsl_injection:
            state.dsl_injection = "[Important] Strictly follow the DSL state constraints below:\n" + state.dsl_injection

    # ------------------------------------------------------------------
    # 成功处理
    # ------------------------------------------------------------------

    def _on_section_success(
        self,
        section_id: str,
        content: str,
        decision: Decision,
        state: GenerationState,
        generated_content: Dict[str, str],
        section_queue: List[str],
        plan_state: PlanState,
        attempt: int,
        dcas: float,
    ) -> None:
        """
        验证通过后的统一处理逻辑

        功能：
            1. 保存内容和决策
            2. 更新 state 和 plan_state
            3. 从生成内容中提取承诺并写入 DSL
            4. 更新 DSL 条目稳定性
            5. 更新 MetaState.memory_trust_level
            6. 记录 DSL 快照（MetricCollector）
            7. 打印成功信息

        参数：
            section_id: 节 ID
            content: 生成内容
            decision: 决策对象
            state: 生成状态
            generated_content: 已生成内容字典
            section_queue: 全局节列表
            plan_state: 规划状态
            attempt: 本次为第几次尝试（0-based）
            dcas: DCAS 评分
        """
        generated_content[section_id] = content
        state.generated_sections.append(section_id)
        state.section_snippets[section_id] = content[:300]
        state.section_summaries[section_id] = content[:500]
        state.update_progress()
        self.dtg.add_decision(decision)
        self.correction_log.add_success(section_id, attempt + 1)

        # 提取承诺并写入 DSL
        new_entries: List[Any] = []
        try:
            new_entries = self.commitment_extractor.extract(
                section_content=content,
                section_id=section_id,
                decision_id=decision.decision_id,
                existing_summary="; ".join(state.section_summaries.get(sid, "")[:100] for sid in state.generated_sections[-3:-1]),
            )
            for entry in new_entries:
                self.dsl.add_entry(entry)
        except Exception as e:
            self.logger.warning("承诺提取失败（跳过）：%s", e)

        # 更新 DSL 稳定性
        self.dsl.update_entry_stability(section_id, state.generated_sections)

        # 更新 MetaState
        self.meta_state.memory_trust_level = self.dsl.compute_memory_trust_level()
        self.meta_state.update_contamination_risk(
            low_trust_ref_ratio=self._compute_low_trust_ratio(section_id),
            recent_same_tier_failure_rate=0.0,
        )

        # 记录 DSL 快照
        active = self.dsl.get_active_entries()
        self.metric_collector.record_dsl_snapshot(
            section_id=section_id,
            total_entries=len(active),
            high_stability_count=sum(1 for e in active if e.stability_score > 0.7),
            revoked_count=0,
            open_loop_count=len(self.dsl.get_open_loops()),
            closed_loop_count=0,
            memory_trust_level=self.meta_state.memory_trust_level,
        )

        if self.run_logger is not None:
            self.run_logger.log_section_success(
                section_id=section_id,
                total_attempts=attempt + 1,
                dcas=dcas,
                new_entries=new_entries,
                total_active_entries=len(active),
                memory_trust=self.meta_state.memory_trust_level,
            )

        self._log_postprocess_skipped(section_id)
        self._print_success(section_id, attempt + 1, dcas)

    # ------------------------------------------------------------------
    # 回退
    # ------------------------------------------------------------------

    def _log_postprocess_skipped(self, section_id: str) -> None:
        """记录 postprocess 默认关闭的原因。"""
        reason = "feature_disabled_by_default"
        self.logger.info("postprocess_skipped: section=%s reason=%s", section_id, reason)
        if getattr(self, "run_logger", None) is not None:
            self.run_logger.log_postprocess_skipped(section_id, reason)


    def _execute_rollback(
        self,
        target_section: str,
        current_section: str,
        reason: str,
        state: GenerationState,
        generated_content: Dict[str, str],
        section_queue: List[str],
        plan_state: PlanState,
    ) -> bool:
        """
        执行回退操作

        功能：
            1. 验证目标节存在
            2. 清除 generated_content、state 和 DTGStore 中的被回退节数据
            3. 回退 DiscourseLedger（清除被回退节引入的 DSL 条目）
            4. 回退 PlanState（清除 intent）
            5. 清除 flagged_issues 中的相关标记

        参数：
            target_section: 回退目标节 ID（从此节重新生成）
            current_section: 当前失败节 ID
            reason: 回退原因
            state: 生成状态
            generated_content: 已生成内容字典
            section_queue: 全局节列表
            plan_state: 规划状态

        返回值：
            bool：回退是否成功
        """
        if target_section not in section_queue:
            self.logger.warning("回退目标 '%s' 不在 section_queue 中，跳过", target_section)
            return False

        target_idx = section_queue.index(target_section)
        current_idx = (
            section_queue.index(current_section)
            if current_section in section_queue
            else len(section_queue)
        )
        sections_to_remove = section_queue[target_idx: current_idx + 1]

        # 清除内容和状态
        for sec in sections_to_remove:
            generated_content.pop(sec, None)
        state.generated_sections = [
            s for s in state.generated_sections if s not in sections_to_remove
        ]
        state.update_progress()

        # 回退 DTGStore
        prev_section = section_queue[target_idx - 1] if target_idx > 0 else None
        self.dtg.rollback_to_section(prev_section)

        # 回退 DiscourseLedger（清除目标节及之后引入的条目）
        cutoff = section_queue[target_idx - 1] if target_idx > 0 else section_queue[0]
        self.dsl.rollback_to_section(cutoff, section_queue)

        # 回退 PlanState（清除 intent）
        plan_state.rollback_intents_from(target_section, section_queue)

        # 清除 flagged_issues
        state.flagged_issues = [
            issue for issue in state.flagged_issues
            if not any(sec in issue for sec in sections_to_remove)
        ]

        self.logger.info(
            "回退完成：%s → %s，清除 %d 节，原因：%s",
            current_section, target_section, len(sections_to_remove), reason
        )
        return True

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _initialize_state(
        self,
        constraints: List[str],
        outline: Dict[str, str],
    ) -> GenerationState:
        """
        初始化 GenerationState

        参数：
            constraints: 全局约束列表
            outline: 章节大纲

        返回值：
            GenerationState：初始化后的生成状态
        """
        first_section = next(iter(outline))
        return GenerationState(
            current_section=first_section,
            progress=0.0,
            global_constraints=constraints,
            outline=outline,
            generated_sections=[],
        )

    def _get_recent_content(self) -> str:
        """
        获取最近 2 个决策的 expected_effect 作为上下文摘要

        返回值：
            str：最近内容摘要字符串
        """
        recent = self.dtg.decision_log[-2:]
        if not recent:
            return ""
        return "\n".join(f"[{d.target_section}] {d.expected_effect}" for d in recent)

    def _get_temperature(self, consecutive_failures: int) -> float:
        """
        根据连续失败次数决定生成温度

        关键实现细节：
            0 次失败 → 0.7（正常生成）
            1 次失败 → 0.5（中等保守）
            2+ 次失败 → 0.3（低温保守重写）

        参数：
            consecutive_failures: 当前节连续失败次数

        返回值：
            float：生成温度
        """
        if consecutive_failures == 0:
            return 0.7
        elif consecutive_failures == 1:
            return 0.5
        else:
            return 0.3

    def _compute_low_trust_ratio(self, section_id: str) -> float:
        """
        计算当前节引用的低信任 DSL 条目比例

        参数：
            section_id: 当前节 ID

        返回值：
            float：低信任条目比例 [0.0, 1.0]
        """
        active = self.dsl.get_active_entries()
        if not active:
            return 0.0
        low_trust = self.dsl.get_low_trust_entry_ids(threshold=0.5)
        return len(low_trust) / len(active)

    def _assemble_text(
        self,
        outline: Dict[str, str],
        generated_content: Dict[str, str],
    ) -> str:
        """
        按大纲顺序组装最终文本

        参数：
            outline: 章节大纲
            generated_content: 已生成内容字典

        返回值：
            str：完整文本
        """
        parts = []
        for section_id, title in outline.items():
            content = generated_content.get(section_id, "")
            parts.append(f"## {title}\n\n{content}")
        return "\n\n---\n\n".join(parts)

    def _coerce_degraded_section_content(self, content: Optional[str]) -> str:
        """Normalize degraded section content so final text never includes failure placeholders."""
        return content.strip() if content is not None else ""

    # ------------------------------------------------------------------
    # Rich 打印
    # ------------------------------------------------------------------

    def _print_header(self, task: str, outline: Dict[str, str]) -> None:
        self.console.print(Panel(
            f"[bold cyan]MetaWriter v4.0[/bold cyan]\n"
            f"Task: {task}\n"
            f"Sections: {len(outline)}",
            title="Generation Start",
            border_style="cyan",
        ))

    def _print_section_start(
        self, section_id: str, title: str, idx: int, total: int
    ) -> None:
        # 目的：
        #   Windows 上常见的 gbk 控制台无法稳定输出 ▶ / — 等字符。
        #   这里统一改用 ASCII，避免真实 benchmark 运行因为打印阶段报编码错而中断。
        self.console.print(
            f"\n[bold blue][{idx+1}/{total}] {section_id}[/bold blue] - {title}"
        )

    def _print_success(self, section_id: str, attempt: int, dcas: float) -> None:
        attempt_str = f"(attempt {attempt})" if attempt > 1 else "(first pass)"
        self.console.print(
            f"  [green][OK] {section_id} passed {attempt_str} DCAS={dcas:.3f}[/green]"
        )

    def _print_failure(self, section_id: str, attempt: int, diagnosis, report) -> None:
        issues_str = " | ".join(i.description[:40] for i in report.issues[:3])
        self.console.print(
            f"  [yellow][FAIL] {section_id} failed on attempt {attempt} -> "
            f"{diagnosis.repair_scope}({diagnosis.error_tier.value}/"
            f"{diagnosis.error_source.value})[/yellow]\n"
            f"    [dim]{issues_str}[/dim]"
        )

    def _print_summary(self) -> None:
        stats = self.correction_log.get_statistics()
        metric_summary = self.metric_collector.compute_repair_efficiency()
        self.console.print(Panel(
            f"Sections: {stats['total_sections']}   "
            f"First-pass rate: {stats['success_rate_first_try']:.0%}   "
            f"Retries: {stats['total_retries']}   "
            f"Rollbacks: {stats['total_rollbacks']}   "
            f"Failed sections: {stats['total_failures']}\n"
            f"False Rollback Rate: {metric_summary.get('false_rollback_rate', 'N/A')}   "
            f"DSL Trust Level: {self.meta_state.memory_trust_level:.3f}",
            title="[bold green]Generation Complete[/bold green]",
            border_style="green",
        ))
