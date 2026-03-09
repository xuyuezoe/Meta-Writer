from typing import Tuple, List, Dict, Optional
import logging

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .agents.generator import Generator
from .memory.dtg_store import DTGStore
from .algorithms.debugger import DTGDebugger
from .metrics.alignment import AlignmentScorer
from .validators.online_validator import OnlineValidator
from .logging.correction_log import CorrectionLog
from .core.state import GenerationState
from .core.decision import Decision


class SelfCorrectingOrchestrator:
    """
    自我修正协调器

    核心功能：
    1. 主循环（带自我修正）：逐节生成，验证失败时执行修正
    2. 修正策略执行：RETRY_SIMPLE / STRENGTHEN_CONSTRAINT /
                     ROLLBACK_TO / RETRY_WITH_STRONGER_PROMPT
    3. 回退机制：清除错误决策和内容，跳回目标节重新生成
    4. 降级策略：超过最大重试次数后接受最后一次生成的内容，即使有问题也不管了，继续前进
    5. 修正日志：完整记录每次成功、重试、回退、失败
    """

    MAX_RETRIES_PER_SECTION = 3
    # 单次生成的最大回退次数（防止无限循环）
    MAX_ROLLBACKS = 5

    def __init__(self, llm_client, memory_path: str = "./sessions", session_name: str = "session"):
        """
        初始化自我修正协调器

        参数：
            llm_client:   LLM客户端实例
            memory_path:  DTG存储路径（默认 ./sessions）
            session_name: 会话名称，用于文件命名和自动清理
        """
        self.generator        = Generator(llm_client)
        self.dtg              = DTGStore(memory_path, session_name=session_name)
        self.debugger         = DTGDebugger(self.dtg)
        self.alignment_scorer = AlignmentScorer(llm_client)
        self.online_validator = OnlineValidator(
            llm_client, self.dtg, self.debugger, self.alignment_scorer
        )
        self.correction_log = CorrectionLog()
        self.console        = Console()
        self.logger         = logging.getLogger(__name__)

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

        核心流程：
            for each section:
                attempt = 0
                while attempt < max_retries:
                    生成 → 验证 →
                    if 通过: 保存并继续到下一节
                    else:    诊断 → 执行修正策略 → 重试或回退

        :param task:        生成任务描述
        :param constraints: 全局约束列表
        :param outline:     有序章节大纲 {section_id: section_title}
        :return:            (最终文本, 决策日志列表, 修正日志)
        """
        self._print_header(task, outline)

        state            = self._initialize_state(task, constraints, outline)
        generated_content: Dict[str, str] = {}
        section_queue    = list(outline.keys())
        rollback_count   = 0
        current_idx      = 0

        while current_idx < len(section_queue):
            section_id = section_queue[current_idx]
            state.current_section = section_id

            self._print_section_start(section_id, outline[section_id], current_idx, len(section_queue))

            # 每节独立的修正上下文
            correction_context: Dict = {
                "last_failure_reason":       None,
                "strengthened_constraints":  [],
            }

            rolled_back = False
            report = None          # 确保降级路径可访问

            for attempt in range(self.MAX_RETRIES_PER_SECTION):
                # ── 步骤1：生成 ──────────────────────────────────────
                try:
                    content, decision = self._generate_section(
                        state, task, correction_context
                    )
                except Exception as e:
                    self.logger.error("生成异常（section=%s attempt=%d）: %s", section_id, attempt + 1, e)
                    correction_context["last_failure_reason"] = f"生成异常：{e}"
                    self.correction_log.add_retry(section_id, attempt + 1, "RETRY_SIMPLE", [str(e)])
                    continue

                # ── 步骤2：验证 ──────────────────────────────────────
                try:
                    report = self.online_validator.validate_and_diagnose(
                        decision, content, state
                    )
                except Exception as e:
                    self.logger.error("验证异常（section=%s）: %s", section_id, e)
                    report = None

                # ── 步骤3：处理验证结果 ──────────────────────────────
                if report is None or report.passed:
                    # ✓ 通过（或验证器崩溃时降级接受）
                    generated_content[section_id] = content
                    state.generated_sections.append(section_id)
                    state.section_snippets[section_id] = content[:300]  # 供后续一致性检查
                    state.update_progress()
                    self.dtg.add_decision(decision)
                    self.correction_log.add_success(section_id, attempt + 1)

                    dcas = report.dcas_score if report else 1.0
                    self._print_success(section_id, attempt + 1, dcas)
                    current_idx += 1
                    break

                else:
                    # ✗ 失败 — 执行修正策略
                    strategy = report.suggested_strategy
                    params   = report.strategy_params

                    self._print_failure(section_id, attempt + 1, strategy, report)

                    if strategy == "RETRY_SIMPLE":
                        correction_context["last_failure_reason"] = (
                            "; ".join(i.description for i in report.issues)
                        )
                        self.correction_log.add_retry(
                            section_id, attempt + 1, strategy, report.issues
                        )
                        continue

                    elif strategy == "STRENGTHEN_CONSTRAINT":
                        violations = params.get("violated_constraints", [])
                        correction_context["strengthened_constraints"].extend(violations)
                        correction_context["last_failure_reason"] = (
                            f"约束违反：{violations}"
                        )
                        self.correction_log.add_retry(
                            section_id, attempt + 1, strategy, report.issues
                        )
                        continue

                    elif strategy == "RETRY_WITH_STRONGER_PROMPT":
                        correction_context["last_failure_reason"] = (
                            "; ".join(i.description for i in report.issues)
                        )
                        self.correction_log.add_retry(
                            section_id, attempt + 1, strategy, report.issues
                        )
                        continue

                    elif strategy.startswith("ROLLBACK_TO:"):
                        target_section = strategy.split(":", 1)[1]

                        if rollback_count >= self.MAX_ROLLBACKS:
                            self.logger.warning(
                                "已达到最大回退次数(%d)，跳过回退，继续重试", self.MAX_ROLLBACKS
                            )
                            correction_context["last_failure_reason"] = "回退次数超限，改为重试"
                            self.correction_log.add_retry(
                                section_id, attempt + 1, "RETRY_SIMPLE(rollback_limit)", report.issues
                            )
                            continue

                        success = self._execute_rollback(
                            target_section, section_id,
                            reason="; ".join(i.description for i in report.issues),
                            state=state,
                            generated_content=generated_content,
                            section_queue=section_queue,
                        )

                        if success:
                            rollback_count += 1
                            self.correction_log.add_rollback(
                                from_section=section_id,
                                to_section=target_section,
                                reason=strategy,
                            )
                            current_idx = section_queue.index(target_section)
                            rolled_back = True
                            break   # 跳出重试循环，从 target 节重新开始
                        else:
                            # 回退失败（目标节不存在），降级为简单重试
                            correction_context["last_failure_reason"] = f"回退目标 {target_section} 不存在"
                            self.correction_log.add_retry(
                                section_id, attempt + 1, "RETRY_SIMPLE(rollback_failed)", report.issues
                            )
                            continue

                    else:
                        # 未知策略 → 简单重试
                        correction_context["last_failure_reason"] = str(report.issues)
                        self.correction_log.add_retry(
                            section_id, attempt + 1, strategy or "UNKNOWN", report.issues
                        )
                        continue

            else:
                # ── 超过最大重试次数：降级策略 ──────────────────────
                if not rolled_back:
                    self.logger.warning(
                        "section=%s 超过最大重试次数，以最后一次内容降级继续", section_id
                    )
                    # 使用最后一次生成的内容（变量在 for 循环中保持）
                    try:
                        fallback_content  = content   # noqa: F821 — 至少执行过一次
                        fallback_decision = decision  # noqa: F821
                    except NameError:
                        fallback_content  = f"[{section_id} 生成失败]"
                        fallback_decision = None

                    generated_content[section_id] = fallback_content
                    state.generated_sections.append(section_id)
                    state.section_snippets[section_id] = fallback_content[:300]
                    state.update_progress()
                    if fallback_decision:
                        self.dtg.add_decision(fallback_decision)

                    last_issues = report.issues if report else []
                    self.correction_log.add_failure(section_id, last_issues)
                    state.flagged_issues.append(f"{section_id}：降级内容（验证未通过）")
                    current_idx += 1

        # ── 组装最终文本 ──────────────────────────────────────────────
        final_text = self._assemble_text(outline, generated_content)
        self._print_summary()

        return final_text, self.dtg.decision_log, self.correction_log

    # ------------------------------------------------------------------
    # 生成单节
    # ------------------------------------------------------------------

    def _generate_section(
        self,
        state: GenerationState,
        task: str,
        correction_context: Dict,
    ) -> Tuple[str, Decision]:
        """
        生成单个section（注入修正上下文）

        若 correction_context 含失败原因或加强约束，附加到任务描述末尾
        """
        task_with_context = task

        if correction_context.get("last_failure_reason"):
            task_with_context += (
                f"\n\n⚠️ 上次失败原因（请针对性改进）：{correction_context['last_failure_reason']}"
            )

        if correction_context.get("strengthened_constraints"):
            extra = "、".join(correction_context["strengthened_constraints"])
            task_with_context += f"\n⚠️ 特别注意以下约束必须严格满足：{extra}"

        recent_content = self._get_recent_content(state)
        return self.generator.generate_with_decision(state, task_with_context, recent_content)

    # ------------------------------------------------------------------
    # 回退
    # ------------------------------------------------------------------

    def _execute_rollback(
        self,
        target_section: str,
        current_section: str,
        reason: str,
        state: GenerationState,
        generated_content: Dict[str, str],
        section_queue: List[str],
    ) -> bool:
        """
        执行回退操作

        步骤：
        1. 验证目标节存在于 section_queue
        2. 确定需要清除的节（target 之后到 current 之间）
        3. 清除 generated_content 和 state.generated_sections
        4. 清除 DTGStore 中对应决策
        5. 清除 state.flagged_issues 中相关标记

        :return: 回退是否成功
        """
        if target_section not in section_queue:
            self.logger.warning("回退目标 '%s' 不在 section_queue 中，跳过", target_section)
            return False

        target_idx  = section_queue.index(target_section)
        current_idx = section_queue.index(current_section) if current_section in section_queue else len(section_queue)

        # 需要清除的节：[target_idx, current_idx]（含目标节，目标节本身也需重新生成）
        sections_to_remove = section_queue[target_idx: current_idx + 1]

        # 清除生成内容
        for sec in sections_to_remove:
            generated_content.pop(sec, None)

        # 清除状态中的已生成列表
        state.generated_sections = [
            s for s in state.generated_sections if s not in sections_to_remove
        ]
        state.update_progress()

        # 清除 DTGStore 决策（回退到 target 节之前的状态）
        prev_section = section_queue[target_idx - 1] if target_idx > 0 else None
        self.dtg.rollback_to_section(prev_section)

        # 清除与被删除节相关的 flagged_issues
        state.flagged_issues = [
            issue for issue in state.flagged_issues
            if not any(sec in issue for sec in sections_to_remove)
        ]

        self.logger.info(
            "回退完成：%s → %s，清除节：%s，原因：%s",
            current_section, target_section, sections_to_remove, reason
        )
        return True

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _initialize_state(
        self,
        task: str,
        constraints: List[str],
        outline: Dict[str, str],
    ) -> GenerationState:
        """初始化生成状态"""
        first_section = next(iter(outline))
        return GenerationState(
            current_section=first_section,
            progress=0.0,
            global_constraints=constraints,
            pending_goals=[task],
            outline=outline,
            generated_sections=[],
            flagged_issues=[],
        )

    def _get_recent_content(self, state: GenerationState) -> str:
        """
        获取最近生成内容（用于上下文）

        策略：取最近2个已生成节的 decision 中 reasoning 字段，
        避免把完整内容全部传入（节省token）。
        实际内容存在 generated_content 中由调用方传递，
        这里通过 DTGStore 的 decision_log 读取最近决策的 expected_effect 作为摘要。
        """
        recent = self.dtg.decision_log[-2:]
        if not recent:
            return ""
        return "\n".join(
            f"[{d.target_section}] {d.expected_effect}" for d in recent
        )

    def _assemble_text(
        self,
        outline: Dict[str, str],
        generated_content: Dict[str, str],
    ) -> str:
        """按大纲顺序组装最终文本"""
        parts = []
        for section_id, title in outline.items():
            content = generated_content.get(section_id, f"[{section_id} 内容缺失]")
            parts.append(f"## {title}\n\n{content}")
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Rich 打印
    # ------------------------------------------------------------------

    def _print_header(self, task: str, outline: Dict[str, str]):
        self.console.print(Panel(
            f"[bold cyan]MetaWriter v4.0[/bold cyan]\n"
            f"任务：{task}\n"
            f"章节数：{len(outline)}",
            title="开始生成",
            border_style="cyan",
        ))

    def _print_section_start(self, section_id: str, title: str, idx: int, total: int):
        self.console.print(
            f"\n[bold blue]▶ [{idx+1}/{total}] {section_id}[/bold blue] — {title}"
        )

    def _print_success(self, section_id: str, attempt: int, dcas: float):
        attempt_str = f"(第{attempt}次)" if attempt > 1 else "(一次通过)"
        self.console.print(
            f"  [green]✓ {section_id} 通过 {attempt_str} DCAS={dcas:.3f}[/green]"
        )

    def _print_failure(self, section_id: str, attempt: int, strategy: str, report):
        issues_str = " | ".join(i.description[:40] for i in report.issues[:3])
        self.console.print(
            f"  [yellow]✗ {section_id} 第{attempt}次失败 → {strategy}[/yellow]\n"
            f"    [dim]{issues_str}[/dim]"
        )

    def _print_summary(self):
        stats = self.correction_log.get_statistics()
        self.console.print(Panel(
            f"总节数：{stats['total_sections']}   "
            f"一次通过率：{stats['success_rate_first_try']:.0%}   "
            f"重试：{stats['total_retries']}次   "
            f"回退：{stats['total_rollbacks']}次   "
            f"失败：{stats['total_failures']}节",
            title="[bold green]生成完成[/bold green]",
            border_style="green",
        ))
