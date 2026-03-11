from typing import List, Dict
import json
import re
import time
from pathlib import Path


class CorrectionLog:
    """
    修正日志

    记录类型：
    - SUCCESS：生成成功（包含尝试次数）
    - RETRY：因验证失败触发重试
    - ROLLBACK：因历史决策错误触发回退
    - FAILURE：超过最大重试次数后仍失败

    主要用途：
    1. 记录系统运行中的修正行为
    2. 统计分析（支持实验评估 RQ2）
    3. 可视化时间线（辅助调试）
    """

    _ICONS = {
        "SUCCESS": "✓",
        "RETRY":   "↺",
        "ROLLBACK": "⬅",
        "FAILURE": "✗",
    }

    def __init__(self):
        self.events: List[Dict] = []

    # ------------------------------------------------------------------
    # 事件记录
    # ------------------------------------------------------------------

    def add_success(self, section: str, attempts: int):
        """记录成功生成"""
        self.events.append({
            "type": "SUCCESS",
            "section": section,
            "attempts": attempts,
            "timestamp": int(time.time()),
        })

    def add_retry(self, section: str, attempt: int, action: str, issues: List):
        """
        记录重试

        :param section: 当前章节ID
        :param attempt: 当前是第几次尝试（从1开始）
        :param action: 触发重试的策略名（如 RETRY_SIMPLE、STRENGTHEN_CONSTRAINT）
        :param issues: 验证失败的 Issue 列表
        """
        self.events.append({
            "type": "RETRY",
            "section": section,
            "attempt": attempt,
            "action": action,
            "issues": [str(issue) for issue in issues],
            "timestamp": int(time.time()),
        })

    def add_rollback(self, from_section: str, to_section: str, reason: str):
        """
        记录回退

        :param from_section: 出错的章节ID
        :param to_section:   回退到的目标章节ID
        :param reason:       触发回退的原因描述
        """
        self.events.append({
            "type": "ROLLBACK",
            "from_section": from_section,
            "to_section": to_section,
            "reason": reason,
            "distance": self._calculate_distance(from_section, to_section),
            "timestamp": int(time.time()),
        })

    def add_failure(self, section: str, issues: List):
        """
        记录彻底失败（超过最大重试次数）

        :param section: 失败的章节ID
        :param issues:  最后一次验证的 Issue 列表
        """
        self.events.append({
            "type": "FAILURE",
            "section": section,
            "issues": [str(issue) for issue in issues],
            "timestamp": int(time.time()),
        })

    # ------------------------------------------------------------------
    # 统计分析
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict:
        """
        统计分析

        返回：
        {
            'total_sections':         int,   # 尝试过的独立章节数
            'success_first_try':      int,   # 一次成功的章节数
            'success_rate_first_try': float, # 一次成功率
            'total_retries':          int,   # 总重试次数
            'retry_by_action':        Dict,  # 按策略分组的重试次数
            'total_rollbacks':        int,   # 总回退次数
            'avg_rollback_distance':  float, # 平均回退距离
            'total_failures':         int,   # 彻底失败章节数
            'avg_attempts':           float, # 所有节（成功+失败）的平均实际尝试次数
        }
        """
        successes  = [e for e in self.events if e["type"] == "SUCCESS"]
        retries    = [e for e in self.events if e["type"] == "RETRY"]
        rollbacks  = [e for e in self.events if e["type"] == "ROLLBACK"]
        failures   = [e for e in self.events if e["type"] == "FAILURE"]

        # 独立章节：出现在 SUCCESS 或 FAILURE 中的 section
        all_sections = {e["section"] for e in successes} | {e["section"] for e in failures}
        total_sections = len(all_sections)

        # 一次成功：attempts == 1 的 SUCCESS
        success_first_try = sum(1 for e in successes if e.get("attempts", 1) == 1)
        success_rate = success_first_try / total_sections if total_sections else 0.0

        # 按策略分组的重试次数
        retry_by_action: Dict[str, int] = {}
        for e in retries:
            action = e.get("action", "UNKNOWN")
            retry_by_action[action] = retry_by_action.get(action, 0) + 1

        # 平均回退距离
        distances = [e.get("distance", 0) for e in rollbacks]
        avg_rollback_distance = sum(distances) / len(distances) if distances else 0.0

        # 所有节（成功+失败）的实际尝试次数
        # 成功节：来自 SUCCESS 事件的 attempts 字段
        # 失败节：该节 RETRY 事件数 + 1（最后一次未通过的尝试）
        section_attempts: Dict[str, int] = {}
        for e in successes:
            section_attempts[e["section"]] = e.get("attempts", 1)

        retry_per_section: Dict[str, int] = {}
        for e in retries:
            sec = e["section"]
            retry_per_section[sec] = retry_per_section.get(sec, 0) + 1

        for e in failures:
            sec = e["section"]
            section_attempts[sec] = retry_per_section.get(sec, 0) + 1

        all_attempt_counts = list(section_attempts.values())
        avg_attempts = sum(all_attempt_counts) / len(all_attempt_counts) if all_attempt_counts else 0.0

        return {
            "total_sections":          total_sections,
            "success_first_try":       success_first_try,
            "success_rate_first_try":  round(success_rate, 3),
            "total_retries":           len(retries),
            "retry_by_action":         retry_by_action,
            "total_rollbacks":         len(rollbacks),
            "avg_rollback_distance":   round(avg_rollback_distance, 2),
            "total_failures":          len(failures),
            "avg_attempts":            round(avg_attempts, 2),
        }

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def save(self, filename: str):
        """
        保存到JSON文件

        参数：
            filename: 目标文件路径

        关键行为：
            保存前先删除旧文件，确保输出文件是本次运行的结果
        """
        filepath = Path(filename)
        if filepath.exists():
            filepath.unlink()

        filepath.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "events": self.events,
            "statistics": self.get_statistics(),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def visualize_timeline(self) -> str:
        """
        可视化时间线

        输出示例：
          sec1: ✓ (1次)
          sec2: ↺ ↺ ✓ (3次)
          sec3: ⬅ 回退到sec1
          sec4: ✗
        """
        # 按章节分组，保持事件顺序
        section_events: Dict[str, List[Dict]] = {}
        for e in self.events:
            key = e.get("section") or e.get("from_section", "?")
            section_events.setdefault(key, []).append(e)

        lines = []
        for section, evts in section_events.items():
            parts = []
            for e in evts:
                t = e["type"]
                if t == "SUCCESS":
                    parts.append(f"{self._ICONS['SUCCESS']} ({e.get('attempts', 1)}次)")
                elif t == "RETRY":
                    parts.append(self._ICONS["RETRY"])
                elif t == "ROLLBACK":
                    parts.append(f"{self._ICONS['ROLLBACK']} 回退到{e.get('to_section', '?')}")
                elif t == "FAILURE":
                    parts.append(self._ICONS["FAILURE"])
            lines.append(f"  {section}: {' '.join(parts)}")

        return "\n".join(lines) if lines else "（无事件）"

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _calculate_distance(self, from_sec: str, to_sec: str) -> int:
        """
        计算回退距离

        策略：从章节ID末尾提取数字，计算差值；若无数字则返回1
        示例：section-5 → section-2 距离为3
        """
        def extract_num(s: str) -> int:
            nums = re.findall(r"\d+", s)
            return int(nums[-1]) if nums else 0

        n_from = extract_num(from_sec)
        n_to   = extract_num(to_sec)
        return abs(n_from - n_to) if (n_from or n_to) else 1
