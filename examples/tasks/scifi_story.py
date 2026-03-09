"""
任务配置：科幻短篇故事

演示系统在创意写作场景下的自我修正能力：
角色设定约束 + 情节结构约束 + 风格约束的联合满足。
"""
from typing import Dict, List


def get_task_config() -> Dict:
    """
    返回科幻故事任务配置

    返回：
        {
            'task':         任务描述字符串
            'constraints':  全局约束列表
            'outline':      章节大纲 {section_id: section_title}
            'session_name': 会话名称（用于文件命名）
        }
    """
    task: str = "写一个简短的科幻故事"

    constraints: List[str] = [
        "主角名叫Alex",
        "背景是火星殖民地",
        "包含一个技术故障事件",
        "故事约500字",
        "必须有明确的结局",
    ]

    outline: Dict[str, str] = {
        "sec1": "描述火星殖民地环境和Alex的日常",
        "sec2": "技术故障发生",
        "sec3": "Alex的应对与结局",
    }

    return {
        "task":         task,
        "constraints":  constraints,
        "outline":      outline,
        "session_name": "scifi_story",
    }
