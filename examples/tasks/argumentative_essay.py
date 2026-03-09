"""
任务配置：议论文——大数据时代的个人隐私保护

演示系统在非虚构写作场景下的自我修正能力：
逻辑严密性约束 + 立场约束 + 结构一致性约束的联合满足。
与 scifi_story 形成对比，验证系统的任务无关泛化性。
"""
from typing import Dict, List


def get_task_config() -> Dict:
    """
    返回议论文任务配置

    返回：
        {
            'task':         任务描述字符串
            'constraints':  全局约束列表
            'outline':      章节大纲 {section_id: section_title}
            'session_name': 会话名称（用于文件命名）
        }
    """
    task: str = "写一篇关于大数据时代个人隐私保护的议论文"

    constraints: List[str] = [
        "必须指出至少两种具体的隐私威胁场景",
        "必须提出至少一项可操作的技术或制度层面解决方案",
        "语言客观严谨，避免情绪化表达",
        "各段落围绕中心论点展开，逻辑连贯",
        "结尾必须给出明确立场",
    ]

    outline: Dict[str, str] = {
        "intro":      "引言：大数据时代的便利与隐忧",
        "threat":     "当前个人隐私面临的主要威胁",
        "solution":   "技术与制度层面的保护路径",
        "conclusion": "结论：在便利与隐私间寻求平衡",
    }

    return {
        "task":         task,
        "constraints":  constraints,
        "outline":      outline,
        "session_name": "argumentative_essay",
    }
