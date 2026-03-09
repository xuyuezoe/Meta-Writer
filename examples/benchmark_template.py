"""
Benchmark 任务模板

TODO: 后续接入 benchmark 时在这里实现

预期接口设计如下，具体实现取决于所选 benchmark 的数据格式和评估标准。
"""
from typing import Dict, List


def load_benchmark_task(task_id: str) -> Dict:
    """
    从 benchmark 加载任务

    参数：
        task_id: benchmark 中的任务标识符

    返回：
        {
            'task': str,                   # 任务描述
            'constraints': List[str],      # 约束列表
            'outline': Dict[str, str],     # 章节大纲 {section_id: section_title}
            'reference': str               # 可选的参考答案
        }

    TODO: 实现时需要根据具体 benchmark 的数据格式进行解析
    """
    raise NotImplementedError("load_benchmark_task 尚未实现，请接入具体 benchmark 数据")


def evaluate_output(generated_text: str, reference: str) -> Dict:
    """
    评估生成结果

    参数：
        generated_text: MetaWriter 生成的文本
        reference:      benchmark 提供的参考答案

    返回：
        {
            'constraint_violation_rate': float,    # 约束违反率（越低越好）
            'entity_consistency_score': float,     # 实体一致性评分（越高越好）
            'logical_coherence': float             # 逻辑连贯性评分（越高越好）
        }

    TODO: 实现时需要根据具体 benchmark 的评估协议进行打分
    """
    raise NotImplementedError("evaluate_output 尚未实现，请接入具体 benchmark 评估协议")
