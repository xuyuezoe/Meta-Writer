"""生成本地医学综述 benchmark 资产。

功能：
    以确定性方式生成 400 条医学综述类 benchmark 样本，并同步写出与之对齐的
    `samples.jsonl`、`outputs.jsonl` 与 `metrics.jsonl`。

参数：
    无。通过直接运行脚本触发生成。

返回值：
    无。

关键实现细节：
    1. 保持现有 JSONL schema，不新增运行时依赖。
    2. 通过不规则配额、交错调度与多原型写法降低任务的机械整齐感。
    3. 同步生成 demo outputs 与 metrics，确保 `sample_id` 在 examples 链路中完整对齐。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


CURRENT_DIR: Path = Path(__file__).resolve().parent
SAMPLES_PATH: Path = CURRENT_DIR / "samples.jsonl"
OUTPUTS_PATH: Path = CURRENT_DIR / "outputs.jsonl"
METRICS_PATH: Path = CURRENT_DIR / "metrics.jsonl"


MEDICAL_DOMAINS: List[Dict[str, object]] = [
    {"name": "心血管医学", "subtopics": ["急性冠脉综合征", "心力衰竭", "房颤", "高血压", "动脉粥样硬化"], "keywords": ["血流动力学", "危险分层", "二级预防"]},
    {"name": "肿瘤学", "subtopics": ["肺癌", "乳腺癌", "结直肠癌", "胰腺癌", "免疫治疗"], "keywords": ["分期", "精准治疗", "真实世界证据"]},
    {"name": "血液学", "subtopics": ["白血病", "淋巴瘤", "骨髓瘤", "贫血", "止凝血异常"], "keywords": ["分层治疗", "出血风险", "复发监测"]},
    {"name": "神经病学", "subtopics": ["卒中", "癫痫", "阿尔茨海默病", "帕金森病", "神经免疫疾病"], "keywords": ["神经功能评估", "影像整合", "长期预后"]},
    {"name": "呼吸与重症医学", "subtopics": ["ARDS", "慢阻肺", "哮喘", "肺栓塞", "机械通气"], "keywords": ["呼吸支持", "重症分层", "器官保护"]},
    {"name": "感染病学", "subtopics": ["脓毒症", "耐药菌感染", "病毒性肺炎", "结核病", "真菌感染"], "keywords": ["病原学证据", "抗菌药物管理", "传播控制"]},
    {"name": "内分泌与代谢医学", "subtopics": ["糖尿病", "肥胖", "甲状腺疾病", "骨代谢异常", "代谢综合征"], "keywords": ["代谢通路", "并发症防控", "生活方式干预"]},
    {"name": "肾脏与泌尿医学", "subtopics": ["慢性肾病", "急性肾损伤", "肾小球疾病", "泌尿系肿瘤", "透析管理"], "keywords": ["肾功能分层", "液体管理", "肾替代治疗"]},
    {"name": "消化与肝胆胰医学", "subtopics": ["肝硬化", "炎症性肠病", "急性胰腺炎", "脂肪肝", "消化道出血"], "keywords": ["肠道微环境", "并发症预警", "循证治疗"]},
    {"name": "风湿免疫医学", "subtopics": ["类风湿关节炎", "系统性红斑狼疮", "血管炎", "强直性脊柱炎", "免疫抑制治疗"], "keywords": ["自身免疫机制", "器官受累", "疾病活动度"]},
    {"name": "妇产与生殖医学", "subtopics": ["妊娠期高血压", "子宫内膜异位症", "不孕症", "宫颈癌筛查", "围产期管理"], "keywords": ["母胎安全", "生殖结局", "风险干预"]},
    {"name": "儿科与新生儿医学", "subtopics": ["早产儿管理", "儿童哮喘", "遗传代谢病", "儿童感染", "发育行为问题"], "keywords": ["生长发育", "年龄分层", "家庭参与"]},
    {"name": "老年医学", "subtopics": ["衰弱", "多病共存", "认知障碍", "跌倒", "缓和医疗"], "keywords": ["功能状态", "照护路径", "综合评估"]},
    {"name": "精神医学", "subtopics": ["抑郁障碍", "双相障碍", "精神分裂症", "睡眠障碍", "成瘾医学"], "keywords": ["症状维度", "药物与心理治疗", "社会功能"]},
    {"name": "皮肤病学", "subtopics": ["银屑病", "特应性皮炎", "黑色素瘤", "皮肤感染", "美容与修复"], "keywords": ["屏障功能", "炎症通路", "长期管理"]},
    {"name": "眼耳鼻喉口腔医学", "subtopics": ["青光眼", "糖网病", "鼻窦炎", "听力损失", "口腔种植"], "keywords": ["器官功能保护", "影像评估", "微创干预"]},
    {"name": "骨科与康复医学", "subtopics": ["骨质疏松", "骨关节炎", "运动损伤", "脊柱退变", "卒中后康复"], "keywords": ["功能重建", "运动处方", "疼痛管理"]},
    {"name": "外科与围术期医学", "subtopics": ["创伤救治", "微创外科", "围术期优化", "术后并发症", "感染控制"], "keywords": ["手术时机", "风险评估", "恢复增强"]},
    {"name": "影像与病理医学", "subtopics": ["胸部影像", "分子影像", "数字病理", "介入放射", "AI辅助诊断"], "keywords": ["多模态证据", "判读一致性", "质量控制"]},
    {"name": "公共卫生与医学伦理", "subtopics": ["疫苗策略", "慢病防控", "卫生技术评估", "药物监管", "临床研究伦理"], "keywords": ["人群健康", "政策评估", "公平性"]},
]

TASK_ANGLES: List[Dict[str, str]] = [
    {"name": "机制与病理生理", "focus": "机制", "compare_axes": "病理机制、关键通路与转化证据"},
    {"name": "流行病学与风险分层", "focus": "风险分层", "compare_axes": "流行病学负担、风险因素与分层工具"},
    {"name": "临床表现与自然史", "focus": "自然史", "compare_axes": "症状谱、疾病分型与自然病程"},
    {"name": "诊断路径与鉴别诊断", "focus": "诊断路径", "compare_axes": "筛查、诊断流程与鉴别诊断证据"},
    {"name": "严重度评估与分期", "focus": "严重度评估", "compare_axes": "分期工具、严重度评分与预警指标"},
    {"name": "药物与程序治疗比较", "focus": "治疗策略", "compare_axes": "药物方案、程序选择与获益风险"},
    {"name": "影像病理实验室整合", "focus": "证据整合", "compare_axes": "影像、病理与实验室证据协同"},
    {"name": "预后、复发与随访", "focus": "预后", "compare_axes": "短期结局、长期复发与随访路径"},
    {"name": "指南、争议与证据综合", "focus": "指南比较", "compare_axes": "不同指南、证据等级与争议焦点"},
    {"name": "伦理、监管与卫生系统", "focus": "卫生系统", "compare_axes": "伦理边界、监管路径与系统实施"},
]

CONTEXTS: List[Dict[str, str]] = [
    {"name": "成人住院场景", "keyword": "成人住院", "detail": "成人住院患者管理路径"},
    {"name": "儿科与青春期场景", "keyword": "儿科", "detail": "儿童与青少年照护需求"},
    {"name": "老年与多病共存场景", "keyword": "老年", "detail": "老年多病共存与脆弱性管理"},
    {"name": "急诊与重症场景", "keyword": "急诊重症", "detail": "急诊分诊与重症资源配置"},
    {"name": "基层与门诊场景", "keyword": "基层门诊", "detail": "基层医疗与门诊随访路径"},
    {"name": "围术期场景", "keyword": "围术期", "detail": "围术期风险控制与恢复优化"},
    {"name": "低资源环境场景", "keyword": "低资源环境", "detail": "资源受限地区的实施可行性"},
    {"name": "数字健康场景", "keyword": "数字健康", "detail": "数字工具、远程监测与数据治理"},
]

DOMAIN_QUOTAS: List[int] = [28, 27, 25, 24, 24, 22, 21, 21, 20, 20, 19, 19, 18, 18, 17, 16, 16, 15, 15, 15]
LENGTH_TARGETS: List[int] = [2000, 2200, 2400, 2600, 3000, 3400, 3800, 4200, 4600, 5200, 5800, 6400, 7200, 8400, 9600, 11200, 12800, 14400, 16000, 18000, 20000]
LENGTH_QUOTAS: List[int] = [15, 12, 18, 11, 21, 24, 16, 27, 19, 23, 26, 18, 22, 17, 21, 16, 24, 19, 17, 14, 20]
ARCHETYPE_QUOTAS: List[int] = [78, 64, 58, 52, 46, 41, 33, 28]
PROFILE_QUOTAS: List[int] = [96, 74, 68, 61, 54, 47]
TARGET_SAMPLE_COUNT: int = 400

ARCHETYPE_SPECS: List[Dict[str, object]] = [
    {
        "name": "canonical_review",
        "task_type": "analysis",
        "deliverable": "综述论文",
        "audience": "学术读者",
        "extra_anchor": "证据整合",
        "angle_route": [0, 1, 2, 6, 8],
        "context_route": [0, 4, 7, 2, 6],
        "prompt_forms": [
            "请撰写一篇约{length}字的{deliverable}，主题聚焦{domain_name}中的{subtopic}，面向{audience}展开。",
            "请完成一篇约{length}字的{deliverable}，围绕{domain_name}中的{subtopic}进行系统综述。",
        ],
        "clause_order": ["scope", "organizer", "compare", "close"],
    },
    {
        "name": "therapy_comparison",
        "task_type": "writing",
        "deliverable": "治疗比较性综述",
        "audience": "临床决策团队",
        "extra_anchor": "风险收益",
        "angle_route": [5, 7, 8, 4, 6],
        "context_route": [0, 3, 5, 6, 4],
        "prompt_forms": [
            "请写一篇约{length}字的{deliverable}，针对{domain_name}中的{subtopic}梳理不同策略的证据差异。",
            "请生成一篇约{length}字的{deliverable}，重点比较{domain_name}中{subtopic}相关方案的优势与局限。",
        ],
        "clause_order": ["compare", "scope", "organizer", "close"],
    },
    {
        "name": "diagnostic_pathway",
        "task_type": "analysis",
        "deliverable": "诊断路径综述",
        "audience": "专科培训团队",
        "extra_anchor": "鉴别重点",
        "angle_route": [3, 4, 6, 1, 8],
        "context_route": [3, 0, 4, 1, 6],
        "prompt_forms": [
            "请完成一篇约{length}字的{deliverable}，梳理{domain_name}中{subtopic}的识别、分层与证据组织方式。",
            "请准备一篇约{length}字的{deliverable}，围绕{domain_name}中的{subtopic}构建决策路径。",
        ],
        "clause_order": ["scope", "compare", "organizer", "close"],
    },
    {
        "name": "guideline_conflict",
        "task_type": "analysis",
        "deliverable": "争议比较稿",
        "audience": "指南起草小组",
        "extra_anchor": "指南差异",
        "angle_route": [8, 5, 3, 9, 1],
        "context_route": [0, 6, 4, 2, 7],
        "prompt_forms": [
            "请撰写一篇约{length}字的{deliverable}，聚焦{domain_name}中的{subtopic}及其证据分歧。",
            "请形成一篇约{length}字的{deliverable}，比较{domain_name}领域{subtopic}相关观点、路径与证据冲突。",
        ],
        "clause_order": ["compare", "organizer", "scope", "close"],
    },
    {
        "name": "implementation_brief",
        "task_type": "writing",
        "deliverable": "实施评估型综述",
        "audience": "卫生系统评估者",
        "extra_anchor": "系统适配",
        "angle_route": [9, 4, 5, 8, 7],
        "context_route": [6, 7, 4, 0, 3],
        "prompt_forms": [
            "请完成一篇约{length}字的{deliverable}，讨论{domain_name}中{subtopic}在不同资源场景下的落地条件。",
            "请写一篇约{length}字的{deliverable}，围绕{domain_name}中的{subtopic}分析系统实施与现实约束。",
        ],
        "clause_order": ["scope", "organizer", "close", "compare"],
    },
    {
        "name": "subgroup_focus",
        "task_type": "writing",
        "deliverable": "子群聚焦综述",
        "audience": "跨学科照护团队",
        "extra_anchor": "子群分层",
        "angle_route": [1, 2, 7, 5, 4],
        "context_route": [1, 2, 0, 3, 6],
        "prompt_forms": [
            "请生成一篇约{length}字的{deliverable}，聚焦{domain_name}中的{subtopic}在特定人群或场景下的差异。",
            "请准备一篇约{length}字的{deliverable}，围绕{domain_name}中的{subtopic}比较不同子群的证据与管理重点。",
        ],
        "clause_order": ["scope", "compare", "close", "organizer"],
    },
    {
        "name": "translational_bridge",
        "task_type": "analysis",
        "deliverable": "转化桥接型综述",
        "audience": "临床与转化研究协作组",
        "extra_anchor": "转化意义",
        "angle_route": [0, 6, 5, 2, 7],
        "context_route": [7, 0, 4, 6, 2],
        "prompt_forms": [
            "请撰写一篇约{length}字的{deliverable}，把{domain_name}中{subtopic}的基础机制与临床应用连接起来。",
            "请写一篇约{length}字的{deliverable}，解释{domain_name}中的{subtopic}如何从机制研究走向实践决策。",
        ],
        "clause_order": ["organizer", "scope", "compare", "close"],
    },
    {
        "name": "evidence_gap_agenda",
        "task_type": "writing",
        "deliverable": "证据缺口导向综述",
        "audience": "研究设计团队",
        "extra_anchor": "研究议程",
        "angle_route": [8, 9, 1, 7, 3],
        "context_route": [6, 4, 7, 2, 3],
        "prompt_forms": [
            "请完成一篇约{length}字的{deliverable}，围绕{domain_name}中的{subtopic}总结已知证据与待解问题。",
            "请形成一篇约{length}字的{deliverable}，在{domain_name}领域的{subtopic}上梳理争议、缺口与后续研究方向。",
        ],
        "clause_order": ["scope", "close", "organizer", "compare"],
    },
]

PROFILE_SPECS: List[Dict[str, object]] = [
    {
        "name": "taxonomy_profile",
        "organizer_anchor": "分类框架",
        "bridge_anchor": "证据整合",
        "closing_anchor": "开放问题",
        "periodic_requirements": [
            ["主体部分每隔两段需要出现一次比较或综合判断", "后半部分必须持续讨论局限性、证据缺口与未来工作"],
            ["正文中段应周期性回到比较与综合，不得只做定义堆叠", "末段之前至少有一段专门讨论局限性与开放问题"],
        ],
        "periodic_keywords": [
            [{"keyword": "比较", "every": 2, "start": 2}],
            [{"keyword": "综合", "every": 3, "start": 2}, {"keyword": "比较", "every": 2, "start": 3}],
        ],
    },
    {
        "name": "pathway_profile",
        "organizer_anchor": "临床路径",
        "bridge_anchor": "风险收益",
        "closing_anchor": "未来工作",
        "periodic_requirements": [
            ["中段需要多次回到临床路径的比较与重组", "结尾前必须讨论局限性与未来工作"],
            ["主体部分应反复比较不同临床路径的取舍逻辑", "后半部分必须说明局限性和未来工作"],
        ],
        "periodic_keywords": [
            [{"keyword": "路径", "every": 2, "start": 2}],
            [{"keyword": "比较", "every": 3, "start": 3}, {"keyword": "路径", "every": 2, "start": 2}],
        ],
    },
    {
        "name": "controversy_profile",
        "organizer_anchor": "争议焦点",
        "bridge_anchor": "指南差异",
        "closing_anchor": "证据缺口",
        "periodic_requirements": [
            ["正文中段需要周期性解释争议焦点与证据冲突", "收束部分必须讨论局限性、证据缺口与未来工作"],
            ["主体段落应反复回到争议焦点，不得只做单方立场陈述", "最后三分之一部分要交代局限性、证据缺口和未来工作"],
        ],
        "periodic_keywords": [
            [{"keyword": "争议", "every": 2, "start": 2}],
            [{"keyword": "比较", "every": 3, "start": 2}, {"keyword": "争议", "every": 2, "start": 3}],
        ],
    },
    {
        "name": "subgroup_profile",
        "organizer_anchor": "子群分层",
        "bridge_anchor": "适应证边界",
        "closing_anchor": "未来工作",
        "periodic_requirements": [
            ["主体部分需要定期回到子群分层和场景差异", "结尾必须讨论局限性与未来工作"],
            ["中段段落应持续比较不同子群的异同", "收束部分要说明局限性与未来工作"],
        ],
        "periodic_keywords": [
            [{"keyword": "子群", "every": 2, "start": 2}],
            [{"keyword": "比较", "every": 3, "start": 3}, {"keyword": "子群", "every": 2, "start": 2}],
        ],
    },
    {
        "name": "implementation_profile",
        "organizer_anchor": "实施障碍",
        "bridge_anchor": "系统适配",
        "closing_anchor": "证据缺口",
        "periodic_requirements": [
            ["正文应周期性回到实施障碍与系统适配问题", "后半部分必须同时说明局限性、证据缺口与未来工作"],
            ["主体分析需要反复处理实施障碍与场景适配，而不是只总结理想路径", "结尾前必须讨论局限性与证据缺口"],
        ],
        "periodic_keywords": [
            [{"keyword": "实施", "every": 2, "start": 2}],
            [{"keyword": "系统", "every": 3, "start": 2}, {"keyword": "实施", "every": 2, "start": 3}],
        ],
    },
    {
        "name": "evidence_map_profile",
        "organizer_anchor": "证据地图",
        "bridge_anchor": "转化意义",
        "closing_anchor": "研究议程",
        "periodic_requirements": [
            ["主体段落需要反复连接证据地图与综合判断", "结尾应明确写出局限性、研究议程与未来工作"],
            ["中段需要持续回到证据地图和证据空白", "最后两段必须保留局限性和未来工作"],
        ],
        "periodic_keywords": [
            [{"keyword": "证据", "every": 2, "start": 2}],
            [{"keyword": "综合", "every": 3, "start": 2}, {"keyword": "证据", "every": 2, "start": 3}],
        ],
    },
]

ORGANIZER_ANCHORS: List[str] = ["分类框架", "临床路径", "争议焦点", "子群分层", "实施障碍", "证据地图"]
OPTIONAL_ANCHORS: List[str] = ["证据整合", "风险收益", "指南差异", "适应证边界", "系统适配", "转化意义", "开放问题", "证据缺口", "研究议程", "实践启示", "鉴别重点", "比较"]


def _write_jsonl(file_path: Path, rows: List[Dict[str, object]]) -> None:
    """写入 JSONL 文件。

    参数：
        file_path: 目标文件路径。
        rows: 待写入的字典列表。

    返回值：
        无。

    关键实现细节：
        统一使用 UTF-8 与 JSONL 单行格式，避免 benchmark 读取侧再做额外适配。
    """
    file_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _validate_quota_sum(quota_list: List[int], expected_total: int, quota_name: str) -> None:
    """校验配额总数。

    参数：
        quota_list: 配额列表。
        expected_total: 期望总数。
        quota_name: 配额名称。

    返回值：
        无。

    关键实现细节：
        不使用默认补齐，若配额和不正确则立即报错，避免数据分布被静默篡改。
    """
    if sum(quota_list) != expected_total:
        raise ValueError(f"{quota_name} 总和必须为 {expected_total}，当前为 {sum(quota_list)}")


def _build_irregular_sequence(quota_list: List[int], stride: int, offset: int) -> List[int]:
    """根据配额构造不规则但确定性的索引序列。

    参数：
        quota_list: 各类别配额。
        stride: 步长参数。
        offset: 偏移参数。

    返回值：
        List[int]：长度等于配额和的索引序列。

    关键实现细节：
        使用带近期惩罚的贪心编排，避免相同标签长串连续出现，同时保持完全确定性。
    """
    remaining = list(quota_list)
    sequence: List[int] = []
    pointer = offset % len(quota_list)
    total_count = sum(quota_list)

    while len(sequence) < total_count:
        best_index: int | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None

        for shift_index in range(len(quota_list)):
            candidate_index = (pointer + shift_index * stride + len(sequence) * 2 + offset) % len(quota_list)
            if remaining[candidate_index] <= 0:
                continue

            streak_count = 0
            for previous_index in reversed(sequence):
                if previous_index != candidate_index:
                    break
                streak_count += 1

            recent_count = sequence[-5:].count(candidate_index)
            used_count = quota_list[candidate_index] - remaining[candidate_index]
            rotation_penalty = shift_index
            phase_bias = (candidate_index * 7 + len(sequence) * 3 + offset) % 11
            score = (
                -streak_count,
                -recent_count,
                remaining[candidate_index],
                -used_count,
                -rotation_penalty,
                -phase_bias,
            )

            if best_score is None or score > best_score:
                best_score = score
                best_index = candidate_index

        if best_index is None:
            raise ValueError("无法继续生成不规则索引序列")

        sequence.append(best_index)
        remaining[best_index] -= 1
        pointer = (best_index + stride + 1 + len(sequence) % 4 + offset) % len(quota_list)

    return sequence


def _compute_expected_blocks(required_length_words: int, profile_name: str) -> int:
    """根据目标长度和约束风格计算段落数。

    参数：
        required_length_words: 目标字数。
        profile_name: 约束风格名称。

    返回值：
        int：建议段落数。

    关键实现细节：
        在长度基础上叠加 profile 偏置，使不同类型任务的段落密度不完全一致。
    """
    if required_length_words <= 2400:
        base_blocks = 5
    elif required_length_words <= 3800:
        base_blocks = 6
    elif required_length_words <= 5800:
        base_blocks = 7
    elif required_length_words <= 8400:
        base_blocks = 8
    elif required_length_words <= 11200:
        base_blocks = 9
    elif required_length_words <= 14400:
        base_blocks = 10
    elif required_length_words <= 18000:
        base_blocks = 11
    else:
        base_blocks = 12

    profile_bias_map: Dict[str, int] = {
        "taxonomy_profile": 0,
        "pathway_profile": 1,
        "controversy_profile": 0,
        "subgroup_profile": 1,
        "implementation_profile": 1,
        "evidence_map_profile": 0,
    }
    if profile_name not in profile_bias_map:
        raise ValueError(f"未知的 profile_name: {profile_name}")
    return base_blocks + profile_bias_map[profile_name]


def _select_item(items: List[str], local_index: int, slot_index: int, multiplier: int) -> str:
    """按不规则步长选择列表元素。

    参数：
        items: 候选字符串列表。
        local_index: 域内已使用次数。
        slot_index: 全局样本索引。
        multiplier: 选择步长。

    返回值：
        str：选中的元素。

    关键实现细节：
        使用局部索引和全局索引交叉编排，避免简单顺序轮转造成过度整齐。
    """
    if len(items) == 0:
        raise ValueError("items 不能为空")
    selected_index = (local_index * multiplier + slot_index * (multiplier + 1) + len(items)) % len(items)
    return items[selected_index]


def _parse_int_value(raw_value: object, field_name: str) -> int:
    """解析整数字段。

    参数：
        raw_value: 原始字段值。
        field_name: 字段名称。

    返回值：
        int：解析后的整数值。

    关键实现细节：
        仅接受整数或整数字符串，避免在生成与评分辅助阶段吞掉结构错误。
    """
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} 必须是整数或整数字符串")
    return int(raw_value)


def _build_prompt(
    archetype_spec: Dict[str, object],
    profile_spec: Dict[str, object],
    domain_name: str,
    subtopic: str,
    angle_name: str,
    compare_axes: str,
    context_name: str,
    context_detail: str,
    required_length_words: int,
    slot_index: int,
) -> str:
    """构造单条任务 prompt。

    参数：
        archetype_spec: 任务原型配置。
        profile_spec: 约束风格配置。
        domain_name: 医学域名称。
        subtopic: 具体主题。
        angle_name: 任务方向名称。
        compare_axes: 比较维度。
        context_name: 场景名称。
        context_detail: 场景细节描述。
        required_length_words: 目标字数。
        slot_index: 全局样本索引。

    返回值：
        str：完整提示词。

    关键实现细节：
        原型控制交付形式与句法顺序，profile 控制组织锚点与收束方式，从而降低统一模板感。
    """
    prompt_forms_object = archetype_spec["prompt_forms"]
    if not isinstance(prompt_forms_object, list):
        raise TypeError("prompt_forms 必须是列表")
    prompt_forms = [str(item) for item in prompt_forms_object]
    clause_order_object = archetype_spec["clause_order"]
    if not isinstance(clause_order_object, list):
        raise TypeError("clause_order 必须是列表")
    clause_order = [str(item) for item in clause_order_object]

    organizer_anchor = str(profile_spec["organizer_anchor"])
    bridge_anchor = str(profile_spec["bridge_anchor"])
    closing_anchor = str(profile_spec["closing_anchor"])
    prompt_template = prompt_forms[slot_index % len(prompt_forms)]
    prefix_text = prompt_template.format(
        length=required_length_words,
        deliverable=str(archetype_spec["deliverable"]),
        domain_name=domain_name,
        subtopic=subtopic,
        audience=str(archetype_spec["audience"]),
    )

    clause_map: Dict[str, str] = {
        "scope": f"开篇先完成研究范围界定，并交代讨论边界、术语口径与{context_name}下的关注重点",
        "organizer": f"正文需要围绕{organizer_anchor}组织内容，把{context_detail}纳入主线，而不是写成零散笔记",
        "compare": f"主体部分必须围绕{compare_axes}做横向比较和综合判断，并把{bridge_anchor}作为连接不同证据链的重要节点",
        "close": f"结尾必须明确讨论局限性、{closing_anchor}与未来工作，不要只给笼统结论",
    }

    ordered_clauses = [clause_map[item] for item in clause_order]
    return (
        f"{prefix_text}当前主线为{angle_name}，主题限定在{domain_name}中的{subtopic}。"
        f"请写成长篇医学综述而不是病例报告、科普短文或提纲。"
        f"{';'.join(ordered_clauses)}。"
    )


def _build_must_include(
    archetype_spec: Dict[str, object],
    profile_spec: Dict[str, object],
    domain_name: str,
    subtopic: str,
    focus_name: str,
    context_keyword: str,
    evidence_keyword: str,
    slot_index: int,
) -> List[str]:
    """构造任务必须覆盖的锚点列表。

    参数：
        archetype_spec: 任务原型配置。
        profile_spec: 约束风格配置。
        domain_name: 医学域名称。
        subtopic: 具体主题。
        focus_name: 核心焦点。
        context_keyword: 场景关键词。
        evidence_keyword: 医学证据关键词。
        slot_index: 全局样本索引。

    返回值：
        List[str]：必须覆盖项列表。

    关键实现细节：
        共享硬锚点与可变补充锚点混合使用，使任务结构有共识但不完全同构。
    """
    organizer_anchor = str(profile_spec["organizer_anchor"])
    bridge_anchor = str(profile_spec["bridge_anchor"])
    closing_anchor = str(profile_spec["closing_anchor"])
    base_items: List[str] = [
        "研究范围",
        organizer_anchor,
        domain_name,
        subtopic,
        focus_name,
        context_keyword,
        evidence_keyword,
        "局限性",
        "未来工作",
    ]

    extras_pool: List[str] = [
        bridge_anchor,
        closing_anchor,
        str(archetype_spec["extra_anchor"]),
        "比较",
        "证据缺口",
        "实践启示",
        "风险收益",
    ]

    target_length = 8 + (slot_index % 5)
    must_include: List[str] = []
    for item in base_items + extras_pool:
        if item not in must_include:
            must_include.append(item)
        if len(must_include) >= target_length:
            break

    if len(must_include) < 8:
        raise ValueError("must_include 长度不足，至少需要 8 个锚点")
    return must_include


def _build_checklist(
    profile_spec: Dict[str, object],
    domain_name: str,
    subtopic: str,
    focus_name: str,
    context_keyword: str,
    slot_index: int,
) -> List[str]:
    """构造多风格 checklist。

    参数：
        profile_spec: 约束风格配置。
        domain_name: 医学域名称。
        subtopic: 具体主题。
        focus_name: 核心焦点。
        context_keyword: 场景关键词。
        slot_index: 全局样本索引。

    返回值：
        List[str]：checklist 文本列表。

    关键实现细节：
        checklist 文案随 profile 变化，但保留可被本地评估识别的结构与语义锚点。
    """
    organizer_anchor = str(profile_spec["organizer_anchor"])
    bridge_anchor = str(profile_spec["bridge_anchor"])
    closing_anchor = str(profile_spec["closing_anchor"])
    template_groups: List[List[str]] = [
        [
            f"是否说明了{subtopic}的研究范围与问题边界",
            f"是否以{organizer_anchor}组织{domain_name}中的材料",
            f"是否围绕{focus_name}进行横向比较与综合",
            f"是否纳入{context_keyword}相关的实践约束",
            f"是否讨论局限性、{closing_anchor}与未来工作",
        ],
        [
            f"文稿前部是否交代{subtopic}讨论范围与术语口径",
            f"正文是否把{organizer_anchor}作为主组织轴",
            f"是否处理{bridge_anchor}与{focus_name}之间的关系",
            f"是否结合{context_keyword}讨论现实应用问题",
            f"结尾是否保留局限性和未来工作两个清晰板块",
        ],
        [
            f"是否完成{subtopic}的范围界定",
            f"是否提出可解释的{organizer_anchor}",
            f"是否比较不同路径在{focus_name}上的差异",
            f"是否呈现{context_keyword}下的情境化判断",
            f"是否明确说明局限性与未来工作",
            f"是否交代{closing_anchor}",
        ],
    ]
    selected_templates = template_groups[slot_index % len(template_groups)]
    return list(selected_templates)


def _build_proxy_questions(profile_spec: Dict[str, object], slot_index: int) -> List[Dict[str, str]]:
    """构造代理问题。

    参数：
        profile_spec: 约束风格配置。
        slot_index: 全局样本索引。

    返回值：
        List[Dict[str, str]]：代理问题列表。

    关键实现细节：
        问法允许变化，但 `answer` 字段始终使用显式锚点，保证本地评估稳定。
    """
    organizer_anchor = str(profile_spec["organizer_anchor"])
    closing_anchor = str(profile_spec["closing_anchor"])
    question_groups: List[List[Dict[str, str]]] = [
        [
            {"qid": "q1", "question": "文中是否说明了研究范围？", "answer": "研究范围"},
            {"qid": "q2", "question": f"文中是否呈现了{organizer_anchor}？", "answer": organizer_anchor},
            {"qid": "q3", "question": "文中是否讨论了局限性？", "answer": "局限性"},
            {"qid": "q4", "question": "文中是否提出了未来工作？", "answer": "未来工作"},
        ],
        [
            {"qid": "q1", "question": "开头是否完成研究范围界定？", "answer": "研究范围"},
            {"qid": "q2", "question": f"作者是否用{organizer_anchor}组织内容？", "answer": organizer_anchor},
            {"qid": "q3", "question": f"是否讨论了{closing_anchor}？", "answer": closing_anchor},
        ],
        [
            {"qid": "q1", "question": "文章有没有先说明研究范围？", "answer": "研究范围"},
            {"qid": "q2", "question": "文章有没有进行比较？", "answer": "比较"},
            {"qid": "q3", "question": "文章有没有说明局限性？", "answer": "局限性"},
            {"qid": "q4", "question": f"文章有没有涉及{organizer_anchor}？", "answer": organizer_anchor},
            {"qid": "q5", "question": "文章有没有提出未来工作？", "answer": "未来工作"},
        ],
    ]
    return [dict(item) for item in question_groups[slot_index % len(question_groups)]]


def _build_range_keywords(
    profile_spec: Dict[str, object],
    must_include: List[str],
    expected_blocks: int,
    slot_index: int,
) -> List[Dict[str, int | str]]:
    """构造区间关键词约束。

    参数：
        profile_spec: 约束风格配置。
        must_include: 必须覆盖项列表。
        expected_blocks: 目标段落数。
        slot_index: 全局样本索引。

    返回值：
        List[Dict[str, int | str]]：区间约束列表。

    关键实现细节：
        区间关键词数量在 3 到 5 个之间变化，从而避免所有任务共享完全相同的段落角色分配。
    """
    organizer_anchor = str(profile_spec["organizer_anchor"])
    bridge_anchor = str(profile_spec["bridge_anchor"])
    closing_anchor = str(profile_spec["closing_anchor"])
    range_keywords: List[Dict[str, int | str]] = [
        {"keyword": "研究范围", "start": 1, "end": 2},
        {"keyword": organizer_anchor, "start": 2, "end": min(expected_blocks - 2, 4)},
        {"keyword": "局限性", "start": max(1, expected_blocks - 2), "end": expected_blocks - 1},
        {"keyword": "未来工作", "start": expected_blocks, "end": expected_blocks},
    ]
    if bridge_anchor in must_include and slot_index % 2 == 0:
        range_keywords.insert(2, {"keyword": bridge_anchor, "start": max(3, expected_blocks // 2 - 1), "end": max(3, expected_blocks // 2 + 1)})
    if closing_anchor in must_include and closing_anchor not in {"未来工作", "局限性"} and slot_index % 3 == 0:
        range_keywords.insert(len(range_keywords) - 1, {"keyword": closing_anchor, "start": max(1, expected_blocks - 2), "end": expected_blocks})
    return range_keywords


def _build_periodic_requirements(profile_spec: Dict[str, object], slot_index: int) -> List[str]:
    """构造周期性要求文本。

    参数：
        profile_spec: 约束风格配置。
        slot_index: 全局样本索引。

    返回值：
        List[str]：周期性要求列表。

    关键实现细节：
        使用 profile 内部的多种措辞变体，保持约束意图稳定但表达不整齐。
    """
    periodic_requirements_object = profile_spec["periodic_requirements"]
    if not isinstance(periodic_requirements_object, list):
        raise TypeError("periodic_requirements 必须是列表")
    requirement_groups = [list(item) for item in periodic_requirements_object if isinstance(item, list)]
    if len(requirement_groups) == 0:
        raise ValueError("periodic_requirements 不能为空")
    return requirement_groups[slot_index % len(requirement_groups)]


def _build_periodic_keywords(profile_spec: Dict[str, object], slot_index: int) -> List[Dict[str, int | str]]:
    """构造周期关键词约束。

    参数：
        profile_spec: 约束风格配置。
        slot_index: 全局样本索引。

    返回值：
        List[Dict[str, int | str]]：周期关键词列表。

    关键实现细节：
        周期关键词数量在 1 到 2 个之间变化，用于打散段落节奏但保持可评估性。
    """
    periodic_keywords_object = profile_spec["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("periodic_keywords 必须是列表")
    keyword_groups = [list(item) for item in periodic_keywords_object if isinstance(item, list)]
    if len(keyword_groups) == 0:
        raise ValueError("periodic_keywords 不能为空")
    selected_group = keyword_groups[slot_index % len(keyword_groups)]
    return [dict(item) for item in selected_group if isinstance(item, dict)]


def _build_sample_row(
    slot_index: int,
    domain_index: int,
    local_domain_index: int,
    length_index: int,
    archetype_index: int,
    profile_index: int,
) -> Dict[str, object]:
    """构造单条 benchmark 样本。

    参数：
        slot_index: 全局样本索引。
        domain_index: 医学域索引。
        local_domain_index: 当前医学域内的样本序号。
        length_index: 长度档索引。
        archetype_index: 任务原型索引。
        profile_index: 约束风格索引。

    返回值：
        Dict[str, object]：可直接写入 samples.jsonl 的字典。

    关键实现细节：
        任务字段完全由确定性 schedule 推导，不使用随机数或兜底默认分支。
    """
    domain_spec = MEDICAL_DOMAINS[domain_index]
    archetype_spec = ARCHETYPE_SPECS[archetype_index]
    profile_spec = PROFILE_SPECS[profile_index]

    subtopics_object = domain_spec["subtopics"]
    if not isinstance(subtopics_object, list):
        raise TypeError("subtopics 必须是列表")
    domain_keywords_object = domain_spec["keywords"]
    if not isinstance(domain_keywords_object, list):
        raise TypeError("keywords 必须是列表")
    angle_route_object = archetype_spec["angle_route"]
    if not isinstance(angle_route_object, list):
        raise TypeError("angle_route 必须是列表")
    context_route_object = archetype_spec["context_route"]
    if not isinstance(context_route_object, list):
        raise TypeError("context_route 必须是列表")

    subtopic = _select_item([str(item) for item in subtopics_object], local_domain_index, slot_index, 3)
    angle_route = [int(item) for item in angle_route_object]
    context_route = [int(item) for item in context_route_object]
    angle_spec = TASK_ANGLES[angle_route[(local_domain_index + slot_index) % len(angle_route)] % len(TASK_ANGLES)]
    context_spec = CONTEXTS[context_route[(local_domain_index * 2 + slot_index) % len(context_route)] % len(CONTEXTS)]
    required_length_words = LENGTH_TARGETS[length_index]
    expected_blocks = _compute_expected_blocks(required_length_words, str(profile_spec["name"]))
    sample_id = f"med_s{slot_index + 1:03d}"
    focus_name = str(angle_spec["focus"])
    evidence_keyword = str(domain_keywords_object[(local_domain_index + slot_index) % len(domain_keywords_object)])

    prompt_text = _build_prompt(
        archetype_spec=archetype_spec,
        profile_spec=profile_spec,
        domain_name=str(domain_spec["name"]),
        subtopic=subtopic,
        angle_name=str(angle_spec["name"]),
        compare_axes=str(angle_spec["compare_axes"]),
        context_name=str(context_spec["name"]),
        context_detail=str(context_spec["detail"]),
        required_length_words=required_length_words,
        slot_index=slot_index,
    )
    must_include = _build_must_include(
        archetype_spec=archetype_spec,
        profile_spec=profile_spec,
        domain_name=str(domain_spec["name"]),
        subtopic=subtopic,
        focus_name=focus_name,
        context_keyword=str(context_spec["keyword"]),
        evidence_keyword=evidence_keyword,
        slot_index=slot_index,
    )
    checklist = _build_checklist(
        profile_spec=profile_spec,
        domain_name=str(domain_spec["name"]),
        subtopic=subtopic,
        focus_name=focus_name,
        context_keyword=str(context_spec["keyword"]),
        slot_index=slot_index,
    )
    proxy_questions = _build_proxy_questions(profile_spec=profile_spec, slot_index=slot_index)
    periodic_requirements = _build_periodic_requirements(profile_spec=profile_spec, slot_index=slot_index)
    periodic_keywords = _build_periodic_keywords(profile_spec=profile_spec, slot_index=slot_index)
    range_keywords = _build_range_keywords(
        profile_spec=profile_spec,
        must_include=must_include,
        expected_blocks=expected_blocks,
        slot_index=slot_index,
    )

    return {
        "sample_id": sample_id,
        "task_type": str(archetype_spec["task_type"]),
        "prompt": prompt_text,
        "constraints": {
            "required_length_words": required_length_words,
            "must_include": must_include,
            "periodic_requirements": periodic_requirements,
            "expected_blocks": expected_blocks,
            "once_keywords": must_include,
            "range_keywords": range_keywords,
            "periodic_keywords": periodic_keywords,
        },
        "proxy_questions": proxy_questions,
        "checklist": checklist,
    }


def _extract_constraints(sample_row: Dict[str, object]) -> Dict[str, object]:
    """提取并校验样本约束字典。

    参数：
        sample_row: 样本字典。

    返回值：
        Dict[str, object]：约束字典。

    关键实现细节：
        输出生成与 metrics 生成都依赖该函数，避免重复的结构校验逻辑。
    """
    constraints_object = sample_row["constraints"]
    if not isinstance(constraints_object, dict):
        raise TypeError("constraints 必须是字典")
    return constraints_object


def _choose_organizer_anchor(must_include: List[str]) -> str:
    """从必须覆盖项中找出主组织锚点。

    参数：
        must_include: 必须覆盖项列表。

    返回值：
        str：组织锚点。

    关键实现细节：
        若当前样本缺少任何已知组织锚点则直接报错，避免输出生成与任务定义脱节。
    """
    for organizer_anchor in ORGANIZER_ANCHORS:
        if organizer_anchor in must_include:
            return organizer_anchor
    raise ValueError(f"未在 must_include 中找到组织锚点：{must_include}")


def _ensure_keywords(blocks: List[str], keywords: List[str]) -> List[str]:
    """确保指定关键词出现在正文中。

    参数：
        blocks: 段落列表。
        keywords: 需要出现的关键词列表。

    返回值：
        List[str]：补齐后的段落列表。

    关键实现细节：
        缺失锚点会被定向追加到不同段落，而不是统一堆到结尾，以减少人工拼接痕迹。
    """
    updated_blocks = list(blocks)
    for keyword_index, keyword in enumerate(keywords):
        joined_text = "\n\n".join(updated_blocks)
        if keyword in joined_text:
            continue
        block_position = keyword_index % len(updated_blocks)
        updated_blocks[block_position] += f" 本段补充强调{keyword}。"
    return updated_blocks


def _ensure_range_keywords(blocks: List[str], range_specs: List[Dict[str, int | str]]) -> List[str]:
    """确保区间关键词命中。

    参数：
        blocks: 段落列表。
        range_specs: 区间关键词约束。

    返回值：
        List[str]：补齐后的段落列表。

    关键实现细节：
        每个区间关键词只补到其合法区间内，避免为了命中指标破坏段落角色。
    """
    updated_blocks = list(blocks)
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = max(1, int(spec["start"])) - 1
        end_index = min(len(updated_blocks), int(spec["end"]))
        if end_index <= start_index:
            continue
        target_slice = updated_blocks[start_index:end_index]
        if keyword in "\n".join(target_slice):
            continue
        updated_blocks[start_index] += f" 这里单独点出{keyword}。"
    return updated_blocks


def _ensure_periodic_keywords(blocks: List[str], periodic_specs: List[Dict[str, int | str]]) -> List[str]:
    """确保周期关键词命中。

    参数：
        blocks: 段落列表。
        periodic_specs: 周期关键词约束。

    返回值：
        List[str]：补齐后的段落列表。

    关键实现细节：
        只在需要的段落位置追加关键词，保证 build_metrics 的周期检测可以稳定通过。
    """
    updated_blocks = list(blocks)
    for spec in periodic_specs:
        keyword = str(spec["keyword"])
        every_value = int(spec["every"])
        start_index = max(1, int(spec["start"])) - 1
        if every_value <= 0:
            raise ValueError("periodic keyword 的 every 必须大于 0")
        current_index = start_index
        while current_index < len(updated_blocks):
            if keyword not in updated_blocks[current_index]:
                updated_blocks[current_index] += f" 本段继续围绕{keyword}推进综合判断。"
            current_index += every_value
    return updated_blocks


def _build_block_text(block_index: int, sample_row: Dict[str, object]) -> str:
    """构造示例输出中的单段正文。

    参数：
        block_index: 当前段落编号，从 0 开始。
        sample_row: 样本字典。

    返回值：
        str：单段文本。

    关键实现细节：
        不再使用单一“范围—分类—比较—局限—未来”模板，而是根据组织锚点与样本风格选择不同段落角色。
    """
    constraints = _extract_constraints(sample_row)
    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include 必须是列表")
    must_include = [str(item) for item in must_include_object]
    organizer_anchor = _choose_organizer_anchor(must_include)
    expected_blocks = _parse_int_value(constraints["expected_blocks"], "constraints.expected_blocks")

    domain_name = str(must_include[2])
    subtopic = str(must_include[3])
    focus_name = str(must_include[4])
    context_keyword = str(must_include[5])
    evidence_keyword = str(must_include[6])
    optional_items = [item for item in must_include if item not in {"研究范围", "局限性", "未来工作", domain_name, subtopic, focus_name, context_keyword, evidence_keyword, organizer_anchor}]
    style_index = (sum(ord(character) for character in str(sample_row["sample_id"])) + len(optional_items)) % 4
    opening_variants = ["开篇部分", "前言部分", "起始段", "第一段"]
    organizer_variants = ["第二个主要段落", "随后的结构段", "中前段", "主体组织段"]
    evidence_variants = ["这一段强调不同证据链的互补关系。", "本段重点梳理分歧来源与比较依据。", "这一段转向横向比较和综合判断。", "本段将不同证据路径重新并置。"]
    limitation_variants = ["临近结尾的段落集中讨论局限性。", "倒数第二段专门处理局限性。", "结尾前一段用于收束局限性。", "收束前的主要段落回到局限性。"]
    future_variants = ["最后一段单列未来工作。", "最终段落专门展开未来工作。", "结论段回到未来工作。", "收束段单独讨论未来工作。"]

    if block_index == 0:
        return (
            f"{opening_variants[style_index]}先界定研究范围，说明本文围绕{domain_name}中的{subtopic}展开，"
            f"只讨论与{focus_name}、{context_keyword}及{evidence_keyword}直接相关的证据链。"
            f"本段同时处理术语边界、纳入问题和排除问题，使后文比较不至于失去边界。"
        )
    if block_index == 1:
        return (
            f"{organizer_variants[style_index]}把全文的主组织轴明确为{organizer_anchor}，"
            f"并据此重排{subtopic}相关研究，避免把材料写成按年份平铺的知识清单。"
        )
    if block_index < expected_blocks - 2:
        extra_anchor = optional_items[(block_index + style_index) % len(optional_items)] if len(optional_items) > 0 else organizer_anchor
        return (
            f"第{block_index + 1}段继续展开{subtopic}在{domain_name}中的关键证据。{evidence_variants[(block_index + style_index) % len(evidence_variants)]}"
            f"这里不仅比较不同路径在{focus_name}上的差异，也把{extra_anchor}与{context_keyword}下的实践取舍放到同一分析框架中。"
        )
    if block_index == expected_blocks - 2:
        return (
            f"{limitation_variants[style_index]}当前关于{subtopic}的研究仍受到样本异质性、终点定义不一致、"
            f"外部可推广性不足以及{context_keyword}资料欠缺等因素影响，因此结论虽然有启发性，仍需要结合具体系统条件重新校准。"
        )
    return (
        f"{future_variants[style_index]}未来工作应继续围绕{focus_name}分层、{evidence_keyword}整合、"
        f"多中心数据共享以及面向{context_keyword}的实施研究展开，并进一步说明不同策略在长期结局与公平性上的综合含义。"
    )


def _build_output_row(sample_row: Dict[str, object], sample_index: int) -> Dict[str, object]:
    """构造单条示例输出。

    参数：
        sample_row: 样本字典。
        sample_index: 全局样本索引。

    返回值：
        Dict[str, object]：可直接写入 outputs.jsonl 的字典。

    关键实现细节：
        先生成多风格基础段落，再用后处理方式补齐 must_include、区间关键词与周期关键词，保证示例链路稳定。
    """
    constraints = _extract_constraints(sample_row)
    expected_blocks = _parse_int_value(constraints["expected_blocks"], "constraints.expected_blocks")
    range_keywords_object = constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("range_keywords 必须是列表")
    periodic_keywords_object = constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("periodic_keywords 必须是列表")
    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include 必须是列表")

    response_blocks = [_build_block_text(block_index, sample_row) for block_index in range(expected_blocks)]
    response_blocks = _ensure_keywords(response_blocks, [str(item) for item in must_include_object])
    response_blocks = _ensure_range_keywords(response_blocks, [dict(item) for item in range_keywords_object if isinstance(item, dict)])
    response_blocks = _ensure_periodic_keywords(response_blocks, [dict(item) for item in periodic_keywords_object if isinstance(item, dict)])
    response_text = "\n\n".join(response_blocks)

    usage_total_tokens = 900 + sample_index * 17
    latency_seconds = round(16.0 + (sample_index % 23) * 0.73, 2)
    return {
        "sample_id": str(sample_row["sample_id"]),
        "model_name": "demo-model",
        "response": response_text,
        "latency_seconds": latency_seconds,
        "usage_total_tokens": usage_total_tokens,
    }


def _contains_keyword(text: str, keyword: str) -> bool:
    """判断文本是否包含关键词。

    参数：
        text: 待检查文本。
        keyword: 关键词。

    返回值：
        bool：是否命中。

    关键实现细节：
        使用统一小写比较，兼容中英文混合关键词。
    """
    return keyword.lower() in text.lower()


def _compute_range_hit_rate(response_text: str, range_specs: List[Dict[str, int | str]]) -> float:
    """计算区间关键词命中率。

    参数：
        response_text: 模型输出文本。
        range_specs: 区间关键词约束。

    返回值：
        float：区间命中率。

    关键实现细节：
        与本地 benchmark 的段落切分方式保持一致，避免 metrics 与示例评估口径偏离。
    """
    blocks = [block.strip() for block in response_text.split("\n\n") if block.strip() != ""]
    if len(range_specs) == 0:
        return 1.0
    hit_count = 0
    for spec in range_specs:
        keyword = str(spec["keyword"])
        start_index = max(1, int(spec["start"])) - 1
        end_index = min(len(blocks), int(spec["end"]))
        if keyword in "\n".join(blocks[start_index:end_index]):
            hit_count += 1
    return hit_count / len(range_specs)


def _compute_periodic_hit_rate(response_text: str, periodic_specs: List[Dict[str, int | str]]) -> float:
    """计算周期关键词命中率。

    参数：
        response_text: 模型输出文本。
        periodic_specs: 周期关键词约束。

    返回值：
        float：周期命中率。

    关键实现细节：
        逐个目标位置检查关键词，保证与 build_metrics 的位置化逻辑一致。
    """
    blocks = [block.strip() for block in response_text.split("\n\n") if block.strip() != ""]
    if len(periodic_specs) == 0:
        return 1.0
    per_spec_scores: List[float] = []
    for spec in periodic_specs:
        keyword = str(spec["keyword"])
        every_value = int(spec["every"])
        start_index = max(1, int(spec["start"]))
        target_positions: List[int] = []
        current_position = start_index
        while current_position <= len(blocks):
            target_positions.append(current_position)
            current_position += every_value
        if len(target_positions) == 0:
            per_spec_scores.append(1.0)
            continue
        hit_count = 0
        for position in target_positions:
            if keyword in blocks[position - 1]:
                hit_count += 1
        per_spec_scores.append(hit_count / len(target_positions))
    return sum(per_spec_scores) / len(per_spec_scores)


def _build_metric_row(
    sample_row: Dict[str, object],
    output_row: Dict[str, object],
    sample_index: int,
) -> Dict[str, object]:
    """构造单条示例 metrics。

    参数：
        sample_row: 样本字典。
        output_row: 输出字典。
        sample_index: 全局样本索引。

    返回值：
        Dict[str, object]：可直接写入 metrics.jsonl 的字典。

    关键实现细节：
        metrics 以当前示例输出为基础做确定性计算，避免与重生成后的约束结构脱节。
    """
    constraints = _extract_constraints(sample_row)
    must_include_object = constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("must_include 必须是列表")
    checklist_object = sample_row["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("checklist 必须是列表")
    proxy_questions_object = sample_row["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("proxy_questions 必须是列表")
    range_keywords_object = constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("range_keywords 必须是列表")
    periodic_keywords_object = constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("periodic_keywords 必须是列表")

    must_include = [str(item) for item in must_include_object]
    checklist = [str(item) for item in checklist_object]
    response_text = str(output_row["response"])
    hard_hits = sum(1 for item in must_include if _contains_keyword(response_text, item))
    soft_hits = sum(1 for item in checklist if any(_contains_keyword(response_text, token) for token in item.replace("是否", "").replace("文稿", "").split("与")))
    proxy_hits = 0
    for proxy_question in proxy_questions_object:
        if isinstance(proxy_question, dict) and _contains_keyword(response_text, str(proxy_question["answer"])):
            proxy_hits += 1

    quality_base = 4.05 + (sample_index % 7) * 0.09
    return {
        "sample_id": str(sample_row["sample_id"]),
        "completion_rate": 1.0,
        "acc_once": round(hard_hits / len(must_include), 4),
        "acc_range": round(_compute_range_hit_rate(response_text, [dict(item) for item in range_keywords_object if isinstance(item, dict)]), 4),
        "acc_periodic": round(_compute_periodic_hit_rate(response_text, [dict(item) for item in periodic_keywords_object if isinstance(item, dict)]), 4),
        "quality_scores": {
            "Accuracy": round(min(4.9, quality_base), 2),
            "Coherence": round(min(4.9, quality_base + 0.08), 2),
            "Clarity": round(min(4.9, quality_base + 0.04), 2),
            "ReadingExperience": round(min(4.9, quality_base + 0.1), 2),
        },
        "instruction_hits": hard_hits + soft_hits,
        "instruction_total": len(must_include) + len(checklist),
        "instruction_hard_hits": hard_hits,
        "instruction_hard_total": len(must_include),
        "instruction_soft_hits": soft_hits,
        "instruction_soft_total": len(checklist),
        "syntax_pass_rate": 1.0,
        "schema_pass_rate": 1.0,
        "proxy_qa_correct": proxy_hits,
        "proxy_qa_total": len(proxy_questions_object),
    }


def main() -> None:
    """生成 400 条医学综述 benchmark 资产。

    参数：
        无。

    返回值：
        无。

    关键实现细节：
        先根据配额生成不规则调度，再构造样本、示例输出与 metrics，最后统一落盘。
    """
    _validate_quota_sum(DOMAIN_QUOTAS, TARGET_SAMPLE_COUNT, "DOMAIN_QUOTAS")
    _validate_quota_sum(LENGTH_QUOTAS, TARGET_SAMPLE_COUNT, "LENGTH_QUOTAS")
    _validate_quota_sum(ARCHETYPE_QUOTAS, TARGET_SAMPLE_COUNT, "ARCHETYPE_QUOTAS")
    _validate_quota_sum(PROFILE_QUOTAS, TARGET_SAMPLE_COUNT, "PROFILE_QUOTAS")

    domain_sequence = _build_irregular_sequence(DOMAIN_QUOTAS, stride=7, offset=3)
    length_sequence = _build_irregular_sequence(LENGTH_QUOTAS, stride=5, offset=1)
    archetype_sequence = _build_irregular_sequence(ARCHETYPE_QUOTAS, stride=3, offset=4)
    profile_sequence = _build_irregular_sequence(PROFILE_QUOTAS, stride=5, offset=2)

    if not (len(domain_sequence) == len(length_sequence) == len(archetype_sequence) == len(profile_sequence) == TARGET_SAMPLE_COUNT):
        raise ValueError("调度序列长度不一致")

    sample_rows: List[Dict[str, object]] = []
    output_rows: List[Dict[str, object]] = []
    metric_rows: List[Dict[str, object]] = []
    domain_seen_counts: List[int] = [0 for _ in MEDICAL_DOMAINS]

    for slot_index in range(TARGET_SAMPLE_COUNT):
        domain_index = domain_sequence[slot_index]
        local_domain_index = domain_seen_counts[domain_index]
        sample_row = _build_sample_row(
            slot_index=slot_index,
            domain_index=domain_index,
            local_domain_index=local_domain_index,
            length_index=length_sequence[slot_index],
            archetype_index=archetype_sequence[slot_index],
            profile_index=profile_sequence[slot_index],
        )
        sample_rows.append(sample_row)
        domain_seen_counts[domain_index] += 1

    for sample_index, sample_row in enumerate(sample_rows):
        output_row = _build_output_row(sample_row, sample_index)
        output_rows.append(output_row)
        metric_rows.append(_build_metric_row(sample_row, output_row, sample_index))

    if not (len(sample_rows) == len(output_rows) == len(metric_rows) == TARGET_SAMPLE_COUNT):
        raise ValueError("医学综述 benchmark 资产数量不一致")

    _write_jsonl(SAMPLES_PATH, sample_rows)
    _write_jsonl(OUTPUTS_PATH, output_rows)
    _write_jsonl(METRICS_PATH, metric_rows)


if __name__ == "__main__":
    main()