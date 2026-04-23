"""Benchmark 任务适配器。

功能：
    将仓库内 `metabench/examples/samples.jsonl` 中的样本转换为 Meta-Writer
    可直接消费的任务配置，并提供一个无需外部裁判模型的本地评估函数。

关键实现细节：
    1. `load_benchmark_task` 直接读取本地 JSONL 样本。
    2. `evaluate_output` 使用规则化启发式检查约束满足、信息命中与结构完整性。
    3. `reference` 返回结构化字典，保留评估所需最小信息，不依赖外部文档。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Union, cast


BENCHMARK_ROOT: Path = Path(__file__).resolve().parent.parent / "metabench"
SAMPLES_PATH: Path = BENCHMARK_ROOT / "examples" / "samples.jsonl"
DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "Document-level requirement: "
LEGACY_DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "整篇要求："


TERM_TRANSLATIONS: Dict[str, str] = {
    "研究范围": "scope",
    "分类框架": "classification framework",
    "临床路径": "clinical pathway",
    "争议焦点": "controversy focus",
    "子群分层": "subgroup stratification",
    "实施障碍": "implementation barriers",
    "证据地图": "evidence map",
    "证据整合": "evidence integration",
    "风险收益": "risk-benefit balance",
    "指南差异": "guideline differences",
    "适应证边界": "indication boundaries",
    "系统适配": "system adaptation",
    "转化意义": "translational significance",
    "开放问题": "open questions",
    "证据缺口": "evidence gaps",
    "研究议程": "research agenda",
    "实践启示": "practice implications",
    "鉴别重点": "diagnostic priorities",
    "比较": "comparison",
    "综合": "synthesis",
    "路径": "pathway",
    "争议": "controversy",
    "子群": "subgroup",
    "实施": "implementation",
    "系统": "system",
    "证据": "evidence",
    "局限性": "limitations",
    "未来工作": "future work",
    "心血管医学": "cardiovascular medicine",
    "肿瘤学": "oncology",
    "血液学": "hematology",
    "神经病学": "neurology",
    "呼吸与重症医学": "pulmonary and critical care medicine",
    "感染病学": "infectious diseases",
    "内分泌与代谢医学": "endocrinology and metabolism",
    "肾脏与泌尿医学": "nephrology and urology",
    "消化与肝胆胰医学": "gastroenterology, hepatology, and pancreatology",
    "风湿免疫医学": "rheumatology and clinical immunology",
    "妇产与生殖医学": "obstetrics, gynecology, and reproductive medicine",
    "儿科与新生儿医学": "pediatrics and neonatology",
    "老年医学": "geriatrics",
    "精神医学": "psychiatry",
    "皮肤病学": "dermatology",
    "眼耳鼻喉口腔医学": "ophthalmology, otolaryngology, and oral medicine",
    "骨科与康复医学": "orthopedics and rehabilitation medicine",
    "外科与围术期医学": "surgery and perioperative medicine",
    "影像与病理医学": "medical imaging and pathology",
    "公共卫生与医学伦理": "public health and medical ethics",
    "急性冠脉综合征": "acute coronary syndrome",
    "心力衰竭": "heart failure",
    "房颤": "atrial fibrillation",
    "高血压": "hypertension",
    "动脉粥样硬化": "atherosclerosis",
    "肺癌": "lung cancer",
    "乳腺癌": "breast cancer",
    "结直肠癌": "colorectal cancer",
    "胰腺癌": "pancreatic cancer",
    "免疫治疗": "immunotherapy",
    "白血病": "leukemia",
    "淋巴瘤": "lymphoma",
    "骨髓瘤": "multiple myeloma",
    "贫血": "anemia",
    "止凝血异常": "bleeding and coagulation disorders",
    "卒中": "stroke",
    "癫痫": "epilepsy",
    "阿尔茨海默病": "Alzheimer's disease",
    "帕金森病": "Parkinson's disease",
    "神经免疫疾病": "neuroimmunologic disorders",
    "ARDS": "ARDS",
    "慢阻肺": "COPD",
    "哮喘": "asthma",
    "肺栓塞": "pulmonary embolism",
    "机械通气": "mechanical ventilation",
    "脓毒症": "sepsis",
    "耐药菌感染": "antimicrobial-resistant infections",
    "病毒性肺炎": "viral pneumonia",
    "结核病": "tuberculosis",
    "真菌感染": "fungal infections",
    "糖尿病": "diabetes",
    "肥胖": "obesity",
    "甲状腺疾病": "thyroid disorders",
    "骨代谢异常": "bone metabolism disorders",
    "代谢综合征": "metabolic syndrome",
    "慢性肾病": "chronic kidney disease",
    "急性肾损伤": "acute kidney injury",
    "肾小球疾病": "glomerular disease",
    "泌尿系肿瘤": "urologic malignancies",
    "透析管理": "dialysis management",
    "肝硬化": "cirrhosis",
    "炎症性肠病": "inflammatory bowel disease",
    "急性胰腺炎": "acute pancreatitis",
    "脂肪肝": "fatty liver disease",
    "消化道出血": "gastrointestinal bleeding",
    "类风湿关节炎": "rheumatoid arthritis",
    "系统性红斑狼疮": "systemic lupus erythematosus",
    "血管炎": "vasculitis",
    "强直性脊柱炎": "ankylosing spondylitis",
    "免疫抑制治疗": "immunosuppressive therapy",
    "妊娠期高血压": "hypertensive disorders of pregnancy",
    "子宫内膜异位症": "endometriosis",
    "不孕症": "infertility",
    "宫颈癌筛查": "cervical cancer screening",
    "围产期管理": "perinatal care",
    "早产儿管理": "preterm infant management",
    "儿童哮喘": "pediatric asthma",
    "遗传代谢病": "inherited metabolic disorders",
    "儿童感染": "pediatric infections",
    "发育行为问题": "developmental and behavioral problems",
    "衰弱": "frailty",
    "多病共存": "multimorbidity",
    "认知障碍": "cognitive impairment",
    "跌倒": "falls",
    "缓和医疗": "palliative care",
    "抑郁障碍": "depressive disorders",
    "双相障碍": "bipolar disorder",
    "精神分裂症": "schizophrenia",
    "睡眠障碍": "sleep disorders",
    "成瘾医学": "addiction medicine",
    "银屑病": "psoriasis",
    "特应性皮炎": "atopic dermatitis",
    "黑色素瘤": "melanoma",
    "皮肤感染": "skin infections",
    "美容与修复": "aesthetic and reconstructive dermatology",
    "青光眼": "glaucoma",
    "糖网病": "diabetic retinopathy",
    "鼻窦炎": "sinusitis",
    "听力损失": "hearing loss",
    "口腔种植": "dental implantology",
    "骨质疏松": "osteoporosis",
    "骨关节炎": "osteoarthritis",
    "运动损伤": "sports injuries",
    "脊柱退变": "degenerative spine disease",
    "卒中后康复": "post-stroke rehabilitation",
    "创伤救治": "trauma care",
    "微创外科": "minimally invasive surgery",
    "围术期优化": "perioperative optimization",
    "术后并发症": "postoperative complications",
    "感染控制": "infection control",
    "胸部影像": "thoracic imaging",
    "分子影像": "molecular imaging",
    "数字病理": "digital pathology",
    "介入放射": "interventional radiology",
    "AI辅助诊断": "AI-assisted diagnosis",
    "疫苗策略": "vaccine strategy",
    "慢病防控": "chronic disease prevention and control",
    "卫生技术评估": "health technology assessment",
    "药物监管": "drug regulation",
    "临床研究伦理": "clinical research ethics",
    "血流动力学": "hemodynamics",
    "风险分层": "risk stratification",
    "危险分层": "risk stratification",
    "二级预防": "secondary prevention",
    "分期": "staging",
    "精准治疗": "precision medicine",
    "真实世界证据": "real-world evidence",
    "分层治疗": "risk-adapted therapy",
    "出血风险": "bleeding risk",
    "复发监测": "relapse monitoring",
    "神经功能评估": "neurologic functional assessment",
    "影像整合": "imaging integration",
    "长期预后": "long-term prognosis",
    "呼吸支持": "respiratory support",
    "重症分层": "critical care stratification",
    "器官保护": "organ protection",
    "病原学证据": "microbiologic evidence",
    "抗菌药物管理": "antimicrobial stewardship",
    "传播控制": "transmission control",
    "代谢通路": "metabolic pathways",
    "并发症防控": "complication prevention and control",
    "生活方式干预": "lifestyle intervention",
    "肾功能分层": "kidney function stratification",
    "液体管理": "fluid management",
    "肾替代治疗": "kidney replacement therapy",
    "肠道微环境": "gut microenvironment",
    "并发症预警": "complication warning",
    "循证治疗": "evidence-based therapy",
    "自身免疫机制": "autoimmune mechanisms",
    "器官受累": "organ involvement",
    "疾病活动度": "disease activity",
    "母胎安全": "maternal-fetal safety",
    "生殖结局": "reproductive outcomes",
    "风险干预": "risk intervention",
    "生长发育": "growth and development",
    "年龄分层": "age stratification",
    "家庭参与": "family involvement",
    "功能状态": "functional status",
    "照护路径": "care pathway",
    "综合评估": "comprehensive assessment",
    "症状维度": "symptom dimensions",
    "药物与心理治疗": "pharmacologic and psychotherapeutic treatment",
    "社会功能": "social functioning",
    "屏障功能": "barrier function",
    "炎症通路": "inflammatory pathways",
    "长期管理": "long-term management",
    "器官功能保护": "organ function preservation",
    "影像评估": "imaging assessment",
    "微创干预": "minimally invasive intervention",
    "功能重建": "functional reconstruction",
    "运动处方": "exercise prescription",
    "疼痛管理": "pain management",
    "手术时机": "timing of surgery",
    "风险评估": "risk assessment",
    "恢复增强": "enhanced recovery",
    "多模态证据": "multimodal evidence",
    "判读一致性": "reading consistency",
    "质量控制": "quality control",
    "人群健康": "population health",
    "政策评估": "policy assessment",
    "公平性": "equity",
    "机制": "mechanism",
    "自然史": "natural history",
    "诊断路径": "diagnostic pathway",
    "严重度评估": "severity assessment",
    "治疗策略": "treatment strategy",
    "预后": "prognosis",
    "指南比较": "guideline comparison",
    "卫生系统": "health system",
    "成人住院": "adult inpatient",
    "儿科": "pediatrics",
    "老年": "older adults",
    "急诊重症": "emergency and critical care",
    "基层门诊": "primary care and outpatient",
    "围术期": "perioperative",
    "低资源环境": "low-resource settings",
    "数字健康": "digital health",
}

PERIODIC_REQUIREMENT_TRANSLATIONS: Dict[str, str] = {
    "主体部分每隔两段需要出现一次比较或综合判断": "The main body should include an explicit comparison or synthesis at least once every two paragraphs.",
    "后半部分必须持续讨论局限性、证据缺口与未来工作": "The second half of the article must keep discussing limitations, evidence gaps, and future work.",
    "正文中段应周期性回到比较与综合，不得只做定义堆叠": "The middle sections should repeatedly return to comparison and synthesis rather than stacking definitions.",
    "末段之前至少有一段专门讨论局限性与开放问题": "Before the final paragraph, dedicate at least one full paragraph to limitations and open questions.",
    "中段需要多次回到临床路径的比较与重组": "The middle sections should revisit the comparison and reconfiguration of clinical pathways multiple times.",
    "结尾前必须讨论局限性与未来工作": "Before the conclusion, the article must discuss limitations and future work.",
    "主体部分应反复比较不同临床路径的取舍逻辑": "The main body should repeatedly compare the trade-offs among different clinical pathways.",
    "后半部分必须说明局限性和未来工作": "The second half of the article must explain the limitations and future work.",
    "正文中段需要周期性解释争议焦点与证据冲突": "The middle sections should periodically explain the controversy focus and evidence conflicts.",
    "收束部分必须讨论局限性、证据缺口与未来工作": "The closing sections must discuss limitations, evidence gaps, and future work.",
    "主体段落应反复回到争议焦点，不得只做单方立场陈述": "The main paragraphs should repeatedly return to the controversy focus rather than presenting only one-sided positions.",
    "最后三分之一部分要交代局限性、证据缺口和未来工作": "The final third of the article should address limitations, evidence gaps, and future work.",
    "主体部分需要定期回到子群分层和场景差异": "The main body should regularly return to subgroup stratification and scenario-specific differences.",
    "结尾必须讨论局限性与未来工作": "The ending must discuss limitations and future work.",
    "中段段落应持续比较不同子群的异同": "The middle paragraphs should continue comparing similarities and differences across subgroups.",
    "收束部分要说明局限性与未来工作": "The closing sections should explain limitations and future work.",
    "正文应周期性回到实施障碍与系统适配问题": "The article should periodically return to implementation barriers and system adaptation issues.",
    "后半部分必须同时说明局限性、证据缺口与未来工作": "The second half of the article must explain limitations, evidence gaps, and future work together.",
    "主体分析需要反复处理实施障碍与场景适配，而不是只总结理想路径": "The main analysis should repeatedly address implementation barriers and contextual adaptation rather than summarizing only ideal pathways.",
    "结尾前必须讨论局限性与证据缺口": "Before the conclusion, the article must discuss limitations and evidence gaps.",
    "主体段落需要反复连接证据地图与综合判断": "The main paragraphs should repeatedly connect the evidence map to synthesis-driven judgments.",
    "结尾应明确写出局限性、研究议程与未来工作": "The ending should explicitly state limitations, the research agenda, and future work.",
    "中段需要持续回到证据地图和证据空白": "The middle sections should repeatedly return to the evidence map and evidence gaps.",
    "最后两段必须保留局限性和未来工作": "The final two paragraphs must preserve explicit discussion of limitations and future work.",
}

CONTEXT_DETAIL_TRANSLATIONS: Dict[str, str] = {
    "adult inpatient": "adult inpatient care pathways",
    "pediatrics": "care needs in children and adolescents",
    "older adults": "multimorbidity and frailty management in older adults",
    "emergency and critical care": "emergency triage and critical-care resource allocation",
    "primary care and outpatient": "primary-care delivery and outpatient follow-up pathways",
    "perioperative": "perioperative risk control and recovery optimization",
    "low-resource settings": "implementation feasibility in resource-limited settings",
    "digital health": "digital tools, remote monitoring, and data governance",
}

ORGANIZER_CANDIDATES_EN = [
    "classification framework",
    "clinical pathway",
    "controversy focus",
    "subgroup stratification",
    "implementation barriers",
    "evidence map",
]

CLOSING_CANDIDATES_EN = [
    "evidence gaps",
    "open questions",
    "research agenda",
    "future work",
]


def is_document_level_constraint(requirement: str) -> bool:
    """判断约束是否标记为整篇级要求。"""
    return requirement.startswith(
        (DOCUMENT_LEVEL_CONSTRAINT_PREFIX, LEGACY_DOCUMENT_LEVEL_CONSTRAINT_PREFIX)
    )


def _translate_term(text: str) -> str:
    """将 benchmark 术语从中文转换为英文。"""
    normalized = text.strip()
    if normalized in TERM_TRANSLATIONS:
        return TERM_TRANSLATIONS[normalized]
    if re.search(r"[\u4e00-\u9fff]", normalized):
        raise KeyError(f"Missing English translation for benchmark term: {normalized}")
    return normalized


def _translate_periodic_requirement(text: str) -> str:
    """将周期性要求转换为英文。"""
    normalized = text.strip()
    if normalized in PERIODIC_REQUIREMENT_TRANSLATIONS:
        return PERIODIC_REQUIREMENT_TRANSLATIONS[normalized]
    if re.search(r"[\u4e00-\u9fff]", normalized):
        raise KeyError(
            f"Missing English translation for benchmark periodic requirement: {normalized}"
        )
    return normalized


def _translate_checklist_item(item: str) -> str:
    """将 checklist 文案转换为英文。"""
    patterns = [
        (
            r"^是否说明了(.+)的研究范围与问题边界$",
            lambda m: f"Does the article define the scope and problem boundary for {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否以(.+)组织(.+)中的材料$",
            lambda m: f"Does the article organize material in {_translate_term(m.group(2))} around a {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否围绕(.+)进行横向比较与综合$",
            lambda m: f"Does the article compare and synthesize the literature around {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否纳入(.+)相关的实践约束$",
            lambda m: f"Does the article incorporate practice constraints related to {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否讨论局限性、(.+)与未来工作$",
            lambda m: f"Does the article discuss limitations, {_translate_term(m.group(1))}, and future work?",
        ),
        (
            r"^文稿前部是否交代(.+)讨论范围与术语口径$",
            lambda m: f"Does the opening section define the discussion scope and terminology for {_translate_term(m.group(1))}?",
        ),
        (
            r"^正文是否把(.+)作为主组织轴$",
            lambda m: f"Does the main body use {_translate_term(m.group(1))} as the main organizing axis?",
        ),
        (
            r"^是否处理(.+)与(.+)之间的关系$",
            lambda m: f"Does the article address the relationship between {_translate_term(m.group(1))} and {_translate_term(m.group(2))}?",
        ),
        (
            r"^是否结合(.+)讨论现实应用问题$",
            lambda m: f"Does the article discuss real-world implementation issues in the context of {_translate_term(m.group(1))}?",
        ),
        (
            r"^结尾是否保留局限性和未来工作两个清晰板块$",
            lambda _m: "Does the conclusion keep separate sections for limitations and future work?",
        ),
        (
            r"^是否完成(.+)的范围界定$",
            lambda m: f"Does the article define the scope of {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否提出可解释的(.+)$",
            lambda m: f"Does the article present an interpretable {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否比较不同路径在(.+)上的差异$",
            lambda m: f"Does the article compare differences across alternative pathways in terms of {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否呈现(.+)下的情境化判断$",
            lambda m: f"Does the article provide context-sensitive judgments for {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否明确说明局限性与未来工作$",
            lambda _m: "Does the article explicitly discuss limitations and future work?",
        ),
        (
            r"^是否交代(.+)$",
            lambda m: f"Does the article address {_translate_term(m.group(1))}?",
        ),
    ]

    for pattern, renderer in patterns:
        match = re.match(pattern, item)
        if match:
            return renderer(match)

    if re.search(r"[\u4e00-\u9fff]", item):
        raise KeyError(f"Missing English translation for checklist item: {item}")
    return item


def _translate_proxy_question(question: str) -> str:
    """将 proxy question 文案转换为英文。"""
    patterns = [
        (
            r"^文中是否说明了研究范围？$",
            lambda _m: "Does the article explain the scope?",
        ),
        (
            r"^文中是否呈现了(.+)？$",
            lambda m: f"Does the article present the {_translate_term(m.group(1))}?",
        ),
        (
            r"^文中是否讨论了局限性？$",
            lambda _m: "Does the article discuss limitations?",
        ),
        (
            r"^文中是否提出了未来工作？$",
            lambda _m: "Does the article propose future work?",
        ),
        (
            r"^开头是否完成研究范围界定？$",
            lambda _m: "Does the opening section define the scope?",
        ),
        (
            r"^作者是否用(.+)组织内容？$",
            lambda m: f"Does the author organize the content around {_translate_term(m.group(1))}?",
        ),
        (
            r"^是否讨论了(.+)？$",
            lambda m: f"Does the article discuss {_translate_term(m.group(1))}?",
        ),
        (
            r"^文章有没有先说明研究范围？$",
            lambda _m: "Does the article state the scope upfront?",
        ),
        (
            r"^文章有没有进行比较？$",
            lambda _m: "Does the article make explicit comparisons?",
        ),
        (
            r"^文章有没有说明局限性？$",
            lambda _m: "Does the article explain limitations?",
        ),
        (
            r"^文章有没有涉及(.+)？$",
            lambda m: f"Does the article address {_translate_term(m.group(1))}?",
        ),
        (
            r"^文章有没有提出未来工作？$",
            lambda _m: "Does the article propose future work?",
        ),
    ]

    for pattern, renderer in patterns:
        match = re.match(pattern, question)
        if match:
            return renderer(match)

    if re.search(r"[\u4e00-\u9fff]", question):
        raise KeyError(f"Missing English translation for proxy question: {question}")
    return question


def _infer_benchmark_semantics(
    must_include: List[str],
    range_keywords: List[Dict[str, object]],
) -> Dict[str, str]:
    """从 benchmark 锚点中恢复英文任务描述所需的语义槽位。"""
    if len(must_include) < 8:
        raise ValueError("benchmark must_include must contain at least 8 items")

    extras = must_include[8:]
    closing_candidates = [
        str(item["keyword"])
        for item in range_keywords
        if isinstance(item, dict) and str(item.get("keyword")) in {"未来工作", "开放问题", "证据缺口", "研究议程"}
    ]

    closing_anchor = next(
        (
            item
            for item in extras + closing_candidates
            if item in {"未来工作", "开放问题", "证据缺口", "研究议程"}
        ),
        "未来工作",
    )
    bridge_anchor = next(
        (
            item
            for item in extras
            if item
            in {"证据整合", "风险收益", "指南差异", "适应证边界", "系统适配", "转化意义"}
        ),
        "证据整合",
    )

    return {
        "organizer": _translate_term(must_include[1]),
        "domain": _translate_term(must_include[2]),
        "subtopic": _translate_term(must_include[3]),
        "focus": _translate_term(must_include[4]),
        "context": _translate_term(must_include[5]),
        "evidence": _translate_term(must_include[6]),
        "bridge": _translate_term(bridge_anchor),
        "closing": _translate_term(closing_anchor),
    }


def _build_english_task_description(
    task_type: str,
    required_length_words: int,
    expected_blocks: int,
    must_include: List[str],
    range_keywords: List[Dict[str, object]],
) -> str:
    """基于 benchmark 结构信息重建英文任务描述。"""
    semantics = _infer_benchmark_semantics(must_include, range_keywords)
    doc_type = (
        "comparative medical review"
        if task_type == "writing"
        else "medical review article"
    )
    context_detail = CONTEXT_DETAIL_TRANSLATIONS.get(
        semantics["context"],
        f"{semantics['context']} practice context",
    )
    closing_phrase = (
        "future work"
        if semantics["closing"] == "future work"
        else f"{semantics['closing']}, and future work"
    )

    return " ".join(
        [
            f"Write an approximately {required_length_words}-word {doc_type} in English on {semantics['subtopic']} within {semantics['domain']}.",
            "Keep the piece as a long-form scholarly review rather than a case report, a popular-science summary, or an outline.",
            f"Open by defining the scope, discussion boundaries, and terminology, while keeping {semantics['context']} as an explicit practice context.",
            f"Organize the main body around a {semantics['organizer']} and keep {context_detail} on the main line of discussion.",
            f"Compare and synthesize the literature around {semantics['focus']}, {semantics['evidence']}, and {semantics['bridge']}.",
            f"Develop at least {expected_blocks} natural paragraphs and close with limitations, {closing_phrase}.",
        ]
    )


def _mark_document_level_constraint(requirement: str) -> str:
    """给整篇级 benchmark 约束打上统一前缀。

    设计目的：
        benchmark 样本里的字数、整体结构、must_include 和周期性要求本质上都是
        “整篇输出”约束，而不是单个 section 的即时校验条件。
        这里显式加前缀，是为了让生成主流程仍然能看到这些要求，同时让
        section 级 validator 可以稳定识别并跳过误判。
    """
    return f"{DOCUMENT_LEVEL_CONSTRAINT_PREFIX}{requirement}"


def _extract_paragraph_blocks(text: str) -> List[str]:
    """从最终生成文本中提取可用于 benchmark 评估的正文段落。

    设计目的：
        Meta-Writer 会在最终输出里插入 `##` 标题和 `---` 分隔线来保留章节结构。
        如果直接按非空行切段，这些装配标记会污染段落计数和位置约束判断，
        导致 benchmark 评分被 markdown 外壳而不是正文内容干扰。
    """
    normalized_text = text.replace("\r\n", "\n").strip()
    if normalized_text == "":
        return []

    paragraph_blocks: List[str] = []
    for raw_block in normalized_text.split("\n\n"):
        block = raw_block.strip()
        if block == "" or block == "---":
            continue

        if block.startswith("## "):
            continue

        paragraph_blocks.append(block)
    return paragraph_blocks


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """读取 JSONL 文件。

    参数：
        file_path: JSONL 文件路径。

    返回值：
        List[Dict[str, object]]：逐行解析后的字典列表。

    关键实现细节：
        遇到空行时跳过，避免文稿整理后出现格式噪声导致加载失败。
    """
    if not file_path.exists():
        raise FileNotFoundError(f"未找到 benchmark 样本文件：{file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        rows.append(json.loads(stripped_line))
    return rows


def list_benchmark_task_ids() -> List[str]:
    """列出当前可用的 benchmark 任务 ID。

    参数：
        无。

    返回值：
        List[str]：按文件顺序返回样本 ID 列表。

    关键实现细节：
        直接复用本地 JSONL 解析结果，确保注册表与样本源始终一致。
    """
    task_ids: List[str] = []
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        sample_id = str(sample_row["sample_id"])
        if sample_id == "":
            raise ValueError("benchmark sample_id 不能为空")
        task_ids.append(sample_id)
    return task_ids


def _build_outline(task_type: str, must_include: List[str]) -> Dict[str, str]:
    """根据任务类型与锚点生成最小可用大纲。

    参数：
        task_type: benchmark 样本中的任务类型。
        must_include: 当前样本的核心锚点列表。

    返回值：
        Dict[str, str]：适配 Meta-Writer 的 section 大纲。

    关键实现细节：
        保持固定三段式接口，但根据样本中的主组织锚点动态改变二级结构语义，
        避免所有医学任务暴露完全相同的 outline。
    """
    organizer_candidates = [
        item for item in ORGANIZER_CANDIDATES_EN if item in must_include
    ]
    closing_candidates = [
        item for item in CLOSING_CANDIDATES_EN if item in must_include
    ]
    organizer_label = (
        organizer_candidates[0] if organizer_candidates else "core organizing axis"
    )
    closing_label = closing_candidates[0] if closing_candidates else "future work"
    if task_type == "analysis":
        return {
            "sec1": "Scope, core concepts, and analytical perspective",
            "sec2": f"Comparative and synthetic analysis organized around the {organizer_label}",
            "sec3": f"Limitations, {closing_label}, and closing judgment",
        }
    return {
        "sec1": "Background, problem framing, and review scope",
        "sec2": f"Comparison, integration, and discussion under the {organizer_label}",
        "sec3": f"Reflections on limitations, {closing_label}, and forward-looking discussion",
    }


def _normalize_reference(reference: Union[str, Dict[str, object]]) -> Dict[str, object]:
    """规范化评估参考信息。

    参数：
        reference: 结构化参考字典，或 JSON 字符串形式的参考信息。

    返回值：
        Dict[str, object]：统一后的参考信息字典。

    关键实现细节：
        若传入字符串，则必须是合法 JSON，避免默认值掩盖字段缺失问题。
    """
    if isinstance(reference, dict):
        return reference
    if isinstance(reference, str):
        parsed_reference = json.loads(reference)
        if not isinstance(parsed_reference, dict):
            raise ValueError("reference JSON 必须解析为字典")
        return parsed_reference
    raise TypeError("reference 必须是 Dict[str, object] 或 JSON 字符串")


def _contains_keyword(text: str, keyword: str) -> bool:
    """判断文本中是否包含关键字。

    参数：
        text: 待检查文本。
        keyword: 关键字。

    返回值：
        bool：是否命中。

    关键实现细节：
        统一转为小写，兼容中英文字段混写场景。
    """
    return keyword.lower() in text.lower()


def _parse_int_field(raw_value: object, field_name: str) -> int:
    """解析并校验整数字段。

    参数：
        raw_value: 原始字段值。
        field_name: 字段名称。

    返回值：
        int：解析后的整数值。

    关键实现细节：
        仅接受整数或整数字符串，避免在 benchmark 配置加载时吞掉结构错误。
    """
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} 必须是整数或整数字符串")
    return int(raw_value)


def load_benchmark_task(task_id: str) -> Dict[str, object]:
    """从本地 benchmark 样本加载任务。

    参数：
        task_id: benchmark 中的任务标识符。

    返回值：
        Dict[str, object]：
            - task: 任务描述
            - constraints: 约束列表
            - outline: 章节大纲
            - reference: 评估参考信息

    关键实现细节：
        直接从 `metabench/examples/samples.jsonl` 检索匹配样本，并将约束压平为
        Meta-Writer 能直接消费的文本约束列表，同时显式加入综述型长文风格提示。
    """
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        if str(sample_row.get("sample_id")) != task_id:
            continue

        raw_constraints_object = sample_row["constraints"]
        if not isinstance(raw_constraints_object, dict):
            raise TypeError("constraints 必须是字典")
        raw_constraints = cast(Dict[str, object], raw_constraints_object)

        must_include_object = raw_constraints["must_include"]
        if not isinstance(must_include_object, list):
            raise TypeError("constraints.must_include 必须是列表")
        must_include = [str(item) for item in must_include_object]

        periodic_requirements_object = raw_constraints["periodic_requirements"]
        if not isinstance(periodic_requirements_object, list):
            raise TypeError("constraints.periodic_requirements 必须是列表")
        periodic_requirements = [str(item) for item in periodic_requirements_object]

        required_length_words = _parse_int_field(
            raw_constraints["required_length_words"],
            "constraints.required_length_words",
        )
        expected_blocks = _parse_int_field(
            raw_constraints["expected_blocks"],
            "constraints.expected_blocks",
        )

        range_keywords_object = raw_constraints["range_keywords"]
        if not isinstance(range_keywords_object, list):
            raise TypeError("constraints.range_keywords 必须是列表")
        range_keywords_zh = [
            dict(item) if isinstance(item, dict) else {"keyword": str(item)}
            for item in range_keywords_object
        ]

        periodic_keywords_object = raw_constraints["periodic_keywords"]
        if not isinstance(periodic_keywords_object, list):
            raise TypeError("constraints.periodic_keywords 必须是列表")
        periodic_keywords_zh = [
            dict(item) if isinstance(item, dict) else {"keyword": str(item)}
            for item in periodic_keywords_object
        ]
        task_type = str(sample_row["task_type"])

        proxy_questions_object = sample_row["proxy_questions"]
        if not isinstance(proxy_questions_object, list):
            raise TypeError("proxy_questions 必须是列表")
        checklist_object = sample_row["checklist"]
        if not isinstance(checklist_object, list):
            raise TypeError("checklist 必须是列表")
        proxy_questions_list = cast(List[object], proxy_questions_object)
        checklist_list = cast(List[object], checklist_object)

        must_include_en = [_translate_term(item) for item in must_include]
        periodic_requirements_en = [
            _translate_periodic_requirement(item) for item in periodic_requirements
        ]
        range_keywords = [
            {**item, "keyword": _translate_term(str(item["keyword"]))}
            for item in range_keywords_zh
        ]
        periodic_keywords = [
            {**item, "keyword": _translate_term(str(item["keyword"]))}
            for item in periodic_keywords_zh
        ]
        task_description = _build_english_task_description(
            task_type=task_type,
            required_length_words=required_length_words,
            expected_blocks=expected_blocks,
            must_include=must_include,
            range_keywords=range_keywords_zh,
        )

        constraints: List[str] = [
            _mark_document_level_constraint("Write the entire article in English."),
            _mark_document_level_constraint(
                f"Target length is about {required_length_words} words."
            ),
            _mark_document_level_constraint(
                f"The body should contain at least {expected_blocks} natural paragraphs and read like a long-form review rather than a brief or outline."
            ),
            _mark_document_level_constraint(
                "Use a survey-paper style: define the scope first, then organize, compare, and synthesize the evidence before closing with limitations and future work."
            ),
            *[
                _mark_document_level_constraint(f"Must explicitly cover: {item}.")
                for item in must_include_en
            ],
            *[
                _mark_document_level_constraint(f"Periodic requirement: {item}")
                for item in periodic_requirements_en
            ],
        ]

        reference: Dict[str, object] = {
            "sample_id": str(sample_row["sample_id"]),
            "task_type": task_type,
            "language": "en",
            "prompt": task_description,
            "constraints": {
                "required_length_words": required_length_words,
                "must_include": must_include_en,
                "periodic_requirements": periodic_requirements_en,
                "expected_blocks": expected_blocks,
                "range_keywords": range_keywords,
                "periodic_keywords": periodic_keywords,
            },
            "proxy_questions": [
                {
                    "qid": str(question["qid"]),
                    "question": _translate_proxy_question(str(question["question"])),
                    "answer": _translate_term(str(question["answer"])),
                }
                for question in proxy_questions_list
                if isinstance(question, dict)
            ],
            "checklist": [_translate_checklist_item(str(item)) for item in checklist_list],
        }

        return {
            "task": task_description,
            "constraints": constraints,
            "outline": _build_outline(task_type, must_include_en),
            "reference": reference,
        }

    raise ValueError(f"未找到 benchmark 任务：{task_id}")


def evaluate_output(
    generated_text: str, reference: Union[str, Dict[str, object]]
) -> Dict[str, object]:
    """评估生成结果。

    参数：
        generated_text: Meta-Writer 生成的文本。
        reference: benchmark 任务参考信息。

    返回值：
        Dict[str, object]：
            - constraint_violation_rate: 约束违反率
            - entity_consistency_score: 关键实体一致性评分
            - logical_coherence: 逻辑连贯性评分
            - diagnostics: 详细诊断信息

    关键实现细节：
        采用本地可复现的启发式规则，不依赖外部 judge 模型，便于 example 接口直连跑通。
        在原有关键词命中基础上，额外利用段落数量、位置约束与周期性关键词，
        对综述型长文的结构信号进行软性评分。
    """
    normalized_reference = _normalize_reference(reference)
    normalized_text = generated_text.strip()
    if normalized_text == "":
        raise ValueError("generated_text 不能为空")

    raw_constraints_object = normalized_reference["constraints"]
    if not isinstance(raw_constraints_object, dict):
        raise TypeError("reference.constraints 必须是字典")
    raw_constraints = cast(Dict[str, object], raw_constraints_object)

    must_include_object = raw_constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("reference.constraints.must_include 必须是列表")
    must_include = [str(item) for item in must_include_object]

    expected_blocks = _parse_int_field(
        raw_constraints["expected_blocks"],
        "reference.constraints.expected_blocks",
    )

    range_keywords_object = raw_constraints["range_keywords"]
    if not isinstance(range_keywords_object, list):
        raise TypeError("reference.constraints.range_keywords 必须是列表")
    range_keywords = [
        dict(item) for item in range_keywords_object if isinstance(item, dict)
    ]

    periodic_keywords_object = raw_constraints["periodic_keywords"]
    if not isinstance(periodic_keywords_object, list):
        raise TypeError("reference.constraints.periodic_keywords 必须是列表")
    periodic_keywords = [
        dict(item) for item in periodic_keywords_object if isinstance(item, dict)
    ]

    checklist_object = normalized_reference["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("reference.checklist 必须是列表")
    checklist_list = cast(List[object], checklist_object)
    checklist = [str(item) for item in checklist_list]

    proxy_questions_object = normalized_reference["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("reference.proxy_questions 必须是列表")
    proxy_questions_list = cast(List[object], proxy_questions_object)
    proxy_questions = [
        dict(item) for item in proxy_questions_list if isinstance(item, dict)
    ]

    matched_keywords = [
        item for item in must_include if _contains_keyword(normalized_text, item)
    ]
    matched_proxy_answers = [
        str(item["qid"])
        for item in proxy_questions
        if _contains_keyword(normalized_text, str(item["answer"]))
    ]
    paragraph_blocks = _extract_paragraph_blocks(normalized_text)
    body_text = "\n\n".join(paragraph_blocks)
    sentence_parts = re.split(r"[.!?。！？]+", body_text)
    sentence_count = sum(1 for part in sentence_parts if part.strip() != "")

    if len(must_include) == 0:
        raise ValueError("reference.constraints.must_include 不能为空")
    if len(proxy_questions) == 0:
        raise ValueError("reference.proxy_questions 不能为空")

    range_keyword_hits: List[str] = []
    missing_range_keywords: List[str] = []
    for item in range_keywords:
        keyword = str(item["keyword"])
        start_index = max(1, int(item["start"])) - 1
        end_index = min(len(paragraph_blocks), int(item["end"]))
        candidate_blocks = paragraph_blocks[start_index:end_index]
        if any(_contains_keyword(block, keyword) for block in candidate_blocks):
            range_keyword_hits.append(keyword)
        else:
            missing_range_keywords.append(keyword)

    periodic_keyword_hits: List[str] = []
    missing_periodic_keywords: List[str] = []
    for item in periodic_keywords:
        keyword = str(item["keyword"])
        every_value = int(item["every"])
        start_paragraph = max(1, int(item["start"]))
        if every_value <= 0:
            raise ValueError("periodic_keywords.every 必须为正整数")

        target_hit_count = 0
        current_paragraph = start_paragraph
        while current_paragraph <= len(paragraph_blocks):
            target_hit_count += 1
            current_paragraph += every_value

        actual_hit_count = sum(
            1
            for block in paragraph_blocks[start_paragraph - 1 :]
            if _contains_keyword(block, keyword)
        )
        if actual_hit_count >= target_hit_count and target_hit_count > 0:
            periodic_keyword_hits.append(keyword)
        else:
            missing_periodic_keywords.append(keyword)

    entity_consistency_score = len(matched_keywords) / len(must_include)
    proxy_hit_rate = len(matched_proxy_answers) / len(proxy_questions)
    paragraph_signal = min(1.0, len(paragraph_blocks) / expected_blocks)
    sentence_signal = 1.0 if sentence_count >= expected_blocks * 2 else 0.5
    range_signal = (
        len(range_keyword_hits) / len(range_keywords) if len(range_keywords) > 0 else 1.0
    )
    periodic_signal = (
        len(periodic_keyword_hits) / len(periodic_keywords)
        if len(periodic_keywords) > 0
        else 1.0
    )
    structure_signal = min(
        1.0,
        0.35 * paragraph_signal
        + 0.2 * sentence_signal
        + 0.25 * range_signal
        + 0.2 * periodic_signal,
    )
    checklist_signal = len(matched_keywords) / max(len(checklist), len(must_include))
    logical_coherence = min(
        1.0, 0.5 * structure_signal + 0.3 * proxy_hit_rate + 0.2 * checklist_signal
    )
    constraint_violation_rate = 1.0 - entity_consistency_score

    return {
        "constraint_violation_rate": constraint_violation_rate,
        "entity_consistency_score": entity_consistency_score,
        "logical_coherence": logical_coherence,
        "diagnostics": {
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                item for item in must_include if item not in matched_keywords
            ],
            "matched_proxy_question_ids": matched_proxy_answers,
            "paragraph_count": len(paragraph_blocks),
            "sentence_count": sentence_count,
            "expected_blocks": expected_blocks,
            "range_keyword_hits": range_keyword_hits,
            "missing_range_keywords": missing_range_keywords,
            "periodic_keyword_hits": periodic_keyword_hits,
            "missing_periodic_keywords": missing_periodic_keywords,
            "checklist": checklist,
        },
    }


def build_benchmark_task_config(task_id: str) -> Dict[str, object]:
    """构造可注册到 Meta-Writer 的 benchmark 任务配置。

    参数：
        task_id: benchmark 样本 ID。

    返回值：
        Dict[str, object]：包含任务、约束、大纲、参考信息与会话名。

    关键实现细节：
        统一在这里生成 `session_name`，避免不同 example 文件重复拼接命名逻辑。
    """
    benchmark_task = load_benchmark_task(task_id)
    constraints_object = benchmark_task["constraints"]
    if not isinstance(constraints_object, list):
        raise TypeError("benchmark_task.constraints 必须是列表")
    outline_object = benchmark_task["outline"]
    if not isinstance(outline_object, dict):
        raise TypeError("benchmark_task.outline 必须是字典")
    reference_object = benchmark_task["reference"]
    if not isinstance(reference_object, dict):
        raise TypeError("benchmark_task.reference 必须是字典")
    constraints_list = cast(List[object], constraints_object)
    outline_dict = cast(Dict[str, object], outline_object)
    reference_dict = cast(Dict[str, object], reference_object)

    return {
        "task": str(benchmark_task["task"]),
        "constraints": list(constraints_list),
        "outline": dict(outline_dict),
        "reference": dict(reference_dict),
        "language": "en",
        "session_name": f"metabench_{task_id}",
    }
