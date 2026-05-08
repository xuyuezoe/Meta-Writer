"""English benchmark adapter for MetaWriter.

This module loads benchmark samples from the local `metabench/examples/samples.jsonl`
bundle, converts them into task configs that MetaWriter can run directly, and
provides a lightweight local evaluator that does not depend on an external judge.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Union, cast


BENCHMARK_ROOT: Path = Path(__file__).resolve().parent.parent / "metabench"
METABENCH_SRC_ROOT: Path = BENCHMARK_ROOT / "src"
SAMPLES_PATH: Path = BENCHMARK_ROOT / "examples" / "samples.jsonl"
DOCUMENT_LEVEL_CONSTRAINT_PREFIX = "Document-level requirement: "

if str(METABENCH_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(METABENCH_SRC_ROOT))

from metabench.local_metrics import (
    compute_acc_once,
    compute_acc_periodic,
    compute_acc_range,
    compute_completion_rate,
    compute_instruction_hits,
    compute_proxy_qa,
    compute_structure_scores,
    contains_keyword,
    count_length_units,
    split_blocks,
)
from metabench.scoring import compute_s_length

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


def _contains_cjk(text: str) -> bool:
    """Return True when the text still contains Chinese characters."""
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def is_document_level_constraint(requirement: str) -> bool:
    """Return whether the constraint is tagged as document-level."""
    return requirement.startswith(DOCUMENT_LEVEL_CONSTRAINT_PREFIX)


def _mark_document_level_constraint(requirement: str) -> str:
    """Prefix a document-level benchmark requirement for the validator layer."""
    return f"{DOCUMENT_LEVEL_CONSTRAINT_PREFIX}{requirement}"


def _extract_paragraph_blocks(text: str) -> List[str]:
    """Extract body paragraphs while skipping MetaWriter assembly wrappers."""
    normalized_text = text.replace("\r\n", "\n").strip()
    if normalized_text == "":
        return []
    return split_blocks(normalized_text, drop_markdown_wrappers=True)


def _extract_section_blocks(text: str) -> List[str]:
    """
    按 ## 标题分割文本，每个 section 作为一个整体块返回（标题文本 + 正文内容合并）。

    设计原理：
        benchmark 的 range_keyword 位置约束本质上是 section 级的语义约束——
        "scope" 应在第一节，"future work" 应在最后一节。
        但 split_blocks 按段落分割，当每节包含多段时块数远超 expected_blocks，
        导致绝对位置约束失效。
        本函数将每节（## 标题 + 其下正文）视为一个语义单元，
        使 range_keyword 位置检查与 section 序号精确对齐。

    返回：
        List[str]：每项 = "标题文本\n\n正文内容"，长度等于 ## 标题数量。
        若无 ## 标题则返回空列表（调用方应回退到 _extract_paragraph_blocks）。
    """
    normalized = text.replace("\r\n", "\n").strip()
    # 找到所有 ## 标题行的位置
    heading_re = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)
    matches = list(heading_re.finditer(normalized))
    if not matches:
        return []

    # 已知非内容标题集合：参考文献列表等不计入 expected_blocks
    _NON_CONTENT_HEADINGS = {"references", "bibliography", "参考文献"}

    blocks: List[str] = []
    for i, m in enumerate(matches):
        title_text = m.group(1).strip()
        if title_text.lower() in _NON_CONTENT_HEADINGS:
            continue
        # 正文：从标题行结束到下一个标题行开始
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(normalized)
        body = normalized[body_start:body_end].strip()
        # 过滤掉 --- 分隔符
        body = re.sub(r"^---\s*$", "", body, flags=re.MULTILINE).strip()
        block = f"{title_text}\n\n{body}" if body else title_text
        blocks.append(block)

    return blocks


def _get_evaluation_blocks(text: str, expected_blocks: int) -> List[str]:
    """
    选择评估粒度：
        1. 若文本含 ## 标题且数量 == expected_blocks → section 级（精确语义匹配）
        2. 否则 → 段落级（兜底，兼容旧格式文本）

    section 级评估的优点：
        - range_keyword 位置 = section 位置，语义正确
        - 无论 section 内部有多少段落，都计为 1 个语义单元
    """
    section_blocks = _extract_section_blocks(text)
    if len(section_blocks) == expected_blocks:
        return section_blocks
    return _extract_paragraph_blocks(text)


def _read_jsonl_rows(file_path: Path) -> List[Dict[str, object]]:
    """Read a JSONL file into a list of dictionaries."""
    if not file_path.exists():
        raise FileNotFoundError(f"Benchmark sample file not found: {file_path}")

    rows: List[Dict[str, object]] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        stripped_line = raw_line.strip()
        if stripped_line == "":
            continue
        rows.append(json.loads(stripped_line))
    return rows


def list_benchmark_task_ids() -> List[str]:
    """Return benchmark task IDs in file order."""
    task_ids: List[str] = []
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        sample_id = str(sample_row["sample_id"])
        if sample_id == "":
            raise ValueError("benchmark sample_id must not be empty")
        task_ids.append(sample_id)
    return task_ids


def _build_outline(
    task_type: str,
    must_include: List[str],
    expected_blocks: int = 7,
) -> Dict[str, str]:
    """
    动态构建恰好 expected_blocks 节的大纲，与 benchmark range_keyword 位置对齐。

    设计原则：
        benchmark 的 range_keywords 使用 section 级绝对位置：
            scope         → [1, 2]         → sec1 负责（标题含 "scope"）
            organizer kw  → [2, 4]         → sec2 负责（标题含 organizer label）
            limitations   → [N-2, N-1]     → sec(N-1) 负责（标题含 "limitations"）
            future work   → [N, N]         → secN 负责（标题含 "future work"）

        中间节数量 = N - 4（最少 0）：
            N=7  → 3 个中间节（sec3, sec4, sec5）
            N=10 → 6 个中间节（sec3..sec8）

        配合 _get_evaluation_blocks 的 section 级评估，无需强制每节单段。

    参数：
        task_type:      任务类型（当前支持 "analysis" 及通用 fallback）
        must_include:   benchmark must_include 关键词列表（含 organizer/closing 候选）
        expected_blocks: benchmark 指定的节数（默认 7）

    返回：
        Dict[str, str]：{"sec1": "标题", ..., "secN": "标题"}，共 N 项
    """
    N = max(expected_blocks, 3)

    organizer_label = next(
        (item for item in ORGANIZER_CANDIDATES_EN if item in must_include),
        "core organizing axis",
    )
    closing_label = next(
        (item for item in CLOSING_CANDIDATES_EN if item in must_include),
        "future work",
    )

    sections: Dict[str, str] = {}

    # ── 固定首节（sec1）：scope ──────────────────────────────────────────────
    sections["sec1"] = (
        "Scope, terminology, and practice context — "
        "define the scope and situate this review within the field"
    )

    # ── 固定末节（secN）：future work ────────────────────────────────────────
    sections[f"sec{N}"] = (
        f"Future work and research agenda — "
        f"outline priorities for future investigation and {closing_label}"
    )

    # ── 固定倒数第二节（sec(N-1)）：limitations ──────────────────────────────
    if N >= 2:
        sections[f"sec{N - 1}"] = (
            "Limitations and evidence gaps — "
            "assess methodological constraints, measurement limitations, and sources of bias"
        )

    # ── 中间节：sec2 到 sec(N-2)，共 N-3 节 ─────────────────────────────────
    # sec2 优先承担 organizer keyword（range [2,4] 覆盖 sec2..sec4）
    _body_templates = [
        f"Organizing framework — compare and structure the discussion around {organizer_label}",
        "Mechanisms and pathophysiology — synthesize mechanistic evidence and compare competing explanations",
        "Evidence synthesis — compare and integrate research findings across populations and care settings",
        "Clinical applications and real-world evidence — compare findings across patient subgroups",
        "Implementation and practice considerations — discuss barriers, facilitators, and system-level factors",
        "Heterogeneity, subgroup analysis, and moderating factors — examine differential effects",
        "Evidence gaps and controversies — identify unresolved questions and compare conflicting findings",
    ]

    body_count = N - 3  # sec2 到 sec(N-2) 的数量
    for i in range(body_count):
        sec_num = i + 2
        template = _body_templates[i % len(_body_templates)]
        sections[f"sec{sec_num}"] = template

    # ── 处理 N <= 3 时 sec2 与 sec(N-1) 重叠的退化情况 ──────────────────────
    # N=3: sec1=scope, sec2=limits, sec3=future → 无中间节，合并 organizer 到 sec2
    if N == 3:
        sections["sec2"] = (
            f"Evidence synthesis and limitations — compare the literature around "
            f"{organizer_label}, assess evidence quality and methodological constraints"
        )

    return dict(sorted(sections.items(), key=lambda kv: int(kv[0][3:])))


def _normalize_reference(reference: Union[str, Dict[str, object]]) -> Dict[str, object]:
    """Normalize a structured reference payload or its JSON serialization."""
    if isinstance(reference, dict):
        return reference
    if isinstance(reference, str):
        parsed_reference = json.loads(reference)
        if not isinstance(parsed_reference, dict):
            raise ValueError("reference JSON must decode to a dictionary")
        return parsed_reference
    raise TypeError("reference must be a dict or a JSON string")


def _contains_keyword(text: str, keyword: str) -> bool:
    """Return whether a keyword appears in a case-insensitive match."""
    return contains_keyword(text, keyword)


def _parse_int_field(raw_value: object, field_name: str) -> int:
    """Parse an integer field and reject non-integral types early."""
    if not isinstance(raw_value, (int, str)):
        raise TypeError(f"{field_name} must be an integer or an integer string")
    return int(raw_value)


def _normalize_keyword_specs(raw_items: object, field_name: str) -> List[Dict[str, object]]:
    """Normalize keyword constraint specs into dictionaries."""
    if not isinstance(raw_items, list):
        raise TypeError(f"{field_name} must be a list")
    return [
        dict(item) if isinstance(item, dict) else {"keyword": str(item)}
        for item in raw_items
    ]


def _validate_english_sample(sample_row: Dict[str, object]) -> None:
    """Guard against accidentally reintroducing Chinese benchmark assets."""
    serialized = json.dumps(sample_row, ensure_ascii=False)
    if _contains_cjk(serialized):
        sample_id = str(sample_row.get("sample_id", "unknown"))
        raise ValueError(
            f"Benchmark sample {sample_id} still contains Chinese text. "
            "Regenerate the benchmark assets in English before running."
        )


def load_benchmark_task(task_id: str) -> Dict[str, object]:
    """Load one benchmark sample and adapt it into a MetaWriter task config."""
    for sample_row in _read_jsonl_rows(SAMPLES_PATH):
        if str(sample_row.get("sample_id")) != task_id:
            continue

        _validate_english_sample(sample_row)

        prompt = str(sample_row["prompt"]).strip()
        task_type = str(sample_row["task_type"])

        raw_constraints_object = sample_row["constraints"]
        if not isinstance(raw_constraints_object, dict):
            raise TypeError("constraints must be a dictionary")
        raw_constraints = cast(Dict[str, object], raw_constraints_object)

        must_include_object = raw_constraints["must_include"]
        if not isinstance(must_include_object, list):
            raise TypeError("constraints.must_include must be a list")
        must_include = [str(item) for item in must_include_object]

        periodic_requirements_object = raw_constraints["periodic_requirements"]
        if not isinstance(periodic_requirements_object, list):
            raise TypeError("constraints.periodic_requirements must be a list")
        periodic_requirements = [str(item) for item in periodic_requirements_object]

        required_length_words = _parse_int_field(
            raw_constraints["required_length_words"],
            "constraints.required_length_words",
        )
        expected_blocks = _parse_int_field(
            raw_constraints["expected_blocks"],
            "constraints.expected_blocks",
        )
        range_keywords = _normalize_keyword_specs(
            raw_constraints["range_keywords"],
            "constraints.range_keywords",
        )
        periodic_keywords = _normalize_keyword_specs(
            raw_constraints["periodic_keywords"],
            "constraints.periodic_keywords",
        )
        once_keywords = [str(item) for item in must_include]
        if "once_keywords" in raw_constraints:
            once_keywords_object = raw_constraints["once_keywords"]
            if not isinstance(once_keywords_object, list):
                raise TypeError("constraints.once_keywords must be a list")
            once_keywords = [str(item) for item in once_keywords_object]

        proxy_questions_object = sample_row["proxy_questions"]
        if not isinstance(proxy_questions_object, list):
            raise TypeError("proxy_questions must be a list")
        checklist_object = sample_row["checklist"]
        if not isinstance(checklist_object, list):
            raise TypeError("checklist must be a list")
        proxy_questions_list = cast(List[object], proxy_questions_object)
        checklist_list = cast(List[object], checklist_object)

        constraints: List[str] = [
            _mark_document_level_constraint("Write the entire article in English."),
            _mark_document_level_constraint(
                f"Target length is about {required_length_words} words."
            ),
            _mark_document_level_constraint(
                f"The body should contain at least {expected_blocks} natural paragraphs "
                f"and read like a long-form review rather than a brief or outline."
            ),
            _mark_document_level_constraint(
                "Use a survey-paper style: define the scope first, then organize, "
                "compare, and synthesize the evidence before closing with limitations "
                "and future work."
            ),
            # Fix D：强制每节写成单段连续段落，使 drop_markdown_wrappers 后产生的
            # paragraph_blocks 数量与 expected_blocks 对齐，满足 range_keyword 绝对位置约束。
            _mark_document_level_constraint(
                f"Structure the article as exactly {expected_blocks} major paragraphs, "
                f"one per section. Write each section as a single continuous prose paragraph "
                f"without internal blank lines or sub-paragraph breaks."
            ),
            # Fix C：用精确词形约束替换"Must explicitly cover"，确保 contains_keyword
            # 的精确匹配通过（如 hemodynamics 而非 hemodynamic）。
            *[
                _mark_document_level_constraint(
                    f"You must use the exact term '{item}' verbatim in the text at least once."
                )
                for item in must_include
            ],
            *[
                _mark_document_level_constraint(f"Periodic requirement: {item}")
                for item in periodic_requirements
            ],
            # Fix E：将 periodic_keywords 中的关键词转化为逐字约束，确保精确词形出现。
            # 仅注入不在 must_include 中的 periodic 关键词（避免重复）。
            *[
                _mark_document_level_constraint(
                    f"Verbatim periodic keyword: use the exact word '{kw_spec['keyword']}' "
                    f"in at least one paragraph out of every {kw_spec['every']} paragraphs, "
                    f"starting from paragraph {kw_spec['start']}."
                )
                for kw_spec in periodic_keywords
                if str(kw_spec["keyword"]) not in must_include
            ],
        ]

        reference: Dict[str, object] = {
            "sample_id": str(sample_row["sample_id"]),
            "task_type": task_type,
            "language": "en",
            "prompt": prompt,
            "constraints": {
                "required_length_words": required_length_words,
                "must_include": must_include,
                "once_keywords": once_keywords,
                "periodic_requirements": periodic_requirements,
                "expected_blocks": expected_blocks,
                "range_keywords": range_keywords,
                "periodic_keywords": periodic_keywords,
            },
            "proxy_questions": [
                {
                    "qid": str(question["qid"]),
                    "question": str(question["question"]),
                    "answer": str(question["answer"]),
                }
                for question in proxy_questions_list
                if isinstance(question, dict)
            ],
            "checklist": [str(item) for item in checklist_list],
        }

        return {
            "task": prompt,
            "constraints": constraints,
            "outline": _build_outline(task_type, must_include, expected_blocks),
            "reference": reference,
        }

    raise ValueError(f"Unknown benchmark task: {task_id}")


def evaluate_output(
    generated_text: str, reference: Union[str, Dict[str, object]]
) -> Dict[str, object]:
    """Evaluate a generated benchmark answer with lightweight local heuristics."""
    normalized_reference = _normalize_reference(reference)
    normalized_text = generated_text.strip()
    if normalized_text == "":
        raise ValueError("generated_text must not be empty")

    raw_constraints_object = normalized_reference["constraints"]
    if not isinstance(raw_constraints_object, dict):
        raise TypeError("reference.constraints must be a dictionary")
    raw_constraints = cast(Dict[str, object], raw_constraints_object)

    must_include_object = raw_constraints["must_include"]
    if not isinstance(must_include_object, list):
        raise TypeError("reference.constraints.must_include must be a list")
    must_include = [str(item) for item in must_include_object]
    once_keywords_object = raw_constraints.get("once_keywords", must_include)
    if not isinstance(once_keywords_object, list):
        raise TypeError("reference.constraints.once_keywords must be a list")
    once_keywords = [str(item) for item in once_keywords_object]

    required_length_words = _parse_int_field(
        raw_constraints["required_length_words"],
        "reference.constraints.required_length_words",
    )

    expected_blocks = _parse_int_field(
        raw_constraints["expected_blocks"],
        "reference.constraints.expected_blocks",
    )

    range_keywords = _normalize_keyword_specs(
        raw_constraints["range_keywords"],
        "reference.constraints.range_keywords",
    )
    periodic_keywords = _normalize_keyword_specs(
        raw_constraints["periodic_keywords"],
        "reference.constraints.periodic_keywords",
    )

    checklist_object = normalized_reference["checklist"]
    if not isinstance(checklist_object, list):
        raise TypeError("reference.checklist must be a list")
    checklist_list = cast(List[object], checklist_object)
    checklist = [str(item) for item in checklist_list]

    proxy_questions_object = normalized_reference["proxy_questions"]
    if not isinstance(proxy_questions_object, list):
        raise TypeError("reference.proxy_questions must be a list")
    proxy_questions_list = cast(List[object], proxy_questions_object)
    proxy_questions = [
        dict(item) for item in proxy_questions_list if isinstance(item, dict)
    ]

    if len(must_include) == 0:
        raise ValueError("reference.constraints.must_include must not be empty")
    if len(proxy_questions) == 0:
        raise ValueError("reference.proxy_questions must not be empty")

    # 优先使用 section 级分块（标题数 == expected_blocks 时），
    # 保证 range_keyword 位置检查与 section 序号语义对齐。
    # 若标题数不匹配（旧格式文本或无标题），回退到段落级分块。
    paragraph_blocks = _get_evaluation_blocks(normalized_text, expected_blocks)
    body_text = "\n\n".join(paragraph_blocks)
    sentence_parts = re.split(r"[.!?]+", body_text)
    sentence_count = sum(1 for part in sentence_parts if part.strip() != "")

    matched_keywords = [
        item for item in must_include if _contains_keyword(normalized_text, item)
    ]
    matched_proxy_answers = [
        str(item["qid"])
        for item in proxy_questions
        if _contains_keyword(normalized_text, str(item["answer"]))
    ]

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
            raise ValueError("periodic_keywords.every must be positive")

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

    response_word_count = count_length_units(normalized_text)
    length_ratio = response_word_count / required_length_words
    length_score = compute_s_length(required_length_words, normalized_text)
    completion_rate = compute_completion_rate(paragraph_blocks, expected_blocks)
    once_signal = compute_acc_once(normalized_text, once_keywords)
    range_signal = compute_acc_range(paragraph_blocks, range_keywords)
    periodic_signal = compute_acc_periodic(paragraph_blocks, periodic_keywords)
    proxy_hit_count, proxy_total = compute_proxy_qa(normalized_text, proxy_questions)
    instruction_hits, instruction_total = compute_instruction_hits(
        normalized_text,
        must_include,
    )
    syntax_pass_rate, schema_pass_rate = compute_structure_scores(
        normalized_text,
        drop_markdown_wrappers=True,
    )

    entity_consistency_score = instruction_hits / instruction_total
    proxy_hit_rate = proxy_hit_count / proxy_total
    checklist_signal = len(matched_keywords) / max(len(checklist), len(must_include))
    structure_signal = min(
        1.0,
        0.2 * completion_rate
        + 0.15 * once_signal
        + 0.15 * range_signal
        + 0.15 * periodic_signal
        + 0.15 * syntax_pass_rate
        + 0.2 * schema_pass_rate,
    )
    logical_coherence = min(
        1.0,
        0.35 * structure_signal
        + 0.25 * proxy_hit_rate
        + 0.15 * checklist_signal
        + 0.25 * length_score,
    )
    constraint_violation_rate = 1.0 - min(entity_consistency_score, length_score)

    return {
        "constraint_violation_rate": constraint_violation_rate,
        "entity_consistency_score": entity_consistency_score,
        "logical_coherence": logical_coherence,
        "length_score": length_score,
        "diagnostics": {
            "matched_keywords": matched_keywords,
            "missing_keywords": [
                item for item in must_include if item not in matched_keywords
            ],
            "matched_proxy_question_ids": matched_proxy_answers,
            "response_word_count": response_word_count,
            "required_length_words": required_length_words,
            "length_ratio": length_ratio,
            "length_within_tolerance": 0.8 <= length_ratio <= 1.2,
            "paragraph_count": len(paragraph_blocks),
            "sentence_count": sentence_count,
            "expected_blocks": expected_blocks,
            "completion_rate": completion_rate,
            "once_signal": once_signal,
            "range_keyword_hits": range_keyword_hits,
            "missing_range_keywords": missing_range_keywords,
            "periodic_keyword_hits": periodic_keyword_hits,
            "missing_periodic_keywords": missing_periodic_keywords,
            "proxy_hit_count": proxy_hit_count,
            "proxy_total": proxy_total,
            "syntax_pass_rate": syntax_pass_rate,
            "schema_pass_rate": schema_pass_rate,
            "structure_signal": structure_signal,
            "checklist": checklist,
        },
    }


def build_benchmark_task_config(task_id: str) -> Dict[str, object]:
    """Build the benchmark task config registered into the main task registry."""
    benchmark_task = load_benchmark_task(task_id)
    constraints_object = benchmark_task["constraints"]
    if not isinstance(constraints_object, list):
        raise TypeError("benchmark_task.constraints must be a list")
    outline_object = benchmark_task["outline"]
    if not isinstance(outline_object, dict):
        raise TypeError("benchmark_task.outline must be a dictionary")
    reference_object = benchmark_task["reference"]
    if not isinstance(reference_object, dict):
        raise TypeError("benchmark_task.reference must be a dictionary")

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
