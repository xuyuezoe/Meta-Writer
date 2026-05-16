"""Clean MetaBench scoring package.

This package is intentionally independent from the legacy ``metabench`` folder.
New benchmark dimensions should be added here as small, explicit modules.
"""

from .content import (
    LLMProxyQuestionJudge,
    ProxyQuestionDecision,
    ProxyQuestionSpec,
    create_default_proxy_question_judge,
    evaluate_content_dimension,
    score_entity_consistency,
    score_proxy_hit_rate,
)
from .citation import (
    CitationChunk,
    CitationCountScoreResult,
    CitationEvent,
    ChunkRoleClassification,
    SourceFidelityResult,
    SourceBalanceResult,
    classify_chunk_role,
    classify_chunk_roles,
    evaluate_citation_dimension,
    score_citation_count,
    score_section_distribution,
    score_source_balance,
    score_source_fidelity,
    split_into_seven_sections,
)
from .evaluator import META_BENCH_METRIC_ORDER, evaluate_meta_bench
from .schemas import GeneratedTask, TaskSpec
from .section_budget import (
    RELATIVE_JITTER,
    SECTION_SHARE_PRIORS,
    SectionBudget,
    build_section_budget,
)
from .structure import (
    evaluate_structure_dimension,
    score_completion_rate,
    score_length,
)
from .task_generator import (
    build_generation_constraints,
    build_main_task_config,
    build_must_include,
    generate_outline,
    generate_prompt,
    generate_proxy_questions,
    generate_review_task,
)

__all__ = [
    "GeneratedTask",
    "LLMProxyQuestionJudge",
    "ProxyQuestionDecision",
    "ProxyQuestionSpec",
    "TaskSpec",
    "CitationChunk",
    "CitationCountScoreResult",
    "CitationEvent",
    "ChunkRoleClassification",
    "SourceFidelityResult",
    "SourceBalanceResult",
    "META_BENCH_METRIC_ORDER",
    "build_generation_constraints",
    "build_main_task_config",
    "build_must_include",
    "classify_chunk_role",
    "classify_chunk_roles",
    "evaluate_citation_dimension",
    "create_default_proxy_question_judge",
    "evaluate_content_dimension",
    "evaluate_meta_bench",
    "evaluate_structure_dimension",
    "generate_outline",
    "generate_prompt",
    "generate_proxy_questions",
    "generate_review_task",
    "score_entity_consistency",
    "score_completion_rate",
    "score_length",
    "score_proxy_hit_rate",
    "score_citation_count",
    "score_section_distribution",
    "score_source_balance",
    "score_source_fidelity",
    "SECTION_SHARE_PRIORS",
    "RELATIVE_JITTER",
    "SectionBudget",
    "build_section_budget",
    "split_into_seven_sections",
]
