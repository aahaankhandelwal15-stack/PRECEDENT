"""Core module exports for Precedent V1."""

from app.core.budget import ExecutionBudget, BudgetViolation, BudgetRecord
from app.core.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderModelError,
    EmbeddingError,
    InfrastructureError,
    ModelOutputValidationError,
    wrap_provider_error,
    PrecedentError,
    PROVIDER_EXCEPTION_MAP,
    map_provider_exception,
    ErrorCategory,
)
from app.core.validation import (
    validate_structured_output,
    validate_required_fields,
    validate_confidence_range,
    validate_enum_value,
    validate_non_empty_string,
    validate_list_of_strings,
    validate_claim_verification_result,
    extract_json_from_text,
)
from app.core.timeouter import (
    TimeoutGuard,
    with_timeout,
    check_budget_or_timeout,
    SyncTimeoutGuard,
    check_budget_or_timeout_sync,
)
from app.core.chunking import chunk_text, chunk_by_sentences, estimate_token_count
from app.core.confidence_scorer import (
    compute_confidence_from_inputs,
    compute_confidence_from_retrieval,
    compute_confidence_from_generation,
    should_refuse,
    compute_authority_score,
    compute_claim_verification_score,
    ConfidenceComponents,
)
from app.core.retrieval import (
    decompose_question,
    retrieve_chunks,
    BoundedIterator,
    DEFAULT_REFUSAL_MESSAGES,
    _determine_refusal_category,
    _build_refusal_response,
    _should_short_circuit_to_refusal,
    DEFAULT_CONFIDENCE_THRESHOLD,
    run_retrieval_pipeline,
    filter_chunks_by_authority,
    assess_evidence_sufficiency,
    assess_retrieval_relevance,
)

__all__ = [
    # Budget
    "ExecutionBudget",
    "BudgetViolation",
    "BudgetRecord",
    # Errors
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "ProviderModelError",
    "EmbeddingError",
    "InfrastructureError",
    "ModelOutputValidationError",
    "wrap_provider_error",
    "PrecedentError",
    "PROVIDER_EXCEPTION_MAP",
    "map_provider_exception",
    "ErrorCategory",
    # Validation
    "validate_structured_output",
    "validate_required_fields",
    "validate_confidence_range",
    "validate_enum_value",
    "validate_non_empty_string",
    "validate_list_of_strings",
    "validate_claim_verification_result",
    "extract_json_from_text",
    # Timeouter
    "TimeoutGuard",
    "with_timeout",
    "check_budget_or_timeout",
    "SyncTimeoutGuard",
    "check_budget_or_timeout_sync",
    # Chunking
    "chunk_text",
    "chunk_by_sentences",
    "estimate_token_count",
    # Confidence
    "compute_confidence_from_inputs",
    "compute_confidence_from_retrieval",
    "compute_confidence_from_generation",
    "should_refuse",
    "compute_authority_score",
    "compute_claim_verification_score",
    "ConfidenceComponents",
    # Retrieval
    "decompose_question",
    "retrieve_chunks",
    "BoundedIterator",
    "DEFAULT_REFUSAL_MESSAGES",
    "_determine_refusal_category",
    "_build_refusal_response",
    "_should_short_circuit_to_refusal",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "run_retrieval_pipeline",
    "filter_chunks_by_authority",
    "assess_evidence_sufficiency",
    "assess_retrieval_relevance",
]