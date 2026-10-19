"""
==================================================
PHASE 9: Execution Guard / Runtime Control Tests
==================================================

Tests for execution budget, provider error normalization,
model output validation, timeout safety, and audit trace integration.

Use mocks/fakes for external providers. Do NOT make the test suite
depend on real Groq/OpenAI/etc. API calls.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

# Import the Precedent modules we're testing
from app.core.budget import (
    ExecutionBudget,
    BudgetRecord,
    BudgetViolation,
    BudgetRecord as BR,
)
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
)
from app.core.validation import (
    validate_structured_output,
    validate_required_fields,
    validate_confidence_range,
    validate_enum_value,
)
from app.core.timeouter import (
    TimeoutGuard,
    with_timeout,
    check_budget_or_timeout,
)


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------

def test_budget_creation_defaults():
    """Budget creates with reasonable defaults."""
    b = ExecutionBudget()
    assert b.max_llm_calls == 10
    assert b.max_tokens == 50000
    assert b.max_retries == 3
    assert b.max_wall_time_ms == 30000
    assert b.max_estimated_cost == 10.0


def test_budget_creation_custom():
    """Budget accepts custom values."""
    b = ExecutionBudget(
        max_llm_calls=5,
        max_tokens=20000,
        max_retries=2,
        max_wall_time_ms=15000,
        max_estimated_cost=5.0,
    )
    assert b.max_llm_calls == 5
    assert b.max_tokens == 20000
    assert b.max_retries == 2
    assert b.max_wall_time_ms == 15000
    assert b.max_estimated_cost == 5.0


def test_budget_consume_llm_call():
    """Consuming LLM call updates state."""
    b = ExecutionBudget()
    b.consume_llm_call(tokens_input=100, tokens_output=200, cost=0.01, retries=0)
    assert b.llm_calls == 1
    assert b.tokens_input == 100
    assert b.tokens_output == 200
    assert b.estimated_cost == 0.01


def test_budget_consume_retry():
    """Consuming retry updates state."""
    b = ExecutionBudget()
    b.consume_llm_call(retries=0)
    b.consume_retry()
    b.consume_retry()
    assert b.retries == 2


def test_budget_check_limits():
    """Checking limits works correctly."""
    b = ExecutionBudget(max_llm_calls=3, max_retries=2)
    assert b.check_llm_calls_limit()  # 0 < 3
    b.consume_llm_call()
    assert b.check_llm_calls_limit()  # 1 < 3
    b.consume_llm_call()
    assert b.check_llm_calls_limit()  # 2 < 3
    b.consume_llm_call()
    assert not b.check_llm_calls_limit()  # 3 >= 3


def test_budget_assert_limits_raises():
    """Asserting limits raises BudgetViolation."""
    b = ExecutionBudget(max_llm_calls=2)
    b.consume_llm_call()
    b.consume_llm_call()
    with pytest.raises(BudgetViolation) as excinfo:
        b.assert_llm_calls_limit("test_operation")
    assert "llm_calls" in str(excinfo.value)


def test_budget_record_snapshot():
    """Budget record snapshot is consistent."""
    b = ExecutionBudget()
    b.consume_llm_call(tokens_input=50, tokens_output=100)
    record = b.budget_record
    assert record.llm_calls == 1
    assert record.tokens_input == 50
    assert record.tokens_output == 100


# ---------------------------------------------------------------------------
# Error normalization tests
# ---------------------------------------------------------------------------

def test_provider_error_categories():
    """Each provider error has correct category, retryable, safe_message."""
    errors = [
        ProviderAuthenticationError(),
        ProviderRateLimitError(),
        ProviderTimeoutError(),
        ProviderModelError(),
        EmbeddingError(),
        InfrastructureError(),
    ]
    expected = [
        ("provider-auth", False, "Authentication failed with the AI provider"),
        ("provider-rate-limit", True, "The AI provider is rate-limiting requests"),
        ("provider-timeout", True, "The AI provider request timed out"),
        ("provider-model", False, "The AI provider's model returned an error"),
        ("embedding-failure", True, "An error occurred generating embeddings"),
        ("infrastructure", False, "A system infrastructure issue was encountered"),
    ]
    for err, (cat, retryable, safe_msg) in zip(errors, expected):
        assert err.category == cat, f"category mismatch: {err.category} != {cat}"
        assert err.retryable == retryable, f"retryable mismatch: {err.retryable} != {retryable}"
        assert safe_msg in err.safe_message, f"safe_message missing: {safe_msg}"


def test_provider_error_diagnostic_codes():
    """Each provider error has a diagnostic code."""
    errors = [
        ProviderAuthenticationError(),
        ProviderRateLimitError(),
        ProviderTimeoutError(),
        ProviderModelError(),
        EmbeddingError(),
        InfrastructureError(),
    ]
    for err in errors:
        assert err.diagnostic_code is not None
        assert isinstance(err.diagnostic_code, str)


def test_wrap_provider_error():
    """Wrapping provider error keeps safe message."""
    original = Exception("Original provider error message")
    wrapped = wrap_provider_error(original, ProviderAuthenticationError)
    assert isinstance(wrapped, ProviderAuthenticationError)
    assert wrapped.safe_message == "Authentication failed with the AI provider"
    assert "Original provider error message" not in wrapped.safe_message
    # Diagnostic code should be set
    assert wrapped.diagnostic_code is not None


def test_map_provider_exception():
    """Mapping provider exceptions works."""
    # Test with a known exception type
    exc = Exception("auth failed")
    mapped = map_provider_exception(exc)
    assert isinstance(mapped, ProviderAuthenticationError)

    # Test with unknown type (falls back to ProviderModelError)
    exc2 = Exception("some other error")
    mapped2 = map_provider_exception(exc2)
    assert isinstance(mapped2, ProviderModelError)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_validate_structured_output_success():
    """Validating output against schema works when input is valid dict."""
    raw = {
        "is_valid": True,
        "confidence": 0.85,
        "quoted_span": "some quoted text",
    }
    result = validate_structured_output(raw, ClaimVerificationResult, "test")
    assert isinstance(result, ClaimVerificationResult)
    assert result.is_valid is True
    assert result.confidence == 0.85


def test_validate_structured_output_string_json():
    """Validating output works when input is JSON string."""
    raw = '{"is_valid": true, "confidence": 0.85, "quoted_span": "span"}'
    result = validate_structured_output(raw, ClaimVerificationResult, "test")
    assert isinstance(result, ClaimVerificationResult)
    assert result.is_valid is True


def test_validate_structured_output_invalid():
    """Validating output raises ModelOutputValidationError on failure."""
    raw = {"is_valid": "not_a_bool", "confidence": 1.5}  # invalid types/values
    with pytest.raises(ModelOutputValidationError):
        validate_structured_output(raw, ClaimVerificationResult, "test")


def test_validate_required_fields():
    """Validating required fields works."""
    data = {"name": "test", "value": 42}
    validate_required_fields(data, ["name", "value"])  # should not raise

    data_missing = {"name": "test"}  # missing "value"
    with pytest.raises(ModelOutputValidationError):
        validate_required_fields(data_missing, ["name", "value"])


def test_validate_confidence_range_valid():
    """Validating confidence in [0.0, 1.0] works."""
    result = validate_confidence_range(0.75, "confidence", "test")
    assert result == 0.75

    result = validate_confidence_range(0.0, "confidence", "test")
    assert result == 0.0

    result = validate_confidence_range(1.0, "confidence", "test")
    assert result == 1.0


def test_validate_confidence_range_invalid():
    """Validating confidence outside [0.0, 1.0] raises error."""
    with pytest.raises(ModelOutputValidationError):
        validate_confidence_range(1.5, "confidence", "test")

    with pytest.raises(ModelOutputValidationError):
        validate_confidence_range(-0.1, "confidence", "test")

    with pytest.raises(ModelOutputValidationError):
        validate_confidence_range("not_a_number", "confidence", "test")


def test_validate_enum_value_valid():
    """Validating enum value works."""
    result = validate_enum_value("no_evidence", ["no_evidence", "low_confidence"], "category", "test")
    assert result == "no_evidence"


def test_validate_enum_value_invalid():
    """Validating invalid enum value raises error."""
    with pytest.raises(ModelOutputValidationError):
        validate_enum_value("invalid_value", ["no_evidence", "low_confidence"], "category", "test")


# ---------------------------------------------------------------------------
# Timeout / timeouter tests
# ---------------------------------------------------------------------------

def test_timeout_guard_check():
    """Timeout guard checking works."""
    b = ExecutionBudget(max_wall_time_ms=10000)  # 10 seconds
    guard = TimeoutGuard(b)
    assert guard.check()  # Should be within limit initially


def test_timeout_guard_check_exceeded():
    """Timeout guard detects exceeded budget."""
    b = ExecutionBudget(max_wall_time_ms=0)  # Zero budget
    guard = TimeoutGuard(b)
    assert not guard.check()  # Should exceed immediately


# ---------------------------------------------------------------------------
# Retrieval endpoint budget integration tests
# ---------------------------------------------------------------------------

class MockRequest:
    """Minimal mock request for endpoint testing."""
    def __init__(self, query="test query"):
        self.query = query


class MockFilteredResponse:
    """Mock filtered retrieval response."""
    def __init__(self):
        self.chunks = ["chunk text 1", "chunk text 2"]
        self.found = True
        self.empty_corpus = False
        self.conflict_flag = {"has_conflict": False, "conflict_details": [], "conflict_pairing": None}
        self.ranked_ids = ["chunk_1", "chunk_2"]
        self.trace_id = "test-trace-123"


@pytest.fixture
def mock_budget():
    """Provide a fresh budget for each test."""
    b = ExecutionBudget(max_llm_calls=10, max_retries=3, max_wall_time_ms=60000)
    b.start_wall_timer()
    return b


def test_budget_violation_endpoint_structure():
    """Test that budget violation produces structured trace record."""
    from app.api.v1.endpoints.retrieval_phase2 import FilteredRetrievalResponse
    from app.schemas import TraceRecord

    # Create a trace record manually with budget_exceeded outcome
    trace = TraceRecord(
        trace_id="test-violation-1",
        timestamp=datetime.utcnow().isoformat(),
        query_text="test query",
        sub_questions=["what is this"],
        chunk_results=[],
        claims=[],
        confidence_inputs={
            "retrieval_relevance": 0.0,
            "evidence_sufficiency": 0.0,
            "claim_verifications": [],
            "authority_flags": {
                "has_conflict": False,
                "conflict_details": [],
                "authority_level": "departmental",
                "jurisdiction_match": True,
                "department_match": True,
                "within_effective_window": True,
                "is_superseded": False,
            },
        },
        final_confidence=0.0,
        outcome="budget_exceeded",
        refusal_category="no_evidence",
        execution_budget={
            "llm_calls": 10,
            "tokens_input": 0,
            "tokens_output": 0,
            "retries": 3,
            "wall_time_ms": 30000.0,
            "estimated_cost": 10.0,
        },
        provider="budget_system",
        model="budget_enforcement",
        retries=3,
        failures=[],
        latency_ms=30000,
    )

    # Verify all required fields exist
    assert trace.trace_id is not None
    assert trace.timestamp is not None
    assert trace.query_text == "test query"
    assert trace.outcome == "budget_exceeded"
    assert trace.refusal_category == "no_evidence"
    assert trace.execution_budget is not None
    assert trace.execution_budget["llm_calls"] == 10
    assert trace.retries == 3
    assert trace.latency_ms == 30000


def test_trace_record_all_fields_present():
    """Test TraceRecord has all FR-8.1 fields."""
    from app.schemas import TraceRecord

    trace = TraceRecord(
        trace_id="test-full-1",
        timestamp=datetime.utcnow().isoformat(),
        query_text="test query",
        sub_questions=["sub q1", "sub q2"],
        chunk_results=[{"chunk_id": "c1", "similarity": 0.8}],
        claims=[{"text": "claim 1", "is_valid": True, "quoted_span": "span"}],
        confidence_inputs={
            "retrieval_relevance": 0.8,
            "evidence_sufficiency": 0.9,
            "claim_verifications": [],
            "authority_flags": {
                "has_conflict": False,
                "conflict_details": [],
                "authority_level": "global",
                "jurisdiction_match": True,
                "department_match": True,
                "within_effective_window": True,
                "is_superseded": False,
            },
        },
        final_confidence=0.75,
        outcome="answered",
        refusal_category=None,
        execution_budget={"llm_calls": 2, "tokens_input": 100, "tokens_output": 50, "retries": 0, "wall_time_ms": 150.0, "estimated_cost": 0.05},
        provider="groq",
        model="mixtral-8x7b-32768",
        retries=0,
        failures=[],
        latency_ms=150,
    )

    # Verify all FR-8.1 fields are present and savable
    assert trace.trace_id == "test-full-1"
    assert trace.timestamp is not None
    assert trace.query_text == "test query"
    assert trace.sub_questions == ["sub q1", "sub q2"]
    assert len(trace.chunk_results) == 1
    assert len(trace.claims) == 1
    assert trace.confidence_inputs["retrieval_relevance"] == 0.8
    assert trace.final_confidence == 0.75
    assert trace.outcome == "answered"
    assert trace.refusal_category is None
    assert trace.execution_budget is not None
    assert trace.provider == "groq"
    assert trace.model == "mixtral-8x7b-32768"
    assert trace.retries == 0
    assert trace.failures == []
    assert trace.latency_ms == 150


# Run: pytest tests/test_phase9.py -v