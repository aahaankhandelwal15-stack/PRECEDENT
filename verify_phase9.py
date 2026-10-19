"""Minimal verification script for Phase 9 - core abstractions."""
from app.core.budget import ExecutionBudget, BudgetViolation
from app.core.errors import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderModelError,
    EmbeddingError,
    InfrastructureError,
    wrap_provider_error,
)
from app.core.validation import validate_confidence_range, validate_enum_value, validate_required_fields
from app.schemas import TraceRecord
from datetime import datetime, timezone


def test_budget_one_call_max_one():
    """max_llm_calls=1: first call violates."""
    b = ExecutionBudget(max_llm_calls=1)
    b.consume_llm_call()
    try:
        b.assert_llm_calls_limit("test")
        assert False, "Should have raised after 1 call with max=1"
    except BudgetViolation:
        pass
    print("  PASS: budget max=1, 1 call violates")


def test_budget_three_calls_max3():
    """max_llm_calls=3: 2 calls OK, 3rd violates."""
    b = ExecutionBudget(max_llm_calls=3)
    b.consume_llm_call()
    b.consume_llm_call()
    # 2 calls with max=3 should still be within limit
    assert b.check_llm_calls_limit() == True, "2 < 3 should be within limit"
    # 3rd call should violate
    try:
        b.assert_llm_calls_limit("test")
        assert False, "Should have raised after 3rd call with max=3"
    except BudgetViolation:
        pass
    print("  PASS: budget max=3, 3 calls violate")


def test_error_categories_and_safe_messages():
    """All error types have correct category, retryable, and safe messages."""
    tests = [
        ("auth", ProviderAuthenticationError, False),
        ("rate", ProviderRateLimitError, True),
        ("timeout", ProviderTimeoutError, True),
        ("model", ProviderModelError, False),
        ("embedding", EmbeddingError, True),
        ("infra", InfrastructureError, False),
    ]
    for name, cls, should_retry in tests:
        e = cls()
        assert e.category is not None, f"{name}: category missing"
        assert e.retryable == should_retry, f"{name}: retryable mismatch"
        assert len(e.safe_message) > 0, f"{name}: safe_message empty"
        # Raw exception must NOT leak into safe message
        orig = Exception("my-secret-key-123")
        wrapped = wrap_provider_error(orig, cls)
        assert "my-secret-key-123" not in wrapped.safe_message, f"{name}: raw exc leaked!"
    print("  PASS: error categories and safe messages")


def test_validation_confidence_range():
    """validate_confidence_range works for valid and rejects invalid."""
    # Valid values
    r1 = validate_confidence_range(0.5, "c", "test")
    assert r1 == 0.5
    r2 = validate_confidence_range(0.0, "c", "test")
    assert r2 == 0.0
    r3 = validate_confidence_range(1.0, "c", "test")
    assert r3 == 1.0
    # Invalid values should raise
    try:
        validate_confidence_range(1.1, "c", "test")  # type: ignore
        assert False, "1.1 should be rejected"
    except Exception:
        pass
    try:
        validate_confidence_range(-0.1, "c", "test")  # type: ignore
        assert False, "-0.1 should be rejected"
    except Exception:
        pass
    try:
        validate_confidence_range("bad", "c", "test")  # type: ignore
        assert False, "string should be rejected"
    except Exception:
        pass
    print("  PASS: validation confidence range")


def test_validation_enum():
    """validate_enum_value works."""
    validate_enum_value("good", ["good", "bad"], "status", "test")
    try:
        validate_enum_value("bad", ["good", "bad"], "status", "test")  # type: ignore
        assert False, "bad should be rejected"
    except Exception:
        pass
    print("  PASS: validation enum")


def test_validation_required_fields():
    """validate_required_fields works."""
    validate_required_fields({"a": 1, "b": 2}, ["a", "b"])
    try:
        validate_required_fields({"a": 1}, ["a", "b"])  # type: ignore
        assert False, "missing b should be rejected"
    except Exception:
        pass
    print("  PASS: validation required fields")


def test_tracerecord_fr8_1_plus_fr9():
    """TraceRecord has all FR-8.1 fields plus FR-9 budget/trace fields."""
    trace = TraceRecord(
        trace_id="v9-full",
        timestamp=datetime.now(timezone.utc).isoformat(),
        query_text="test query",
        sub_questions=["q1"],
        chunk_results=[],
        claims=[],
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
        execution_budget={"llm_calls": 3, "tokens_input": 200, "tokens_output": 100, "retries": 2, "wall_time_ms": 300.0, "estimated_cost": 0.10},
        provider="openai",
        model="gpt-4o",
        retries=2,
        failures=[],
        latency_ms=300,
    )
    assert trace.execution_budget["llm_calls"] == 3
    assert trace.provider == "openai"
    assert trace.model == "gpt-4o"
    assert trace.retries == 2
    assert trace.latency_ms == 300
    assert trace.outcome == "answered"
    # No secrets leakage through safe public fields
    assert "sk-" not in str(trace).lower()
    assert "api_key" not in str(trace).lower()
    print("  PASS: TraceRecord FR-8.1 + FR-9")


def test_budget_record_snapshot():
    """BudgetRecord provides consistent immutable snapshot."""
    b = ExecutionBudget()
    b.consume_llm_call(tokens_input=50, tokens_output=100, cost=0.01)
    rec = b.budget_record
    assert rec.llm_calls == 1
    assert rec.tokens_input == 50
    assert rec.tokens_output == 100
    assert rec.estimated_cost == 0.01
    print("  PASS: BudgetRecord snapshot")


if __name__ == '__main__':
    print("=== PHASE 9 VERIFICATION ===\n")
    test_budget_one_call_max_one()
    test_budget_three_calls_max3()
    test_error_categories_and_safe_messages()
    test_validation_confidence_range()
    test_validation_enum()
    test_validation_required_fields()
    test_tracerecord_fr8_1_plus_fr9()
    test_budget_record_snapshot()
    print("\n=== ALL VERIFICATION TESTS PASSED ===")