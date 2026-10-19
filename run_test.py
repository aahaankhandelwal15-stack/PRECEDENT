import sys
sys.path.insert(0, "D:/Projects/PRECEDENT")

# Test basic Phase 9 modules
from app.core.budget import ExecutionBudget
from app.core.errors import wrap_provider_error, ProviderTimeoutError, ProviderAuthenticationError, ProviderRateLimitError
from app.core.validation import validate_confidence_range, validate_enum_value, validate_required_fields
from app.schemas import TraceRecord
from datetime import datetime, timezone

# Test budget
b = ExecutionBudget(max_llm_calls=1)
b.consume_llm_call()
try:
    b.assert_llm_calls_limit("test")
    print("FAIL: Should have raised")
except Exception as e:
    print("PASS: Budget test")

# Test errors
e = ProviderTimeoutError()
print(f"PASS: ProviderTimeoutError: category={e.category}, retryable={e.retryable}")

e = ProviderAuthenticationError()
print(f"PASS: ProviderAuthenticationError: category={e.category}, retryable={e.retryable}")

e = ProviderRateLimitError()
print(f"PASS: ProviderRateLimitError: category={e.category}, retryable={e.retryable}")

# Test validation
r = validate_confidence_range(0.5, "c", "test")
assert r == 0.5
print("PASS: validate_confidence_range valid")

try:
    validate_confidence_range(1.5, "c", "test")
    assert False
except:
    print("PASS: validate_confidence_range rejects 1.5")

try:
    validate_confidence_range(-0.1, "c", "test")
    assert False
except:
    print("PASS: validate_confidence_range rejects -0.1")

try:
    validate_enum_value("good", ["good", "bad"], "status", "test")
    print("PASS: validate_enum_value valid")
except:
    assert False

try:
    validate_enum_value("bad", ["good", "bad"], "status", "test")
    assert False
except:
    print("PASS: validate_enum_value rejects bad")

try:
    validate_required_fields({"a": 1, "b": 2}, ["a", "b"])
    print("PASS: validate_required_fields valid")
except:
    assert False

try:
    validate_required_fields({"a": 1}, ["a", "b"])
    assert False
except:
    print("PASS: validate_required_fields rejects missing")

# Test TraceRecord
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
assert trace.outcome == "answered"
assert trace.provider == "openai"
assert trace.model == "gpt-4o"
print(f"PASS: TraceRecord - outcome={trace.outcome}, provider={trace.provider}")

print("\n=== All Phase 9 basic tests passed ===")