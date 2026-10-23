"""Generation Engine for Precedent V1.

Integrates LLM provider abstraction with generation prompts to produce
grounded, verifiable answers with audit trails.
"""

from __future__ import annotations

import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import date

from app.schemas import (
    SubQuestion,
    FilteredRetrievalResponse,
    ClaimVerificationResult,
    ClaimWithCitations,
    AuthorityApplicabilityFlags,
    RefusalCategory,
    Outcome,
    TraceRecord,
    DEFAULT_REFUSAL_MESSAGES,
)
from app.core.llm_provider import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    get_default_provider,
    generate_response,
)
from app.core.generation_prompts import (
    build_generation_prompt,
    build_claim_extraction_prompt,
    parse_generation_response,
    validate_generation_response,
)
from app.core.confidence_scorer import (
    compute_confidence_from_generation,
    should_refuse,
)
from app.core.budget import ExecutionBudget
from app.core.errors import ProviderError, ModelOutputValidationError
from app.core.retrieval import _aggregate_authority_flags


@dataclass
class GenerationResult:
    """Result of the generation phase."""
    answer: Optional[str]
    claims: List[ClaimVerificationResult]
    refusal: Optional[Dict[str, Any]]
    confidence: float
    trace_id: str
    raw_llm_response: Optional[LLMResponse] = None
    verification_failures: List[Dict[str, Any]] = None


async def run_generation_phase(
    retrieval_response: FilteredRetrievalResponse,
    query: str,
    sub_questions: List[SubQuestion],
    query_context: Optional[Dict[str, Any]] = None,
    budget: Optional[ExecutionBudget] = None,
    provider: Optional[LLMProvider] = None,
) -> Tuple[GenerationResult, TraceRecord]:
    """
    Run the generation phase of the pipeline.

    Takes retrieval results and generates a grounded answer with citations.
    Returns (GenerationResult, TraceRecord).
    """
    trace_id = str(uuid.uuid4())
    if budget is None:
        budget = ExecutionBudget()
    budget.start_wall_timer()

    trace_chunks = []
    trace_failures = []
    verification_failures = []

    try:
        # Check if we should refuse based on retrieval results
        authority_flags = _aggregate_authority_flags(
            [],  # We'll rebuild from retrieval_response
            retrieval_response.conflict_flag,
        )

        # Update authority flags from retrieval response
        authority_flags.update(retrieval_response.authority_flags)

        # Check for early refusal conditions
        should_refuse_early, refusal_cat = _check_generation_refusal(
            retrieval_response, authority_flags
        )

        if should_refuse_early:
            return _build_refusal_result(
                trace_id, query, sub_questions, retrieval_response,
                refusal_cat, budget, trace_chunks, trace_failures
            )

        # Build generation prompt
        context = _build_generation_context(
            retrieval_response, query, sub_questions, query_context
        )
        messages = build_generation_prompt(context)

        # Generate answer
        llm_response = await generate_response(
            messages=messages,
            budget=budget,
            provider=provider,
        )

        # Parse and validate response
        try:
            parsed_response = parse_generation_response(llm_response.content)
        except (ValueError, json.JSONDecodeError) as e:
            # Validation failed - record failure and refuse
            trace_failures.append({
                "category": "validation_error",
                "message": f"Generation output validation failed: {str(e)}",
                "retryable": False,
                "details": {"raw_response": llm_response.content[:500]},
            })
            return _build_refusal_result(
                trace_id, query, sub_questions, retrieval_response,
                RefusalCategory.VALIDATION_ERROR, budget, trace_chunks, trace_failures
            )

        # Extract claims and verify them
        claims, claim_verification_failures = await _extract_and_verify_claims(
            parsed_response, context, budget, provider
        )
        verification_failures.extend(claim_verification_failures)

        # Compute generation-phase confidence
        confidence_components = compute_confidence_from_generation(
            retrieval_relevance=retrieval_response.retrieval_relevance.score,
            evidence_sufficiency=retrieval_response.evidence_sufficiency.score,
            claim_verifications=claims,
            authority_flags=authority_flags,
        )

        final_confidence = confidence_components.final_confidence

        # Check if we should refuse based on confidence
        refuse, refusal_cat = should_refuse(
            final_confidence=final_confidence,
            authority_flags=authority_flags,
            evidence_sufficiency=retrieval_response.evidence_sufficiency.score,
            retrieval_relevance=retrieval_response.retrieval_relevance.score,
        )

        if refuse:
            return _build_refusal_result(
                trace_id, query, sub_questions, retrieval_response,
                refusal_cat, budget, trace_chunks, trace_failures + verification_failures,
                final_confidence
            )

        # Build successful result
        answer = parsed_response.get("answer")

        trace = TraceRecord(
            trace_id=trace_id,
            query_text=query,
            sub_questions=[sq.question for sq in sub_questions],
            chunk_results=trace_chunks,
            claims=[_claim_to_dict(c) for c in claims],
            confidence_inputs={
                "retrieval_relevance": retrieval_response.retrieval_relevance.score,
                "evidence_sufficiency": retrieval_response.evidence_sufficiency.score,
                "claim_verifications": [_claim_to_dict(c) for c in claims],
                "authority_flags": authority_flags,
            },
            final_confidence=final_confidence,
            outcome=Outcome.ANSWERED,
            refusal_category=None,
            execution_budget=budget.to_dict(),
            provider=llm_response.provider,
            model=llm_response.model,
            retries=budget.retries,
            failures=trace_failures + verification_failures,
            latency_ms=int(budget.wall_time_ms),
        )

        result = GenerationResult(
            answer=answer,
            claims=claims,
            refusal=None,
            confidence=final_confidence,
            trace_id=trace_id,
            raw_llm_response=llm_response,
            verification_failures=verification_failures,
        )

        return result, trace

    except ProviderError as e:
        budget.consume_retry()
        trace_failures.append({
            "category": e.category.value,
            "message": e.safe_message,
            "retryable": e.retryable,
            "diagnostic_code": e.diagnostic_code,
        })
        raise


def _check_generation_refusal(
    retrieval_response: FilteredRetrievalResponse,
    authority_flags: Dict[str, Any],
) -> Tuple[bool, Optional[RefusalCategory]]:
    """Check if we should refuse before generation."""

    # Conflict detection
    if retrieval_response.conflict_flag.has_conflict:
        return True, RefusalCategory.CONFLICTING_EVIDENCE

    # Stale evidence
    if not authority_flags.get("within_effective_window", True):
        return True, RefusalCategory.STALE_EVIDENCE
    if authority_flags.get("is_superseded", False):
        return True, RefusalCategory.STALE_EVIDENCE

    # Inapplicable evidence
    if not authority_flags.get("jurisdiction_match", True):
        return True, RefusalCategory.INAPPLICABLE_EVIDENCE
    if not authority_flags.get("department_match", True):
        return True, RefusalCategory.INAPPLICABLE_EVIDENCE

    # Insufficient authority
    if not authority_flags.get("is_authoritative", False):
        return True, RefusalCategory.INSUFFICIENT_AUTHORITY

    # Evidence sufficiency
    if retrieval_response.evidence_sufficiency.score < 0.3:  # Hard threshold
        return True, RefusalCategory.INAPPLICABLE_EVIDENCE

    return False, None


def _build_generation_context(
    retrieval_response: FilteredRetrievalResponse,
    query: str,
    sub_questions: List[SubQuestion],
    query_context: Optional[Dict[str, Any]] = None,
):
    """Build generation context from retrieval response."""
    from app.core.generation_prompts import GenerationContext
    from app.schemas import ChunkApplicability, ConflictFlag, EvidenceSufficiency, RetrievalRelevance

    # Convert chunk applicability
    applicability_list = []
    for meta in retrieval_response.chunk_metadata:
        # Reconstruct ChunkApplicability from metadata
        # This is a simplified version - in production, pass the full objects
        applicability_list.append(ChunkApplicability(
            chunk_id=meta.get("chunk_id", ""),
            authority_level=meta.get("authority_level", "departmental"),
            jurisdiction_match=True,
            department_match=True,
            within_effective_window=True,
            is_superseded=False,
            is_authoritative=True,
        ))

    return GenerationContext(
        query=query,
        sub_questions=[sq.question for sq in sub_questions],
        chunks=retrieval_response.chunks,
        chunk_ids=retrieval_response.chunk_ids,
        chunk_metadata=retrieval_response.chunk_metadata,
        chunk_applicability=applicability_list,
        conflict_flag=retrieval_response.conflict_flag,
        evidence_sufficiency=retrieval_response.evidence_sufficiency,
        retrieval_relevance=retrieval_response.retrieval_relevance,
        query_context=query_context,
    )


async def _extract_and_verify_claims(
    parsed_response: Dict[str, Any],
    context,
    budget: ExecutionBudget,
    provider: Optional[LLMProvider] = None,
) -> Tuple[List[ClaimVerificationResult], List[Dict[str, Any]]]:
    """Extract claims from generated answer and verify against chunks."""
    claims = []
    failures = []

    # Get claims from parsed response
    response_claims = parsed_response.get("claims", [])

    # Verify each claim
    for claim_data in response_claims:
        try:
            claim = ClaimVerificationResult(
                is_valid=claim_data["is_valid"],
                confidence=claim_data["confidence"],
                quoted_span=claim_data["quoted_span"],
                chunk_id=claim_data["chunk_id"],
                claim_text=claim_data["claim_text"],
                verification_notes=claim_data.get("verification_notes"),
            )
            claims.append(claim)

            # Additional validation: verify quoted_span actually exists in chunk
            if claim.is_valid:
                _verify_quoted_span(claim, context.chunks, context.chunk_ids)

        except Exception as e:
            failures.append({
                "category": "claim_verification_error",
                "message": f"Failed to verify claim: {str(e)}",
                "retryable": False,
                "details": {"claim_data": claim_data},
            })

    # Check for unverified claims
    unverified = parsed_response.get("unverified_claims", [])
    for uc in unverified:
        failures.append({
            "category": "unverified_claim",
            "message": f"Claim could not be verified: {uc}",
            "retryable": False,
            "details": {"unverified_claim": uc},
        })

    return claims, failures


def _verify_quoted_span(
    claim: ClaimVerificationResult,
    chunks: List[str],
    chunk_ids: List[str],
) -> bool:
    """Verify that the quoted_span exists in the referenced chunk."""
    # Find the chunk
    chunk_idx = None
    for i, cid in enumerate(chunk_ids):
        if cid == claim.chunk_id:
            chunk_idx = i
            break

    if chunk_idx is None:
        raise ValueError(f"Chunk {claim.chunk_id} not found in retrieved chunks")

    chunk_text = chunks[chunk_idx]
    if claim.quoted_span not in chunk_text:
        # Try fuzzy matching (substring)
        if claim.quoted_span.lower() not in chunk_text.lower():
            raise ValueError(
                f"Quoted span not found in chunk {claim.chunk_id}: "
                f"'{claim.quoted_span[:100]}...'"
            )

    return True


def _build_refusal_result(
    trace_id: str,
    query: str,
    sub_questions: List[SubQuestion],
    retrieval_response: FilteredRetrievalResponse,
    refusal_category: RefusalCategory,
    budget: ExecutionBudget,
    trace_chunks: List[Dict[str, Any]],
    trace_failures: List[Dict[str, Any]],
    final_confidence: float = 0.0,
) -> Tuple[GenerationResult, TraceRecord]:
    """Build a refusal result with trace."""

    from app.core.retrieval import _build_refusal_response

    refusal_response = _build_refusal_response(refusal_category, trace_id)

    trace = TraceRecord(
        trace_id=trace_id,
        query_text=query,
        sub_questions=[sq.question for sq in sub_questions],
        chunk_results=trace_chunks,
        claims=[],
        confidence_inputs={
            "retrieval_relevance": retrieval_response.retrieval_relevance.score,
            "evidence_sufficiency": retrieval_response.evidence_sufficiency.score,
            "claim_verifications": [],
            "authority_flags": retrieval_response.authority_flags,
        },
        final_confidence=final_confidence,
        outcome=Outcome(f"refused_{refusal_category.value}"),
        refusal_category=refusal_category,
        execution_budget=budget.to_dict(),
        provider="mock",
        model="retrieval_pipeline_v1",
        retries=budget.retries,
        failures=trace_failures,
        latency_ms=int(budget.wall_time_ms),
    )

    result = GenerationResult(
        answer=None,
        claims=[],
        refusal={
            "refusal_category": refusal_category.value,
            "reason": refusal_response.reason,
            "trace_id": trace_id,
        },
        confidence=final_confidence,
        trace_id=trace_id,
        verification_failures=trace_failures,
    )

    return result, trace


def _claim_to_dict(claim: ClaimVerificationResult) -> Dict[str, Any]:
    """Convert ClaimVerificationResult to dict for trace."""
    return {
        "claim_text": claim.claim_text,
        "is_valid": claim.is_valid,
        "confidence": claim.confidence,
        "quoted_span": claim.quoted_span,
        "chunk_id": claim.chunk_id,
        "verification_notes": claim.verification_notes,
    }


# ============================================================
# Full Pipeline Integration
# ============================================================

async def run_full_pipeline(
    session,
    query: str,
    k: int = 5,
    query_context: Optional[Dict[str, Any]] = None,
    budget: Optional[ExecutionBudget] = None,
    provider: Optional[LLMProvider] = None,
) -> Tuple[Dict[str, Any], TraceRecord]:
    """
    Run the complete pipeline: retrieval + generation.

    Returns (QueryResponse dict, TraceRecord).
    """
    from app.core.retrieval import run_retrieval_pipeline

    trace_id = str(uuid.uuid4())
    if budget is None:
        budget = ExecutionBudget()
    budget.start_wall_timer()

    # Phase 1: Retrieval
    retrieval_response, retrieval_trace = await run_retrieval_pipeline(
        session=session,
        query=query,
        k=k,
        query_context=query_context,
        budget=budget,
    )

    # Check if retrieval already determined a refusal
    if retrieval_response.trace_id and retrieval_trace.outcome != Outcome.ANSWERED:
        # Retrieval phase already refused
        return _build_query_response_from_trace(retrieval_trace), retrieval_trace

    # Phase 2: Generation
    # Convert string sub_questions to SubQuestion objects
    from app.schemas import SubQuestion
    sub_questions = [
        SubQuestion(question=sq, focus="general")
        for sq in retrieval_trace.sub_questions
    ]
    generation_result, generation_trace = await run_generation_phase(
        retrieval_response=retrieval_response,
        query=query,
        sub_questions=sub_questions,
        query_context=query_context,
        budget=budget,
        provider=provider,
    )

    # Merge traces
    merged_trace = _merge_traces(retrieval_trace, generation_trace)

    # Build final response
    query_response = _build_query_response(generation_result, merged_trace)

    return query_response, merged_trace


def _merge_traces(retrieval_trace: TraceRecord, generation_trace: TraceRecord) -> TraceRecord:
    """Merge retrieval and generation traces."""
    return TraceRecord(
        trace_id=generation_trace.trace_id,
        timestamp=generation_trace.timestamp,
        query_text=generation_trace.query_text,
        sub_questions=generation_trace.sub_questions,
        chunk_results=retrieval_trace.chunk_results + generation_trace.chunk_results,
        claims=generation_trace.claims,
        confidence_inputs=generation_trace.confidence_inputs,
        final_confidence=generation_trace.final_confidence,
        outcome=generation_trace.outcome,
        refusal_category=generation_trace.refusal_category,
        execution_budget=generation_trace.execution_budget,
        provider=generation_trace.provider,
        model=generation_trace.model,
        retries=generation_trace.retries,
        failures=retrieval_trace.failures + generation_trace.failures,
        latency_ms=generation_trace.latency_ms,
        numeric_validations=generation_trace.numeric_validations,
    )


def _build_query_response(
    generation_result: GenerationResult,
    trace: TraceRecord,
) -> Dict[str, Any]:
    """Build QueryResponse from generation result."""
    from app.schemas import QueryResponse, RefusalResponse

    if generation_result.refusal:
        refusal = RefusalResponse(
            refusal_category=RefusalCategory(generation_result.refusal["refusal_category"]),
            reason=generation_result.refusal["reason"],
            trace_id=generation_result.refusal["trace_id"],
        )
        return QueryResponse(
            answer=None,
            refusal=refusal,
            trace_record=trace,
            confidence=generation_result.confidence,
        ).model_dump()

    return QueryResponse(
        answer=generation_result.answer,
        refusal=None,
        trace_record=trace,
        confidence=generation_result.confidence,
    ).model_dump()


def _build_query_response_from_trace(trace: TraceRecord) -> Dict[str, Any]:
    """Build QueryResponse from a refusal trace."""
    from app.schemas import QueryResponse, RefusalResponse

    if trace.refusal_category:
        refusal = RefusalResponse(
            refusal_category=trace.refusal_category,
            reason=DEFAULT_REFUSAL_MESSAGES.get(trace.refusal_category, "Unable to answer"),
            trace_id=trace.trace_id,
        )
        return QueryResponse(
            answer=None,
            refusal=refusal,
            trace_record=trace,
            confidence=trace.final_confidence,
        ).model_dump()

    return QueryResponse(
        answer="No answer generated",
        refusal=None,
        trace_record=trace,
        confidence=trace.final_confidence,
    ).model_dump()