"""POST /retrieval/retrieve endpoint for Precedent V1 (FR-2.x, FR-3.x, FR-4.x, FR-7.x, Section 6.2)."""

from __future__ import annotations

from typing import Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlmodel import Session
from pydantic import BaseModel, Field

from app.models import PolicyChunk
from app.schemas import (
    QueryRequest,
    QueryResponse,
    FilteredRetrievalResponse,
    RefusalResponse,
    TraceRecord,
    Outcome,
    RefusalCategory,
)
from app.core.database import get_session
from app.core.retrieval import run_retrieval_pipeline
from app.core.budget import ExecutionBudget
from app.core.config import settings
from app.api.v1.endpoints.trace import store_trace

router = APIRouter(prefix="/retrieval", tags=["retrieval-phase2"])


class RetrieveRequest(BaseModel):
    """Request for retrieval endpoint."""
    query: str = Field(default="", max_length=5000)
    k: int = Field(default=5, ge=1, le=20)
    jurisdiction: Optional[str] = Field(default=None)
    department: Optional[str] = Field(default=None)
    authority_level: Optional[str] = Field(default=None)


class RetrieveResponse(BaseModel):
    """Response from retrieval endpoint."""
    chunks: list[str]
    chunk_ids: list[str]
    chunk_metadata: list[Dict[str, Any]]
    found: bool
    empty_corpus: bool
    conflict_flag: Dict[str, Any]
    ranked_ids: list[str]
    trace_id: str
    retrieval_relevance: Dict[str, Any]
    evidence_sufficiency: Dict[str, Any]
    authority_flags: Dict[str, Any]
    trace_record: Optional[TraceRecord] = None


@router.post("/retrieve", response_model=RetrieveResponse, status_code=status.HTTP_200_OK)
async def retrieve(
    request: RetrieveRequest,
    session: Session = Depends(get_session),
    authorization: str = Header(...),
) -> RetrieveResponse:
    """
    Retrieve and filter policy chunks for a query.

    Pipeline:
    1. Decompose question into sub-questions
    2. Retrieve relevant chunks (simple text matching in V1)
    3. Filter by authority, applicability, temporal validity
    4. Detect conflicts
    5. Assess evidence sufficiency
    6. Compute confidence
    7. Return filtered results with trace
    """
    # Verify authorization
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    token = authorization[7:]
    if token != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )

    # Build query context
    query_context = {}
    if request.jurisdiction:
        query_context["jurisdiction"] = request.jurisdiction
    if request.department:
        query_context["department"] = request.department
    if request.authority_level:
        query_context["authority_level"] = request.authority_level

    # Create budget
    budget = ExecutionBudget(
        max_llm_calls=settings.max_llm_calls,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_retries,
        max_wall_time_ms=settings.max_wall_time_ms,
        max_estimated_cost=settings.max_estimated_cost,
    )

    try:
        # Run retrieval pipeline
        filtered_response, trace = await run_retrieval_pipeline(
            session=session,
            query=request.query,
            k=request.k,
            query_context=query_context,
            budget=budget,
        )

        # Store trace for later retrieval
        store_trace(trace)

        # Determine if we should include full trace
        include_trace = True  # Could be controlled by request parameter

        return RetrieveResponse(
            chunks=filtered_response.chunks,
            chunk_ids=filtered_response.chunk_ids,
            chunk_metadata=filtered_response.chunk_metadata,
            found=filtered_response.found,
            empty_corpus=filtered_response.empty_corpus,
            conflict_flag={
                "has_conflict": filtered_response.conflict_flag.has_conflict,
                "conflict_details": filtered_response.conflict_flag.conflict_details,
                "conflict_pairing": filtered_response.conflict_flag.conflict_pairing,
            },
            ranked_ids=filtered_response.ranked_ids,
            trace_id=filtered_response.trace_id,
            retrieval_relevance={
                "score": filtered_response.retrieval_relevance.score,
                "reason": filtered_response.retrieval_relevance.reason,
            },
            evidence_sufficiency={
                "score": filtered_response.evidence_sufficiency.score,
                "covered_sub_questions": filtered_response.evidence_sufficiency.covered_sub_questions,
                "total_sub_questions": filtered_response.evidence_sufficiency.total_sub_questions,
                "missing_sub_questions": filtered_response.evidence_sufficiency.missing_sub_questions,
            },
            authority_flags=filtered_response.authority_flags,
            trace_record=trace if include_trace else None,
        )

    except Exception as e:
        # Handle budget violations and other errors
        from app.core.errors import BudgetViolation, ProviderError
        if isinstance(e, BudgetViolation):
            trace_id = str(__import__("uuid").uuid4())
            trace = TraceRecord(
                trace_id=trace_id,
                query_text=request.query,
                sub_questions=[],
                chunk_results=[],
                claims=[],
                confidence_inputs={},
                final_confidence=0.0,
                outcome=Outcome.REFUSED_BUDGET_EXCEEDED,
                refusal_category=RefusalCategory.BUDGET_EXCEEDED,
                execution_budget=budget.to_dict(),
                provider="budget_system",
                model="budget_enforcement",
                retries=budget.retries,
                failures=[{"category": "budget", "message": str(e)}],
                latency_ms=int(budget.wall_time_ms),
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "category": "budget",
                        "retryable": False,
                        "safe_message": "Query exceeded execution budget limits",
                        "diagnostic_code": f"ERR-BUDGET-{type(e).__name__}",
                    },
                    "trace_id": trace_id,
                    "execution_budget": budget.to_dict(),
                },
            )
        elif isinstance(e, ProviderError):
            trace_id = str(__import__("uuid").uuid4())
            trace = TraceRecord(
                trace_id=trace_id,
                query_text=request.query,
                sub_questions=[],
                chunk_results=[],
                claims=[],
                confidence_inputs={},
                final_confidence=0.0,
                outcome=Outcome.REFUSED_PROVIDER_ERROR,
                refusal_category=RefusalCategory.PROVIDER_ERROR,
                execution_budget=budget.to_dict(),
                provider="mock",
                model="retrieval_pipeline_v1",
                retries=budget.retries,
                failures=[{"category": e.category.value, "message": e.safe_message, "retryable": e.retryable}],
                latency_ms=int(budget.wall_time_ms),
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "category": e.category.value,
                        "retryable": e.retryable,
                        "safe_message": e.safe_message,
                        "diagnostic_code": e.diagnostic_code,
                    },
                    "trace_id": trace_id,
                    "execution_budget": budget.to_dict(),
                },
            )
        raise


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query(
    request: QueryRequest,
    session: Session = Depends(get_session),
    authorization: str = Header(...),
) -> QueryResponse:
    """
    Full query endpoint: retrieve + generate answer.

    For V1, this runs the retrieval pipeline and returns the trace.
    Generation phase would be added here in full implementation.
    """
    # Verify authorization
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header",
        )
    token = authorization[7:]
    if token != settings.api_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )

    # Build query context (could be extended with user context)
    query_context = {}

    # Create budget
    budget = ExecutionBudget(
        max_llm_calls=settings.max_llm_calls,
        max_tokens=settings.max_tokens,
        max_retries=settings.max_retries,
        max_wall_time_ms=settings.max_wall_time_ms,
        max_estimated_cost=settings.max_estimated_cost,
    )

    try:
        # Run retrieval pipeline
        filtered_response, trace = await run_retrieval_pipeline(
            session=session,
            query=request.query,
            k=request.k,
            query_context=query_context,
            budget=budget,
        )

        # Determine response based on trace outcome
        if trace.outcome == Outcome.ANSWERED:
            # In full implementation, generate answer here
            # For V1, return a placeholder answer based on retrieved chunks
            answer = _generate_placeholder_answer(filtered_response.chunks, request.query)
            return QueryResponse(
                answer=answer,
                refusal=None,
                trace_record=trace if request.trace else None,
                confidence=trace.final_confidence,
            )
        else:
            # Refusal
            refusal = RefusalResponse(
                refusal_category=trace.refusal_category or RefusalCategory.LOW_CONFIDENCE,
                reason=trace.confidence_inputs.get("authority_flags", {}).get("refusal_reason", "Unable to answer"),
                trace_id=trace.trace_id,
            )
            return QueryResponse(
                answer=None,
                refusal=refusal,
                trace_record=trace if request.trace else None,
                confidence=trace.final_confidence,
            )

    except Exception as e:
        from app.core.errors import BudgetViolation, ProviderError
        if isinstance(e, BudgetViolation):
            trace_id = str(__import__("uuid").uuid4())
            refusal = RefusalResponse(
                refusal_category=RefusalCategory.BUDGET_EXCEEDED,
                reason="Query exceeded execution budget limits",
                trace_id=trace_id,
            )
            return QueryResponse(
                answer=None,
                refusal=refusal,
                trace_record=None,
                confidence=0.0,
            )
        elif isinstance(e, ProviderError):
            trace_id = str(__import__("uuid").uuid4())
            refusal = RefusalResponse(
                refusal_category=RefusalCategory.PROVIDER_ERROR,
                reason=e.safe_message,
                trace_id=trace_id,
            )
            return QueryResponse(
                answer=None,
                refusal=refusal,
                trace_record=None,
                confidence=0.0,
            )
        raise


def _generate_placeholder_answer(chunks: list[str], query: str) -> str:
    """Generate a placeholder answer from retrieved chunks."""
    if not chunks:
        return "No relevant policy information found."

    # Simple concatenation for V1
    combined = " ".join(chunks[:3])  # Use top 3 chunks
    return f"Based on the retrieved policy documents: {combined[:500]}..."