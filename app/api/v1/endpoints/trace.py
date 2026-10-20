"""GET /trace/{trace_id} endpoint for Precedent V1 (FR-8.x, Section 6.3)."""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlmodel import Session, select

from app.models import Document, PolicyChunk
from app.schemas import TraceRecord, TraceResponse
from app.core.database import get_session
from app.core.config import settings

router = APIRouter(prefix="/trace", tags=["trace"])


# In-memory trace store for V1 (replace with persistent storage in production)
_trace_store: dict[str, TraceRecord] = {}


def store_trace(trace: TraceRecord) -> None:
    """Store trace record in memory."""
    _trace_store[trace.trace_id] = trace


def get_trace(trace_id: str) -> Optional[TraceRecord]:
    """Retrieve trace record from memory."""
    return _trace_store.get(trace_id)


@router.get("/{trace_id}", response_model=TraceResponse, status_code=status.HTTP_200_OK)
async def get_trace_endpoint(
    trace_id: str,
    authorization: str = Header(...),
) -> TraceResponse:
    """
    Retrieve audit trace for a query.

    Returns the complete TraceRecord per FR-8.1 including:
    - Query and sub-questions
    - Retrieved chunks and relevance scores
    - Claims and citations
    - Confidence inputs and final confidence
    - Outcome and refusal category (if any)
    - Execution budget
    - Provider and model info
    - Failures and latency
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

    trace = get_trace(trace_id)
    if trace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace not found: {trace_id}",
        )

    return TraceResponse(trace=trace)