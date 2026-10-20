"""Pydantic schemas for Precedent V1 API and internal data structures."""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional, List, Dict, Any, Literal, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# ============================================================
# Enums
# ============================================================

class RefusalCategory(str, Enum):
    """Refusal categories per FR-7.x."""
    NO_EVIDENCE = "no_evidence"
    LOW_CONFIDENCE = "low_confidence"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    INAPPLICABLE_EVIDENCE = "inapplicable_evidence"
    STALE_EVIDENCE = "stale_evidence"
    INSUFFICIENT_AUTHORITY = "insufficient_authority"
    BUDGET_EXCEEDED = "budget_exceeded"
    PROVIDER_ERROR = "provider_error"
    VALIDATION_ERROR = "validation_error"


class Outcome(str, Enum):
    """Query outcome per FR-7.x and FR-8.x."""
    ANSWERED = "answered"
    REFUSED_NO_EVIDENCE = "refused_no_evidence"
    REFUSED_LOW_CONFIDENCE = "refused_low_confidence"
    REFUSED_CONFLICTING_EVIDENCE = "refused_conflicting_evidence"
    REFUSED_INAPPLICABLE_EVIDENCE = "refused_inapplicable_evidence"
    REFUSED_STALE_EVIDENCE = "refused_stale_evidence"
    REFUSED_INSUFFICIENT_AUTHORITY = "refused_insufficient_authority"
    REFUSED_BUDGET_EXCEEDED = "refused_budget_exceeded"
    REFUSED_PROVIDER_ERROR = "refused_provider_error"
    REFUSED_VALIDATION_ERROR = "refused_validation_error"
    REFUSED_LOW_CONFIDENCE_GENERATION = "refused_low_confidence_generation"


class AuthorityLevel(str, Enum):
    """Authority levels per FR-3.x."""
    GLOBAL = "global"
    REGIONAL = "regional"
    DEPARTMENTAL = "departmental"


class DocumentStatus(str, Enum):
    """Document status per FR-3.x."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


class ChunkStatus(str, Enum):
    """Chunk status derived from document."""
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"


# ============================================================
# Ingestion Schemas (FR-1.x, Section 6.1)
# ============================================================

class IngestRequest(BaseModel):
    """Request to ingest a policy document."""
    text: str = Field(..., min_length=1, description="Full document text to ingest")
    policy_id: str = Field(..., min_length=1, description="Unique policy identifier")
    version: str = Field(default="1.0", description="Policy version")
    owner: str = Field(default="system", description="Document owner")
    authority_level: AuthorityLevel = Field(default=AuthorityLevel.GLOBAL, description="Authority level")
    jurisdiction: str = Field(default="headquarters", description="Jurisdiction")
    department: str = Field(default="all", description="Department")
    audience: str = Field(default="internal", description="Intended audience")
    status: DocumentStatus = Field(default=DocumentStatus.ACTIVE, description="Document status")
    effective_from: Optional[date] = Field(default=None, description="Effective from date")
    effective_until: Optional[date] = Field(default=None, description="Effective until date")
    supersedes: Optional[str] = Field(default=None, description="Policy ID this document supersedes")

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text must not be empty or whitespace only")
        return v

    @field_validator("effective_until")
    @classmethod
    def effective_until_after_from(cls, v: Optional[date], info) -> Optional[date]:
        if v is not None and info.data.get("effective_from") is not None:
            if v < info.data["effective_from"]:
                raise ValueError("effective_until must be after effective_from")
        return v


class IngestResponse(BaseModel):
    """Response from document ingestion."""
    document_id: str
    chunk_count: int
    superseded: List[str] = Field(default_factory=list, description="Policy IDs that were superseded")
    content_hash: str


# ============================================================
# Retrieval & Query Schemas (FR-2.x, FR-3.x, FR-4.x, Section 6.2)
# ============================================================

class DecompositionRequest(BaseModel):
    """Request for question decomposition."""
    query: str = Field(..., min_length=1)
    max_sub_questions: int = Field(default=5, ge=1, le=10)


class SubQuestion(BaseModel):
    """A single sub-question from decomposition."""
    question: str
    focus: Optional[str] = None  # e.g., "eligibility", "amount", "process"


class DecompositionResponse(BaseModel):
    """Response from question decomposition."""
    sub_questions: List[SubQuestion]
    original_query: str


class RetrievalRelevance(BaseModel):
    """Retrieval relevance score per FR-2.x."""
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str


class EvidenceSufficiency(BaseModel):
    """Evidence sufficiency assessment per FR-4.x."""
    score: float = Field(..., ge=0.0, le=1.0)
    covered_sub_questions: int
    total_sub_questions: int
    missing_sub_questions: List[str] = Field(default_factory=list)


class ChunkApplicability(BaseModel):
    """Applicability flags for a retrieved chunk per FR-3.x."""
    chunk_id: str
    authority_level: AuthorityLevel
    jurisdiction_match: bool
    department_match: bool
    within_effective_window: bool
    is_superseded: bool
    superseded_by: Optional[str] = None
    is_authoritative: bool  # Combined flag


class ConflictFlag(BaseModel):
    """Conflict detection between chunks per FR-3.4."""
    has_conflict: bool
    conflict_details: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_pairing: Optional[tuple[str, str]] = None  # (chunk_id_a, chunk_id_b)


class AuthorityApplicabilityFlags(BaseModel):
    """Authority and applicability flags for confidence scoring (FR-3.x, FR-6.x)."""
    has_conflict: bool = False
    conflict_details: List[Dict[str, Any]] = Field(default_factory=list)
    authority_level: str = "departmental"
    jurisdiction_match: bool = True
    department_match: bool = True
    within_effective_window: bool = True
    is_superseded: bool = False
    is_authoritative: bool = False


class FilteredRetrievalResponse(BaseModel):
    """Response from retrieval + filtering pipeline (FR-2, FR-3, FR-4)."""
    chunks: List[str] = Field(description="Text content of filtered chunks")
    chunk_ids: List[str]
    chunk_metadata: List[Dict[str, Any]] = Field(default_factory=list)
    found: bool
    empty_corpus: bool = False
    conflict_flag: ConflictFlag
    ranked_ids: List[str]
    trace_id: str
    retrieval_relevance: RetrievalRelevance
    evidence_sufficiency: EvidenceSufficiency
    authority_flags: Dict[str, Any]  # Serialized ChunkApplicability list + conflict


# ============================================================
# Claim Extraction & Verification (FR-5.x)
# ============================================================

class ClaimVerificationResult(BaseModel):
    """Result of verifying a single claim against evidence per FR-5.x."""
    is_valid: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    quoted_span: str = Field(..., description="Exact quote from chunk supporting claim")
    chunk_id: str
    claim_text: str
    verification_notes: Optional[str] = None


class ClaimWithCitations(BaseModel):
    """A claim with its supporting citations per FR-5.x."""
    claim_text: str
    is_valid: bool
    confidence: float
    citations: List[ClaimVerificationResult]


# ============================================================
# Confidence Scoring (FR-6.x)
# ============================================================

class ConfidenceInputs(BaseModel):
    """Inputs to confidence computation per FR-6.x."""
    retrieval_relevance: float = Field(..., ge=0.0, le=1.0)
    evidence_sufficiency: float = Field(..., ge=0.0, le=1.0)
    claim_verifications: List[ClaimVerificationResult] = Field(default_factory=list)
    authority_flags: Dict[str, Any] = Field(default_factory=dict)


class ConfidenceScore(BaseModel):
    """Computed confidence score per FR-6.x."""
    final_confidence: float = Field(..., ge=0.0, le=1.0)
    component_scores: Dict[str, float]
    penalty_applied: bool = False
    penalty_reason: Optional[str] = None


# ============================================================
# Refusal & Response Schemas (FR-7.x)
# ============================================================

class RefusalResponse(BaseModel):
    """Structured refusal response per FR-7.x."""
    refusal_category: RefusalCategory
    reason: str
    trace_id: str
    suggestions: List[str] = Field(default_factory=list)


class SystemRefusal(BaseModel):
    """Internal refusal representation."""
    category: RefusalCategory
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


DEFAULT_REFUSAL_MESSAGES: Dict[RefusalCategory, str] = {
    RefusalCategory.NO_EVIDENCE: "I couldn't find any relevant policy documents to answer your question.",
    RefusalCategory.LOW_CONFIDENCE: "The available evidence is insufficient to provide a confident answer.",
    RefusalCategory.CONFLICTING_EVIDENCE: "Multiple authoritative policies conflict on this topic. Please consult your compliance team.",
    RefusalCategory.INAPPLICABLE_EVIDENCE: "The retrieved policies don't apply to your specific situation (jurisdiction, department, or role).",
    RefusalCategory.STALE_EVIDENCE: "The relevant policy has been superseded or is no longer in effect.",
    RefusalCategory.INSUFFICIENT_AUTHORITY: "The retrieved policy lacks sufficient authority for this question.",
    RefusalCategory.BUDGET_EXCEEDED: "The query exceeded execution budget limits.",
    RefusalCategory.PROVIDER_ERROR: "A system error occurred while processing your query.",
    RefusalCategory.VALIDATION_ERROR: "The generated response failed validation checks.",
}


# ============================================================
# Trace Record (FR-8.x)
# ============================================================

class TraceRecord(BaseModel):
    """Complete audit trace for a query per FR-8.1 + FR-9."""
    trace_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    query_text: str
    sub_questions: List[str] = Field(default_factory=list)
    chunk_results: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_inputs: Dict[str, Any] = Field(default_factory=dict)
    final_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    outcome: Outcome
    refusal_category: Optional[RefusalCategory] = None
    execution_budget: Dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    retries: int = 0
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0
    numeric_validations: List[Dict[str, Any]] = Field(default_factory=list)


# ============================================================
# API Endpoint Schemas (Section 6)
# ============================================================

class QueryRequest(BaseModel):
    """POST /query request."""
    query: str = Field(..., min_length=1, max_length=5000)
    k: int = Field(default=5, ge=1, le=20)
    trace: bool = Field(default=True, description="Include full trace in response")


class QueryResponse(BaseModel):
    """POST /query response."""
    answer: Optional[str] = None
    refusal: Optional[RefusalResponse] = None
    trace_record: Optional[TraceRecord] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class TraceResponse(BaseModel):
    """GET /trace/{trace_id} response."""
    trace: TraceRecord


# ============================================================
# Iteration Request (for multi-step retrieval if needed)
# ============================================================

class IterationRequest(BaseModel):
    """Request for iterative retrieval step."""
    query: str
    sub_questions: List[str]
    previous_chunks: List[str]
    iteration: int
    max_iterations: int = 3


class IterationResponse(BaseModel):
    """Response from iterative retrieval."""
    new_chunks: List[str]
    new_chunk_ids: List[str]
    should_continue: bool
    reason: str


# ============================================================
# Retrieval Response (internal)
# ============================================================

class RetrievalResponse(BaseModel):
    """Internal retrieval response before filtering."""
    chunks: List[str]
    chunk_ids: List[str]
    similarities: List[float]
    metadatas: List[Dict[str, Any]]