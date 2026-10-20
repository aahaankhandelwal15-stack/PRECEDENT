"""Retrieval pipeline for Precedent V1 (FR-1.x through FR-7.x)."""

from __future__ import annotations

import uuid
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import date

from app.models import PolicyChunk
from app.schemas import (
    SubQuestion,
    DecompositionResponse,
    FilteredRetrievalResponse,
    RetrievalRelevance,
    EvidenceSufficiency,
    ConflictFlag,
    ChunkApplicability,
    ClaimVerificationResult,
    AuthorityApplicabilityFlags,
    RefusalCategory,
    SystemRefusal,
    RefusalResponse,
    TraceRecord,
    DEFAULT_REFUSAL_MESSAGES,
    Outcome,
)
from app.core.config import settings
from app.core.budget import ExecutionBudget
from app.core.confidence_scorer import (
    compute_confidence_from_retrieval,
    compute_confidence_from_generation,
    should_refuse,
)
from app.core.errors import ProviderError
from sqlmodel import Session, select, func
import re


# ============================================================
# Constants
# ============================================================

DEFAULT_CONFIDENCE_THRESHOLD = settings.confidence_threshold
MAX_ITERATIONS = 3

# Simple keyword-based decomposition patterns
DECOMPOSITION_PATTERNS = [
    (r"what (?:is|are) (?:the )?(.+?)(?:\?|$)", "definition"),
    (r"how (?:to|do|does) (.+?)(?:\?|$)", "process"),
    (r"when (.+?)(?:\?|$)", "timing"),
    (r"where (.+?)(?:\?|$)", "location"),
    (r"who (.+?)(?:\?|$)", "actor"),
    (r"why (.+?)(?:\?|$)", "reason"),
    (r"(?:eligibility|eligible) (.+?)(?:\?|$)", "eligibility"),
    (r"(?:maximum|max|limit|cap) (.+?)(?:\?|$)", "maximum"),
    (r"(?:minimum|min) (.+?)(?:\?|$)", "minimum"),
    (r"(?:approval|approve) (.+?)(?:\?|$)", "approval"),
    (r"(?:reimburse|reimbursement) (.+?)(?:\?|$)", "reimbursement"),
]


# ============================================================
# Question Decomposition (FR-2.1)
# ============================================================

def decompose_question(query: str, max_sub_questions: int = 5) -> DecompositionResponse:
    """
    Decompose a complex question into sub-questions.
    Uses simple pattern matching for V1 (can be replaced with LLM later).
    """
    query_lower = query.lower().strip()
    sub_questions = []

    # Try to split by common conjunctions first
    # Split on "and", "or", ",", ";" when they separate question parts
    parts = re.split(r'\s+(?:and|or)\s+|[,;]\s*', query_lower)

    # Filter out empty parts and very short parts
    parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 10]

    if len(parts) > 1:
        # Multiple parts detected
        for part in parts[:max_sub_questions]:
            # Ensure it ends with question mark
            if not part.endswith('?'):
                part = part + '?'
            sub_questions.append(SubQuestion(question=part, focus=_infer_focus(part)))
    else:
        # Single question - try pattern-based decomposition
        sub_questions = _pattern_decompose(query, max_sub_questions)

    # If no decomposition found, use original query as single sub-question
    if not sub_questions:
        sub_questions = [SubQuestion(question=query, focus=_infer_focus(query))]

    return DecompositionResponse(
        sub_questions=sub_questions[:max_sub_questions],
        original_query=query,
    )


def _pattern_decompose(query: str, max_sub: int) -> List[SubQuestion]:
    """Decompose using regex patterns."""
    sub_questions = []
    query_lower = query.lower()

    for pattern, focus in DECOMPOSITION_PATTERNS:
        matches = re.findall(pattern, query_lower)
        for match in matches:
            if len(match.strip()) > 5:
                sq = match.strip()
                if not sq.endswith('?'):
                    sq += '?'
                sub_questions.append(SubQuestion(question=sq, focus=focus))

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for sq in sub_questions:
        if sq.question not in seen:
            seen.add(sq.question)
            unique.append(sq)

    return unique[:max_sub]


def _infer_focus(question: str) -> str:
    """Infer focus area from question text."""
    q = question.lower()
    if any(w in q for w in ["eligib", "qualif"]):
        return "eligibility"
    if any(w in q for w in ["maximum", "max", "limit", "cap", "amount"]):
        return "maximum"
    if any(w in q for w in ["minimum", "min"]):
        return "minimum"
    if any(w in q for w in ["approval", "approve", "authoriz"]):
        return "approval"
    if any(w in q for w in ["reimburse", "reimbursement", "expense"]):
        return "reimbursement"
    if any(w in q for w in ["process", "procedure", "how"]):
        return "process"
    return "general"


# ============================================================
# Chunk Retrieval (FR-2.2, FR-2.3)
# ============================================================

def retrieve_chunks(
    session: Session,
    query: str,
    k: int = 5,
    sub_questions: Optional[List[str]] = None,
) -> Tuple[List[PolicyChunk], List[float]]:
    """
    Retrieve relevant chunks for a query.
    V1 uses simple text matching (replace with vector search in production).
    Returns (chunks, similarity_scores).
    """
    # For V1, use simple ILIKE matching on text
    # In production, this would use vector embeddings
    query_terms = _extract_query_terms(query)

    # Also search for sub-questions if provided
    all_terms = set(query_terms)
    if sub_questions:
        for sq in sub_questions:
            all_terms.update(_extract_query_terms(sq))

    # Build query conditions
    conditions = []
    for term in all_terms:
        if len(term) > 2:  # Skip very short terms
            conditions.append(PolicyChunk.text.ilike(f"%{term}%"))

    if not conditions:
        return [], []

    # Combine with OR
    from sqlalchemy import or_
    stmt = select(PolicyChunk).where(or_(*conditions)).limit(k * 2)

    chunks = session.exec(stmt).all()

    # Score by term overlap (simple heuristic)
    scored_chunks = []
    for chunk in chunks:
        score = _compute_overlap_score(chunk.text, all_terms)
        scored_chunks.append((chunk, score))

    # Sort by score descending
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top k
    top_chunks = [c for c, s in scored_chunks[:k]]
    scores = [s for c, s in scored_chunks[:k]]

    return top_chunks, scores


def _extract_query_terms(query: str) -> List[str]:
    """Extract meaningful terms from query."""
    # Remove common stop words - keep domain-specific terms!
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "what", "how", "when", "where", "who", "why",
        "for", "of", "to", "in", "on", "at", "by", "with", "from",
    }

    # Extract words
    words = re.findall(r'\b\w+\b', query.lower())
    terms = [w for w in words if w not in stop_words and len(w) > 2]

    # Also extract multi-word phrases (bigrams)
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stop_words and words[i+1] not in stop_words:
            bigrams.append(f"{words[i]} {words[i+1]}")

    return terms + bigrams


def _compute_overlap_score(text: str, query_terms: set) -> float:
    """Simple term overlap scoring."""
    text_lower = text.lower()
    matches = sum(1 for term in query_terms if term in text_lower)
    return matches / max(1, len(query_terms))


# ============================================================
# Authority & Applicability Filtering (FR-3.x)
# ============================================================

def filter_chunks_by_authority(
    chunks: List[PolicyChunk],
    query_context: Dict[str, Any],
    current_date: Optional[date] = None,
) -> Tuple[List[PolicyChunk], List[ChunkApplicability], ConflictFlag]:
    """
    Filter chunks by authority, applicability, and temporal validity.
    Returns (filtered_chunks, applicability_flags, conflict_flag).
    """
    if current_date is None:
        current_date = date.today()

    query_jurisdiction = query_context.get("jurisdiction", "headquarters")
    query_department = query_context.get("department", "all")
    query_authority_min = query_context.get("authority_level", "departmental")

    authority_order = {"global": 3, "regional": 2, "departmental": 1}
    min_authority_level = authority_order.get(query_authority_min, 1)

    filtered = []
    applicability_list = []
    chunk_by_policy = {}  # For conflict detection

    for chunk in chunks:
        # Check temporal validity (FR-3.2, FR-3.3)
        within_window = True
        if chunk.effective_from and chunk.effective_from > current_date:
            within_window = False
        if chunk.effective_until and chunk.effective_until < current_date:
            within_window = False

        # Check status
        is_active = chunk.status == "active"
        is_superseded = chunk.status == "superseded" or chunk.superseded_by is not None

        # Check jurisdiction match
        jurisdiction_match = (
            chunk.jurisdiction == query_jurisdiction or
            chunk.jurisdiction == "all" or
            query_jurisdiction == "all"
        )

        # Check department match
        department_match = (
            chunk.department == query_department or
            chunk.department == "all" or
            query_department == "all"
        )

        # Check authority level meets minimum
        chunk_authority = authority_order.get(chunk.authority_level, 1)
        authority_sufficient = chunk_authority >= min_authority_level

        # Overall applicability
        is_applicable = (
            is_active and
            within_window and
            jurisdiction_match and
            department_match and
            authority_sufficient and
            not is_superseded
        )

        applicability = ChunkApplicability(
            chunk_id=chunk.chunk_id,
            authority_level=chunk.authority_level,
            jurisdiction_match=jurisdiction_match,
            department_match=department_match,
            within_effective_window=within_window,
            is_superseded=is_superseded,
            superseded_by=chunk.superseded_by,
            is_authoritative=is_applicable,
        )
        applicability_list.append(applicability)

        # Track for conflict detection
        policy_key = f"{chunk.policy_id}:{chunk.version}"
        if policy_key not in chunk_by_policy:
            chunk_by_policy[policy_key] = []
        chunk_by_policy[policy_key].append(chunk)

        if is_applicable:
            filtered.append(chunk)

    # Conflict detection (FR-3.4)
    conflict_flag = _detect_conflicts(chunk_by_policy, filtered)

    return filtered, applicability_list, conflict_flag


def _detect_conflicts(
    chunk_by_policy: Dict[str, List[PolicyChunk]],
    filtered_chunks: List[PolicyChunk],
) -> ConflictFlag:
    """Detect conflicting policies among ALL retrieved chunks (not just filtered)."""
    # Use all chunks from chunk_by_policy (which contains all retrieved chunks)
    all_chunks = []
    for chunks in chunk_by_policy.values():
        all_chunks.extend(chunks)

    # Group by policy_id AND version
    policies = {}
    for chunk in all_chunks:
        policy_key = f"{chunk.policy_id}:{chunk.version}"
        if policy_key not in policies:
            policies[policy_key] = []
        policies[policy_key].append(chunk)

    # Check for multiple active versions of same policy without supersedes
    conflict_details = []
    conflict_pairing = None

    # Group by policy_id (ignoring version) to detect version conflicts
    policy_versions = {}
    for chunk in all_chunks:
        if chunk.policy_id not in policy_versions:
            policy_versions[chunk.policy_id] = []
        policy_versions[chunk.policy_id].append(chunk)

    for policy_id, chunks in policy_versions.items():
        # Get unique versions
        versions = {}
        for c in chunks:
            if c.version not in versions:
                versions[c.version] = []
            versions[c.version].append(c)

        # Check if there are multiple versions that are both active
        active_versions = [v for v, cs in versions.items() if any(c.status == "active" for c in cs)]
        if len(active_versions) > 1:
            # Check if they have supersedes relationship
            all_active = [c for c in chunks if c.status == "active"]
            has_supersedes = any(c.supersedes or c.superseded_by for c in all_active)
            if not has_supersedes:
                # Conflict between versions!
                for i, v1 in enumerate(active_versions):
                    for v2 in active_versions[i+1:]:
                        c1 = versions[v1][0]  # representative chunk
                        c2 = versions[v2][0]
                        conflict_details.append({
                            "policy_id": policy_id,
                            "version_a": v1,
                            "version_b": v2,
                            "chunk_a": c1.chunk_id,
                            "chunk_b": c2.chunk_id,
                            "text_a": c1.text[:200],
                            "text_b": c2.text[:200],
                        })
                        if conflict_pairing is None:
                            conflict_pairing = (c1.chunk_id, c2.chunk_id)

    return ConflictFlag(
        has_conflict=len(conflict_details) > 0,
        conflict_details=conflict_details,
        conflict_pairing=conflict_pairing,
    )


# ============================================================
# Evidence Sufficiency (FR-4.x)
# ============================================================

def assess_evidence_sufficiency(
    sub_questions: List[SubQuestion],
    filtered_chunks: List[PolicyChunk],
    chunk_applicability: List[ChunkApplicability],
) -> EvidenceSufficiency:
    """
    Assess if retrieved evidence sufficiently covers all sub-questions.
    """
    if not sub_questions:
        return EvidenceSufficiency(
            score=1.0,
            covered_sub_questions=0,
            total_sub_questions=0,
            missing_sub_questions=[],
        )

    # For each sub-question, check if we have at least one applicable chunk
    covered = 0
    missing = []

    for sq in sub_questions:
        # Simple keyword matching to see if any chunk covers this sub-question
        sq_terms = set(_extract_query_terms(sq.question))
        has_coverage = False

        for chunk, applicability in zip(filtered_chunks, chunk_applicability):
            if not applicability.is_authoritative:
                continue
            chunk_terms = set(_extract_query_terms(chunk.text))
            # Flexible matching: check if any query term is substring of chunk term or vice versa
            overlap = False
            for qterm in sq_terms:
                for cterm in chunk_terms:
                    if qterm in cterm or cterm in qterm:
                        overlap = True
                        break
                if overlap:
                    break
            if overlap:
                has_coverage = True
                break

        if has_coverage:
            covered += 1
        else:
            missing.append(sq.question)

    total = len(sub_questions)
    score = covered / total if total > 0 else 1.0

    return EvidenceSufficiency(
        score=score,
        covered_sub_questions=covered,
        total_sub_questions=total,
        missing_sub_questions=missing,
    )


# ============================================================
# Retrieval Relevance (FR-2.x)
# ============================================================

def assess_retrieval_relevance(
    query: str,
    retrieved_chunks: List[PolicyChunk],
) -> RetrievalRelevance:
    """Assess how relevant retrieved chunks are to the query."""
    if not retrieved_chunks:
        return RetrievalRelevance(score=0.0, reason="No chunks retrieved")

    query_terms = set(_extract_query_terms(query))
    total_score = 0.0

    for chunk in retrieved_chunks:
        chunk_terms = set(_extract_query_terms(chunk.text))
        overlap = query_terms & chunk_terms
        score = len(overlap) / max(1, len(query_terms))
        total_score += score

    avg_score = total_score / len(retrieved_chunks)

    if avg_score >= 0.7:
        reason = "High relevance - strong term overlap"
    elif avg_score >= 0.4:
        reason = "Moderate relevance - partial term overlap"
    elif avg_score > 0:
        reason = "Low relevance - minimal term overlap"
    else:
        reason = "No relevance - no term overlap"

    return RetrievalRelevance(score=avg_score, reason=reason)


# ============================================================
# Refusal Logic (FR-7.x)
# ============================================================

def _determine_refusal_category(
    found: bool,
    empty_corpus: bool,
    conflict_flag: ConflictFlag,
    authority_flags: Dict[str, Any],
    final_confidence: float,
    evidence_sufficiency: float,
    retrieval_relevance: float,
) -> RefusalCategory:
    """
    Determine the appropriate refusal category based on pipeline state.
    This is the corrected version per diagnosis_table.md.
    """
    # Empty corpus short-circuit
    if empty_corpus or not found:
        return RefusalCategory.NO_EVIDENCE

    # Conflict detection (highest priority after empty corpus)
    if conflict_flag.has_conflict:
        return RefusalCategory.CONFLICTING_EVIDENCE

    # Authority/applicability issues
    if not authority_flags.get("within_effective_window", True):
        return RefusalCategory.STALE_EVIDENCE

    if authority_flags.get("is_superseded", False):
        return RefusalCategory.STALE_EVIDENCE

    if not authority_flags.get("jurisdiction_match", True):
        return RefusalCategory.INAPPLICABLE_EVIDENCE

    if not authority_flags.get("department_match", True):
        return RefusalCategory.INAPPLICABLE_EVIDENCE

    if not authority_flags.get("is_authoritative", True):
        return RefusalCategory.INSUFFICIENT_AUTHORITY

    # Evidence sufficiency
    if evidence_sufficiency < settings.min_evidence_sufficiency:
        return RefusalCategory.INAPPLICABLE_EVIDENCE

    if retrieval_relevance < settings.min_retrieval_relevance:
        return RefusalCategory.NO_EVIDENCE

    # Confidence-based
    if final_confidence < DEFAULT_CONFIDENCE_THRESHOLD:
        return RefusalCategory.LOW_CONFIDENCE

    # Should not reach here if called correctly
    return RefusalCategory.LOW_CONFIDENCE


def _build_refusal_response(
    refusal_category: RefusalCategory,
    trace_id: str,
    details: Optional[Dict[str, Any]] = None,
) -> RefusalResponse:
    """Build a structured refusal response."""
    return RefusalResponse(
        refusal_category=refusal_category,
        reason=DEFAULT_REFUSAL_MESSAGES.get(refusal_category, "Unable to answer"),
        trace_id=trace_id,
        suggestions=_get_refusal_suggestions(refusal_category),
    )


def _get_refusal_suggestions(category: RefusalCategory) -> List[str]:
    """Get helpful suggestions for each refusal category."""
    suggestions = {
        RefusalCategory.NO_EVIDENCE: [
            "Try rephrasing your question with different keywords",
            "Check if the policy document has been ingested",
            "Verify the policy ID and version",
        ],
        RefusalCategory.LOW_CONFIDENCE: [
            "The available evidence is insufficient for a confident answer",
            "Consider consulting the policy document directly",
            "Contact your compliance team for clarification",
        ],
        RefusalCategory.CONFLICTING_EVIDENCE: [
            "Multiple authoritative policies conflict on this topic",
            "Consult your compliance team for resolution",
            "Check for supersession metadata in policy documents",
        ],
        RefusalCategory.INAPPLICABLE_EVIDENCE: [
            "The retrieved policies may not apply to your jurisdiction/department",
            "Verify your query context (jurisdiction, department, role)",
            "Check policy effective dates",
        ],
        RefusalCategory.STALE_EVIDENCE: [
            "The relevant policy has been superseded or expired",
            "Check for updated policy versions",
            "Verify effective dates in the policy metadata",
        ],
        RefusalCategory.INSUFFICIENT_AUTHORITY: [
            "The retrieved policy lacks sufficient authority for this question",
            "A higher-authority policy may be needed",
            "Consult your compliance team",
        ],
        RefusalCategory.BUDGET_EXCEEDED: [
            "The query exceeded execution budget limits",
            "Try a simpler or more specific question",
        ],
        RefusalCategory.PROVIDER_ERROR: [
            "A system error occurred",
            "Please try again later",
        ],
        RefusalCategory.VALIDATION_ERROR: [
            "The generated response failed validation",
            "Please try again",
        ],
    }
    return suggestions.get(category, [])


def _should_short_circuit_to_refusal(
    found: bool,
    empty_corpus: bool,
    conflict_flag: ConflictFlag,
    authority_flags: Dict[str, Any],
    evidence_sufficiency: EvidenceSufficiency,
    retrieval_relevance: RetrievalRelevance,
) -> Tuple[bool, Optional[RefusalCategory]]:
    """
    Check if we should short-circuit to refusal without generation.
    Returns (should_refuse, refusal_category).
    """
    # Empty corpus
    if empty_corpus or not found:
        return True, RefusalCategory.NO_EVIDENCE

    # Unresolved conflict
    if conflict_flag.has_conflict:
        return True, RefusalCategory.CONFLICTING_EVIDENCE

    # Stale evidence
    if not authority_flags.get("within_effective_window", True):
        return True, RefusalCategory.STALE_EVIDENCE

    if authority_flags.get("is_superseded", False):
        return True, RefusalCategory.STALE_EVIDENCE

    # Insufficient evidence
    if evidence_sufficiency.score < settings.min_evidence_sufficiency:
        return True, RefusalCategory.INAPPLICABLE_EVIDENCE

    if retrieval_relevance.score < settings.min_retrieval_relevance:
        return True, RefusalCategory.NO_EVIDENCE

    return False, None


# ============================================================
# Bounded Iterator (for iterative retrieval if needed)
# ============================================================

@dataclass
class BoundedIterator:
    """Iterator with budget and iteration bounds."""
    budget: ExecutionBudget
    max_iterations: int = MAX_ITERATIONS
    current_iteration: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.current_iteration >= self.max_iterations:
            raise StopIteration
        self.current_iteration += 1
        self.budget.assert_all_limits(f"iteration_{self.current_iteration}")
        return self.current_iteration

    def has_next(self) -> bool:
        return self.current_iteration < self.max_iterations


# ============================================================
# Main Pipeline Function
# ============================================================

async def run_retrieval_pipeline(
    session: Session,
    query: str,
    k: int = 5,
    query_context: Optional[Dict[str, Any]] = None,
    budget: Optional[ExecutionBudget] = None,
) -> Tuple[FilteredRetrievalResponse, TraceRecord]:
    """
    Run the complete retrieval pipeline:
    1. Decompose question
    2. Retrieve chunks
    3. Filter by authority/applicability
    4. Assess evidence sufficiency
    5. Compute confidence
    6. Determine refusal or proceed to generation

    Returns (FilteredRetrievalResponse, TraceRecord)
    """
    trace_id = str(uuid.uuid4())
    if budget is None:
        budget = ExecutionBudget()
    budget.start_wall_timer()

    query_context = query_context or {}
    trace_chunks = []
    trace_failures = []

    try:
        # Step 1: Decompose
        budget.assert_llm_calls_limit("decompose_question")
        decomposition = decompose_question(query, max_sub_questions=5)
        budget.consume_llm_call(cost=0.001)  # Mock cost

        # Step 2: Retrieve
        budget.assert_llm_calls_limit("retrieve_chunks")
        chunks, similarities = retrieve_chunks(
            session, query, k=k, sub_questions=[sq.question for sq in decomposition.sub_questions]
        )
        budget.consume_llm_call(cost=0.001)

        # Build trace chunk results
        for chunk, sim in zip(chunks, similarities):
            trace_chunks.append({
                "chunk_id": chunk.chunk_id,
                "similarity": sim,
                "text_preview": chunk.text[:200],
                "policy_id": chunk.policy_id,
                "version": chunk.version,
                "authority_level": chunk.authority_level,
            })

        # Step 3: Filter by authority/applicability
        filtered_chunks, applicability_list, conflict_flag = filter_chunks_by_authority(
            chunks, query_context
        )

        # Step 4: Assess evidence sufficiency
        evidence_sufficiency = assess_evidence_sufficiency(
            decomposition.sub_questions, filtered_chunks, applicability_list
        )

        # Step 5: Assess retrieval relevance
        retrieval_relevance = assess_retrieval_relevance(query, chunks)

        # Step 6: Compute authority flags for confidence
        authority_flags = _aggregate_authority_flags(applicability_list, conflict_flag)

        # Step 7: Compute retrieval-phase confidence
        confidence = compute_confidence_from_retrieval(
            retrieval_relevance=retrieval_relevance.score,
            evidence_sufficiency=evidence_sufficiency.score,
            authority_flags=authority_flags,
        )

        # Step 8: Check for short-circuit refusal
        should_refuse_early, refusal_cat = _should_short_circuit_to_refusal(
            found=len(chunks) > 0,
            empty_corpus=len(chunks) == 0,
            conflict_flag=conflict_flag,
            authority_flags=authority_flags,
            evidence_sufficiency=evidence_sufficiency,
            retrieval_relevance=retrieval_relevance,
        )

        # Record conflict in trace failures if detected
        if conflict_flag.has_conflict:
            for detail in conflict_flag.conflict_details:
                trace_failures.append({
                    "category": "conflicting_evidence",
                    "message": f"Conflict detected between {detail.get('chunk_a')} and {detail.get('chunk_b')}",
                    "retryable": False,
                    "details": detail,
                })

        if should_refuse_early:
            refusal_response = _build_refusal_response(refusal_cat, trace_id)
            trace = TraceRecord(
                trace_id=trace_id,
                query_text=query,
                sub_questions=[sq.question for sq in decomposition.sub_questions],
                chunk_results=trace_chunks,
                claims=[],
                confidence_inputs={
                    "retrieval_relevance": retrieval_relevance.score,
                    "evidence_sufficiency": evidence_sufficiency.score,
                    "claim_verifications": [],
                    "authority_flags": authority_flags,
                },
                final_confidence=confidence.final_confidence,
                outcome=Outcome(f"refused_{refusal_cat.value}"),
                refusal_category=refusal_cat,
                execution_budget=budget.to_dict(),
                provider="mock",
                model="retrieval_pipeline_v1",
                retries=budget.retries,
                failures=trace_failures,
                latency_ms=int(budget.wall_time_ms),
            )
            response = FilteredRetrievalResponse(
                chunks=[c.text for c in filtered_chunks],
                chunk_ids=[c.chunk_id for c in filtered_chunks],
                chunk_metadata=[{
                    "policy_id": c.policy_id,
                    "version": c.version,
                    "authority_level": c.authority_level,
                } for c in filtered_chunks],
                found=len(filtered_chunks) > 0,
                empty_corpus=len(chunks) == 0,
                conflict_flag=conflict_flag,
                ranked_ids=[c.chunk_id for c in filtered_chunks],
                trace_id=trace_id,
                retrieval_relevance=retrieval_relevance,
                evidence_sufficiency=evidence_sufficiency,
                authority_flags=authority_flags,
            )
            return response, trace

        # Step 9: If not refused, prepare for generation phase
        # (Generation would happen here in full pipeline)
        # For now, return the filtered response with confidence
        trace = TraceRecord(
            trace_id=trace_id,
            query_text=query,
            sub_questions=[sq.question for sq in decomposition.sub_questions],
            chunk_results=trace_chunks,
            claims=[],  # Would be populated in generation phase
            confidence_inputs={
                "retrieval_relevance": retrieval_relevance.score,
                "evidence_sufficiency": evidence_sufficiency.score,
                "claim_verifications": [],
                "authority_flags": authority_flags,
            },
            final_confidence=confidence.final_confidence,
            outcome=Outcome.ANSWERED if confidence.final_confidence >= DEFAULT_CONFIDENCE_THRESHOLD else Outcome.REFUSED_LOW_CONFIDENCE,
            refusal_category=None if confidence.final_confidence >= DEFAULT_CONFIDENCE_THRESHOLD else RefusalCategory.LOW_CONFIDENCE,
            execution_budget=budget.to_dict(),
            provider="mock",
            model="retrieval_pipeline_v1",
            retries=budget.retries,
            failures=trace_failures,
            latency_ms=int(budget.wall_time_ms),
        )

        response = FilteredRetrievalResponse(
            chunks=[c.text for c in filtered_chunks],
            chunk_ids=[c.chunk_id for c in filtered_chunks],
            chunk_metadata=[{
                "policy_id": c.policy_id,
                "version": c.version,
                "authority_level": c.authority_level,
            } for c in filtered_chunks],
            found=len(filtered_chunks) > 0,
            empty_corpus=len(chunks) == 0,
            conflict_flag=conflict_flag,
            ranked_ids=[c.chunk_id for c in filtered_chunks],
            trace_id=trace_id,
            retrieval_relevance=retrieval_relevance,
            evidence_sufficiency=evidence_sufficiency,
            authority_flags=authority_flags,
        )

        return response, trace

    except ProviderError as e:
        budget.consume_retry()
        trace_failures.append({
            "category": e.category.value,
            "message": e.safe_message,
            "retryable": e.retryable,
            "diagnostic_code": e.diagnostic_code,
        })
        # Re-raise to be handled by endpoint
        raise


def _aggregate_authority_flags(
    applicability_list: List[ChunkApplicability],
    conflict_flag: ConflictFlag,
) -> Dict[str, Any]:
    """Aggregate authority flags from chunk applicability list."""
    if not applicability_list:
        return {
            "has_conflict": conflict_flag.has_conflict,
            "conflict_details": conflict_flag.conflict_details,
            "authority_level": "none",
            "jurisdiction_match": False,
            "department_match": False,
            "within_effective_window": False,
            "is_superseded": False,
            "is_authoritative": False,
        }

    # Use the highest authority level among applicable chunks
    authoritative_chunks = [a for a in applicability_list if a.is_authoritative]
    if authoritative_chunks:
        # Get max authority level
        authority_order = {"global": 3, "regional": 2, "departmental": 1}
        max_auth = max(authoritative_chunks, key=lambda a: authority_order.get(a.authority_level, 0))
        return {
            "has_conflict": conflict_flag.has_conflict,
            "conflict_details": conflict_flag.conflict_details,
            "authority_level": max_auth.authority_level,
            "jurisdiction_match": max_auth.jurisdiction_match,
            "department_match": max_auth.department_match,
            "within_effective_window": max_auth.within_effective_window,
            "is_superseded": max_auth.is_superseded,
            "is_authoritative": True,
        }
    else:
        # No authoritative chunks - use first chunk's flags
        first = applicability_list[0]
        return {
            "has_conflict": conflict_flag.has_conflict,
            "conflict_details": conflict_flag.conflict_details,
            "authority_level": first.authority_level,
            "jurisdiction_match": first.jurisdiction_match,
            "department_match": first.department_match,
            "within_effective_window": first.within_effective_window,
            "is_superseded": first.is_superseded,
            "is_authoritative": False,
        }