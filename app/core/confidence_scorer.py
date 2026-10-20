"""Confidence scoring for Precedent V1 (FR-6.x).

Key Design Decision (per diagnosis_table.md):
- The retrieval phase should NOT penalize for missing claim_verifications
  because claim extraction (FR-5) happens AFTER retrieval in the pipeline.
- Claim verification penalties only apply in the generation phase.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.schemas import ClaimVerificationResult, AuthorityApplicabilityFlags
from app.core.config import settings


@dataclass
class ConfidenceComponents:
    """Individual components of the confidence score."""
    retrieval_relevance: float  # 0.0 - 1.0
    evidence_sufficiency: float  # 0.0 - 1.0
    claim_verification_score: float  # 0.0 - 1.0 (based on claim validations)
    authority_score: float  # 0.0 - 1.0 (based on authority/applicability flags)
    final_confidence: float  # Combined score


def compute_authority_score(authority_flags: Dict[str, Any]) -> float:
    """
    Compute authority/applicability score from flags.
    Returns 1.0 if all checks pass, lower if issues detected.
    """
    if not authority_flags:
        return 0.5  # Neutral if no flags provided

    score = 1.0
    penalties = []

    # Check for conflicts
    if authority_flags.get("has_conflict", False):
        score *= 0.3
        penalties.append("conflict_detected")

    # Check authority level
    auth_level = authority_flags.get("authority_level", "departmental")
    if auth_level == "global":
        pass  # No penalty
    elif auth_level == "regional":
        score *= 0.9
    elif auth_level == "departmental":
        score *= 0.7
    else:
        score *= 0.5
        penalties.append(f"unknown_authority_level:{auth_level}")

    # Jurisdiction match
    if not authority_flags.get("jurisdiction_match", True):
        score *= 0.5
        penalties.append("jurisdiction_mismatch")

    # Department match
    if not authority_flags.get("department_match", True):
        score *= 0.7
        penalties.append("department_mismatch")

    # Effective window
    if not authority_flags.get("within_effective_window", True):
        score *= 0.3
        penalties.append("outside_effective_window")

    # Superseded
    if authority_flags.get("is_superseded", False):
        score *= 0.2
        penalties.append("superseded")

    return max(0.0, min(1.0, score)), penalties


def compute_claim_verification_score(
    claim_verifications: List[ClaimVerificationResult],
) -> tuple[float, List[str]]:
    """
    Compute score from claim verification results.
    Returns (score, penalty_reasons).
    """
    if not claim_verifications:
        # No claim verifications yet - this is EXPECTED during retrieval phase
        # Return neutral score (1.0) with no penalty
        # The penalty for missing claims is applied in GENERATION phase only
        return 1.0, []

    valid_count = sum(1 for cv in claim_verifications if cv.is_valid)
    total_count = len(claim_verifications)

    if total_count == 0:
        return 1.0, []

    validity_ratio = valid_count / total_count

    # Weight by individual confidences
    weighted_sum = sum(cv.confidence for cv in claim_verifications if cv.is_valid)
    max_possible = sum(cv.confidence for cv in claim_verifications) if claim_verifications else 1.0

    if max_possible > 0:
        confidence_weighted = weighted_sum / max_possible
    else:
        confidence_weighted = 0.0

    # Combined score: validity ratio * confidence weighted
    score = validity_ratio * confidence_weighted

    penalties = []
    if validity_ratio < 1.0:
        penalties.append(f"invalid_claims:{total_count - valid_count}/{total_count}")

    return max(0.0, min(1.0, score)), penalties


def compute_confidence_from_inputs(
    retrieval_relevance: float,
    evidence_sufficiency: float,
    claim_verifications: List[ClaimVerificationResult],
    authority_flags: Dict[str, Any],
    phase: str = "retrieval",  # "retrieval" or "generation"
) -> ConfidenceComponents:
    """
    Compute overall confidence from pipeline inputs.

    Args:
        retrieval_relevance: Score from retrieval relevance (FR-2.x)
        evidence_sufficiency: Score from evidence sufficiency (FR-4.x)
        claim_verifications: List of claim verification results (FR-5.x)
        authority_flags: Authority/applicability flags (FR-3.x)
        phase: Pipeline phase - "retrieval" or "generation"
               CRITICAL: In retrieval phase, claim_verifications is expected to be empty.
               Do NOT penalize for empty claim_verifications in retrieval phase.

    Returns:
        ConfidenceComponents with final_confidence and component scores
    """
    # Validate inputs
    retrieval_relevance = max(0.0, min(1.0, retrieval_relevance))
    evidence_sufficiency = max(0.0, min(1.0, evidence_sufficiency))

    # Authority score
    authority_score, auth_penalties = compute_authority_score(authority_flags)

    # Claim verification score
    claim_score, claim_penalties = compute_claim_verification_score(claim_verifications)

    # Phase-specific logic
    penalties = list(auth_penalties) + list(claim_penalties)
    penalty_applied = len(penalties) > 0
    penalty_reason = "; ".join(penalties) if penalties else None

    # In retrieval phase, claim_verifications is expected to be empty
    # We use a neutral claim_score (1.0) so it doesn't affect retrieval confidence
    if phase == "retrieval":
        # Retrieval confidence based on retrieval + evidence + authority only
        # Claim verification hasn't happened yet
        final_confidence = (
            0.35 * retrieval_relevance +
            0.35 * evidence_sufficiency +
            0.30 * authority_score
        )
        # Normalize since claim_verification not included
        final_confidence = final_confidence / 1.0  # Already sums to 1.0
    else:
        # Generation phase: all four components
        final_confidence = (
            0.25 * retrieval_relevance +
            0.25 * evidence_sufficiency +
            0.25 * claim_score +
            0.25 * authority_score
        )

    final_confidence = max(0.0, min(1.0, final_confidence))

    return ConfidenceComponents(
        retrieval_relevance=retrieval_relevance,
        evidence_sufficiency=evidence_sufficiency,
        claim_verification_score=claim_score if phase == "generation" else 1.0,
        authority_score=authority_score,
        final_confidence=final_confidence,
    )


def compute_confidence_from_retrieval(
    retrieval_relevance: float,
    evidence_sufficiency: float,
    authority_flags: Dict[str, Any],
) -> ConfidenceComponents:
    """
    Convenience function for retrieval-phase confidence (no claim verifications yet).
    """
    return compute_confidence_from_inputs(
        retrieval_relevance=retrieval_relevance,
        evidence_sufficiency=evidence_sufficiency,
        claim_verifications=[],  # Empty in retrieval phase
        authority_flags=authority_flags,
        phase="retrieval",
    )


def compute_confidence_from_generation(
    retrieval_relevance: float,
    evidence_sufficiency: float,
    claim_verifications: List[ClaimVerificationResult],
    authority_flags: Dict[str, Any],
) -> ConfidenceComponents:
    """
    Convenience function for generation-phase confidence (with claim verifications).
    """
    return compute_confidence_from_inputs(
        retrieval_relevance=retrieval_relevance,
        evidence_sufficiency=evidence_sufficiency,
        claim_verifications=claim_verifications,
        authority_flags=authority_flags,
        phase="generation",
    )


def should_refuse(
    final_confidence: float,
    authority_flags: Dict[str, Any],
    evidence_sufficiency: float,
    retrieval_relevance: float,
    threshold: Optional[float] = None,
) -> tuple[bool, Optional[str]]:
    """
    Determine if query should be refused based on confidence and flags.
    Returns (should_refuse, refusal_category).
    """
    threshold = threshold or settings.confidence_threshold

    # Hard refusal conditions (override confidence)
    if authority_flags.get("has_conflict", False):
        return True, "conflicting_evidence"

    if not authority_flags.get("within_effective_window", True):
        return True, "stale_evidence"

    if authority_flags.get("is_superseded", False):
        return True, "stale_evidence"

    if not authority_flags.get("jurisdiction_match", True):
        return True, "inapplicable_evidence"

    if not authority_flags.get("department_match", True):
        return True, "inapplicable_evidence"

    if evidence_sufficiency < settings.min_evidence_sufficiency:
        return True, "inapplicable_evidence"

    if retrieval_relevance < settings.min_retrieval_relevance:
        return True, "no_evidence"

    # Confidence-based refusal
    if final_confidence < threshold:
        return True, "low_confidence"

    return False, None