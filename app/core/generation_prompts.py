"""Generation Prompts for Precedent V1.

Contains structured prompts for the generation phase with exclusivity clause
to prevent hallucination and ensure answers are grounded in retrieved evidence.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.models import PolicyChunk
from app.schemas import ChunkApplicability, ConflictFlag, EvidenceSufficiency, RetrievalRelevance
from app.core.llm_provider import LLMMessage


# ============================================================
# Exclusivity Clause - Core Anti-Hallucination Mechanism
# ============================================================

EXCLUSIVITY_CLAUSE = """
=== EXCLUSIVITY CLAUSE (MANDATORY) ===

You are a policy assurance engine. Your answers MUST be grounded EXCLUSIVELY in the
provided policy chunks. You are FORBIDDEN from:

1. Using ANY knowledge not present in the provided chunks
2. Inferring, assuming, or extrapolating beyond what is explicitly stated
3. Providing general knowledge, best practices, or "common sense" answers
4. Filling gaps with plausible-sounding but unsupported information
5. Answering questions that the provided evidence does not address

If the provided chunks do not contain sufficient information to answer the question,
you MUST refuse to answer. State clearly what information is missing.

Every claim in your answer MUST be directly supported by a quoted span from the chunks.
Format each claim as: [CLAIM: "exact claim text"] [EVIDENCE: "exact quote from chunk"] [CHUNK_ID: chunk_id]

VIOLATION OF THIS CLAUSE IS A CRITICAL FAILURE.
"""

# ============================================================
# System Prompt with Exclusivity Clause
# ============================================================

SYSTEM_PROMPT = f"""You are Precedent, an Internal Policy Assurance Engine. Your role is to answer
policy questions using ONLY the provided policy document chunks.

{EXCLUSIVITY_CLAUSE}

=== OUTPUT FORMAT ===
Your response must be a JSON object with the following structure:
{{
    "answer": "Your natural language answer, with each claim grounded in evidence",
    "claims": [
        {{
            "claim_text": "Exact claim statement",
            "is_valid": true,
            "confidence": 0.95,
            "quoted_span": "Exact quote from chunk supporting this claim",
            "chunk_id": "chunk_id_from_metadata",
            "verification_notes": "Optional notes on verification"
        }}
    ],
    "refusal": null,
    "confidence": 0.9
}}

If you cannot answer due to insufficient evidence, set "refusal" to:
{{
    "refusal_category": "no_evidence|low_confidence|conflicting_evidence|inapplicable_evidence|stale_evidence|insufficient_authority",
    "reason": "Explanation of why you cannot answer",
    "missing_information": ["List of specific missing information"]
}}

=== VERIFICATION REQUIREMENTS ===
- Each claim MUST have a quoted_span that EXACTLY matches text in the provided chunks
- Each claim MUST reference a valid chunk_id from the provided metadata
- Confidence scores MUST be between 0.0 and 1.0
- is_valid MUST be true only if the quoted_span directly supports the claim_text
- If any claim cannot be verified, set is_valid to false and explain in verification_notes
"""

# ============================================================
# Decomposition Prompt
# ============================================================

DECOMPOSITION_PROMPT = """You are a question decomposition specialist. Break down the user's question
into 1-5 focused sub-questions that can be answered independently from policy documents.

Return a JSON object:
{
    "sub_questions": [
        {"question": "Specific sub-question 1", "focus": "eligibility|maximum|process|timing|general"},
        {"question": "Specific sub-question 2", "focus": "..."}
    ]
}

Guidelines:
- Each sub-question should address ONE specific aspect
- Focus areas: eligibility, maximum, minimum, process, timing, approval, reimbursement, general
- Do not create sub-questions that require external knowledge
- Maximum 5 sub-questions
"""

# ============================================================
# Claim Extraction Prompt
# ============================================================

CLAIM_EXTRACTION_PROMPT = f"""You are a claim extraction specialist. Extract verifiable claims from
the provided answer text and match them to supporting evidence in the policy chunks.

{EXCLUSIVITY_CLAUSE}

=== INPUT ===
- Answer text to verify
- Policy chunks with metadata (chunk_id, text, authority info)

=== OUTPUT ===
Return a JSON object:
{{
    "claims": [
        {{
            "claim_text": "Exact claim from the answer",
            "is_valid": true,
            "confidence": 0.95,
            "quoted_span": "Exact quote from chunk that supports this claim",
            "chunk_id": "chunk_id",
            "verification_notes": "How the span supports the claim"
        }}
    ],
    "unverified_claims": [
        "Claims from the answer that could not be matched to any chunk"
    ]
}}

Rules:
- A claim is ONLY valid if there is an EXACT or near-exact quote in the chunks
- Paraphrasing is NOT sufficient - the quoted_span must be from the chunk text
- If a claim mentions specific numbers, dates, or names, the chunk MUST contain them
- List any claims from the answer that lack supporting evidence in unverified_claims
"""

# ============================================================
# Answer Generation Prompt Builder
# ============================================================

@dataclass
class GenerationContext:
    """Context for building generation prompts."""
    query: str
    sub_questions: List[str]
    chunks: List[str]
    chunk_ids: List[str]
    chunk_metadata: List[Dict[str, Any]]
    chunk_applicability: List[ChunkApplicability]
    conflict_flag: ConflictFlag
    evidence_sufficiency: EvidenceSufficiency
    retrieval_relevance: RetrievalRelevance
    query_context: Optional[Dict[str, Any]] = None


def build_generation_prompt(context: GenerationContext) -> List[LLMMessage]:
    """Build the complete prompt for answer generation with exclusivity clause."""

    # Format chunks with metadata
    chunk_blocks = []
    for i, (chunk_text, chunk_id, metadata, applicability) in enumerate(zip(
        context.chunks, context.chunk_ids, context.chunk_metadata, context.chunk_applicability
    )):
        auth_info = (
            f"Authority: {metadata.get('authority_level', 'unknown')}, "
            f"Jurisdiction: {metadata.get('jurisdiction', 'unknown')}, "
            f"Department: {metadata.get('department', 'unknown')}, "
            f"Status: {applicability.status if hasattr(applicability, 'status') else 'unknown'}, "
            f"Authoritative: {applicability.is_authoritative}"
        )
        chunk_blocks.append(
            f"=== CHUNK {i+1} (ID: {chunk_id}) ===\n"
            f"{auth_info}\n"
            f"TEXT: {chunk_text}\n"
        )

    chunks_text = "\n".join(chunk_blocks)

    # Format sub-questions
    sub_q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(context.sub_questions))

    # Format query context
    context_text = ""
    if context.query_context:
        context_text = "\nQUERY CONTEXT:\n"
        for k, v in context.query_context.items():
            context_text += f"  {k}: {v}\n"

    # Conflict warning
    conflict_text = ""
    if context.conflict_flag.has_conflict:
        conflict_text = "\n⚠️ CONFLICT DETECTED: Multiple authoritative policies conflict on this topic. "
        conflict_text += "You MUST refuse with conflicting_evidence category.\n"
        for detail in context.conflict_flag.conflict_details:
            conflict_text += f"  - Policy {detail.get('policy_id')}: v{detail.get('version_a')} vs v{detail.get('version_b')}\n"

    # Evidence sufficiency warning
    evidence_text = ""
    if context.evidence_sufficiency.score < 1.0:
        missing = context.evidence_sufficiency.missing_sub_questions
        if missing:
            evidence_text = f"\n⚠️ EVIDENCE GAP: The following sub-questions lack supporting evidence:\n"
            for m in missing:
                evidence_text += f"  - {m}\n"
            evidence_text += "You MUST address these gaps in your response or refuse.\n"

    user_prompt = f"""ORIGINAL QUESTION: {context.query}

SUB-QUESTIONS TO ADDRESS:
{sub_q_text}
{context_text}
{conflict_text}
{evidence_text}

RETRIEVED POLICY CHUNKS:
{chunks_text}

Generate a JSON response following the output format in the system prompt.
Remember: You MUST ground every claim in the provided chunks. If evidence is insufficient, refuse."""

    return [
        LLMMessage(role="system", content=SYSTEM_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]


def build_decomposition_prompt(query: str) -> List[LLMMessage]:
    """Build prompt for question decomposition."""
    return [
        LLMMessage(role="system", content=DECOMPOSITION_PROMPT),
        LLMMessage(role="user", content=f"Decompose this question: {query}"),
    ]


def build_claim_extraction_prompt(
    answer: str,
    chunks: List[str],
    chunk_ids: List[str],
    chunk_metadata: List[Dict[str, Any]],
) -> List[LLMMessage]:
    """Build prompt for claim extraction and verification."""

    chunk_blocks = []
    for chunk_text, chunk_id, metadata in zip(chunks, chunk_ids, chunk_metadata):
        chunk_blocks.append(f"CHUNK {chunk_id}:\n{chunk_text}\n")

    chunks_text = "\n".join(chunk_blocks)

    user_prompt = f"""ANSWER TO VERIFY:
{answer}

POLICY CHUNKS FOR VERIFICATION:
{chunks_text}

Extract and verify all claims from the answer against the chunks."""

    return [
        LLMMessage(role="system", content=CLAIM_EXTRACTION_PROMPT),
        LLMMessage(role="user", content=user_prompt),
    ]


# ============================================================
# Prompt Validation
# ============================================================

def validate_generation_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize a generation response."""
    required_fields = ["answer", "claims", "confidence"]

    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing required field: {field}")

    # Validate claims structure
    if not isinstance(response["claims"], list):
        raise ValueError("claims must be a list")

    for i, claim in enumerate(response["claims"]):
        required_claim_fields = ["claim_text", "is_valid", "confidence", "quoted_span", "chunk_id"]
        for field in required_claim_fields:
            if field not in claim:
                raise ValueError(f"Claim {i} missing required field: {field}")

        # Validate confidence range
        conf = claim["confidence"]
        if not isinstance(conf, (int, float)) or conf < 0.0 or conf > 1.0:
            raise ValueError(f"Claim {i} confidence must be between 0.0 and 1.0")

        # Validate is_valid is boolean
        if not isinstance(claim["is_valid"], bool):
            raise ValueError(f"Claim {i} is_valid must be boolean")

    # Validate overall confidence
    overall_conf = response["confidence"]
    if not isinstance(overall_conf, (int, float)) or overall_conf < 0.0 or overall_conf > 1.0:
        raise ValueError("Overall confidence must be between 0.0 and 1.0")

    # Handle refusal
    if response.get("refusal"):
        refusal = response["refusal"]
        required_refusal_fields = ["refusal_category", "reason"]
        for field in required_refusal_fields:
            if field not in refusal:
                raise ValueError(f"Refusal missing required field: {field}")

        valid_categories = {
            "no_evidence", "low_confidence", "conflicting_evidence",
            "inapplicable_evidence", "stale_evidence", "insufficient_authority"
        }
        if refusal["refusal_category"] not in valid_categories:
            raise ValueError(f"Invalid refusal category: {refusal['refusal_category']}")

    return response


def parse_generation_response(raw_response: str) -> Dict[str, Any]:
    """Parse and validate a generation response from LLM output."""
    try:
        # Try to extract JSON from the response
        # Handle cases where LLM wraps JSON in markdown code blocks
        content = raw_response.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        response = json.loads(content)
        return validate_generation_response(response)

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except ValueError:
        raise


import json