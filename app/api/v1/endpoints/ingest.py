"""POST /ingest endpoint for Precedent V1 (FR-1.x, Section 6.1)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import date
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlmodel import Session, select, delete
from sqlalchemy import func

from app.models import Document, PolicyChunk
from app.schemas import IngestRequest, IngestResponse
from app.core.database import get_session
from app.core.chunking import chunk_text
from app.core.config import settings

router = APIRouter(prefix="/ingest", tags=["ingest"])


# ============================================================
# Helper Functions
# ============================================================

def compute_content_hash(text: str) -> str:
    """Compute SHA256 hash of document text for idempotency."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_existing_document(session: Session, content_hash: str) -> Optional[Document]:
    """Find existing document by content hash."""
    stmt = select(Document).where(Document.content_hash == content_hash)
    return session.exec(stmt).first()


def find_superseded_policies(
    session: Session,
    policy_id: str,
    new_version: str,
) -> List[str]:
    """Find policy IDs that should be superseded by this new version."""
    # Find active documents with same policy_id but different version
    stmt = select(Document).where(
        Document.document_id != policy_id,  # Not the same document
        # We need to check via chunks for policy_id
    )
    # Actually, we need to check chunks for policy_id
    from sqlalchemy import exists
    subq = select(PolicyChunk.document_id).where(
        PolicyChunk.policy_id == policy_id,
        PolicyChunk.version != new_version,
        PolicyChunk.status == "active",
    ).distinct()
    superseded_doc_ids = session.exec(subq).all()
    return superseded_doc_ids


def mark_chunks_superseded(
    session: Session,
    document_ids: List[str],
    superseded_by_policy_id: str,
) -> None:
    """Mark chunks of superseded documents as superseded."""
    for doc_id in document_ids:
        stmt = select(PolicyChunk).where(
            PolicyChunk.document_id == doc_id,
            PolicyChunk.status == "active",
        )
        chunks = session.exec(stmt).all()
        for chunk in chunks:
            chunk.status = "superseded"
            chunk.superseded_by = superseded_by_policy_id
            session.add(chunk)


def create_chunks_from_request(
    request: IngestRequest,
    document_id: str,
    chunks_text: List[str],
) -> List[PolicyChunk]:
    """Create PolicyChunk objects from request and chunked text."""
    chunks = []
    for index, chunk_text_content in enumerate(chunks_text):
        chunk_id = f"{document_id}_{index}"
        chunk = PolicyChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            text=chunk_text_content,
            policy_id=request.policy_id,
            version=request.version,
            status=request.status.value,
            authority_level=request.authority_level.value,
            jurisdiction=request.jurisdiction,
            department=request.department,
            audience=request.audience,
            effective_from=request.effective_from,
            effective_until=request.effective_until,
            owner=request.owner,
            supersedes=request.supersedes,
            superseded_by=None,
        )
        chunks.append(chunk)
    return chunks


# ============================================================
# Endpoint
# ============================================================

@router.post("", response_model=IngestResponse, status_code=status.HTTP_200_OK)
async def ingest_document(
    request: IngestRequest,
    session: Session = Depends(get_session),
    authorization: str = Header(...),
) -> IngestResponse:
    """
    Ingest a policy document.

    - Splits text into chunks with overlap
    - Stores document metadata and chunks with authority metadata
    - Handles idempotency via content hash
    - Handles supersession of previous policy versions
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

    # Compute content hash for idempotency
    content_hash = compute_content_hash(request.text)

    # Check for existing document with same content
    existing_doc = find_existing_document(session, content_hash)
    if existing_doc:
        # Return existing document info (idempotent)
        chunk_count_stmt = select(func.count(PolicyChunk.chunk_id)).where(
            PolicyChunk.document_id == existing_doc.document_id
        )
        chunk_count = session.exec(chunk_count_stmt).one()
        return IngestResponse(
            document_id=existing_doc.document_id,
            chunk_count=chunk_count,
            superseded=[],
            content_hash=content_hash,
        )

    # Check for supersession
    superseded_policy_ids = []
    if request.supersedes:
        # Find documents with the superseded policy_id
        superseded_doc_ids = find_superseded_policies(
            session, request.supersedes, request.version
        )
        if superseded_doc_ids:
            mark_chunks_superseded(session, superseded_doc_ids, request.policy_id)
            superseded_policy_ids = superseded_doc_ids

    # Create new document
    document_id = request.policy_id  # Use policy_id as document_id for simplicity
    # But ensure uniqueness - if collision, add UUID suffix
    existing = session.get(Document, document_id)
    if existing:
        document_id = f"{request.policy_id}_{uuid.uuid4().hex[:8]}"

    document = Document(
        document_id=document_id,
        document_name=request.policy_id,
        content_hash=content_hash,
    )
    session.add(document)
    session.flush()

    # Chunk the text
    chunks_text = chunk_text(
        request.text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    # Create chunk records
    chunk_objects = create_chunks_from_request(request, document_id, chunks_text)
    for chunk in chunk_objects:
        session.add(chunk)

    session.commit()

    return IngestResponse(
        document_id=document_id,
        chunk_count=len(chunk_objects),
        superseded=superseded_policy_ids,
        content_hash=content_hash,
    )