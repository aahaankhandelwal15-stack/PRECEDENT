"""SQLModel database models for Precedent V1."""

from __future__ import annotations

from datetime import datetime, date
from typing import Optional
from sqlmodel import SQLModel, Field, Column, Index
from sqlalchemy import Text, DateTime, Date


class Document(SQLModel, table=True):
    """Document metadata for ingested policy documents."""

    __tablename__ = "documents"

    document_id: str = Field(primary_key=True, index=True)
    document_name: str = Field(index=True)
    content_hash: str = Field(unique=True, index=True)
    ingested_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        sa_column=Column(DateTime(timezone=True), nullable=False, index=True),
    )


class PolicyChunk(SQLModel, table=True):
    """Individual chunk of an ingested policy document with authority metadata."""

    __tablename__ = "policy_chunks"

    chunk_id: str = Field(primary_key=True, index=True)
    document_id: str = Field(foreign_key="documents.document_id", index=True)
    text: str = Field(sa_column=Column(Text, nullable=False))

    # Policy identity
    policy_id: str = Field(index=True)
    version: str = Field(index=True)
    status: str = Field(index=True)  # active, superseded, draft

    # Authority metadata (FR-3.x)
    authority_level: str = Field(index=True)  # global, regional, departmental
    jurisdiction: str = Field(index=True)
    department: str = Field(index=True)
    audience: str = Field(default="internal")

    # Temporal validity (FR-3.2, FR-3.3)
    effective_from: Optional[date] = Field(default=None, sa_column=Column(Date, nullable=True, index=True))
    effective_until: Optional[date] = Field(default=None, sa_column=Column(Date, nullable=True, index=True))

    # Supersession chain (FR-3.1)
    supersedes: Optional[str] = Field(default=None, index=True)  # policy_id this chunk's document supersedes
    superseded_by: Optional[str] = Field(default=None, index=True)  # policy_id that supersedes this chunk's document

    # Owner
    owner: str = Field(default="system")

    __table_args__ = (
        Index("ix_policy_chunks_policy_version", "policy_id", "version"),
        Index("ix_policy_chunks_status_effective", "status", "effective_from", "effective_until"),
        Index("ix_policy_chunks_authority", "authority_level", "jurisdiction", "department"),
    )