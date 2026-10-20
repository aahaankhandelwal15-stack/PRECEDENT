"""Database engine and session management for Precedent V1."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, Optional
from sqlmodel import SQLModel, Session, create_engine
from sqlalchemy.engine import Engine

from app.core.config import settings


_engine: Optional[Engine] = None


def get_engine() -> Engine:
    """Get or create the database engine (singleton)."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.database_echo,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
        )
    return _engine


def create_engine_and_tables(database_url: Optional[str] = None) -> Engine:
    """Create engine and all tables. Used for testing and initialization."""
    global _engine
    url = database_url or settings.database_url
    _engine = create_engine(
        url,
        echo=settings.database_echo,
        connect_args={"check_same_thread": False} if "sqlite" in url else {},
    )
    SQLModel.metadata.create_all(_engine)
    return _engine


def drop_all_tables(engine: Optional[Engine] = None) -> None:
    """Drop all tables. Used for testing."""
    eng = engine or get_engine()
    SQLModel.metadata.drop_all(eng)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    engine = get_engine()
    with Session(engine) as session:
        yield session


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for database sessions outside of FastAPI."""
    engine = get_engine()
    session = Session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(database_url: Optional[str] = None) -> None:
    """Initialize database - create tables if they don't exist."""
    engine = create_engine_and_tables(database_url)
    # Tables are created in create_engine_and_tables