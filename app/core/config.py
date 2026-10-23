"""Configuration settings for Precedent V1."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API
    api_token: str = "dev-token-123"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "sqlite:///./precedent.db"
    database_echo: bool = False

    # LLM Provider (swappable per SRS)
    llm_provider: str = "mock"  # mock, nvidia, openai, anthropic, groq
    llm_model: str = "mock-model"
    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096
    llm_timeout: float = 30.0

    # NVIDIA NIM specific settings
    nvidia_api_key: Optional[str] = None
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "meta/llama-3.1-405b-instruct"

    # OpenAI specific settings
    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Anthropic specific settings
    anthropic_api_key: Optional[str] = None
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_model: str = "claude-3-haiku-20240307"

    # Embedding Provider
    embedding_provider: str = "mock"  # mock, sentence-transformers, openai
    embedding_model: str = "mock-embedding"
    embedding_api_key: Optional[str] = None

    # Retrieval
    default_top_k: int = 5
    max_top_k: int = 20
    similarity_threshold: float = 0.3

    # Confidence & Refusal (FR-6, FR-7)
    confidence_threshold: float = 0.5
    min_evidence_sufficiency: float = 0.3
    min_retrieval_relevance: float = 0.3

    # Execution Budget (FR-9, FR-10)
    max_llm_calls: int = 10
    max_tokens: int = 50000
    max_retries: int = 3
    max_wall_time_ms: int = 30000
    max_estimated_cost: float = 10.0

    # Chunking (FR-1)
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Logging
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()