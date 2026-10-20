"""Error types and normalization for Precedent V1 (FR-7, FR-9)."""

from __future__ import annotations

from typing import Optional, Dict, Type, Any
from enum import Enum


class ErrorCategory(str, Enum):
    """Error categories for classification and retry logic."""
    PROVIDER_AUTH = "provider-auth"
    PROVIDER_RATE_LIMIT = "provider-rate-limit"
    PROVIDER_TIMEOUT = "provider-timeout"
    PROVIDER_MODEL = "provider-model"
    EMBEDDING_FAILURE = "embedding-failure"
    INFRASTRUCTURE = "infrastructure"
    VALIDATION = "validation"
    BUDGET = "budget"
    INTERNAL = "internal"


class PrecedentError(Exception):
    """Base exception for all Precedent errors."""
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        retryable: bool = False,
        safe_message: Optional[str] = None,
        diagnostic_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.category = category
        self.retryable = retryable
        self.safe_message = safe_message or self._default_safe_message()
        self.diagnostic_code = diagnostic_code or self._generate_diagnostic_code()
        self.details = details or {}
        super().__init__(message)

    def _default_safe_message(self) -> str:
        return "An error occurred while processing your request."

    def _generate_diagnostic_code(self) -> str:
        import uuid
        return f"ERR-{self.category.value.upper()}-{uuid.uuid4().hex[:8]}"


class ProviderError(PrecedentError):
    """Base class for provider-related errors."""
    pass


class ProviderAuthenticationError(ProviderError):
    """Authentication failed with the AI provider."""
    def __init__(self, message: str = "Authentication failed with the AI provider", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROVIDER_AUTH,
            retryable=False,
            safe_message="Authentication failed with the AI provider",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class ProviderRateLimitError(ProviderError):
    """The AI provider is rate-limiting requests."""
    def __init__(self, message: str = "The AI provider is rate-limiting requests", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROVIDER_RATE_LIMIT,
            retryable=True,
            safe_message="The AI provider is rate-limiting requests. Please try again later.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class ProviderTimeoutError(ProviderError):
    """The AI provider request timed out."""
    def __init__(self, message: str = "The AI provider request timed out", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROVIDER_TIMEOUT,
            retryable=True,
            safe_message="The AI provider request timed out. Please try again.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class ProviderModelError(ProviderError):
    """The AI provider's model returned an error."""
    def __init__(self, message: str = "The AI provider's model returned an error", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.PROVIDER_MODEL,
            retryable=False,
            safe_message="The AI provider's model returned an error.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class EmbeddingError(PrecedentError):
    """An error occurred generating embeddings."""
    def __init__(self, message: str = "An error occurred generating embeddings", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.EMBEDDING_FAILURE,
            retryable=True,
            safe_message="An error occurred generating embeddings. Please try again.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class InfrastructureError(PrecedentError):
    """A system infrastructure issue was encountered."""
    def __init__(self, message: str = "A system infrastructure issue was encountered", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INFRASTRUCTURE,
            retryable=False,
            safe_message="A system infrastructure issue was encountered.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class ModelOutputValidationError(PrecedentError):
    """Model output failed validation checks."""
    def __init__(self, message: str = "Model output failed validation", **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            retryable=False,
            safe_message="The generated response failed validation checks.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


class BudgetViolation(PrecedentError):
    """Execution budget limit exceeded."""
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.BUDGET,
            retryable=False,
            safe_message="The query exceeded execution budget limits.",
            diagnostic_code=kwargs.pop("diagnostic_code", None),
            **kwargs
        )


# Mapping from provider exception types to Precedent error classes
PROVIDER_EXCEPTION_MAP: Dict[str, Type[ProviderError]] = {
    "auth": ProviderAuthenticationError,
    "authentication": ProviderAuthenticationError,
    "unauthorized": ProviderAuthenticationError,
    "permission": ProviderAuthenticationError,
    "rate_limit": ProviderRateLimitError,
    "ratelimit": ProviderRateLimitError,
    "too_many_requests": ProviderRateLimitError,
    "timeout": ProviderTimeoutError,
    "timed_out": ProviderTimeoutError,
    "model": ProviderModelError,
    "invalid_request": ProviderModelError,
    "bad_request": ProviderModelError,
}


def map_provider_exception(exc: Exception) -> ProviderError:
    """
    Map a generic provider exception to a specific Precedent provider error.
    Uses keyword matching on exception type/message.
    """
    exc_type = type(exc).__name__.lower()
    exc_msg = str(exc).lower()

    for keyword, error_class in PROVIDER_EXCEPTION_MAP.items():
        if keyword in exc_type or keyword in exc_msg:
            return error_class()

    # Default fallback
    return ProviderModelError()


def wrap_provider_error(
    original: Exception,
    error_class: Optional[Type[ProviderError]] = None,
) -> ProviderError:
    """
    Wrap a provider exception in a Precedent error.
    If error_class is provided, use it; otherwise infer from exception.
    """
    if error_class is not None:
        wrapped = error_class()
        # Preserve diagnostic code from original if it's already a PrecedentError
        if isinstance(original, PrecedentError):
            wrapped.diagnostic_code = original.diagnostic_code
        return wrapped

    return map_provider_exception(original)


# Convenience function for creating error responses
def create_error_response(
    error: PrecedentError,
    trace_id: str,
    execution_budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a standardized error response dict."""
    return {
        "error": {
            "category": error.category.value,
            "retryable": error.retryable,
            "safe_message": error.safe_message,
            "diagnostic_code": error.diagnostic_code,
        },
        "trace_id": trace_id,
        "execution_budget": execution_budget or {},
    }