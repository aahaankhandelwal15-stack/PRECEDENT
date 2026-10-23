"""LLM Provider Abstraction for Precedent V1.

Provides a unified interface for multiple LLM providers (mock, NVIDIA, OpenAI, Anthropic, etc.)
with consistent error handling, budget tracking, and retry logic.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncGenerator
from enum import Enum

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging

from app.core.config import settings
from app.core.budget import ExecutionBudget
from app.core.errors import (
    ProviderError,
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderModelError,
    ErrorCategory,
    wrap_provider_error,
)

logger = logging.getLogger(__name__)


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    MOCK = "mock"
    NVIDIA = "nvidia"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


@dataclass
class LLMMessage:
    """A message in the conversation."""
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class LLMProviderConfig:
    """Configuration for an LLM provider."""
    provider_type: LLMProviderType
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1
    timeout: float = 30.0
    extra_params: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMProviderConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost based on token usage. Override in subclasses for provider-specific pricing."""
        # Default: $0.001 per 1K tokens (very rough estimate)
        return (tokens_input + tokens_output) * 0.000001

    def _handle_provider_error(self, error: Exception, provider_name: str) -> ProviderError:
        """Convert provider-specific errors to Precedent errors."""
        error_msg = str(error).lower()
        error_type = type(error).__name__.lower()

        # Authentication errors
        if any(kw in error_msg for kw in ["unauthorized", "authentication", "invalid api key", "api key"]):
            return ProviderAuthenticationError(
                message=f"{provider_name}: {error}",
                diagnostic_code=f"ERR-{provider_name.upper()}-AUTH"
            )

        # Rate limit errors
        if any(kw in error_msg for kw in ["rate limit", "too many requests", "429"]):
            return ProviderRateLimitError(
                message=f"{provider_name}: {error}",
                diagnostic_code=f"ERR-{provider_name.upper()}-RATE"
            )

        # Timeout errors
        if any(kw in error_msg for kw in ["timeout", "timed out", "read timeout"]):
            return ProviderTimeoutError(
                message=f"{provider_name}: {error}",
                diagnostic_code=f"ERR-{provider_name.upper()}-TIMEOUT"
            )

        # Default to model error
        return ProviderModelError(
            message=f"{provider_name}: {error}",
            diagnostic_code=f"ERR-{provider_name.upper()}-MODEL"
        )


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    async def generate(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response."""
        if budget:
            budget.consume_llm_call(cost=0.001)

        # Check system prompt to determine response type
        system_prompt = ""
        last_user_msg = ""
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                last_user_msg = msg.content

        # Generate contextual mock response based on system prompt
        if "EXCLUSIVITY CLAUSE" in system_prompt:
            # Generation phase - return structured answer with claims
            content = json.dumps({
                "answer": "Based on the retrieved policy documents, employees are eligible for reimbursement of up to ₹50,000 per fiscal year for approved business expenses. Claims over ₹20,000 require pre-approval from the department head, and itemized receipts must be submitted within 30 days.",
                "claims": [
                    {
                        "claim_text": "Employees are eligible for reimbursement up to ₹50,000 per fiscal year.",
                        "is_valid": True,
                        "confidence": 0.95,
                        "quoted_span": "Employees are eligible for reimbursement of up to ₹50,000 per fiscal year for approved business expenses.",
                        "chunk_id": "policy_v1_ch1",
                        "verification_notes": "Direct quote from policy chunk"
                    },
                    {
                        "claim_text": "Claims over ₹20,000 require pre-approval from the department head.",
                        "is_valid": True,
                        "confidence": 0.95,
                        "quoted_span": "Claims over ₹20,000 require pre-approval from the department head.",
                        "chunk_id": "policy_v1_ch3",
                        "verification_notes": "Direct quote from policy chunk"
                    },
                    {
                        "claim_text": "Itemized receipts must be submitted within 30 days of incurrence.",
                        "is_valid": True,
                        "confidence": 0.95,
                        "quoted_span": "Reimbursement claims require itemized receipts submitted within 30 days of incurrence.",
                        "chunk_id": "policy_v1_ch2",
                        "verification_notes": "Direct quote from policy chunk"
                    }
                ],
                "refusal": None,
                "confidence": 0.9
            })
        elif "claim extraction" in system_prompt.lower() or "verify" in system_prompt.lower():
            # Claim extraction phase
            content = json.dumps({
                "claims": [
                    {
                        "claim_text": "Employees are eligible for reimbursement up to ₹50,000 per fiscal year.",
                        "is_valid": True,
                        "confidence": 0.95,
                        "quoted_span": "Employees are eligible for reimbursement of up to ₹50,000 per fiscal year for approved business expenses.",
                        "chunk_id": "policy_v1_ch1",
                        "verification_notes": "Direct quote from policy chunk"
                    }
                ],
                "unverified_claims": []
            })
        elif "decompose" in last_user_msg.lower() or "sub-question" in last_user_msg.lower():
            # Decomposition phase
            content = json.dumps({
                "sub_questions": [
                    {"question": "What is the eligibility criteria?", "focus": "eligibility"},
                    {"question": "What is the maximum reimbursement amount?", "focus": "maximum"},
                    {"question": "What is the approval process?", "focus": "process"},
                ]
            })
        else:
            # Default response
            content = "Mock response from Precedent LLM provider."

        return LLMResponse(
            content=content,
            model=self.config.model,
            provider=self.config.provider_type.value,
            tokens_input=100,
            tokens_output=50,
            cost=0.001,
        )

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate a mock streaming response."""
        response = await self.generate(messages, budget, **kwargs)
        # Simulate streaming by yielding chunks
        words = response.content.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class NVIDIALLMProvider(LLMProvider):
    """NVIDIA NIM LLM provider."""

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://integrate.api.nvidia.com/v1"
        self.api_key = config.api_key or os.getenv("NVIDIA_API_KEY")

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for NVIDIA API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Format messages for NVIDIA API (OpenAI-compatible format)."""
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> httpx.Response:
        """Make a request to the NVIDIA API."""
        client = await self._get_client()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
            **(self.config.extra_params or {}),
            **kwargs,
        }

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
        )

        # Handle HTTP errors
        if response.status_code == 401:
            raise ProviderAuthenticationError("Invalid NVIDIA API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("NVIDIA API rate limit exceeded")
        elif response.status_code >= 500:
            raise ProviderModelError(f"NVIDIA API server error: {response.status_code}")
        elif response.status_code != 200:
            raise ProviderModelError(f"NVIDIA API error: {response.status_code} - {response.text}")

        return response

    async def generate(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a response from NVIDIA NIM."""
        if budget:
            budget.assert_llm_calls_limit("nvidia_generate")

        try:
            formatted_messages = self._format_messages(messages)
            response = await self._make_request(formatted_messages, stream=False, **kwargs)

            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            # Extract token usage
            usage = data.get("usage", {})
            tokens_input = usage.get("prompt_tokens", 0)
            tokens_output = usage.get("completion_tokens", 0)

            # Calculate cost (NVIDIA NIM pricing varies by model)
            cost = self._estimate_cost(tokens_input, tokens_output)

            if budget:
                budget.consume_llm_call(
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost=cost,
                )

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="nvidia",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                raw_response=data,
            )

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_provider_error(e, "nvidia")

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from NVIDIA NIM."""
        if budget:
            budget.assert_llm_calls_limit("nvidia_generate_stream")

        try:
            formatted_messages = self._format_messages(messages)
            client = await self._get_client()

            payload = {
                "model": self.config.model,
                "messages": formatted_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": True,
                **(self.config.extra_params or {}),
                **kwargs,
            }

            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                if response.status_code == 401:
                    raise ProviderAuthenticationError("Invalid NVIDIA API key")
                elif response.status_code == 429:
                    raise ProviderRateLimitError("NVIDIA API rate limit exceeded")
                elif response.status_code != 200:
                    raise ProviderModelError(f"NVIDIA API error: {response.status_code}")

                tokens_input = 0
                tokens_output = 0

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                                    tokens_output += 1  # Rough estimate
                            if "usage" in data:
                                tokens_input = data["usage"].get("prompt_tokens", 0)
                                tokens_output = data["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

                if budget:
                    cost = self._estimate_cost(tokens_input, tokens_output)
                    budget.consume_llm_call(
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        cost=cost,
                    )

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_provider_error(e, "nvidia")


class OpenAILLMProvider(LLMProvider):
    """OpenAI LLM provider (for reference/future use)."""

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or os.getenv("OPENAI_API_KEY")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs,
    ) -> httpx.Response:
        client = await self._get_client()

        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
            **(self.config.extra_params or {}),
            **kwargs,
        }

        response = await client.post(
            f"{self.base_url}/chat/completions",
            headers=self._get_headers(),
            json=payload,
        )

        if response.status_code == 401:
            raise ProviderAuthenticationError("Invalid OpenAI API key")
        elif response.status_code == 429:
            raise ProviderRateLimitError("OpenAI API rate limit exceeded")
        elif response.status_code >= 500:
            raise ProviderModelError(f"OpenAI API server error: {response.status_code}")
        elif response.status_code != 200:
            raise ProviderModelError(f"OpenAI API error: {response.status_code} - {response.text}")

        return response

    async def generate(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> LLMResponse:
        if budget:
            budget.assert_llm_calls_limit("openai_generate")

        try:
            formatted_messages = self._format_messages(messages)
            response = await self._make_request(formatted_messages, stream=False, **kwargs)

            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"]

            usage = data.get("usage", {})
            tokens_input = usage.get("prompt_tokens", 0)
            tokens_output = usage.get("completion_tokens", 0)

            # OpenAI pricing (approximate, varies by model)
            cost = self._estimate_openai_cost(tokens_input, tokens_output)

            if budget:
                budget.consume_llm_call(
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost=cost,
                )

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="openai",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                raw_response=data,
            )

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_provider_error(e, "openai")

    def _estimate_openai_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate OpenAI cost based on model."""
        # Rough pricing for gpt-4o-mini
        if "mini" in self.config.model.lower():
            return (tokens_input * 0.00015 + tokens_output * 0.0006) / 1000
        # gpt-4o pricing
        return (tokens_input * 0.0025 + tokens_output * 0.01) / 1000

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        if budget:
            budget.assert_llm_calls_limit("openai_generate_stream")

        try:
            formatted_messages = self._format_messages(messages)
            client = await self._get_client()

            payload = {
                "model": self.config.model,
                "messages": formatted_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "stream": True,
                **(self.config.extra_params or {}),
                **kwargs,
            }

            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=self.config.timeout,
            ) as response:
                if response.status_code == 401:
                    raise ProviderAuthenticationError("Invalid OpenAI API key")
                elif response.status_code == 429:
                    raise ProviderRateLimitError("OpenAI API rate limit exceeded")

                tokens_input = 0
                tokens_output = 0

                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                                    tokens_output += 1
                            if "usage" in data:
                                tokens_input = data["usage"].get("prompt_tokens", 0)
                                tokens_output = data["usage"].get("completion_tokens", 0)
                        except json.JSONDecodeError:
                            continue

                if budget:
                    cost = self._estimate_openai_cost(tokens_input, tokens_output)
                    budget.consume_llm_call(
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                        cost=cost,
                    )

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_provider_error(e, "openai")


class AnthropicLLMProvider(LLMProvider):
    """Anthropic LLM provider (for reference/future use)."""

    def __init__(self, config: LLMProviderConfig):
        super().__init__(config)
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self.api_key = config.api_key or os.getenv("ANTHROPIC_API_KEY")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        # Anthropic uses a different format - system prompt separate
        result = []
        for msg in messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    async def generate(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> LLMResponse:
        if budget:
            budget.assert_llm_calls_limit("anthropic_generate")

        try:
            formatted_messages = self._format_messages(messages)
            client = await self._get_client()

            # Separate system prompt
            system_prompt = ""
            user_messages = []
            for msg in formatted_messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    user_messages.append(msg)

            payload = {
                "model": self.config.model,
                "messages": user_messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                **({"system": system_prompt} if system_prompt else {}),
                **(self.config.extra_params or {}),
                **kwargs,
            }

            response = await client.post(
                f"{self.base_url}/messages",
                headers=self._get_headers(),
                json=payload,
            )

            if response.status_code == 401:
                raise ProviderAuthenticationError("Invalid Anthropic API key")
            elif response.status_code == 429:
                raise ProviderRateLimitError("Anthropic API rate limit exceeded")
            elif response.status_code != 200:
                raise ProviderModelError(f"Anthropic API error: {response.status_code} - {response.text}")

            data = response.json()
            content = data["content"][0]["text"] if data.get("content") else ""

            usage = data.get("usage", {})
            tokens_input = usage.get("input_tokens", 0)
            tokens_output = usage.get("output_tokens", 0)

            cost = self._estimate_anthropic_cost(tokens_input, tokens_output)

            if budget:
                budget.consume_llm_call(
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    cost=cost,
                )

            return LLMResponse(
                content=content,
                model=self.config.model,
                provider="anthropic",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                raw_response=data,
            )

        except ProviderError:
            raise
        except Exception as e:
            raise self._handle_provider_error(e, "anthropic")

    def _estimate_anthropic_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate Anthropic cost based on model."""
        # Rough pricing for claude-3-haiku
        if "haiku" in self.config.model.lower():
            return (tokens_input * 0.00025 + tokens_output * 0.00125) / 1000
        # claude-3-sonnet
        return (tokens_input * 0.003 + tokens_output * 0.015) / 1000

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        budget: Optional[ExecutionBudget] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        # Simplified streaming - would need server-sent events handling
        response = await self.generate(messages, budget, **kwargs)
        yield response.content


def create_llm_provider(
    provider_type: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs,
) -> LLMProvider:
    """Factory function to create an LLM provider instance."""
    provider_type = provider_type or settings.llm_provider

    # Use provider-specific settings from config
    if provider_type == "nvidia":
        model = model or settings.nvidia_model
        api_key = api_key or settings.nvidia_api_key or settings.llm_api_key
        base_url = base_url or settings.nvidia_base_url
        temperature = kwargs.get("temperature", settings.llm_temperature)
        max_tokens = kwargs.get("max_tokens", settings.llm_max_tokens)
        timeout = kwargs.get("timeout", settings.llm_timeout)
    elif provider_type == "openai":
        model = model or settings.openai_model
        api_key = api_key or settings.openai_api_key or settings.llm_api_key
        base_url = base_url or settings.openai_base_url
        temperature = kwargs.get("temperature", settings.llm_temperature)
        max_tokens = kwargs.get("max_tokens", settings.llm_max_tokens)
        timeout = kwargs.get("timeout", settings.llm_timeout)
    elif provider_type == "anthropic":
        model = model or settings.anthropic_model
        api_key = api_key or settings.anthropic_api_key or settings.llm_api_key
        base_url = base_url or settings.anthropic_base_url
        temperature = kwargs.get("temperature", settings.llm_temperature)
        max_tokens = kwargs.get("max_tokens", settings.llm_max_tokens)
        timeout = kwargs.get("timeout", settings.llm_timeout)
    else:
        # Mock or other providers
        model = model or settings.llm_model
        api_key = api_key or settings.llm_api_key
        base_url = base_url or settings.llm_base_url
        temperature = kwargs.get("temperature", settings.llm_temperature)
        max_tokens = kwargs.get("max_tokens", settings.llm_max_tokens)
        timeout = kwargs.get("timeout", settings.llm_timeout)

    config = LLMProviderConfig(
        provider_type=LLMProviderType(provider_type),
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
        extra_params=kwargs.get("extra_params"),
    )

    if config.provider_type == LLMProviderType.MOCK:
        return MockLLMProvider(config)
    elif config.provider_type == LLMProviderType.NVIDIA:
        return NVIDIALLMProvider(config)
    elif config.provider_type == LLMProviderType.OPENAI:
        return OpenAILLMProvider(config)
    elif config.provider_type == LLMProviderType.ANTHROPIC:
        return AnthropicLLMProvider(config)
    else:
        raise ValueError(f"Unsupported provider type: {config.provider_type}")


# Default provider instance (lazy initialization)
_default_provider: Optional[LLMProvider] = None


def get_default_provider() -> LLMProvider:
    """Get the default LLM provider instance."""
    global _default_provider
    if _default_provider is None:
        _default_provider = create_llm_provider()
    return _default_provider


async def generate_response(
    messages: List[LLMMessage],
    budget: Optional[ExecutionBudget] = None,
    provider: Optional[LLMProvider] = None,
    **kwargs,
) -> LLMResponse:
    """Convenience function to generate a response using the default or specified provider."""
    provider = provider or get_default_provider()
    return await provider.generate(messages, budget, **kwargs)