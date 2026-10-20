"""Timeout guard and budget checking for Precedent V1 (FR-9, FR-10)."""

from __future__ import annotations

from typing import Optional, Callable, TypeVar, Awaitable
from functools import wraps
import asyncio
import time

from app.core.budget import ExecutionBudget, BudgetViolation
from app.core.errors import InfrastructureError, ErrorCategory

T = TypeVar("T")


class TimeoutGuard:
    """
    Guard that checks both wall-clock timeout and budget limits.
    """

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget

    def check(self) -> bool:
        """Check if within all limits. Returns True if OK, False if any limit exceeded."""
        return self.budget.check_all_limits()

    def assert_ok(self, operation: str) -> None:
        """Assert all limits, raising appropriate exception if exceeded."""
        if not self.budget.check_wall_time_limit():
            raise InfrastructureError(
                f"Wall time limit exceeded during {operation}",
                diagnostic_code=f"ERR-TIMEOUT-{operation.upper()}",
            )
        self.budget.assert_all_limits(operation)


def with_timeout(
    timeout_ms: int,
    budget: Optional[ExecutionBudget] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Decorator to add timeout to an async function.
    Optionally checks budget limits as well.
    """
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=timeout_ms / 1000.0,
                )
            except asyncio.TimeoutError:
                if budget:
                    budget.consume_cost(0.0)  # Record attempt
                raise InfrastructureError(
                    f"Operation timed out after {timeout_ms}ms",
                    diagnostic_code=f"ERR-TIMEOUT-{func.__name__.upper()}",
                )
        return wrapper
    return decorator


def check_budget_or_timeout(
    budget: ExecutionBudget,
    operation: str,
) -> None:
    """
    Check budget limits and wall time, raising appropriate exceptions.
    Convenience function for inline checks.
    """
    if not budget.check_wall_time_limit():
        raise InfrastructureError(
            f"Wall time limit exceeded during {operation}",
            diagnostic_code=f"ERR-TIMEOUT-{operation.upper()}",
        )
    budget.assert_all_limits(operation)


class SyncTimeoutGuard:
    """Synchronous version of TimeoutGuard for non-async contexts."""

    def __init__(self, budget: ExecutionBudget):
        self.budget = budget
        self._start_time = time.perf_counter()

    def check(self) -> bool:
        return self.budget.check_all_limits()

    def assert_ok(self, operation: str) -> None:
        if not self.budget.check_wall_time_limit():
            raise InfrastructureError(
                f"Wall time limit exceeded during {operation}",
                diagnostic_code=f"ERR-TIMEOUT-{operation.upper()}",
            )
        self.budget.assert_all_limits(operation)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start_time) * 1000


def check_budget_or_timeout_sync(
    budget: ExecutionBudget,
    operation: str,
) -> None:
    """Synchronous version of check_budget_or_timeout."""
    if not budget.check_wall_time_limit():
        raise InfrastructureError(
            f"Wall time limit exceeded during {operation}",
            diagnostic_code=f"ERR-TIMEOUT-{operation.upper()}",
        )
    budget.assert_all_limits(operation)