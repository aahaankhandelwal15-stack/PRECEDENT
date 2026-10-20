"""Execution budget tracking for Precedent V1 (FR-9, FR-10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


@dataclass
class BudgetRecord:
    """Immutable snapshot of budget consumption at a point in time."""
    llm_calls: int
    tokens_input: int
    tokens_output: int
    retries: int
    wall_time_ms: float
    estimated_cost: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BudgetViolation(Exception):
    """Raised when a budget limit is exceeded."""
    def __init__(self, limit_name: str, current: int, limit: int, operation: str):
        self.limit_name = limit_name
        self.current = current
        self.limit = limit
        self.operation = operation
        super().__init__(
            f"Budget violation: {limit_name} limit ({limit}) exceeded during {operation}. Current: {current}"
        )


class ExecutionBudget:
    """
    Tracks execution budget for a single query pipeline run.
    Enforces limits on LLM calls, tokens, retries, wall time, and cost.
    """

    def __init__(
        self,
        max_llm_calls: int = 10,
        max_tokens: int = 50000,
        max_retries: int = 3,
        max_wall_time_ms: int = 30000,
        max_estimated_cost: float = 10.0,
    ):
        self.max_llm_calls = max_llm_calls
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.max_wall_time_ms = max_wall_time_ms
        self.max_estimated_cost = max_estimated_cost

        # Current consumption
        self.llm_calls: int = 0
        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.retries: int = 0
        self.estimated_cost: float = 0.0

        # Wall time tracking
        self._start_time: Optional[datetime] = None
        self._wall_time_ms: float = 0.0

    def start_wall_timer(self) -> None:
        """Start the wall clock timer."""
        self._start_time = datetime.now(timezone.utc)

    def _update_wall_time(self) -> None:
        """Update wall time from start timer."""
        if self._start_time is not None:
            elapsed = datetime.now(timezone.utc) - self._start_time
            self._wall_time_ms = elapsed.total_seconds() * 1000

    @property
    def wall_time_ms(self) -> float:
        """Current wall time in milliseconds."""
        self._update_wall_time()
        return self._wall_time_ms

    def consume_llm_call(
        self,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0.0,
        retries: int = 0,
    ) -> None:
        """Record an LLM call consumption."""
        self.llm_calls += 1
        self.tokens_input += tokens_input
        self.tokens_output += tokens_output
        self.estimated_cost += cost
        self.retries += retries

    def consume_retry(self) -> None:
        """Record a retry attempt."""
        self.retries += 1

    def consume_tokens(self, tokens_input: int, tokens_output: int) -> None:
        """Record token consumption without an LLM call (e.g., embedding)."""
        self.tokens_input += tokens_input
        self.tokens_output += tokens_output

    def consume_cost(self, cost: float) -> None:
        """Record cost consumption."""
        self.estimated_cost += cost

    # Limit checks (return True if within limit, False if exceeded)
    def check_llm_calls_limit(self) -> bool:
        return self.llm_calls < self.max_llm_calls

    def check_tokens_limit(self) -> bool:
        return (self.tokens_input + self.tokens_output) < self.max_tokens

    def check_retries_limit(self) -> bool:
        return self.retries < self.max_retries

    def check_wall_time_limit(self) -> bool:
        return self.wall_time_ms < self.max_wall_time_ms

    def check_cost_limit(self) -> bool:
        return self.estimated_cost < self.max_estimated_cost

    def check_all_limits(self) -> bool:
        """Check all limits at once."""
        return all([
            self.check_llm_calls_limit(),
            self.check_tokens_limit(),
            self.check_retries_limit(),
            self.check_wall_time_limit(),
            self.check_cost_limit(),
        ])

    # Assert versions (raise BudgetViolation if exceeded)
    def assert_llm_calls_limit(self, operation: str) -> None:
        if not self.check_llm_calls_limit():
            raise BudgetViolation("llm_calls", self.llm_calls, self.max_llm_calls, operation)

    def assert_tokens_limit(self, operation: str) -> None:
        if not self.check_tokens_limit():
            raise BudgetViolation("tokens", self.tokens_input + self.tokens_output, self.max_tokens, operation)

    def assert_retries_limit(self, operation: str) -> None:
        if not self.check_retries_limit():
            raise BudgetViolation("retries", self.retries, self.max_retries, operation)

    def assert_wall_time_limit(self, operation: str) -> None:
        if not self.check_wall_time_limit():
            raise BudgetViolation("wall_time_ms", int(self.wall_time_ms), self.max_wall_time_ms, operation)

    def assert_cost_limit(self, operation: str) -> None:
        if not self.check_cost_limit():
            raise BudgetViolation("estimated_cost", self.estimated_cost, self.max_estimated_cost, operation)

    def assert_all_limits(self, operation: str) -> None:
        """Assert all limits, raising on first violation."""
        self.assert_llm_calls_limit(operation)
        self.assert_tokens_limit(operation)
        self.assert_retries_limit(operation)
        self.assert_wall_time_limit(operation)
        self.assert_cost_limit(operation)

    @property
    def budget_record(self) -> BudgetRecord:
        """Get immutable snapshot of current budget state."""
        return BudgetRecord(
            llm_calls=self.llm_calls,
            tokens_input=self.tokens_input,
            tokens_output=self.tokens_output,
            retries=self.retries,
            wall_time_ms=self.wall_time_ms,
            estimated_cost=self.estimated_cost,
        )

    def to_dict(self) -> dict:
        """Serialize for trace record."""
        return {
            "llm_calls": self.llm_calls,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "retries": self.retries,
            "wall_time_ms": self.wall_time_ms,
            "estimated_cost": self.estimated_cost,
        }