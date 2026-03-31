"""Deterministic Clock for consistent time handling."""

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

@runtime_checkable
class IClock(Protocol):
    """Interface for time-keeping."""

    def now(self) -> datetime:
        """Get current time."""
        ...

class SystemClock:
    """Standard system clock."""

    def now(self) -> datetime:
        """Get current UTC time from system."""
        return datetime.now(timezone.utc)

class DeterministicClock:
    """Fixed or incrementing clock for deterministic testing."""

    def __init__(self, start_time: datetime | None = None) -> None:
        """Initialize with a fixed start time."""
        self._current = start_time or datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self._step_ms: int = 0

    def set_time(self, new_time: datetime) -> None:
        """Manually set the current time."""
        if new_time.tzinfo is None:
            new_time = new_time.replace(tzinfo=timezone.utc)
        self._current = new_time

    def set_auto_increment(self, ms: int) -> None:
        """Enable auto-increment by milliseconds on each call."""
        self._step_ms = ms

    def now(self) -> datetime:
        """Get current time and optionally increment."""
        current = self._current
        if self._step_ms > 0:
            from datetime import timedelta
            self._current += timedelta(milliseconds=self._step_ms)
        return current
