"""Abstract cache provider interface for RAE-core."""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ICacheProvider(Protocol):
    """Abstract interface for cache providers (Redis, in-memory, etc.)."""

    async def get(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> Any | None:
        """Get value from cache."""
        ...

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Set value in cache with optional TTL in seconds."""
        ...

    async def delete(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Delete value from cache."""
        ...

    async def exists(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Check if key exists in cache."""
        ...

    async def clear(
        self,
        pattern: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear cache keys matching pattern (all if None)."""
        ...
