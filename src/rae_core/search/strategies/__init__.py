"""Search strategy interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID


class SearchStrategy(ABC):
    """Base class for all search strategies."""

    @abstractmethod
    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        """Execute search and return (memory_id, score, importance) tuples.

        Args:
            query: Search query string
            tenant_id: Tenant identifier
            filters: Optional filters (layer, agent_id, tags, etc.)
            limit: Maximum number of results
            project: Optional project identifier

        Returns:
            List of (memory_id, score, importance) tuples sorted by score descending
        """
        pass  # pragma: no cover

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return the name of this strategy."""
        pass  # pragma: no cover

    @abstractmethod
    def get_strategy_weight(self) -> float:
        """Return default weight for hybrid search fusion (0.0-1.0)."""
        pass  # pragma: no cover
