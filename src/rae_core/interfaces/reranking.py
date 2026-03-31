"""Abstract reranking interface for RAE-core."""

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class IReranker(Protocol):
    """Abstract interface for reranking strategies."""

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[UUID, float, float]],
        tenant_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        """
        Re-rank candidate memories.

        Args:
            query: The search query.
            candidates: List of (memory_id, score, importance) tuples.
            tenant_id: Tenant context.
            limit: Number of results to return.
        """
        ...
