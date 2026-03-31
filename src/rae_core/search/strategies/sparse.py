from typing import Any
from uuid import UUID

from ...interfaces.storage import IMemoryStorage
from . import SearchStrategy


class SparseVectorStrategy(SearchStrategy):
    """Sparse vector search strategy (e.g. BM25)."""

    def __init__(
        self, memory_storage: IMemoryStorage, default_weight: float = 0.7
    ) -> None:
        self.storage = memory_storage
        self.default_weight = default_weight

    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        # Placeholder
        return []

    def get_strategy_name(self) -> str:
        return "sparse"

    def get_strategy_weight(self) -> float:
        return self.default_weight
