from typing import Any
from uuid import UUID

from ...interfaces.embedding import IEmbeddingProvider
from ...interfaces.vector import IVectorStore
from . import SearchStrategy


class VectorSearchStrategy(SearchStrategy):
    """Dense vector search strategy."""

    def __init__(
        self,
        vector_store: IVectorStore,
        embedding_provider: IEmbeddingProvider,
        default_weight: float = 1.0,
        vector_name: str | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.default_weight = default_weight
        self.vector_name = vector_name

    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        # Prepare search kwargs
        search_kwargs = kwargs.copy()
        if self.vector_name:
            search_kwargs["vector_name"] = self.vector_name

        # Generate embedding for the query
        query_embedding = await self.embedding_provider.embed_text(
            query, task_type="search_query"
        )

        results = await self.vector_store.search_similar(
            query_embedding=query_embedding, tenant_id=tenant_id, limit=limit, **search_kwargs
        )

        # Convert to 3-tuple (id, score, importance)
        # Importance is 0.0 for now, as it's typically fetched from storage later
        return [(r[0], r[1], 0.0) for r in results]

    def get_strategy_name(self) -> str:
        return "vector"

    def get_strategy_weight(self) -> float:
        return self.default_weight
