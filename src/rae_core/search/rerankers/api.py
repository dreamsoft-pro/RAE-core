from typing import Any
from uuid import UUID

from ...interfaces.reranking import IReranker


class ApiReranker(IReranker):
    """Reranker using external API."""

    def __init__(self, api_url: str, api_key: str | None = None):
        self.api_url = api_url
        self.api_key = api_key

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[UUID, float, float]],
        tenant_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        # Placeholder for real API call
        return candidates[:limit]
