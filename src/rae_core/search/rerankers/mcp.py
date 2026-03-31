from typing import Any
from uuid import UUID

from ...interfaces.reranking import IReranker


class McpReranker(IReranker):
    """Reranker using MCP tool."""

    def __init__(self, tool_name: str = "rerank"):
        self.tool_name = tool_name

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[UUID, float, float]],
        tenant_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        # Placeholder for real MCP call
        return candidates[:limit]
