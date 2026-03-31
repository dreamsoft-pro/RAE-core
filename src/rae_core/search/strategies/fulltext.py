from typing import Any
from uuid import UUID

from ...interfaces.storage import IMemoryStorage
from . import SearchStrategy


class FullTextStrategy(SearchStrategy):
    """Full-text lexical search strategy."""

    def __init__(
        self, memory_storage: IMemoryStorage, default_weight: float = 1.0
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
        agent_id = kwargs.get("agent_id") or (filters or {}).get("agent_id", "default")
        # Default to None (search all layers) if not specified
        layer = kwargs.get("layer") or (filters or {}).get("layer")
        project = project or (filters or {}).get("project")

        results = await self.storage.search_memories(
            query=query,
            tenant_id=tenant_id,
            limit=limit,
            agent_id=agent_id,
            layer=layer,
            project=project,
        )

        output: list[tuple[UUID, float, float]] = []
        for r in results:
            m_id_raw = r.get("id")
            if isinstance(m_id_raw, str):
                m_id = UUID(m_id_raw)
            elif isinstance(m_id_raw, UUID):
                m_id = m_id_raw
            else:
                continue

            score = float(r.get("score", 0.0))
            importance = float(r.get("importance", 0.0))
            output.append((m_id, score, importance))

        return output

    def get_strategy_name(self) -> str:
        return "fulltext"

    def get_strategy_weight(self) -> float:
        return self.default_weight
