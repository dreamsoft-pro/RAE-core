from typing import Any
from uuid import UUID

from ...interfaces.graph import IGraphStore
from ...interfaces.storage import IMemoryStorage
from . import SearchStrategy


class GraphTraversalStrategy(SearchStrategy):
    """Search strategy using graph traversal from seed memories."""

    def __init__(
        self,
        graph_store: IGraphStore,
        memory_storage: IMemoryStorage,
        default_weight: float = 0.5,
    ) -> None:
        self.graph_store = graph_store
        self.memory_storage = memory_storage
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
        """Traverse graph from seeds and return neighbors."""
        search_filters = filters or {}
        seed_ids = kwargs.get("seed_ids") or search_filters.get("seed_ids", [])

        if not seed_ids:
            return []

        # Convert string IDs to UUIDs
        seeds = []
        for sid in seed_ids:
            if isinstance(sid, str):
                try:
                    seeds.append(UUID(sid))
                except ValueError:
                    continue
            else:
                seeds.append(sid)

        visited = set(seeds)
        to_visit = list(seeds)
        results: dict[UUID, float] = {}

        # Simple BFS traversal
        depth = 0
        max_depth = 2

        while to_visit and depth < max_depth:
            current_layer = to_visit
            to_visit = []
            depth += 1

            for node_id in current_layer:
                neighbors = await self.graph_store.get_neighbors(node_id, tenant_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        to_visit.append(neighbor_id)
                        # Multi-path boost: increase score if reached via multiple paths
                        results[neighbor_id] = results.get(neighbor_id, 0.0) + (
                            1.0 / depth
                        )

        # Sort and return
        final = sorted(results.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(m_id, score, 0.0) for m_id, score in final]

    def get_strategy_name(self) -> str:
        return "graph"

    def get_strategy_weight(self) -> float:
        return self.default_weight
