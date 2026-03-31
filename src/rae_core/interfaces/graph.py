"""Abstract graph store interface for RAE-core."""

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class IGraphStore(Protocol):
    """Abstract interface for knowledge graph storage."""

    async def create_node(
        self,
        node_id: UUID,
        node_type: str,
        tenant_id: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a graph node."""
        ...

    async def create_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        tenant_id: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a graph edge."""
        ...

    async def get_neighbors(
        self,
        node_id: UUID,
        tenant_id: str,
        edge_type: str | None = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[UUID]:
        """Get neighboring nodes."""
        ...

    async def delete_node(self, node_id: UUID, tenant_id: str) -> bool:
        """Delete a node and its edges."""
        ...

    async def delete_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        tenant_id: str,
    ) -> bool:
        """Delete an edge."""
        ...

    async def shortest_path(
        self,
        source_id: UUID,
        target_id: UUID,
        tenant_id: str,
        max_depth: int = 5,
    ) -> list[UUID] | None:
        """Find shortest path between nodes."""
        ...

    async def get_subgraph(
        self, node_ids: list[UUID], tenant_id: str, include_edges: bool = True
    ) -> dict[str, Any]:
        """Extract a subgraph."""
        ...
