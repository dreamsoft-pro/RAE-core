"""Base class for all memory layers in RAE-core."""

from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from ..interfaces.storage import IMemoryStorage
from ..models.memory import MemoryItem, ScoredMemoryItem


class MemoryLayerBase(ABC):
    """Abstract base class for memory layers.

    All memory layers (Sensory, Working, LongTerm, Reflective) inherit from this
    and implement layer-specific logic for storage, retrieval, and lifecycle.
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        layer_name: str,
        tenant_id: str,
        agent_id: str,
    ):
        """Initialize memory layer.

        Args:
            storage: Storage backend implementation
            layer_name: Name of this layer (sensory, working, episodic, semantic, reflective)
            tenant_id: Tenant identifier for multi-tenancy
            agent_id: Agent identifier
        """
        self.storage = storage
        self.layer_name = layer_name
        self.tenant_id = tenant_id
        self.agent_id = agent_id

    @abstractmethod
    async def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
    ) -> UUID:
        """Add a memory to this layer.

        Returns:
            UUID of the created memory
        """
        pass  # pragma: no cover

    @abstractmethod
    async def get_memory(self, memory_id: UUID) -> MemoryItem | None:
        """Retrieve a memory by ID."""
        pass  # pragma: no cover

    @abstractmethod
    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredMemoryItem]:
        """Search memories in this layer.

        Args:
            query: Search query
            limit: Maximum number of results
            filters: Optional filters (tags, date ranges, etc.)

        Returns:
            List of scored memory items
        """
        pass  # pragma: no cover

    @abstractmethod
    async def cleanup(self) -> int:
        """Perform layer-specific cleanup (decay, consolidation, etc.).

        Returns:
            Number of memories affected
        """
        pass  # pragma: no cover

    async def count_memories(self) -> int:
        """Count total memories in this layer."""
        return await self.storage.count_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
        )

    async def delete_memory(self, memory_id: UUID) -> bool:
        """Delete a memory from this layer."""
        return await self.storage.delete_memory(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
        )
