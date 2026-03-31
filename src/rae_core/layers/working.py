"""Working memory layer - active context with capacity limits."""

from typing import Any
from uuid import UUID

from ..interfaces.storage import IMemoryStorage
from ..models.memory import MemoryItem, MemoryLayer, ScoredMemoryItem
from .base import MemoryLayerBase


class WorkingLayer(MemoryLayerBase):
    """Working memory layer implementation.

    Characteristics:
    - Limited capacity (typically 7±2 items, not tokens)
    - High importance threshold
    - Active manipulation of information
    - No automatic decay, but capacity-based eviction
    - Feeds into long-term consolidation

    Typical capacity: 5-9 items
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        tenant_id: str,
        agent_id: str,
        max_capacity: int = 7,
        importance_threshold: float = 0.5,
    ):
        """Initialize working layer.

        Args:
            storage: Storage backend
            tenant_id: Tenant ID
            agent_id: Agent ID
            max_capacity: Maximum number of active items (default: 7)
            importance_threshold: Minimum importance to keep in working memory
        """
        super().__init__(storage, MemoryLayer.WORKING.value, tenant_id, agent_id)
        self.max_capacity = max_capacity
        self.importance_threshold = importance_threshold

    async def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
    ) -> UUID:
        """Add memory to working layer.

        If capacity is exceeded, lowest importance items are evicted
        (potentially consolidated to long-term first).

        Args:
            content: Memory content
            tags: Optional tags
            metadata: Optional metadata
            embedding: Optional vector embedding
            importance: Importance score (should be >= threshold)

        Returns:
            Memory UUID
        """
        # Working memories should have at least threshold importance
        if importance is None:
            importance = self.importance_threshold
        elif importance < self.importance_threshold:
            importance = self.importance_threshold

        memory_id = await self.storage.store_memory(
            content=content,
            layer=self.layer_name,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            tags=tags,
            metadata=metadata,
            embedding=embedding,
            importance=importance,
        )

        # Check capacity
        count = await self.count_memories()
        if count > self.max_capacity:
            await self._evict_least_important()

        return memory_id

    async def get_memory(self, memory_id: UUID) -> MemoryItem | None:
        """Get memory by ID and update access time."""
        memory_dict = await self.storage.get_memory(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
        )

        if not memory_dict:
            return None

        # Update last_accessed_at and usage_count
        await self.storage.update_memory_access(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
        )

        return MemoryItem(**memory_dict)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredMemoryItem]:
        """Search working memories.

        Args:
            query: Search query
            limit: Max results
            filters: Optional filters

        Returns:
            List of scored memories
        """
        results = await self.storage.search_memories(
            query=query,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
            limit=limit,
            filters=filters,
        )

        return [
            ScoredMemoryItem(memory=MemoryItem(**r["memory"]), score=r["score"])
            for r in results
        ]

    async def cleanup(self) -> int:
        """Remove items below importance threshold.

        Returns:
            Number of memories removed
        """
        deleted_count = await self.storage.delete_memories_below_importance(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
            importance_threshold=self.importance_threshold,
        )
        return deleted_count

    async def _evict_least_important(self) -> None:
        """Evict lowest importance item when capacity exceeded."""
        # Get all working memories sorted by importance
        memories = await self.storage.list_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
            order_by="importance",
            order_direction="asc",
            limit=1,
        )

        if memories:
            # Delete the least important
            await self.delete_memory(memories[0]["id"])

    async def promote_to_working(
        self,
        memory: MemoryItem,
        new_importance: float | None = None,
    ) -> UUID:
        """Promote a memory from sensory to working layer.

        Args:
            memory: Memory to promote
            new_importance: Optional new importance (default: keep existing or threshold)

        Returns:
            UUID of new working memory
        """
        importance = (
            new_importance
            if new_importance is not None
            else max(memory.importance, self.importance_threshold)
        )

        return await self.add_memory(
            content=memory.content,
            tags=memory.tags,
            metadata=memory.metadata,
            embedding=memory.embedding,
            importance=importance,
        )

    async def get_capacity_status(self) -> dict[str, Any]:
        """Get current capacity status.

        Returns:
            Dict with current_count, max_capacity, available_slots
        """
        current_count = await self.count_memories()
        return {
            "current_count": current_count,
            "max_capacity": self.max_capacity,
            "available_slots": max(0, self.max_capacity - current_count),
            "is_full": current_count >= self.max_capacity,
            "utilization_pct": (current_count / self.max_capacity) * 100,
        }
