"""Sensory memory layer - short-term buffer with automatic decay."""

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from ..interfaces.storage import IMemoryStorage
from ..models.memory import MemoryItem, MemoryLayer, ScoredMemoryItem
from .base import MemoryLayerBase


class SensoryLayer(MemoryLayerBase):
    """Sensory memory layer implementation.

    Characteristics:
    - Very short retention (seconds to minutes)
    - Automatic decay based on TTL
    - High capacity, low importance threshold
    - Acts as buffer before working memory

    Typical TTL: 60-300 seconds
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        tenant_id: str,
        agent_id: str,
        default_ttl_seconds: int = 60,
        max_capacity: int = 1000,
    ):
        """Initialize sensory layer.

        Args:
            storage: Storage backend
            tenant_id: Tenant ID
            agent_id: Agent ID
            default_ttl_seconds: Default time-to-live for memories (default: 60s)
            max_capacity: Maximum number of memories before cleanup
        """
        super().__init__(storage, MemoryLayer.SENSORY.value, tenant_id, agent_id)
        self.default_ttl_seconds = default_ttl_seconds
        self.max_capacity = max_capacity

    async def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
        ttl_seconds: int | None = None,
    ) -> UUID:
        """Add memory to sensory layer with TTL.

        Args:
            content: Memory content
            tags: Optional tags
            metadata: Optional metadata
            embedding: Optional vector embedding
            importance: Importance score (default: 0.1 for sensory)
            ttl_seconds: Custom TTL, uses default if not specified

        Returns:
            Memory UUID
        """
        # Sensory memories have low importance by default
        if importance is None:
            importance = 0.1

        # Calculate expiration time
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl_seconds
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)

        # Add metadata about TTL
        if metadata is None:
            metadata = {}
        metadata["ttl_seconds"] = ttl
        metadata["expires_at"] = expires_at.isoformat()

        memory_id = await self.storage.store_memory(
            content=content,
            layer=self.layer_name,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            tags=tags,
            metadata=metadata,
            embedding=embedding,
            importance=importance,
            expires_at=expires_at,
        )

        # Check capacity and trigger cleanup if needed
        count = await self.count_memories()
        if count > self.max_capacity:
            await self.cleanup()

        return memory_id

    async def get_memory(self, memory_id: UUID) -> MemoryItem | None:
        """Get memory by ID, returns None if expired."""
        memory_dict = await self.storage.get_memory(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
        )

        if not memory_dict:
            return None

        # Check if expired
        expires_at = memory_dict.get("expires_at")
        if expires_at and expires_at < datetime.now(timezone.utc):
            # Delete expired memory
            await self.delete_memory(memory_id)
            return None

        return MemoryItem(**memory_dict)

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredMemoryItem]:
        """Search sensory memories, excluding expired ones.

        Args:
            query: Search query
            limit: Max results
            filters: Optional filters

        Returns:
            List of scored memories
        """
        # Add filter to exclude expired memories
        if filters is None:
            filters = {}
        filters["not_expired"] = True

        # Delegate to storage implementation
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
        """Remove expired memories from sensory layer.

        Returns:
            Number of memories deleted
        """
        deleted_count = await self.storage.delete_expired_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
        )
        return deleted_count

    async def extend_ttl(self, memory_id: UUID, additional_seconds: int) -> bool:
        """Extend TTL for a memory (e.g., if still relevant).

        Args:
            memory_id: Memory to extend
            additional_seconds: Seconds to add to current expiration

        Returns:
            True if successful
        """
        memory = await self.get_memory(memory_id)
        if not memory or not memory.expires_at:
            return False

        new_expires_at = memory.expires_at + timedelta(seconds=additional_seconds)

        return await self.storage.update_memory_expiration(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
            expires_at=new_expires_at,
        )
