"""Long-term memory layer - persistent storage (episodic + semantic)."""

from typing import Any
from uuid import UUID

from ..interfaces.storage import IMemoryStorage
from ..models.memory import MemoryItem, MemoryLayer, ScoredMemoryItem
from .base import MemoryLayerBase


class LongTermLayer(MemoryLayerBase):
    """Long-term memory layer implementation.

    Combines both episodic and semantic memory:
    - Episodic: Time-bound experiences and events
    - Semantic: Facts, concepts, and knowledge (decontextualized)

    Characteristics:
    - Persistent storage (no automatic decay)
    - High capacity
    - Indexed for efficient retrieval
    - Supports consolidation from working memory
    - Can be upgraded from episodic to semantic through reflection
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        tenant_id: str,
        agent_id: str,
    ):
        """Initialize long-term layer.

        Args:
            storage: Storage backend
            tenant_id: Tenant ID
            agent_id: Agent ID
        """
        # Long-term encompasses both episodic and semantic
        # We use "episodic" as default but allow semantic via metadata
        super().__init__(storage, MemoryLayer.EPISODIC.value, tenant_id, agent_id)

    async def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
        is_semantic: bool = False,
    ) -> UUID:
        """Add memory to long-term storage.

        Args:
            content: Memory content
            tags: Optional tags
            metadata: Optional metadata
            embedding: Optional vector embedding
            importance: Importance score
            is_semantic: True for semantic memory, False for episodic

        Returns:
            Memory UUID
        """
        # Default importance for long-term
        if importance is None:
            importance = 0.7 if is_semantic else 0.5

        # Mark as semantic in metadata
        if metadata is None:
            metadata = {}
        metadata["is_semantic"] = is_semantic
        metadata["memory_subtype"] = "semantic" if is_semantic else "episodic"

        # Use appropriate layer
        layer = (
            MemoryLayer.SEMANTIC.value if is_semantic else MemoryLayer.EPISODIC.value
        )

        memory_id = await self.storage.store_memory(
            content=content,
            layer=layer,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            tags=tags,
            metadata=metadata,
            embedding=embedding,
            importance=importance,
        )

        return memory_id

    async def get_memory(self, memory_id: UUID) -> MemoryItem | None:
        """Get memory by ID."""
        memory_dict = await self.storage.get_memory(
            memory_id=memory_id,
            tenant_id=self.tenant_id,
        )

        if not memory_dict:
            return None

        # Update access stats
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
        semantic_only: bool = False,
        episodic_only: bool = False,
    ) -> list[ScoredMemoryItem]:
        """Search long-term memories.

        Args:
            query: Search query
            limit: Max results
            filters: Optional filters
            semantic_only: Only search semantic memories
            episodic_only: Only search episodic memories

        Returns:
            List of scored memories
        """
        if filters is None:
            filters = {}

        # Add subtype filter
        if semantic_only:
            filters["metadata.is_semantic"] = True
        elif episodic_only:
            filters["metadata.is_semantic"] = False

        # Search both episodic and semantic layers
        layers = []
        if not semantic_only:
            layers.append(MemoryLayer.EPISODIC.value)
        if not episodic_only:
            layers.append(MemoryLayer.SEMANTIC.value)

        all_results = []
        for layer in layers:
            results = await self.storage.search_memories(
                query=query,
                tenant_id=self.tenant_id,
                agent_id=self.agent_id,
                layer=layer,
                limit=limit,
                filters=filters,
            )
            all_results.extend(results)

        # Sort by score and limit
        all_results.sort(key=lambda x: x["score"], reverse=True)
        all_results = all_results[:limit]

        return [
            ScoredMemoryItem(memory=MemoryItem(**r["memory"]), score=r["score"])
            for r in all_results
        ]

    async def cleanup(self) -> int:
        """Long-term memories don't auto-decay, but can prune very low importance.

        Returns:
            Number of memories removed
        """
        # Only remove if importance is extremely low (< 0.1)
        deleted_count = await self.storage.delete_memories_below_importance(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=MemoryLayer.EPISODIC.value,
            importance_threshold=0.1,
        )

        # Also check semantic layer
        semantic_deleted = await self.storage.delete_memories_below_importance(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=MemoryLayer.SEMANTIC.value,
            importance_threshold=0.1,
        )

        return deleted_count + semantic_deleted

    async def consolidate_from_working(
        self,
        working_memory: MemoryItem,
        as_semantic: bool = False,
    ) -> UUID:
        """Consolidate a working memory into long-term storage.

        Args:
            working_memory: Memory from working layer
            as_semantic: True to store as semantic, False for episodic

        Returns:
            UUID of new long-term memory
        """
        # Increase importance for consolidated memories
        new_importance = min(working_memory.importance + 0.2, 1.0)

        return await self.add_memory(
            content=working_memory.content,
            tags=working_memory.tags,
            metadata=working_memory.metadata,
            embedding=working_memory.embedding,
            importance=new_importance,
            is_semantic=as_semantic,
        )

    async def upgrade_to_semantic(
        self,
        episodic_memory_id: UUID,
        generalized_content: str | None = None,
    ) -> UUID:
        """Upgrade an episodic memory to semantic by decontextualizing.

        Args:
            episodic_memory_id: ID of episodic memory to upgrade
            generalized_content: Optional generalized/abstracted content

        Returns:
            UUID of new semantic memory
        """
        episodic_memory = await self.get_memory(episodic_memory_id)
        if not episodic_memory:
            raise ValueError(f"Memory {episodic_memory_id} not found")

        # Use generalized content or original
        content = (
            generalized_content if generalized_content else episodic_memory.content
        )

        # Create semantic version
        semantic_id = await self.add_memory(
            content=content,
            tags=episodic_memory.tags,
            metadata={
                **episodic_memory.metadata,
                "derived_from_episodic": str(episodic_memory_id),
            },
            embedding=episodic_memory.embedding,
            importance=min(episodic_memory.importance + 0.1, 1.0),
            is_semantic=True,
        )

        return semantic_id

    async def count_episodic(self) -> int:
        """Count episodic memories."""
        return await self.storage.count_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=MemoryLayer.EPISODIC.value,
        )

    async def count_semantic(self) -> int:
        """Count semantic memories."""
        return await self.storage.count_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=MemoryLayer.SEMANTIC.value,
        )
