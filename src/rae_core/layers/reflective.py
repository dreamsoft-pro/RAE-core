"""Reflective memory layer - meta-cognitive insights and patterns."""

from typing import Any
from uuid import UUID

from ..interfaces.storage import IMemoryStorage
from ..models.memory import MemoryItem, MemoryLayer, ScoredMemoryItem
from ..models.reflection import Reflection, ReflectionType
from .base import MemoryLayerBase


class ReflectiveLayer(MemoryLayerBase):
    """Reflective memory layer implementation.

    Characteristics:
    - Stores higher-order insights derived from other memories
    - Pattern recognition across episodic/semantic layers
    - Meta-cognitive awareness (contradictions, trends, insights)
    - Links back to source memories
    - Highest importance scores

    Typical use cases:
    - "User prefers Python over JavaScript" (pattern)
    - "Contradiction: user said X but did Y" (contradiction)
    - "Trend: user increasingly interested in AI" (trend)
    """

    def __init__(
        self,
        storage: IMemoryStorage,
        tenant_id: str,
        agent_id: str,
        min_source_memories: int = 3,
    ):
        """Initialize reflective layer.

        Args:
            storage: Storage backend
            tenant_id: Tenant ID
            agent_id: Agent ID
            min_source_memories: Minimum source memories required for reflection
        """
        super().__init__(storage, MemoryLayer.REFLECTIVE.value, tenant_id, agent_id)
        self.min_source_memories = min_source_memories

    async def add_memory(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
    ) -> UUID:
        """Add reflective memory (use add_reflection instead for proper structure)."""
        # Reflections should have high importance
        if importance is None:
            importance = 0.8

        return await self.storage.store_memory(
            content=content,
            layer=self.layer_name,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            tags=tags,
            metadata=metadata,
            embedding=embedding,
            importance=importance,
        )

    async def add_reflection(
        self,
        reflection: Reflection,
        embedding: list[float] | None = None,
    ) -> UUID:
        """Add a structured reflection.

        Args:
            reflection: Reflection object with content, type, source memories
            embedding: Optional vector embedding

        Returns:
            Memory UUID
        """
        # Validate minimum source memories
        if len(reflection.source_memory_ids) < self.min_source_memories:
            raise ValueError(
                f"Reflection requires at least {self.min_source_memories} "
                f"source memories, got {len(reflection.source_memory_ids)}"
            )

        # Store as reflective memory with structured metadata
        metadata = {
            "reflection_id": str(reflection.id),
            "reflection_type": reflection.reflection_type.value,
            "priority": reflection.priority.value,
            "source_memory_ids": [str(mid) for mid in reflection.source_memory_ids],
            "confidence": reflection.confidence,
        }

        # Map priority to importance
        priority_to_importance = {
            "low": 0.6,
            "medium": 0.7,
            "high": 0.85,
            "critical": 0.95,
        }
        importance = priority_to_importance.get(reflection.priority.value, 0.8)

        memory_id = await self.storage.store_memory(
            content=reflection.content,
            layer=self.layer_name,
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            tags=reflection.tags,
            metadata=metadata,
            embedding=embedding,
            importance=importance,
        )

        return memory_id

    async def get_memory(self, memory_id: UUID) -> MemoryItem | None:
        """Get reflective memory by ID."""
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

    async def get_reflection(self, memory_id: UUID) -> Reflection | None:
        """Get reflection as structured object."""
        memory = await self.get_memory(memory_id)
        if not memory:
            return None

        metadata = memory.metadata

        return Reflection(
            id=UUID(metadata.get("reflection_id", str(memory.id))),
            content=memory.content,
            reflection_type=ReflectionType(metadata.get("reflection_type", "insight")),
            priority=metadata.get("priority", "medium"),
            source_memory_ids=[
                UUID(mid) for mid in metadata.get("source_memory_ids", [])
            ],
            tenant_id=memory.tenant_id,
            agent_id=memory.agent_id,
            tags=memory.tags,
            confidence=metadata.get("confidence", 0.5),
            created_at=memory.created_at,
            last_updated_at=memory.last_accessed_at,
        )

    async def search_memories(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
        reflection_type: ReflectionType | None = None,
    ) -> list[ScoredMemoryItem]:
        """Search reflective memories.

        Args:
            query: Search query
            limit: Max results
            filters: Optional filters
            reflection_type: Filter by reflection type

        Returns:
            List of scored memories
        """
        if filters is None:
            filters = {}

        if reflection_type:
            filters["metadata.reflection_type"] = reflection_type.value

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
        """Remove low-confidence reflections.

        Returns:
            Number of reflections removed
        """
        # Remove reflections with very low confidence
        deleted_count = await self.storage.delete_memories_with_metadata_filter(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
            metadata_filter={"confidence__lt": 0.3},
        )
        return deleted_count

    async def get_reflections_by_type(
        self,
        reflection_type: ReflectionType,
        limit: int = 10,
    ) -> list[Reflection]:
        """Get reflections of a specific type.

        Args:
            reflection_type: Type of reflection to retrieve
            limit: Maximum number of results

        Returns:
            List of Reflection objects
        """
        memories = await self.storage.list_memories(
            tenant_id=self.tenant_id,
            agent_id=self.agent_id,
            layer=self.layer_name,
            filters={"metadata.reflection_type": reflection_type.value},
            limit=limit,
            order_by="importance",
            order_direction="desc",
        )

        reflections = []
        for mem_dict in memories:
            memory = MemoryItem(**mem_dict)
            metadata = memory.metadata

            reflection = Reflection(
                id=UUID(metadata.get("reflection_id", str(memory.id))),
                content=memory.content,
                reflection_type=reflection_type,
                priority=metadata.get("priority", "medium"),
                source_memory_ids=[
                    UUID(mid) for mid in metadata.get("source_memory_ids", [])
                ],
                tenant_id=memory.tenant_id,
                agent_id=memory.agent_id,
                tags=memory.tags,
                confidence=metadata.get("confidence", 0.5),
                created_at=memory.created_at,
                last_updated_at=memory.last_accessed_at,
            )
            reflections.append(reflection)

        return reflections

    async def find_contradictions(self) -> list[Reflection]:
        """Find all contradiction reflections.

        Returns:
            List of contradiction reflections
        """
        return await self.get_reflections_by_type(
            ReflectionType.CONTRADICTION,
            limit=50,
        )

    async def find_patterns(self) -> list[Reflection]:
        """Find all pattern reflections.

        Returns:
            List of pattern reflections
        """
        return await self.get_reflections_by_type(
            ReflectionType.PATTERN,
            limit=50,
        )
