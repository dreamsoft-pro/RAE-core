"""Abstract storage interface for RAE-core.

This module defines the storage interface that all storage adapters must implement.
Storage can be PostgreSQL, SQLite, in-memory, or any other backend.
"""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class IMemoryStorage(Protocol):
    """Abstract interface for memory storage.

    Implementations must provide persistent storage for memories across
    all memory layers (sensory, working, episodic, semantic, reflective).
    """

    async def store_memory(self, **kwargs: Any) -> UUID:
        """Store a new memory."""
        ...

    async def store_reflection_audit(
        self,
        query_id: str,
        tenant_id: str,
        fsi_score: float,
        final_decision: str,
        l1_report: dict[str, Any],
        l2_report: dict[str, Any],
        l3_report: dict[str, Any],
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Store a 3-layer reflection audit result."""
        ...

    async def get_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a memory by ID."""
        ...

    async def get_memories_batch(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve multiple memories by IDs."""
        ...

    async def update_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update a memory."""
        ...

    async def delete_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a memory."""
        ...

    async def list_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """List memories with filtering and sorting."""
        ...

    async def delete_memories_with_metadata_filter(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories matching metadata filter."""
        ...

    async def delete_memories_below_importance(
        self,
        tenant_id: str,
        agent_id: str,
        layer: str,
        importance_threshold: float,
    ) -> int:
        """Delete memories below importance threshold."""
        ...

    async def count_memories(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> int:
        """Count memories matching filters."""
        ...

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        agent_id: str,
        layer: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search memories."""
        ...

    async def delete_expired_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> int:
        """Delete expired memories."""
        ...

    async def update_memory_access(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Update last access time and increment usage count."""
        ...

    async def increment_access_count(self, memory_id: UUID, tenant_id: str) -> bool:
        """Alias for update_memory_access (Legacy test support)."""
        ...

    async def update_memory_expiration(
        self,
        memory_id: UUID,
        tenant_id: str,
        expires_at: datetime | None,
    ) -> bool:
        """Update memory expiration time."""
        ...

    async def get_metric_aggregate(
        self,
        tenant_id: str,
        metric: str,
        func: str,
        filters: dict[str, Any] | None = None,
    ) -> float:
        """Calculate aggregate metric."""
        ...

    async def update_memory_access_batch(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
    ) -> bool:
        """Update access count for multiple memories."""
        ...

    async def adjust_importance(
        self,
        memory_id: UUID,
        delta: float,
        tenant_id: str,
    ) -> float:
        """Adjust memory importance."""
        ...

    async def save_embedding(
        self,
        memory_id: UUID,
        model_name: str,
        embedding: list[float],
        tenant_id: str,
        **kwargs: Any,
    ) -> bool:
        """Save a vector embedding."""
        ...

    async def decay_importance(
        self,
        tenant_id: str,
        decay_factor: float,
    ) -> int:
        """Apply importance decay."""
        ...

    async def clear_tenant(self, tenant_id: str) -> int:
        """Delete all memories for a tenant."""
        ...

    async def close(self) -> None:
        """Close storage connection."""
        ...
