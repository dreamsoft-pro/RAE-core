"""Abstract sync provider interface for RAE-core."""

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class ISyncProvider(Protocol):
    """Abstract interface for sync protocol (RAE-Sync)."""

    async def push_changes(self, tenant_id: str, changes: list[dict[str, Any]]) -> bool:
        """Push local changes to sync endpoint."""
        ...

    async def pull_changes(
        self, tenant_id: str, since_timestamp: str
    ) -> list[dict[str, Any]]:
        """Pull changes from sync endpoint since timestamp."""
        ...

    async def resolve_conflict(
        self,
        memory_id: UUID,
        local_version: dict[str, Any],
        remote_version: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve conflict using CRDT merge strategy."""
        ...

    async def push_memories(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Push memories to remote.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            memory_ids: Optional specific memory IDs to push
            since: Optional timestamp to push memories modified since

        Returns:
            Dictionary with sync results
        """
        ...

    async def pull_memories(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Pull memories from remote.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            memory_ids: Optional specific memory IDs to pull
            since: Optional timestamp to pull memories modified since

        Returns:
            Dictionary with sync results
        """
        ...

    async def sync_memories(
        self,
        tenant_id: str,
        agent_id: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Bidirectional sync of memories.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            since: Optional timestamp to sync memories modified since

        Returns:
            Dictionary with sync results
        """
        ...

    async def get_sync_status(
        self,
        tenant_id: str,
        agent_id: str,
        sync_id: str,
    ) -> dict[str, Any]:
        """Get status of a sync operation.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            sync_id: Sync operation ID

        Returns:
            Sync status information
        """
        ...

    async def handshake(
        self,
        tenant_id: str,
        agent_id: str,
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        """Perform handshake to negotiate sync capabilities.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            capabilities: Dictionary of supported capabilities and versions

        Returns:
            Negotiated capabilities and session info
        """
        ...
