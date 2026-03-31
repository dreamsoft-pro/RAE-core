"""Sync protocol for memory synchronization."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from rae_core.interfaces.sync import ISyncProvider


class SyncMetadata(BaseModel):
    """Metadata for sync operations."""

    sync_id: str = Field(description="Unique sync operation ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_agent: str = Field(description="Source agent ID")
    target_agent: str | None = Field(default=None, description="Target agent ID")
    sync_type: str = Field(description="Type of sync: push, pull, sync")
    memory_count: int = Field(default=0, description="Number of memories synced")
    success: bool = Field(default=False)
    error_message: str | None = Field(default=None)


class SyncRequest(BaseModel):
    """Request for sync operation."""

    tenant_id: str
    agent_id: str
    sync_type: str  # push, pull, sync
    memory_ids: list[str] | None = Field(default=None)
    since: datetime | None = Field(default=None)
    batch_size: int = Field(default=100)
    encryption_enabled: bool = Field(default=True)


class SyncResponse(BaseModel):
    """Response from sync operation."""

    success: bool
    metadata: SyncMetadata
    synced_memory_ids: list[str] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


class SyncProtocol:
    """Protocol for memory synchronization operations.

    Implements push/pull/sync operations with conflict resolution.
    """

    def __init__(
        self,
        sync_provider: ISyncProvider,
        encryption_enabled: bool = True,
    ):
        """Initialize sync protocol.

        Args:
            sync_provider: Sync provider implementation
            encryption_enabled: Enable E2E encryption
        """
        self.sync_provider = sync_provider
        self.encryption_enabled = encryption_enabled

    async def push(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> SyncResponse:
        """Push local memories to remote.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            memory_ids: Optional specific memory IDs to push
            since: Optional timestamp to push memories modified since

        Returns:
            Sync response with results
        """
        request = SyncRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            sync_type="push",
            memory_ids=[str(mid) for mid in memory_ids] if memory_ids else None,
            since=since,
            encryption_enabled=self.encryption_enabled,
        )

        try:
            result = await self.sync_provider.push_memories(
                tenant_id=tenant_id,
                agent_id=agent_id,
                memory_ids=(
                    [UUID(mid) for mid in request.memory_ids]
                    if request.memory_ids
                    else None
                ),
                since=since,
            )

            metadata = SyncMetadata(
                sync_id=result.get("sync_id", ""),
                source_agent=agent_id,
                sync_type="push",
                memory_count=len(result.get("synced_memory_ids", [])),
                success=result.get("success", False),
            )

            return SyncResponse(
                success=result.get("success", False),
                metadata=metadata,
                synced_memory_ids=result.get("synced_memory_ids", []),
                conflicts=result.get("conflicts", []),
            )

        except Exception as e:
            metadata = SyncMetadata(
                sync_id="",
                source_agent=agent_id,
                sync_type="push",
                success=False,
                error_message=str(e),
            )
            return SyncResponse(
                success=False,
                metadata=metadata,
                error_message=str(e),
            )

    async def pull(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> SyncResponse:
        """Pull remote memories to local.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            memory_ids: Optional specific memory IDs to pull
            since: Optional timestamp to pull memories modified since

        Returns:
            Sync response with results
        """
        request = SyncRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            sync_type="pull",
            memory_ids=[str(mid) for mid in memory_ids] if memory_ids else None,
            since=since,
            encryption_enabled=self.encryption_enabled,
        )

        try:
            result = await self.sync_provider.pull_memories(
                tenant_id=tenant_id,
                agent_id=agent_id,
                memory_ids=(
                    [UUID(mid) for mid in request.memory_ids]
                    if request.memory_ids
                    else None
                ),
                since=since,
            )

            metadata = SyncMetadata(
                sync_id=result.get("sync_id", ""),
                source_agent=agent_id,
                sync_type="pull",
                memory_count=len(result.get("synced_memory_ids", [])),
                success=result.get("success", False),
            )

            return SyncResponse(
                success=result.get("success", False),
                metadata=metadata,
                synced_memory_ids=result.get("synced_memory_ids", []),
                conflicts=result.get("conflicts", []),
            )

        except Exception as e:
            metadata = SyncMetadata(
                sync_id="",
                source_agent=agent_id,
                sync_type="pull",
                success=False,
                error_message=str(e),
            )
            return SyncResponse(
                success=False,
                metadata=metadata,
                error_message=str(e),
            )

    async def sync(
        self,
        tenant_id: str,
        agent_id: str,
        since: datetime | None = None,
    ) -> SyncResponse:
        """Bidirectional sync between local and remote.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            since: Optional timestamp to sync memories modified since

        Returns:
            Sync response with results
        """
        SyncRequest(
            tenant_id=tenant_id,
            agent_id=agent_id,
            sync_type="sync",
            since=since,
            encryption_enabled=self.encryption_enabled,
        )

        try:
            result = await self.sync_provider.sync_memories(
                tenant_id=tenant_id,
                agent_id=agent_id,
                since=since,
            )

            metadata = SyncMetadata(
                sync_id=result.get("sync_id", ""),
                source_agent=agent_id,
                sync_type="sync",
                memory_count=len(result.get("synced_memory_ids", [])),
                success=result.get("success", False),
            )

            return SyncResponse(
                success=result.get("success", False),
                metadata=metadata,
                synced_memory_ids=result.get("synced_memory_ids", []),
                conflicts=result.get("conflicts", []),
            )

        except Exception as e:
            metadata = SyncMetadata(
                sync_id="",
                source_agent=agent_id,
                sync_type="sync",
                success=False,
                error_message=str(e),
            )
            return SyncResponse(
                success=False,
                metadata=metadata,
                error_message=str(e),
            )

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
        return await self.sync_provider.get_sync_status(
            tenant_id=tenant_id,
            agent_id=agent_id,
            sync_id=sync_id,
        )

    async def handshake(
        self,
        tenant_id: str,
        agent_id: str,
        capabilities: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Perform handshake to negotiate sync capabilities.

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            capabilities: Optional dictionary of supported capabilities

        Returns:
            Negotiated capabilities and session info
        """
        return await self.sync_provider.handshake(
            tenant_id=tenant_id,
            agent_id=agent_id,
            capabilities=capabilities or {},
        )
