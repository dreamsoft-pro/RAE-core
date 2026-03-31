"""Sync models for RAE-Sync protocol."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SyncOperation(str, Enum):
    """Sync operation type."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class SyncChange(BaseModel):
    """Single sync change."""

    memory_id: UUID
    operation: SyncOperation
    data: dict[str, Any]
    timestamp: datetime
    version: int = Field(description="Version number for conflict resolution")


class SyncState(BaseModel):
    """Sync state for a tenant."""

    tenant_id: str
    last_sync_timestamp: datetime
    pending_changes: int = Field(default=0)
    sync_enabled: bool = Field(default=False)


class SyncConflict(BaseModel):
    """Sync conflict."""

    memory_id: UUID
    local_version: dict[str, Any]
    remote_version: dict[str, Any]
    resolved: bool = Field(default=False)
    resolution: dict[str, Any] | None = None


class SyncPeer(BaseModel):
    """Represents a connected RAE instance (node)."""

    peer_id: str
    role: str = Field(description="e.g., 'server', 'lite', 'mobile'")
    address: str | None = None
    last_seen: datetime = Field(default_factory=datetime.now)
    protocol_version: str = "1.0"
    tags: dict[str, str] = Field(default_factory=dict)


class SyncHandshake(BaseModel):
    """Handshake payload for establishing sync session."""

    peer_id: str
    role: str
    protocol_version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    capabilities: list[str] = Field(default_factory=list)


class SyncLog(BaseModel):
    """Log entry for sync events (Layer 2 Telemetry)."""

    id: UUID = Field(description="Unique sync event ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    peer_id: str
    direction: str  # "incoming", "outgoing"
    status: str  # "success", "failed", "conflict"
    items_synced: int = 0
    conflicts_resolved: int = 0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
