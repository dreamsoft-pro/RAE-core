"""Memory diff calculation for synchronization."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ChangeType(str, Enum):
    """Type of change in memory."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    UNCHANGED = "unchanged"


class MemoryChange(BaseModel):
    """Represents a change in memory."""

    memory_id: UUID
    change_type: ChangeType
    local_version: int | None = None
    remote_version: int | None = None
    local_modified: datetime | None = None
    remote_modified: datetime | None = None
    conflicts: bool = Field(default=False)
    conflict_fields: list[str] = Field(default_factory=list)


class DiffResult(BaseModel):
    """Result of memory diff operation."""

    tenant_id: str
    agent_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created: list[MemoryChange] = Field(default_factory=list)
    modified: list[MemoryChange] = Field(default_factory=list)
    deleted: list[MemoryChange] = Field(default_factory=list)
    unchanged: list[MemoryChange] = Field(default_factory=list)
    conflicts: list[MemoryChange] = Field(default_factory=list)

    @property
    def has_changes(self) -> bool:
        """Check if there are any changes."""
        return bool(self.created or self.modified or self.deleted)

    @property
    def has_conflicts(self) -> bool:
        """Check if there are any conflicts."""
        return bool(self.conflicts)

    @property
    def total_changes(self) -> int:
        """Get total number of changes."""
        return len(self.created) + len(self.modified) + len(self.deleted)


def calculate_memory_diff(
    local_memories: list[dict[str, Any]],
    remote_memories: list[dict[str, Any]],
    tenant_id: str,
    agent_id: str,
) -> DiffResult:
    """Calculate differences between local and remote memories.

    Args:
        local_memories: List of local memory records
        remote_memories: List of remote memory records
        tenant_id: Tenant identifier
        agent_id: Agent identifier

    Returns:
        DiffResult with categorized changes
    """
    result = DiffResult(tenant_id=tenant_id, agent_id=agent_id)

    # Create lookup dictionaries
    local_dict = {str(mem["id"]): mem for mem in local_memories}
    remote_dict = {str(mem["id"]): mem for mem in remote_memories}

    local_ids = set(local_dict.keys())
    remote_ids = set(remote_dict.keys())

    # Find created memories (in remote but not in local)
    created_ids = remote_ids - local_ids
    for mem_id in created_ids:
        remote_mem = remote_dict[mem_id]
        change = MemoryChange(
            memory_id=UUID(mem_id),
            change_type=ChangeType.CREATED,
            remote_version=remote_mem.get("version", 1),
            remote_modified=remote_mem.get("modified_at"),
        )
        result.created.append(change)

    # Find deleted memories (in local but not in remote)
    deleted_ids = local_ids - remote_ids
    for mem_id in deleted_ids:
        local_mem = local_dict[mem_id]
        change = MemoryChange(
            memory_id=UUID(mem_id),
            change_type=ChangeType.DELETED,
            local_version=local_mem.get("version", 1),
            local_modified=local_mem.get("modified_at"),
        )
        result.deleted.append(change)

    # Find modified or unchanged memories (in both)
    common_ids = local_ids & remote_ids
    for mem_id in common_ids:
        local_mem = local_dict[mem_id]
        remote_mem = remote_dict[mem_id]

        local_version = local_mem.get("version", 1)
        remote_version = remote_mem.get("version", 1)
        local_modified = local_mem.get("modified_at")
        remote_modified = remote_mem.get("modified_at")

        # Check if modified
        if _is_memory_modified(local_mem, remote_mem):
            # Check for conflicts
            conflicts, conflict_fields = _detect_conflicts(
                local_mem, remote_mem, local_modified, remote_modified
            )

            change = MemoryChange(
                memory_id=UUID(mem_id),
                change_type=ChangeType.MODIFIED,
                local_version=local_version,
                remote_version=remote_version,
                local_modified=local_modified,
                remote_modified=remote_modified,
                conflicts=conflicts,
                conflict_fields=conflict_fields,
            )

            if conflicts:
                result.conflicts.append(change)
            else:
                result.modified.append(change)
        else:
            change = MemoryChange(
                memory_id=UUID(mem_id),
                change_type=ChangeType.UNCHANGED,
                local_version=local_version,
                remote_version=remote_version,
                local_modified=local_modified,
                remote_modified=remote_modified,
            )
            result.unchanged.append(change)

    return result


def _is_memory_modified(local: dict[str, Any], remote: dict[str, Any]) -> bool:
    """Check if memory has been modified.

    Args:
        local: Local memory record
        remote: Remote memory record

    Returns:
        True if modified, False otherwise
    """
    # Compare key fields
    compare_fields = ["content", "importance", "tags", "metadata", "version"]

    for field in compare_fields:
        local_value = local.get(field)
        remote_value = remote.get(field)

        # Normalize None and empty values
        if local_value != remote_value:
            return True

    return False


def _detect_conflicts(
    local: dict[str, Any],
    remote: dict[str, Any],
    local_modified: datetime | None,
    remote_modified: datetime | None,
) -> tuple[bool, list[str]]:
    """Detect conflicts between local and remote versions.

    Conflicts occur when:
    1. Both versions have been modified since last sync
    2. Different fields have been changed

    Args:
        local: Local memory record
        remote: Remote memory record
        local_modified: Local modification timestamp
        remote_modified: Remote modification timestamp

    Returns:
        Tuple of (has_conflicts, list_of_conflicting_fields)
    """
    conflict_fields = []

    # Check if both have been modified (potential conflict)
    if local_modified and remote_modified:
        # If timestamps are very close (within 1 second), no conflict
        if abs((local_modified - remote_modified).total_seconds()) < 1:
            return False, []

        # Check which fields differ
        compare_fields = ["content", "importance", "tags", "metadata"]
        for field in compare_fields:
            local_value = local.get(field)
            remote_value = remote.get(field)

            if local_value != remote_value:
                conflict_fields.append(field)

    has_conflicts = len(conflict_fields) > 0
    return has_conflicts, conflict_fields


def get_sync_direction(change: MemoryChange) -> str:
    """Determine sync direction based on change.

    Args:
        change: Memory change

    Returns:
        Sync direction: "push", "pull", or "conflict"
    """
    if change.conflicts:
        return "conflict"

    if change.change_type == ChangeType.CREATED:
        return "pull"  # Pull from remote to local

    if change.change_type == ChangeType.DELETED:
        return "push"  # Push deletion to remote

    if change.change_type == ChangeType.MODIFIED:
        # Use timestamps to determine direction
        if change.local_modified and change.remote_modified:
            if change.local_modified > change.remote_modified:
                return "push"
            else:
                return "pull"

    return "pull"  # Default to pull
