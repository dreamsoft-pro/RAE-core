"""CRDT-based memory merge and conflict resolution."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ConflictResolutionStrategy(str, Enum):
    """Strategy for resolving conflicts."""

    LAST_WRITE_WINS = "last_write_wins"
    MERGE_FIELDS = "merge_fields"
    KEEP_LOCAL = "keep_local"
    KEEP_REMOTE = "keep_remote"
    MANUAL = "manual"


class MergeResult(BaseModel):
    """Result of memory merge operation."""

    memory_id: UUID | None = None
    success: bool
    strategy_used: ConflictResolutionStrategy
    merged_memory: dict[str, Any] | None = None
    conflicts_resolved: list[str] = []
    error_message: str | None = None


class ConflictResolver:
    """Resolves conflicts between memory versions using CRDT principles."""

    def __init__(
        self,
        default_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS,
    ):
        """Initialize conflict resolver.

        Args:
            default_strategy: Default strategy for conflict resolution
        """
        self.default_strategy = default_strategy

    def resolve(
        self,
        local: dict[str, Any],
        remote: dict[str, Any],
        conflict_fields: list[str],
        strategy: ConflictResolutionStrategy | None = None,
    ) -> MergeResult:
        """Resolve conflicts between local and remote versions.

        Args:
            local: Local memory version
            remote: Remote memory version
            conflict_fields: List of fields with conflicts
            strategy: Resolution strategy (uses default if not specified)

        Returns:
            MergeResult with resolved memory
        """
        strategy = strategy or self.default_strategy
        memory_id: UUID | None = None

        try:
            memory_id = UUID(local["id"])
            if strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
                merged = self._last_write_wins(local, remote, conflict_fields)
            elif strategy == ConflictResolutionStrategy.MERGE_FIELDS:
                merged = self._merge_fields(local, remote, conflict_fields)
            elif strategy == ConflictResolutionStrategy.KEEP_LOCAL:
                merged = local.copy()
            elif strategy == ConflictResolutionStrategy.KEEP_REMOTE:
                merged = remote.copy()
            else:
                return MergeResult(
                    memory_id=memory_id,
                    success=False,
                    strategy_used=strategy,
                    error_message="Manual resolution required",
                )

            return MergeResult(
                memory_id=memory_id,
                success=True,
                strategy_used=strategy,
                merged_memory=merged,
                conflicts_resolved=conflict_fields,
            )

        except Exception as e:
            return MergeResult(
                memory_id=memory_id,
                success=False,
                strategy_used=strategy,
                error_message=str(e),
            )

    def _last_write_wins(
        self,
        local: dict[str, Any],
        remote: dict[str, Any],
        conflict_fields: list[str],
    ) -> dict[str, Any]:
        """Resolve conflicts using Last-Write-Wins strategy.

        Args:
            local: Local memory version
            remote: Remote memory version
            conflict_fields: List of fields with conflicts

        Returns:
            Merged memory
        """
        local_modified = local.get("modified_at")
        remote_modified = remote.get("modified_at")

        # Use version if timestamps are equal
        if local_modified == remote_modified:
            local_version = local.get("version", 0)
            remote_version = remote.get("version", 0)
            winner = local if local_version >= remote_version else remote
        else:
            winner = (
                local
                if (local_modified or datetime.min) > (remote_modified or datetime.min)
                else remote
            )

        return winner.copy()

    def _merge_fields(
        self,
        local: dict[str, Any],
        remote: dict[str, Any],
        conflict_fields: list[str],
    ) -> dict[str, Any]:
        """Resolve conflicts by merging fields intelligently.

        Args:
            local: Local memory version
            remote: Remote memory version
            conflict_fields: List of fields with conflicts

        Returns:
            Merged memory
        """
        # Start with newer base version
        local_modified = local.get("modified_at")
        remote_modified = remote.get("modified_at")

        if (local_modified or datetime.min) > (remote_modified or datetime.min):
            merged = local.copy()
        else:
            merged = remote.copy()

        # Merge specific fields
        for field in conflict_fields:
            if field == "tags":
                # Merge tags (union)
                local_tags = set(local.get("tags", []))
                remote_tags = set(remote.get("tags", []))
                merged["tags"] = list(local_tags | remote_tags)

            elif field == "metadata":
                # Merge metadata (union, prefer newer values)
                local_meta = local.get("metadata", {})
                remote_meta = remote.get("metadata", {})
                merged_meta = {**local_meta, **remote_meta}
                merged["metadata"] = merged_meta

            elif field == "importance":
                # Use maximum importance
                local_imp = local.get("importance", 0.0)
                remote_imp = remote.get("importance", 0.0)
                merged["importance"] = max(local_imp, remote_imp)

            elif field == "content":
                # Keep content from newer version (already set in merged)
                pass

        # Update version and timestamp
        merged["version"] = (
            max(
                local.get("version", 1),
                remote.get("version", 1),
            )
            + 1
        )
        merged["modified_at"] = datetime.now(timezone.utc)

        return merged


def merge_memories(
    local_memories: list[dict[str, Any]],
    remote_memories: list[dict[str, Any]],
    resolver: ConflictResolver | None = None,
) -> list[MergeResult]:
    """Merge local and remote memories using CRDT principles.

    Args:
        local_memories: List of local memory records
        remote_memories: List of remote memory records
        resolver: Optional conflict resolver (creates default if not provided)

    Returns:
        List of merge results
    """
    if resolver is None:
        resolver = ConflictResolver()

    results = []

    # Create lookup dictionaries
    local_dict = {str(mem["id"]): mem for mem in local_memories}
    remote_dict = {str(mem["id"]): mem for mem in remote_memories}

    # Process all unique memory IDs
    all_ids = set(local_dict.keys()) | set(remote_dict.keys())

    for mem_id in all_ids:
        local_mem = local_dict.get(mem_id)
        remote_mem = remote_dict.get(mem_id)

        if local_mem and remote_mem:
            # Both exist - check for conflicts
            from rae_core.sync.diff import _detect_conflicts

            conflicts, conflict_fields = _detect_conflicts(
                local_mem,
                remote_mem,
                local_mem.get("modified_at"),
                remote_mem.get("modified_at"),
            )

            if conflicts:
                # Resolve conflicts
                result = resolver.resolve(
                    local_mem,
                    remote_mem,
                    conflict_fields,
                )
                results.append(result)
            else:
                # No conflicts - use newer version
                result = resolver.resolve(
                    local_mem,
                    remote_mem,
                    [],
                    ConflictResolutionStrategy.LAST_WRITE_WINS,
                )
                results.append(result)

        elif local_mem:
            # Only local exists - keep it
            results.append(
                MergeResult(
                    memory_id=UUID(mem_id),
                    success=True,
                    strategy_used=ConflictResolutionStrategy.KEEP_LOCAL,
                    merged_memory=local_mem,
                )
            )

        elif remote_mem:
            # Only remote exists - keep it
            results.append(
                MergeResult(
                    memory_id=UUID(mem_id),
                    success=True,
                    strategy_used=ConflictResolutionStrategy.KEEP_REMOTE,
                    merged_memory=remote_mem,
                )
            )

    return results


async def apply_merge_results(
    merge_results: list[MergeResult],
    storage_update_fn: Any,
) -> dict[str, Any]:
    """Apply merge results to storage.

    Args:
        merge_results: List of merge results
        storage_update_fn: Function to update storage (async callable)

    Returns:
        Summary of applied changes
    """
    summary: dict[str, Any] = {
        "total": len(merge_results),
        "successful": 0,
        "failed": 0,
        "errors": [],
    }

    for result in merge_results:
        if result.success and result.merged_memory:
            try:
                await storage_update_fn(result.merged_memory)
                summary["successful"] += 1
            except Exception as e:
                summary["failed"] += 1
                summary["errors"].append(
                    {"memory_id": str(result.memory_id), "error": str(e)}
                )
        else:
            summary["failed"] += 1
            if result.error_message:
                summary["errors"].append(
                    {"memory_id": str(result.memory_id), "error": result.error_message}
                )

    return summary
