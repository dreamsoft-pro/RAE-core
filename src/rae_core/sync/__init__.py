"""Sync protocol module for RAE-core."""

from rae_core.sync.diff import (
    ChangeType,
    DiffResult,
    MemoryChange,
    calculate_memory_diff,
    get_sync_direction,
)
from rae_core.sync.encryption import E2EEncryption, decrypt_batch, encrypt_batch
from rae_core.sync.merge import (
    ConflictResolutionStrategy,
    ConflictResolver,
    MergeResult,
    apply_merge_results,
    merge_memories,
)
from rae_core.sync.protocol import SyncMetadata, SyncProtocol, SyncRequest, SyncResponse

__all__ = [
    # Protocol
    "SyncProtocol",
    "SyncRequest",
    "SyncResponse",
    "SyncMetadata",
    # Diff
    "calculate_memory_diff",
    "DiffResult",
    "MemoryChange",
    "ChangeType",
    "get_sync_direction",
    # Merge
    "merge_memories",
    "ConflictResolver",
    "ConflictResolutionStrategy",
    "MergeResult",
    "apply_merge_results",
    # Encryption
    "E2EEncryption",
    "encrypt_batch",
    "decrypt_batch",
]
