"""Sync Manager for RAE-Sync Protocol.

Handles peer discovery, handshakes, and orchestration of sync operations.
"""

import logging
from datetime import datetime
from uuid import uuid4

from rae_core.interfaces.sync import ISyncProvider
from rae_core.models.sync import SyncHandshake, SyncLog, SyncPeer
from rae_core.sync.protocol import SyncProtocol

logger = logging.getLogger(__name__)


class SyncManager:
    """Manages synchronization lifecycle and peers."""

    def __init__(self, agent_id: str, role: str, sync_provider: ISyncProvider):
        self.agent_id = agent_id
        self.role = role
        self.peers: dict[str, SyncPeer] = {}
        self.provider = sync_provider
        self.protocol = SyncProtocol(sync_provider)

    async def initiate_handshake(self, target_peer_id: str) -> SyncHandshake:
        """Create a handshake payload to send to a peer."""
        return SyncHandshake(
            peer_id=self.agent_id,
            role=self.role,
            protocol_version="1.0",
            capabilities=["push", "pull", "encryption"],
        )

    async def receive_handshake(self, handshake: SyncHandshake) -> bool:
        """Process an incoming handshake."""
        logger.info(f"Received handshake from {handshake.peer_id} ({handshake.role})")

        # Verify protocol compatibility
        if handshake.protocol_version != "1.0":
            logger.warning(
                f"Incompatible protocol version: {handshake.protocol_version}"
            )
            return False

        # Register peer
        self.peers[handshake.peer_id] = SyncPeer(
            peer_id=handshake.peer_id,
            role=handshake.role,
            protocol_version=handshake.protocol_version,
            last_seen=datetime.now(),
        )
        return True

    async def sync_with_peer(self, peer_id: str, tenant_id: str) -> SyncLog:
        """Perform a full sync with a connected peer."""
        if peer_id not in self.peers:
            raise ValueError(f"Peer {peer_id} not connected")

        start_time = datetime.now()

        # logic for bidirectional sync using protocol
        # For Phase 3 MVP, we assume a 'pull' then 'push' strategy or just delegate to provider

        try:
            # 1. Pull changes
            pull_result = await self.protocol.pull(tenant_id, self.agent_id)

            # 2. Push changes
            push_result = await self.protocol.push(tenant_id, self.agent_id)

            duration = (datetime.now() - start_time).total_seconds() * 1000

            # Create Log
            return SyncLog(
                id=uuid4(),
                peer_id=peer_id,
                direction="bidirectional",
                status="success",
                items_synced=(
                    len(pull_result.synced_memory_ids)
                    + len(push_result.synced_memory_ids)
                ),
                conflicts_resolved=(
                    len(pull_result.conflicts) + len(push_result.conflicts)
                ),
                duration_ms=duration,
                metadata={
                    "pull_count": len(pull_result.synced_memory_ids),
                    "push_count": len(push_result.synced_memory_ids),
                },
            )

        except Exception as e:
            logger.error(f"Sync failed with {peer_id}: {e}")
            return SyncLog(
                id=uuid4(),
                peer_id=peer_id,
                direction="bidirectional",
                status="failed",
                duration_ms=(datetime.now() - start_time).total_seconds() * 1000,
                metadata={"error": str(e)},
            )
