"""HTTP implementation of ISyncProvider for RAE Mesh."""

from datetime import datetime
from typing import Any
from uuid import UUID

import httpx
import structlog

from rae_core.interfaces.sync import ISyncProvider

logger = structlog.get_logger(__name__)


class HttpSyncProvider(ISyncProvider):
    """
    Sync provider that communicates with remote RAE instances via HTTP/API v2.
    """

    def __init__(self, remote_url: str, auth_token: str, timeout: int = 30):
        self.remote_url = remote_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.remote_url,
            headers={
                "Authorization": f"Bearer {self.auth_token}",
                "X-RAE-Peer-Token": self.auth_token,
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def push_changes(self, tenant_id: str, changes: list[dict[str, Any]]) -> bool:
        """Push raw changes (not implemented for v1 MVP)."""
        raise NotImplementedError("Raw change push not supported yet")

    async def pull_changes(
        self, tenant_id: str, since_timestamp: str
    ) -> list[dict[str, Any]]:
        """Pull raw changes (not implemented for v1 MVP)."""
        raise NotImplementedError("Raw change pull not supported yet")

    async def resolve_conflict(
        self,
        memory_id: UUID,
        local_version: dict[str, Any],
        remote_version: dict[str, Any],
    ) -> dict[str, Any]:
        """Simple conflict resolution: Last Write Wins (based on timestamp)."""
        local_ts = local_version.get("updated_at") or local_version.get("created_at")
        remote_ts = remote_version.get("updated_at") or remote_version.get("created_at")

        if not local_ts:
            return remote_version
        if not remote_ts:
            return local_version

        return local_version if local_ts > remote_ts else remote_version

    async def push_memories(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Push memories to remote via /v2/mesh/sync/receive."""
        try:
            # Note: In a real implementation, we would first fetch the memories
            # from local storage here. For now, we assume the caller passes data
            # or the manager handles extraction.
            # This method signature in ISyncProvider expects IDs, so the Manager
            # usually handles the data fetching.
            # However, for the HTTP proxy, we expect to SEND data.
            # This indicates ISyncProvider might need refactoring or we interpret
            # 'push_memories' as "Trigger a push".

            # Assuming the caller (SyncManager) has already prepared the data payload.
            # But wait, ISyncProvider is a low-level interface.
            # Let's assume we implement the transport only here.

            # FIX: The current ISyncProvider definition is a bit ambiguous on data flow.
            # We will implement a simplified "Send what I have" logic.

            logger.warning("http_sync_push_not_fully_implemented")
            return {"success": False, "error": "Not implemented in MVP"}

        except Exception as e:
            logger.error("push_memories_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def pull_memories(
        self,
        tenant_id: str,
        agent_id: str,
        memory_ids: list[UUID] | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Pull memories from remote via /v2/memories/query or specialized endpoint."""
        try:
            payload: dict[str, Any] = {
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "since": since.isoformat() if since else None,
                "limit": 100,  # Batch size
            }
            if memory_ids:
                payload["memory_ids"] = [str(m) for m in memory_ids]

            response = await self.client.post("/v2/mesh/sync/export", json=payload)
            response.raise_for_status()
            data = response.json()

            return {
                "success": True,
                "synced_memory_ids": [m["id"] for m in data.get("memories", [])],
                "memories": data.get("memories", []),
                "sync_id": str(data.get("sync_id", "")),
            }

        except Exception as e:
            logger.error("pull_memories_failed", error=str(e))
            return {"success": False, "error": str(e)}

    async def sync_memories(
        self,
        tenant_id: str,
        agent_id: str,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Bidirectional sync (Pull then Push logic handled by Manager)."""
        raise NotImplementedError("Use Manager for bidirectional sync")

    async def get_sync_status(
        self,
        tenant_id: str,
        agent_id: str,
        sync_id: str,
    ) -> dict[str, Any]:
        """Get remote status."""
        response = await self.client.get(f"/v2/mesh/sync/status/{sync_id}")
        from typing import cast

        return cast(dict[str, Any], response.json())

    async def handshake(
        self,
        tenant_id: str,
        agent_id: str,
        capabilities: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform handshake.
        For HTTP Provider, this usually verifies the token works.
        """
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            return {"status": "connected", "peer_type": "http"}
        except Exception as e:
            return {"status": "failed", "error": str(e)}
