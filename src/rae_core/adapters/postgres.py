import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4
import re

import asyncpg

from ..interfaces.storage import IMemoryStorage


class PostgreSQLStorage(IMemoryStorage):
    def __init__(
        self,
        dsn: str | None = None,
        pool: asyncpg.Pool | None = None,
        **pool_kwargs: Any,
    ) -> None:
        self.dsn = dsn
        self._pool = pool
        self._pool_kwargs = pool_kwargs

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            if not self.dsn and not self._pool:
                raise ValueError("Either dsn or pool must be provided")
            self._pool = await asyncpg.create_pool(self.dsn, **self._pool_kwargs)
        return self._pool

    async def store_memory(self, **kwargs: Any) -> UUID:
        pool = await self._get_pool()
        m_id = uuid4()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO memories (id, content, layer, tenant_id, agent_id, tags, metadata, importance, created_at, project) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)",
                m_id,
                kwargs.get("content"),
                kwargs.get("layer"),
                kwargs.get("tenant_id"),
                kwargs.get("agent_id"),
                kwargs.get("tags", []),
                json.dumps(kwargs.get("metadata", {})),
                kwargs.get("importance", 0.5),
                datetime.now(timezone.utc).replace(tzinfo=None),
                kwargs.get("project"),
            )
        return m_id

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
        pool = await self._get_pool()
        audit_id = uuid4()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO reflection_audits (
                    id, query_id, tenant_id, agent_id, fsi_score, 
                    final_decision, l1_report, l2_report, l3_report, 
                    metadata, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
                audit_id,
                query_id,
                tenant_id,
                agent_id,
                fsi_score,
                final_decision,
                json.dumps(l1_report),
                json.dumps(l2_report),
                json.dumps(l3_report),
                json.dumps(metadata or {}),
                datetime.now(timezone.utc).replace(tzinfo=None),
            )
        return audit_id

    def _row_to_dict(self, row: asyncpg.Record | None) -> dict[str, Any] | None:
        if row is None: return None
        d = dict(row)
        meta = d.get("metadata")
        while isinstance(meta, str):
            try:
                parsed = json.loads(meta)
                if isinstance(parsed, (dict, list)): meta = parsed
                else: break
            except: break
        d["metadata"] = meta if isinstance(meta, dict) else {}
        return d

    def _get_stem(self, token: str) -> str:
        t = token.lower()
        if len(t) < 4: return t
        if t.endswith("ies"): return t[:-3]
        if t.endswith("s") and not t.endswith("ss"): return t[:-1]
        return t

    async def search_memories(
        self, query: str, tenant_id: str, agent_id: str, layer: str, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        
        ts_query = query.strip()
        if '"' not in ts_query:
            ts_query = re.sub(r'\s+', ' | ', ts_query)
            
        # System 40.16: SQL Linguistic Expansion
        # We search for both original words AND their stems in content/metadata
        words = [w for w in re.findall(r'\w+', query.lower()) if len(w) > 2]
        stems = [self._get_stem(w) for w in words]
        unique_patterns = list(set(words + stems))[:5] # Limit to avoid slow SQL
        
        project = kwargs.get("project")
        
        sql = f"""
        WITH candidates AS (
            SELECT *, 
                   ts_rank_cd(to_tsvector('english', coalesce(content, '')), websearch_to_tsquery('english', $1)) as ts_rank
            FROM memories
            WHERE tenant_id = $2 AND agent_id = $3 AND layer = $4
            {"AND project = $6" if project else ""}
            AND (
                to_tsvector('english', coalesce(content, '')) @@ websearch_to_tsquery('english', $1)
                OR content ILIKE $5
                { "".join([f"OR content ILIKE '%{p}%' OR metadata->>'id' ILIKE '%{p}%'" for p in unique_patterns]) }
            )
        )
        SELECT * FROM candidates ORDER BY ts_rank DESC LIMIT $7
        """
        
        async with pool.acquire() as conn:
            if project:
                rows = await conn.fetch(sql, ts_query, tenant_id, agent_id, layer, f"%{query}%", project, limit)
            else:
                rows = await conn.fetch(sql, ts_query, tenant_id, agent_id, layer, f"%{query}%", None, limit)
                
        return [self._row_to_dict(r) for r in rows if r]

    async def get_memory(self, memory_id: UUID, tenant_id: str) -> dict[str, Any] | None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM memories WHERE id = $1 AND tenant_id = $2", memory_id, tenant_id)
        return self._row_to_dict(row)

    async def list_memories(self, tenant_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        limit = kwargs.get("limit", 100)
        async with pool.acquire() as conn:
            rows = await conn.fetch("SELECT * FROM memories WHERE tenant_id = $1 LIMIT $2", tenant_id, limit)
        return [self._row_to_dict(r) for r in rows if r]

    async def close(self) -> None:
        if self._pool: await self._pool.close()

    async def delete_memories_with_metadata_filter(self, tenant_id=None, agent_id=None, layer=None, metadata_filter=None) -> int: return 0
    async def count_memories(self, tenant_id=None, agent_id=None, layer=None) -> int: return 0
    async def update_memory_access(self, memory_id, tenant_id) -> bool: return True
    async def delete_expired_memories(self, tenant_id, agent_id=None, layer=None) -> int: return 0
    async def update_memory(self, memory_id, tenant_id, updates) -> bool: return True
    async def delete_memory(self, memory_id, tenant_id) -> bool: return True
    async def get_metric_aggregate(self, tenant_id, metric, func, filters=None) -> float: return 0.0
    async def update_memory_access_batch(self, memory_ids, tenant_id) -> bool: return True
    async def adjust_importance(self, memory_id, delta, tenant_id) -> float: return 0.5
    async def delete_memories_below_importance(self, tenant_id, agent_id, layer, importance_threshold) -> int: return 0
    async def decay_importance(self, tenant_id, decay_factor) -> int: return 0
    async def save_embedding(self, memory_id, model_name, embedding, tenant_id) -> bool: return True
    async def update_memory_expiration(self, memory_id, tenant_id, expires_at) -> bool: return True
