"""SQLite storage adapter for RAE-core with FTS5 full-text search."""

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import aiosqlite

from rae_core.interfaces.storage import IMemoryStorage


class SQLiteStorage(IMemoryStorage):
    """SQLite implementation of IMemoryStorage with FTS5 search."""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    layer TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    modified_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    version INTEGER DEFAULT 1,
                    expires_at TEXT,
                    project TEXT
                )
            """
            )
            # Support for embeddings table used in tests
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_embeddings (
                    memory_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (memory_id, model_name)
                )
            """
            )
            # FTS5 virtual table for lightning-fast keyword search
            await db.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, content='memories')"
            )
            
            # Triggers to keep FTS index synced with main table
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
            """)
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                END;
            """)
            await db.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content);
                    INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content);
                END;
            """)
            
            await db.commit()
        self._initialized = True

    async def store_memory(self, **kwargs: Any) -> UUID:
        await self.initialize()
        m_id = uuid4()
        now = datetime.now(timezone.utc).isoformat()

        tags = kwargs.get("tags") or []
        metadata = kwargs.get("metadata") or {}

        if "info_class" not in metadata:
            metadata["info_class"] = "internal"

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO memories (id, content, layer, tenant_id, agent_id, tags, metadata, importance, created_at, modified_at, last_accessed_at, project, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(m_id),
                    kwargs.get("content"),
                    kwargs.get("layer"),
                    kwargs.get("tenant_id"),
                    kwargs.get("agent_id"),
                    json.dumps(tags),
                    json.dumps(metadata),
                    kwargs.get("importance", 0.5),
                    now,
                    now,
                    now,
                    kwargs.get("project"),
                    (
                        kwargs.get("expires_at").isoformat()
                        if kwargs.get("expires_at")
                        else None
                    ),
                ),
            )
            await db.commit()
        return m_id

    async def get_memory(
        self, memory_id: UUID, tenant_id: str
    ) -> dict[str, Any] | None:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM memories WHERE id = ? AND tenant_id = ?",
                (str(memory_id), tenant_id),
            ) as cursor:
                row = await cursor.fetchone()
                return self._row_to_dict(row) if row else None

    async def get_memories_batch(
        self, memory_ids: list[UUID], tenant_id: str
    ) -> list[dict[str, Any]]:
        await self.initialize()
        ids = [str(mid) for mid in memory_ids]
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            placeholders = ",".join(["?"] * len(ids))
            async with db.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders}) AND tenant_id = ?",
                (*ids, tenant_id),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    async def update_memory(
        self, memory_id: UUID, tenant_id: str, updates: dict[str, Any]
    ) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT id FROM memories WHERE id = ? AND tenant_id = ?",
                (str(memory_id), tenant_id),
            ) as cursor:
                if not await cursor.fetchone():
                    return False

            if not updates:
                return False

            cols = []
            vals = []
            valid_fields = ["content", "importance", "layer", "tags", "metadata"]
            for k, v in updates.items():
                if k in valid_fields:
                    cols.append(f"{k} = ?")
                    vals.append(json.dumps(v) if k in ["tags", "metadata"] else v)

            if not cols:
                return False

            cols.append("version = version + 1")
            cols.append("modified_at = ?")
            vals.append(datetime.now(timezone.utc).isoformat())

            sql = (
                f"UPDATE memories SET {', '.join(cols)} WHERE id = ? AND tenant_id = ?"
            )
            vals.extend([str(memory_id), tenant_id])

            await db.execute(sql, vals)
            await db.commit()
            return True

    async def delete_memory(self, memory_id: UUID, tenant_id: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE id = ? AND tenant_id = ?",
                (str(memory_id), tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def list_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        limit = kwargs.get("limit", 100)
        tags_filter = kwargs.get("tags")
        filters = kwargs.get("filters", {})

        # Safe column list for ordering
        safe_cols = [
            "created_at",
            "modified_at",
            "importance",
            "access_count",
            "content",
        ]
        order_by = kwargs.get("order_by", "created_at")
        if order_by not in safe_cols:
            order_by = "created_at"

        direction = kwargs.get("order_direction", "desc")
        if direction.lower() not in ["asc", "desc"]:
            direction = "desc"

        where_clauses = ["tenant_id = ?"]
        params = [tenant_id]

        if agent_id:
            where_clauses.append("agent_id = ?")
            params.append(agent_id)
        if layer:
            where_clauses.append("layer = ?")
            params.append(layer)

        for k, v in filters.items():
            where_clauses.append(f"json_extract(metadata, '$.{k}') = ?")
            params.append(str(v))

        sql = f"SELECT * FROM memories WHERE {' AND '.join(where_clauses)} ORDER BY {order_by} {direction}"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                memories = [self._row_to_dict(r) for r in rows]

                if tags_filter:
                    filtered = []
                    for m in memories:
                        if any(tag in m["tags"] for tag in tags_filter):
                            filtered.append(m)
                    memories = filtered

                offset = kwargs.get("offset", 0)
                return memories[offset : offset + limit]

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        agent_id: str,
        layer: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        await self.initialize()
        where_clauses = ["tenant_id = ?", "agent_id = ?", "content LIKE ?"]
        params = [tenant_id, agent_id, f"%{query}%"]
        if layer:
            where_clauses.append("layer = ?")
            params.append(layer)

        params.append(limit)
        sql = f"SELECT * FROM memories WHERE {' AND '.join(where_clauses)} LIMIT ?"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "memory": self._row_to_dict(r),
                        "score": 1.0,
                        "id": r["id"],
                        "content": r["content"],
                    }
                    for r in rows
                ]

    async def search_full_text(
        self, query: str, tenant_id: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        await self.initialize()
        # Clean query for FTS5
        search_term = query.strip('"').replace("'", "")
        if not search_term:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Use FTS5 MATCH for high precision search
            # We join with the main memories table to get the tenant_id filtering and full metadata
            sql = """
                SELECT m.* 
                FROM memories m
                JOIN memories_fts f ON m.rowid = f.rowid
                WHERE m.tenant_id = ? AND memories_fts MATCH ?
                LIMIT ?
            """
            try:
                async with db.execute(sql, (tenant_id, search_term, limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_dict(r) for r in rows]
            except aiosqlite.OperationalError:
                # Fallback to LIKE if MATCH fails (e.g. invalid syntax or FTS table missing)
                sql_fallback = "SELECT * FROM memories WHERE tenant_id = ? AND content LIKE ? LIMIT ?"
                async with db.execute(sql_fallback, (tenant_id, f"%{search_term}%", limit)) as cursor:
                    rows = await cursor.fetchall()
                    return [self._row_to_dict(r) for r in rows]

    async def count_memories(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> int:
        await self.initialize()
        where_clauses = []
        params = []
        if tenant_id:
            where_clauses.append("tenant_id = ?")
            params.append(tenant_id)
        if agent_id:
            where_clauses.append("agent_id = ?")
            params.append(agent_id)
        if layer:
            where_clauses.append("layer = ?")
            params.append(layer)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT COUNT(*) FROM memories {where_sql}", params
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 0

    async def delete_memories_with_metadata_filter(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        await self.initialize()
        count = 0
        mems = await self.list_memories(
            tenant_id or "default",
            agent_id=agent_id,
            layer=layer,
            filters=metadata_filter or {},
        )
        for m in mems:
            if await self.delete_memory(m["id"], m["tenant_id"]):
                count += 1
        return count

    async def delete_memories_below_importance(
        self, tenant_id: str, agent_id: str, layer: str, importance_threshold: float
    ) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE tenant_id = ? AND agent_id = ? AND layer = ? AND importance < ?",
                (tenant_id, agent_id, layer, importance_threshold),
            )
            await db.commit()
            return cursor.rowcount

    async def delete_expired_memories(
        self, tenant_id: str, agent_id: str | None = None, layer: str | None = None
    ) -> int:
        await self.initialize()
        now = datetime.now(timezone.utc).isoformat()

        where = ["tenant_id = ?", "expires_at < ?"]
        params = [tenant_id, now]
        if agent_id:
            where.append("agent_id = ?")
            params.append(agent_id)
        if layer:
            where.append("layer = ?")
            params.append(layer)

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"DELETE FROM memories WHERE {' AND '.join(where)}", params
            )
            await db.commit()
            return cursor.rowcount

    async def update_memory_access(self, memory_id: UUID, tenant_id: str) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ? AND tenant_id = ?",
                (datetime.now(timezone.utc).isoformat(), str(memory_id), tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def increment_access_count(self, memory_id: UUID, tenant_id: str) -> bool:
        return await self.update_memory_access(memory_id, tenant_id)

    async def update_memory_expiration(
        self, memory_id: UUID, tenant_id: str, expires_at: datetime | None
    ) -> bool:
        await self.initialize()
        exp_str = expires_at.isoformat() if expires_at else None
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE memories SET expires_at = ? WHERE id = ? AND tenant_id = ?",
                (exp_str, str(memory_id), tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_metric_aggregate(
        self,
        tenant_id: str,
        metric: str,
        func: str,
        filters: dict[str, Any] | None = None,
    ) -> float:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                async with db.execute(
                    f"SELECT {func}({metric}) FROM memories WHERE tenant_id = ?",
                    (tenant_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                    return float(row[0] or 0.0)
            except Exception:
                return 0.0

    async def update_memory_access_batch(
        self, memory_ids: list[UUID], tenant_id: str
    ) -> bool:
        for mid in memory_ids:
            await self.update_memory_access(mid, tenant_id)
        return True

    async def adjust_importance(
        self, memory_id: UUID, delta: float, tenant_id: str
    ) -> float:
        await self.initialize()
        mem = await self.get_memory(memory_id, tenant_id)
        if not mem:
            return 0.0
        new_imp = max(0.0, min(1.0, mem["importance"] + delta))
        await self.update_memory(memory_id, tenant_id, {"importance": new_imp})
        return new_imp

    async def save_embedding(
        self,
        memory_id: UUID,
        model_name: str,
        embedding: list[float],
        tenant_id: str,
        **kwargs: Any,
    ) -> bool:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            # Check existence WITHOUT tenant for test expectation
            async with db.execute(
                "SELECT tenant_id FROM memories WHERE id = ?", (str(memory_id),)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return False
                if row[0] != tenant_id:
                    raise ValueError(
                        f"Access Denied: Memory {memory_id} not found for tenant {tenant_id}"
                    )

            await db.execute(
                "INSERT OR REPLACE INTO memory_embeddings (memory_id, model_name, embedding, created_at) VALUES (?, ?, ?, ?)",
                (
                    str(memory_id),
                    model_name,
                    json.dumps(embedding),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            await db.commit()
            return True

    async def decay_importance(self, tenant_id: str, decay_factor: float) -> int:
        return 0

    async def clear_tenant(self, tenant_id: str) -> int:
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM memories WHERE tenant_id = ?", (tenant_id,)
            )
            await db.commit()
            return cursor.rowcount

    async def close(self) -> None:
        pass

    def _row_to_dict(self, row: aiosqlite.Row) -> dict[str, Any]:
        d = dict(row)
        d["id"] = UUID(d["id"])
        d["tags"] = json.loads(d["tags"]) if d["tags"] else []
        d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
        d["usage_count"] = d["access_count"]
        return d