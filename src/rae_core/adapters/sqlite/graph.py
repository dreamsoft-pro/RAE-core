"""SQLite graph store adapter for RAE-core.

Lightweight graph storage ideal for RAE-Lite offline-first architecture.
"""

import json
from typing import Any
from uuid import UUID

import aiosqlite

from rae_core.interfaces.graph import IGraphStore


class SQLiteGraphStore(IGraphStore):
    """SQLite implementation of IGraphStore for knowledge graph operations."""

    def __init__(self, db_path: str = ":memory:"):
        """Initialize SQLite graph store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize database schema for graph operations."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")

            # Nodes table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    properties TEXT,  -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Edges table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    weight REAL DEFAULT 1.0,
                    tenant_id TEXT NOT NULL,
                    properties TEXT,  -- JSON object
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, type, tenant_id),
                    FOREIGN KEY (source_id) REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES knowledge_graph_nodes(id) ON DELETE CASCADE
                )
            """
            )

            # Indexes
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_tenant ON knowledge_graph_nodes(tenant_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_tenant ON knowledge_graph_edges(tenant_id)"
            )

            await db.commit()

        self._initialized = True

    async def create_node(
        self,
        node_id: UUID,
        node_type: str,
        tenant_id: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a graph node."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO knowledge_graph_nodes (id, type, tenant_id, properties)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT (id) DO UPDATE SET properties = excluded.properties
                    """,
                    (str(node_id), node_type, tenant_id, json.dumps(properties or {})),
                )
                await db.commit()
                return True
            except Exception:  # pragma: no cover
                return False  # pragma: no cover

    async def create_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        tenant_id: str,
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        """Create a graph edge."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute(
                    """
                    INSERT INTO knowledge_graph_edges
                    (source_id, target_id, type, weight, tenant_id, properties)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT (source_id, target_id, type, tenant_id) DO UPDATE
                    SET weight = excluded.weight, properties = excluded.properties
                    """,
                    (
                        str(source_id),
                        str(target_id),
                        edge_type,
                        weight,
                        tenant_id,
                        json.dumps(properties or {}),
                    ),
                )
                await db.commit()
                return True
            except Exception:  # pragma: no cover
                return False  # pragma: no cover

    async def get_neighbors(
        self,
        node_id: UUID,
        tenant_id: str,
        edge_type: str | None = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[UUID]:
        """Get neighboring nodes using BFS-like traversal."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            query_parts = []
            params = []

            if direction in ["out", "both"]:
                sql = "SELECT target_id FROM knowledge_graph_edges WHERE source_id = ? AND tenant_id = ?"
                if edge_type:
                    sql += " AND type = ?"
                    params.extend([str(node_id), tenant_id, edge_type])
                else:
                    params.extend([str(node_id), tenant_id])
                query_parts.append(sql)

            if direction in ["in", "both"]:
                sql = "SELECT source_id FROM knowledge_graph_edges WHERE target_id = ? AND tenant_id = ?"
                if edge_type:
                    sql += " AND type = ?"
                    params.extend([str(node_id), tenant_id, edge_type])
                else:
                    params.extend([str(node_id), tenant_id])
                query_parts.append(sql)

            full_query = " UNION ".join(query_parts)

            async with db.execute(full_query, params) as cursor:
                rows = await cursor.fetchall()
                return [UUID(row[0]) for row in rows]

    async def delete_node(self, node_id: UUID, tenant_id: str) -> bool:
        """Delete a node and its edges."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM knowledge_graph_nodes WHERE id = ? AND tenant_id = ?",
                (str(node_id), tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        edge_type: str,
        tenant_id: str,
    ) -> bool:
        """Delete an edge."""
        await self.initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM knowledge_graph_edges WHERE source_id = ? AND target_id = ? AND type = ? AND tenant_id = ?",
                (str(source_id), str(target_id), edge_type, tenant_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def shortest_path(
        self,
        source_id: UUID,
        target_id: UUID,
        tenant_id: str,
        max_depth: int = 5,
    ) -> list[UUID] | None:
        """Basic BFS shortest path implementation."""
        # Simplified for Lite version
        queue = [(source_id, [source_id])]
        visited = {source_id}

        while queue:
            current_node, path = queue.pop(0)
            if current_node == target_id:
                return path

            if len(path) <= max_depth:
                neighbors = await self.get_neighbors(current_node, tenant_id)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        return None

    async def get_subgraph(
        self, node_ids: list[UUID], tenant_id: str, include_edges: bool = True
    ) -> dict[str, Any]:
        """Extract a subgraph."""
        await self.initialize()
        result: dict[str, Any] = {"nodes": [], "edges": []}

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            placeholders = ",".join(["?" for _ in node_ids])
            params = [str(nid) for nid in node_ids] + [tenant_id]

            async with db.execute(
                f"SELECT * FROM knowledge_graph_nodes WHERE id IN ({placeholders}) AND tenant_id = ?",
                params,
            ) as cursor:
                result["nodes"] = [dict(row) for row in await cursor.fetchall()]

            if include_edges:
                # Edges between any of the specified nodes
                params = (
                    [str(nid) for nid in node_ids]
                    + [str(nid) for nid in node_ids]
                    + [tenant_id]
                )
                async with db.execute(
                    f"SELECT * FROM knowledge_graph_edges WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders}) AND tenant_id = ?",
                    params,
                ) as cursor:
                    result["edges"] = [dict(row) for row in await cursor.fetchall()]

        return result
