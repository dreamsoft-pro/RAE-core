"""SQLite vector store adapter using sqlite-vec extension.

Lightweight vector storage for RAE-Lite offline-first architecture.
Uses sqlite-vec for efficient vector similarity search.
"""

import json
import struct
from typing import Any
from uuid import UUID

import aiosqlite
import numpy as np

from rae_core.interfaces.vector import IVectorStore


class SQLiteVectorStore(IVectorStore):
    """SQLite implementation of IVectorStore using sqlite-vec.

    Features:
    - File-based vector storage
    - Cosine similarity search (optimized with numpy)
    - Metadata storage alongside vectors
    - Layer filtering support
    - Batch operations
    - ACID transactions

    Note: Requires sqlite-vec extension. Falls back to numpy
    cosine similarity if extension is not available.
    """

    def __init__(self, db_path: str = ":memory:"):
        """Initialize SQLite vector store.

        Args:
            db_path: Path to SQLite database file, or ":memory:" for in-memory DB
        """
        self.db_path = db_path
        self._initialized = False
        self._has_vec_extension = False

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL")

            # Try to load sqlite-vec extension
            try:
                await db.enable_load_extension(True)
                await db.load_extension("vec0")
                self._has_vec_extension = True  # pragma: no cover
            except Exception:
                # Extension not available, will use manual similarity
                self._has_vec_extension = False

            # Vectors table
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS vectors (
                    memory_id TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    dimension INTEGER NOT NULL,
                    tenant_id TEXT NOT NULL,
                    metadata TEXT  -- JSON object
                )
            """
            )

            # Indexes
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_vectors_tenant_id
                ON vectors(tenant_id)
            """
            )

            await db.commit()

        self._initialized = True

    async def store_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a vector embedding."""
        await self.initialize()

        # Handle multi-vector
        if isinstance(embedding, dict):
            vec_list = (
                embedding.get("default")
                or embedding.get("dense")
                or next(iter(embedding.values()))
            )
        else:
            vec_list = embedding

        # Serialize embedding as BLOB (float32 array)
        embedding_bytes = struct.pack(f"{len(vec_list)}f", *vec_list)
        metadata_json = json.dumps(metadata or {})

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                INSERT OR REPLACE INTO vectors (memory_id, embedding, dimension, tenant_id, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(memory_id),
                    embedding_bytes,
                    len(vec_list),
                    tenant_id,
                    metadata_json,
                ),
            )
            await db.commit()

        return True

    async def search_similar(
        self,
        query_embedding: list[float],
        tenant_id: str,
        layer: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        """Search for similar vectors using cosine similarity."""
        await self.initialize()

        if self._has_vec_extension:  # pragma: no cover
            # TODO: Implement sqlite-vec specific search when extension is present
            pass  # pragma: no cover

        # Fallback to optimized numpy search
        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return []

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Build WHERE clause
            where_clauses = ["tenant_id = ?"]
            params = [tenant_id]

            if layer:
                where_clauses.append("json_extract(metadata, '$.layer') = ?")
                params.append(layer)

            if agent_id:
                where_clauses.append("json_extract(metadata, '$.agent_id') = ?")
                params.append(agent_id)

            where_clause = " AND ".join(where_clauses)

            # Fetch all vectors for this tenant/layer
            async with db.execute(
                f"""
                SELECT memory_id, embedding, dimension
                FROM vectors
                WHERE {where_clause}
                """,
                params,
            ) as cursor:
                rows = await cursor.fetchall()

                if not rows:
                    return []

                # Extract and deserialize all embeddings at once
                ids = [row["memory_id"] for row in rows]
                embeddings = []
                for row in rows:
                    embeddings.append(np.frombuffer(row["embedding"], dtype=np.float32))

                # Convert to matrix for bulk calculation
                matrix = np.stack(embeddings)

                # Bulk cosine similarity calculation
                dot_products = np.dot(matrix, query_vec)
                norms = np.linalg.norm(matrix, axis=1)
                similarities = dot_products / (query_norm * norms)

                # Filter and format results
                results = []
                for i, similarity in enumerate(similarities):
                    if score_threshold is not None and similarity < score_threshold:
                        continue
                    results.append((UUID(ids[i]), float(similarity)))

                # Sort by similarity (descending) and limit
                results.sort(key=lambda x: x[1], reverse=True)
                return results[:limit]

    async def delete_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a vector."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                DELETE FROM vectors
                WHERE memory_id = ? AND tenant_id = ?
                """,
                (str(memory_id), tenant_id),
            )
            await db.commit()

            return cursor.rowcount > 0

    async def update_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector embedding."""
        await self.initialize()

        # Check if vector exists
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT 1 FROM vectors
                WHERE memory_id = ? AND tenant_id = ?
                """,
                (str(memory_id), tenant_id),
            ) as cursor:
                exists = await cursor.fetchone()

                if not exists:
                    return False

        # Update using store_vector (INSERT OR REPLACE)
        return await self.store_vector(memory_id, embedding, tenant_id, metadata)

    async def get_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> list[float] | None:
        """Retrieve a vector embedding."""
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT embedding, dimension FROM vectors
                WHERE memory_id = ? AND tenant_id = ?
                """,
                (str(memory_id), tenant_id),
            ) as cursor:
                row = await cursor.fetchone()

                if not row:
                    return None

                embedding_bytes = row[0]
                dimension = row[1]

                # Deserialize embedding
                embedding = struct.unpack(f"{dimension}f", embedding_bytes)
                return list(embedding)

    async def batch_store_vectors(
        self,
        vectors: list[
            tuple[UUID, list[float] | dict[str, list[float]], dict[str, Any]]
        ],
        tenant_id: str,
    ) -> int:
        """Store multiple vectors in a batch."""
        await self.initialize()

        count = 0

        async with aiosqlite.connect(self.db_path) as db:
            for memory_id, embedding, metadata in vectors:
                try:
                    # Handle multi-vector
                    if isinstance(embedding, dict):
                        vec_list = (
                            embedding.get("default")
                            or embedding.get("dense")
                            or next(iter(embedding.values()))
                        )
                    else:
                        vec_list = embedding

                    embedding_bytes = struct.pack(f"{len(vec_list)}f", *vec_list)
                    metadata_json = json.dumps(metadata)

                    await db.execute(
                        """
                        INSERT OR REPLACE INTO vectors (memory_id, embedding, dimension, tenant_id, metadata)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            str(memory_id),
                            embedding_bytes,
                            len(vec_list),
                            tenant_id,
                            metadata_json,
                        ),
                    )
                    count += 1
                except Exception:
                    # Skip invalid vectors
                    continue

            await db.commit()

        return count

    async def clear_tenant(self, tenant_id: str) -> int:
        """Clear all vectors for a tenant.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Number of vectors deleted
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """
                SELECT COUNT(*) FROM vectors
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            ) as cursor:
                row = await cursor.fetchone()
                count = row[0] if row else 0

            await db.execute(
                """
                DELETE FROM vectors
                WHERE tenant_id = ?
                """,
                (tenant_id,),
            )
            await db.commit()

            return count

    async def get_statistics(self) -> dict[str, Any]:
        """Get vector store statistics.

        Returns:
            Dictionary with statistics
        """
        await self.initialize()

        async with aiosqlite.connect(self.db_path) as db:
            # Total vectors
            async with db.execute("SELECT COUNT(*) FROM vectors") as cursor:
                row = await cursor.fetchone()
                total_vectors = row[0] if row else 0

            # Unique tenants
            async with db.execute(
                "SELECT COUNT(DISTINCT tenant_id) FROM vectors"
            ) as cursor:
                row = await cursor.fetchone()
                tenants = row[0] if row else 0

            # Dimensions distribution
            async with db.execute(
                "SELECT DISTINCT dimension FROM vectors ORDER BY dimension"
            ) as cursor:
                dimensions = [row[0] for row in await cursor.fetchall()]

            return {
                "total_vectors": total_vectors,
                "tenants": tenants,
                "dimensions": dimensions,
                "has_vec_extension": self._has_vec_extension,
            }

    async def close(self) -> None:
        """Close database connection."""
        # aiosqlite uses context managers, so explicit close not needed
        pass
