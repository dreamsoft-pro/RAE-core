"""Qdrant vector store adapter."""

import math
from typing import Any, cast
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from rae_core.interfaces.vector import IVectorStore

logger = structlog.get_logger(__name__)


class QdrantVectorStore(IVectorStore):
    """Qdrant implementation of the Vector Store interface."""

    def __init__(
        self,
        client: AsyncQdrantClient | None = None,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "memories",
        embedding_dim: int = 384,
        vector_name: str = "dense",
    ):
        """Initialize Qdrant Vector Store.

        Args:
            client: Existing Qdrant client instance.
            url: Qdrant URL (if client is not provided).
            api_key: Qdrant API Key (if client is not provided).
            collection_name: Name of the collection.
            embedding_dim: Dimension of embeddings (default 384).
            vector_name: Name of the named vector (default "dense").
        """
        if client:
            self.client = client
        else:
            if not url:
                url = "http://localhost:6333"  # Default fallback
            self.client = AsyncQdrantClient(url=url, api_key=api_key)

        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.vector_name = vector_name
        self._initialized = False
        self._known_vectors: set[str] = set()

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists and is compatible with current config."""
        if self._initialized:
            return

        try:
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                # Case 1: Collection does not exist. Create it.
                await self._create_collection()
            else:
                # Case 2: Collection exists. Validate schema compliance.
                collection_info = await self.client.get_collection(self.collection_name)
                vectors_config = collection_info.config.params.vectors

                is_valid = True

                # Check compatibility of the primary vector
                if isinstance(vectors_config, dict):
                    if self.vector_name in vectors_config:
                        existing_dim = vectors_config[self.vector_name].size
                        if existing_dim != self.embedding_dim:
                            logger.warning(
                                "qdrant_schema_mismatch",
                                collection=self.collection_name,
                                vector=self.vector_name,
                                expected=self.embedding_dim,
                                actual=existing_dim,
                                action="recreating_collection",
                            )
                            is_valid = False
                    # If vector doesn't exist, we can add it later via ensure_vector_config, so technically valid structure
                else:
                    # Single unnamed vector
                    # Check if we expect a named vector but got unnamed
                    if self.vector_name != "":
                        # This is complex. For simplicity, if we have unnamed and want named (or vice versa), recreate.
                        # Or check if size matches.
                        pass

                if not is_valid:
                    await self.client.delete_collection(self.collection_name)
                    await self._create_collection()
                else:
                    # Load known vectors
                    if isinstance(vectors_config, dict):
                        self._known_vectors.update(vectors_config.keys())

            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to ensure Qdrant collection: {e}")
            # Do not set initialized to True if failed, so we retry

    async def _create_collection(self) -> None:
        """Helper to create collection with current settings."""
        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                self.vector_name: VectorParams(
                    size=self.embedding_dim, distance=Distance.COSINE
                )
            },
        )
        logger.info(f"Created/Recreated Qdrant collection: {self.collection_name}")
        self._known_vectors.add(self.vector_name)

    async def ensure_vector_config(self, vector_name: str, dim: int) -> None:
        """Dynamically add a new named vector config if missing."""
        await self._ensure_collection()

        if vector_name in self._known_vectors:
            return

        try:
            # check again against API to be safe (race conditions)
            collection_info = await self.client.get_collection(self.collection_name)
            existing_vectors = {}
            if isinstance(collection_info.config.params.vectors, dict):
                existing_vectors = collection_info.config.params.vectors

            if vector_name not in existing_vectors:
                logger.info(f"Adding new vector config: {vector_name} ({dim}d)")
                await self.client.update_collection(
                    collection_name=self.collection_name,
                    vectors_config={
                        vector_name: VectorParams(size=dim, distance=Distance.COSINE)
                    },
                )

            self._known_vectors.add(vector_name)
        except Exception as e:
            logger.error(f"Failed to update vector config for {vector_name}: {e}")

    def _build_filter(
        self,
        tenant_id: str,
        layer: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        project: str | None = None,
        extra_filters: dict[str, Any] | None = None,
    ) -> Filter:
        """Build Qdrant Filter from RAE metadata."""
        must_conditions = [
            FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))
        ]

        if layer:
            must_conditions.append(
                FieldCondition(key="layer", match=MatchValue(value=layer))
            )
        if agent_id:
            must_conditions.append(
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id))
            )
        if session_id:
            must_conditions.append(
                FieldCondition(key="session_id", match=MatchValue(value=session_id))
            )
        if project:
            must_conditions.append(
                FieldCondition(key="project", match=MatchValue(value=project))
            )

        if extra_filters:
            for key, value in extra_filters.items():
                # Basic support for exact match
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )

        return Filter(must=must_conditions)

    async def store_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a single vector."""
        return (
            await self.batch_store_vectors(
                [(memory_id, embedding, metadata or {})], tenant_id
            )
            == 1
        )

    async def batch_store_vectors(
        self,
        vectors: list[
            tuple[UUID, list[float] | dict[str, list[float]], dict[str, Any]]
        ],
        tenant_id: str,
    ) -> int:
        """Store multiple vectors in a batch."""
        if not vectors:
            return 0

        await self._ensure_collection()
        points = []

        for mem_id, emb, meta in vectors:
            # Handle named vectors vs list
            if isinstance(emb, dict):
                vector_data = emb
                # Dynamically register any new vectors found
                for v_name, v_vec in emb.items():
                    if v_name not in self._known_vectors:
                        await self.ensure_vector_config(v_name, len(v_vec))
            else:
                vector_data = {self.vector_name: emb}
                # Ensure default vector is registered (should be done in _ensure_collection but safe to check)
                if self.vector_name not in self._known_vectors:
                    await self.ensure_vector_config(self.vector_name, len(emb))

            # Prepare payload
            payload = {
                "memory_id": str(mem_id),
                "tenant_id": str(tenant_id),
                **(meta or {}),
            }

            # DEBUG: Inspect payload
            logger.info(
                f"DEBUG_QDRANT_PAYLOAD: vector_keys={list(vector_data.keys())} target_collection={self.collection_name}"
            )

            points.append(
                PointStruct(id=str(mem_id), vector=vector_data, payload=payload)
            )

        try:
            await self.client.upsert(
                collection_name=self.collection_name, points=points
            )
            return len(points)
        except Exception as e:
            # Enhanced logging for debugging
            error_details = str(e)
            if hasattr(e, "content"):  # Qdrant client specific
                error_details += f" | Content: {e.content}"
            if hasattr(e, "response"):
                error_details += f" | Response: {e.response}"

            logger.error(f"Qdrant batch upsert failed: {error_details}")
            return 0

    async def get_vector(
        self, memory_id: UUID, tenant_id: str, vector_name: str | None = None
    ) -> list[float] | None:
        """Retrieve a vector embedding by ID."""
        target_vector = vector_name or self.vector_name
        await self._ensure_collection()
        try:
            results = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(memory_id)],
                with_vectors=True,
                with_payload=True,
            )

            if not results:
                return None

            record = results[0]
            # Security check: Ensure tenant matches
            if not record.payload or record.payload.get("tenant_id") != tenant_id:
                return None

            if record.vector:
                if isinstance(record.vector, dict):
                    return cast(list[float] | None, record.vector.get(target_vector))
                return cast(list[float] | None, record.vector)

            return None
        except Exception as e:
            logger.error(f"Qdrant get_vector failed: {e}")
            return None

    async def update_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector embedding."""
        return await self.store_vector(memory_id, embedding, tenant_id, metadata)

    async def delete_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a vector."""
        await self._ensure_collection()
        try:
            # First check if it belongs to tenant
            existing = await self.client.retrieve(
                collection_name=self.collection_name,
                ids=[str(memory_id)],
                with_payload=True,
            )
            if (
                not existing
                or not existing[0].payload
                or existing[0].payload.get("tenant_id") != tenant_id
            ):
                return False

            await self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[str(memory_id)]),
            )
            return True
        except Exception as e:
            logger.error(f"Qdrant delete failed: {e}")
            return False

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
        vector_name: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        """Search for similar vectors."""
        target_vector = vector_name or self.vector_name
        await self._ensure_collection()

        search_filter = self._build_filter(
            tenant_id=tenant_id,
            layer=layer,
            agent_id=agent_id,
            session_id=session_id,
            project=project,
            extra_filters=filters,
        )

        try:
            # Use query_points instead of search (deprecated/removed in AsyncQdrantClient)
            response = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                using=target_vector,
                query_filter=search_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )

            output = []
            for r in response.points:
                if r.payload and "memory_id" in r.payload:
                    output.append((UUID(r.payload["memory_id"]), float(r.score)))

            return output
        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

    # --- Extra Methods Required by Tests / Legacy Support ---

    async def add_vector(
        self,
        memory_id: UUID,
        embedding: list[float],
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> bool:
        """Alias for store_vector (Legacy)."""
        metadata = {}
        if agent_id:
            metadata["agent_id"] = agent_id
        if layer:
            metadata["layer"] = layer
        return await self.store_vector(memory_id, embedding, tenant_id, metadata)

    async def count_vectors(self, tenant_id: str, layer: str | None = None) -> int:
        """Count vectors for a tenant."""
        await self._ensure_collection()
        try:
            count_filter = self._build_filter(tenant_id=tenant_id, layer=layer)
            result = await self.client.count(
                collection_name=self.collection_name, count_filter=count_filter
            )
            return result.count
        except Exception as e:
            logger.error(f"Qdrant count failed: {e}")
            return 0

    async def delete_by_layer(self, tenant_id: str, agent_id: str, layer: str) -> int:
        """Delete vectors matching criteria."""
        await self._ensure_collection()
        try:
            del_filter = self._build_filter(
                tenant_id=tenant_id, agent_id=agent_id, layer=layer
            )
            # Qdrant python client delete_by_filter returns UpdateResult?
            _result = await self.client.delete(
                collection_name=self.collection_name, points_selector=del_filter
            )
            # Cannot easily get count of deleted items without extra query
            return 1  # Assume success
        except Exception as e:
            logger.error(f"Qdrant delete_by_layer failed: {e}")
            return 0

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self.client.close()

    async def search_with_contradiction_penalty(
        self,
        query_embedding: list[float],
        tenant_id: str,
        penalty_factor: float = 0.5,
        limit: int = 10,
        contradiction_threshold: float = 0.15,
    ) -> list[tuple[UUID, float]]:
        """Experimental: Search and penalize results that contradict the query."""
        # 1. Standard search
        results = await self.search_similar(query_embedding, tenant_id, limit=limit)
        if not results:
            return []

        # 2. Re-score based on contradiction detection
        penalized_results = []
        for memory_id, score in results:
            memory_vector = await self.get_vector(memory_id, tenant_id)
            final_score = score
            if memory_vector:
                # Calculate cosine similarity
                dot = sum(a * b for a, b in zip(query_embedding, memory_vector))
                mag1 = math.sqrt(sum(a * a for a in query_embedding))
                mag2 = math.sqrt(sum(b * b for b in memory_vector))
                similarity = dot / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0

                if similarity < contradiction_threshold:
                    final_score = score * penalty_factor

            penalized_results.append((memory_id, final_score))

        return sorted(penalized_results, key=lambda x: x[1], reverse=True)
