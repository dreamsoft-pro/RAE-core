"""Abstract vector store interface for RAE-core.

This module defines the vector store interface for similarity search.
Implementations can use Qdrant, sqlite-vec, FAISS, or any other vector DB.
"""

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class IVectorStore(Protocol):
    """Abstract interface for vector storage and similarity search.

    Implementations must provide vector storage and efficient similarity search
    capabilities for memory embeddings.
    """

    async def store_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a vector embedding.

        Args:
            memory_id: UUID of the memory
            embedding: Vector embedding (list) or dict of named vectors
            tenant_id: Tenant identifier
            metadata: Optional metadata to store with the vector

        Returns:
            True if successful, False otherwise
        """
        ...

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
        """Search for similar vectors using cosine similarity.

        Args:
            query_embedding: Query vector
            tenant_id: Tenant identifier
            layer: Optional layer filter
            limit: Maximum number of results
            score_threshold: Optional minimum similarity score (0.0-1.0)
            agent_id: Optional agent identifier for filtering
            session_id: Optional session identifier for filtering
            filters: Optional dictionary of generic metadata filters
            project: Optional project identifier for filtering
            **kwargs: Additional backend-specific arguments

        Returns:
            List of (memory_id, similarity_score) tuples, sorted by score descending
        """
        ...

    async def delete_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a vector.

        Args:
            memory_id: UUID of the memory
            tenant_id: Tenant identifier

        Returns:
            True if successful, False otherwise
        """
        ...

    async def update_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector embedding.

        Args:
            memory_id: UUID of the memory
            embedding: New vector embedding (list) or dict of named vectors
            tenant_id: Tenant identifier
            metadata: Optional new metadata

        Returns:
            True if successful, False otherwise
        """
        ...

    async def get_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> list[float] | None:
        """Retrieve a vector embedding.

        Args:
            memory_id: UUID of the memory
            tenant_id: Tenant identifier

        Returns:
            Vector embedding or None if not found
        """
        ...

    async def batch_store_vectors(
        self,
        vectors: list[
            tuple[UUID, list[float] | dict[str, list[float]], dict[str, Any]]
        ],
        tenant_id: str,
    ) -> int:
        """Store multiple vectors in a batch.

        Args:
            vectors: List of (memory_id, embedding, metadata) tuples
            tenant_id: Tenant identifier

        Returns:
            Number of successfully stored vectors
        """
        ...
