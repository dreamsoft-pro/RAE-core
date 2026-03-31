"""Abstract embedding provider interface for RAE-core."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class IEmbeddingProvider(Protocol):
    """Abstract interface for embedding providers."""

    async def embed_text(
        self, text: str, task_type: str = "search_document"
    ) -> list[float]:
        """Generate embedding for text.

        Args:
            text: The text to embed.
            task_type: Task type hint ("search_query" or "search_document").
        """
        ...

    async def embed_batch(
        self, texts: list[str], task_type: str = "search_document"
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            task_type: Task type hint ("search_query" or "search_document").
        """
        ...

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        ...
