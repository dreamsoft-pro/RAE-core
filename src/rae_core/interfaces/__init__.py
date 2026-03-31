"""Abstract interfaces for RAE-core dependency injection.

All storage adapters must implement these interfaces to be compatible with RAE-core.
"""

from .cache import ICacheProvider
from .embedding import IEmbeddingProvider
from .graph import IGraphStore
from .llm import ILLMProvider
from .storage import IMemoryStorage
from .sync import ISyncProvider
from .vector import IVectorStore

__all__ = [
    "IMemoryStorage",
    "IVectorStore",
    "IGraphStore",
    "ICacheProvider",
    "ILLMProvider",
    "IEmbeddingProvider",
    "ISyncProvider",
]
