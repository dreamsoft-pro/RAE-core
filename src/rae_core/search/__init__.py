"""Search module for RAE-core.

Provides hybrid search capabilities combining multiple strategies:
- Vector search (semantic similarity)
- Graph traversal (relationship-based)
- Sparse vectors (BM25)
- Full-text (keyword matching)
"""

from rae_core.search.cache import SearchCache
from rae_core.search.engine import HybridSearchEngine
from rae_core.search.strategies import SearchStrategy
from rae_core.search.strategies.fulltext import FullTextStrategy
from rae_core.search.strategies.graph import GraphTraversalStrategy
from rae_core.search.strategies.sparse import SparseVectorStrategy
from rae_core.search.strategies.vector import VectorSearchStrategy

__all__ = [
    "SearchStrategy",
    "VectorSearchStrategy",
    "GraphTraversalStrategy",
    "SparseVectorStrategy",
    "FullTextStrategy",
    "HybridSearchEngine",
    "SearchCache",
]
