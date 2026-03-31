"""Search models for RAE-core hybrid search."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchStrategy(str, Enum):
    """Search strategy enumeration."""

    VECTOR = "vector"  # Pure vector similarity
    GRAPH = "graph"  # Graph traversal (GraphRAG)
    SPARSE = "sparse"  # Sparse vector (BM25-like)
    FULLTEXT = "fulltext"  # Full-text search
    HYBRID = "hybrid"  # Fusion of multiple strategies


class SearchQuery(BaseModel):
    """Search query model."""

    query: str = Field(description="Search query text")
    strategy: SearchStrategy = Field(
        default=SearchStrategy.HYBRID, description="Search strategy to use"
    )

    # Filters
    tenant_id: str = Field(description="Tenant identifier")
    agent_id: str | None = Field(default=None, description="Filter by agent")
    layer: str | None = Field(default=None, description="Filter by memory layer")
    tags: list[str] | None = Field(
        default=None, description="Filter by tags (OR logic)"
    )
    min_importance: float | None = Field(
        default=None, ge=0.0, description="Minimum importance threshold"
    )

    # Search parameters
    limit: int = Field(default=10, ge=1, le=100, description="Max results")
    score_threshold: float | None = Field(
        default=None, ge=0.0, description="Minimum score threshold"
    )

    # Fusion weights (for hybrid search)
    weights: dict[str, float] | None = Field(
        default=None,
        description="Strategy weights for fusion (vector, graph, sparse, fulltext)",
    )

    # Graph-specific parameters
    graph_depth: int = Field(
        default=2, ge=1, le=5, description="Max graph traversal depth"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What are the user's programming preferences?",
                "strategy": "hybrid",
                "tenant_id": "tenant_123",
                "layer": "episodic",
                "limit": 10,
                "weights": {
                    "vector": 0.4,
                    "graph": 0.3,
                    "sparse": 0.2,
                    "fulltext": 0.1,
                },
            }
        }
    )


class SearchResult(BaseModel):
    """Single search result."""

    memory_id: str = Field(description="UUID of the memory")
    content: str = Field(description="Memory content")
    score: float = Field(description="Relevance score", ge=0.0)
    strategy_used: SearchStrategy = Field(
        description="Which strategy produced this result"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional result metadata"
    )


class SearchResponse(BaseModel):
    """Search response with results and metadata."""

    results: list[SearchResult] = Field(description="Search results")
    query: str = Field(description="Original query")
    strategy: SearchStrategy = Field(description="Strategy used")
    total_found: int = Field(description="Total matching results before limit")
    execution_time_ms: float = Field(description="Execution time in milliseconds")
    fusion_details: dict[str, Any] | None = Field(
        default=None, description="Details about fusion process"
    )
    synthesized_context: str | None = Field(
        default=None, description="Combined knowledge context (for GraphRAG)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "results": [
                    {
                        "memory_id": "550e8400-e29b-41d4-a716-446655440000",
                        "content": "User prefers Python",
                        "score": 0.92,
                        "strategy_used": "vector",
                    }
                ],
                "query": "What does user prefer?",
                "strategy": "hybrid",
                "total_found": 15,
                "execution_time_ms": 45.2,
            }
        }
    )


class ScoringWeights(BaseModel):
    """Weights for unified memory scoring."""

    similarity: float = Field(
        default=0.4, ge=0.0, description="Vector similarity weight"
    )
    importance: float = Field(
        default=0.3, ge=0.0, description="Importance score weight"
    )
    recency: float = Field(
        default=0.3, ge=0.0, description="Recency/decay weight"
    )

    def validate_sum(self) -> bool:
        """Validate that weights sum to 1.0."""
        total = self.similarity + self.importance + self.recency
        return abs(total - 1.0) < 0.01  # Allow small floating point errors
