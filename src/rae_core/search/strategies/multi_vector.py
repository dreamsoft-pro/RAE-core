import asyncio
import math
from typing import Any
from uuid import UUID

from ...interfaces.embedding import IEmbeddingProvider
from ...interfaces.vector import IVectorStore
from . import SearchStrategy


class MultiVectorSearchStrategy(SearchStrategy):
    """
    Hybrid search strategy using multiple vector spaces AND multiple reflection layers.
    SYSTEM 96.0: Neural Polyglot (Concurrent Layer & Space Exploration)
    """

    def __init__(
        self,
        strategies: list[tuple[IVectorStore, IEmbeddingProvider, str]],
        default_weight: float = 0.5,
    ) -> None:
        self.strategies_list = strategies
        self.default_weight = default_weight
        # Define the 3 core layers of RAE memory
        self.target_layers = ["episodic", "semantic", "reflective"]

    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        """Search across all registered vector spaces and reflection layers concurrently."""
        if not self.strategies_list:
            return []

        # Determine which layers to search
        safe_filters = filters.copy() if filters else {}
        layers_to_search = self.target_layers
        
        # If a specific layer is forced via filters or kwargs, respect it
        forced_layer = safe_filters.get("layer") or kwargs.get("layer")
        if forced_layer:
            layers_to_search = [forced_layer]
            
        # Remove 'layer' from base kwargs so we can inject it dynamically per task
        safe_kwargs = kwargs.copy()
        safe_kwargs.pop("layer", None)
        safe_filters.pop("layer", None)

        tasks = []
        task_metadata = [] # Keep track of (vector_name, layer) for each task

        # Generate embeddings for all providers first (can be done sequentially as it's usually fast, 
        # or we could parallelize it too, but API limits might be an issue. Let's do it per provider).
        embeddings_cache = {}
        for store, embedder, vector_name in self.strategies_list:
            try:
                embeddings_cache[vector_name] = await embedder.embed_text(query, task_type="search_query")
            except Exception:
                continue

        # Create a massive parallel grid: [Vector Spaces] x [Layers]
        for store, embedder, vector_name in self.strategies_list:
            if vector_name not in embeddings_cache:
                continue
                
            query_embedding = embeddings_cache[vector_name]

            for layer in layers_to_search:
                layer_filters = safe_filters.copy()
                layer_filters["layer"] = layer
                
                tasks.append(
                    store.search_similar(
                        query_embedding=query_embedding,
                        tenant_id=tenant_id,
                        limit=limit,
                        vector_name=vector_name,
                        filters=layer_filters,
                        project=project,
                        **safe_kwargs,
                    )
                )
                task_metadata.append(f"{vector_name}::{layer}")

        if not tasks:
            return []

        # Execute all searches concurrently
        all_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Fuse results using Exponential Rank Sharpening
        fused_scores: dict[UUID, float] = {}
        
        for meta, strategy_res in zip(task_metadata, all_results):
            if isinstance(strategy_res, Exception) or not strategy_res:
                continue
                
            for rank, result in enumerate(strategy_res):
                m_id = result[0]
                # Using the same constant as System 37.0 for consistency
                fused_scores[m_id] = fused_scores.get(m_id, 0.0) + math.exp(-rank / 3.0)

        # Sort and return top K
        final = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [(m_id, score, 0.0) for m_id, score in final]

    def get_strategy_name(self) -> str:
        return "multi_vector_fusion"

    def get_strategy_weight(self) -> float:
        return self.default_weight
