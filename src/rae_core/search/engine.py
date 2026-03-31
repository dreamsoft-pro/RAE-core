"""Hybrid search engine implementation."""

import math
from typing import Any
from uuid import UUID

import structlog

from rae_core.interfaces.embedding import IEmbeddingProvider
from rae_core.interfaces.reranking import IReranker
from rae_core.interfaces.storage import IMemoryStorage
from rae_core.math.fusion import FusionStrategy
from rae_core.math.metadata_injector import MetadataInjector
from rae_core.search.strategies import SearchStrategy

logger = structlog.get_logger(__name__)


class EmeraldReranker(IReranker):
    """Emerald Reranker v1 - Semantic Cross-Validation."""

    def __init__(self, embedding_provider: Any, memory_storage: Any):
        self.embedding_provider = embedding_provider
        self.memory_storage = memory_storage

    async def rerank(
        self,
        query: str,
        candidates: list[tuple[UUID, float, float]],
        tenant_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        if self.embedding_provider is None or self.memory_storage is None:
            return candidates[:limit]

        try:
            query_emb = await self.embedding_provider.embed_text(
                query, task_type="search_query"
            )
            reranked: list[tuple[UUID, float, float]] = []

            for item in candidates:
                m_id, original_score, importance, audit_log = self._unpack_candidate_with_audit(item)

                # Get full content for deep comparison
                memory = await self.memory_storage.get_memory(m_id, tenant_id)
                if not memory:
                    reranked.append((m_id, original_score, importance, audit_log))
                    continue

                # Semantic Scoring (Emerald Logic)
                memory_emb = await self.embedding_provider.embed_text(
                    memory.content, task_type="search_document"
                )
                
                # Manual Cosine Similarity (No external dependencies needed here)
                v1 = query_emb
                v2 = memory_emb
                dot = sum(a * b for a, b in zip(v1, v2))
                mag1 = math.sqrt(sum(a * a for a in v1))
                mag2 = math.sqrt(sum(a * a for a in v2))
                semantic_score = dot / (mag1 * mag2) if (mag1 * mag2) > 0 else 0.0

                # Synergy Score: 70% Semantic, 30% Original Rank
                final_score = (semantic_score * 0.7) + (original_score * 0.3)
                audit_log["emerald_semantic"] = semantic_score
                reranked.append((m_id, final_score, importance, audit_log))

            return sorted(reranked, key=lambda x: x[1], reverse=True)[:limit]
        except Exception as e:
            logger.error("rerank_failed", error=str(e))
            return candidates[:limit]

    def _unpack_candidate_with_audit(self, item: tuple) -> tuple[UUID, float, float, dict]:
        """Safely unpack candidate tuple with audit log (4 values)."""
        if len(item) == 4:
            return item[0], item[1], item[2], item[3]
        elif len(item) == 3:
            return item[0], item[1], item[2], {}
        elif len(item) == 2:
            return item[0], item[1], 0.0, {}
        else:
            raise ValueError(f"Invalid candidate tuple length: {len(item)}")

    def _unpack_candidate(self, item: tuple) -> tuple[UUID, float, float]:
        """Safely unpack candidate tuple (handles 2, 3, or 4 values)."""
        if len(item) == 4:
            return item[0], item[1], item[2]
        elif len(item) == 3:
            return item[0], item[1], item[2]
        elif len(item) == 2:
            return item[0], item[1], 0.0
        else:
            raise ValueError(f"Invalid candidate tuple length: {len(item)}")


class HybridSearchEngine:
    """
    Orchestrates multiple search strategies (Vector + FullText) and fuses them using LogicGateway.
    """

    def __init__(
        self,
        strategies: dict[str, SearchStrategy],
        embedding_provider: IEmbeddingProvider,
        memory_storage: IMemoryStorage,
        reranker: IReranker | None = None,
        graph_store: Any | None = None,
        math_controller: Any | None = None,
    ):
        self.strategies = strategies
        self.embedding_provider = embedding_provider
        self.memory_storage = memory_storage
        self.graph_store = graph_store
        self._reranker = reranker
        self.math_controller = math_controller
        self.injector = MetadataInjector()

        # Initialize Fusion Strategy (LogicGateway wrapper)
        config = self.math_controller.config if self.math_controller else {}
        self.fusion_strategy = FusionStrategy(config)
        if hasattr(self.fusion_strategy, "gateway"):
            self.fusion_strategy.gateway.storage = memory_storage
            self.fusion_strategy.gateway.graph_store = graph_store

    async def search(
        self,
        query: str,
        tenant_id: str,
        agent_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        strategies: list[str] | None = None,
        strategy_weights: dict[str, float] | None = None,
        enable_reranking: bool = False,
        math_controller: Any = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        """
        Execute search across active strategies and fuse results.
        Supports System 7.2 gateway_config override.
        """
        active_strategies = strategies or list(self.strategies.keys())
        weights = strategy_weights or {}

        # Extract Gateway Config Override (System 7.2)
        gateway_config = kwargs.get("gateway_config")

        # SYSTEM 43.0: No Enrichment. Enrichment dilutes technical signal.
        enriched_query = query 

        # 1. PARALLEL EXECUTION (Full Spectrum Synergy)
        strategy_results: dict[str, list[tuple[UUID, float, float]]] = {}
        
        # SYSTEM 40.8: No Fast Path skipping. We want mathematical consensus from all strategies.
        should_skip_vector = False

        async def run_strategies():
            tasks = []
            task_names = []
            for name in active_strategies:
                if name not in self.strategies:
                    continue

                strategy = self.strategies[name]
                task_names.append(name)
                
                # Split-Brain Querying: Vector gets enriched, FullText gets precise
                target_query = enriched_query if name == "vector" else query
                
                tasks.append(strategy.search(
                    query=target_query,
                    tenant_id=tenant_id,
                    filters=filters,
                    limit=limit,
                    **kwargs,
                ))
            
            if not tasks:
                return {}
                
            import asyncio
            all_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            results_dict = {}
            for name, results in zip(task_names, all_results):
                if isinstance(results, Exception):
                    logger.error("strategy_failed", name=name, error=str(results))
                    results_dict[name] = []
                else:
                    results_dict[name] = [
                        (r[0], r[1], r[2] if len(r) > 2 else 0.0) for r in results
                    ]
            return results_dict

        # Run all strategies in parallel
        strategy_results = await run_strategies()
        
        if not strategy_results:
            logger.warning("no_active_strategies_for_search")

        # 2. FUSION (LogicGateway)
        # Fetch contents for reranking
        all_ids = set()
        for results in strategy_results.values():
            for r in results:
                all_ids.add(r[0])  # m_id

        memory_data = {}
        if all_ids and hasattr(self.memory_storage, "get_memories_batch"):
            try:
                mems = await self.memory_storage.get_memories_batch(
                    list(all_ids), tenant_id
                )
                for mem in mems:
                    # Store full memory object instead of just content
                    memory_data[mem["id"]] = mem
            except Exception as e:
                logger.warning("batch_fetch_failed", error=str(e))

        # Note: We pass 'gateway_config' to fuse if supported
        # SYSTEM 41.1: We pass the ORIGINAL query to LogicGateway (Resonance/Neural)
        fused_results = await self.fusion_strategy.fuse(
            strategy_results,
            weights,
            query=query,
            config_override=gateway_config,
            memory_contents=memory_data,
        )

        # 3. OPTIONAL RERANKING
        if enable_reranking and len(fused_results) > 1:
            # Skip if LogicGateway already reranked (Neural Scalpel active)
            if (
                hasattr(self.fusion_strategy, "gateway")
                and self.fusion_strategy.gateway.reranker
            ):
                return fused_results[:limit]

            reranker = self._reranker
            if reranker is None and self.embedding_provider and self.memory_storage:
                reranker = EmeraldReranker(self.embedding_provider, self.memory_storage)

            if reranker:
                logger.info("reranking_activated", count=len(fused_results[:20]))
                # SYSTEM 41.1: Use original query for reranking
                return await reranker.rerank(
                    query, fused_results[:20], tenant_id, limit=limit
                )

        return fused_results[:limit]
