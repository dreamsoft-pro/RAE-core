"""RAE Engine - The Intelligent Memory Manifold."""

import math
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RAEEngine:
    """
    RAE Engine: A self-tuning memory manifold that uses designed math
    to navigate vector spaces more intelligently than standard RAG.
    """

    def __init__(
        self,
        memory_storage: Any,
        vector_store: Any,
        embedding_provider: Any,
        llm_provider: Any = None,
        settings: Any = None,
        cache_provider: Any = None,
        search_engine: Any = None,
        math_controller: Any = None,
        resonance_engine: Any = None,
    ):
        self.memory_storage = memory_storage
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.settings = settings
        self.cache_provider = cache_provider

        # Initialize Math Layer Controller (The Brain)
        from rae_core.math.controller import MathLayerController
        from rae_core.math.resonance import SemanticResonanceEngine

        self.math_ctrl = math_controller or MathLayerController(config=settings)

        # Load resonance factor from config
        res_factor = self.math_ctrl.get_engine_param("resonance_factor", 0.4)

        self.resonance_engine = resonance_engine or SemanticResonanceEngine(
            resonance_factor=float(res_factor)
        )

        # Flattened Search Engine Strategies (System 43.0)
        # We register all available vector spaces as independent strategies for global fusion
        strategies = {
            "fulltext": self._init_fulltext_strategy(),
            "anchor": self._init_anchor_strategy(),
        }
        
        # Dynamic Multi-Vector Registration
        from rae_core.embedding.manager import EmbeddingManager
        from rae_core.search.strategies.vector import VectorSearchStrategy

        if isinstance(self.embedding_provider, EmbeddingManager):
            for name, provider in self.embedding_provider.providers.items():
                # Register each named vector space as a first-class strategy
                strategy_key = f"vector_{name}"
                strategies[strategy_key] = VectorSearchStrategy(
                    self.vector_store, provider, vector_name=name
                )
        else:
            strategies["vector"] = VectorSearchStrategy(
                self.vector_store, self.embedding_provider
            )

        if search_engine:
            self.search_engine = search_engine
        else:
            from rae_core.search.engine import HybridSearchEngine

            self.search_engine = HybridSearchEngine(
                strategies=strategies,
                embedding_provider=self.embedding_provider,
                memory_storage=self.memory_storage,
            )

    def _init_anchor_strategy(self):
        from rae_core.search.strategies.anchor import AnchorStrategy

        return AnchorStrategy(self.memory_storage)

    def _init_vector_strategy(self):
        from rae_core.embedding.manager import EmbeddingManager

        if isinstance(self.embedding_provider, EmbeddingManager):
            from rae_core.search.strategies.multi_vector import (
                MultiVectorSearchStrategy,
            )

            strategies = []
            for name, provider in self.embedding_provider.providers.items():
                strategies.append((self.vector_store, provider, name))
            return MultiVectorSearchStrategy(strategies=strategies)

        from rae_core.search.strategies.vector import VectorSearchStrategy

        return VectorSearchStrategy(self.vector_store, self.embedding_provider)

    def _init_fulltext_strategy(self):
        from rae_core.search.strategies.fulltext import FullTextStrategy

        return FullTextStrategy(self.memory_storage)

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        RAE Reflective Search: Retrieval -> Math Scoring -> Manifold Adjustment.
        """
        search_filters = {**(filters or {})}
        if agent_id:
            search_filters["agent_id"] = agent_id
        if project:
            search_filters["project"] = project
        if layer:
            search_filters["layer"] = layer

        # 1. BANDIT TUNING: Get "weights" and PARAMS (Spectrum Strategy)
        custom_weights = kwargs.get("custom_weights")
        strategy_weights = None
        engine_params = {}

        if isinstance(custom_weights, dict):
            strategy_weights = custom_weights

        if not strategy_weights:
            # This now returns weights + _params + _arm_id
            strategy_config = self.math_ctrl.get_retrieval_weights(query)

            # Extract internal params
            engine_params = strategy_config.pop("_params", {})
            arm_id = strategy_config.pop("_arm_id", "unknown")

            strategy_weights = strategy_config
            logger.info("spectrum_strategy_active", arm=arm_id, params=engine_params)

        # Apply Dynamic Params to Engine Components
        # 1. Resonance
        if "resonance_factor" in engine_params:
            self.resonance_engine.resonance_factor = float(
                engine_params["resonance_factor"]
            )

        # 2. Rerank Gate & Limits (LogicGateway override)
        gateway_config_override = engine_params.copy()
        if "rerank_gate" in engine_params:
            gateway_config_override["confidence_gate"] = float(
                engine_params["rerank_gate"]
            )

        # --- SYSTEM 7.2: MAP WEIGHTS TO THRESHOLDS (Legacy override removed, Bandit rules now) ---

        # Prepare arguments safely
        active_strategies = kwargs.get("strategies") or search_filters.get("strategies")
        engine_limit = self.math_ctrl.get_engine_param("limit", 100)
        enable_reranking = kwargs.get("enable_reranking", False)

        # Clean kwargs to avoid duplicates in **search_kwargs
        search_kwargs = kwargs.copy()
        for k in [
            "strategies",
            "strategy_weights",
            "enable_reranking",
            "custom_weights",
            "limit",
        ]:
            search_kwargs.pop(k, None)

        if gateway_config_override:
            search_kwargs["gateway_config"] = gateway_config_override

        # EXECUTE HYBRID RETRIEVAL (Enrichment handled internally by SearchEngine)
        candidates = await self.search_engine.search(
            query=query,
            tenant_id=tenant_id,
            agent_id=agent_id,
            filters=search_filters,
            limit=int(engine_limit),
            strategies=active_strategies,
            strategy_weights=strategy_weights,
            enable_reranking=enable_reranking,
            **search_kwargs,
        )

        # 2. DESIGNED MATH SCORING
        from rae_core.math.structure import ScoringWeights

        scoring_weights = None
        if isinstance(custom_weights, dict):
            valid_fields = {
                k: v
                for k, v in custom_weights.items()
                if k in ["alpha", "beta", "gamma"]
            }
            scoring_weights = ScoringWeights(**valid_fields)
        elif custom_weights:
            scoring_weights = custom_weights

        # 1. RETRIEVAL & DEDUPLICATION
        # We ensure each memory ID appears only once, with its best score
        best_candidates = {}
        for item in candidates:
            m_id = item[0]
            sim_score = item[1]
            # Use tuple structure: (score, importance, audit_log)
            if m_id not in best_candidates or sim_score > best_candidates[m_id][0]:
                best_candidates[m_id] = item[1:]

        memories = []
        for m_id, (sim_score, importance, audit_log) in best_candidates.items():
            memory = await self.memory_storage.get_memory(m_id, tenant_id)
            if memory:
                # 2. DESIGNED MATH SCORING
                # We calculate math score, but if we have a Symbolic Hard-Lock from Stage 1,
                # we must ensure it remains the dominant factor.
                math_score = self.math_ctrl.score_memory(
                    memory, query_similarity=sim_score, weights=scoring_weights
                )
                
                # Silicon Oracle: If audit_log contains SIC (Symbolic Information Content), 
                # we use the fused sim_score directly as it already contains the SIC multiplier.
                if audit_log.get("sic_boost"):
                    math_score = sim_score
                
                if isinstance(memory, object) and hasattr(memory, "to_dict"):
                    memory_dict = memory.to_dict()
                elif not isinstance(memory, dict):
                    # Fallback for unexpected types
                    memory_dict = dict(memory)
                else:
                    memory_dict = memory

                memory_dict["math_score"] = math_score
                memory_dict["search_score"] = sim_score
                memory_dict["importance"] = importance or memory_dict.get("importance", 0.5)
                memory_dict["audit_trail"] = audit_log
                
                # Explicitly log the winning feature for auditability
                if math_score > 0.8:
                    win_feature = "unknown"
                    if audit_log.get("sic_boost"): win_feature = "symbolic_hard_lock"
                    elif audit_log.get("hard_lock"): win_feature = "symbolic_hard_lock"
                    elif audit_log.get("anchor_hit"): win_feature = "anchor_match"
                    elif audit_log.get("cat_boost"): win_feature = "category_match"
                    elif audit_log.get("quant_boost"): win_feature = "quantitative_resonance"
                    
                    logger.info("memory_selection_proof", 
                                 id=str(m_id), 
                                 score=math_score, 
                                 feature=win_feature,
                                 audit=audit_log)
                
                memories.append(memory_dict)

        # 3. SEMANTIC RESONANCE
        if hasattr(self.memory_storage, "get_neighbors_batch") and memories:
            m_ids = [m["id"] for m in memories]
            edges = await self.memory_storage.get_neighbors_batch(m_ids, tenant_id)
            if edges:
                candidate_ids = {str(m["id"]) for m in memories}
                memories, energy_map = self.resonance_engine.compute_resonance(
                    memories, edges
                )

                induced_ids = []
                if energy_map:
                    max_e = max(energy_map.values())
                    dyn_threshold = self.math_ctrl.get_resonance_threshold(query)
                    threshold = max_e * dyn_threshold

                    for node_id, energy in energy_map.items():
                        if node_id not in candidate_ids and energy > threshold:
                            induced_ids.append(node_id)

                if induced_ids:
                    logger.info(
                        "reflection_induction_triggered", count=len(induced_ids)
                    )
                    for mid_str in induced_ids[:5]:
                        try:
                            from uuid import UUID

                            induced_mem = await self.memory_storage.get_memory(
                                UUID(mid_str), tenant_id
                            )
                            if induced_mem:
                                if isinstance(induced_mem, object) and hasattr(induced_mem, "to_dict"):
                                    induced_mem_dict = induced_mem.to_dict()
                                elif not isinstance(induced_mem, dict):
                                    induced_mem_dict = dict(induced_mem)
                                else:
                                    induced_mem_dict = induced_mem

                                induced_mem_dict["math_score"] = float(
                                    np.tanh(energy_map[mid_str])
                                )
                                induced_mem_dict["resonance_metadata"] = {
                                    "induced": True,
                                    "boost": float(energy_map[mid_str]),
                                }
                                memories.append(induced_mem_dict)
                        except Exception:
                            continue

        # SYSTEM 40.17: Guaranteed Tier Isolation
        memories.sort(key=lambda x: (x.get("audit_trail", {}).get("tier", 2), -x.get("math_score", 0.0)))

        # 4. ACTIVE SZUBAR LOOP (System 52.0 - Neighbor Recruitment)
        # "Success from Failure": If confidence is low, explore the graph for missing links.
        top_score = memories[0].get("math_score", 0.0) if memories else 0.0

        if top_score < 0.75 and not kwargs.get("_is_retry") and hasattr(self.memory_storage, "get_neighbors_batch"):
            logger.info("active_szubar_expansion_triggered", query=query, top_score=top_score)

            # 1. Identify seed anchors (Top candidates that almost made it)
            seed_ids = [m["id"] for m in memories[:10]]
            
            # 2. Fetch adjacency list from Knowledge Graph
            edges = await self.memory_storage.get_neighbors_batch(seed_ids, tenant_id)
            if edges:
                # 3. Identify recruited candidates (Neighbors NOT in original results)
                current_ids = {m["id"] for m in memories}
                recruited_ids = []
                for _, neighbors in edges.items():
                    for n_id, _ in neighbors:
                        if n_id not in current_ids: recruited_ids.append(n_id)
                
                if recruited_ids:
                    # 4. Fetch full content for deep verification
                    new_mems_data = await self.memory_storage.get_memories_batch(
                        recruited_ids[:30], tenant_id
                    )
                    
                    if new_mems_data:
                        logger.info("szubar_neighbor_recruitment", count=len(new_mems_data))
                        
                        # 5. RE-SCORING: Run recruited neighbors through the same logic
                        # We simulate a "mini-search" for these specific items
                        recruited_results = []
                        for m in new_mems_data:
                            # Re-run Math + Neural logic for the new candidate
                            # We use LogicGateway directly if available
                            multiplier, audit = self.search_engine.fusion_strategy.gateway._apply_mathematical_logic(
                                query, m["content"], m.get("metadata", {})
                            )
                            
                            # Probabilistic score for the newcomer
                            # We treat it as if it was found at a baseline rank
                            base_p = 0.5 # Neutral prior for recruited node
                            final_logit = math.log(base_p / (1.0 - base_p)) + math.log(max(multiplier, 1e-9))
                            
                            # Neural Verification if enabled
                            if kwargs.get("enable_reranking") and self.search_engine.fusion_strategy.gateway.reranker:
                                pair = (query, f"[CONTENT] {m['content']}")
                                logit = self.search_engine.fusion_strategy.gateway.reranker.predict([pair])[0]
                                final_logit += logit
                                audit["neural_logit"] = logit

                            m_dict = dict(m) if not isinstance(m, dict) else m
                            m_dict["math_score"] = self.search_engine.fusion_strategy.gateway.sigmoid(final_logit)
                            m_dict["audit_trail"] = audit
                            m_dict["audit_trail"]["szubar_recruited"] = True
                            recruited_results.append(m_dict)
                        
                        # Inject and re-sort
                        memories.extend(recruited_results)
                        memories.sort(key=lambda x: (x.get("audit_trail", {}).get("tier", 2), -x.get("math_score", 0.0)))
                        
                        # Check if Szubar saved the day
                        new_top_score = memories[0].get("math_score", 0.0)
                        if new_top_score > top_score:
                            logger.info("szubar_recovery_success", 
                                        old_score=top_score, 
                                        new_score=new_top_score,
                                        recovered_id=str(memories[0]["id"]))

        return memories[:top_k]

    async def generate_text(self, prompt: str, **kwargs) -> str:
        if not self.llm_provider:
            raise RuntimeError("LLM provider not configured")
        from typing import cast

        return cast(str, await self.llm_provider.generate_text(prompt=prompt, **kwargs))

    async def store_memory(self, **kwargs):
        content = kwargs.get("content", "")
        tenant_id = kwargs.get("tenant_id")
        project = kwargs.get("project", "default")
        
        # SYSTEM 92.4: Quality Guard at Ingestion (Autonomous Firewall)
        if kwargs.get("validate") and content.strip():
            logger.info("ingestion_quality_guard_active", project=project)
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    q_resp = await client.post(f"{self.settings.QUALITY_API_URL}/v2/quality/audit", json={
                        "code": content,
                        "project": project,
                        "importance": "medium"
                    })
                    if q_resp.status_code == 200:
                        verdict = q_resp.json()
                        if verdict.get("verdict") == "REJECTED":
                            logger.error("ingestion_rejected_by_quality_guard", reasoning=verdict.get("reasoning"))
                            return None
            except Exception as e:
                logger.warning("quality_guard_bypass_due_to_error", error=str(e))

        # Ensure layer is set (default to episodic if not provided)
        if "layer" not in kwargs:
            kwargs["layer"] = "episodic"

        # SYSTEM 40.14: Universal Ingest Pipeline (UICTC)
        from rae_core.ingestion.pipeline import UniversalIngestPipeline
        pipeline = UniversalIngestPipeline()
        
        # Process text through the 5-stage pipeline
        chunks, signature, audit_trail, policy = await pipeline.process(
            content, 
            metadata=kwargs.get("metadata")
        )
        
        if not chunks:
            return None

        # SYSTEM 92.2: Dedup Hash Check (Stop the spiral)
        if self.cache_provider and chunks:
            agent_id = kwargs.get("agent_id", "default")
            project = kwargs.get("project", "default")
            text_hash = chunks[0].metadata.get("content_hash")
            
            cache_key = f"last_mem_hash:{project}:{agent_id}"
            last_hash = await self.cache_provider.get(cache_key)
            
            if last_hash == text_hash:
                logger.info("skipping_duplicate_memory_write", project=project, agent_id=agent_id)
                return None
            
            await self.cache_provider.set(cache_key, text_hash, ttl=300) # 5 min protection

        # SYSTEM 92.3: Operational State Isolation
        # If it's a fallback, we don't want it to be highly retrievable or vectorized
        is_operational = (policy == "POLICY_FALLBACK")
        
        import uuid
        parent_id = str(uuid.uuid4())
        memory_ids = []
        
        # Store chunks with full provenance
        for i, chunk in enumerate(chunks):
            chunk_kwargs = kwargs.copy()
            chunk_kwargs["content"] = chunk.content
            chunk_kwargs["metadata"] = kwargs.get("metadata", {}).copy()
            chunk_kwargs["metadata"].update(chunk.metadata)
            chunk_kwargs["metadata"].update({
                "parent_id": parent_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "is_chunk": True,
                "ingest_audit": [a.__dict__ for a in audit_trail] if audit_trail else [],
                "is_operational": is_operational
            })
            
            # Use original source if provided, otherwise default to chunk index
            base_source = kwargs.get("source", "universal_ingest")
            chunk_kwargs["source"] = f"{base_source} [p{i+1}/{len(chunks)}]"
            
            # Operational data has zero importance for semantic retrieval
            if is_operational:
                chunk_kwargs["importance"] = 0.0
                chunk_kwargs["tags"] = chunk_kwargs.get("tags", []) + ["operational", "non_retrievable"]
            
            m_id = await self.memory_storage.store_memory(**chunk_kwargs)
            
            # Skip vector store for operational/fallback data (Anti-Echo)
            if not is_operational:
                # Embed and store vector
                embed_kwargs = chunk_kwargs.copy()
                if "content" in embed_kwargs: del embed_kwargs["content"]
                if "tenant_id" in embed_kwargs: del embed_kwargs["tenant_id"]
                
                await self._embed_and_store_vector(m_id, chunk.content, tenant_id, **embed_kwargs)
            else:
                logger.info("skipping_vector_store_for_operational_data", memory_id=str(m_id))
                
            memory_ids.append(m_id)
            
        return memory_ids[0]

    async def _embed_and_store_vector(self, m_id, content, tenant_id, **kwargs):
        if hasattr(self.embedding_provider, "generate_all_embeddings"):
            embs_dict = await self.embedding_provider.generate_all_embeddings(
                [content], task_type="search_document"
            )
            emb = {name: e[0] for name, e in embs_dict.items() if e}
        else:
            emb = await self.embedding_provider.embed_text(
                content, task_type="search_document"
            )

        vector_meta = kwargs.copy()

        await self.vector_store.store_vector(m_id, emb, tenant_id, metadata=vector_meta)

    def get_status(self) -> dict[str, Any]:
        return {
            "engine": "RAE-Core v2.9.0",
            "search_strategies": list(self.search_engine.strategies.keys()),
            "components": {
                "storage": type(self.memory_storage).__name__,
                "vector_store": type(self.vector_store).__name__,
                "embedding": type(self.embedding_provider).__name__,
            },
        }

    async def get_statistics(self, tenant_id: str = "local") -> dict[str, Any]:
        """Get memory statistics."""
        stats = {"total_count": 0, "layer_counts": {}}
        
        if hasattr(self.memory_storage, "count_memories"):
            try:
                # Count total
                total = await self.memory_storage.count_memories(tenant_id=tenant_id)
                stats["total_count"] = total
                
                # Count per layer
                for layer in ["working", "semantic", "episodic"]:
                    count = await self.memory_storage.count_memories(tenant_id=tenant_id, layer=layer)
                    stats["layer_counts"][layer] = count
            except Exception as e:
                logger.warning("get_statistics_failed", error=str(e))
        
        return stats

    async def run_reflection_cycle(self, **kwargs) -> dict[str, Any]:
        return {"status": "completed", "reflections_created": 0}

    async def generate_reflections(self, tenant_id: str, project: str) -> list[Any]:
        """Generate reflections (Lite version placeholder)."""
        # In a full version, this would call Reflection Engine.
        # For Lite/Offline, we can return empty or basic clusters.
        logger.info("generating_reflections", tenant_id=tenant_id, project=project)
        return []
