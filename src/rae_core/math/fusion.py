"""
RAE Fusion Strategies (System 40.12 - Unified Tier Alignment).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from uuid import UUID
import math
import structlog

logger = structlog.get_logger(__name__)

class AbstractFusionStrategy(ABC):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def fuse(self, strategy_results, query, h_sys=13.0, memory_contents=None, weights=None, **kwargs):
        pass

class Legacy416Strategy(AbstractFusionStrategy):
    async def fuse(self, strategy_results, query, h_sys=13.0, memory_contents=None, weights=None, **kwargs):
        k = self.config.get("strategies_config", {}).get("legacy_416", {}).get("k_factor", 1.0)
        types = ["incident", "ticket", "metric", "log", "alert", "doc", "question", "bug"]
        query_types = [t for t in types if t in query.lower() or (t + "s") in query.lower()]
        
        fused_scores = {}
        for strategy, results in strategy_results.items():
            if not results: continue
            for rank, item in enumerate(results):
                m_id = item[0] if isinstance(item, tuple) else (item.get("id") or item.get("memory_id"))
                if isinstance(m_id, str):
                    try: m_id = UUID(m_id)
                    except: pass
                fused_scores[m_id] = fused_scores.get(m_id, 0.0) + (1.0 / (rank + k))

        processed = []
        for m_id, rrf_score in fused_scores.items():
            mem_obj = (memory_contents or {}).get(m_id, {})
            content = mem_obj.get("content", "").lower()
            res_id = str(m_id).lower()
            
            type_multiplier = 1.0
            tier = 2
            if query_types:
                matches_type = any(t in res_id for t in query_types) or any(t in content for t in query_types)
                if not matches_type:
                    type_multiplier = 0.001
                    tier = 3 # Trash Tier
                else:
                    type_multiplier = 2.0 # Entity Boost
            
            meta = mem_obj.get("metadata", {})
            importance = float(meta.get("importance", 0.5) if isinstance(meta, dict) else 0.5)
            processed.append((m_id, rrf_score * type_multiplier, importance, {"strategy": "Legacy416", "tier": tier, "type_penalty": tier == 3}))
        
        return sorted(processed, key=lambda x: (x[3]["tier"], -x[1]))

class SiliconOracleStrategy(AbstractFusionStrategy):
    async def fuse(self, strategy_results, query, h_sys=13.0, memory_contents=None, weights=None, **kwargs):
        divisor = self.config.get("strategies_config", {}).get("silicon_oracle", {}).get("rank_sharpening_divisor", 3.0)
        types = ["incident", "ticket", "metric", "log", "alert", "doc", "question", "bug"]
        query_types = [t for t in types if t in query.lower() or (t + "s") in query.lower()]
        
        fused_scores = {}
        actual_weights = weights or {}
        for strategy, results in strategy_results.items():
            w = actual_weights.get(strategy, 1.0)
            for rank, item in enumerate(results):
                m_id = item[0] if isinstance(item, tuple) else (item.get("id") or item.get("memory_id"))
                if isinstance(m_id, str):
                    try: m_id = UUID(m_id)
                    except: pass
                fused_scores[m_id] = fused_scores.get(m_id, 0.0) + w * math.exp(-rank / divisor)

        processed = []
        for m_id, score in fused_scores.items():
            mem_obj = (memory_contents or {}).get(m_id, {})
            res_id = str(m_id).lower()
            content = mem_obj.get("content", "").lower()
            
            type_multiplier, tier = 1.0, 2
            if query_types:
                matches_type = any(t in res_id for t in query_types) or any(t in content for t in query_types)
                if not matches_type:
                    type_multiplier, tier = 0.001, 3 # Trash Tier
                else:
                    type_multiplier = 2.0 # Entity Boost
            
            meta = mem_obj.get("metadata", {})
            importance = float(meta.get("importance", 0.5) if isinstance(meta, dict) else 0.5)
            processed.append((m_id, score * type_multiplier, importance, {"strategy": "SiliconOracle", "tier": tier, "type_penalty": tier == 3}))
            
        return sorted(processed, key=lambda x: (x[3]["tier"], -x[1]))

class FusionStrategy:
    def __init__(self, config=None):
        self.config = config or {}
        from rae_core.math.logic_gateway import LogicGateway
        self.gateway = LogicGateway(self.config)

    async def fuse(self, strategy_results=None, weights=None, query="", **kwargs):
        return await self.gateway.fuse(strategy_results=strategy_results, weights=weights, query=query, **kwargs)
