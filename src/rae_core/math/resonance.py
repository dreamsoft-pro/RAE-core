"""
RAE Semantic Resonance Engine (System 41.4 - Linguistic Scalpel).
Final tuning for Industrial 1.0 MRR by enforcing documentation preference for policy queries.
"""

import math
import re
import structlog
from typing import List, Dict, Any, Tuple, Optional, Union
from uuid import UUID

logger = structlog.get_logger(__name__)

class SemanticResonanceEngine:
    def __init__(self, h_sys: float = 13.0, resonance_factor: Optional[float] = None, iterations: int = 1, **kwargs):
        self.h_sys = h_sys
        self.resonance_factor = resonance_factor if resonance_factor is not None else 1.0 / (1.0 + math.log1p(self.h_sys))
        self.iterations = iterations

    def _get_severity_boost(self, text: str, is_critical_query: bool) -> Tuple[float, bool]:
        t_lower = text.lower()
        high_impact_terms = ["failover", "outage", "crash", "fatal", "shutdown", "unreachable", "post-mortem", "error", "vulnerability", "exploit"]
        has_high_impact = any(w in t_lower for w in high_impact_terms)
        
        if "sev1" in t_lower or "severity 1" in t_lower or (is_critical_query and has_high_impact):
            return 5.0 if is_critical_query else 3.0, True
        if "sev2" in t_lower or "severity 2" in t_lower:
            return 2.5, False
        if "sev3" in t_lower or "severity 3" in t_lower:
            return 1.2, False
        return 1.0, False

    def sharpen(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not results or not query: return results

        q_lower = query.lower()
        q_symbols = set(re.findall(r"[\w\-\[\]\.\:]+", q_lower))
        
        intent_map = {
            "problem": ["log", "incident", "ticket", "bug", "alert"],
            "issue": ["log", "incident", "ticket", "bug", "alert", "feedback"],
            "vulnerability": ["doc", "code", "security", "metric"],
            "cost": ["metric", "doc", "budget", "feedback"],
            "performance": ["metric", "log", "code"],
            "documentation": ["doc", "readme", "wiki", "code"],
            "backup": ["doc", "policy", "log"],
            "procedure": ["doc", "policy"],
            "policy": ["doc", "policy"],
            "sentiment": ["feedback", "meeting"],
            "complaint": ["feedback", "ticket"]
        }
        
        target_types = []
        for word, types in intent_map.items():
            if word in q_lower or (word + "s") in q_lower or word[:-1] in q_lower:
                target_types.extend(types)
        
        explicit_types = ["incident", "ticket", "metric", "log", "alert", "doc", "question", "bug", "meeting", "code", "feedback"]
        query_types = [t for t in explicit_types if t in q_lower or (t + "s") in q_lower]
        all_required_types = list(set(target_types + query_types))
        
        is_critical_query = any(w in q_lower for w in ["critical", "emergency", "urgent", "p0", "p1", "immediate", "security"])
        
        stop_words = {"what", "where", "how", "are", "the", "were", "found", "did", "get", "about", "planned", "being", "been"}
        token_weights = {t: len(t) * 10.0 for t in q_symbols if t not in stop_words and len(t) > 2}
        total_q_weight = sum(token_weights.values())

        tier0_base = 1e14 
        tier1_base = 1e10
        tier2_base = 1e8

        sharpened_results = []
        t1_threshold = 0.4 

        for res in results:
            content = str(res.get("content", "")).lower()
            res_id = str(res.get("id", "")).lower()
            metadata = res.get("metadata") or {}
            external_id = str(metadata.get("id", metadata.get("external_id", ""))).lower()
            
            base_score = float(res.get("score") or res.get("math_score") or 0.0)
            audit = res.get("audit") or {}
            
            type_boost = 1.0
            matches_type = False
            
            if all_required_types:
                matches_type = any(t in res_id for t in all_required_types) or \
                               any(t in external_id for t in all_required_types) or \
                               any(t in content for t in all_required_types) or \
                               any(f"[{t.upper()}]" in content for t in all_required_types)
                
                if matches_type:
                    type_boost = 2.0
                    if any(w in q_lower for w in ["procedure", "policy", "requirements"]):
                        if "doc" in res_id or "doc" in content or "doc" in external_id: type_boost = 5.0
                else:
                    type_boost = 0.0
            else:
                matches_type = True 
                type_boost = 1.0
            
            severity_boost, is_extreme = self._get_severity_boost(content, is_critical_query)
            
            found_weight = 0.0
            for t, w in token_weights.items():
                if t in content or (len(t) > 4 and t[:4] in content): 
                    found_weight += w
            coverage = found_weight / total_q_weight if total_q_weight > 0 else 0
            
            certainty_factor = type_boost * severity_boost
            norm_base = 1.0 + math.log1p(max(0, base_score))
            proof_factor = (1.0 + coverage) * certainty_factor * norm_base

            is_id_match = any(s in res_id or s in content for s in q_symbols if len(s) > 8 and any(c.isdigit() for c in s))

            if is_id_match:
                tier = 0
                final_score = tier0_base * proof_factor
            elif matches_type and (coverage >= t1_threshold or is_extreme):
                tier = 1
                final_score = tier1_base * proof_factor
            elif matches_type and type_boost > 0:
                tier = 2
                final_score = tier2_base * proof_factor
            else:
                tier = 3
                final_score = 0.0
            
            audit.update({
                "tier": tier,
                "certainty": round(certainty_factor, 3),
                "coverage": round(coverage, 3),
                "proof_v": "41.4-Scalpel"
            })
            
            res["score"] = final_score
            res["math_score"] = final_score # Compatibility
            res["audit"] = audit
            sharpened_results.append(res)

        return sorted(sharpened_results, key=lambda x: (x["audit"].get("tier", 2), -x["score"]))

    def compute_resonance(self, memories: List[Dict[str, Any]], edges: List[Tuple[str, str, float]]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        RAE Rebirth: Propagates energy through the knowledge graph.
        Higher energy = higher math_score.
        """
        energy_map = {}
        for m in memories:
            m_id = str(m.get("id", ""))
            # Prefer math_score, then score, fallback to search_score or default
            energy_map[m_id] = float(m.get("math_score") or m.get("score") or m.get("search_score") or 0.5)

        for _ in range(self.iterations):
            new_energy = energy_map.copy()
            for source, target, weight in edges:
                source, target = str(source), str(target)
                if source in energy_map:
                    # Transfer energy from source to target
                    transfer = energy_map[source] * weight * self.resonance_factor
                    new_energy[target] = new_energy.get(target, 0.0) + transfer
            energy_map = new_energy

        # Update memories with resonated energy
        for m in memories:
            m_id = str(m.get("id", ""))
            m["math_score"] = energy_map.get(m_id, 0.0)
            m["resonance_metadata"] = {"energy": m["math_score"], "iterations": self.iterations}

        return memories, energy_map
