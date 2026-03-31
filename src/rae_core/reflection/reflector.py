"""Reflector component for Reflection V2.

The Reflector generates insights and reflection patterns from memories.
"""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from rae_core.interfaces.llm import ILLMProvider
from rae_core.interfaces.storage import IMemoryStorage
from rae_core.reflection.layers import ReflectionCoordinator


class Reflector:
    """Reflector component that generates meta-cognitive insights.

    Implements the "Reflect" phase of the Actor-Evaluator-Reflector pattern.
    Now enriched with 3-Layer QFT-inspired Structural Reflection.
    """

    def __init__(
        self,
        memory_storage: IMemoryStorage,
        llm_provider: ILLMProvider | None = None,
        reflection_mode: str = "standard",
    ):
        """Initialize Reflector.

        Args:
            memory_storage: Memory storage for retrieval
            llm_provider: Optional LLM for intelligent reflection
            reflection_mode: "minimal", "standard" or "advanced"
        """
        self.memory_storage = memory_storage
        self.llm_provider = llm_provider
        self.coordinator = ReflectionCoordinator(
            mode=reflection_mode, 
            enforce_hard_frames=True,
            storage=memory_storage
        )

    async def evaluate_answer_draft(
        self,
        query_id: str,
        retrieved_sources: list[Any],
        answer_draft: str,
        tenant_id: str,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluates an answer draft using the 3-Layer Reflection Architecture.

        Args:
            query_id: Identifier of the query.
            retrieved_sources: Sources fetched to construct the answer.
            answer_draft: The LLM generated response to be audited.
            tenant_id: Tenant identifier for storage.
            agent_id: Optional agent identifier.
            metadata: Contextual data.

        Returns:
            Dict containing L1, L2, L3 analysis and final decision (block/pass).
        """
        payload = {
            "query_id": query_id,
            "retrieved_sources": retrieved_sources,
            "answer_draft": answer_draft,
            "metadata": metadata or {}
        }
        
        return await self.coordinator.run_and_store_reflections(
            payload=payload,
            tenant_id=tenant_id,
            agent_id=agent_id
        )

    async def generate_reflection(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
        agent_id: str,
        reflection_type: str = "consolidation",
    ) -> dict[str, Any]:
        """Generate a reflection from memories.

        Args:
            memory_ids: List of memory IDs to reflect on
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            reflection_type: Type of reflection (consolidation, pattern, insight)

        Returns:
            Generated reflection
        """
        # Retrieve memories
        memories = []
        for mem_id in memory_ids:
            memory = await self.memory_storage.get_memory(mem_id, tenant_id)
            if memory:
                memories.append(memory)

        if not memories:
            return {"success": False, "error": "No valid memories found"}

        # --- NOISE GATE (RST Optimization) ---
        # Filter out low-importance memories (noise) to improve stability
        # Only apply if we have enough memories to spare
        if len(memories) > 3:
            filtered_memories = [m for m in memories if m.get("importance", 0.0) >= 0.2]
            # Fallback if filtering removed too many
            if len(filtered_memories) >= 2:
                memories = filtered_memories
        # -------------------------------------

        # Generate reflection based on type
        if reflection_type == "consolidation":
            return await self._generate_consolidation_reflection(
                memories, tenant_id, agent_id
            )
        elif reflection_type == "pattern":
            return await self._generate_pattern_reflection(
                memories, tenant_id, agent_id
            )
        elif reflection_type == "insight":
            return await self._generate_insight_reflection(
                memories, tenant_id, agent_id
            )
        else:
            return {
                "success": False,
                "error": f"Unknown reflection type: {reflection_type}",
            }

    async def _generate_consolidation_reflection(
        self,
        memories: list[dict[str, Any]],
        tenant_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Generate consolidation reflection."""
        # Extract common themes
        all_content = " ".join([m.get("content", "") for m in memories])
        all_tags = []
        for m in memories:
            all_tags.extend(m.get("tags", []))

        # Find common tags
        tag_counts: dict[str, int] = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        common_tags = [tag for tag, count in tag_counts.items() if count >= 2]

        # Generate reflection content
        if self.llm_provider:
            f"Consolidate these {len(memories)} memory fragments into a coherent insight:\n{all_content[:1000]}"
            reflection_content = await self.llm_provider.summarize(
                all_content, max_length=300
            )
        else:
            # Rule-based consolidation
            reflection_content = f"Consolidated {len(memories)} memories with common themes: {', '.join(common_tags[:5]) if common_tags else 'various topics'}"

        # Store reflection
        reflection_id = await self.memory_storage.store_memory(
            content=reflection_content,
            layer="reflective",
            tenant_id=tenant_id,
            agent_id=agent_id,
            tags=["reflection", "consolidation"] + common_tags[:3],
            metadata={
                "reflection_type": "consolidation",
                "source_memory_count": len(memories),
                "source_memory_ids": [str(m["id"]) for m in memories],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            importance=0.9,
        )

        return {
            "success": True,
            "reflection_id": str(reflection_id),
            "type": "consolidation",
            "content": reflection_content,
        }

    async def _generate_pattern_reflection(
        self,
        memories: list[dict[str, Any]],
        tenant_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Generate pattern recognition reflection."""
        # Detect patterns in memories
        patterns = []

        # Temporal patterns
        timestamps = [m.get("created_at") for m in memories if m.get("created_at")]
        if len(timestamps) >= 3:
            patterns.append("temporal_clustering")

        # Tag patterns
        all_tags = []
        for m in memories:
            all_tags.extend(m.get("tags", []))

        tag_freq: dict[str, int] = {}
        for tag in all_tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1

        if tag_freq:
            most_common = max(tag_freq.items(), key=lambda x: x[1])
            if most_common[1] >= 3:
                patterns.append(f"tag_pattern:{most_common[0]}")

        # Generate reflection
        pattern_desc = ", ".join(patterns) if patterns else "no clear patterns"
        reflection_content = (
            f"Detected patterns across {len(memories)} memories: {pattern_desc}"
        )

        reflection_id = await self.memory_storage.store_memory(
            content=reflection_content,
            layer="reflective",
            tenant_id=tenant_id,
            agent_id=agent_id,
            tags=["reflection", "pattern"] + patterns[:3],
            metadata={
                "reflection_type": "pattern",
                "patterns": patterns,
                "source_memory_count": len(memories),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            importance=0.85,
        )

        return {
            "success": True,
            "reflection_id": str(reflection_id),
            "type": "pattern",
            "patterns": patterns,
            "content": reflection_content,
        }

    async def _generate_insight_reflection(
        self,
        memories: list[dict[str, Any]],
        tenant_id: str,
        agent_id: str,
    ) -> dict[str, Any]:
        """Generate high-level insight reflection."""
        # Analyze memory importance distribution
        importances = [m.get("importance", 0.5) for m in memories]
        avg_importance = sum(importances) / len(importances) if importances else 0.5

        # Analyze layers
        layers = [m.get("layer", "unknown") for m in memories]
        layer_dist: dict[str, int] = {}
        for layer in layers:
            layer_dist[layer] = layer_dist.get(layer, 0) + 1

        # Generate insight
        if self.llm_provider:
            all_content = " ".join([m.get("content", "") for m in memories[:10]])
            prompt = f"Generate a meta-cognitive insight from these memories:\n{all_content[:800]}"
            try:
                insight_content = await self.llm_provider.generate(
                    prompt, max_tokens=200
                )
            except Exception:
                insight_content = f"Key insight: {len(memories)} memories with average importance {avg_importance:.2f}"
        else:
            insight_content = f"Analyzed {len(memories)} memories across layers: {', '.join(layer_dist.keys())}"

        reflection_id = await self.memory_storage.store_memory(
            content=insight_content,
            layer="reflective",
            tenant_id=tenant_id,
            agent_id=agent_id,
            tags=["reflection", "insight", "meta-cognitive"],
            metadata={
                "reflection_type": "insight",
                "avg_importance": avg_importance,
                "layer_distribution": layer_dist,
                "source_memory_count": len(memories),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
            importance=0.95,
        )

        return {
            "success": True,
            "reflection_id": str(reflection_id),
            "type": "insight",
            "content": insight_content,
            "metrics": {
                "avg_importance": avg_importance,
                "layer_distribution": layer_dist,
            },
        }

    async def identify_reflection_candidates(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        min_memories: int = 5,
        max_age_hours: int = 24,
    ) -> list[dict[str, Any]]:
        """Identify memories that are candidates for reflection.

        Args:
            tenant_id: Tenant identifier
            agent_id: Optional agent filter
            min_memories: Minimum memories to trigger reflection
            max_age_hours: Maximum age in hours

        Returns:
            List of memory groups suitable for reflection
        """
        # Get recent memories
        memories = await self.memory_storage.list_memories(
            tenant_id=tenant_id,
            agent_id=agent_id,
            limit=100,
        )

        if len(memories) < min_memories:
            return []

        # Group by tags
        tag_groups: dict[str, list[UUID]] = {}
        for memory in memories:
            tags = memory.get("tags", [])
            for tag in tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(memory["id"])

        # Find groups with enough memories
        candidates = []
        for tag, mem_ids in tag_groups.items():
            if len(mem_ids) >= min_memories:
                candidates.append(
                    {
                        "tag": tag,
                        "memory_ids": mem_ids,
                        "count": len(mem_ids),
                        "type": "tag_group",
                    }
                )

        return candidates
