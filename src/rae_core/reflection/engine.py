"""Reflection Engine V2 - Orchestrates Actor-Evaluator-Reflector pattern."""

from typing import Any, TypedDict
from uuid import UUID

from rae_core.interfaces.llm import ILLMProvider
from rae_core.interfaces.storage import IMemoryStorage
from rae_core.reflection.actor import Actor
from rae_core.reflection.evaluator import Evaluator
from rae_core.reflection.reflector import Reflector


class ReflectionCycleResult(TypedDict):
    """Structure for reflection cycle results."""

    trigger_type: str
    reflections_generated: int
    actions_executed: int
    evaluations_performed: int
    success: bool
    reason: str | None


class ReflectionEngine:
    """Reflection Engine V2 implementing Actor-Evaluator-Reflector pattern.

    Coordinates the three components:
    - Actor: Executes actions based on reflections
    - Evaluator: Assesses quality and outcomes
    - Reflector: Generates meta-cognitive insights
    """

    def __init__(
        self,
        memory_storage: IMemoryStorage,
        llm_provider: ILLMProvider | None = None,
        reflection_mode: str = "standard",
    ):
        """Initialize reflection engine.

        Args:
            memory_storage: Memory storage for persistence
            llm_provider: Optional LLM provider for intelligent reflection
            reflection_mode: "minimal", "standard" or "advanced"
        """
        self.memory_storage = memory_storage
        self.llm_provider = llm_provider

        # Initialize components
        self.actor = Actor(memory_storage, llm_provider)
        self.evaluator = Evaluator(memory_storage)
        self.reflector = Reflector(
            memory_storage, 
            llm_provider, 
            reflection_mode=reflection_mode
        )

    async def run_reflection_cycle(
        self,
        tenant_id: str,
        agent_id: str,
        trigger_type: str = "scheduled",
    ) -> ReflectionCycleResult:
        """Run a complete reflection cycle.

        Steps:
        1. Identify reflection candidates
        2. Generate reflections
        3. Execute actions based on insights
        4. Evaluate outcomes

        Args:
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            trigger_type: Type of trigger (scheduled, manual, threshold)

        Returns:
            Cycle execution summary
        """
        results: ReflectionCycleResult = {
            "trigger_type": trigger_type,
            "reflections_generated": 0,
            "actions_executed": 0,
            "evaluations_performed": 0,
            "success": True,
            "reason": None,
        }

        # Step 1: Identify candidates
        candidates = await self.reflector.identify_reflection_candidates(
            tenant_id=tenant_id,
            agent_id=agent_id,
            min_memories=5,
        )

        if not candidates:
            results["success"] = False
            results["reason"] = "No reflection candidates found"
            return results

        # Step 2: Generate reflections
        for candidate in candidates[:3]:  # Limit to 3 candidates per cycle
            memory_ids = [
                UUID(mid) if isinstance(mid, str) else mid
                for mid in candidate["memory_ids"][:10]
            ]

            reflection_result = await self.reflector.generate_reflection(
                memory_ids=memory_ids,
                tenant_id=tenant_id,
                agent_id=agent_id,
                reflection_type="consolidation",
            )

            if reflection_result.get("success"):
                results["reflections_generated"] += 1

                # Step 3: Execute actions based on reflection
                action_context = {
                    "memory_ids": [str(mid) for mid in memory_ids],
                    "agent_id": agent_id,
                    "reflection_id": reflection_result.get("reflection_id"),
                }

                action_result = await self.actor.execute_action(
                    action_type="consolidate_memories",
                    context=action_context,
                    tenant_id=tenant_id,
                )

                if action_result.get("success"):
                    results["actions_executed"] += 1

                    # Step 4: Evaluate outcome
                    await self.evaluator.evaluate_action_outcome(
                        action_type="consolidate_memories",
                        action_result=action_result,
                        context=action_context,
                    )

                    results["evaluations_performed"] += 1

        return results

    async def generate_reflection(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
        agent_id: str,
        reflection_type: str = "consolidation",
    ) -> dict[str, Any]:
        """Generate a reflection from specific memories.

        Args:
            memory_ids: List of memory IDs
            tenant_id: Tenant identifier
            agent_id: Agent identifier
            reflection_type: Type of reflection

        Returns:
            Reflection result
        """
        return await self.reflector.generate_reflection(
            memory_ids=memory_ids,
            tenant_id=tenant_id,
            agent_id=agent_id,
            reflection_type=reflection_type,
        )

    async def execute_action(
        self,
        action_type: str,
        context: dict[str, Any],
        tenant_id: str,
        evaluate: bool = True,
    ) -> dict[str, Any]:
        """Execute an action and optionally evaluate it.

        Args:
            action_type: Type of action
            context: Action context
            tenant_id: Tenant identifier
            evaluate: Whether to evaluate outcome

        Returns:
            Action result with optional evaluation
        """
        action_result = await self.actor.execute_action(
            action_type=action_type,
            context=context,
            tenant_id=tenant_id,
        )

        if evaluate and action_result.get("success"):
            evaluation = await self.evaluator.evaluate_action_outcome(
                action_type=action_type,
                action_result=action_result,
                context=context,
            )
            action_result["evaluation"] = evaluation

        return action_result

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
            Dict containing L1, L2, L3 analysis and final decision.
        """
        return await self.reflector.evaluate_answer_draft(
            query_id=query_id,
            retrieved_sources=retrieved_sources,
            answer_draft=answer_draft,
            tenant_id=tenant_id,
            agent_id=agent_id,
            metadata=metadata
        )

    async def evaluate_memory_quality(
        self,
        memory_id: UUID,
        tenant_id: str,
        context: str | None = None,
    ) -> dict[str, float]:
        """Evaluate quality of a memory.

        Args:
            memory_id: Memory identifier
            tenant_id: Tenant identifier
            context: Optional context

        Returns:
            Quality metrics
        """
        return await self.evaluator.evaluate_memory_quality(
            memory_id=memory_id,
            tenant_id=tenant_id,
            context=context,
        )

    async def identify_low_quality_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        quality_threshold: float = 0.4,
        limit: int = 50,
    ) -> list[UUID]:
        """Identify low-quality memories for pruning.

        Args:
            tenant_id: Tenant identifier
            agent_id: Optional agent filter
            quality_threshold: Minimum quality threshold
            limit: Maximum memories to check

        Returns:
            List of low-quality memory IDs
        """
        memories = await self.memory_storage.list_memories(
            tenant_id=tenant_id,
            agent_id=agent_id,
            limit=limit,
        )

        low_quality = []
        for memory in memories:
            memory_id = memory["id"]
            if isinstance(memory_id, str):
                memory_id = UUID(memory_id)

            metrics = await self.evaluator.evaluate_memory_quality(
                memory_id=memory_id,
                tenant_id=tenant_id,
            )

            if metrics.get("quality", 1.0) < quality_threshold:
                low_quality.append(memory_id)

        return low_quality

    async def prune_low_quality_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        quality_threshold: float = 0.4,
    ) -> dict[str, Any]:
        """Identify and prune low-quality memories.

        Args:
            tenant_id: Tenant identifier
            agent_id: Optional agent filter
            quality_threshold: Quality threshold

        Returns:
            Pruning result
        """
        low_quality_ids = await self.identify_low_quality_memories(
            tenant_id=tenant_id,
            agent_id=agent_id,
            quality_threshold=quality_threshold,
        )

        if not low_quality_ids:
            return {
                "success": True,
                "pruned_count": 0,
                "reason": "No low-quality memories found",
            }

        return await self.actor.execute_action(
            action_type="prune_duplicates",
            context={
                "memory_ids": [str(mid) for mid in low_quality_ids],
                "agent_id": agent_id,
            },
            tenant_id=tenant_id,
        )
