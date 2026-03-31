"""Actor component for Reflection V2.

The Actor executes actions based on reflection insights.
"""

from typing import Any
from uuid import UUID

from rae_core.interfaces.llm import ILLMProvider
from rae_core.interfaces.storage import IMemoryStorage


class Actor:
    """Actor component that executes actions based on reflections.

    Implements the "Act" phase of the Actor-Evaluator-Reflector pattern.
    """

    def __init__(
        self,
        memory_storage: IMemoryStorage,
        llm_provider: ILLMProvider | None = None,
    ):
        """Initialize Actor.

        Args:
            memory_storage: Memory storage for persistence
            llm_provider: Optional LLM for intelligent action selection
        """
        self.memory_storage = memory_storage
        self.llm_provider = llm_provider

    async def execute_action(
        self,
        action_type: str,
        context: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Execute a specific action.

        Args:
            action_type: Type of action to execute
            context: Action context and parameters
            tenant_id: Tenant identifier

        Returns:
            Action execution result
        """
        if action_type == "consolidate_memories":
            return await self._consolidate_memories(context, tenant_id)
        elif action_type == "update_importance":
            return await self._update_importance(context, tenant_id)
        elif action_type == "create_semantic_link":
            return await self._create_semantic_link(context, tenant_id)
        elif action_type == "prune_duplicates":
            return await self._prune_duplicates(context, tenant_id)
        else:
            return {"success": False, "error": f"Unknown action: {action_type}"}

    async def _consolidate_memories(
        self,
        context: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Consolidate related memories into semantic memory.

        Args:
            context: Context with memory IDs to consolidate
            tenant_id: Tenant identifier

        Returns:
            Consolidation result
        """
        memory_ids = context.get("memory_ids", [])
        if not memory_ids:
            return {"success": False, "error": "No memories provided"}

        # Retrieve memories
        memories = []
        for mem_id in memory_ids:
            memory = await self.memory_storage.get_memory(
                memory_id=UUID(mem_id) if isinstance(mem_id, str) else mem_id,
                tenant_id=tenant_id,
            )
            if memory:
                memories.append(memory)

        if not memories:
            return {"success": False, "error": "No valid memories found"}

        # Create consolidated memory
        combined_content = " ".join([m.get("content", "") for m in memories])
        summary = (
            combined_content[:500] + "..."
            if len(combined_content) > 500
            else combined_content
        )

        new_memory_id = await self.memory_storage.store_memory(
            content=summary,
            layer="semantic",
            tenant_id=tenant_id,
            agent_id=context.get("agent_id", "system"),
            tags=["consolidated"],
            metadata={"source_memories": memory_ids},
            importance=0.8,
        )

        return {
            "success": True,
            "new_memory_id": str(new_memory_id),
            "consolidated_count": len(memories),
        }

    async def _update_importance(
        self,
        context: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Update importance scores for memories.

        Args:
            context: Context with memory IDs and new importance scores
            tenant_id: Tenant identifier

        Returns:
            Update result
        """
        updates = context.get("updates", [])
        success_count = 0

        for update in updates:
            memory_id = update.get("memory_id")
            new_importance = update.get("importance")

            if memory_id and new_importance is not None:
                success = await self.memory_storage.update_memory(
                    memory_id=(
                        UUID(memory_id) if isinstance(memory_id, str) else memory_id
                    ),
                    tenant_id=tenant_id,
                    updates={"importance": new_importance},
                )
                if success:
                    success_count += 1

        return {
            "success": True,
            "updated_count": success_count,
            "total": len(updates),
        }

    async def _create_semantic_link(
        self,
        context: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Create semantic link between memories.

        Args:
            context: Context with source and target memory IDs
            tenant_id: Tenant identifier

        Returns:
            Link creation result
        """
        source_id = context.get("source_id")
        target_id = context.get("target_id")
        relation_type = context.get("relation_type", "relates_to")

        if not source_id or not target_id:
            return {"success": False, "error": "Missing source or target ID"}

        # Store as metadata in source memory
        success = await self.memory_storage.update_memory(
            memory_id=UUID(source_id) if isinstance(source_id, str) else source_id,
            tenant_id=tenant_id,
            updates={"metadata": {f"link_{relation_type}": str(target_id)}},
        )

        return {
            "success": success,
            "link": {
                "source": str(source_id),
                "target": str(target_id),
                "type": relation_type,
            },
        }

    async def _prune_duplicates(
        self,
        context: dict[str, Any],
        tenant_id: str,
    ) -> dict[str, Any]:
        """Prune duplicate or low-value memories.

        Args:
            context: Context with memory IDs to prune
            tenant_id: Tenant identifier

        Returns:
            Pruning result
        """
        memory_ids = context.get("memory_ids", [])
        pruned_count = 0

        for mem_id in memory_ids:
            success = await self.memory_storage.delete_memory(
                memory_id=UUID(mem_id) if isinstance(mem_id, str) else mem_id,
                tenant_id=tenant_id,
            )
            if success:
                pruned_count += 1

        return {
            "success": True,
            "pruned_count": pruned_count,
            "total": len(memory_ids),
        }
