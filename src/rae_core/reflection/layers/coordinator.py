from typing import Any, Dict, List, Optional, Callable
import uuid
import structlog
from rae_core.reflection.layers.l1_operational import L1OperationalReflection
from rae_core.reflection.layers.l2_structural import L2StructuralReflection
from rae_core.reflection.layers.l3_meta import L3MetaFieldReflection
from rae_core.reflection.layers.l4_cognitive import L4CognitiveReflection
from rae_core.interfaces.storage import IMemoryStorage

logger = structlog.get_logger(__name__)

class ReflectionCoordinator:
    """
    Coordinates L1, L2, L3 (Deterministic) and L4 (Cognitive) reflections.
    Ensures Hard Frames enforcement while allowing optional LLM-based insights.
    """
    def __init__(
        self, 
        mode: str = "standard", 
        enforce_hard_frames: bool = True,
        storage: IMemoryStorage | None = None,
        llm_provider: Any = None,
        llm_model: str = "ollama/qwen3.5:9b",
        strategy: str = "math"
    ):
        self.mode = mode.lower()
        self.enforce_hard_frames = enforce_hard_frames
        self.storage = storage
        self.strategy = strategy
        
        self.l1 = L1OperationalReflection()
        self.l2 = L2StructuralReflection()
        self.l3 = L3MetaFieldReflection()
        self.l4 = L4CognitiveReflection(llm_provider=llm_provider, model_name=llm_model)

    async def run_reflections(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs active reflection layers.
        """
        result = {
            "l1_operational": {},
            "l2_structural": {},
            "l3_meta_field": {},
            "l4_cognitive": {},
            "final_decision": "pass",
            "block_reasons": []
        }
        
        l1_res = self.l1.reflect(payload)
        result["l1_operational"] = l1_res
        
        if self.enforce_hard_frames and l1_res.get("block"):
            result["final_decision"] = "blocked"
            result["block_reasons"].append("L1 Operational constraints violated")
            logger.warning("reflection_blocked_l1", reasons=result["block_reasons"])

        if self.mode in ["standard", "advanced"] and result["final_decision"] == "pass":
            l2_res = self.l2.reflect(payload)
            result["l2_structural"] = l2_res

        if self.mode == "advanced" and result["final_decision"] == "pass":
            l3_res = self.l3.reflect(payload)
            result["l3_meta_field"] = l3_res
            if self.enforce_hard_frames and l3_res.get("block"):
                result["final_decision"] = "blocked"
                result["block_reasons"].append("L3 Meta-Field anomalies critical")
                logger.warning("reflection_blocked_l3", reasons=result["block_reasons"])

        if self.strategy == "hybrid" and result["final_decision"] == "pass":
            logger.info("l4_reflection_starting", strategy=self.strategy, model=self.l4.model_name)
            l4_res = await self.l4.reflect(payload)
            result["l4_cognitive"] = l4_res
            logger.info("l4_reflection_finished", status=l4_res.get("status"))

        return result

    async def run_and_store_reflections(
        self, 
        payload: Dict[str, Any], 
        tenant_id: str, 
        agent_id: str | None = None,
        store_callback: Optional[Callable] = None # NEW: Support for vectored storage
    ) -> Dict[str, Any]:
        """
        Runs reflections and persists result to the audit table.
        """
        result = await self.run_reflections(payload)
        
        if self.storage:
            query_id = payload.get("query_id", str(uuid.uuid4()))
            fsi = result.get("l3_meta_field", {}).get("field_stability_index", 1.0)
            
            await self.storage.store_reflection_audit(
                query_id=query_id,
                tenant_id=tenant_id,
                agent_id=agent_id,
                fsi_score=fsi,
                final_decision=result["final_decision"],
                l1_report=result["l1_operational"],
                l2_report=result["l2_structural"],
                l3_report=result["l3_meta_field"],
                metadata={
                    **payload.get("metadata", {}),
                    "l4_report": result["l4_cognitive"],
                    "strategy": self.strategy
                }
            )
            
            # Hybrid Vector Storage Support - Executed in background to avoid DB locks
            if result["l4_cognitive"].get("status") == "success":
                import asyncio
                insight = result["l4_cognitive"]["insight"]
                lesson_content = f"LESSON LEARNED: {insight['lesson']}"
                
                # Background storage task
                async def store_lesson():
                    try:
                        if store_callback:
                            await store_callback(
                                content=lesson_content,
                                layer="reflective",
                                tenant_id=tenant_id,
                                agent_id=agent_id or "l4_sage",
                                tags=insight.get("tags", []) + ["l4_lesson"],
                                metadata={"confidence": insight.get("confidence"), "origin": "l4_cognitive"},
                                project=payload.get("metadata", {}).get("project")
                            )
                        else:
                            await self.storage.store_memory(
                                content=lesson_content,
                                layer="reflective",
                                tenant_id=tenant_id,
                                agent_id=agent_id or "l4_sage",
                                tags=insight.get("tags", []) + ["l4_lesson"],
                                metadata={"confidence": insight.get("confidence"), "origin": "l4_cognitive"},
                                project=payload.get("metadata", {}).get("project")
                            )
                    except Exception as e:
                        logger.warning("failed_to_store_l4_lesson", error=str(e))

                asyncio.create_task(store_lesson())
            
        return result
