import structlog

from rae_core.interfaces.agent import BaseAgent
from rae_core.interfaces.storage import IMemoryStorage
from rae_core.models.interaction import AgentAction, AgentActionType, RAEInput

logger = structlog.get_logger(__name__)


class RAERuntime:
    """
    The Operating System for Agents.
    Orchestrates the lifecycle: Input -> Agent -> Action -> Memory -> Output.
    """

    def __init__(self, storage: IMemoryStorage, agent: BaseAgent | None = None):
        self.storage = storage
        self.agent = agent

    async def process(self, input_payload: RAEInput) -> AgentAction:
        """
        Executes the agent within the RAE boundaries.
        Enforces memory persistence and policy checks.
        """
        if not self.agent:
            raise RuntimeError("No agent configured for Runtime")

        logger.info("rae_runtime_start", request_id=str(input_payload.request_id))

        # 1. Execute Agent (Pure Function)
        try:
            action = await self.agent.run(input_payload)
        except Exception as e:
            logger.error("agent_execution_failed", error=str(e))
            raise RuntimeError(f"Agent execution failed: {e}")

        # 2. Validation (Architecture Enforcement)
        if not isinstance(action, AgentAction):
            raise TypeError(
                f"Agent returned {type(action)} instead of AgentAction. "
                "Direct string return is FORBIDDEN."
            )

        # 3. Memory Hook (The "Side Effect")
        # Agent doesn't know this happens.
        await self._handle_memory_policy(input_payload, action)

        logger.info(
            "rae_runtime_success", action_type=action.type, confidence=action.confidence
        )

        return action

    async def _handle_memory_policy(self, input_payload: RAEInput, action: AgentAction):
        """
        Decides if and how to store the action in memory.
        Enforces "Implicit Capture" policy.
        """
        agent_id = input_payload.context.get("agent_id", "agent-runtime")
        project = input_payload.context.get("project")
        session_id = input_payload.context.get("session_id")

        # Base tags and metadata
        base_tags = ["rae-first", f"action-{action.type.value}"] + action.signals
        base_metadata = {
            "request_id": str(input_payload.request_id),
            "confidence": action.confidence,
            "reasoning": action.reasoning,
            "input_preview": input_payload.content[:100],
        }

        # Policy: Capture EVERYTHING significant (RAE-First Enforcement)
        # 1. Final Answers -> Episodic (Knowledge) or Working (Operational)
        if action.type == AgentActionType.FINAL_ANSWER:
            # SYSTEM 92.4: Fallback isolation in Runtime
            is_fallback = "fallback" in action.signals
            target_layer = "working" if is_fallback else "episodic"
            
            logger.info("memory_policy_triggered", 
                        rule="final_answer_store", 
                        is_fallback=is_fallback,
                        layer=target_layer)
                        
            await self.storage.store_memory(
                content=str(action.content),
                layer=target_layer,
                tenant_id=input_payload.tenant_id,
                agent_id=agent_id,
                tags=base_tags + (["final_answer"] if not is_fallback else ["operational_fallback"]),
                metadata=base_metadata,
                project=project,
                session_id=session_id,
                source="RAERuntime",
            )

        # 2. Thoughts & Decisions -> Working
        elif action.type == AgentActionType.THOUGHT:
            logger.info("memory_policy_triggered", rule="cognitive_trace_store")
            await self.storage.store_memory(
                content=f"Reasoning: {action.reasoning} | Output: {str(action.content)}",
                layer="working",
                tenant_id=input_payload.tenant_id,
                agent_id=agent_id,
                tags=base_tags + ["trace"],
                metadata=base_metadata,
                project=project,
                session_id=session_id,
                source="RAERuntime",
            )

        # 3. Tool Invocations -> Working (The Audit Trail)
        elif action.type == AgentActionType.TOOL_CALL:
            logger.info("memory_policy_triggered", rule="tool_audit_store")
            await self.storage.store_memory(
                content=f"Tool Call: {action.content} | Reasoning: {action.reasoning}",
                layer="working",
                tenant_id=input_payload.tenant_id,
                agent_id=agent_id,
                tags=base_tags + ["audit", "tool_call"],
                metadata=base_metadata,
                project=project,
                session_id=session_id,
                source="RAERuntime",
            )
