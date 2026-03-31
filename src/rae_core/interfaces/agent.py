from abc import ABC, abstractmethod
from typing import NoReturn

from rae_core.models.interaction import AgentAction, RAEInput


class BaseAgent(ABC):
    """
    RAE-First Agent Contract.
    Enforces the architecture where Agents are pure cognitive functions.
    """

    @abstractmethod
    async def run(self, input_payload: RAEInput) -> AgentAction:
        """
        The ONLY entry point for agent execution.
        Must return a structured AgentAction.
        Returning strings or interacting with IO directly is forbidden.
        """
        pass

    # --- Forbidden Methods (Architectural Safeguards) ---
    # These raise errors to prevent developers/LLMs from slipping into old habits.

    def save_memory(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            "VIOLATION: Agents CANNOT save memory. "
            "Emit an AgentAction, and RAE will handle persistence."
        )

    def print(self, *args, **kwargs) -> NoReturn:
        raise NotImplementedError(
            "VIOLATION: Agents CANNOT use print(). "
            "Use structlog or emit an AgentAction."
        )

    def respond(self, text: str) -> NoReturn:
        raise NotImplementedError(
            "VIOLATION: Agents CANNOT return raw strings. "
            "Return AgentAction(type='final_answer', content=...)."
        )
