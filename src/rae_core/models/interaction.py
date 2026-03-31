from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AgentActionType(str, Enum):
    """
    Defines the permissible types of actions an agent can emit.
    RAE-First: Agents do not 'speak', they emit typed actions.
    """

    FINAL_ANSWER = "final_answer"  # The final response to the user
    THOUGHT = "thought"  # Internal reasoning (Chain of Thought)
    TOOL_CALL = "tool_call"  # Request to execute a tool
    DELEGATION = "delegation"  # Delegating sub-task to another agent
    ANALYSIS = "analysis"  # Deep analysis result
    CLARIFICATION = "clarification"  # Requesting more info from user/system


class RAEInput(BaseModel):
    """
    The standardized input payload for any Agent in the RAE ecosystem.
    Agents MUST accept this and ONLY this structure.
    """

    request_id: UUID
    tenant_id: str
    user_id: str | None = None

    # The actual content to process (could be user query or system directive)
    content: str

    # Context provided by RAE (memories, history, graph data)
    # Agents do NOT query RAE themselves; they receive context here.
    context: dict[str, Any] = Field(default_factory=dict)

    # Constraints and policies for this execution
    constraints: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class AgentAction(BaseModel):
    """
    The ONLY allowed output from an Agent.
    Wraps the 'what' and 'why' into a structured event.
    """

    type: AgentActionType

    # The main payload (e.g., the answer text, or tool arguments)
    content: Any

    # Reasoning/Justification for this action (Crucial for Reflective Memory)
    reasoning: str | None = None

    # Confidence score (0.0 - 1.0). Low confidence might trigger RAE intervention.
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Semantic signals for the Memory Policy Engine
    # e.g., ["decision", "critical", "proposal"]
    signals: list[str] = Field(default_factory=list)

    # Metadata for tools or specific extensions
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def validate_content_not_empty(cls, v):
        if v is None:
            raise ValueError("Content cannot be None")
        return v
