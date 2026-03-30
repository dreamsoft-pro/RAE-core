from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Any
import uuid
from rae_core.models.cost import CostVector

class ActionRecord(BaseModel):
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    department: str
    role: str
    action_type: str
    goal: str
    tools_used: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    trace_id: str

class DecisionEvidenceRecord(BaseModel):
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_summary: str
    reasoning_summary: str
    policy_basis: Optional[str] = None
    input_refs: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    cost: Optional[float] = None
    cost_vector: Optional[CostVector] = None

class OutcomeRecord(BaseModel):
    outcome_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_id: str
    result: str  # success | failure | partial
    quality_result: Optional[str] = None
    latency_ms: Optional[int] = None
    cost: Optional[float] = None
    observed_effect: Optional[str] = None
    cost_vector: Optional[CostVector] = None
