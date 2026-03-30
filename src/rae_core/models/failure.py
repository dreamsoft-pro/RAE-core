from pydantic import BaseModel, Field
from typing import Optional
import uuid

class FailureLearningRecord(BaseModel):
    failure_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    failure_type: str
    failure_stage: str
    impact_scope: str = "local"
    wasted_cost: Optional[float] = None
    root_cause_hypothesis: Optional[str] = None
    lesson_learned: Optional[str] = None
    future_guardrail: Optional[str] = None
    retry_recommendation: Optional[str] = None
