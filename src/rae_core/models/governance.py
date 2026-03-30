from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class QualityGate(BaseModel):
    gate_id: str
    name: str
    required_coverage: float = 0.8
    allow_warnings: bool = False
    enforcement_mode: str = "strict"  # "strict" | "advisory"

class PolicyRule(BaseModel):
    rule_id: str
    description: str
    severity: str = "high"
    auto_fix: bool = False

class GovernancePolicy(BaseModel):
    policy_id: str
    version: str
    rules: List[PolicyRule]
    gates: List[QualityGate]
    updated_at: datetime = Field(default_factory=datetime.utcnow)
