from __future__ import annotations
from typing import Any, List, Literal, Optional
from uuid import uuid4
from pydantic import BaseModel, Field

VerificationMethod = Literal["payload_check", "outcome_check", "pattern_check"]
BehaviorSignalKind = Literal["self_report", "runtime_observation", "policy_signal", "telemetry_signal"]
BehaviorSeverity = Literal["low", "medium", "high", "critical"]

class BehaviorGuarantee(BaseModel):
    guarantee_id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    verifiable: bool = True
    verification_method: VerificationMethod
    description: Optional[str] = None
    required_evidence_types: List[str] = Field(default_factory=list)

class BehaviorSignal(BaseModel):
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    department: str
    signal_kind: BehaviorSignalKind
    guarantee_id: Optional[str] = None
    trace_id: Optional[str] = None
    observed_payload: dict[str, Any] = Field(default_factory=dict)
    observed_outcome: dict[str, Any] = Field(default_factory=dict)
    reason: str
    emitted_by: str
    severity_hint: BehaviorSeverity = "medium"

class BehaviorViolation(BaseModel):
    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    department: str
    guarantee_id: str
    observed_payload: dict[str, Any] = Field(default_factory=dict)
    reason: str
    severity: BehaviorSeverity = "medium"
    source_signal_ids: List[str] = Field(default_factory=list)
    issued_by: str = "auditor"

class DepartmentBehaviorContract(BaseModel):
    department: str
    version: str
    guarantees: List[BehaviorGuarantee] = Field(default_factory=list)
    forbidden_behaviors: List[str] = Field(default_factory=list)
    required_evidence: List[str] = Field(default_factory=list)
    escalation_rules: List[str] = Field(default_factory=list)
