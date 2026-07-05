from __future__ import annotations
from typing import Any, List, Optional, Dict
from uuid import uuid4
from pydantic import BaseModel, Field

class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    motivation: str
    target_metric: str
    origin: str = "lab"

class Experiment(BaseModel):
    experiment_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    hypothesis_id: str
    candidate: str
    success_criteria: str

class FailurePatternPack(BaseModel):
    pack_id: str = Field(default_factory=lambda: str(uuid4()))
    version: str = "1.0.0"
    patterns: List[Dict[str, Any]]
    source_failures: List[str]

class InsightPack(BaseModel):
    pack_id: str = Field(default_factory=lambda: str(uuid4()))
    insights: List[Dict[str, Any]]
    recommendations: List[str]

class ImprovementProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    experiment_id: Optional[str] = None
    rationale: Optional[str] = None
    proposed_patch: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"
    promotion_requirements: Any = Field(default_factory=list)

class PolicyPatchProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    target_policy_id: str
    rationale: str
    proposed_patch: dict[str, Any] = Field(default_factory=dict)

class ExperimentRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    proposal_id: Optional[str] = None
    experiment_id: Optional[str] = None
    mode: str  # shadow | canary | offline
    result: str  # pass | fail
    metrics: Dict[str, Any] = Field(default_factory=dict)
    trace_id: Optional[str] = None

class PromotionDecision(BaseModel):
    proposal_id: str
    approved: bool
    reason: str

class RollbackDecision(BaseModel):
    proposal_id: str
    rollback_triggered: bool = False
    reason: str
