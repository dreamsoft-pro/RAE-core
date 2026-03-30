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
    rationale: str
    proposed_patch: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"
    promotion_requirements: List[str] = Field(default_factory=list)

class PolicyPatchProposal(BaseModel):
    proposal_id: str = Field(default_factory=lambda: str(uuid4()))
    target_policy_id: str
    rationale: str
    proposed_patch: dict[str, Any] = Field(default_factory=dict)

class ExperimentRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    proposal_id: str
    mode: str  # shadow | canary | offline
    result: str  # pass | fail

class PromotionDecision(BaseModel):
    proposal_id: str
    approved: bool
    reason: str

class RollbackDecision(BaseModel):
    proposal_id: str
    reason: str
