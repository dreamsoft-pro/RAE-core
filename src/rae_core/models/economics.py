from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class CostRecord(BaseModel):
    transaction_id: str
    agent_id: str
    action_id: str
    ncu_consumed: float
    model_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class BudgetEnvelope(BaseModel):
    policy_id: str
    agent_id: str
    limit_ncu_daily: float
    limit_ncu_monthly: float
    current_usage_ncu: float = 0.0
    currency: str = "USD"
