from pydantic import BaseModel, Field
from typing import Dict, Optional

class CostVector(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    wall_time_s: float = 0.0
    cpu_s: float = 0.0
    gpu_s: float = 0.0
    ram_gb_s: float = 0.0
    storage_gb_s: float = 0.0
    transfer_mb: float = 0.0
    external_api_currency: float = 0.0
    operator_minutes: float = 0.0

class CostPolicy(BaseModel):
    policy_id: str
    name: str
    weights: Dict[str, float] = Field(default_factory=dict)
    budget_ncu_daily: Optional[float] = None
    currency: str = "USD"

class BudgetEnvelope(BaseModel):
    policy_id: str
    daily_limit_ncu: Optional[float] = None
    monthly_limit_ncu: Optional[float] = None
    hard_stop: bool = True
