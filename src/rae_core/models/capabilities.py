from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AgentCapability(BaseModel):
    name: str
    description: str
    version: str
    parameters_schema: Dict[str, Any] = Field(default_factory=dict)

class AgentRegistration(BaseModel):
    agent_id: str
    department: str
    role: str
    capabilities: List[AgentCapability]
    endpoint: Optional[str] = None
    status: str = "active"
