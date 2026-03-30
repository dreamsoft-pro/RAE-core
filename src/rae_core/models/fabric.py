from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
from datetime import datetime

class MessageEnvelope(BaseModel):
    message_id: str
    source_node: str
    target_node: str
    protocol_version: str = "3.0"
    payload_type: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class NodeStatus(BaseModel):
    node_id: str
    hostname: str
    is_active: bool = True
    load_factor: float = 0.0
    active_agents: int = 0
