import time
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4

class MeshPeer(BaseModel):
    peer_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    endpoint: str
    public_key: Optional[str] = None

class ConsentGrant(BaseModel):
    grant_id: str = Field(default_factory=lambda: f"gnt-{uuid4().hex[:6]}")
    peer_id: str
    scope: List[str] = Field(default_factory=list)  # e.g., ["InsightPack", "FailurePatternPack"]
    ttl_days: int = 30
    created_at: float = Field(default_factory=time.time)

class MeshExchangeEnvelope(BaseModel):
    envelope_id: str = Field(default_factory=lambda: str(uuid4()))
    source_instance: str = "rae-suite-main"
    target_peer_id: str
    pack_type: str
    pack_id: str
    consent_ref: str
    expires_at: datetime
    sensitivity_label: str = "internal"
    payload_data: Dict[str, Any] = Field(default_factory=dict)
    contract_version: str = "1.0.0"
    provenance: Dict[str, Any] = Field(default_factory=dict)
