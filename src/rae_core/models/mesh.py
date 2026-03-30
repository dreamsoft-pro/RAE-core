from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from uuid import uuid4

class MeshPeer(BaseModel):
    peer_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    endpoint: str

class MeshExchangeEnvelope(BaseModel):
    envelope_id: str = Field(default_factory=lambda: str(uuid4()))
    pack_type: str
    contract_version: str = "1.0.0"
