"""
RAE Universal Ingest Interfaces.
Defines the contracts for the 5-stage pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

@dataclass
class IngestAudit:
    """Audit trail for a single ingest step."""
    stage: str
    action: str
    trace: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContentSignature:
    """Universal signature of input content."""
    struct: Dict[str, Any]  # Structural features (S-Layer)
    dist: Dict[str, Any]    # Distributional features (D-Layer)
    stab: Dict[str, Any]    # Stability features (O-Layer)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "struct": self.struct,
            "dist": self.dist,
            "stab": self.stab
        }

class INormalizer(ABC):
    @abstractmethod
    def normalize(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> tuple[str, IngestAudit]:
        pass

class ISignatureDetector(ABC):
    @abstractmethod
    def detect(self, text: str) -> tuple[ContentSignature, IngestAudit]:
        pass

class IPolicySelector(ABC):
    @abstractmethod
    def select_policy(self, signature: ContentSignature) -> tuple[str, IngestAudit]:
        pass

@dataclass
class IngestChunk:
    content: str
    metadata: Dict[str, Any]
    offset: int
    length: int

class ISegmenter(ABC):
    @abstractmethod
    async def segment(self, text: str, policy: str, signature: ContentSignature) -> tuple[List[IngestChunk], IngestAudit]:
        pass

class ICompressor(ABC):
    @abstractmethod
    def compress(self, chunks: List[IngestChunk], policy: str) -> tuple[List[IngestChunk], Dict[str, Any], IngestAudit]:
        pass
