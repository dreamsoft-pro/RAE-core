"""Reflection models for RAE-core reflection system."""

from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ReflectionType(str, Enum):
    """Type of reflection."""

    PATTERN = "pattern"
    INSIGHT = "insight"
    CONTRADICTION = "contradiction"
    TREND = "trend"
    SUMMARY = "summary"


class ReflectionPriority(str, Enum):
    """Priority of reflection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Reflection(BaseModel):
    """Reflection model."""

    id: UUID = Field(default_factory=uuid4)
    content: str
    reflection_type: ReflectionType
    priority: ReflectionPriority = Field(default=ReflectionPriority.MEDIUM)

    # Links to source memories
    source_memory_ids: list[UUID] = Field(default_factory=list)

    # Metadata
    tenant_id: str
    agent_id: str
    tags: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class ReflectionPolicy(BaseModel):
    """Policy for reflection generation."""

    min_memories: int = Field(
        default=5, description="Min memories to trigger reflection"
    )
    max_age_hours: int = Field(default=24, description="Max age before reflection")
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    enabled_types: list[ReflectionType] = Field(
        default_factory=lambda: list(ReflectionType)
    )
