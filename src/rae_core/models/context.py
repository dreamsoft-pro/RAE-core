"""Context models for RAE-core working memory."""

from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class ContextWindow(BaseModel):
    """Context window for working memory."""

    max_tokens: int = Field(default=4096, description="Maximum tokens in window")
    current_tokens: int = Field(default=0, description="Current token count")
    items: list[UUID] = Field(
        default_factory=list, description="Memory item IDs in window"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Window metadata"
    )


class WorkingContext(BaseModel):
    """Working context model for active processing."""

    tenant_id: str = Field(description="Tenant identifier")
    agent_id: str | None = Field(default=None, description="Agent identifier")

    window: ContextWindow = Field(
        default_factory=ContextWindow, description="Context window"
    )

    focus_items: list[UUID] = Field(
        default_factory=list, description="Currently focused memory items"
    )

    priority_score: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Context priority"
    )

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )


class ContextMetadata(BaseModel):
    """Metadata for context management."""

    total_items: int = Field(default=0, description="Total items in context")
    active_items: int = Field(default=0, description="Active items")
    token_usage: int = Field(default=0, description="Current token usage")
    last_compaction: datetime | None = Field(
        default=None, description="Last context compaction time"
    )
    statistics: dict[str, Any] = Field(
        default_factory=dict, description="Context statistics"
    )
