"""Graph models for RAE-core knowledge graph."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """Graph node type."""

    ENTITY = "entity"
    CONCEPT = "concept"
    MEMORY = "memory"
    AGENT = "agent"
    EVENT = "event"


class EdgeType(str, Enum):
    """Graph edge relationship type."""

    RELATES_TO = "relates_to"
    CAUSES = "causes"
    FOLLOWS = "follows"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"


class GraphNode(BaseModel):
    """Graph node model."""

    id: UUID
    node_type: NodeType
    properties: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphEdge(BaseModel):
    """Graph edge model."""

    source_id: UUID
    target_id: UUID
    edge_type: EdgeType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: dict[str, Any] = Field(default_factory=dict)
    tenant_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GraphPath(BaseModel):
    """Path in the graph."""

    nodes: list[UUID]
    edges: list[GraphEdge]
    total_weight: float


class Subgraph(BaseModel):
    """Extracted subgraph."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_node: UUID | None = None
