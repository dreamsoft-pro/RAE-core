"""
Feature extraction for Math Layer Controller

Features form the observation space for decision making.
In Iteration 1: used for rule-based decisions
In Iteration 2+: become state vector for learning algorithms
"""

from dataclasses import dataclass, field
from typing import Any

from rae_core.math.types import MathLevel, TaskType


@dataclass
class Features:
    """
    Extracted features from task context for decision making.

    These features form the observation space for the controller.
    In Iteration 1, they are used for rule-based decisions.
    In later iterations, they become the state vector for learning algorithms.

    Attributes:
        task_type: Type of memory operation being performed
        memory_count: Current number of memories in the system
        graph_density: Density of knowledge graph (edges / max_edges)
        session_length: Number of turns in current session
        memory_entropy: Entropy of memory distribution (from math_metrics)
        recent_mrr: Mean Reciprocal Rank from recent queries
        recent_gcs: Graph Connectivity Score
        recent_scs: Semantic Coherence Score
        cost_budget: Remaining cost budget (USD), None if unlimited
        latency_budget_ms: Maximum allowed latency, None if unlimited
        previous_level: Level used in previous decision (for stability)
        previous_level_success: Whether previous decision was successful
        custom: Additional custom features for extensibility
    """

    # Task context
    task_type: TaskType

    # Memory state
    memory_count: int = 0
    graph_density: float = 0.0
    session_length: int = 0
    memory_entropy: float = 0.0

    # Performance metrics
    recent_mrr: float = 0.0
    recent_gcs: float = 0.0  # Graph Connectivity Score
    recent_scs: float = 0.0  # Semantic Coherence Score

    # Budget constraints
    cost_budget: float | None = None  # USD remaining
    latency_budget_ms: int | None = None  # ms

    # History (for stability)
    previous_level: MathLevel | None = None
    previous_level_success: bool | None = None

    # Extensibility
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize features for logging"""
        return {
            "task_type": self.task_type.value,
            "memory_count": self.memory_count,
            "graph_density": self.graph_density,
            "session_length": self.session_length,
            "memory_entropy": self.memory_entropy,
            "recent_mrr": self.recent_mrr,
            "recent_gcs": self.recent_gcs,
            "recent_scs": self.recent_scs,
            "cost_budget": self.cost_budget,
            "latency_budget_ms": self.latency_budget_ms,
            "previous_level": (
                self.previous_level.value if self.previous_level else None
            ),
            "previous_level_success": self.previous_level_success,
            "custom": self.custom,
        }

    def is_budget_constrained(self) -> bool:
        """Check if we have strict budget constraints"""
        return self.cost_budget is not None and self.cost_budget < 0.01

    def is_latency_constrained(self) -> bool:
        """Check if we have strict latency constraints"""
        return self.latency_budget_ms is not None and self.latency_budget_ms < 100
