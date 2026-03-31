"""
RAE Math - Structural Components

This module defines the core data structures used in RAE's mathematical
formulations:
- Configuration classes for scoring parameters
- Result containers for scoring outputs

These are pure data structures with no external dependencies.

License: Apache-2.0
Author: Grzegorz Leśniowski <lesniowskig@gmail.com>
"""

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoringWeights:
    """
    Weights for combining scoring components in unified memory scoring.

    The unified memory score combines three components:
        score = alpha * relevance + beta * importance + gamma * recency

    Default weights (optimized for general RAE usage):
    - alpha (relevance): 0.5 - Semantic similarity is most important
    - beta (importance): 0.3 - Content importance matters
    - gamma (recency): 0.2 - Time decay has moderate impact

    Weights should sum to 1.0 for normalized scores in [0, 1].

    Attributes:
        alpha: Weight for relevance/similarity component (0.0-1.0)
        beta: Weight for importance component (0.0-1.0)
        gamma: Weight for recency/temporal component (0.0-1.0)

    Mathematical notation:
        S(m, q, t) = α·sim(m, q) + β·imp(m) + γ·rec(m, t)
        where α + β + γ = 1

    Example:
        >>> # Default balanced weights
        >>> weights = ScoringWeights()
        >>> print(weights.alpha, weights.beta, weights.gamma)
        0.5 0.3 0.2

        >>> # Custom weights favoring recency
        >>> recent_weights = ScoringWeights(alpha=0.3, beta=0.2, gamma=0.5)
    """

    alpha: float = 0.5  # Relevance weight
    beta: float = 0.3  # Importance weight
    gamma: float = 0.2  # Recency weight

    @classmethod
    def szubar_profile(cls) -> "ScoringWeights":
        """
        Get weights for Szubar Mode.
        Focuses on Importance (Pressure) and Usage over simple Similarity.
        """
        return cls(alpha=0.3, beta=0.5, gamma=0.2)

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0 (with small tolerance for float precision)"""
        total = self.alpha + self.beta + self.gamma
        if not math.isclose(total, 1.0, rel_tol=1e-5):
            import warnings

            warnings.warn(
                f"ScoringWeights do not sum to 1.0: {total:.6f}. "
                f"Consider normalizing: alpha={self.alpha}, beta={self.beta}, gamma={self.gamma}",
                UserWarning,
            )

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary"""
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}


@dataclass
class DecayConfig:
    """
    Configuration for recency decay calculation in temporal scoring.

    Implements exponential decay with access count consideration:
        recency_score = exp(-effective_decay * time_elapsed)

    Where:
        effective_decay = base_decay_rate / (log(1 + access_count) + 1)

    This means frequently accessed memories decay slower (remain relevant longer).

    Attributes:
        base_decay_rate: Base decay rate per second (default: 0.001 = ~50% after 11.5 min)
        access_count_boost: Whether to reduce decay for frequently accessed memories
        min_decay_rate: Minimum effective decay rate (for very frequently accessed)
        max_decay_rate: Maximum effective decay rate (for rarely accessed)

    Mathematical properties:
        - Half-life at base_decay: t_1/2 = ln(2) / base_decay_rate
        - For base_decay=0.001: t_1/2 ≈ 693 seconds ≈ 11.5 minutes
        - For base_decay=0.0001: t_1/2 ≈ 6930 seconds ≈ 1.9 hours

    Example:
        >>> # Standard decay (11.5 min half-life)
        >>> config = DecayConfig()
        >>> print(config.base_decay_rate)
        0.001

        >>> # Slower decay for long-term memory
        >>> ltm_config = DecayConfig(base_decay_rate=0.0001)
    """

    base_decay_rate: float = 0.001  # Base decay rate per second
    access_count_boost: bool = True  # Enable access count consideration
    min_decay_rate: float = 0.0001  # Minimum effective decay (for frequent access)
    max_decay_rate: float = 0.01  # Maximum effective decay (for rare access)

    def to_dict(self) -> dict[str, float]:
        """Serialize to dictionary"""
        return {
            "base_decay_rate": self.base_decay_rate,
            "access_count_boost": self.access_count_boost,
            "min_decay_rate": self.min_decay_rate,
            "max_decay_rate": self.max_decay_rate,
        }


@dataclass
class MemoryScoreResult:
    """
    Result container for unified memory scoring.

    Contains the final score and detailed breakdown of components for
    interpretability and debugging.

    Attributes:
        final_score: Weighted combination of all components (0.0-1.0)
        relevance_score: Semantic similarity component (0.0-1.0)
        importance_score: Content importance component (0.0-1.0)
        recency_score: Temporal recency component (0.0-1.0)
        memory_id: Identifier of the scored memory
        age_seconds: Age of memory in seconds since last access/creation
        access_count: Number of times memory was accessed
        effective_decay_rate: Actual decay rate used (after access count adjustment)

    Mathematical formulation:
        final_score = α·relevance + β·importance + γ·recency
        where recency = exp(-effective_decay * age_seconds)

    Example:
        >>> result = MemoryScoreResult(
        ...     final_score=0.82,
        ...     relevance_score=0.85,
        ...     importance_score=0.75,
        ...     recency_score=0.90,
        ...     memory_id="mem_123",
        ...     age_seconds=600.0,
        ...     access_count=5,
        ...     effective_decay_rate=0.0004
        ... )
        >>> print(f"Score: {result.final_score:.2f}")
        Score: 0.82
    """

    # Final score
    final_score: float

    # Component scores (for interpretability)
    relevance_score: float
    importance_score: float
    recency_score: float

    # Metadata
    memory_id: str | None
    age_seconds: float
    access_count: int
    effective_decay_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "final_score": self.final_score,
            "relevance_score": self.relevance_score,
            "importance_score": self.importance_score,
            "recency_score": self.recency_score,
            "memory_id": self.memory_id,
            "age_seconds": self.age_seconds,
            "access_count": self.access_count,
            "effective_decay_rate": self.effective_decay_rate,
        }


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Formula: (A · B) / (||A|| * ||B||)
    """
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = sum(a * a for a in vec1) ** 0.5
    norm_b = sum(b * b for b in vec2) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))
