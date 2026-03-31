"""
RAE Math - Pure Mathematical Functions

This module provides pure mathematical functions for RAE's memory scoring
and temporal dynamics, with no external dependencies.

Modules:
    structure: Data structures (ScoringWeights, DecayConfig, MemoryScoreResult)
    dynamics: Temporal dynamics (recency calculation, decay functions)
    policy: Scoring policy (unified memory scoring, ranking)
    reasoning: Reasoning controller for graph reasoning (NEW)

License: Apache-2.0
Author: Grzegorz Leśniowski <lesniowskig@gmail.com>
"""

# Temporal dynamics
from rae_core.math.dynamics import (
    calculate_decay_rate_from_half_life,
    calculate_half_life,
    calculate_recency_score,
    ensure_utc,
)

# Scoring policy
from rae_core.math.policy import (
    compute_batch_scores,
    compute_memory_score,
    compute_score_with_custom_weights,
    rank_memories_by_score,
)

# Reasoning (NEW - Iteration 3)
from rae_core.math.reasoning import ReasoningController, ReasoningPath

# Data structures
from rae_core.math.structure import DecayConfig, MemoryScoreResult, ScoringWeights

__all__ = [
    # Data structures
    "ScoringWeights",
    "DecayConfig",
    "MemoryScoreResult",
    # Temporal dynamics
    "calculate_recency_score",
    "ensure_utc",
    "calculate_half_life",
    "calculate_decay_rate_from_half_life",
    # Scoring policy
    "compute_memory_score",
    "compute_batch_scores",
    "rank_memories_by_score",
    "compute_score_with_custom_weights",
    # Reasoning
    "ReasoningController",
    "ReasoningPath",
]
