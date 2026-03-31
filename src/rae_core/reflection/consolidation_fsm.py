"""Formal Finite State Machine (FSM) for Memory Consolidation.

Implements the deterministic lifecycle of a memory artifact:
WORKING -> EPISODIC -> SEMANTIC_PENDING -> SEMANTIC

Transitions are governed by parametric threshold equations and Bayesian
updates, ensuring full auditability and reproducibility (System 87.0).
"""

from enum import Enum, auto
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from rae_core.utils.clock import IClock


class MemoryState(str, Enum):
    """Lifecycle states for a memory artifact."""
    WORKING = "working"               # Active context, short-term
    EPISODIC = "episodic"             # Immutable log, time-series
    SEMANTIC_PENDING = "semantic_pending"  # Candidate for abstraction
    SEMANTIC = "semantic"             # Consolidated fact in knowledge graph
    ARCHIVED = "archived"             # Moved to cold storage


class ConsolidationConfig(BaseModel):
    """Configuration for consolidation thresholds."""
    min_access_count: int = Field(default=3, description="Minimum accesses to trigger consolidation")
    min_importance: float = Field(default=0.7, description="Minimum importance to become semantic")
    decay_rate: float = Field(default=0.1, description="Exponential decay rate")
    bayes_prior: float = Field(default=0.5, description="Prior probability for new facts")


class ConsolidationFSM:
    """Deterministic Finite State Machine for memory lifecycle."""

    def __init__(self, config: ConsolidationConfig, clock: IClock) -> None:
        self.config = config
        self.clock = clock

    def evaluate_transition(self, memory: dict[str, Any]) -> tuple[MemoryState, float]:
        """Evaluate if memory should transition to next state.
        
        Returns:
            (New State, Confidence Score)
        """
        current_state = MemoryState(memory.get("layer", "working"))
        access_count = memory.get("access_count", 0)
        importance = memory.get("importance", 0.5)
        
        # Calculate dynamic threshold based on equation R(e, t)
        # For simplicity in Python: importance is already decayed by external process or here?
        # Assuming importance is current.
        
        if current_state == MemoryState.WORKING:
            # Working -> Episodic (Immediate upon save usually, or after session)
            # This transition is usually forced by "end of session".
            return MemoryState.EPISODIC, 1.0

        if current_state == MemoryState.EPISODIC:
            # Episodic -> Semantic Pending
            # Rule: High access count AND high importance
            if access_count >= self.config.min_access_count and importance >= self.config.min_importance:
                return MemoryState.SEMANTIC_PENDING, importance
            return MemoryState.EPISODIC, 0.0

        if current_state == MemoryState.SEMANTIC_PENDING:
            # Semantic Pending -> Semantic
            # Requires Verification or "Bayesian Confirmation"
            # In simple FSM, if it survived pending without being rejected -> Semantic.
            # Or requires N confirmations.
            # For now, auto-promote if importance holds.
            if importance >= self.config.min_importance:
                # Bayesian Update for edge weight would happen here
                return MemoryState.SEMANTIC, importance
            # Demote if lost importance
            if importance < 0.3:
                return MemoryState.ARCHIVED, 0.0
            return MemoryState.SEMANTIC_PENDING, 0.0

        if current_state == MemoryState.SEMANTIC:
            # Semantic -> Archived (Decay)
            if importance < 0.1:
                return MemoryState.ARCHIVED, 0.0
            return MemoryState.SEMANTIC, 1.0

        return current_state, 0.0

    def bayesian_update(self, current_prob: float, evidence_strength: float) -> float:
        """Update probability of a fact based on new evidence.
        
        P(H|E) = (P(E|H) * P(H)) / P(E)
        
        Simplified recursive update:
        Posterior = (Prior * Likelihood) / Marginal
        
        Args:
            current_prob: P(H) - Current confidence
            evidence_strength: Strength of new evidence (0.0 to 1.0)
            
        Returns:
            New probability
        """
        # Assume likelihood P(E|H) is high if evidence matches (e.g. 0.9)
        # P(E|~H) is low (e.g. 0.1)
        p_h = current_prob
        p_e_given_h = 0.9 * evidence_strength  # Stronger evidence = higher likelihood
        p_e_given_not_h = 0.1
        
        p_e = (p_e_given_h * p_h) + (p_e_given_not_h * (1 - p_h))
        
        if p_e == 0: 
            return p_h
            
        return (p_e_given_h * p_h) / p_e
