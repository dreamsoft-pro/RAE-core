"""Reasoning controller for graph reasoning with configurable parameters.

Centralizes control of reasoning depth, uncertainty handling, and path pruning.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReasoningPath:
    """Represents a reasoning chain/path through the knowledge graph.

    A path consists of:
    - nodes: Sequence of node IDs visited
    - steps: Human-readable step descriptions
    - uncertainty: Confidence score (0-1, higher is better)
    - contradictions: Any detected contradictions
    - metadata: Additional path information
    """

    nodes: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    uncertainty: float = 1.0  # Start with full confidence
    contradictions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens_used: int = 0

    @property
    def depth(self) -> int:
        """Current depth of reasoning path."""
        return len(self.steps)

    @property
    def is_contradictory(self) -> bool:
        """Check if path has contradictions."""
        return len(self.contradictions) > 0

    def add_step(
        self,
        node_id: str,
        description: str,
        uncertainty_delta: float = 0.0,
        tokens: int = 0,
    ) -> None:
        """Add a reasoning step to the path.

        Args:
            node_id: Node ID being visited
            description: Human-readable step description
            uncertainty_delta: Change in uncertainty (-1 to +1)
            tokens: Tokens consumed by this step
        """
        self.nodes.append(node_id)
        self.steps.append(description)
        self.uncertainty = max(0.0, min(1.0, self.uncertainty + uncertainty_delta))
        self.tokens_used += tokens

    def aligns_with(self, memory: dict[str, Any]) -> bool:
        """Check if path aligns with a memory.

        Args:
            memory: Memory dictionary with content and metadata

        Returns:
            True if path content is consistent with memory
        """
        # Simple heuristic: check if memory content appears in any step
        memory_content = memory.get("content", "").lower()
        if not memory_content:
            return False

        # Check if any step mentions similar content
        for step in self.steps:
            if memory_content in step.lower():
                return True

        return False

    def count_unverified_assumptions(self, verified_facts: set[str]) -> int:
        """Count unverified assumptions in the path.

        Args:
            verified_facts: Set of verified fact strings

        Returns:
            Number of steps that are not verified by known facts
        """
        unverified_count = 0
        for step in self.steps:
            step_lower = step.lower()
            # Check if this step is supported by any verified fact
            is_verified = any(fact.lower() in step_lower for fact in verified_facts)
            if not is_verified:
                unverified_count += 1
        return unverified_count

    def similarity_to(self, other_path: "ReasoningPath") -> float:
        """Compute similarity to another path.

        Args:
            other_path: Another reasoning path

        Returns:
            Similarity score (0-1) based on overlapping steps
        """
        if not self.steps or not other_path.steps:
            return 0.0

        # Simple Jaccard similarity on step content
        self_content = set(word.lower() for step in self.steps for word in step.split())
        other_content = set(
            word.lower() for step in other_path.steps for word in step.split()
        )

        if not self_content or not other_content:  # pragma: no cover
            return 0.0  # pragma: no cover

        intersection = len(self_content & other_content)
        union = len(self_content | other_content)

        return intersection / union if union > 0 else 0.0


class ReasoningController:
    """Centralized controller for graph reasoning with configurable parameters.

    Features:
    - Configurable max depth, uncertainty threshold, token budget
    - Decision logic for continuing vs stopping reasoning
    - Path pruning based on contradictions and memory alignment
    - Statistics tracking for debugging

    Usage:
        controller = ReasoningController(
            max_depth=12,
            uncertainty_threshold=0.3,
            token_budget_per_step=500,
        )

        # During reasoning
        if controller.should_continue_reasoning(
            current_depth=path.depth,
            uncertainty=path.uncertainty,
            tokens_used=path.tokens_used,
        ):
            # Continue reasoning
            pass
        else:
            # Stop reasoning
            pass
    """

    def __init__(
        self,
        max_depth: int = 12,
        uncertainty_threshold: float = 0.6,  # Increased from 0.3 to improve coherence
        token_budget_per_step: int = 500,
        enable_pruning: bool = True,
        max_unverified_assumptions: int = 2,
        known_false_similarity_threshold: float = 0.7,  # Decreased from 0.8
    ):
        """Initialize reasoning controller.

        Args:
            max_depth: Maximum reasoning depth before stopping
            uncertainty_threshold: Stop if uncertainty drops below this
            token_budget_per_step: Maximum tokens per reasoning step
            enable_pruning: Enable automatic path pruning
            max_unverified_assumptions: Maximum unverified assumptions before pruning
            known_false_similarity_threshold: Similarity threshold for known false paths
        """
        self.max_depth = max_depth
        self.uncertainty_threshold = uncertainty_threshold
        self.token_budget = token_budget_per_step
        self.enable_pruning = enable_pruning
        self.max_unverified_assumptions = max_unverified_assumptions
        self.known_false_similarity_threshold = known_false_similarity_threshold

        # Track known false paths for pruning
        self.known_false_paths: list[ReasoningPath] = []

        # Statistics
        self.stats = {
            "paths_evaluated": 0,
            "paths_pruned": 0,
            "paths_pruned_unverified": 0,
            "paths_pruned_similar_to_false": 0,
            "paths_pruned_contradictory": 0,
            "max_depth_reached": 0,
            "uncertainty_stops": 0,
            "budget_exceeded": 0,
        }

    def should_continue_reasoning(
        self,
        current_depth: int,
        uncertainty: float,
        tokens_used: int,
    ) -> bool:
        """Decide whether to continue or stop reasoning.

        Args:
            current_depth: Current depth of reasoning path
            uncertainty: Current uncertainty score (0-1)
            tokens_used: Tokens consumed so far

        Returns:
            True if should continue, False if should stop
        """
        self.stats["paths_evaluated"] += 1

        # Check max depth
        if current_depth >= self.max_depth:
            self.stats["max_depth_reached"] += 1
            logger.debug(f"Stopping: Max depth {self.max_depth} reached")
            return False

        # Check uncertainty (higher uncertainty = less confident)
        # Apply depth penalty: deeper paths require higher base confidence
        depth_penalty = 0.05 * current_depth
        effective_uncertainty = uncertainty - depth_penalty

        # If uncertainty drops below threshold, we're too uncertain to continue
        if effective_uncertainty < self.uncertainty_threshold:
            self.stats["uncertainty_stops"] += 1
            logger.debug(
                f"Stopping: Effective uncertainty {effective_uncertainty:.2f} "
                f"(base {uncertainty:.2f} - penalty {depth_penalty:.2f}) "
                f"below threshold {self.uncertainty_threshold}"
            )
            return False

        # Check token budget
        if tokens_used > self.token_budget:
            self.stats["budget_exceeded"] += 1
            logger.debug(
                f"Stopping: Tokens {tokens_used} exceed budget {self.token_budget}"
            )
            return False

        return True

    def prune_contradictory_paths(
        self,
        paths: list[ReasoningPath],
        episodic_memories: list[dict[str, Any]] | None = None,
        semantic_memories: list[dict[str, Any]] | None = None,
        verified_facts: set[str] | None = None,
    ) -> list[ReasoningPath]:
        """Cut paths that contradict memory layers or have too many unverified assumptions.

        Enhanced with three heuristics:
        1. Paths contradicting episodic memory → prune
        2. Paths with >max_unverified_assumptions → prune
        3. Paths similar (>threshold) to known-false paths → prune

        Args:
            paths: List of reasoning paths to evaluate
            episodic_memories: Recent episodic memories for validation
            semantic_memories: Semantic knowledge for validation
            verified_facts: Set of verified fact strings

        Returns:
            Filtered list of non-contradictory paths
        """
        if not self.enable_pruning:
            return paths

        pruned_paths = []

        for path in paths:
            # Heuristic 1: Skip paths with detected contradictions
            if path.is_contradictory:
                self.stats["paths_pruned"] += 1
                self.stats["paths_pruned_contradictory"] += 1
                logger.debug(f"Pruning path (depth {path.depth}): Has contradictions")
                continue

            # Heuristic 1a: Check alignment with combined memories
            all_memories = (episodic_memories or []) + (semantic_memories or [])
            if all_memories:
                is_contradictory = self._contradicts_memories(path, all_memories)
                if is_contradictory:
                    self.stats["paths_pruned"] += 1
                    self.stats["paths_pruned_contradictory"] += 1
                    logger.debug(
                        f"Pruning path (depth {path.depth}): "
                        "Contradicts available memories"
                    )
                    continue

            # Heuristic 2: Prune paths with too many unverified assumptions
            if verified_facts is not None:
                unverified_count = path.count_unverified_assumptions(verified_facts)
                if unverified_count > self.max_unverified_assumptions:
                    self.stats["paths_pruned"] += 1
                    self.stats["paths_pruned_unverified"] += 1
                    logger.debug(
                        f"Pruning path (depth {path.depth}): "
                        f"{unverified_count} unverified assumptions "
                        f"(max: {self.max_unverified_assumptions})"
                    )
                    continue

            # Heuristic 3: Prune paths similar to known-false paths
            if self.known_false_paths:
                is_similar_to_false = False
                for false_path in self.known_false_paths:
                    similarity = path.similarity_to(false_path)
                    if similarity > self.known_false_similarity_threshold:
                        is_similar_to_false = True
                        self.stats["paths_pruned"] += 1
                        self.stats["paths_pruned_similar_to_false"] += 1
                        logger.debug(
                            f"Pruning path (depth {path.depth}): "
                            f"Similar ({similarity:.2f}) to known-false path"
                        )
                        break
                if is_similar_to_false:
                    continue

            # Path passes all checks
            pruned_paths.append(path)

        return pruned_paths

    def _contradicts_memories(
        self, path: ReasoningPath, memories: list[dict[str, Any]]
    ) -> bool:
        """Check if path contradicts any memories.

        Simple heuristic: If path doesn't align with any memory,
        it might be contradictory.

        Args:
            path: Reasoning path to check
            memories: Memories to validate against

        Returns:
            True if path contradicts memories
        """
        if not memories:
            return False

        # Check if path aligns with at least one memory
        aligns_with_any = any(path.aligns_with(mem) for mem in memories)

        # If path doesn't align with any memory, consider it contradictory
        return not aligns_with_any

    def get_stats(self) -> dict[str, int]:
        """Get reasoning statistics.

        Returns:
            Dictionary with reasoning stats
        """
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        for key in self.stats:
            self.stats[key] = 0

    def mark_path_as_false(self, path: ReasoningPath) -> None:
        """Mark a path as known-false for future pruning.

        Args:
            path: Path that was determined to be false/incorrect
        """
        self.known_false_paths.append(path)
        logger.debug(f"Marked path (depth {path.depth}) as known-false")

    def clear_known_false_paths(self) -> None:
        """Clear the list of known-false paths."""
        count = len(self.known_false_paths)
        self.known_false_paths.clear()
        logger.debug(f"Cleared {count} known-false paths")
