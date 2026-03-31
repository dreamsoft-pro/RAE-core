"""
Type definitions for Math Layer Controller

Defines the core enums used throughout the controller:
- MathLevel: Which mathematical level to use (L1, L2, L3)
- TaskType: Types of memory operations
"""

from enum import Enum


class MathLevel(str, Enum):
    """
    Mathematical layer levels for memory operations.

    L1 - Deterministic/Heuristic: Rule-based scoring with configurable formulas
    L2 - Information-Theoretic: Entropy-based optimization with IB principles
    L3 - Adaptive/Hybrid: Meta-learning that combines L1+L2 based on context
    """

    L1 = "deterministic_heuristic"
    L2 = "information_theoretic"
    L3 = "adaptive_hybrid"

    @property
    def cost_multiplier(self) -> float:
        """Relative computational cost of each level"""
        return {
            MathLevel.L1: 1.0,
            MathLevel.L2: 2.5,
            MathLevel.L3: 4.0,
        }[self]

    @property
    def description(self) -> str:
        """Human-readable description"""
        return {
            MathLevel.L1: "Rule-based scoring (fastest, lowest cost)",
            MathLevel.L2: "Information-theoretic optimization (balanced)",
            MathLevel.L3: "Adaptive hybrid meta-controller (highest quality)",
        }[self]


class TaskType(str, Enum):
    """Types of tasks that influence math level selection"""

    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_CONSOLIDATE = "memory_consolidate"
    REFLECTION_LIGHT = "reflection_light"
    REFLECTION_DEEP = "reflection_deep"
    GRAPH_UPDATE = "graph_update"
    CONTEXT_SELECT = "context_select"

    @property
    def preferred_level(self) -> MathLevel:
        """Default preferred level for this task type"""
        mapping = {
            TaskType.MEMORY_STORE: MathLevel.L1,
            TaskType.MEMORY_RETRIEVE: MathLevel.L1,
            TaskType.MEMORY_CONSOLIDATE: MathLevel.L2,
            TaskType.REFLECTION_LIGHT: MathLevel.L1,
            TaskType.REFLECTION_DEEP: MathLevel.L2,
            TaskType.GRAPH_UPDATE: MathLevel.L2,
            TaskType.CONTEXT_SELECT: MathLevel.L2,
        }
        return mapping[self]
