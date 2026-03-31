"""
Arm representation for Multi-Armed Bandit

An arm represents a specific (level, strategy) combination that the bandit can choose.
"""

import json
from dataclasses import dataclass, field

from rae_core.math.types import MathLevel


@dataclass
class Arm:
    """
    Represents a single arm in the multi-armed bandit.

    An arm is a (level, strategy) pair that can be pulled (selected).
    Each arm maintains statistics about its performance across different contexts.

    Attributes:
        level: Math level (L1, L2, or L3)
        strategy: Strategy within that level
        arm_id: Unique identifier (auto-generated)
        config: Dictionary storing arm-specific parameters (weights, resonance, etc.)

        # Global statistics (all contexts)
        pulls: Total number of times this arm was selected
        total_reward: Cumulative reward from all pulls

        # Context-specific statistics (81 context buckets)
        context_pulls: Dict mapping context_id -> pull count
        context_rewards: Dict mapping context_id -> cumulative reward

        # Metadata
        last_pulled: Timestamp of last pull
        confidence: Confidence in this arm's estimates [0, 1]

        # Sliding Window (System 3.4 Adaptive Determinism)
        history: list[float] = field(default_factory=list)
        window_size: int = 100
    """

    level: MathLevel
    strategy: str
    arm_id: str = field(default="")
    config: dict = field(default_factory=dict)

    # Global statistics
    pulls: int = 0
    total_reward: float = 0.0

    # Context-specific statistics
    context_pulls: dict[int, int] = field(default_factory=dict)
    context_rewards: dict[int, float] = field(default_factory=dict)

    # Metadata
    last_pulled: float | None = None
    confidence: float = 0.0

    # Sliding Window
    history: list[float] = field(default_factory=list)
    window_size: int = 100

    def __post_init__(self):
        """Generate arm_id if not provided"""
        if not self.arm_id:
            self.arm_id = f"{self.level.value}:{self.strategy}"

    def mean_reward(self, context_id: int | None = None) -> float:
        """
        Calculate mean reward for this arm using Sliding Window.

        Args:
            context_id: If provided, return context-specific mean (NOT WINDOWED YET)

        Returns:
            Mean reward (0.0 if never pulled)
        """
        if context_id is not None:
            # Context-specific mean (Legacy/Global for now)
            pulls = self.context_pulls.get(context_id, 0)
            if pulls == 0:
                return 0.0
            total = self.context_rewards.get(context_id, 0.0)
            return total / pulls
        else:
            # Sliding Window Mean (Adaptive)
            if not self.history:
                return 0.0
            return sum(self.history) / len(self.history)

    def ucb_score(
        self,
        total_pulls: int,
        c: float = 1.0,
        context_id: int | None = None,
        context_bonus: float = 0.0,
    ) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for this arm.

        UCB formula: mean_reward + c * sqrt(ln(N) / n) + context_bonus

        Args:
            total_pulls: Total pulls across all arms (N)
            c: Confidence parameter (higher = more exploration)
            context_id: Context for context-specific UCB
            context_bonus: Additional bonus for context match

        Returns:
            UCB score (higher = should be selected)
        """
        import math

        # Get arm-specific pulls
        if context_id is not None:
            arm_pulls = self.context_pulls.get(context_id, 0)
        else:
            # For Sliding Window UCB, we use the effective window size (n)
            # But strictly SW-UCB uses local count in window.
            # Approximating with len(history) as the effective 'n'
            arm_pulls = len(self.history)

        # If never pulled, return infinity (explore first)
        if arm_pulls == 0:
            return float("inf")

        # Calculate UCB
        mean = self.mean_reward(context_id)

        # For SW-UCB, N should effectively be min(total_pulls, window_size * num_arms)
        # or just total_pulls. Standard UCB uses total_pulls.
        exploration_bonus = c * math.sqrt(math.log(max(total_pulls, 1)) / arm_pulls)

        return mean + exploration_bonus + context_bonus

    def update(
        self,
        reward: float,
        context_id: int | None = None,
        timestamp: float | None = None,
    ):
        """
        Update arm statistics with a new reward observation.

        Args:
            reward: Reward received from pulling this arm
            context_id: Context in which arm was pulled
            timestamp: When the arm was pulled
        """
        # Update global statistics
        self.pulls += 1
        self.total_reward += reward

        # Update context-specific statistics
        if context_id is not None:
            self.context_pulls[context_id] = self.context_pulls.get(context_id, 0) + 1
            self.context_rewards[context_id] = (
                self.context_rewards.get(context_id, 0.0) + reward
            )

        # Update metadata
        if timestamp is not None:
            self.last_pulled = timestamp

        # Update confidence (more pulls = higher confidence)
        self.confidence = 1.0 - 1.0 / (1.0 + self.pulls)

        # Update Sliding Window
        self.history.append(reward)
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def reset_window(self):
        """Reset sliding window stats (Change Point Detection)."""
        self.history = []

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "arm_id": self.arm_id,
            "level": self.level.value,
            "strategy": self.strategy,
            "config": self.config,
            "pulls": self.pulls,
            "total_reward": self.total_reward,
            "mean_reward": self.mean_reward(),
            "context_pulls": self.context_pulls,
            "context_rewards": self.context_rewards,
            "last_pulled": self.last_pulled,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "Arm":
        """Deserialize from dictionary"""
        level = MathLevel(data["level"])
        return cls(
            level=level,
            strategy=data["strategy"],
            arm_id=data.get("arm_id", ""),
            config=data.get("config", {}),
            pulls=data.get("pulls", 0),
            total_reward=data.get("total_reward", 0.0),
            context_pulls=data.get("context_pulls", {}),
            context_rewards=data.get("context_rewards", {}),
            last_pulled=data.get("last_pulled"),
            confidence=data.get("confidence", 0.0),
        )


def create_default_arms() -> list[Arm]:
    """
    Create the default set of arms for the bandit.

    Includes:
    - 21 Granular L1 arms for Text:Vector ratios (from 10:1 to 1:10)
    - 4 L2 arms for information-theoretic strategies
    - 2 L3 arms for hybrid ensemble logic
    """
    arms = []

    # L1: Granular Hybrid Ratios (21 arms)
    # Ratios from 10.0:1.0 down to 1.0:10.0
    ratios = [
        (10.0, 1.0),
        (8.0, 1.0),
        (6.0, 1.0),
        (5.0, 1.0),
        (4.0, 1.0),
        (3.0, 1.0),
        (2.0, 1.0),
        (1.5, 1.0),
        (1.2, 1.0),
        (1.0, 1.0),
        (1.0, 1.2),
        (1.0, 1.5),
        (1.0, 2.0),
        (1.0, 3.0),
        (1.0, 4.0),
        (1.0, 5.0),
        (1.0, 6.0),
        (1.0, 8.0),
        (1.0, 10.0),
        # Extreme cases for exploration
        (25.0, 1.0),
        (1.0, 25.0),
    ]

    for txt_w, vec_w in ratios:
        # Using 'w_txt' and 'vec' prefixes to make parsing unambiguous
        txt_str = str(txt_w).replace(".", "p")
        vec_str = str(vec_w).replace(".", "p")
        strategy_name = f"w_txt{txt_str}_vec{vec_str}"
        arms.append(Arm(level=MathLevel.L1, strategy=strategy_name))

    # Keep legacy L1 names for compatibility
    arms.append(Arm(level=MathLevel.L1, strategy="default"))
    arms.append(Arm(level=MathLevel.L1, strategy="relevance_scoring"))
    arms.append(Arm(level=MathLevel.L1, strategy="importance_scoring"))

    # L2 arms (4)
    arms.append(Arm(level=MathLevel.L2, strategy="default"))
    arms.append(Arm(level=MathLevel.L2, strategy="entropy_minimization"))
    arms.append(Arm(level=MathLevel.L2, strategy="information_bottleneck"))
    arms.append(Arm(level=MathLevel.L2, strategy="mutual_information"))

    # L3 arms (2)
    arms.append(Arm(level=MathLevel.L3, strategy="hybrid_default"))
    arms.append(Arm(level=MathLevel.L3, strategy="weighted_combination"))

    return arms
