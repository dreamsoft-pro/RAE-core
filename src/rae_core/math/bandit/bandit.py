"""
Multi-Armed Bandit for Math Level Selection

Implements UCB (Upper Confidence Bound) algorithm with contextual information
and safety guardrails for online learning.
"""

import json
import time
from dataclasses import dataclass
from pathlib import Path

from rae_core.math.features_v2 import FeaturesV2
from rae_core.math.types import MathLevel

from .arm import Arm, create_default_arms


@dataclass
class BanditConfig:
    """
    Configuration for Multi-Armed Bandit.

    Attributes:
        c: UCB exploration parameter (higher = more exploration)
        context_bonus: Bonus for context match
        exploration_rate: Probability of random exploration [0, 1]
        max_exploration_rate: Maximum allowed exploration (safety limit)
        degradation_threshold: Max acceptable reward drop before rollback
        min_pulls_for_confidence: Minimum pulls before arm is considered reliable
        save_frequency: Save arm weights every N decisions
        persistence_path: Path to save/load arm statistics
    """

    c: float = 1.0
    context_bonus: float = 0.1
    exploration_rate: float = 0.1
    max_exploration_rate: float = 0.2
    degradation_threshold: float = 0.15
    min_pulls_for_confidence: int = 10
    save_frequency: int = 50
    persistence_path: Path | None = None

    def __post_init__(self):
        """Validate configuration"""
        if self.persistence_path and isinstance(self.persistence_path, str):
            self.persistence_path = Path(self.persistence_path)

        if self.exploration_rate is None:
            self.exploration_rate = 0.1

        if self.exploration_rate > self.max_exploration_rate:
            raise ValueError(
                f"exploration_rate ({self.exploration_rate}) exceeds "
                f"max_exploration_rate ({self.max_exploration_rate})"
            )
        if not 0 <= self.exploration_rate <= 1:
            raise ValueError(
                f"exploration_rate must be in [0, 1], got {self.exploration_rate}"
            )


class MultiArmedBandit:
    """
    Multi-Armed Bandit for Math Level Selection.

    Uses UCB algorithm with contextual information and safety guardrails.
    Maintains 9 arms representing (level, strategy) combinations.

    Decision Flow:
    1. Extract context features and discretize to bucket ID
    2. Compute UCB scores for all arms
    3. Select arm with highest UCB score (or explore randomly)
    4. Return (level, strategy) recommendation
    5. After execution, update arm with observed reward

    Safety Features:
    - Exploration rate caps (production: 0%, research: 20%)
    - Degradation detection (rollback if performance drops)
    - Minimum confidence thresholds
    """

    def __init__(
        self,
        config: BanditConfig,
        arms: list[Arm] | None = None,
    ):
        """
        Initialize bandit.

        Args:
            config: Bandit configuration
            arms: List of arms (defaults to create_default_arms())
        """
        self.config = config
        self.arms = arms if arms is not None else create_default_arms()

        # Create arm lookup by (level, strategy)
        self.arm_map: dict[tuple[MathLevel, str], Arm] = {
            (arm.level, arm.strategy): arm for arm in self.arms
        }

        # Statistics
        self.total_pulls = 0
        self.total_reward = 0.0
        self.decisions_since_save = 0

        # Safety tracking
        self.baseline_mean_reward = 0.0
        self.last_100_rewards: list[float] = []

        # Load persisted state if available
        if self.config.persistence_path and self.config.persistence_path.exists():
            self.load_state()

    def select_arm(
        self,
        features: FeaturesV2,
        force_exploration: bool = False,
    ) -> tuple[Arm, bool]:
        """
        Select arm using UCB algorithm.

        Args:
            features: Task features for context
            force_exploration: Force random exploration (for testing)

        Returns:
            Tuple of (selected_arm, was_exploration)
        """
        import random

        # Discretize context
        context_id = self._discretize_context(features)

        # Decide: explore or exploit
        should_explore = force_exploration or (
            random.random() < self.config.exploration_rate
        )

        if should_explore and self.config.exploration_rate > 0:
            # Random exploration
            arm = random.choice(self.arms)
            return arm, True
        else:
            # UCB exploitation
            best_arm = max(
                self.arms,
                key=lambda arm: arm.ucb_score(
                    total_pulls=max(self.total_pulls, 1),
                    c=self.config.c,
                    context_id=context_id,
                    context_bonus=self.config.context_bonus,
                ),
            )
            return best_arm, False

    def update(
        self,
        arm: Arm,
        reward: float,
        features: FeaturesV2,
    ):
        """
        Update arm with observed reward.

        Args:
            arm: Arm that was pulled
            reward: Observed reward
            features: Task features for context
        """
        # Discretize context
        context_id = self._discretize_context(features)

        # Update arm statistics
        timestamp = time.time()
        arm.update(reward=reward, context_id=context_id, timestamp=timestamp)

        # Update global statistics
        self.total_pulls += 1
        self.total_reward += reward

        # Track recent rewards for degradation detection
        self.last_100_rewards.append(reward)
        if len(self.last_100_rewards) > 100:
            self.last_100_rewards.pop(0)

        # Update baseline if we have enough data
        if self.total_pulls >= 20:
            if self.baseline_mean_reward == 0:
                self.baseline_mean_reward = self.total_reward / self.total_pulls

            # CHECK FOR DRIFT (System 3.4 Adaptive Determinism)
            # This will automatically reset all arms if significant degradation is found
            self.check_degradation()

        # Periodic save
        self.decisions_since_save += 1
        if self.decisions_since_save >= self.config.save_frequency:
            self.save_state()
            self.decisions_since_save = 0

    def check_degradation(self) -> tuple[bool, float]:
        """
        Check if performance has degraded significantly (CUSUM-style drift detection).
        If drift is detected, RESET all arms to adapt to the new context (Change Point).

        Returns:
            Tuple of (is_degraded, current_drop)
        """
        if len(self.last_100_rewards) < 20:
            return False, 0.0

        if self.baseline_mean_reward == 0:
            return False, 0.0

        recent_mean = sum(self.last_100_rewards) / len(self.last_100_rewards)
        drop = (self.baseline_mean_reward - recent_mean) / abs(
            self.baseline_mean_reward
        )

        is_degraded = drop > self.config.degradation_threshold

        if is_degraded:
            # CHANGE POINT DETECTED!
            # The world has changed (e.g. log -> commercial data).
            # Old statistics are now misleading. Reset everything.
            print(
                f"📉 DRIFT DETECTED! Drop: {drop:.2%} > {self.config.degradation_threshold:.2%}. Resetting Bandit Memory."
            )

            for arm in self.arms:
                arm.reset_window()

            # Reset global baselines
            self.last_100_rewards = []
            self.baseline_mean_reward = 0.0
            # We don't reset total_pulls/total_reward purely for historical logging,
            # but the arms' decision making is now fresh.

        return is_degraded, drop

    def get_best_arm(self, features: FeaturesV2) -> Arm:
        """
        Get best arm by mean reward (no exploration).

        Args:
            features: Task features for context

        Returns:
            Arm with highest mean reward
        """
        context_id = self._discretize_context(features)

        # Filter arms with minimum confidence
        confident_arms = [
            arm
            for arm in self.arms
            if arm.pulls >= self.config.min_pulls_for_confidence
        ]

        if not confident_arms:
            # No confident arms yet - return default L1
            return self.arm_map[(MathLevel.L1, "default")]

        # Return arm with highest mean reward
        return max(confident_arms, key=lambda arm: arm.mean_reward(context_id))

    def _discretize_context(self, features: FeaturesV2) -> int:
        """
        Discretize context features into bucket ID.

        Updated for System 4.0 Spectrum:
        - Uses 'is_industrial' flag directly
        - Uses entropy and density
        """
        # If industrial, use special high-range buckets (100+)
        if getattr(features, "is_industrial", False):
            base = 100
            if getattr(features, "is_quantitative", False):
                base += 50  # 150+ for quantitative industrial

            if features.term_density > 0.8:
                return base + 1  # 101 or 151 (High Density)
            return base  # 100 or 150 (Normal)

        # Legacy discretization for general queries
        # Discretize memory_count into 3 bins
        if features.memory_count < 30:
            memory_bin = 0  # small
        elif features.memory_count < 200:
            memory_bin = 1  # medium
        else:
            memory_bin = 2  # large

        # Discretize graph_density into 3 bins
        if features.graph_density < 0.3:
            density_bin = 0  # sparse
        elif features.graph_density < 0.7:
            density_bin = 1  # medium
        else:
            density_bin = 2  # dense

        # Discretize memory_entropy into 3 bins
        if features.memory_entropy < 0.3:
            entropy_bin = 0  # low
        elif features.memory_entropy < 0.7:
            entropy_bin = 1  # medium
        else:
            entropy_bin = 2  # high

        # Discretize task affinity (use L3 affinity as proxy)
        affinity = features.get_task_affinity_l3()
        if affinity < 0.3:
            affinity_bin = 0  # low
        elif affinity < 0.7:
            affinity_bin = 1  # medium
        else:
            affinity_bin = 2  # high

        # Combine into single bucket ID: 0-80
        # Formula: memory*27 + density*9 + entropy*3 + affinity
        bucket_id = memory_bin * 27 + density_bin * 9 + entropy_bin * 3 + affinity_bin

        return bucket_id

    def save_state(self):
        """Save arm statistics to disk"""
        if not self.config.persistence_path:
            return

        state = {
            "timestamp": time.time(),
            "total_pulls": self.total_pulls,
            "total_reward": self.total_reward,
            "baseline_mean_reward": self.baseline_mean_reward,
            "arms": [arm.to_dict() for arm in self.arms],
        }

        self.config.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.persistence_path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self):
        """Load arm statistics from disk"""
        if (
            not self.config.persistence_path
            or not self.config.persistence_path.exists()
        ):
            return

        with open(self.config.persistence_path) as f:
            state = json.load(f)

        self.total_pulls = state.get("total_pulls", 0)
        self.total_reward = state.get("total_reward", 0.0)
        self.baseline_mean_reward = state.get("baseline_mean_reward", 0.0)

        # Restore arm statistics
        arms_data = state.get("arms", [])
        for arm_data in arms_data:
            arm_id = arm_data["arm_id"]
            # Find matching arm
            for arm in self.arms:
                if arm.arm_id == arm_id:
                    arm.pulls = arm_data.get("pulls", 0)
                    arm.total_reward = arm_data.get("total_reward", 0.0)
                    arm.context_pulls = arm_data.get("context_pulls", {})
                    # Convert string keys back to int
                    arm.context_pulls = {
                        int(k): v for k, v in arm.context_pulls.items()
                    }
                    arm.context_rewards = arm_data.get("context_rewards", {})
                    arm.context_rewards = {
                        int(k): v for k, v in arm.context_rewards.items()
                    }
                    arm.last_pulled = arm_data.get("last_pulled")
                    arm.confidence = arm_data.get("confidence", 0.0)
                    break

    def get_statistics(self) -> dict:
        """Get bandit statistics for monitoring"""
        is_degraded, drop = self.check_degradation()

        return {
            "total_pulls": self.total_pulls,
            "total_reward": self.total_reward,
            "mean_reward": self.total_reward / max(self.total_pulls, 1),
            "baseline_mean_reward": self.baseline_mean_reward,
            "is_degraded": is_degraded,
            "degradation_drop": drop,
            "exploration_rate": self.config.exploration_rate,
            "arms": [
                {
                    "arm_id": arm.arm_id,
                    "pulls": arm.pulls,
                    "mean_reward": arm.mean_reward(),
                    "confidence": arm.confidence,
                }
                for arm in sorted(self.arms, key=lambda a: a.pulls, reverse=True)
            ],
        }

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "config": {
                "c": self.config.c,
                "exploration_rate": self.config.exploration_rate,
                "max_exploration_rate": self.config.max_exploration_rate,
            },
            "statistics": self.get_statistics(),
        }
