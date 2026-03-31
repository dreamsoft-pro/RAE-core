"""Importance decay functions for memory aging."""

import math
from datetime import datetime, timedelta, timezone

from rae_core.models.scoring import DecayConfig, DecayResult


class ImportanceDecay:
    """
    Handles time-based importance decay for memories.

    Implements multiple decay strategies:
    - Exponential decay (default)
    - Linear decay
    - Step decay
    - Logarithmic decay
    """

    def __init__(self, config: DecayConfig | None = None):
        """Initialize with decay configuration."""
        self.config = config or DecayConfig()

    def exponential_decay(
        self, importance: float, time_elapsed: timedelta, layer: str = "working"
    ) -> DecayResult:
        """
        Apply exponential decay to importance.

        Formula: I(t) = I₀ * e^(-λt)
        where λ is the decay rate
        """
        decay_rate = self.config.layer_rates.get(layer, self.config.decay_rate)

        # Convert time to periods
        periods = time_elapsed / self.config.decay_period

        # Calculate decay
        decay_factor = math.exp(-decay_rate * periods)
        decayed = importance * decay_factor

        decayed = max(
            self.config.min_importance, min(self.config.max_importance, decayed)
        )

        return DecayResult(
            original_importance=importance,
            decayed_importance=decayed,
            decay_amount=importance - decayed,
            time_elapsed=time_elapsed,
            next_decay_at=datetime.now(timezone.utc) + self.config.decay_period,
        )

    def linear_decay(
        self, importance: float, time_elapsed: timedelta, layer: str = "working"
    ) -> DecayResult:
        """
        Apply linear decay to importance.

        Formula: I(t) = I₀ - (λ * t)
        """
        decay_rate = self.config.layer_rates.get(layer, self.config.decay_rate)
        periods = time_elapsed / self.config.decay_period

        decay_amount = decay_rate * periods
        decayed = importance - decay_amount

        decayed = max(
            self.config.min_importance, min(self.config.max_importance, decayed)
        )

        return DecayResult(
            original_importance=importance,
            decayed_importance=decayed,
            decay_amount=importance - decayed,
            time_elapsed=time_elapsed,
            next_decay_at=datetime.now(timezone.utc) + self.config.decay_period,
        )

    def logarithmic_decay(
        self, importance: float, time_elapsed: timedelta, layer: str = "working"
    ) -> DecayResult:
        """
        Apply logarithmic decay (slow initial, then faster).

        Formula: I(t) = I₀ / (1 + λ * ln(1 + t))
        """
        decay_rate = self.config.layer_rates.get(layer, self.config.decay_rate)
        periods = time_elapsed / self.config.decay_period

        decay_factor = 1 + (decay_rate * math.log(1 + periods))
        decayed = importance / decay_factor

        decayed = max(
            self.config.min_importance, min(self.config.max_importance, decayed)
        )

        return DecayResult(
            original_importance=importance,
            decayed_importance=decayed,
            decay_amount=importance - decayed,
            time_elapsed=time_elapsed,
            next_decay_at=datetime.now(timezone.utc) + self.config.decay_period,
        )

    def step_decay(
        self,
        importance: float,
        time_elapsed: timedelta,
        layer: str = "working",
        step_size: float = 0.1,
    ) -> DecayResult:
        """
        Apply step decay (discrete drops at intervals).
        """
        decay_rate = self.config.layer_rates.get(layer, self.config.decay_rate)
        periods = int(time_elapsed / self.config.decay_period)

        decay_amount = step_size * periods * decay_rate
        decayed = importance - decay_amount

        decayed = max(
            self.config.min_importance, min(self.config.max_importance, decayed)
        )

        return DecayResult(
            original_importance=importance,
            decayed_importance=decayed,
            decay_amount=importance - decayed,
            time_elapsed=time_elapsed,
            next_decay_at=datetime.now(timezone.utc) + self.config.decay_period,
        )


# Utility functions
def calculate_half_life(decay_rate: float) -> timedelta:
    """Calculate half-life for given decay rate."""
    if decay_rate <= 0:
        return timedelta(days=36500)  # ~100 years
    periods_to_half = math.log(2) / decay_rate
    return timedelta(days=periods_to_half)


def time_to_threshold(
    initial_importance: float, threshold: float, decay_rate: float
) -> timedelta:
    """Calculate time until importance reaches threshold."""
    if initial_importance <= threshold or decay_rate <= 0:
        return timedelta(0)

    periods = -math.log(threshold / initial_importance) / decay_rate
    return timedelta(days=periods)
