"""
RAE Math - Temporal Dynamics

This module implements temporal dynamics for memory systems:
- Recency scoring with exponential decay
- Access count consideration (frequent access slows decay)
- Time-aware utility functions

These are pure mathematical functions with no external dependencies.

Mathematical Foundation:
    Recency score uses exponential decay:
        R(t) = exp(-λ·Δt)

    Where:
        - λ = effective decay rate (adjusted by access count)
        - Δt = time elapsed since last access or creation
        - R(t) ∈ [0, 1], with R(0) = 1 and lim R(t) → 0 as t → ∞

    Access count adjustment:
        λ_effective = λ_base / (ln(1 + n_access) + 1)

    This implements "use it or lose it" - frequently accessed memories
    decay slower, staying relevant longer.

License: Apache-2.0
Author: Grzegorz Leśniowski <lesniowskig@gmail.com>
"""

import math
from datetime import datetime, timezone

from rae_core.math.structure import DecayConfig


def calculate_recency_score(
    last_accessed_at: datetime | None,
    created_at: datetime,
    access_count: int,
    now: datetime,
    decay_config: DecayConfig | None = None,
) -> tuple[float, float, float]:
    """
    Calculate recency component with access count consideration.

    Implements exponential decay with usage-based decay rate adjustment:
        1. Determine reference time (last access or creation)
        2. Calculate time elapsed since reference
        3. Adjust decay rate based on access count
        4. Apply exponential decay formula
        5. Clamp result to [0, 1]

    Formula:
        time_ref = last_accessed_at or created_at
        time_diff = (now - time_ref).total_seconds()

        if access_count_boost:
            effective_decay = base_decay / (log(1 + access_count) + 1)
        else:
            effective_decay = base_decay

        recency_score = exp(-effective_decay * time_diff)

    Args:
        last_accessed_at: Last access timestamp (None if never accessed)
        created_at: Memory creation timestamp
        access_count: Number of times memory was accessed
        now: Current time for age calculation
        decay_config: Decay configuration (default: DecayConfig())

    Returns:
        Tuple of (recency_score, age_seconds, effective_decay_rate):
            - recency_score: Temporal relevance score (0.0-1.0)
            - age_seconds: Time elapsed since reference (float)
            - effective_decay_rate: Actual decay rate used (float)

    Mathematical Properties:
        - Half-life: t_1/2 = ln(2) / λ_effective
        - For base_decay=0.001, no access: t_1/2 ≈ 693s ≈ 11.5 min
        - For base_decay=0.001, access_count=9: t_1/2 ≈ 1594s ≈ 26.6 min
        - For base_decay=0.001, access_count=99: t_1/2 ≈ 3882s ≈ 64.7 min

    Example:
        >>> from datetime import datetime, timedelta
        >>> created = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        >>> now = datetime(2024, 1, 1, 12, 10, 0, tzinfo=timezone.utc)  # 10 min later
        >>>
        >>> # Recently created, never accessed (access_count=0)
        >>> score, age, decay = calculate_recency_score(
        ...     last_accessed_at=None,
        ...     created_at=created,
        ...     access_count=0,
        ...     now=now
        ... )
        >>> print(f"Score: {score:.3f}, Age: {age:.0f}s")
        Score: 0.549, Age: 600s
        >>>
        >>> # Frequently accessed (access_count=10)
        >>> score, age, decay = calculate_recency_score(
        ...     last_accessed_at=None,
        ...     created_at=created,
        ...     access_count=10,
        ...     now=now
        ... )
        >>> print(f"Score: {score:.3f} (higher due to frequent access)")
        Score: 0.773 (higher due to frequent access)
    """
    # Initialize config if not provided
    if decay_config is None:
        decay_config = DecayConfig()

    # Ensure timestamps are timezone-aware
    created_at = ensure_utc(created_at)
    if last_accessed_at:
        last_accessed_at = ensure_utc(last_accessed_at)

    # Determine reference time (last access or creation)
    time_ref = last_accessed_at if last_accessed_at else created_at
    time_diff = (now - time_ref).total_seconds()

    # Handle edge case: future timestamps (shouldn't happen, but safeguard)
    if time_diff < 0:
        import warnings

        warnings.warn(
            f"Future timestamp detected: time_ref={time_ref.isoformat()}, now={now.isoformat()}. "
            f"Returning perfect recency score (1.0).",
            UserWarning,
        )
        return 1.0, 0.0, 0.0

    # Calculate effective decay rate based on access count
    if decay_config.access_count_boost and access_count > 0:
        # More accessed memories decay slower
        # Mathematical justification:
        #   - access_count=0: divisor=1 (no reduction)
        #   - access_count=9: divisor≈2.3 (decay rate ~43% of base)
        #   - access_count=99: divisor≈5.6 (decay rate ~18% of base)
        #
        # This creates logarithmic diminishing returns:
        # - First few accesses have large impact
        # - Additional accesses have smaller impact
        effective_decay = decay_config.base_decay_rate / (
            math.log(1 + access_count) + 1
        )
    else:
        effective_decay = decay_config.base_decay_rate

    # Clamp to configured min/max
    effective_decay = max(
        decay_config.min_decay_rate, min(decay_config.max_decay_rate, effective_decay)
    )

    # Exponential decay: score = e^(-λ·t)
    recency_score = math.exp(-effective_decay * time_diff)

    # Ensure score is in [0, 1]
    recency_score = max(0.0, min(1.0, recency_score))

    return recency_score, time_diff, effective_decay


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware (UTC).

    If the datetime is naive (no timezone info), assumes UTC.
    If already timezone-aware, returns as-is.

    Args:
        dt: Input datetime (may be naive or aware)

    Returns:
        Timezone-aware datetime in UTC

    Example:
        >>> from datetime import datetime
        >>> naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        >>> aware_dt = ensure_utc(naive_dt)
        >>> print(aware_dt.tzinfo)
        UTC
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def calculate_half_life(decay_rate: float) -> float:
    """
    Calculate half-life (time for score to reach 0.5) given decay rate.

    Mathematical formula:
        t_1/2 = ln(2) / λ

    Where:
        - t_1/2 = half-life (seconds)
        - λ = decay rate (per second)
        - ln(2) ≈ 0.693

    Args:
        decay_rate: Decay rate per second (λ)

    Returns:
        Half-life in seconds

    Example:
        >>> # Standard decay rate
        >>> half_life = calculate_half_life(0.001)
        >>> print(f"Half-life: {half_life:.1f}s = {half_life/60:.1f} min")
        Half-life: 693.1s = 11.6 min
        >>>
        >>> # Slower decay (e.g., for LTM)
        >>> half_life = calculate_half_life(0.0001)
        >>> print(f"Half-life: {half_life/3600:.1f} hours")
        Half-life: 1.9 hours
    """
    if decay_rate <= 0:
        raise ValueError(f"Decay rate must be positive, got {decay_rate}")
    return math.log(2) / decay_rate


def calculate_decay_rate_from_half_life(half_life_seconds: float) -> float:
    """
    Calculate decay rate from desired half-life.

    Inverse of calculate_half_life():
        λ = ln(2) / t_1/2

    Args:
        half_life_seconds: Desired half-life in seconds

    Returns:
        Decay rate per second

    Example:
        >>> # Want memories to decay to 50% after 1 hour
        >>> decay = calculate_decay_rate_from_half_life(3600)
        >>> print(f"Decay rate: {decay:.6f}")
        Decay rate: 0.000193
    """
    if half_life_seconds <= 0:
        raise ValueError(f"Half-life must be positive, got {half_life_seconds}")
    return math.log(2) / half_life_seconds
