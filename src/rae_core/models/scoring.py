"""Scoring models for RAE-core memory importance."""

from datetime import datetime, timedelta

from pydantic import BaseModel, Field


class ScoringWeights(BaseModel):
    """Weights for memory scoring components."""

    recency: float = Field(default=0.3, ge=0.0, le=1.0)
    relevance: float = Field(default=0.4, ge=0.0, le=1.0)
    importance: float = Field(default=0.2, ge=0.0, le=1.0)
    usage: float = Field(default=0.1, ge=0.0, le=1.0)

    @classmethod
    def get_szubar_weights(cls) -> "ScoringWeights":
        """Get predefined weights for Szubar Mode (Pressure on Importance and Usage)."""
        return cls(
            recency=0.1,
            relevance=0.2,
            importance=0.5,
            usage=0.2,
        )


class QualityMetrics(BaseModel):
    """Quality metrics for memory assessment."""

    coherence: float = Field(default=0.5, ge=0.0, le=1.0)
    completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    accuracy: float = Field(default=0.5, ge=0.0, le=1.0)
    relevance: float = Field(default=0.5, ge=0.0, le=1.0)

    overall_quality: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Computed overall quality"
    )

    def compute_overall(self) -> float:
        """Compute overall quality score."""
        self.overall_quality = (
            self.coherence * 0.25
            + self.completeness * 0.25
            + self.accuracy * 0.30
            + self.relevance * 0.20
        )
        return self.overall_quality


class DecayConfig(BaseModel):
    """Configuration for importance decay."""

    decay_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Base decay rate per period"
    )
    decay_period: timedelta = Field(
        default=timedelta(days=1), description="Time period for decay application"
    )
    min_importance: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum importance floor"
    )
    max_importance: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Maximum importance ceiling"
    )

    # Layer-specific decay rates
    layer_rates: dict[str, float] = Field(
        default_factory=lambda: {
            "sensory": 0.5,  # Fast decay
            "working": 0.2,  # Medium decay
            "episodic": 0.05,  # Slow decay
            "semantic": 0.01,  # Very slow decay
            "reflective": 0.0,  # No decay
        },
        description="Decay rates per memory layer",
    )


class DecayResult(BaseModel):
    """Result of decay calculation."""

    original_importance: float
    decayed_importance: float
    decay_amount: float
    time_elapsed: timedelta
    next_decay_at: datetime
