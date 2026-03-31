"""RAE-core configuration using pydantic-settings."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rae_core.config.defaults import (
    DEFAULT_CACHE_MAX_SIZE,
    DEFAULT_CACHE_TTL,
    DEFAULT_CONSOLIDATION_THRESHOLD,
    DEFAULT_DECAY_INTERVAL_HOURS,
    DEFAULT_DECAY_RATE,
    DEFAULT_EPISODIC_SIZE,
    DEFAULT_LLM_MAX_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_MIN_IMPORTANCE,
    DEFAULT_MIN_MEMORIES_FOR_REFLECTION,
    DEFAULT_OTEL_ENABLED,
    DEFAULT_OTEL_SERVICE_NAME,
    DEFAULT_OTEL_SERVICE_VERSION,
    DEFAULT_PROMOTION_THRESHOLD,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_REFLECTION_MAX_AGE_HOURS,
    DEFAULT_RERANK_TOP_K,
    DEFAULT_SEMANTIC_SIZE,
    DEFAULT_SENSORY_SIZE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_SYNC_BATCH_SIZE,
    DEFAULT_SYNC_RETRY_ATTEMPTS,
    DEFAULT_SYNC_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_WORKING_SIZE,
)


class RAESettings(BaseSettings):
    """RAE-core settings loaded from environment variables.

    All settings can be overridden via environment variables with RAE_ prefix.
    Example: RAE_SENSORY_MAX_SIZE=200
    """

    model_config = SettingsConfigDict(
        env_prefix="RAE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Memory layer sizes
    sensory_max_size: int = Field(
        default=DEFAULT_SENSORY_SIZE,
        description="Maximum size of sensory memory layer",
    )
    working_max_size: int = Field(
        default=DEFAULT_WORKING_SIZE,
        description="Maximum size of working memory layer",
    )
    episodic_max_size: int = Field(
        default=DEFAULT_EPISODIC_SIZE,
        description="Maximum size of episodic memory layer",
    )
    semantic_max_size: int = Field(
        default=DEFAULT_SEMANTIC_SIZE,
        description="Maximum size of semantic memory layer",
    )

    # Memory decay parameters
    decay_rate: float = Field(
        default=DEFAULT_DECAY_RATE,
        ge=0.0,
        le=1.0,
        description="Decay rate for memory importance (0-1)",
    )
    decay_interval_hours: int = Field(
        default=DEFAULT_DECAY_INTERVAL_HOURS,
        ge=1,
        description="Hours between decay cycles",
    )

    # Importance thresholds
    min_importance: float = Field(
        default=DEFAULT_MIN_IMPORTANCE,
        ge=0.0,
        le=1.0,
        description="Minimum importance threshold",
    )
    promotion_threshold: float = Field(
        default=DEFAULT_PROMOTION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Importance threshold for promotion to semantic layer",
    )
    consolidation_threshold: float = Field(
        default=DEFAULT_CONSOLIDATION_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Threshold for memory consolidation",
    )

    # Search parameters
    search_top_k: int = Field(
        default=DEFAULT_TOP_K,
        ge=1,
        description="Number of top results to return in search",
    )
    similarity_threshold: float = Field(
        default=DEFAULT_SIMILARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for search results",
    )
    rerank_top_k: int = Field(
        default=DEFAULT_RERANK_TOP_K,
        ge=1,
        description="Number of results to rerank",
    )

    # Reflection parameters
    min_memories_for_reflection: int = Field(
        default=DEFAULT_MIN_MEMORIES_FOR_REFLECTION,
        ge=1,
        description="Minimum number of memories required for reflection",
    )
    reflection_max_age_hours: int = Field(
        default=DEFAULT_REFLECTION_MAX_AGE_HOURS,
        ge=1,
        description="Maximum age of memories to consider for reflection (hours)",
    )
    quality_threshold: float = Field(
        default=DEFAULT_QUALITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Quality threshold for memory pruning",
    )

    # LLM parameters
    llm_temperature: float = Field(
        default=DEFAULT_LLM_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="LLM temperature parameter",
    )
    llm_max_tokens: int = Field(
        default=DEFAULT_LLM_MAX_TOKENS,
        ge=1,
        description="Maximum tokens for LLM generation",
    )
    llm_timeout: float = Field(
        default=DEFAULT_LLM_TIMEOUT,
        ge=1.0,
        description="LLM request timeout in seconds",
    )

    # Sync parameters
    sync_enabled: bool = Field(
        default=False,
        description="Enable memory synchronization",
    )
    sync_batch_size: int = Field(
        default=DEFAULT_SYNC_BATCH_SIZE,
        ge=1,
        description="Batch size for sync operations",
    )
    sync_timeout: float = Field(
        default=DEFAULT_SYNC_TIMEOUT,
        ge=1.0,
        description="Sync operation timeout in seconds",
    )
    sync_retry_attempts: int = Field(
        default=DEFAULT_SYNC_RETRY_ATTEMPTS,
        ge=1,
        description="Number of retry attempts for sync operations",
    )
    sync_encryption_enabled: bool = Field(
        default=True,
        description="Enable E2E encryption for sync",
    )

    # Cache parameters
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching",
    )
    cache_ttl: int = Field(
        default=DEFAULT_CACHE_TTL,
        ge=1,
        description="Cache TTL in seconds",
    )
    cache_max_size: int = Field(
        default=DEFAULT_CACHE_MAX_SIZE,
        ge=1,
        description="Maximum cache size",
    )

    # OpenTelemetry parameters
    otel_enabled: bool = Field(
        default=DEFAULT_OTEL_ENABLED,
        description="Enable OpenTelemetry tracing",
    )
    otel_service_name: str = Field(
        default=DEFAULT_OTEL_SERVICE_NAME,
        description="Service name for OpenTelemetry",
    )
    otel_service_version: str = Field(
        default=DEFAULT_OTEL_SERVICE_VERSION,
        description="Service version for OpenTelemetry",
    )
    otel_endpoint: str | None = Field(
        default=None,
        description="OpenTelemetry collector endpoint",
    )

    # Vector backend
    vector_backend: str = Field(
        default="qdrant",
        description="Vector database backend (qdrant, pgvector)",
    )
    vector_dimension: int = Field(
        default=384,
        ge=1,
        description="Dimension of embedding vectors",
    )

    def get_layer_config(self, layer: str) -> dict[str, Any]:
        """Get configuration for a specific memory layer.

        Args:
            layer: Layer name (sensory, working, episodic, semantic)

        Returns:
            Dictionary with layer configuration
        """
        layer_configs: dict[str, dict[str, Any]] = {
            "sensory": {
                "max_size": self.sensory_max_size,
                "decay_rate": self.decay_rate,
                "min_importance": self.min_importance,
            },
            "working": {
                "max_size": self.working_max_size,
                "decay_rate": self.decay_rate,
                "min_importance": self.min_importance,
            },
            "episodic": {
                "max_size": self.episodic_max_size,
                "decay_rate": self.decay_rate,
                "min_importance": self.min_importance,
            },
            "semantic": {
                "max_size": self.semantic_max_size,
                "decay_rate": 1.0,  # No decay for semantic memories
                "min_importance": self.promotion_threshold,
            },
        }
        return layer_configs.get(layer, {})

    def get_llm_config(self) -> dict[str, Any]:
        """Get LLM configuration.

        Returns:
            Dictionary with LLM settings
        """
        return {
            "temperature": self.llm_temperature,
            "max_tokens": self.llm_max_tokens,
            "timeout": self.llm_timeout,
        }

    def get_search_config(self) -> dict[str, Any]:
        """Get search configuration.

        Returns:
            Dictionary with search settings
        """
        return {
            "top_k": self.search_top_k,
            "similarity_threshold": self.similarity_threshold,
            "rerank_top_k": self.rerank_top_k,
        }

    def get_reflection_config(self) -> dict[str, Any]:
        """Get reflection configuration.

        Returns:
            Dictionary with reflection settings
        """
        return {
            "min_memories": self.min_memories_for_reflection,
            "max_age_hours": self.reflection_max_age_hours,
            "quality_threshold": self.quality_threshold,
        }
