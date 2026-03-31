"""Default configuration values for RAE-core."""

from typing import Any

# Memory layer sizes
DEFAULT_SENSORY_SIZE = 100
DEFAULT_WORKING_SIZE = 50
DEFAULT_EPISODIC_SIZE = 500
DEFAULT_SEMANTIC_SIZE = 1000

# Memory decay parameters
DEFAULT_DECAY_RATE = 0.95
DEFAULT_DECAY_INTERVAL_HOURS = 24

# Importance thresholds
DEFAULT_MIN_IMPORTANCE = 0.1
DEFAULT_PROMOTION_THRESHOLD = 0.7
DEFAULT_CONSOLIDATION_THRESHOLD = 0.6

# Search parameters
DEFAULT_TOP_K = 10
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_RERANK_TOP_K = 5

# Reflection parameters
DEFAULT_MIN_MEMORIES_FOR_REFLECTION = 5
DEFAULT_REFLECTION_MAX_AGE_HOURS = 24
DEFAULT_QUALITY_THRESHOLD = 0.4

# LLM parameters
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 1000
DEFAULT_LLM_TIMEOUT = 30.0

# Sync parameters
DEFAULT_SYNC_BATCH_SIZE = 100
DEFAULT_SYNC_TIMEOUT = 60.0
DEFAULT_SYNC_RETRY_ATTEMPTS = 3

# Cache parameters
DEFAULT_CACHE_TTL = 3600  # 1 hour
DEFAULT_CACHE_MAX_SIZE = 1000

# OpenTelemetry parameters
DEFAULT_OTEL_ENABLED = True
DEFAULT_OTEL_SERVICE_NAME = "rae-core"
DEFAULT_OTEL_SERVICE_VERSION = "0.4.0"


def get_default_memory_layer_config() -> dict[str, Any]:
    """Get default configuration for memory layers.

    Returns:
        Dictionary with default layer configurations
    """
    return {
        "sensory": {
            "max_size": DEFAULT_SENSORY_SIZE,
            "decay_rate": DEFAULT_DECAY_RATE,
            "min_importance": DEFAULT_MIN_IMPORTANCE,
        },
        "working": {
            "max_size": DEFAULT_WORKING_SIZE,
            "decay_rate": DEFAULT_DECAY_RATE,
            "min_importance": DEFAULT_MIN_IMPORTANCE,
        },
        "episodic": {
            "max_size": DEFAULT_EPISODIC_SIZE,
            "decay_rate": DEFAULT_DECAY_RATE,
            "min_importance": DEFAULT_MIN_IMPORTANCE,
        },
        "semantic": {
            "max_size": DEFAULT_SEMANTIC_SIZE,
            "decay_rate": 1.0,  # No decay for semantic memories
            "min_importance": DEFAULT_PROMOTION_THRESHOLD,
        },
    }


def get_default_llm_config() -> dict[str, Any]:
    """Get default LLM configuration.

    Returns:
        Dictionary with default LLM settings
    """
    return {
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "max_tokens": DEFAULT_LLM_MAX_TOKENS,
        "timeout": DEFAULT_LLM_TIMEOUT,
    }


def get_default_search_config() -> dict[str, Any]:
    """Get default search configuration.

    Returns:
        Dictionary with default search settings
    """
    return {
        "top_k": DEFAULT_TOP_K,
        "similarity_threshold": DEFAULT_SIMILARITY_THRESHOLD,
        "rerank_top_k": DEFAULT_RERANK_TOP_K,
    }


def get_default_reflection_config() -> dict[str, Any]:
    """Get default reflection configuration.

    Returns:
        Dictionary with default reflection settings
    """
    return {
        "min_memories": DEFAULT_MIN_MEMORIES_FOR_REFLECTION,
        "max_age_hours": DEFAULT_REFLECTION_MAX_AGE_HOURS,
        "quality_threshold": DEFAULT_QUALITY_THRESHOLD,
    }
