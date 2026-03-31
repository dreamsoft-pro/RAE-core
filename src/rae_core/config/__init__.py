"""Configuration module for RAE-core."""

from rae_core.config.defaults import (
    DEFAULT_CONSOLIDATION_THRESHOLD,
    DEFAULT_DECAY_INTERVAL_HOURS,
    DEFAULT_DECAY_RATE,
    DEFAULT_EPISODIC_SIZE,
    DEFAULT_MIN_IMPORTANCE,
    DEFAULT_PROMOTION_THRESHOLD,
    DEFAULT_SEMANTIC_SIZE,
    DEFAULT_SENSORY_SIZE,
    DEFAULT_WORKING_SIZE,
    get_default_llm_config,
    get_default_memory_layer_config,
    get_default_reflection_config,
    get_default_search_config,
)
from rae_core.config.settings import RAESettings

__all__ = [
    "RAESettings",
    "DEFAULT_SENSORY_SIZE",
    "DEFAULT_WORKING_SIZE",
    "DEFAULT_EPISODIC_SIZE",
    "DEFAULT_SEMANTIC_SIZE",
    "DEFAULT_DECAY_RATE",
    "DEFAULT_DECAY_INTERVAL_HOURS",
    "DEFAULT_MIN_IMPORTANCE",
    "DEFAULT_PROMOTION_THRESHOLD",
    "DEFAULT_CONSOLIDATION_THRESHOLD",
    "get_default_memory_layer_config",
    "get_default_llm_config",
    "get_default_search_config",
    "get_default_reflection_config",
]
