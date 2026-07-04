"""LLM orchestration module for RAE-core.

Provides LLM provider management, fallback strategies, and load balancing.
"""

from rae_core.llm.config import LLMConfig, LLMProviderType, ProviderConfig
from rae_core.llm.fallback import NoLLMFallback
from rae_core.llm.orchestrator import LLMOrchestrator
from rae_core.llm.strategies import (
    FallbackStrategy,
    LLMStrategy,
    LoadBalancingStrategy,
    RoundRobinStrategy,
    SingleLLMStrategy,
)
from rae_core.llm.descriptor import CapabilityMatrix, ModelDescriptor
from rae_core.llm.runtime import resolve_llm_runtime

__all__ = [
    "LLMConfig",
    "LLMProviderType",
    "ProviderConfig",
    "NoLLMFallback",
    "LLMOrchestrator",
    "LLMStrategy",
    "SingleLLMStrategy",
    "FallbackStrategy",
    "LoadBalancingStrategy",
    "RoundRobinStrategy",
    "CapabilityMatrix",
    "ModelDescriptor",
    "resolve_llm_runtime",
]
