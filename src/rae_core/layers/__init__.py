"""Memory layer implementations for RAE-core.

This module provides concrete implementations of the 4-layer memory architecture:
- SensoryLayer: Short-term buffer with automatic decay (seconds to minutes)
- WorkingLayer: Active context with capacity limits (items, not tokens)
- LongTermLayer: Persistent storage split into episodic and semantic
- ReflectiveLayer: Meta-cognitive insights and patterns

Each layer implements a common interface for storage, retrieval, and lifecycle management.
"""

from .base import MemoryLayerBase
from .longterm import LongTermLayer
from .reflective import ReflectiveLayer
from .sensory import SensoryLayer
from .working import WorkingLayer

__all__ = [
    "MemoryLayerBase",
    "SensoryLayer",
    "WorkingLayer",
    "LongTermLayer",
    "ReflectiveLayer",
]
