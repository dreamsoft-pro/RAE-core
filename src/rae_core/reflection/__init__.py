"""Reflection V2 module for RAE-core.

Implements the Actor-Evaluator-Reflector pattern for meta-cognitive processing.
"""

from rae_core.reflection.actor import Actor
from rae_core.reflection.engine import ReflectionEngine
from rae_core.reflection.evaluator import Evaluator
from rae_core.reflection.reflector import Reflector

__all__ = [
    "Actor",
    "Evaluator",
    "Reflector",
    "ReflectionEngine",
]
