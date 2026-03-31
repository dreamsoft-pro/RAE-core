"""
RAE Manifold Base - Unified Strategy Interface.
Defines the contract for all historical and modern reasoning strategies.
"""

import abc
from typing import Any, Dict, List, Tuple, Optional
from uuid import UUID


class ManifoldArm(abc.ABC):
    """
    Abstract base class for a mathematical reasoning arm (strategy).
    Each arm represents a specific 'theory' of memory retrieval.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = config or {}

    @abc.abstractmethod
    async def fuse(
        self,
        strategy_results: Dict[str, List[Any]],
        query: str,
        h_sys: float,
        memory_contents: Dict[UUID, Dict[str, Any]],
        weights: Dict[str, float] | None = None,
        **kwargs: Any,
    ) -> List[Tuple[UUID, float, float, Dict]]:
        """
        Execute the specific mathematical fusion logic.
        
        Args:
            strategy_results: Raw hits from individual strategies (vector, fulltext, etc.)
            query: The original search query
            h_sys: System entropy factor (log2 of corpus size)
            memory_contents: Actual memory data for deep inspection
            weights: Priority weights for providers
            
        Returns:
            Ranked list of (id, score, importance, audit_info)
        """
        pass

    def get_name(self) -> str:
        return self.__class__.__name__
