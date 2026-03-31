"""RAE Search - Fusion Strategies for Hybrid Retrieval."""

from abc import ABC, abstractmethod
from uuid import UUID


class FusionStrategy(ABC):
    """Base class for hybrid result fusion."""

    @abstractmethod
    def fuse(
        self,
        strategy_results: dict[str, list[tuple[UUID, float]]],
        weights: dict[str, float],
    ) -> list[tuple[UUID, float]]:
        pass


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion (Standard RAG approach)."""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, strategy_results, weights):
        scores: dict[UUID, float] = {}
        for name, results in strategy_results.items():
            w = weights.get(name, 1.0)
            for rank, (m_id, _) in enumerate(results, 1):
                scores[m_id] = scores.get(m_id, 0.0) + (w / (self.k + rank))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class ConfidenceWeightedFusion(FusionStrategy):
    """
    RAE-ORB: Advanced Fusion with Dynamic Confidence Adjustment.

    Analyzes the 'gap' and 'z-score' of results to trust confident strategies more.
    """

    def _analyze_confidence(self, results: list[tuple[UUID, float]]) -> float:
        if not results or len(results) < 2:
            return 0.0
        scores = [r[1] for r in results]
        gap = scores[0] - scores[1]
        mean = sum(scores) / len(scores)
        std = (sum((x - mean) ** 2 for x in scores) / len(scores)) ** 0.5
        z_score = (scores[0] - mean) / std if std > 0 else 0.0
        # More selective: gap matters most, z-score is secondary
        return min(1.0, max(0.0, (gap * 1.5) + (z_score / 10.0)))

    def fuse(self, strategy_results, weights):
        unified_scores: dict[UUID, float] = {}
        confidences = {
            name: self._analyze_confidence(res)
            for name, res in strategy_results.items()
        }

        for name, results in strategy_results.items():
            if not results:
                continue
            # Apply dynamic boost based on confidence
            w = weights.get(name, 1.0) * (1.0 + confidences[name])
            max_s = max(r[1] for r in results) or 1.0

            for m_id, score in results:
                normalized = (score / max_s) * w
                unified_scores[m_id] = max(unified_scores.get(m_id, 0.0), normalized)

        return sorted(unified_scores.items(), key=lambda x: x[1], reverse=True)
