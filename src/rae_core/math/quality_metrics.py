"""
RAE Math - Metrics System

This module defines the extensible metrics system for evaluating memory quality.
It follows a Strategy pattern to allow different implementations (heuristic,
statistical, semantic) depending on the environment (Mobile vs Server).

Core Components:
- IMetric: Abstract interface for all metrics
- MetricResult: Rich return type with score, confidence, and explanation
- QualityScorer: Aggregator for combining multiple metrics

Design Principles:
1. Pluggable: Easy to add new metrics (e.g., LLM-based)
2. Context-Aware: Metrics can use context
3. Explainable: Results include metadata/reasoning

License: Apache-2.0
Author: Grzegorz Leśniowski <lesniowskig@gmail.com>
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    """
    Standardized result from any metric computation.

    Attributes:
        score: Normalized score [0.0, 1.0]
        name: Name of the metric (e.g., "entropy", "coherence")
        confidence: Confidence in the calculation [0.0, 1.0]
        metadata: Additional details (e.g., raw counts, reasoning)
    """

    score: float
    name: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class IMetric(ABC):
    """Abstract base class for all memory quality metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the metric."""
        ...

    @abstractmethod
    def compute(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        """
        Compute the metric score.

        Args:
            content: The content to evaluate (text, list of tokens, etc.)
            context: Optional context dictionary

        Returns:
            MetricResult object
        """
        ...


class TextCoherenceMetric(IMetric):
    """
    Basic heuristic for text coherence (Local/Mobile friendly).

    Uses statistical properties of text (length, sentence structure)
    instead of expensive LLM calls.
    """

    @property
    def name(self) -> str:
        return "text_coherence"

    def compute(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        if not isinstance(content, str):
            return MetricResult(
                score=0.0,
                name=self.name,
                confidence=1.0,
                metadata={"reason": "empty_content"},
            )

        if not content.strip():
            return MetricResult(
                score=0.0,
                name=self.name,
                confidence=1.0,
                metadata={"reason": "empty_content"},
            )

        text = content.strip()
        words = text.split()
        word_count = len(words)

        # Heuristic 1: Very short texts are rarely coherent memories
        if word_count < 3:
            return MetricResult(
                score=0.2,
                name=self.name,
                metadata={"reason": "too_short", "word_count": word_count},
            )

        # Heuristic 2: Sentence structure (punctuation)
        has_punctuation = any(c in text for c in ".!?")
        capitalized = text[0].isupper()

        score = 0.5
        if word_count > 5:
            score += 0.2
        if has_punctuation:
            score += 0.1
        if capitalized:
            score += 0.1

        # Cap at 1.0
        score = min(1.0, score)

        return MetricResult(
            score=score,
            name=self.name,
            metadata={
                "word_count": word_count,
                "has_punctuation": has_punctuation,
                "is_capitalized": capitalized,
            },
        )


class EntropyMetric(IMetric):
    """
    Shannon Entropy metric for information density.

    Higher entropy = more information/surprise.
    Useful for filtering repetitive logs or boilerplate.
    """

    @property
    def name(self) -> str:
        return "shannon_entropy"

    def compute(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        tokens = []
        if isinstance(content, str):
            tokens = content.lower().split()
        elif isinstance(content, list):
            tokens = [str(t).lower() for t in content]

        if not tokens:
            return MetricResult(score=0.0, name=self.name)

        # Calculate frequency distribution
        freq: dict[str, int] = {}
        for token in tokens:
            freq[token] = freq.get(token, 0) + 1

        total = len(tokens)
        entropy = 0.0
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize: H_max = log2(N) where N is number of unique tokens
        # We use log2(len(freq)) as upper bound for perfect uniformity
        unique_count = len(freq)
        max_entropy = math.log2(unique_count) if unique_count > 1 else 1.0

        normalized_score = entropy / max_entropy if max_entropy > 0 else 0.0
        normalized_score = min(1.0, max(0.0, normalized_score))

        return MetricResult(
            score=normalized_score,
            name=self.name,
            metadata={
                "raw_entropy": entropy,
                "unique_tokens": unique_count,
                "total_tokens": total,
            },
        )


class RelevanceMetric(IMetric):
    """
    Measures relevance to context using simple keyword overlap.
    """

    @property
    def name(self) -> str:
        return "keyword_relevance"

    def compute(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        if not context or "query" not in context:
            return MetricResult(
                score=0.0,
                name=self.name,
                confidence=1.0,
                metadata={"reason": "no_query_context"},
            )

        text = str(content).lower()
        query = str(context["query"]).lower()

        text_words = set(text.split())
        query_words = set(query.split())

        if not text_words or not query_words:
            return MetricResult(score=0.0, name=self.name)

        overlap = len(text_words & query_words)
        score = overlap / len(query_words)

        return MetricResult(
            score=min(1.0, score), name=self.name, metadata={"overlap_count": overlap}
        )


class CompletenessMetric(IMetric):
    """
    Measures information completeness based on key fields.
    """

    @property
    def name(self) -> str:
        return "completeness"

    def compute(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        # Expects content to be a dictionary (memory item)
        if not isinstance(content, dict):
            # Try to parse if it's an object with __dict__ or Pydantic model
            if hasattr(content, "model_dump"):
                content = content.model_dump()
            elif hasattr(content, "__dict__"):
                content = content.__dict__
            else:
                return MetricResult(
                    score=0.0,
                    name=self.name,
                    confidence=0.0,
                    metadata={"reason": "invalid_content_type"},
                )

        required_fields = ["content", "created_at"]
        optional_fields = ["tags", "metadata", "importance"]

        required_score = sum(1 for f in required_fields if content.get(f)) / len(
            required_fields
        )
        optional_score = sum(1 for f in optional_fields if content.get(f)) / len(
            optional_fields
        )

        score = required_score * 0.7 + optional_score * 0.3

        return MetricResult(
            score=score,
            name=self.name,
            metadata={
                "missing_required": [f for f in required_fields if not content.get(f)]
            },
        )


class QualityScorer:
    """
    Aggregates multiple metrics into a final quality score.

    Allows configuring weights for different environments (e.g., Mobile
    might prioritize simple metrics, Server might use semantic ones).
    """

    def __init__(self, metrics: list[IMetric], weights: dict[str, float] | None = None):
        self.metrics = metrics
        self.weights = weights or {m.name: 1.0 for m in metrics}

        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def evaluate(
        self, content: Any, context: dict[str, Any] | None = None
    ) -> MetricResult:
        """
        Run all metrics and calculate weighted average.
        """
        results = []
        final_score = 0.0
        details = {}

        for metric in self.metrics:
            weight = self.weights.get(metric.name, 0.0)
            if weight == 0:
                continue

            res = metric.compute(content, context)
            results.append(res)
            final_score += res.score * weight
            details[metric.name] = res

        return MetricResult(
            score=min(1.0, max(0.0, final_score)),
            name="aggregated_quality",
            metadata={
                "components": {k: v.score for k, v in details.items()},
                "breakdown": details,
            },
        )
