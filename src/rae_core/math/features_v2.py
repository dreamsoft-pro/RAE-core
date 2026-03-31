"""
Extended features for Policy v2

Adds derived features and quality tracking for data-driven decisions.
"""

from dataclasses import dataclass, field

import numpy as np

from rae_core.math.features import Features
from rae_core.math.types import TaskType


@dataclass
class FeaturesV2(Features):
    """
    Extended features for Policy v2 with historical tracking and derived metrics.

    Backwards compatible with Features - adds new fields without removing old ones.
    """

    # Historical performance
    recent_quality_scores: list[float] = field(
        default_factory=list
    )  # Last N quality scores
    quality_trend: float = 0.0  # MRR trend over last N queries
    error_rate_recent: float = 0.0  # Error rate in last N operations

    # Level history and stability
    level_history: list[str] = field(default_factory=list)  # Last N levels used
    consecutive_same_level: int = 0  # Stability indicator

    # Task complexity
    query_complexity: float = 0.0  # Estimated query difficulty (0-1)
    time_since_reflection: int = 0  # Operations since last reflection

    # Graph features
    graph_connectivity: float = 0.0  # From GCS metric

    # First turn indicator
    is_first_turn: bool = False

    # Content-aware features (System 3.4 Revival)
    term_density: float = 0.0  # Ratio of unique terms to total tokens
    keyword_ratio: float = 0.0  # Ratio of capitalized/special tokens
    entropy: float = 0.0  # Shannon entropy of the query
    is_industrial: bool = False  # Flag for industrial context
    is_quantitative: bool = False  # Flag for quantitative queries
    symbols: list[str] = field(default_factory=list)  # Technical symbols/IDs detected

    def compute_derived_features(self) -> dict[str, float]:
        """
        Compute derived features for decision making.

        Returns normalized features in range [0, 1] where possible.
        """
        return {
            # Scale features (normalized to 0-1)
            "memory_scale": min(self.memory_count / 1000.0, 1.0),
            "session_scale": min(self.session_length / 50.0, 1.0),
            # Entropy normalization (assume max entropy ~4.0)
            "entropy_normalized": min(self.memory_entropy / 4.0, 1.0),
            # Content-aware features for Industrial/Log detection
            "term_density": self.term_density,
            "keyword_ratio": self.keyword_ratio,
            "is_quantitative": 1.0 if self.is_quantitative else 0.0,
            # Quality indicators
            "quality_declining": 1.0 if self.quality_trend < -0.1 else 0.0,
            "quality_improving": 1.0 if self.quality_trend > 0.1 else 0.0,
            # Stability indicators
            "level_stable": 1.0 if self.consecutive_same_level >= 5 else 0.0,
            "needs_reflection": 1.0 if self.time_since_reflection > 100 else 0.0,
            # Graph connectivity indicator
            "has_graph": 1.0 if self.graph_connectivity > 0.1 else 0.0,
            # Budget pressure
            "budget_tight": (
                1.0 if (self.cost_budget and self.cost_budget < 0.01) else 0.0
            ),
            "latency_tight": (
                1.0
                if (self.latency_budget_ms and self.latency_budget_ms < 100)
                else 0.0
            ),
            # First turn flag
            "first_turn": 1.0 if self.is_first_turn else 0.0,
        }

    def get_task_affinity_l1(self) -> float:
        """Task type affinity for L1 (deterministic heuristic)"""
        affinities = {
            TaskType.MEMORY_RETRIEVE: 0.8,  # Fast retrieval
            TaskType.MEMORY_STORE: 0.9,  # Simple storage
            TaskType.CONTEXT_SELECT: 0.7,  # Context pruning
            TaskType.MEMORY_CONSOLIDATE: 0.4,  # Needs more sophistication
            TaskType.REFLECTION_LIGHT: 0.6,
            TaskType.REFLECTION_DEEP: 0.2,  # Needs L2/L3
            TaskType.GRAPH_UPDATE: 0.5,
        }
        return affinities.get(self.task_type, 0.5)

    def get_task_affinity_l2(self) -> float:
        """Task type affinity for L2 (information theoretic)"""
        affinities = {
            TaskType.MEMORY_RETRIEVE: 0.5,  # Good for complex queries
            TaskType.MEMORY_STORE: 0.3,
            TaskType.CONTEXT_SELECT: 0.6,
            TaskType.MEMORY_CONSOLIDATE: 0.8,  # Information bottleneck shines
            TaskType.REFLECTION_LIGHT: 0.7,
            TaskType.REFLECTION_DEEP: 0.9,  # Deep analysis benefits
            TaskType.GRAPH_UPDATE: 0.7,
        }
        return affinities.get(self.task_type, 0.5)

    def get_task_affinity_l3(self) -> float:
        """Task type affinity for L3 (adaptive hybrid)"""
        affinities = {
            TaskType.MEMORY_RETRIEVE: 0.3,  # Overkill usually
            TaskType.MEMORY_STORE: 0.1,
            TaskType.CONTEXT_SELECT: 0.4,
            TaskType.MEMORY_CONSOLIDATE: 0.7,
            TaskType.REFLECTION_LIGHT: 0.5,
            TaskType.REFLECTION_DEEP: 1.0,  # Best for deep reflection
            TaskType.GRAPH_UPDATE: 0.6,
        }
        return affinities.get(self.task_type, 0.3)

    @classmethod
    def from_features(cls, features: Features, **kwargs) -> "FeaturesV2":
        """
        Create FeaturesV2 from base Features.

        Allows easy upgrade from v1 to v2 in existing code.
        """
        # Copy all fields from base Features
        base_dict = {
            "task_type": features.task_type,
            "memory_count": features.memory_count,
            "graph_density": features.graph_density,
            "session_length": features.session_length,
            "memory_entropy": features.memory_entropy,
            "recent_mrr": features.recent_mrr,
            "recent_gcs": features.recent_gcs,
            "recent_scs": features.recent_scs,
            "cost_budget": features.cost_budget,
            "latency_budget_ms": features.latency_budget_ms,
            "previous_level": features.previous_level,
            "previous_level_success": features.previous_level_success,
            "custom": features.custom,
        }

        # Merge with new fields from kwargs
        base_dict.update(kwargs)

        return cls(**base_dict)  # type: ignore[arg-type]


class FeatureExtractorV2:
    """
    Extracts FeaturesV2 from queries and system state.
    Calculates entropy, density, and detects industrial contexts.
    """

    def extract(self, query: str) -> FeaturesV2:
        """Extract features from a query string."""
        if not query:
            return FeaturesV2(task_type=TaskType.MEMORY_RETRIEVE)

        # 1. Basic Stats
        tokens = query.split()
        token_count = len(tokens)
        unique_tokens = len(set(tokens))

        # 2. Term Density (Lexical Richness)
        term_density = unique_tokens / max(token_count, 1)

        # 3. Entropy (Shannon)
        # Prob of each token
        from collections import Counter

        counts = Counter(tokens)
        probs = [c / token_count for c in counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs)

        # 4. Keyword Ratio (Industrial detection)
        # Count words with digits, caps, or special chars
        special_tokens = sum(
            1 for t in tokens if any(c.isdigit() for c in t) or not t.isalpha()
        )
        keyword_ratio = special_tokens / max(token_count, 1)

        # 5. Industrial Detection
        industrial_keywords = [
            "log",
            "error",
            "warn",
            "info",
            "uuid",
            "0x",
            "exception",
            "trace",
            "stack",
            "deploy",
            "server",
            "db",
            "sql",
            "http",
            "status",
            "code",
            "latency",
            "ms",
            "sloo",
            "sla",
        ]
        has_industrial_kw = any(k in query.lower() for k in industrial_keywords)
        is_industrial = (keyword_ratio > 0.2) or has_industrial_kw

        # 6. Quantitative Detection
        quant_keywords = [
            "high",
            "low",
            "max",
            "min",
            "limit",
            "exceed",
            "percent",
            "%",
            ">",
            "<",
            "latency",
            "slow",
            "fast",
        ]
        is_quantitative = any(k in query.lower() for k in quant_keywords)

        # 7. Symbol Extraction (System 4.15)
        # Detect: UUIDs, Hex codes, technical IDs like #123, or specific industrial codes
        import re

        symbols = []
        # UUID pattern
        uuids = re.findall(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            query.lower(),
        )
        symbols.extend(uuids)

        # Technical IDs: words with digits, hyphens or underscores
        # e.g. incident_0001, srv-06, ticket_123, 0xABC
        # We look for tokens that are not plain common words
        for token in tokens:
            token_clean = token.strip(".,!?()[]{}'\"")
            # If token has digits and letters, or hyphens/underscores between chars
            if (any(c.isdigit() for c in token_clean) and any(c.isalpha() for c in token_clean)) or \
               (re.search(r"[a-z0-9]+[-_][a-z0-9]+", token_clean.lower())):
                symbols.append(token_clean.lower())

        # Hex codes
        hex_codes = re.findall(r"0x[0-9a-f]+", query.lower())
        symbols.extend(hex_codes)

        return FeaturesV2(
            task_type=TaskType.MEMORY_RETRIEVE,
            term_density=term_density,
            keyword_ratio=keyword_ratio,
            entropy=entropy,
            is_industrial=is_industrial,
            is_quantitative=is_quantitative,
            query_complexity=entropy / 4.0,  # Normalized complexity
            symbols=list(set(symbols)),  # De-duplicate
        )
