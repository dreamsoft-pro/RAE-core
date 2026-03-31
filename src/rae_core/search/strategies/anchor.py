"""Anchor Search Strategy - Deterministic Exact Matching."""

import re
from typing import Any
from uuid import UUID

from ...interfaces.storage import IMemoryStorage
from . import SearchStrategy


class AnchorStrategy(SearchStrategy):
    """
    Tier 1 Search Strategy: Deterministic "Anchor" matching.

    Identifies strong entities in the query (UUIDs, Error Codes, Ticket IDs,
    Dates, Contract Signatures) and performs exact lookups.

    If an anchor is found, it guarantees retrieval regardless of semantic drift.
    """

    def __init__(self, storage: IMemoryStorage, default_weight: float = 100.0) -> None:
        self.storage = storage
        self.default_weight = default_weight

        # Regex patterns with TIERED confidence
        # Tier 1: HARD IDs (Unique signatures) -> Weight 100.0
        # Tier 2: SOFT Entities (Contextual) -> Weight 5.0
        self.patterns = {
            # Hard Anchors
            "uuid": (
                r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
                100.0,
            ),
            "error_hex": (r"\b0x[0-9A-Fa-f]{3,}\b", 100.0),  # 0x404
            "ticket_id": (
                r"\b(ticket|issue|pr|bug)[\s#_-]+(\d{3,})\b",
                100.0,
            ),  # ticket_001, PR#123 (min 3 digits)
            # Soft Anchors (Context Boosters)
            "log_level": (r"\[(ERROR|CRITICAL|WARN|INFO)\]", 5.0),
            "http_code": (r"\b[45]\d{2}\b", 5.0),  # 404, 500
            "date_iso": (r"\d{4}-\d{2}-\d{2}", 10.0),  # 2026-01-01
        }

    async def search(
        self,
        query: str,
        tenant_id: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float, float]]:
        """
        Execute anchor search. Returns matches with score 1.0 (will be boosted by weight).
        """
        anchors = self._extract_anchors(query)
        if not anchors:
            return []

        candidates = {}

        for anchor_type, value, weight_mod in anchors:
            # Search for this exact value
            # We treat the anchor value as a "quoted" search to enforce exactness if supported
            exact_query = f'"{value}"'

            results = await self.storage.search_memories(
                query=exact_query, tenant_id=tenant_id, limit=limit, **kwargs
            )

            for res in results:
                # Structure from storage: {'id': ..., 'content': ..., 'score': ...}
                # Adapter compatibility check
                m_id = res.get("id")
                if isinstance(m_id, str):
                    m_id = UUID(m_id)

                # Apply tiered weight immediately or return relative boost
                # Here we return a boost factor that Engine will multiply by default_weight
                # If default_weight is 100, then boost 1.0 = 100, boost 0.05 = 5

                current_boost = weight_mod / self.default_weight

                # Maximize score if multiple anchors hit same doc
                if m_id in candidates:
                    candidates[m_id] = max(candidates[m_id], current_boost)
                else:
                    candidates[m_id] = current_boost

        # Sort by boost score
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return [(mid, score, 0.0) for mid, score in sorted_candidates[:limit]]

    def _extract_anchors(self, query: str) -> list[tuple[str, str, float]]:
        """Extract all matching anchors from query."""
        found = []
        for name, (pattern, weight) in self.patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for m in matches:
                # Handle groups (ticket_id returns tuple)
                if isinstance(m, tuple):
                    # Join specific groups e.g. ("ticket", "001") -> "ticket 001"
                    # We normalize separators for search
                    val = f"{m[0]} {m[1]}"
                    # Also try without space
                    val_alt = f"{m[0]}{m[1]}"
                    found.append((name, val, weight))
                    found.append((name, val_alt, weight))
                else:
                    val = m
                    found.append((name, val, weight))
        return found

    def get_strategy_name(self) -> str:
        return "anchor"

    def get_strategy_weight(self) -> float:
        return self.default_weight
