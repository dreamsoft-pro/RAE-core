"Context builder for assembling LLM-ready context from search results."

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID

from rae_core.context.window import ContextWindowManager, estimate_tokens
from rae_core.models.context import ContextMetadata, WorkingContext


class ContextFormat(str, Enum):
    """Format types for context assembly."""

    CONVERSATIONAL = "conversational"
    STRUCTURED = "structured"
    MINIMAL = "minimal"
    DETAILED = "detailed"


class ContextBuilder:
    """
    Builds LLM-ready context from memory search results.

    Features:
    - Token-aware context assembly
    - Priority-based memory ranking
    - Multiple formatting templates
    - Automatic truncation on overflow
    - Metadata preservation
    """

    def __init__(
        self,
        max_tokens: int = 4096,
        default_format: ContextFormat = ContextFormat.CONVERSATIONAL,
    ):
        """Initialize context builder.

        Args:
            max_tokens: Maximum tokens for assembled context
            default_format: Default context format template
        """
        self.max_tokens = max_tokens
        self.default_format = default_format
        self.window_manager = ContextWindowManager(max_tokens=max_tokens)

    def build_context(
        self,
        memories: list[dict[str, Any]],
        query: str | None = None,
        format_type: ContextFormat | None = None,
        max_memories: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[str, ContextMetadata]:
        """Build context from search results."""
        format_type = format_type or self.default_format

        # Rank memories by priority
        ranked_memories = self._rank_memories(memories)

        # Apply max_memories limit if specified
        if max_memories:
            ranked_memories = ranked_memories[:max_memories]

        # Build context with token management
        context_parts = []
        included_memories = []
        total_tokens = 0

        # Add query header if provided
        if query:
            query_header = self._format_query_header(query, format_type)
            query_tokens = estimate_tokens(query_header)
            if query_tokens <= self.max_tokens:
                context_parts.append(query_header)
                total_tokens += query_tokens

        # Add memories until token limit
        for memory in ranked_memories:
            memory_text = self._format_memory(
                memory, format_type, include_metadata=include_metadata
            )
            memory_tokens = estimate_tokens(memory_text)

            if total_tokens + memory_tokens > self.max_tokens:
                break

            context_parts.append(memory_text)
            included_memories.append(memory)
            total_tokens += memory_tokens

            memory_id = memory.get("id")
            if memory_id and isinstance(memory_id, (str, UUID)):
                if isinstance(memory_id, str):
                    memory_id = UUID(memory_id)
                self.window_manager.add_item(memory_id, memory_tokens)

        context = self._assemble_context(context_parts, format_type)

        avg_tokens = (
            int(total_tokens / len(included_memories)) if included_memories else 0
        )

        metadata = ContextMetadata(
            total_items=len(memories),
            active_items=len(included_memories),
            token_usage=total_tokens,
            last_compaction=(
                datetime.now(timezone.utc)
                if len(included_memories) < len(memories)
                else None
            ),
            statistics={
                "format": format_type,
                "truncated": len(included_memories) < len(memories),
                "query_provided": query is not None,
                "avg_tokens_per_memory": avg_tokens,
            },
        )

        return context, metadata

    def build_working_context(
        self,
        tenant_id: str,
        agent_id: str,
        memories: list[dict[str, Any]],
        focus_items: list[UUID] | None = None,
    ) -> WorkingContext:
        """Build a complete WorkingContext object."""
        focus_items = focus_items or []
        context_text, metadata = self.build_context(memories)

        priority_score = self._calculate_priority_score(memories, focus_items)

        # Ensure window manager has a window
        if not self.window_manager.current_window:
            self.window_manager.create_window()

        return WorkingContext(
            tenant_id=tenant_id,
            agent_id=agent_id,
            window=self.window_manager.current_window,
            focus_items=focus_items,
            priority_score=priority_score,
            metadata=metadata.model_dump(),
        )

    def _calculate_priority_score(
        self, memories: list[dict[str, Any]], focus_items: list[UUID] | None = None
    ) -> float:
        """Calculate overall priority score for the context."""
        if not memories:
            return 0.0

        focus_items = focus_items or []
        total_score = 0.0

        for memory in memories:
            importance = memory.get("importance", 0.5)
            # Boost score if memory is in focus items
            mem_id = memory.get("id")
            if mem_id and (
                mem_id in focus_items or str(mem_id) in [str(f) for f in focus_items]
            ):
                importance = min(1.0, importance * 1.5)
            total_score += importance

        avg_score = total_score / len(memories)

        # Ensure focus items always boost the average if present
        if focus_items and len(focus_items) > 0:
            avg_score = min(1.0, avg_score * 1.1)

        return avg_score

    def _rank_memories(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rank memories by priority (importance, recency, relevance, and layer boost)."""

        def priority_score(memory: dict[str, Any]) -> float:
            importance = memory.get("importance", 0.5)
            created_at = memory.get("created_at")
            recency_bonus = 0.0
            if created_at:
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(
                            created_at.replace("Z", "+00:00")
                        )
                    except ValueError:
                        created_at = None

                if isinstance(created_at, datetime):
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    age_hours = (
                        datetime.now(timezone.utc) - created_at
                    ).total_seconds() / 3600
                    recency_bonus = 0.3 * (1.0 / (1.0 + age_hours / 24))

            layer = memory.get("layer", "episodic")
            layer_boost = {
                "reflective": 0.4,
                "semantic": 0.2,
                "episodic": 0.0,
                "sensory": -0.2,
            }.get(layer, 0.0)

            relevance = memory.get("score", 0.5)
            return float(importance + recency_bonus + layer_boost + (relevance * 0.5))

        return sorted(memories, key=priority_score, reverse=True)

    def _format_query_header(self, query: str, format_type: ContextFormat) -> str:
        if format_type == ContextFormat.MINIMAL:
            return f"Query: {query}\n"
        elif format_type == ContextFormat.STRUCTURED:
            return f"# Query\n{query}\n\n# Relevant Memories\n"
        elif format_type == ContextFormat.DETAILED:
            return f"## Search Query\n{query}\n\n## Retrieved Memories (ranked by relevance)\n"
        else:  # CONVERSATIONAL
            return (
                f"Based on your query: '{query}'\n\nHere are the relevant memories:\n\n"
            )

    def _format_memory(
        self,
        memory: dict[str, Any],
        format_type: ContextFormat,
        include_metadata: bool = True,
    ) -> str:
        content = memory.get("content", "")
        memory_id = memory.get("id", "unknown")
        importance = memory.get("importance", 0.0)
        tags = memory.get("tags", [])

        if format_type == ContextFormat.MINIMAL:
            return f"{content}\n"
        elif format_type == ContextFormat.STRUCTURED:
            parts = [f"## Memory {memory_id}", f"{content}"]
            if include_metadata:
                parts.append(f"Importance: {importance:.2f}")
                if tags:
                    parts.append(f"Tags: {', '.join(tags)}")
            return "\n".join(parts) + "\n\n"
        elif format_type == ContextFormat.DETAILED:
            parts = [f"### Memory: {memory_id}", f"**Content:** {content}"]
            if include_metadata:
                parts.append(f"**Importance:** {importance:.2f}")
                if tags:
                    parts.append(f"**Tags:** {', '.join(tags)}")
                if memory.get("created_at"):
                    parts.append(f"**Created:** {memory.get('created_at')}")
            return "\n".join(parts) + "\n\n"
        else:  # CONVERSATIONAL
            prefix = ""
            if importance > 0.8:
                prefix = "[Important] "
            return f"- {prefix}{content}\n"

    def _assemble_context(self, parts: list[str], format_type: ContextFormat) -> str:
        return "\n".join(parts)

    def reset(self) -> None:
        self.window_manager.create_window()

    def get_statistics(self) -> dict[str, Any]:
        window_tokens = 0
        if self.window_manager.current_window:
            window_tokens = self.window_manager.current_window.current_tokens

        return {
            "max_tokens": self.max_tokens,
            "default_format": self.default_format,
            "window_utilization": self.window_manager.get_utilization(),
            "window_tokens": window_tokens,
        }
