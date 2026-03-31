"""Context Window management for working memory."""

from uuid import UUID

from rae_core.models.context import ContextWindow


class ContextWindowManager:
    """
    Manages context window for working memory.

    Handles:
    - Token counting
    - Window overflow
    - Item prioritization
    - Context compaction
    """

    def __init__(self, max_tokens: int = 4096):
        """Initialize with maximum token limit."""
        self.max_tokens = max_tokens
        self.current_window: ContextWindow | None = None

    def create_window(self) -> ContextWindow:
        """Create a new context window."""
        self.current_window = ContextWindow(max_tokens=self.max_tokens)
        return self.current_window

    def add_item(self, item_id: UUID, tokens: int) -> bool:
        """
        Add item to window if space available.

        Returns:
            True if added, False if would exceed limit
        """
        if not self.current_window:
            self.create_window()

        assert self.current_window is not None
        if self.current_window.current_tokens + tokens > self.max_tokens:
            return False

        self.current_window.items.append(item_id)
        self.current_window.current_tokens += tokens
        return True

    def remove_item(self, item_id: UUID, tokens: int) -> bool:
        """Remove item from window."""
        if not self.current_window or item_id not in self.current_window.items:
            return False

        assert self.current_window is not None
        self.current_window.items.remove(item_id)
        self.current_window.current_tokens = max(
            0, self.current_window.current_tokens - tokens
        )
        return True

    def compact(self, keep_items: list[UUID]) -> int:
        """
        Compact window by keeping only specified items.

        Returns:
            Number of items removed
        """
        if not self.current_window:
            return 0

        assert self.current_window is not None
        original_count = len(self.current_window.items)
        self.current_window.items = [
            item for item in self.current_window.items if item in keep_items
        ]
        # Re-calculate tokens if we had them, but we don't store individual tokens
        # For now, just count items
        return original_count - len(self.current_window.items)

    def get_utilization(self) -> float:
        """Get window utilization percentage (0-1)."""
        if not self.current_window:
            return 0.0
        return self.current_window.current_tokens / self.max_tokens

    def has_space_for(self, tokens: int) -> bool:
        """Check if window has space for N tokens."""
        if not self.current_window:
            return tokens <= self.max_tokens
        return (self.current_window.current_tokens + tokens) <= self.max_tokens


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Simple heuristic: ~4 chars per token for English text.
    """
    return max(1, len(text) // 4)
