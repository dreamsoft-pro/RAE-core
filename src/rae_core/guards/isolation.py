"""Memory isolation guard for RAE-core.

Provides post-search validation to ensure data isolation between agents and sessions.
"""

import logging
from typing import Any

from ..models.memory import MemoryItem

logger = logging.getLogger(__name__)


class MemoryIsolationGuard:
    """
    Guards against multi-layer memory interference (MMIT).

    Validates that search results belong to the correct agent and session
    before returning them to the caller.
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize the guard.

        Args:
            strict_mode: If True, raises exceptions on isolation violations.
        """
        self.strict_mode = strict_mode
        self.leak_count = 0
        self.validation_count = 0

    def validate_search_results(
        self,
        results: list[MemoryItem | dict[str, Any]],
        expected_agent_id: str,
        expected_session_id: str | None = None,
        expected_tenant_id: str | None = None,
        raise_on_leak: bool = False,
    ) -> list[MemoryItem | dict[str, Any]]:
        """
        Filter out any memories that don't match expected IDs.

        Args:
            results: List of memories retrieved from storage (objects or dicts)
            expected_agent_id: The ID of the agent performing the search
            expected_session_id: The current session ID
            expected_tenant_id: The current tenant ID
            raise_on_leak: Whether to raise an exception if a leak is detected

        Returns:
            Filtered list of memories belonging only to the expected agent/session
        """
        self.validation_count += 1
        validated = []
        leaks_detected = 0

        for memory in results:
            if self.validate_single_memory(
                memory, expected_agent_id, expected_session_id, expected_tenant_id
            ):
                validated.append(memory)
            else:
                leaks_detected += 1

        if leaks_detected > 0:
            self.leak_count += leaks_detected
            logger.error(
                f"MMIT Isolation Violation: Blocked {leaks_detected} leaking memories "
                f"for agent {expected_agent_id}"
            )
            if raise_on_leak or self.strict_mode:
                from ..exceptions.base import RAEError

                raise RAEError(
                    f"Security Violation: Cross-agent memory leak detected ({leaks_detected} items)"
                )

        return validated

    def validate_single_memory(
        self,
        memory: MemoryItem | dict[str, Any],
        expected_agent_id: str,
        expected_session_id: str | None = None,
        expected_tenant_id: str | None = None,
    ) -> bool:
        """Validate isolation for a single memory."""

        # Get values from object or dict
        if isinstance(memory, dict):
            agent_id = memory.get("agent_id")
            session_id = memory.get("session_id")
            tenant_id = memory.get("tenant_id")
            memory_id = memory.get("id")
        else:
            agent_id = getattr(memory, "agent_id", None)
            session_id = getattr(memory, "session_id", None)
            tenant_id = getattr(memory, "tenant_id", None)
            memory_id = getattr(memory, "id", None)

        # Check agent_id match
        if agent_id != expected_agent_id:
            logger.warning(
                f"MMIT LEAK DETECTED: Memory {memory_id} has wrong agent_id "
                f"({agent_id} != expected {expected_agent_id})"
            )
            return False

        # Check session_id match (only if expected_session_id is provided)
        if expected_session_id and session_id != expected_session_id:
            logger.warning(
                f"MMIT LEAK DETECTED: Memory {memory_id} has wrong session_id "
                f"({session_id} != expected {expected_session_id})"
            )
            return False

        # Check tenant_id match (only if expected_tenant_id is provided)
        if expected_tenant_id and tenant_id != expected_tenant_id:
            logger.warning(
                f"MMIT LEAK DETECTED: Memory {memory_id} has wrong tenant_id "
                f"({tenant_id} != expected {expected_tenant_id})"
            )
            return False

        return True

    def get_stats(self) -> dict[str, Any]:
        """Get guard statistics."""
        leak_rate = (
            (self.leak_count / self.validation_count)
            if self.validation_count > 0
            else 0.0
        )
        return {
            "leak_count": self.leak_count,
            "validation_count": self.validation_count,
            "leak_rate": leak_rate,
        }

    def reset_stats(self) -> None:
        """Reset guard statistics."""
        self.leak_count = 0
        self.validation_count = 0
