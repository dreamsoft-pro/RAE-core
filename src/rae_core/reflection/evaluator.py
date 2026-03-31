"""Evaluator component for Reflection V2.

The Evaluator assesses the quality and outcomes of actions.
"""

from typing import Any, cast
from uuid import UUID

from rae_core.interfaces.storage import IMemoryStorage
from rae_core.math.quality_metrics import (
    CompletenessMetric,
    EntropyMetric,
    QualityScorer,
    RelevanceMetric,
    TextCoherenceMetric,
)


class Evaluator:
    """Evaluator component that assesses memory quality and action outcomes.

    Implements the "Evaluate" phase of the Actor-Evaluator-Reflector pattern.
    Uses the extensible Metric System for quality assessment.
    """

    def __init__(
        self,
        memory_storage: IMemoryStorage,
    ):
        """Initialize Evaluator.

        Args:
            memory_storage: Memory storage for retrieval
        """
        self.memory_storage = memory_storage

        # Initialize scorer with standard metrics
        self.scorer = QualityScorer(
            [
                TextCoherenceMetric(),
                EntropyMetric(),
                RelevanceMetric(),
                CompletenessMetric(),
            ]
        )

    async def evaluate_memory_quality(
        self,
        memory_id: UUID,
        tenant_id: str,
        context: str | None = None,
    ) -> dict[str, float]:
        """Evaluate quality of a single memory.

        Args:
            memory_id: Memory identifier
            tenant_id: Tenant identifier
            context: Optional context string (treated as query for relevance)

        Returns:
            Dictionary of quality metrics
        """
        memory = await self.memory_storage.get_memory(memory_id, tenant_id)
        if not memory:
            return {"error": 1.0}

        # Prepare evaluation context
        eval_context = {}
        if context:
            eval_context["query"] = context

        # Evaluate content quality (text)
        content_result = self.scorer.evaluate(memory.get("content", ""), eval_context)

        # Evaluate completeness (dict)
        completeness_metric = CompletenessMetric()
        completeness_result = completeness_metric.compute(memory)

        # Merge results (simplified for now)
        # In a real scenario, we might want to run scorer on memory dict as well
        # or have separate scorers for content vs structure.

        metrics = content_result.metadata.get("components", {})
        metrics["quality"] = content_result.score
        metrics["completeness"] = completeness_result.score

        # Recalculate overall quality including completeness
        metrics["quality"] = (metrics["quality"] * 0.7) + (
            completeness_result.score * 0.3
        )

        return cast(dict[str, float], metrics)

    async def evaluate_action_outcome(
        self,
        action_type: str,
        action_result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate the outcome of an action.

        Args:
            action_type: Type of action executed
            action_result: Result from action execution
            context: Original action context

        Returns:
            Evaluation result with success metrics
        """
        if not action_result.get("success"):
            return {
                "outcome": "failure",
                "score": 0.0,
                "reason": action_result.get("error", "Unknown error"),
            }

        # Evaluate based on action type
        if action_type == "consolidate_memories":
            return self._evaluate_consolidation(action_result, context)
        elif action_type == "update_importance":
            return self._evaluate_importance_update(action_result, context)
        elif action_type == "create_semantic_link":
            return self._evaluate_link_creation(action_result, context)
        elif action_type == "prune_duplicates":
            return self._evaluate_pruning(action_result, context)
        else:
            return {
                "outcome": "success",
                "score": 0.5,
                "reason": "Unknown action type",
            }

    def _evaluate_consolidation(
        self,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate memory consolidation."""
        consolidated_count = result.get("consolidated_count", 0)

        if consolidated_count < 2:
            score = 0.3  # Low value if few memories consolidated
        elif consolidated_count < 5:
            score = 0.7
        else:
            score = 1.0  # High value for many memories

        return {
            "outcome": "success",
            "score": score,
            "reason": f"Consolidated {consolidated_count} memories",
            "metrics": {
                "consolidated_count": consolidated_count,
            },
        }

    def _evaluate_importance_update(
        self,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate importance score updates."""
        updated_count = result.get("updated_count", 0)
        total = result.get("total", 0)

        if total == 0:
            score = 0.0
        else:
            score = updated_count / total

        return {
            "outcome": "success",
            "score": score,
            "reason": f"Updated {updated_count}/{total} importance scores",
            "metrics": {
                "success_rate": score,
            },
        }

    def _evaluate_link_creation(
        self,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate semantic link creation."""
        link = result.get("link", {})

        return {
            "outcome": "success",
            "score": 0.8,  # Fixed score for successful link
            "reason": f"Created {link.get('type', 'link')} between memories",
            "metrics": {
                "link_type": link.get("type"),
            },
        }

    def _evaluate_pruning(
        self,
        result: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate memory pruning."""
        pruned_count = result.get("pruned_count", 0)
        total = result.get("total", 0)

        if total == 0:
            score = 0.0
        else:
            score = pruned_count / total

        return {
            "outcome": "success",
            "score": score,
            "reason": f"Pruned {pruned_count}/{total} memories",
            "metrics": {
                "pruning_rate": score,
            },
        }

    async def evaluate_memory_batch(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
        context: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate quality of multiple memories.

        Args:
            memory_ids: List of memory identifiers
            tenant_id: Tenant identifier
            context: Optional context for relevance

        Returns:
            Aggregated quality metrics
        """
        total_metrics = {
            "quality": 0.0,
        }

        count = 0
        for memory_id in memory_ids:
            metrics = await self.evaluate_memory_quality(memory_id, tenant_id, context)
            if "error" not in metrics:
                total_metrics["quality"] += metrics.get("quality", 0.0)
                # Aggregate other keys if present
                for k, v in metrics.items():
                    if k != "quality" and isinstance(v, (int, float)):
                        total_metrics[k] = total_metrics.get(k, 0.0) + v
                count += 1

        # Average
        if count > 0:
            for key in total_metrics:
                total_metrics[key] /= count

        return {
            "evaluated_count": count,
            "total": len(memory_ids),
            "metrics": total_metrics,
        }


class SanityChecker:
    """Checks for logical contradictions and temporal impossibilities."""

    def check_for_contradictions(
        self,
        inputs: list[str],
    ) -> tuple[bool, list[str]]:
        """Check inputs for obvious contradictions.

        Args:
            inputs: List of text statements to check

        Returns:
            Tuple (is_sane, list_of_issues)
        """
        issues = []

        # 1. Check for direct negations (A vs not A)
        # Simplified heuristic for now
        for i, stmt1 in enumerate(inputs):
            for stmt2 in inputs[i + 1 :]:
                if self._are_contradictory(stmt1, stmt2):
                    issues.append(f"Contradiction found: '{stmt1}' vs '{stmt2}'")

        return len(issues) == 0, issues

    def _are_contradictory(self, stmt1: str, stmt2: str) -> bool:
        """Check if two statements are contradictory."""
        s1 = stmt1.lower().strip()
        s2 = stmt2.lower().strip()

        # Simple negation check "is X" vs "is not X"
        # This is very basic, but serves the initial implementation requirement
        if f"not {s1}" in s2 or f"not {s2}" in s1:
            return True

        # Check "is X" vs "is not X" pattern specifically
        if " is " in s1 and " is not " in s2:
            key1 = s1.split(" is ")[1]
            key2 = s2.split(" is not ")[1]
            if key1 == key2:
                return True

        if " is not " in s1 and " is " in s2:
            key1 = s1.split(" is not ")[1]
            key2 = s2.split(" is ")[1]
            if key1 == key2:
                return True

        return False
