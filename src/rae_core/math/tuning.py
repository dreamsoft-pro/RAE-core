"""
RAE Math - Bayesian Policy Tuner

This module implements high-precision weight optimization using
Recursive Bayesian Estimation (Dirichlet-Multinomial conjugate prior).

It treats retrieval weights (alpha, beta, gamma) as a probability simplex
that evolves based on interaction evidence.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class TuningResult:
    new_weights: dict[str, float]
    confidence: float
    entropy: float


class BayesianPolicyTuner:
    """
    Advanced tuner using Thompson Sampling on a Dirichlet Distribution.
    Designed for high-stability optimization in agentic memory systems.
    """

    def __init__(self, baseline_alpha: float = 5.0):
        # Concentration parameters for Dirichlet distribution
        # Higher alpha = higher stability (more evidence needed to shift)
        self.baseline_alpha = baseline_alpha

    def compute_posterior(
        self,
        current_weights: dict[str, float],
        feedback_loop: list[dict],
        learning_rate: float = 1.0,
    ) -> TuningResult:
        """
        Update weights using Bayesian inference.

        Args:
            current_weights: Current alpha, beta, gamma
            feedback_loop: List of dicts with {'score': float, 'weights': dict}
            learning_rate: How much weight to give to new evidence
        """
        # Prior parameters (alpha_0)
        # We transform current weights back to pseudo-counts
        alphas = (
            np.array(
                [
                    current_weights.get("alpha", 0.5),
                    current_weights.get("beta", 0.3),
                    current_weights.get("gamma", 0.2),
                ]
            )
            * self.baseline_alpha
        )

        # Aggregate evidence
        for event in feedback_loop:
            score = event.get("score", 0.0)
            # We only learn from significant signals
            if abs(score) < 0.1:
                continue

            # If positive feedback, increment the weights that produced it
            # If negative, we decrement (with floor at 0.1)
            weights_used = np.array(
                [
                    event["weights"].get("alpha", 0.5),
                    event["weights"].get("beta", 0.3),
                    event["weights"].get("gamma", 0.2),
                ]
            )

            # Bayes Update Step
            # Positive feedback increases concentration, Negative decreases it
            alphas += weights_used * score * learning_rate

        # Ensure positivity
        alphas = np.maximum(alphas, 0.5)

        # Compute Expected Value (Mean of Dirichlet)
        total_alpha = np.sum(alphas)
        expected_weights = alphas / total_alpha

        # Apply Section 14.2 Guardrails (0.05 - 0.85)
        clamped_weights = np.clip(expected_weights, 0.05, 0.85)
        # Re-normalize after clamping
        final_weights = clamped_weights / np.sum(clamped_weights)

        # Calculate Confidence (Inverse of Variance)
        # Variance of Dirichlet: Var(X_i) = (alpha_i * (S - alpha_i)) / (S^2 * (S + 1))
        # Lower variance = Higher confidence
        variance = (alphas * (total_alpha - alphas)) / (
            total_alpha**2 * (total_alpha + 1)
        )
        confidence = 1.0 - float(np.mean(variance)) * 10.0  # Normalized

        return TuningResult(
            new_weights={
                "alpha": float(final_weights[0]),
                "beta": float(final_weights[1]),
                "gamma": float(final_weights[2]),
            },
            confidence=max(0.0, min(1.0, confidence)),
            entropy=float(-np.sum(final_weights * np.log(final_weights + 1e-9))),
        )

    def calculate_intent_adjustment(self, query: str) -> dict[str, float]:
        """
        Heuristic adjustment based on query linguistic complexity (Entropy).
        Specific queries boost Similarity, Vague queries boost Importance.
        """
        words = query.lower().split()
        length = len(words)

        # Simple intent entropy proxy
        if length <= 2:  # "Vague"
            return {"beta": 0.1, "alpha": -0.05, "gamma": -0.05}
        elif "?" in query or length > 6:  # "Specific/Complex"
            return {"alpha": 0.1, "beta": -0.05, "gamma": -0.05}

        return {"alpha": 0.0, "beta": 0.0, "gamma": 0.0}
