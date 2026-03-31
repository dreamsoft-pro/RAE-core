"""
Semantic Resonance Calculator (System 6.4).
Measures query entropy and abstraction level to guide routing decisions.
"""

import math


class SemanticResonance:
    """
    Calculates resonance metrics for a query.
    High Resonance = Abstract, High Entropy (Needs Vector).
    Low Resonance = Specific, Low Entropy (Needs Lexical).
    """

    def calculate(self, query: str) -> float:
        """
        Calculate normalized resonance score [0.0, 1.0].
        0.0 = Very specific (Code, IDs).
        1.0 = Very abstract (Philosophy, Open questions).
        """
        if not query:
            return 0.5

        # 1. Shannon Entropy of characters
        entropy = self._shannon_entropy(query)

        # 2. Token Diversity (Type-Token Ratio)
        tokens = query.lower().split()
        ttr = len(set(tokens)) / len(tokens) if tokens else 0.0

        # 3. Special Character Density (Penalize resonance for IDs/Code)
        special_chars = sum(1 for c in query if not c.isalnum() and not c.isspace())
        special_density = special_chars / len(query) if len(query) > 0 else 0.0

        # Formula:
        # Base is entropy (usually 2.0-5.0 for text). Normalize to 0-1 range roughly.
        # Penalize by special density (IDs have low resonance).

        normalized_entropy = min(entropy / 5.0, 1.0)

        # If density of special chars is high (e.g. > 10%), it's likely code/ID -> Low Resonance
        if special_density > 0.1:
            return 0.1

        # Combine factors
        resonance = (normalized_entropy * 0.7) + (ttr * 0.3)

        return min(max(resonance, 0.0), 1.0)

    def _shannon_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of the string."""
        prob = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
        entropy = -sum([p * math.log(p) / math.log(2.0) for p in prob])
        return entropy
