"""No-LLM fallback implementation using rule-based methods."""

import re
from typing import Any

from rae_core.interfaces.llm import ILLMProvider


class NoLLMFallback(ILLMProvider):
    """Rule-based LLM fallback that doesn't require external API.

    Implements simple text processing operations without actual LLM calls.
    Useful for testing, offline mode, or when LLM budget is exhausted.
    """

    def __init__(self) -> None:
        """Initialize fallback provider."""
        self.name = "NoLLM Fallback"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate rule-based response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt (ignored)
            max_tokens: Maximum tokens (used for length limiting)
            temperature: Temperature (ignored in rule-based)
            stop_sequences: Stop sequences (ignored)

        Returns:
            Rule-based generated text
        """
        # Detect intent from prompt
        prompt_lower = prompt.lower()

        # Summarization
        if any(word in prompt_lower for word in ["summarize", "summary", "tldr"]):
            return self._extractive_summary(prompt, max_length=max_tokens * 4)

        # Entity extraction
        if any(word in prompt_lower for word in ["extract", "entities", "names"]):
            return self._extract_entities_rule_based(prompt)

        # Question answering (simple keyword match)
        if "?" in prompt:
            return self._simple_qa(prompt)

        # Default: Return first sentence or truncate
        sentences = re.split(r"[.!?]+", prompt)
        result = sentences[0].strip() if sentences else prompt[:200]
        return result + "..."

    async def generate_with_context(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate with conversation context.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            Rule-based response
        """
        # Extract last user message
        user_messages = [m for m in messages if m.get("role") == "user"]
        if not user_messages:
            return "No input provided."

        last_message = user_messages[-1].get("content", "")
        return await self.generate(last_message, max_tokens=max_tokens)

    async def count_tokens(self, text: str) -> int:
        """Estimate token count (4 chars per token heuristic)."""
        return max(1, len(text) // 4)

    def supports_function_calling(self) -> bool:
        """No function calling support in fallback."""
        return False

    async def extract_entities(self, text: str) -> list[dict[str, Any]]:
        """Extract entities using simple rules.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity dictionaries
        """
        entities = []

        # Capitalize words likely to be names (very basic)
        words = text.split()
        for word in words:
            clean_word = word.strip(".,!?;:\"'()[]{}")
            if clean_word and clean_word[0].isupper() and len(clean_word) > 2:
                entities.append(
                    {"text": clean_word, "type": "ENTITY", "confidence": 0.5}
                )

        # Numbers
        numbers = re.findall(r"\b\d+\b", text)
        for num in numbers:
            entities.append({"text": num, "type": "NUMBER", "confidence": 0.8})

        return entities[:10]  # Limit to 10

    async def summarize(self, text: str, max_length: int = 200) -> str:
        """Summarize text using extractive method.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters

        Returns:
            Extracted summary
        """
        return self._extractive_summary(text, max_length)

    def _extractive_summary(self, text: str, max_length: int = 200) -> str:
        """Extract first N sentences up to max_length."""
        sentences = re.split(r"[.!?]+", text)
        summary = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(summary) + len(sentence) <= max_length:
                summary += sentence + ". "
            else:
                break

        return summary.strip() or text[:max_length] + "..."

    def _extract_entities_rule_based(self, text: str) -> str:
        """Extract entities and return as formatted string."""
        entities = []

        # Capitalized words
        words = text.split()
        for word in words:
            if word and word[0].isupper() and len(word) > 2:
                if word not in entities:
                    entities.append(word)

        if entities:
            return "Entities found: " + ", ".join(entities[:5])
        return "No entities found."

    def _simple_qa(self, question: str) -> str:
        """Simple question answering using keyword matching."""
        question_lower = question.lower()

        # Common question patterns
        if any(
            word in question_lower
            for word in ["what", "when", "where", "who", "how", "why"]
        ):
            # Extract context after question word
            for word in ["what", "when", "where", "who", "how", "why"]:
                if word in question_lower:
                    parts = question_lower.split(word, 1)
                    if len(parts) > 1:
                        context = parts[1].strip().rstrip("?")
                        return f"Based on the question about {context}, I would need more context to provide a specific answer."

        return "I cannot provide a specific answer with the available context."
