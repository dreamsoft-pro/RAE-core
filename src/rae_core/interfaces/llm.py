"""Abstract LLM provider interface for RAE-core.

This module defines the LLM interface for language model operations.
Implementations can use OpenAI, Anthropic, Ollama, or rule-based fallback.
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ILLMProvider(Protocol):
    """Abstract interface for LLM providers.

    Implementations must provide text generation capabilities for
    reflection, summarization, and other cognitive operations.
    """

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        stop_sequences: list[str] | None = None,
    ) -> str:
        """Generate text completion.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            stop_sequences: Optional stop sequences

        Returns:
            Generated text
        """
        ...

    async def generate_with_context(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate text with conversation context.

        Args:
            messages: List of messages [{"role": "user"|"assistant"|"system", "content": str}]
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        ...

    async def count_tokens(
        self,
        text: str,
    ) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        ...

    def supports_function_calling(self) -> bool:
        """Check if provider supports function calling.

        Returns:
            True if function calling is supported
        """
        ...

    async def extract_entities(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Extract entities from text using LLM.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity dictionaries with 'text', 'type', 'confidence'
        """
        ...

    async def summarize(
        self,
        text: str,
        max_length: int = 200,
    ) -> str:
        """Summarize text.

        Args:
            text: Text to summarize
            max_length: Maximum summary length in characters

        Returns:
            Summary text
        """
        ...
