"""LLM orchestration strategies."""

from abc import ABC, abstractmethod
from typing import Any

from rae_core.interfaces.llm import ILLMProvider


class LLMStrategy(ABC):
    """Base class for LLM invocation strategies."""

    @abstractmethod
    async def execute(
        self,
        providers: dict[str, ILLMProvider],
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute LLM call using strategy.

        Args:
            providers: Available LLM providers
            prompt: Prompt to send
            **kwargs: Additional arguments for generate()

        Returns:
            Tuple of (response, provider_name_used)
        """
        pass  # pragma: no cover


class SingleLLMStrategy(LLMStrategy):
    """Use a single specified LLM provider."""

    def __init__(self, provider_name: str):
        """Initialize with provider name.

        Args:
            provider_name: Name of provider to use
        """
        self.provider_name = provider_name

    async def execute(
        self,
        providers: dict[str, ILLMProvider],
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute using single provider.

        Args:
            providers: Available providers
            prompt: Prompt text
            **kwargs: Additional arguments

        Returns:
            Tuple of (response, provider_name)
        """
        provider = providers.get(self.provider_name)
        if not provider:
            raise ValueError(f"Provider '{self.provider_name}' not found")

        response = await provider.generate(prompt, **kwargs)
        return response, self.provider_name


class FallbackStrategy(LLMStrategy):
    """Try providers in order until one succeeds."""

    def __init__(self, provider_priority: list[str]):
        """Initialize with provider priority order.

        Args:
            provider_priority: List of provider names in priority order
        """
        self.provider_priority = provider_priority

    async def execute(
        self,
        providers: dict[str, ILLMProvider],
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute with fallback.

        Args:
            providers: Available providers
            prompt: Prompt text
            **kwargs: Additional arguments

        Returns:
            Tuple of (response, provider_name_used)
        """
        last_error = None

        for provider_name in self.provider_priority:
            provider = providers.get(provider_name)
            if not provider:
                continue

            try:
                response = await provider.generate(prompt, **kwargs)
                return response, provider_name
            except Exception as e:
                last_error = e
                continue

        # All failed
        if last_error:
            raise RuntimeError(f"All providers failed. Last error: {last_error}")
        else:
            raise ValueError("No providers available")


class LoadBalancingStrategy(LLMStrategy):
    """Distribute requests across multiple providers using round-robin."""

    def __init__(self, provider_names: list[str]):
        """Initialize with provider names for load balancing.

        Args:
            provider_names: List of provider names to balance across
        """
        self.provider_names = provider_names
        self.current_index = 0

    async def execute(
        self,
        providers: dict[str, ILLMProvider],
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute with load balancing.

        Args:
            providers: Available providers
            prompt: Prompt text
            **kwargs: Additional arguments

        Returns:
            Tuple of (response, provider_name_used)
        """
        if not self.provider_names:
            raise ValueError("No providers configured for load balancing")

        # Try current provider
        attempts = 0
        while attempts < len(self.provider_names):
            provider_name = self.provider_names[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.provider_names)

            provider = providers.get(provider_name)
            if not provider:
                attempts += 1
                continue

            try:
                response = await provider.generate(prompt, **kwargs)
                return response, provider_name
            except Exception:
                # Try next provider
                attempts += 1
                continue

        raise RuntimeError("All providers failed in load balancing")


class RoundRobinStrategy(LLMStrategy):
    """Simple round-robin without fallback on failure."""

    def __init__(self, provider_names: list[str]):
        """Initialize with provider names.

        Args:
            provider_names: List of provider names
        """
        self.provider_names = provider_names
        self.current_index = 0

    async def execute(
        self,
        providers: dict[str, ILLMProvider],
        prompt: str,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """Execute round-robin selection.

        Args:
            providers: Available providers
            prompt: Prompt text
            **kwargs: Additional arguments

        Returns:
            Tuple of (response, provider_name_used)
        """
        if not self.provider_names:
            raise ValueError("No providers configured")

        provider_name = self.provider_names[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.provider_names)

        provider = providers.get(provider_name)
        if not provider:
            raise ValueError(f"Provider '{provider_name}' not found")

        response = await provider.generate(prompt, **kwargs)
        return response, provider_name
