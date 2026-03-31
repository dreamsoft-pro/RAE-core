"""LLM Orchestrator for managing multiple LLM providers."""

from typing import Any

from rae_core.interfaces.cache import ICacheProvider
from rae_core.interfaces.llm import ILLMProvider
from rae_core.llm.config import LLMConfig
from rae_core.llm.fallback import NoLLMFallback
from rae_core.llm.strategies import LLMStrategy, SingleLLMStrategy


class LLMOrchestrator:
    """Orchestrates LLM calls across multiple providers.

    Manages provider selection, fallback, load balancing, and caching.
    """

    def __init__(
        self,
        config: LLMConfig,
        providers: dict[str, ILLMProvider] | None = None,
        cache: ICacheProvider | None = None,
    ):
        """Initialize LLM orchestrator.

        Args:
            config: LLM configuration
            providers: Pre-configured provider instances
            cache: Optional cache provider for response caching
        """
        self.config = config
        self.providers: dict[str, ILLMProvider] = providers or {}
        self.cache = cache
        self.fallback_provider = NoLLMFallback()

        # Add fallback provider if enabled
        if config.enable_fallback:
            self.providers["fallback"] = self.fallback_provider

    def register_provider(self, name: str, provider: ILLMProvider) -> None:
        """Register an LLM provider.

        Args:
            name: Provider name
            provider: Provider instance
        """
        self.providers[name] = provider

    def get_provider(self, name: str) -> ILLMProvider | None:
        """Get provider by name.

        Args:
            name: Provider name

        Returns:
            Provider instance or None
        """
        return self.providers.get(name)

    async def generate(
        self,
        prompt: str,
        provider_name: str | None = None,
        strategy: LLMStrategy | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_cache: bool = True,
    ) -> tuple[str, str]:
        """Generate text using configured strategy.

        Args:
            prompt: User prompt
            provider_name: Optional specific provider to use
            strategy: Optional custom strategy
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            use_cache: Whether to use cache

        Returns:
            Tuple of (response, provider_name_used)
        """
        # Check cache first
        if use_cache and self.cache and self.config.cache_responses:
            cache_key = self._generate_cache_key(
                prompt, system_prompt, max_tokens, temperature
            )
            cached = await self.cache.get(cache_key)
            if cached:
                return cached["response"], cached["provider"]

        # Determine strategy
        if strategy is None:
            # Use single provider strategy
            target_provider = provider_name or self.config.default_provider
            if not target_provider:
                raise ValueError("No provider specified and no default configured")
            strategy = SingleLLMStrategy(target_provider)

        # Execute strategy
        try:
            response, used_provider = await strategy.execute(
                providers=self.providers,
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            # Try fallback if enabled
            if self.config.enable_fallback and "fallback" in self.providers:
                fallback = self.providers["fallback"]
                response = await fallback.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                used_provider = "fallback"
            else:
                raise e

        # Cache result
        if use_cache and self.cache and self.config.cache_responses:
            cache_key = self._generate_cache_key(
                prompt, system_prompt, max_tokens, temperature
            )
            await self.cache.set(
                cache_key,
                {"response": response, "provider": used_provider},
                ttl=self.config.cache_ttl,
            )

        return response, used_provider

    async def generate_with_context(
        self,
        messages: list[dict[str, str]],
        provider_name: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> tuple[str, str]:
        """Generate with conversation context.

        Args:
            messages: List of messages
            provider_name: Optional specific provider
            max_tokens: Maximum tokens
            temperature: Temperature

        Returns:
            Tuple of (response, provider_name_used)
        """
        target_provider = provider_name or self.config.default_provider
        if not target_provider:
            raise ValueError("No provider specified and no default configured")

        provider = self.providers.get(target_provider)
        if not provider:
            raise ValueError(f"Provider '{target_provider}' not found")

        response = await provider.generate_with_context(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return response, target_provider

    async def extract_entities(
        self,
        text: str,
        provider_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract entities from text.

        Args:
            text: Text to extract entities from
            provider_name: Optional specific provider

        Returns:
            List of entity dictionaries
        """
        target_provider = provider_name or self.config.default_provider or "fallback"
        provider = self.providers.get(target_provider)

        if not provider:
            # Use fallback
            provider = self.fallback_provider

        return await provider.extract_entities(text)

    async def summarize(
        self,
        text: str,
        max_length: int = 200,
        provider_name: str | None = None,
    ) -> str:
        """Summarize text.

        Args:
            text: Text to summarize
            max_length: Maximum summary length
            provider_name: Optional specific provider

        Returns:
            Summary text
        """
        target_provider = provider_name or self.config.default_provider or "fallback"
        provider = self.providers.get(target_provider)

        if not provider:
            provider = self.fallback_provider

        return await provider.summarize(text, max_length)

    def _generate_cache_key(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generate cache key for request."""
        import hashlib

        key_parts = [
            prompt,
            system_prompt or "",
            str(max_tokens),
            f"{temperature:.2f}",
        ]
        key_string = "|".join(key_parts)
        return f"llm:{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"

    def list_providers(self) -> list[str]:
        """List all registered providers."""
        return list(self.providers.keys())
