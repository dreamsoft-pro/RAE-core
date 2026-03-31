"""LLM configuration models."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    FALLBACK = "fallback"


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider."""

    provider_type: LLMProviderType = Field(description="Type of LLM provider")
    api_key: str | None = Field(default=None, description="API key for provider")
    base_url: str | None = Field(default=None, description="Custom base URL")
    model: str = Field(description="Model identifier (e.g., gpt-4, claude-3)")
    max_tokens: int = Field(default=1000, ge=1, le=100000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout: int = Field(
        default=30, ge=1, le=300, description="Request timeout in seconds"
    )
    max_retries: int = Field(default=3, ge=0, le=10)
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata"
    )

    model_config = ConfigDict(use_enum_values=True)


class LLMConfig(BaseModel):
    """Configuration for LLM orchestration."""

    providers: dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Configured LLM providers by name"
    )
    default_provider: str | None = Field(
        default=None, description="Default provider name to use"
    )
    enable_fallback: bool = Field(
        default=True, description="Enable fallback to NoLLM if all providers fail"
    )
    load_balancing: bool = Field(
        default=False, description="Enable load balancing across providers"
    )
    max_concurrent_requests: int = Field(
        default=10, ge=1, le=100, description="Maximum concurrent LLM requests"
    )
    cache_responses: bool = Field(
        default=True, description="Cache LLM responses for identical prompts"
    )
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")

    def get_provider(self, name: str) -> ProviderConfig | None:
        """Get provider configuration by name."""
        return self.providers.get(name)

    def add_provider(self, name: str, config: ProviderConfig) -> None:
        """Add a provider configuration."""
        self.providers[name] = config
        if self.default_provider is None:
            self.default_provider = name

    def list_providers(self) -> list[str]:
        """List all configured provider names."""
        return list(self.providers.keys())
