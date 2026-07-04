from pydantic import BaseModel, Field

class CapabilityMatrix(BaseModel):
    chat: bool = Field(default=True, description="Supports chat/completions")
    json_schema: bool = Field(default=False, description="Supports structured outputs / JSON mode")
    tools: bool = Field(default=False, description="Supports tool calling / function calling")
    vision: bool = Field(default=False, description="Supports image inputs")
    embeddings: bool = Field(default=False, description="Supports embedding generation")
    streaming: bool = Field(default=True, description="Supports chunked/streamed token output")
    reasoning: bool = Field(default=False, description="Has advanced reasoning/thinking capacity")

class ModelDescriptor(BaseModel):
    id: str = Field(description="Unique identifier for model, e.g. gpt-4o")
    name: str = Field(description="Display/human name for model")
    provider: str = Field(description="Associated provider id, e.g. openai")
    context_window: int = Field(default=128000, description="Delineated input context size in tokens")
    max_tokens: int = Field(default=4096, description="Maximum output tokens limit")
    capabilities: CapabilityMatrix = Field(default_factory=CapabilityMatrix)
    cost: dict[str, float] = Field(
        default_factory=lambda: {"input": 1.0, "output": 5.0, "cacheRead": 0.1, "cacheWrite": 0.0},
        description="Delineated micro-pricing costs"
    )
