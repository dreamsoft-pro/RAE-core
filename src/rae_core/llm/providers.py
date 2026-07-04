import httpx
import structlog
from typing import Any, Optional, Dict, List
from rae_core.interfaces.llm import ILLMProvider

logger = structlog.get_logger(__name__)

class LightweightOpenAIProvider(ILLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_with_context(messages, **kwargs)

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return True

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length]

class LightweightAnthropicProvider(ILLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com/v1", model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_with_context(messages, system_prompt=system_prompt, **kwargs)

    async def generate_with_context(self, messages: List[Dict[str, str]], system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        formatted_messages = []
        for m in messages:
            if m["role"] == "system":
                system_prompt = m["content"]
            else:
                formatted_messages.append({"role": m["role"], "content": m["content"]})
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return True

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length]

class LightweightGoogleProvider(ILLMProvider):
    def __init__(self, api_key: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta", model: str = "gemini-2.5-flash"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        contents = []
        if system_prompt:
            contents.append({"role": "system", "parts": [{"text": system_prompt}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 1000)
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, **kwargs)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return True

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length]

class BridgeLLMProvider(ILLMProvider):
    def __init__(self, api_url: str, target_agent: str = "rae-local-reasoner", strategy: str = "default"):
        self.api_url = api_url.rstrip("/")
        self.target_agent = target_agent
        self.strategy = strategy

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.generate_with_context(messages, **kwargs)

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        url = f"{self.api_url}/v2/bridge/interact"
        payload = {
            "source_agent": "rae-core-client",
            "target_agent": self.target_agent,
            "strategy": self.strategy,
            "payload": {
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            payload_out = data.get("payload", {})
            if "interaction_data" in payload_out:
                return payload_out["interaction_data"].get("text", "")
            return payload_out.get("text", "")

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return True

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length]
