import os
import httpx
import structlog
from typing import Optional
from rae_core.interfaces.llm import ILLMProvider
from rae_core.llm.providers import (
    BridgeLLMProvider,
    LightweightOpenAIProvider,
    LightweightAnthropicProvider,
    LightweightGoogleProvider
)
from rae_core.llm.small_local import SmallLocalLLMProvider

logger = structlog.get_logger(__name__)

async def resolve_llm_runtime(
    requirements: Optional[dict] = None,
    target_agent: Optional[str] = None
) -> ILLMProvider:
    mode = os.getenv("RAE_LLM_MODE", "auto").lower()
    api_url = os.getenv("RAE_API_URL", "http://rae-api-dev:8000")
    
    if mode == "remote":
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{api_url}/health")
                resp.raise_for_status()
            logger.info("llm_runtime_resolved", mode=mode, api_url=api_url)
            return BridgeLLMProvider(api_url, target_agent=target_agent or "rae-local-reasoner")
        except Exception as e:
            raise RuntimeError(f"RAE Bridge is unavailable under RAE_LLM_MODE=remote: {e}")
            
    elif mode == "local":
        logger.info("llm_runtime_resolved", mode=mode)
        return _build_local_runtime(requirements)
        
    else: # auto
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{api_url}/health")
                if resp.status_code == 200:
                    logger.info("llm_runtime_resolved", mode="remote_fallback", api_url=api_url)
                    return BridgeLLMProvider(api_url, target_agent=target_agent or "rae-local-reasoner")
        except Exception:
            pass
            
        logger.info("llm_runtime_resolved", mode="local_fallback")
        return _build_local_runtime(requirements)

def _build_local_runtime(requirements: Optional[dict] = None) -> ILLMProvider:
    req = requirements or {}
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    qwen_key = os.getenv("QWEN_API_KEY")
    
    if req.get("requires_reasoning"):
        if deepseek_key:
            return LightweightOpenAIProvider(api_key=deepseek_key, base_url="https://api.deepseek.com/v1", model="deepseek-reasoner")
        if openai_key:
            return LightweightOpenAIProvider(api_key=openai_key, model="o3-mini")
            
    if req.get("requires_json_schema"):
        if openai_key:
            return LightweightOpenAIProvider(api_key=openai_key, model="gpt-4o-mini")
        if gemini_key:
            return LightweightGoogleProvider(api_key=gemini_key, model="gemini-2.5-flash")
            
    if openai_key:
        return LightweightOpenAIProvider(api_key=openai_key)
    if anthropic_key:
        return LightweightAnthropicProvider(api_key=anthropic_key)
    if gemini_key:
        return LightweightGoogleProvider(api_key=gemini_key)
    if deepseek_key:
        return LightweightOpenAIProvider(api_key=deepseek_key, base_url="https://api.deepseek.com/v1", model="deepseek-chat")
    if qwen_key:
        return LightweightOpenAIProvider(api_key=qwen_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-turbo")

    return SmallLocalLLMProvider()
