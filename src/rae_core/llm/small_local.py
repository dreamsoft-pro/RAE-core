"""
RAE Small Local LLM Provider.
Optimized for Windows Standalone usage.
"""

import re
import httpx
import structlog
from typing import Any, List, Dict, Optional
from rae_core.interfaces.llm import ILLMProvider

from rae_core.llm.local_onnx import LocalOnnxLLMProvider

logger = structlog.get_logger(__name__)

class SmallLocalLLMProvider(ILLMProvider):
    """
    Local LLM provider that prioritizes:
    1. Local ONNX model (bundled with EXE)
    2. Ollama (running on Windows)
    3. Procedural rule-based fallback
    """
    
    def __init__(self, 
                 onnx_model_path: Optional[str] = None,
                 ollama_url: str = "http://localhost:11434", 
                 model: str = "llama3:8b"):
        self.ollama_url = ollama_url
        self.model = model
        self.client = httpx.AsyncClient(base_url=ollama_url, timeout=60.0)
        
        # Local ONNX Provider
        self.onnx_provider = None
        if onnx_model_path:
            self.onnx_provider = LocalOnnxLLMProvider(onnx_model_path)
        
    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        # 1. Try Local ONNX (Primary for EXE)
        if self.onnx_provider:
            res = await self.onnx_provider.generate(prompt, system_prompt, **kwargs)
            if res:
                return res
                
        # 2. Try Ollama
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": kwargs.get("temperature", 0.7)}
            }
            if system_prompt:
                payload["system"] = system_prompt
                
            response = await self.client.post("/api/generate", json=payload)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            logger.debug("ollama_unavailable", error=str(e))
            
        # 3. Procedural Fallback (Designed for Order Entry)
        return self._generate_procedural_fallback(prompt, system_prompt or "")

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Simplistic conversion for Lite mode
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, **kwargs)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return False

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length] + "..."

    def _generate_procedural_fallback(self, prompt: str, context: str) -> str:
        """
        Rule-based procedural synthesis when LLM is offline.
        Extracts lines from search results that look like steps.
        """
        # Combine everything we know
        source_material = context + "\n" + prompt
        lines = source_material.split("\n")
        steps = []
        
        seen = set()
        for line in lines:
            line = line.strip()
            if len(line) < 10 or line in seen: continue
            
            # Match Step 1, Krok 1, 1., Kolejno:
            if re.match(r'^(?:Krok|Step|Kolejno|\d+[\.)])', line, re.I):
                steps.append(line)
                seen.add(line)
            elif any(kw in line.lower() for kw in ["należy", "powinieneś", "musisz", "wymagane", "instrukcja"]):
                if not line.startswith("#"): # Skip headers
                    steps.append(f"- {line}")
                    seen.add(line)
                
        if not steps:
            return "STABILITY MODE (LLM Offline). No clear procedural steps found. Review raw results below."
            
        header = "ORDER ENTRY ASSISTANT (Math-Synthesized Steps):\n\n"
        return header + "\n".join(steps[:12])
