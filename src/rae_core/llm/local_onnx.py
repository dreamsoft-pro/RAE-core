"""
RAE Local ONNX LLM Provider using onnxruntime-genai.
Direct local execution without external APIs or Ollama.
"""

import os
import structlog
from typing import Any, List, Dict, Optional
from rae_core.interfaces.llm import ILLMProvider

logger = structlog.get_logger(__name__)

class LocalOnnxLLMProvider(ILLMProvider):
    """
    Local LLM provider using onnxruntime-genai.
    Expects model files in 'models/llm'.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def _initialize(self):
        if self._initialized:
            return
            
        try:
            import onnxruntime_genai as og
            if not os.path.exists(self.model_path):
                logger.warning("onnx_llm_model_not_found", path=self.model_path)
                return
                
            logger.info("loading_local_onnx_llm", path=self.model_path)
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
            self._initialized = True
            logger.info("local_onnx_llm_ready")
        except ImportError:
            logger.warning("onnxruntime_genai_not_installed")
        except Exception as e:
            logger.error("local_onnx_llm_init_failed", error=str(e))

    async def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        self._initialize()
        if not self._initialized:
            return "" # Fallback to other providers will happen in SmallLocalLLMProvider
            
        import onnxruntime_genai as og
        
        try:
            full_prompt = f"<|system|>
{system_prompt}<|end|>
<|user|>
{prompt}<|end|>
<|assistant|>"
            
            tokens = self.tokenizer.encode(full_prompt)
            
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=kwargs.get("max_tokens", 1024), 
                                    temperature=kwargs.get("temperature", 0.7))
            params.input_ids = tokens
            
            generator = og.Generator(self.model, params)
            
            output_tokens = []
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                output_tokens.append(new_token)
                
            return self.tokenizer.decode(output_tokens).strip()
        except Exception as e:
            logger.error("local_onnx_llm_generation_failed", error=str(e))
            return ""

    async def generate_with_context(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Construct prompt from messages
        prompt = ""
        system = ""
        for msg in messages:
            if msg['role'] == 'system': system = msg['content']
            else: prompt += f"{msg['role']}: {msg['content']}
"
        return await self.generate(prompt, system_prompt=system, **kwargs)

    async def count_tokens(self, text: str) -> int:
        return len(text) // 4

    def supports_function_calling(self) -> bool:
        return False

    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        return []

    async def summarize(self, text: str, max_length: int = 200) -> str:
        return text[:max_length] + "..."
