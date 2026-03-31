from typing import Any, Dict, List, Optional
import structlog

logger = structlog.get_logger(__name__)

class L4CognitiveReflection:
    """
    L4 Cognitive Layer (The Sage).
    Non-deterministic layer using LLM (Qwen 3.5) to synthesize high-level patterns 
    and "Lessons Learned" from agentic activity.
    """
    def __init__(self, llm_provider: Any = None, model_name: str = "ollama/qwen3.5:9b"):
        self.llm_provider = llm_provider
        self.model_name = model_name

    async def reflect(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyzes the context and decision to extract semantic insights.
        """
        print(f"DEBUG L4: Starting reflection for model {self.model_name}")
        if not self.llm_provider:
            print("DEBUG L4: No LLM provider")
            return {"status": "skipped", "reason": "No LLM provider configured for L4"}

        content = payload.get("analysis", "")
        sources = payload.get("retrieved_sources_content", [])
        
        print(f"DEBUG L4: Input content length: {len(content)}")
        
        prompt = f"""You are the RAE L4 Cognitive Reflection engine. 
Analyze the following agent action and its grounding sources to extract a 'Lesson Learned'.
A 'Lesson Learned' should be a concise, reusable piece of knowledge for future sessions.

AGENT ACTION/ANALYSIS:
{content}

GROUNDING SOURCES:
{chr(10).join(sources[:3])}

Format your response as a JSON object:
{{
  "lesson": "The core insight extracted",
  "confidence": 0.0-1.0,
  "tags": ["tag1", "tag2"]
}}
"""
        try:
            print(f"DEBUG L4: Sending request to Ollama...")
            # We use the configured LLM (Qwen 3.5) for synthesis
            response = await self.llm_provider.generate_text(
                prompt=prompt,
                model=self.model_name
            )
            
            if not response:
                print("DEBUG L4: Ollama returned EMPTY response")
                return {"status": "error", "reason": "Empty response from Ollama"}

            print(f"DEBUG L4: Received response from Ollama (Length: {len(response)})")
            print(f"DEBUG L4: Raw response start: {response[:200]}")
            
            import json
            import re
            
            # Robust JSON extraction from LLM output
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    insight = json.loads(match.group())
                    print(f"DEBUG L4: Successfully parsed insight: {insight.get('lesson')}")
                    return {
                        "status": "success",
                        "insight": insight,
                        "raw_response": response,
                        "model": self.model_name
                    }
                except Exception as parse_err:
                    print(f"DEBUG L4: JSON parse error: {str(parse_err)}")
            
            print(f"DEBUG L4: Failed to find valid JSON in response. Content: {response[:500]}")
            # Even if not JSON, return the raw text as an insight attempt
            return {
                "status": "partial", 
                "reason": "Failed to parse JSON, but got text",
                "raw_response": response,
                "insight": {"lesson": response[:500], "confidence": 0.1, "tags": ["unformatted"]}
            }
            
        except Exception as e:
            print(f"DEBUG L4: Exception during reflection: {str(e)}")
            logger.warning("l4_reflection_failed", error=str(e))
            return {"status": "error", "reason": str(e)}
