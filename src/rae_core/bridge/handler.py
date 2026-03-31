from fastapi import APIRouter, Request
from typing import Any, Dict, Optional
from uuid import uuid4, UUID
import logging
import httpx
import os

# Standard logging instead of structlog for maximum compatibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rae-bridge")

class BridgeA2AHandler:
    """
    Standardized A2A Handler for RAE Modules.
    Ensures all interactions follow the Silicon Oracle schema.
    """
    def __init__(self, module_name: str, rae_api_url: Optional[str] = None):
        self.module_name = module_name
        self.rae_api_url = rae_api_url or os.getenv("RAE_API_URL", "http://rae-api-dev:8000")
        self.router = APIRouter(prefix="/v2/bridge", tags=["Bridge"])
        self._setup_routes()

    def _setup_routes(self):
        @self.router.post("/interact")
        async def interact(
            payload: Dict[str, Any],
            request: Request,
            source_agent: str = "unknown",
            target_agent: Optional[str] = None,
            session_id: Optional[str] = None,
            correlation_id: Optional[UUID] = None
        ):
            target = target_agent or self.module_name
            event_id = uuid4()
            corr_id = correlation_id or event_id
            
            logger.info(f"bridge_interaction_received: source={source_agent}, target={target}, event_id={str(event_id)}")

            # 1. Forward to RAE Memory (Implicit Capture)
            await self._capture_interaction(
                source_agent, target, payload, session_id, corr_id, event_id
            )

            # 2. Return success
            return {
                "event_id": str(event_id),
                "correlation_id": str(corr_id),
                "status": "received",
                "module": self.module_name
            }

    async def _capture_interaction(self, source, target, payload, session_id, corr_id, event_id):
        try:
            async with httpx.AsyncClient() as client:
                memory_payload = {
                    "content": f"A2A INTERACTION: {source} -> {target}: {str(payload)[:500]}",
                    "layer": "episodic",
                    "agent_id": source,
                    "metadata": {
                        "event_id": str(event_id),
                        "correlation_id": str(corr_id),
                        "target_agent": target,
                        "interaction_data": payload
                    }
                }
                if session_id:
                    memory_payload["session_id"] = session_id
                
                # We use the internal network URL
                await client.post(
                    f"{self.rae_api_url}/v2/memories",
                    json=memory_payload,
                    headers={
                        "X-Tenant-Id": "00000000-0000-0000-0000-000000000000",
                        "X-Project-Id": "rae-suite"
                    },
                    timeout=5.0
                )
        except Exception as e:
            logger.error(f"bridge_capture_failed: {e}")

def register_bridge(app, module_name: str):
    handler = BridgeA2AHandler(module_name)
    app.include_router(handler.router)
    return handler
