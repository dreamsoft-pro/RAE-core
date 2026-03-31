# rae_core/utils/memory_bridge.py
import os
import requests
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from rae_core.utils.context import RAEContextLocator

class RAEMemoryBridge:
    """Enterprise bridge with automatic context detection and ASYNC background logging."""
    
    # Global executor for all bridge instances to keep threads under control
    _executor = ThreadPoolExecutor(max_workers=4)

    def __init__(self, project_name: str = None):
        self.api_url = os.getenv("RAE_API_URL", "http://rae-api-dev:8000")
        self.tenant_id = RAEContextLocator.get_current_tenant_id()
        self.project = project_name or RAEContextLocator.get_project_name()
        self.logger = logging.getLogger(f"RAE.Bridge.{self.project}")

    def save_event(self, content: str, human_label: str = None, metadata: dict = None, layer: str = "episodic"):
        """Asynchronously saves an event to RAE Memory."""
        if self.tenant_id == "00000000-0000-0000-0000-000000000000" and self.project == "unnamed_production_module":
            return False

        payload = {
            "content": content,
            "project": self.project,
            "human_label": human_label or f"[{self.project.upper()}] Operation Log",
            "layer": layer,
            "metadata": metadata or {},
            "importance": 0.5
        }
        headers = {"X-Tenant-Id": self.tenant_id}
        self._executor.submit(self._send_request, payload, headers)
        return True

    def log_decision(self, action: str, reasoning: str, payload: dict, layer: str = "reflective"):
        """Synchronous, high-priority decision audit for ISO compliance."""
        human_label = f"[{self.project.upper()}] Decision: {action.replace('_', ' ').title()}"
        data = {
            "content": f"{human_label}. Reasoning: {reasoning}",
            "human_label": human_label,
            "project": self.project,
            "layer": layer,
            "metadata": {
                "audit_type": "iso_27001_compliance",
                "action": action,
                "context": payload,
                "agent": self.project
            }
        }
        headers = {"X-Tenant-Id": self.tenant_id}
        return self._send_request(data, headers)

    def _send_request(self, payload: dict, headers: dict):
        """Internal synchronous sender for background execution."""
        url = f"{self.api_url}/v2/memories"
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            if r.status_code not in [200, 201]:
                self.logger.warning(f"⚠️ Memory API returned {r.status_code}: {r.text[:100]}")
        except Exception as e:
            # Silence background errors to prevent main app from crashing
            pass
