import re
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MetadataInjector:
    """
    Standard Metadata Injector - Pure Mathematics of Language.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.synonyms = {
            "slo": ["sla", "reliability", "metrics"],
            "error": ["bug", "failure", "fault"],
            "critical": ["p0", "blocker", "urgent"],
        }

    def enrich_text(self, text: str) -> str:
        if not text: return text
        text_lower = text.lower()
        found = []
        for k, v in self.synonyms.items():
            if k in text_lower and re.search(rf"\b{re.escape(k)}\b", text_lower):
                found.extend(v)
        if not found: return text
        return f"{text} {' '.join(set(found))}"

    def process_query(self, query: str) -> str: return self.enrich_text(query)
    def process_document(self, text: str) -> str: return self.enrich_text(text)
