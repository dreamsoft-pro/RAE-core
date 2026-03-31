"""
RAE Ingest Normalizer (System 42.1).
Stage 1 of the Universal Ingest Pipeline.
Implements Robust Decoding for multi-encoding support (UTF-8, Windows-1250).
"""

from typing import Any, Dict, Optional, Union
import structlog
from .interfaces import INormalizer, IngestAudit

logger = structlog.get_logger(__name__)

class IngestNormalizer(INormalizer):
    """
    Standardizes input text for analysis.
    Performs encoding recovery and basic cleaning.
    """
    
    def normalize(self, data: Union[str, bytes], metadata: Optional[Dict[str, Any]] = None) -> tuple[str, IngestAudit]:
        original_len = len(data)
        detected_encoding = "string_input"
        
        # 1. Robust Decoding (if input is bytes)
        if isinstance(data, bytes):
            text, detected_encoding = self._decode_robustly(data)
        else:
            text = data

        # 2. Standard Cleaning
        # Remove null bytes and standardize line endings
        normalized = text.replace('\x00', '')
        normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
        
        # 3. Trace
        audit = IngestAudit(
            stage="normalize",
            action="robust_decoding",
            trace={
                "original_length": original_len,
                "normalized_length": len(normalized),
                "encoding": detected_encoding,
                "source_metadata": metadata or {}
            }
        )
        
        return normalized, audit

    def _decode_robustly(self, data: bytes) -> tuple[str, str]:
        """
        Attempts to decode bytes using a priority list of encodings.
        Priority: utf-8-sig (BOM), windows-1250 (Polish), utf-8 (strict), utf-8 (ignore).
        """
        encodings = [
            ("utf-8-sig", "strict"),
            ("windows-1250", "strict"),
            ("utf-8", "strict"),
            ("utf-8", "ignore")
        ]
        
        for enc, mode in encodings:
            try:
                text = data.decode(enc, errors=mode)
                return text, enc
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Ultimate fallback
        return data.decode('ascii', errors='ignore'), "fallback_ascii"
