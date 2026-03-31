"""
RAE Content Signature Detector (System 42.0).
Stage 2: Calculating Information Density and Signal-to-Noise Ratio.
"""

import math
import re
from collections import Counter
from typing import Any, Dict, List
from .interfaces import ISignatureDetector, ContentSignature, IngestAudit

class ContentSignatureDetector(ISignatureDetector):
    """
    Calculates structural, distributional and adaptive signatures.
    Derived parameters replace all hardcoded constants.
    """
    
    def detect(self, text: str) -> tuple[ContentSignature, IngestAudit]:
        struct = self._analyze_structure(text)
        dist = self._analyze_distribution(text)
        
        # SYSTEM 42.0: Information Density (ID)
        # Ratio of unique semantic information to total length
        vocab_size = dist.get("vocab_size", 1)
        entropy = dist.get("token_entropy", 1.0)
        text_len = len(text) or 1
        
        # Density: How much "surprise" (entropy) per character
        info_density = (entropy * vocab_size) / text_len
        dist["info_density"] = round(info_density, 5)
        
        # Adaptive Stability
        stab = {
            "conflict": dist["repeatability_score"] > 0.5 and entropy > 7.0,
            "order_sensitive": struct["mode"] in ["LINEAR_LOG_LIKE", "MACHINE_TELEMETRY_LIKE"],
            "signal_resonance_potential": 1.0 / (1.0 + info_density)
        }
        
        signature = ContentSignature(struct=struct, dist=dist, stab=stab)
        
        audit = IngestAudit(
            stage="detect",
            action="autonomous_signature_generation",
            trace={
                "mode": struct["mode"],
                "info_density": dist["info_density"],
                "entropy": entropy
            }
        )
        
        return signature, audit

    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        line_count = len(lines)
        if line_count == 0: return {"mode": "EMPTY"}
        
        time_markers = len(re.findall(r'\d{2,4}[-:/]\d{2}[-:/]\d{2,4}', text))
        machine_markers = sum(1 for l in lines if re.search(r'\b(Speed|Size|Job\s*ID|Ink|Print\s*Area)\b', l, re.I))
        list_markers = sum(1 for l in lines if re.match(r'^(?:Krok|Step|Kolejno|\d+[\.)]|[-*•])', l, re.I))
        
        # SYSTEM 92.1: Operational Fallback Detection (Anti-Echo)
        is_fallback = "STABILITY MODE ACTIVE" in text or "Math Fallback" in text
        
        # Adaptive Mode Selection (No fixed weights, based on marker dominance)
        modes = {
            "MACHINE_TELEMETRY_LIKE": machine_markers / line_count if line_count > 0 else 0,
            "LINEAR_LOG_LIKE": time_markers / line_count if line_count > 0 else 0,
            "LIST_PROCEDURE_LIKE": list_markers / line_count if line_count > 0 else 0
        }
        
        best_mode = max(modes, key=modes.get)
        if is_fallback:
            best_mode = "OPERATIONAL_FALLBACK"
        elif modes[best_mode] < 0.2: # Threshold for structural signal
            best_mode = "PROSE_PARAGRAPH_LIKE"
            
        return {
            "mode": best_mode,
            "line_count": line_count,
            "confidence": 1.0 if is_fallback else round(modes.get(best_mode, 0), 2)
        }

    def _analyze_distribution(self, text: str) -> Dict[str, Any]:
        words = text.split()
        if not words: return {"token_entropy": 0, "vocab_size": 0, "repeatability_score": 0}
        
        counts = Counter(words)
        total = len(words)
        entropy = -sum((count/total) * math.log2(count/total) for count in counts.values())
        
        lines = [l.strip()[:15] for l in text.split('\n') if len(l.strip()) > 10]
        repeatability = 0.0
        if lines:
            line_prefixes = Counter(lines)
            repeatability = sum(count for count in line_prefixes.values() if count > 1) / len(lines)
            
        return {
            "token_entropy": round(entropy, 3),
            "repeatability_score": round(repeatability, 3),
            "vocab_size": len(counts)
        }
