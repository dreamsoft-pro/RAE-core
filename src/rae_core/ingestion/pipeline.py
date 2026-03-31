"""
RAE Universal Ingest Pipeline (UICTC).
Orchestrates the 5-stage deterministic ingestion process.
Propagates global math configuration to all stages.
"""

from typing import Any, Dict, List, Optional
import structlog
import yaml
import os

from .interfaces import IngestChunk, ContentSignature, IngestAudit
from .normalizer import IngestNormalizer
from .detector import ContentSignatureDetector
from .policy import IngestPolicySelector
from .segmenter import IngestSegmenter
from .compressor import IngestCompressor

logger = structlog.get_logger(__name__)

class UniversalIngestPipeline:
    """
    Orchestrates ingestion from raw text to structured, multi-vector memories.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize stages with injected configuration
        self.normalizer = IngestNormalizer()
        self.detector = ContentSignatureDetector()
        self.policy_selector = IngestPolicySelector()
        self.segmenter = IngestSegmenter(config=self.config) # Ingesting rules
        self.compressor = IngestCompressor()

    def _load_config(self, path: Optional[str]) -> Dict[str, Any]:
        if not path:
            # Try default paths
            paths = [
                "config/math_controller.yaml",
                "/app/math_config/math_controller.yaml",
                "../config/math_controller.yaml"
            ]
            for p in paths:
                if os.path.exists(p):
                    path = p
                    break
        
        if path and os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.error("config_load_failed", path=path, error=str(e))
        return {}

    async def process(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> tuple[List[IngestChunk], ContentSignature, List[IngestAudit], str]:
        audit_trail = []
        
        # SYSTEM 92.2: Dedup Hash (Stop the spiral)
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Stage 1: Normalize
        normalized_text, audit_1 = self.normalizer.normalize(text, metadata)
        audit_trail.append(audit_1)
        
        # Stage 2: Detect Signature (S-D-O Layer)
        signature, audit_2 = self.detector.detect(normalized_text)
        audit_trail.append(audit_2)
        
        # Stage 3: Policy Selection
        policy, audit_3 = self.policy_selector.select_policy(signature)
        audit_trail.append(audit_3)
        
        # Stage 4: Content-Aware Segmentation (AFE happens here)
        chunks, audit_4 = await self.segmenter.segment(normalized_text, policy, signature)
        audit_trail.append(audit_4)
        
        # Attach hash to metadata for downstream dedup if needed
        for chunk in chunks:
            if not chunk.metadata: chunk.metadata = {}
            chunk.metadata["content_hash"] = text_hash

        # Stage 5: Compression / Semantic Folding
        compressed_chunks, prov_map, audit_5 = self.compressor.compress(chunks, policy)
        audit_trail.append(audit_5)
        
        # Attach provenance and policy to chunks
        for i, chunk in enumerate(compressed_chunks):
            if not chunk.metadata: chunk.metadata = {}
            chunk.metadata.update({
                "ingest_signature": signature.to_dict(),
                "ingest_policy": policy,
                "ingest_version": "1.0.0",
                "compression_provenance": prov_map.get(i, [i])
            })
            
        logger.info("universal_ingest_complete", 
                    chunk_count=len(compressed_chunks),
                    mode=signature.struct.get("mode"),
                    policy=policy)
        
        return compressed_chunks, signature, audit_trail, policy
