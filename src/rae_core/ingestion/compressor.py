"""
RAE Ingest Compressor (Stage 5).
Reduces redundancy in repetitive data (like logs).
"""

from typing import Any, Dict, List, Tuple
from .interfaces import ICompressor, IngestChunk, IngestAudit

class IngestCompressor(ICompressor):
    """
    Compresses repetitive chunks to save space and reduce noise.
    Primarily used for log streams.
    """
    
    def compress(self, chunks: List[IngestChunk], policy: str) -> tuple[List[IngestChunk], Dict[str, Any], IngestAudit]:
        if policy != "POLICY_LOG_STREAM":
            # No compression for non-log data by default
            return chunks, {}, IngestAudit(stage="compress", action="skip", trace={"reason": "policy_not_eligible"})
            
        original_count = len(chunks)
        compressed_chunks = []
        provenance_map = {}
        
        # Simple deduplication based on content hash/prefix
        seen_patterns = {}
        
        for i, chunk in enumerate(chunks):
            # Extract pattern (e.g. first 40 chars of log line)
            lines = chunk.content.split('\n')
            if not lines: continue
            
            # Use the first line as a pattern representative
            pattern = lines[0][:40] 
            
            if pattern not in seen_patterns:
                compressed_chunks.append(chunk)
                seen_patterns[pattern] = i
                provenance_map[i] = [i]
            else:
                # Duplicate pattern found, record provenance but don't add chunk
                original_index = seen_patterns[pattern]
                # Ensure the original index is in the map
                if original_index not in provenance_map:
                    provenance_map[original_index] = [original_index]
                provenance_map[original_index].append(i)
                
        audit = IngestAudit(
            stage="compress",
            action="deduplication",
            trace={
                "original_chunks": original_count,
                "compressed_chunks": len(compressed_chunks),
                "ratio": round(len(compressed_chunks) / original_count, 2) if original_count > 0 else 1.0
            }
        )
        
        return compressed_chunks, provenance_map, audit
