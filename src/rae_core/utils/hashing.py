"""Deterministic hashing utilities for System 87.0.

Provides stable, seed-based hashing to ensure identical memory layout
and index behavior across restarts and environments, independent of
PYTHONHASHSEED.
"""

import hashlib
from typing import Any

def stable_hash(key: str | bytes) -> int:
    """Compute a stable, deterministic hash for a key.
    
    Uses SHA-256 truncated to 64-bit integer to guarantee
    collisions are mathematically improbable and identical across platforms.
    
    Args:
        key: String or bytes to hash.
        
    Returns:
        Signed 64-bit integer.
    """
    if isinstance(key, str):
        data = key.encode("utf-8")
    else:
        data = key
        
    # SHA-256 is standard and stable
    sha = hashlib.sha256(data).digest()
    
    # Take first 8 bytes for 64-bit int
    # Big-endian for consistency
    val = int.from_bytes(sha[:8], byteorder="big", signed=True)
    
    return val

def bloom_filter_fingerprint(tags: list[str]) -> int:
    """Generate a simple bitmask fingerprint for tags.
    
    This is a simplified "Bloom Filter" (more like a bitmask signature)
    to allow O(1) rejection of non-matching candidates.
    
    Maps each tag to a bit position 0-63.
    """
    mask = 0
    for tag in tags:
        h = stable_hash(tag)
        # Map to 0-63
        bit = h % 64
        mask |= (1 << bit)
    return mask

def compute_provenance_hash(data: dict[str, Any], source_hashes: list[str]) -> str:
    """Compute the Merkle-DAG hash for a node.
    
    Hash(S_n) = SHA256( Data(S_n) || Hash(E_1) || ... || Hash(E_k) )
    
    Args:
        data: Dictionary of node data (must be serializable/hashable).
              Typically includes content, timestamp, type.
        source_hashes: List of SHA-256 hex strings of source nodes.
        
    Returns:
        Hex string of the SHA-256 hash.
    """
    # Deterministic serialization of data
    # Sort keys to ensure stability
    import json
    
    # Filter out volatile fields if any (like access_count)
    # For strict provenance, we hash content + immutable metadata
    clean_data = {k: v for k, v in data.items() if k not in ("access_count", "usage_count", "last_accessed_at")}
    
    data_bytes = json.dumps(clean_data, sort_keys=True, default=str).encode("utf-8")
    
    # Sort source hashes to ensure order independence (unless order matters for DAG?)
    # Usually DAG edges have order if they represent sequence. If just set of sources, sort.
    # Plan says "Hash(E_1) || ...", usually implying order. But for "set of sources", sorting is safer.
    sorted_sources = sorted(source_hashes)
    source_bytes = "".join(sorted_sources).encode("utf-8")
    
    combined = data_bytes + source_bytes
    
    return hashlib.sha256(combined).hexdigest()
