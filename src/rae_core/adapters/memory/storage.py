"""In-Memory storage adapter for RAE-core (Deterministyczny, High-Performance).

Implements both IMemoryStorage and IVectorStore using contiguous memory arenas
and fixed-point arithmetic for absolute determinism (System 87.0).
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, cast
from uuid import UUID, uuid4

from rae_core.interfaces.storage import IMemoryStorage
from rae_core.interfaces.vector import IVectorStore
from rae_core.utils.clock import IClock, SystemClock
from rae_core.math.quantization_bytes import (
    quantize_vector_bytes,
    cosine_similarity_bytes,
    dequantize_vector_bytes
)
from rae_core.utils.hashing import bloom_filter_fingerprint, stable_hash


class InMemoryStorage(IMemoryStorage, IVectorStore):
    """In-memory implementation of IMemoryStorage and IVectorStore.

    Architecture (System 87.0):
    - Contiguous Memory Arenas (bytearray) for vectors to ensure L1/L2 cache locality.
    - Fixed-Point Quantization (int32) for deterministic arithmetic.
    - Offset-based indexing instead of object references.
    """

    def __init__(self, clock: IClock | None = None) -> None:
        """Initialize in-memory storage."""
        self._clock = clock or SystemClock()
        
        # Main storage: {memory_id: memory_dict}
        self._memories: dict[UUID, dict[str, Any]] = {}

        # Reflection Audits storage: {audit_id: audit_dict}
        self._reflection_audits: dict[UUID, dict[str, Any]] = {}

        # Bloom Filters: {memory_id: 64-bit mask} (The "Scalpel" Phase 2)
        self._bloom_filters: dict[UUID, int] = {}

        # Indexes for fast lookups
        self._by_tenant: dict[str, set[UUID]] = defaultdict(set)
        self._by_agent: dict[tuple[str, str], set[UUID]] = defaultdict(set)
        self._by_layer: dict[tuple[str, str], set[UUID]] = defaultdict(set)
        self._by_tags: dict[tuple[str, str], set[UUID]] = defaultdict(set)

        # Vector Arenas: {model_name: bytearray}
        # Stores packed int32 vectors contiguously.
        self._vector_arenas: dict[str, bytearray] = defaultdict(bytearray)
        
        # Vector Index: {model_name: {memory_id: offset}}
        # Maps memory ID to byte offset in the arena.
        self._vector_indices: dict[str, dict[UUID, int]] = defaultdict(dict)
        
        # Vector Metadata: {model_name: {memory_id: metadata}}
        # Stores metadata alongside vectors for filtering.
        self._vector_metadata: dict[str, dict[UUID, dict[str, Any]]] = defaultdict(dict)
        
        # Vector Dimensions: {model_name: dimension_size}
        # Used to validate vector sizes and calculate stride.
        self._vector_dims: dict[str, int] = {}

        # Thread safety
        self._lock = asyncio.Lock()

    # =========================================================================
    # IVectorStore Implementation (The "Arena" & "Scalpel")
    # =========================================================================

    async def store_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Store a vector embedding in contiguous arena."""
        async with self._lock:
            # Check if memory exists before storing vector
            if memory_id not in self._memories:
                return False

            # Normalize input to dict of vectors
            vectors: dict[str, list[float]] = {}
            if isinstance(embedding, list):
                vectors["default"] = embedding
            elif isinstance(embedding, dict):
                vectors = embedding
            else:
                # Invalid embedding type (e.g. string from test)
                return False
    
            for model_name, vector in vectors.items():
                if not isinstance(vector, (list, tuple)):
                    return False
                dim = len(vector)
                
                # Validate dimension consistency
                if model_name not in self._vector_dims:
                    self._vector_dims[model_name] = dim
                elif self._vector_dims[model_name] != dim:
                    raise ValueError(
                        f"Dimension mismatch for model {model_name}: "
                        f"expected {self._vector_dims[model_name]}, got {dim}"
                    )

                # Quantize to bytes (Fixed-Point)
                vector_bytes = quantize_vector_bytes(vector)
                vector_len = len(vector_bytes)

                # Check if update or insert
                if memory_id in self._vector_indices[model_name]:
                    # Update in place (if same size) - simplified: append new, orphan old space
                    # Real arena allocator would manage free list. Here we append-only for simplicity/speed in Python.
                    # Or we overwrite if length matches (which it should for fixed dim).
                    offset = self._vector_indices[model_name][memory_id]
                    self._vector_arenas[model_name][offset : offset + vector_len] = vector_bytes
                else:
                    # Append new
                    offset = len(self._vector_arenas[model_name])
                    self._vector_arenas[model_name].extend(vector_bytes)
                    self._vector_indices[model_name][memory_id] = offset

                # Store metadata
                meta = metadata or {}
                # Ensure tenant_id is in metadata for security filtering
                meta["tenant_id"] = tenant_id
                self._vector_metadata[model_name][memory_id] = meta

            return True

    async def search_similar(
        self,
        query_embedding: list[float],
        tenant_id: str,
        layer: str | None = None,
        limit: int = 10,
        score_threshold: float | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        filters: dict[str, Any] | None = None,
        project: str | None = None,
        **kwargs: Any,
    ) -> list[tuple[UUID, float]]:
        """Search for similar vectors using deterministic fixed-point arithmetic."""
        async with self._lock:
            model_name = kwargs.get("model_name", "default")
            
            if model_name not in self._vector_arenas:
                return []

            # Prepare query
            query_bytes = quantize_vector_bytes(query_embedding)
            dim_bytes = len(query_bytes)
            
            if model_name in self._vector_dims:
                 expected_dim = self._vector_dims[model_name] * 4
                 if dim_bytes != expected_dim:
                     # Fail silently or raise? Standard is usually empty result on mismatch
                     return []

            arena = self._vector_arenas[model_name]
            indices = self._vector_indices[model_name]
            metadatas = self._vector_metadata[model_name]

            results: list[tuple[UUID, float]] = []

            # Bloom Filter Setup (The Scalpel)
            query_mask = 0
            if filters and "tags" in filters:
                tags_list = filters["tags"]
                if isinstance(tags_list, list):
                    query_mask = bloom_filter_fingerprint(tags_list)

            # Linear Scan (Simulating low-level scan)
            # In C++ this would be a SIMD loop. In Python, we iterate keys to look up offsets.
            # Iterating keys is faster than slicing bytearray repeatedly in Python.
            for mem_id, offset in indices.items():
                # 0. Bloom Filter Check (O(1) Bitwise Rejection) - Phase 2
                if query_mask:
                     mem_mask = self._bloom_filters.get(mem_id, 0)
                     # Check if memory has ALL required tag bits
                     if (mem_mask & query_mask) != query_mask:
                         continue

                # 1. Security & Metadata Filtering (The "Scalpel")
                meta = metadatas.get(mem_id, {})
                if meta.get("tenant_id") != tenant_id:
                    continue
                
                if layer and meta.get("layer") != layer:
                    continue
                if agent_id and meta.get("agent_id") != agent_id:
                    continue
                if session_id and meta.get("session_id") != session_id:
                    continue
                if project and meta.get("project") != project:
                    continue
                    
                if filters:
                    match = True
                    for k, v in filters.items():
                        # Special handling for tags (subset check)
                        if k == "tags":
                            # Bloom filter already did probabilistic check. Now verify strictly.
                            # Query tags (v) must be subset of memory tags
                            # Handle different types (list vs set)
                            query_tags = set(v) if isinstance(v, (list, tuple)) else {v}
                            mem_tags = set(meta.get("tags", []))
                            if not query_tags.issubset(mem_tags):
                                match = False
                                break
                            continue

                        if meta.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                # 2. Extract Vector from Arena
                # Slicing creates a copy, but it's needed for the math function
                # (unless we rewrite math to take buffer + offset, which is better but strictly Python bytes are immutable/copy-heavy)
                vec_bytes = arena[offset : offset + dim_bytes]

                # 3. Compute Similarity (Deterministic)
                score = cosine_similarity_bytes(query_bytes, vec_bytes)

                if score <= 0.0:
                    continue

                if score_threshold is not None and score < score_threshold:
                    continue

                results.append((mem_id, score))

            # Sort by score descending (Tie-Breaking by ID for determinism)
            # Python's sort is stable.
            results.sort(key=lambda x: (x[1], x[0].hex), reverse=True)

            return results[:limit]

    async def search_similar_batch(
        self,
        query_embeddings: list[list[float]],
        tenant_id: str,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[list[tuple[UUID, float]]]:
        """Search for multiple embeddings in batch."""
        results = []
        for emb in query_embeddings:
            res = await self.search_similar(emb, tenant_id, limit=limit, **kwargs)
            results.append(res)
        return results

    async def get_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> list[float] | None:
        """Retrieve a vector embedding."""
        async with self._lock:
            # Default model strategy: Try "default", then fallback to any available
            model_name = "default"
            
            # Check if default exists for this ID
            if not (model_name in self._vector_indices and memory_id in self._vector_indices[model_name]):
                # Fallback: Find first model containing this ID
                found_model = None
                for m_name, index in self._vector_indices.items():
                    if memory_id in index:
                        found_model = m_name
                        break
                
                if not found_model:
                    return None
                model_name = found_model
            
            offset = self._vector_indices[model_name][memory_id]
            dim = self._vector_dims[model_name]
            byte_len = dim * 4
            
            vec_bytes = self._vector_arenas[model_name][offset : offset + byte_len]
            
            # Verify tenant
            meta = self._vector_metadata[model_name].get(memory_id, {})
            if meta.get("tenant_id") != tenant_id:
                return None
                
            return dequantize_vector_bytes(bytes(vec_bytes))

    async def delete_vector(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a vector."""
        async with self._lock:
            deleted = False
            for model_name in list(self._vector_indices.keys()):
                if memory_id in self._vector_indices[model_name]:
                    # Check tenant
                    meta = self._vector_metadata[model_name].get(memory_id, {})
                    if meta.get("tenant_id") == tenant_id:
                        del self._vector_indices[model_name][memory_id]
                        del self._vector_metadata[model_name][memory_id]
                        # We don't compact the arena immediately (expensive). 
                        # Fragmentation is accepted in this simulated version.
                        deleted = True
            return deleted

    async def update_vector(
        self,
        memory_id: UUID,
        embedding: list[float] | dict[str, list[float]],
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a vector embedding."""
        return await self.store_vector(memory_id, embedding, tenant_id, metadata)

    async def batch_store_vectors(
        self,
        vectors: list[
            tuple[UUID, list[float] | dict[str, list[float]], dict[str, Any]]
        ],
        tenant_id: str,
    ) -> int:
        """Store multiple vectors."""
        count = 0
        for mid, emb, meta in vectors:
            if await self.store_vector(mid, emb, tenant_id, meta):
                count += 1
        return count

    # =========================================================================
    # IMemoryStorage Implementation (Legacy + Core)
    # =========================================================================

    async def save_embedding(
        self,
        memory_id: UUID,
        model_name: str,
        embedding: list[float],
        tenant_id: str,
        **kwargs: Any,
    ) -> bool:
        """Legacy alias for store_vector."""
        async with self._lock:
            memory = self._memories.get(memory_id)
            if not memory:
                return False
            if memory["tenant_id"] != tenant_id:
                raise ValueError("Access Denied")
        
        # Wrap single vector as dict for store_vector
        return await self.store_vector(
            memory_id, 
            {model_name: embedding}, 
            tenant_id, 
            kwargs.get("metadata")
        )

    async def store_reflection_audit(
        self,
        query_id: str,
        tenant_id: str,
        fsi_score: float,
        final_decision: str,
        l1_report: dict[str, Any],
        l2_report: dict[str, Any],
        l3_report: dict[str, Any],
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Store a reflection audit result."""
        async with self._lock:
            audit_id = uuid4()
            self._reflection_audits[audit_id] = {
                "id": audit_id,
                "query_id": query_id,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "fsi_score": fsi_score,
                "final_decision": final_decision,
                "l1_report": l1_report,
                "l2_report": l2_report,
                "l3_report": l3_report,
                "metadata": metadata or {},
                "created_at": self._clock.now()
            }
            return audit_id

    async def store_memory(self, **kwargs: Any) -> UUID:
        """Store a new memory."""
        async with self._lock:
            memory_id = uuid4()
            now = self._clock.now()

            content = kwargs.get("content", "")
            layer = kwargs.get("layer", "episodic")
            tenant_id = kwargs.get("tenant_id", "default")
            agent_id = kwargs.get("agent_id", "default")
            tags = kwargs.get("tags") or []
            metadata = kwargs.get("metadata") or {}
            embedding = kwargs.get("embedding")
            importance = kwargs.get("importance", 0.5)
            expires_at = kwargs.get("expires_at")
            memory_type = kwargs.get("memory_type", "text")
            strength = kwargs.get("strength", 1.0)

            memory = {
                "id": memory_id,
                "content": content,
                "layer": layer,
                "tenant_id": tenant_id,
                "agent_id": agent_id,
                "tags": tags,
                "metadata": metadata,
                # embedding is stored separately in vector store usually, 
                # but we keep a reference or copy if needed. 
                # InMemoryStorage used to store it in _memories too? 
                # The old code had: "embedding": embedding
                "embedding": embedding, 
                "importance": importance,
                "created_at": now,
                "modified_at": now,
                "last_accessed_at": now,
                "expires_at": expires_at,
                "access_count": 0,
                "usage_count": 0,
                "memory_type": memory_type,
                "strength": strength,
                "version": 1,
            }

            # Store memory
            self._memories[memory_id] = memory

            # Update indexes
            self._by_tenant[tenant_id].add(memory_id)
            self._by_agent[(tenant_id, agent_id)].add(memory_id)
            self._by_layer[(tenant_id, layer)].add(memory_id)

            for tag in tags or []:
                self._by_tags[(tenant_id, tag)].add(memory_id)
                
            # Bloom Filter (Phase 2: The Scalpel)
            if tags:
                mask = bloom_filter_fingerprint(tags)
                self._bloom_filters[memory_id] = mask
                
            # If embedding provided, store in vector store part as well
            if embedding:
                # Need to release lock if calling self.store_vector which acquires lock?
                # self.store_vector is async and uses lock.
                # Re-entrant lock is not available in asyncio.Lock.
                # We must manually invoke vector storage logic WITHOUT acquiring lock again.
                # Refactoring: Extract logic to _store_vector_internal
                
                # For now, simplistic approach: duplicate logic or assume embedding handles it.
                # Actually, self.store_vector is public API.
                # Let's just inline the basic vector storage here since we have the lock.
                
                # Normalize
                vectors = {}
                if isinstance(embedding, list):
                    vectors["default"] = embedding
                elif isinstance(embedding, dict):
                    vectors = embedding
                
                for m_name, vec in vectors.items():
                    # Quantize
                    v_bytes = quantize_vector_bytes(vec)
                    dim = len(vec)
                    if m_name not in self._vector_dims:
                        self._vector_dims[m_name] = dim
                    
                    if m_name not in self._vector_arenas:
                        self._vector_arenas[m_name] = bytearray()
                        
                    offset = len(self._vector_arenas[m_name])
                    self._vector_arenas[m_name].extend(v_bytes)
                    self._vector_indices[m_name][memory_id] = offset
                    
                    # Metadata for vector
                    v_meta = metadata.copy()
                    v_meta.update({
                        "tenant_id": tenant_id,
                        "layer": layer,
                        "agent_id": agent_id,
                        "tags": tags or []
                    })
                    self._vector_metadata[m_name][memory_id] = v_meta

            return memory_id

    async def get_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> dict[str, Any] | None:
        """Retrieve a memory by ID."""
        async with self._lock:
            memory = self._memories.get(memory_id)

            if memory and memory["tenant_id"] == tenant_id:
                return memory.copy()

            return None

    async def get_memories_batch(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
    ) -> list[dict[str, Any]]:
        """Retrieve multiple memories by IDs."""
        async with self._lock:
            results = []
            for mid in memory_ids:
                memory = self._memories.get(mid)
                if memory and memory["tenant_id"] == tenant_id:
                    results.append(memory.copy())
            return results

    async def update_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update a memory."""
        async with self._lock:
            memory = self._memories.get(memory_id)

            if not memory or memory["tenant_id"] != tenant_id:
                return False

            # Handle tag updates (update index)
            if "tags" in updates:
                old_tags = set(memory.get("tags", []))
                new_tags = set(updates.get("tags", []))

                removed_tags = old_tags - new_tags
                added_tags = new_tags - old_tags

                for tag in removed_tags:
                    self._by_tags[(tenant_id, tag)].discard(memory_id)

                for tag in added_tags:
                    self._by_tags[(tenant_id, tag)].add(memory_id)

            # Handle layer updates (update index)
            if "layer" in updates:
                old_layer = memory.get("layer")
                new_layer = updates.get("layer")

                if old_layer != new_layer:
                    self._by_layer[(tenant_id, cast(str, old_layer))].discard(memory_id)
                    self._by_layer[(tenant_id, cast(str, new_layer))].add(memory_id)

            # Update memory
            memory.update(updates)
            memory["modified_at"] = self._clock.now()
            memory["version"] = memory.get("version", 1) + 1

            return True

    async def delete_memory(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Delete a memory."""
        async with self._lock:
            memory = self._memories.get(memory_id)

            if not memory or memory["tenant_id"] != tenant_id:
                return False

            # Remove from indexes
            self._by_tenant[tenant_id].discard(memory_id)
            self._by_agent[(tenant_id, memory["agent_id"])].discard(memory_id)
            self._by_layer[(tenant_id, memory["layer"])].discard(memory_id)

            for tag in memory.get("tags", []):
                self._by_tags[(tenant_id, tag)].discard(memory_id)

            # Remove memory
            del self._memories[memory_id]
            
            # Also remove vectors if present (naive approach without calling delete_vector to avoid deadlock)
            # Just remove from indices, arena remains fragmented
            for model_name in list(self._vector_indices.keys()):
                if memory_id in self._vector_indices[model_name]:
                    del self._vector_indices[model_name][memory_id]
                    del self._vector_metadata[model_name][memory_id]

            return True

    async def list_memories(
        self, tenant_id: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """List memories with filtering."""
        async with self._lock:
            agent_id = kwargs.get("agent_id")
            layer = kwargs.get("layer")
            tags = kwargs.get("tags")
            limit = kwargs.get("limit", 100)
            offset = kwargs.get("offset", 0)

            # Start with tenant memories
            candidate_ids = self._by_tenant[tenant_id].copy()

            # Apply filters using indexes
            if agent_id:
                candidate_ids &= self._by_agent[(tenant_id, agent_id)]

            if layer:
                candidate_ids &= self._by_layer[(tenant_id, layer)]

            if tags:
                # OR logic for tags
                tag_ids = set()
                for tag in tags:
                    tag_ids |= self._by_tags[(tenant_id, tag)]
                candidate_ids &= tag_ids

            # Get memories and sort by created_at
            memories = [
                self._memories[mid].copy()
                for mid in candidate_ids
                if mid in self._memories
            ]
            memories.sort(key=lambda m: m["created_at"], reverse=True)

            # Apply pagination
            return memories[offset : offset + limit]

    async def count_memories(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> int:
        """Count memories matching filters."""
        async with self._lock:
            if not tenant_id:
                return len(self._memories)

            # Start with tenant memories
            candidate_ids = self._by_tenant[tenant_id].copy()

            # Apply filters
            if agent_id:
                candidate_ids &= self._by_agent[(tenant_id, agent_id)]

            if layer:
                candidate_ids &= self._by_layer[(tenant_id, layer)]

            return len(candidate_ids)

    async def get_statistics(self) -> dict[str, Any]:
        """Get storage statistics."""
        async with self._lock:
            return {
                "total_memories": len(self._memories),
                "tenants": len(self._by_tenant),
                "agents": len(self._by_agent),
                "layers": len(self._by_layer),
                "unique_tags": len(self._by_tags),
                "vector_models": list(self._vector_arenas.keys()),
                "vector_arena_sizes_bytes": {k: len(v) for k, v in self._vector_arenas.items()}
            }

    async def clear_all(self) -> int:
        """Clear all data."""
        async with self._lock:
            count = len(self._memories)

            self._memories.clear()
            self._by_tenant.clear()
            self._by_agent.clear()
            self._by_layer.clear()
            self._by_tags.clear()
            
            self._vector_arenas.clear()
            self._vector_indices.clear()
            self._vector_metadata.clear()
            self._vector_dims.clear()

            return count

    async def delete_memories_with_metadata_filter(
        self,
        tenant_id: str | None = None,
        agent_id: str | None = None,
        layer: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories matching metadata filter."""
        async with self._lock:
            matching_ids = []
            for memory_id, memory in self._memories.items():
                if tenant_id and memory["tenant_id"] != tenant_id:
                    continue
                if agent_id and memory["agent_id"] != agent_id:
                    continue
                if layer and memory["layer"] != layer:
                    continue

                if metadata_filter and not self._matches_metadata_filter(
                    memory.get("metadata", {}), metadata_filter
                ):
                    continue

                matching_ids.append(memory_id)

            for memory_id in matching_ids:
                self._delete_memory_sync(memory_id)

            return len(matching_ids)

    async def delete_memories_below_importance(
        self,
        tenant_id: str,
        agent_id: str,
        layer: str,
        importance_threshold: float,
    ) -> int:
        """Delete memories below importance threshold."""
        async with self._lock:
            matching_ids = [
                memory_id
                for memory_id, memory in self._memories.items()
                if (
                    memory["tenant_id"] == tenant_id
                    and memory["agent_id"] == agent_id
                    and memory["layer"] == layer
                    and memory.get("importance", 0) < importance_threshold
                )
            ]

            for memory_id in matching_ids:
                self._delete_memory_sync(memory_id)

            return len(matching_ids)

    async def search_memories(
        self,
        query: str,
        tenant_id: str,
        agent_id: str,
        layer: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search memories using simple substring matching."""
        async with self._lock:
            results = []
            query_lower = query.lower()

            for memory in self._memories.values():
                if (
                    memory["tenant_id"] == tenant_id
                    and memory["agent_id"] == agent_id
                    and (layer is None or memory["layer"] == layer)
                ):
                    # Simple substring search in content
                    content_lower = memory["content"].lower()
                    if query_lower in content_lower:
                        # Apply filters if present
                        filters = kwargs.get("filters")
                        if filters:
                            match = True
                            for k, v in filters.items():
                                if k == "tags":
                                    query_tags = set(v) if isinstance(v, (list, tuple)) else {v}
                                    mem_tags = set(memory.get("tags", []))
                                    if not query_tags.issubset(mem_tags):
                                        match = False
                                        break
                                    continue
                                if memory.get(k) != v and memory.get("metadata", {}).get(k) != v:
                                    match = False
                                    break
                            if not match:
                                continue

                        # Calculate simple score based on position
                        score = 1.0 - (
                            content_lower.index(query_lower) / len(content_lower)
                        )
                        results.append(
                            {
                                "id": memory["id"],
                                "content": memory["content"],
                                "score": score,
                                "importance": memory.get("importance", 0.5),
                                "memory": memory.copy(),
                            }
                        )

            # Sort by score descending
            results.sort(key=lambda x: x["score"], reverse=True)

            return results[:limit]

    async def delete_expired_memories(
        self,
        tenant_id: str,
        agent_id: str | None = None,
        layer: str | None = None,
    ) -> int:
        """Delete expired memories."""
        async with self._lock:
            now = self._clock.now()
            matching_ids = []
            for memory_id, memory in self._memories.items():
                if memory["tenant_id"] != tenant_id:
                    continue
                if agent_id and memory["agent_id"] != agent_id:
                    continue
                if layer and memory["layer"] != layer:
                    continue

                if memory.get("expires_at") and memory["expires_at"] < now:
                    matching_ids.append(memory_id)

            for memory_id in matching_ids:
                self._delete_memory_sync(memory_id)

            return len(matching_ids)

    async def update_memory_access(
        self,
        memory_id: UUID,
        tenant_id: str,
    ) -> bool:
        """Update last access time and increment usage count."""
        async with self._lock:
            memory = self._memories.get(memory_id)

            if not memory or memory["tenant_id"] != tenant_id:
                return False

            memory["last_accessed_at"] = self._clock.now()
            memory["access_count"] = memory.get("access_count", 0) + 1
            memory["usage_count"] = memory.get("usage_count", 0) + 1

            return True

    async def increment_access_count(self, memory_id: UUID, tenant_id: str) -> bool:
        """Alias for update_memory_access (Legacy test support)."""
        return await self.update_memory_access(memory_id, tenant_id)

    async def update_memory_expiration(
        self,
        memory_id: UUID,
        tenant_id: str,
        expires_at: datetime | None,
    ) -> bool:
        """Update memory expiration time."""
        async with self._lock:
            memory = self._memories.get(memory_id)

            if not memory or memory["tenant_id"] != tenant_id:
                return False

            memory["expires_at"] = expires_at
            memory["modified_at"] = self._clock.now()

            return True

    async def get_metric_aggregate(
        self,
        tenant_id: str,
        metric: str,
        func: str,
        filters: dict[str, Any] | None = None,
    ) -> float:
        """Calculate aggregate metric."""
        async with self._lock:
            values = []
            for memory in self._memories.values():
                if memory["tenant_id"] != tenant_id:
                    continue

                # Apply filters
                if filters:
                    match = True
                    for k, v in filters.items():
                        if memory.get(k) != v:
                            match = False
                            break
                    if not match:
                        continue

                val = memory.get(metric)
                if val is not None:
                    values.append(float(val))

            if not values:
                return 0.0

            if func == "sum":
                return sum(values)
            if func == "avg":
                return sum(values) / len(values)
            if func == "max":
                return max(values)
            if func == "min":
                return min(values)
            if func == "count":
                return float(len(values))
            return 0.0

    async def update_memory_access_batch(
        self,
        memory_ids: list[UUID],
        tenant_id: str,
    ) -> bool:
        """Update access count for multiple memories."""
        for mid in memory_ids:
            await self.update_memory_access(mid, tenant_id)
        return True

    async def adjust_importance(
        self,
        memory_id: UUID,
        delta: float,
        tenant_id: str,
    ) -> float:
        """Adjust memory importance."""
        async with self._lock:
            memory = self._memories.get(memory_id)
            if not memory or memory["tenant_id"] != tenant_id:
                return 0.0

            new_imp = float(memory.get("importance", 0.5)) + delta
            new_imp = max(0.0, min(1.0, new_imp))
            memory["importance"] = new_imp
            memory["modified_at"] = self._clock.now()
            return new_imp

    async def decay_importance(
        self,
        tenant_id: str,
        decay_factor: float,
    ) -> int:
        """Apply importance decay to all memories for a tenant."""
        async with self._lock:
            count = 0
            for memory_id in self._by_tenant[tenant_id]:
                memory = self._memories.get(memory_id)
                if not memory:
                    continue

                current = float(memory.get("importance", 0.5))
                new_val = current * decay_factor
                memory["importance"] = new_val
                count += 1
            return count

    async def clear_tenant(self, tenant_id: str) -> int:
        """Delete all memories for a tenant."""
        async with self._lock:
            mids = list(self._by_tenant[tenant_id])
            for mid in mids:
                self._delete_memory_sync(mid)
            
            # Clean up the tenant index key
            if tenant_id in self._by_tenant:
                del self._by_tenant[tenant_id]
                
            return len(mids)

    async def close(self) -> None:
        """Close storage connection."""
        pass

    def _matches_metadata_filter(
        self, metadata: dict[str, Any], filter_dict: dict[str, Any]
    ) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def _delete_memory_sync(self, memory_id: UUID) -> None:
        """Internal delete helper (assumes lock is held)."""
        memory = self._memories.get(memory_id)
        if not memory:
            return

        # Remove from main storage
        del self._memories[memory_id]

        # Remove from indexes
        tenant_id = memory["tenant_id"]
        agent_id = memory["agent_id"]
        layer = memory["layer"]

        self._by_tenant[tenant_id].discard(memory_id)
        self._by_agent[(tenant_id, agent_id)].discard(memory_id)
        self._by_layer[(tenant_id, layer)].discard(memory_id)

        for tag in memory.get("tags", []):
            self._by_tags[(tenant_id, tag)].discard(memory_id)
        
        # Remove from vector index (fragmentation remains)
        for model_name in list(self._vector_indices.keys()):
             if memory_id in self._vector_indices[model_name]:
                 del self._vector_indices[model_name][memory_id]
                 del self._vector_metadata[model_name][memory_id]
