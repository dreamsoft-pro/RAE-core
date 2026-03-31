"""Search result caching for performance optimization."""

import hashlib
import json
from typing import Any
from uuid import UUID

from rae_core.interfaces.cache import ICacheProvider


class SearchCache:
    """Cache for search results to improve performance.

    Caches search results with configurable TTL to reduce
    redundant computations for repeated queries.
    """

    def __init__(
        self,
        cache_provider: ICacheProvider,
        default_ttl: int = 300,  # 5 minutes
        cache_prefix: str = "search:",
    ):
        """Initialize search cache.

        Args:
            cache_provider: Cache provider implementation
            default_ttl: Default TTL in seconds
            cache_prefix: Prefix for cache keys
        """
        self.cache_provider = cache_provider
        self.default_ttl = default_ttl
        self.cache_prefix = cache_prefix

    def _generate_cache_key(
        self,
        query: str,
        tenant_id: str,
        strategy: str,
        filters: dict[str, Any] | None = None,
    ) -> str:
        """Generate deterministic cache key from search parameters.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            strategy: Strategy name
            filters: Optional filters

        Returns:
            Cache key string
        """
        # Create deterministic representation
        params = {
            "query": query,
            "tenant_id": tenant_id,
            "strategy": strategy,
            "filters": filters or {},
        }

        # Sort keys for deterministic serialization
        params_json = json.dumps(params, sort_keys=True)

        # Generate hash
        key_hash = hashlib.sha256(params_json.encode()).hexdigest()[:16]

        return f"{self.cache_prefix}{strategy}:{key_hash}"

    async def get(
        self,
        query: str,
        tenant_id: str,
        strategy: str,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[UUID, float]] | None:
        """Get cached search results.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            strategy: Strategy name
            filters: Optional filters

        Returns:
            Cached results or None if not found/expired
        """
        key = self._generate_cache_key(query, tenant_id, strategy, filters)

        cached_data = await self.cache_provider.get(key)
        if cached_data is None:
            return None

        # Deserialize results
        try:
            results = [(UUID(item["id"]), item["score"]) for item in cached_data]
            return results
        except (KeyError, ValueError, TypeError):
            # Invalid cache data, return None
            return None

    async def set(
        self,
        query: str,
        tenant_id: str,
        strategy: str,
        results: list[tuple[UUID, float]],
        filters: dict[str, Any] | None = None,
        ttl: int | None = None,
    ) -> bool:
        """Cache search results.

        Args:
            query: Search query
            tenant_id: Tenant identifier
            strategy: Strategy name
            results: Search results to cache
            filters: Optional filters
            ttl: Optional custom TTL (uses default if None)

        Returns:
            True if cached successfully
        """
        key = self._generate_cache_key(query, tenant_id, strategy, filters)

        # Serialize results
        serialized = [
            {"id": str(memory_id), "score": score} for memory_id, score in results
        ]

        return await self.cache_provider.set(
            key,
            serialized,
            ttl=ttl or self.default_ttl,
        )

    async def invalidate(
        self,
        tenant_id: str,
        strategy: str | None = None,
    ) -> int:
        """Invalidate cached results.

        Args:
            tenant_id: Tenant identifier
            strategy: Optional strategy name (invalidates all if None)

        Returns:
            Number of keys invalidated
        """
        if strategy:
            pattern = f"{self.cache_prefix}{strategy}:*"
        else:
            pattern = f"{self.cache_prefix}*"

        return await self.cache_provider.clear(pattern)
