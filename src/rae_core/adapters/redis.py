"""Redis cache adapter for RAE-core.

Implements ICacheProvider interface using Redis for caching.
"""

import json
from typing import Any, Optional

try:
    import redis.asyncio as aioredis_mod

    aioredis = aioredis_mod
except ImportError:  # pragma: no cover
    aioredis = None  # type: ignore[assignment]

from ..interfaces.cache import ICacheProvider


class RedisCache(ICacheProvider):
    """Redis implementation of ICacheProvider.

    Requires redis package with async support.

    Features:
    - TTL-based expiration
    - JSON serialization for complex objects
    - Connection pooling
    - Prefix support for namespace isolation
    """

    def __init__(
        self,
        url: str | None = None,
        redis_client: Optional["aioredis.Redis[Any]"] = None,
        prefix: str = "rae:",
        default_ttl: int = 3600,
        **redis_kwargs: Any,
    ) -> None:
        """Initialize Redis cache.

        Args:
            url: Redis connection URL (e.g., redis://localhost:6379/0)
            redis_client: Existing Redis client instance
            prefix: Key prefix for namespace isolation (default: "rae:")
            default_ttl: Default TTL in seconds (default: 3600 = 1 hour)
            **redis_kwargs: Additional arguments for Redis client
        """
        if aioredis is None:
            raise ImportError(
                "redis is required for RedisCache. "
                "Install with: pip install redis[hiredis]"
            )

        self.prefix = prefix
        self.default_ttl = default_ttl

        if redis_client:
            self.redis = redis_client
        elif url:
            self.redis = aioredis.from_url(url, **redis_kwargs)
        else:
            # Default to localhost
            self.redis = aioredis.Redis(
                host="localhost", port=6379, db=0, **redis_kwargs
            )

    def _make_key(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Add prefix and optional namespace scopes to key."""
        base = f"{self.prefix}"
        if agent_id:
            base += f"{agent_id}:"
        if session_id:
            base += f"{session_id}:"
        return f"{base}{key}"

    async def get(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> Any | None:
        """Get value from cache."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            value = await self.redis.get(full_key)
            if value is None:
                return None

            # Try to deserialize as JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Return as string if not JSON
                return value.decode("utf-8") if isinstance(value, bytes) else value
        except Exception:
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Set value in cache with optional TTL."""
        full_key = self._make_key(key, agent_id, session_id)
        ttl = ttl if ttl is not None else self.default_ttl

        try:
            # Serialize complex objects as JSON
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value)
            elif not isinstance(value, (str, bytes, int, float)):
                value = str(value)

            if ttl > 0:
                await self.redis.setex(full_key, ttl, value)
            else:
                await self.redis.set(full_key, value)

            return True
        except Exception:
            return False

    async def delete(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Delete value from cache."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            result = await self.redis.delete(full_key)
            return result > 0
        except Exception:
            return False

    async def exists(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Check if key exists in cache."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            result = await self.redis.exists(full_key)
            return result > 0
        except Exception:
            return False

    async def increment(
        self,
        key: str,
        amount: int = 1,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """Increment a counter."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            result = await self.redis.incrby(full_key, amount)
            return int(result)
        except Exception:
            return None

    async def decrement(
        self,
        key: str,
        amount: int = 1,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """Decrement a counter."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            result = await self.redis.decrby(full_key, amount)
            return int(result)
        except Exception:
            return None

    async def get_ttl(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """Get remaining TTL for a key in seconds."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            ttl = await self.redis.ttl(full_key)
            # TTL returns -1 if key exists but has no expiration
            # Returns -2 if key doesn't exist
            return ttl if ttl >= -1 else None
        except Exception:
            return None

    async def expire(
        self,
        key: str,
        ttl: int,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Set expiration on existing key."""
        full_key = self._make_key(key, agent_id, session_id)

        try:
            result = await self.redis.expire(full_key, ttl)
            return result
        except Exception:
            return False

    async def clear(
        self,
        pattern: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear cache keys matching pattern (all if None)."""
        return await self.clear_prefix(pattern or "*", agent_id, session_id)

    async def clear_prefix(
        self,
        pattern: str = "*",
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear all keys matching pattern under prefix.

        Args:
            pattern: Pattern to match (default: "*" = all keys with prefix)
            agent_id: Optional agent ID for namespace isolation
            session_id: Optional session ID for namespace isolation

        Returns:
            Number of keys deleted
        """
        full_pattern = self._make_key(pattern, agent_id, session_id)

        try:
            keys = []
            async for key in self.redis.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                deleted = await self.redis.delete(*keys)
                return deleted
            return 0
        except Exception:
            return 0

    async def ping(self) -> bool:
        """Check if Redis is accessible."""
        try:
            return await self.redis.ping()
        except Exception:
            return False

    async def close(self) -> None:
        """Close Redis connection."""
        try:
            await self.redis.close()
        except Exception:
            pass
