"""In-Memory cache adapter for RAE-core.

Simple dictionary-based cache with TTL support using asyncio.
Ideal for testing, development, and single-instance deployments.
"""

import asyncio
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from rae_core.interfaces.cache import ICacheProvider


class InMemoryCache(ICacheProvider):
    """In-memory implementation of ICacheProvider.

    Features:
    - Fast dictionary-based storage
    - TTL support with automatic expiration
    - Thread-safe operations with asyncio.Lock
    - Pattern-based key clearing (glob-style)
    - JSON serialization for complex types
    """

    def __init__(self) -> None:
        """Initialize in-memory cache."""
        # Main storage: {key: (value, expiry_time)}
        self._cache: dict[str, tuple[Any, datetime | None]] = {}

        # Thread safety
        self._lock = asyncio.Lock()

    async def get(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            full_key = self._get_full_key(key, agent_id, session_id)
            if full_key not in self._cache:
                return None

            value, expiry = self._cache[full_key]

            if expiry and datetime.now(timezone.utc) > expiry:
                # Remove expired entry
                del self._cache[full_key]
                return None

            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Set value in cache with optional TTL in seconds."""
        async with self._lock:
            full_key = self._get_full_key(key, agent_id, session_id)
            expiry = None
            if ttl:
                expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl)

            # Store value
            self._cache[full_key] = (value, expiry)

            return True

    async def delete(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Delete value from cache."""
        async with self._lock:
            full_key = self._get_full_key(key, agent_id, session_id)
            if full_key in self._cache:
                del self._cache[full_key]
                return True

            return False

    async def exists(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> bool:
        """Check if key exists in cache."""
        async with self._lock:
            full_key = self._get_full_key(key, agent_id, session_id)
            if full_key not in self._cache:
                return False

            # Check if expired
            value, expiry = self._cache[full_key]
            if expiry and datetime.now(timezone.utc) > expiry:
                # Remove expired entry
                del self._cache[full_key]
                return False

            return True

    async def clear(
        self,
        pattern: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> int:
        """Clear cache keys matching pattern (all if None).

        Pattern supports glob-style wildcards:
        - * matches any characters
        - ? matches single character

        Examples:
        - "user:*" matches "user:123", "user:456"
        - "cache:?:data" matches "cache:1:data", "cache:a:data"

        Args:
            pattern: Optional glob-style pattern
            agent_id: Optional agent filter
            session_id: Optional session filter

        Returns:
            Number of keys deleted
        """
        async with self._lock:
            # Construct prefix if agent_id/session_id provided
            prefix = ""
            if agent_id:
                prefix = f"{agent_id}:"
                if session_id:
                    prefix = f"{agent_id}:{session_id}:"

            if pattern is None:
                if not prefix:
                    # Clear all keys
                    count = len(self._cache)
                    self._cache.clear()
                    return count
                else:
                    # Clear all keys for this agent/session
                    matching_keys = [
                        key for key in self._cache.keys() if key.startswith(prefix)
                    ]
                    for key in matching_keys:
                        del self._cache[key]
                    return len(matching_keys)

            # Convert glob pattern to regex, considering prefix
            search_pattern = f"{prefix}{pattern}" if prefix else pattern
            regex_pattern = self._glob_to_regex(search_pattern)
            compiled_pattern = re.compile(regex_pattern)

            # Find matching keys
            matching_keys = [
                key for key in self._cache.keys() if compiled_pattern.match(key)
            ]

            # Delete matching keys
            for key in matching_keys:
                del self._cache[key]

            return len(matching_keys)

    def _get_full_key(
        self,
        key: str,
        agent_id: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Get full cache key incorporating agent and session IDs."""
        if not agent_id:
            return key
        if not session_id:
            return f"{agent_id}:{key}"
        return f"{agent_id}:{session_id}:{key}"

    @staticmethod
    def _glob_to_regex(pattern: str) -> str:
        """Convert glob pattern to regex pattern.

        Args:
            pattern: Glob pattern string

        Returns:
            Regex pattern string
        """
        # Escape special regex characters except * and ?
        pattern = re.escape(pattern)

        # Replace escaped * and ? with regex equivalents
        pattern = pattern.replace(r"\*", ".*")
        pattern = pattern.replace(r"\?", ".")

        # Anchor pattern
        return f"^{pattern}$"

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            expired_keys = [
                key
                for key, (_, expiry) in self._cache.items()
                if expiry and now > expiry
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    async def get_ttl(self, key: str) -> int | None:
        """Get remaining TTL for a key in seconds.

        Args:
            key: Cache key

        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist
        """
        async with self._lock:
            if key not in self._cache:
                return None

            value, expiry = self._cache[key]

            if not expiry:
                return None

            # Check if expired
            now = datetime.now(timezone.utc)
            if now > expiry:
                del self._cache[key]
                return None

            # Calculate remaining seconds
            remaining = (expiry - now).total_seconds()
            return int(remaining)

    async def set_if_not_exists(
        self, key: str, value: Any, ttl: int | None = None
    ) -> bool:
        """Set value only if key doesn't exist.

        Args:
            key: Cache key
            value: Value to set
            ttl: Optional TTL in seconds

        Returns:
            True if value was set, False if key already exists
        """
        async with self._lock:
            # Check if key exists and is not expired
            if key in self._cache:
                _, expiry = self._cache[key]
                if not expiry or datetime.now(timezone.utc) <= expiry:
                    return False

            # Set value
            expiry = None
            if ttl:
                expiry = datetime.now(timezone.utc) + timedelta(seconds=ttl)

            self._cache[key] = (value, expiry)
            return True

    async def increment(self, key: str, delta: int = 1) -> int:
        """Increment numeric value.

        Args:
            key: Cache key
            delta: Amount to increment (default 1)

        Returns:
            New value after increment
        """
        async with self._lock:
            current_value = 0
            expiry = None

            if key in self._cache:
                value, expiry = self._cache[key]
                # Check if expired
                if not expiry or datetime.now(timezone.utc) <= expiry:
                    try:
                        current_value = int(value)
                    except (ValueError, TypeError):
                        current_value = 0

            new_value = current_value + delta
            self._cache[key] = (new_value, expiry)

            return new_value

    async def get_statistics(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            total_keys = len(self._cache)
            expired_keys = sum(
                1 for _, expiry in self._cache.values() if expiry and now > expiry
            )
            active_keys = total_keys - expired_keys

            keys_with_ttl = sum(1 for _, expiry in self._cache.values() if expiry)

            return {
                "total_keys": total_keys,
                "active_keys": active_keys,
                "expired_keys": expired_keys,
                "keys_with_ttl": keys_with_ttl,
            }

    async def clear_all(self) -> int:
        """Clear all data (use with caution!).

        Returns:
            Number of keys deleted
        """
        return await self.clear(pattern=None)
