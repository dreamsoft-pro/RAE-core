from typing import Any

import asyncpg
import redis.asyncio as aioredis
import structlog
from qdrant_client import AsyncQdrantClient

logger = structlog.get_logger(__name__)


class InfrastructureFactory:
    """
    Factory for initializing infrastructure components based on RAE_PROFILE.
    Decouples the application from specific driver instantiation.
    """

    @staticmethod
    async def initialize(app: Any, settings: Any) -> None:
        """
        Initialize infrastructure connections and attach them to app.state.

        Args:
            app: The FastAPI application instance.
            settings: The application settings object.
        """
        profile = getattr(settings, "RAE_PROFILE", "standard")
        logger.info("initializing_infrastructure", profile=profile)

        # 1. Postgres Initialization
        logger.info("connecting_postgres", host=settings.POSTGRES_HOST)
        # Use DATABASE_URL if available (common in Lite/Docker), else individual fields
        dsn = getattr(settings, "DATABASE_URL", None)
        if dsn:
            app.state.pool = await asyncpg.create_pool(dsn=dsn)
        else:
            app.state.pool = await asyncpg.create_pool(
                host=settings.POSTGRES_HOST,
                database=settings.POSTGRES_DB,
                user=settings.POSTGRES_USER,
                password=settings.POSTGRES_PASSWORD,
            )

        # 2. Redis Initialization
        logger.info("connecting_redis", url=settings.REDIS_URL)
        app.state.redis_client = await aioredis.from_url(
            settings.REDIS_URL, encoding="utf-8", decode_responses=True
        )

        # 3. Qdrant Initialization
        # Check for QDRANT_URL (Lite) or split host/port (Standard)
        qdrant_url = getattr(settings, "QDRANT_URL", None)
        if qdrant_url:
            logger.info("connecting_qdrant_url", url=qdrant_url)
            app.state.qdrant_client = AsyncQdrantClient(url=qdrant_url)
        else:
            logger.info("connecting_qdrant_host", host=settings.QDRANT_HOST)
            app.state.qdrant_client = AsyncQdrantClient(
                host=settings.QDRANT_HOST, port=settings.QDRANT_PORT
            )
