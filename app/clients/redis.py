from typing import Optional
import logging
import time

import redis.asyncio as aioredis
from redis.asyncio import Redis

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Module-level Redis client
_redis_client: Redis | None = None


async def init_redis() -> None:
    """Call once at startup (inside lifespan)."""
    global _redis_client
    if _redis_client is not None:
        return

    # redis.asyncio.from_url is synchronous and returns a client instance
    _redis_client = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )


async def close_redis() -> None:
    """Call once at app shutdown (inside lifespan)."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.close()
        except Exception:
            logger.exception("Error closing redis client")
        try:
            # Best-effort disconnect of the connection pool
            await _redis_client.connection_pool.disconnect()
        except Exception:
            pass
        _redis_client = None


def get_redis() -> Redis:
    """Return the live Redis client.

    Raises RuntimeError if called before init_redis().
    """
    if _redis_client is None:
        raise RuntimeError("Redis not initialised — call init_redis() at startup")
    return _redis_client