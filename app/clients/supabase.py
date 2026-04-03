from supabase._async.client import AsyncClient, create_client
from app.config import get_settings

settings = get_settings()

# Module-level singleton — initialised in main.py lifespan.
_supabase_client: AsyncClient | None = None


async def init_supabase() -> None:
    """Call once at app startup (inside lifespan)."""
    global _supabase_client
    # create_client is async in supabase-py v2.10.0
    _supabase_client = await create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )


async def close_supabase() -> None:
    """Call once at app shutdown (inside lifespan)."""
    global _supabase_client
    # supabase-py v2 async client doesn't expose explicit close
    # but we null the reference so GC can clean up the httpx session
    _supabase_client = None


def get_supabase() -> AsyncClient:
    """
    Return the live Supabase service-role client.
    This client bypasses all RLS — only use server-side.
    Raises RuntimeError if called before init_supabase().
    """
    if _supabase_client is None:
        raise RuntimeError("Supabase not initialised — call init_supabase() at startup")
    return _supabase_client