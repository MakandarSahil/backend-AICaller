import json
import logging
from typing import Any

from app.clients.redis import get_redis
from app.clients.supabase import get_supabase
from app.config import get_settings
from app.models.session import SessionState
from app.pipeline.prompt import build_system_prompt_block

settings = get_settings()
logger = logging.getLogger(__name__)

# In-process session store — Dict[call_sid, SessionState]
# One entry per active call. Isolated per call_sid.
_sessions: dict[str, SessionState] = {}


# ── Session CRUD ──────────────────────────────────────────────────────────────

def create_session(call_sid: str, agent_id: str) -> SessionState:
    session = SessionState(call_sid=call_sid, agent_id=agent_id)
    _sessions[call_sid] = session
    logger.info("Session created: call_sid=%s agent_id=%s", call_sid, agent_id)
    return session


def get_session(call_sid: str) -> SessionState | None:
    return _sessions.get(call_sid)


def delete_session(call_sid: str) -> None:
    session = _sessions.pop(call_sid, None)
    if session:
        logger.info(
            "Session deleted: call_sid=%s turns=%d",
            call_sid,
            len(session.conversation_history),
        )


def active_session_count() -> int:
    return len(_sessions)


# ── Context Loading (Redis cache → Supabase fallback) ─────────────────────────

async def load_agent_config(agent_id: str) -> dict[str, Any]:
    """
    Load agent config from Redis cache (5 min TTL).
    On cache miss, fetch from Supabase and populate cache.

    Returns the agent row dict. Keys include:
        system_prompt, tts_voice, tts_provider, tts_model,
        stt_provider, stt_model, llm_provider, llm_model, name
    """
    redis = get_redis()
    cache_key = f"agent:{agent_id}"

    cached = await redis.get(cache_key)
    if cached:
        logger.debug("Agent config cache HIT: %s", agent_id)
        return json.loads(cached)

    logger.debug("Agent config cache MISS: %s — fetching from Supabase", agent_id)
    supabase = get_supabase()
    result = (
        await supabase.table("agents")
        .select("*")
        .eq("id", agent_id)
        .single()
        .execute()
    )

    if not result.data:
        logger.error("Agent not found in Supabase: %s", agent_id)
        return {}

    agent_data = result.data
    await redis.setex(cache_key, settings.cache_ttl_agent, json.dumps(agent_data))
    return agent_data


async def load_kb_documents(agent_id: str, force_refresh: bool = False) -> list[str]:
    """
    Load all KB document texts attached to this agent.
    Redis cache (5 min TTL) → Supabase fallback.

    Returns a list of raw text strings — one per document.
    Only documents with status='ready' and non-empty content are included.
    """
    redis = get_redis()
    cache_key = f"kb:{agent_id}"

    if not force_refresh:
        cached = await redis.get(cache_key)
        if cached:
            logger.info("KB cache HIT: agent=%s", agent_id)
            return json.loads(cached)
    else:
        logger.info("KB cache BYPASS: agent=%s (force_refresh=true)", agent_id)

    logger.info("KB cache MISS: agent=%s — fetching from Supabase", agent_id)
    supabase = get_supabase()

    # Join: agent_knowledge_bases → knowledge_bases → kb_documents
    result = (
        await supabase.table("agent_knowledge_bases")
        .select("knowledge_bases(id, kb_documents(content, status))")
        .eq("agent_id", agent_id)
        .execute()
    )

    docs: list[str] = []
    attached_kb_ids: list[str] = []
    if result.data:
        for akb in result.data:
            kb = akb.get("knowledge_bases") or {}
            kb_id = kb.get("id")
            if kb_id:
                attached_kb_ids.append(str(kb_id))
            for doc in kb.get("kb_documents") or []:
                if doc.get("status") == "ready" and doc.get("content"):
                    docs.append(doc["content"])

    await redis.setex(cache_key, settings.cache_ttl_kb, json.dumps(docs))
    total_size = sum(len(doc) for doc in docs)
    logger.info(
        "KB loaded: agent=%s attached_kbs=%d docs=%d total_chars=%d cache_ttl=%d kb_ids=%s",
        agent_id,
        len(attached_kb_ids),
        len(docs),
        total_size,
        settings.cache_ttl_kb,
        ",".join(attached_kb_ids[:10]) if attached_kb_ids else "none",
    )
    return docs


async def load_caller_history(caller_id: str) -> list[dict]:
    """
    Load past conversation summaries for this caller.
    Used to give the LLM context about previous interactions.
    Redis cache (30 min TTL) → Supabase fallback.
    """
    if not caller_id:
        return []

    redis = get_redis()
    cache_key = f"caller_history:{caller_id}"

    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    supabase = get_supabase()
    result = (
        await supabase.table("conversations")
        .select("started_at, summary")
        .eq("caller_id", caller_id)
        .eq("status", "completed")
        .not_.is_("summary", "null")
        .order("started_at", desc=True)
        .limit(settings.caller_history_max_turns)
        .execute()
    )

    history = result.data or []
    await redis.setex(
        cache_key, settings.cache_ttl_caller_history, json.dumps(history)
    )
    return history


async def resolve_or_create_caller(
    workspace_id: str, phone_number: str
) -> str | None:
    """
    Get or create a callers row for this phone number in the given workspace.
    Returns the caller UUID.

    callers table: UNIQUE(workspace_id, phone_number)
    Uses upsert to handle race conditions cleanly.
    """
    if not phone_number:
        return None

    try:
        supabase = get_supabase()
        result = (
            await supabase.table("callers")
            .upsert(
                {
                    "workspace_id": workspace_id,
                    "phone_number": phone_number,
                    "last_seen_at": "now()",
                },
                on_conflict="workspace_id,phone_number",
                returning="representation",
            )
            .execute()
        )
        return result.data[0]["id"] if result.data else None
    except Exception as exc:
        logger.error("Failed to resolve caller: %s", exc)
        return None


async def populate_session(session: SessionState, caller_phone: str) -> None:
    """
    Load all context into a session after it's created.
    Called once on the Twilio 'start' event.

    Loads: agent_config, kb_documents
    """
    session.caller_phone = caller_phone

    # Load agent config and KB in parallel
    agent_config, kb_docs = await _gather_safe(
        load_agent_config(session.agent_id),
        load_kb_documents(session.agent_id),
    )

    session.agent_config = agent_config or {}
    session.kb_documents = kb_docs or []

    # Pre-build the static system prompt block (system + KB + caller history placeholder)
    # This avoids rebuilding it on every LLM turn.
    # Caller history will be appended later when available.
    session.cached_system_prompt = build_system_prompt_block(
        agent_config=session.agent_config,
        kb_documents=session.kb_documents,
        caller_history=[],  # will be updated when caller history loads
    )

    logger.info(
        "Session populated: call_sid=%s agent=%s kb_docs=%d",
        session.call_sid,
        session.agent_id,
        len(session.kb_documents),
    )


async def preload_caller_data(session: SessionState) -> None:
    """
    Preload caller history in the background after caller_id is resolved.
    Updates the cached system prompt with caller history.
    Called from the background DB setup task.
    """
    if session.caller_id:
        session.caller_history = await load_caller_history(session.caller_id)

        # Rebuild cached system prompt with caller history now available
        session.cached_system_prompt = build_system_prompt_block(
            agent_config=session.agent_config,
            kb_documents=session.kb_documents,
            caller_history=session.caller_history,
        )

        logger.info(
            "Caller history preloaded: call_sid=%s caller_id=%s turns=%d",
            session.call_sid,
            session.caller_id,
            len(session.caller_history),
        )


async def _gather_safe(*coros):
    """Run coroutines concurrently, returning None for any that fail."""
    import asyncio
    results = await asyncio.gather(*coros, return_exceptions=True)
    return [None if isinstance(r, Exception) else r for r in results]