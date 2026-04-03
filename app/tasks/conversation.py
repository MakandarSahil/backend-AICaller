import asyncio
import logging

from celery import Celery

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

celery_app = Celery(
    "aicaller",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name="tasks.save_conversation",
)
def save_conversation(
    self,
    *,
    call_sid: str,
    agent_id: str,
    caller_id: str | None,
    caller_phone: str,
    conversation_history: list[dict[str, str]],
    conversation_id: str | None,       # FIX: passed in from session — UPDATE not INSERT
) -> dict:
    """
    Celery task: finalise a completed call in Supabase.

    BUG FIX: Previously this task always INSERTed a new conversations row.
    Now it receives the conversation_id that was created at call start
    (status=active) and UPDATEs it to status=completed + adds summary.
    This keeps the Supabase row lifecycle correct:
        call start  → INSERT status=active     (call_handler._create_active_conversation)
        call end    → UPDATE status=completed  (this task)

    Steps:
        1. INSERT all message rows (linked to existing conversation_id)
        2. Generate 2-3 sentence summary via Groq
        3. UPDATE conversations: status=completed, summary, ended_at, message_count
        4. Increment agent_usage properly (fetch → add → upsert)

    Retries up to 3 times with 10s delay on failure.
    """
    logger.info(
        "save_conversation: call_sid=%s agent_id=%s turns=%d conv_id=%s",
        call_sid, agent_id, len(conversation_history), conversation_id,
    )

    try:
        result = asyncio.run(
            _save_async(
                call_sid=call_sid,
                agent_id=agent_id,
                caller_id=caller_id,
                caller_phone=caller_phone,
                conversation_history=conversation_history,
                conversation_id=conversation_id,
            )
        )
        return result
    except Exception as exc:
        logger.error("save_conversation failed: %s", exc, exc_info=True)
        raise self.retry(exc=exc)


async def _save_async(
    *,
    call_sid: str,
    agent_id: str,
    caller_id: str | None,
    caller_phone: str,
    conversation_history: list[dict[str, str]],
    conversation_id: str | None,
) -> dict:
    from supabase._async.client import create_client
    from groq import AsyncGroq

    # Worker process — initialise its own clients (no shared state with FastAPI)
    supabase = await create_client(
        settings.supabase_url,
        settings.supabase_service_role_key,
    )
    groq = AsyncGroq(api_key=settings.groq_api_key)

    # ── 1. Handle conversation row ─────────────────────────────────────────
    # If call_handler created an active row → UPDATE it.
    # Fallback: if somehow no conversation_id, INSERT now (safety net).
    if not conversation_id:
        logger.warning(
            "No conversation_id for call_sid=%s — inserting fallback row", call_sid
        )
        conv_payload: dict = {
            "agent_id": agent_id,
            "session_id": call_sid,
            "channel": "twilio",
            "status": "completed",
            "message_count": len(conversation_history),
            "started_at": "now()",
        }
        if caller_id:
            conv_payload["caller_id"] = caller_id
        result = await supabase.table("conversations").insert(conv_payload).execute()
        conversation_id = result.data[0]["id"] if result.data else None

    if not conversation_id:
        logger.error("Could not get conversation_id for call_sid=%s", call_sid)
        return {"status": "error", "reason": "no conversation_id"}

    # ── 2. INSERT all message rows ─────────────────────────────────────────
    if conversation_history:
        messages_payload = [
            {
                "conversation_id": conversation_id,
                "role": turn["role"],
                "content": turn["content"],
            }
            for turn in conversation_history
        ]
        await supabase.table("messages").insert(messages_payload).execute()
        logger.info(
            "Messages saved: conv_id=%s count=%d", conversation_id, len(conversation_history)
        )

    # ── 3. Generate summary ────────────────────────────────────────────────
    summary = await _generate_summary(groq, conversation_history)

    # ── 4. UPDATE conversation row → completed ────────────────────────────
    await (
        supabase.table("conversations")
        .update({
            "status": "completed",
            "summary": summary,
            "ended_at": "now()",
            "message_count": len(conversation_history),
        })
        .eq("id", conversation_id)
        .execute()
    )

    # ── 5. Increment agent_usage properly ─────────────────────────────────
    # BUG FIX: Previous code set total_calls=1 every time (overwrote instead
    # of incremented). Now we fetch current values first, then add to them.
    await _increment_agent_usage(supabase, agent_id, len(conversation_history))

    logger.info(
        "save_conversation complete: call_sid=%s conv_id=%s", call_sid, conversation_id
    )
    return {
        "status": "ok",
        "conversation_id": conversation_id,
        "message_count": len(conversation_history),
    }


async def _increment_agent_usage(supabase, agent_id: str, message_count: int) -> None:
    """
    Safely increment total_calls and total_messages on agent_usage.

    BUG FIX: The old upsert({ total_calls: 1 }) overwrote the count every time.
    Correct approach: fetch current values → add → upsert with updated values.
    Single Celery worker with concurrency=2 makes this safe enough for v1
    (two tasks finishing simultaneously is extremely unlikely for the same agent).
    """
    try:
        current = (
            await supabase.table("agent_usage")
            .select("total_calls, total_messages")
            .eq("agent_id", agent_id)
            .maybe_single()
            .execute()
        )

        if current.data:
            new_calls = (current.data.get("total_calls") or 0) + 1
            new_messages = (current.data.get("total_messages") or 0) + message_count
        else:
            new_calls = 1
            new_messages = message_count

        await (
            supabase.table("agent_usage")
            .upsert(
                {
                    "agent_id": agent_id,
                    "total_calls": new_calls,
                    "total_messages": new_messages,
                    "last_active_at": "now()",
                },
                on_conflict="agent_id",
            )
            .execute()
        )
    except Exception as exc:
        logger.error("agent_usage increment failed for agent=%s: %s", agent_id, exc)


async def _generate_summary(groq, conversation_history: list[dict[str, str]]) -> str:
    """
    Generate a 2-3 sentence call summary via a single non-streaming Groq call.
    Returns empty string on failure — does not block the rest of the task.
    """
    if not conversation_history:
        return ""

    transcript = "\n".join(
        f"{'Caller' if t['role'] == 'user' else 'Assistant'}: {t['content']}"
        for t in conversation_history
    )

    try:
        response = await groq.chat.completions.create(
            model=settings.groq_model,
            max_tokens=200,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a call summarizer. Write a concise 2-3 sentence summary: "
                        "what the caller wanted, what was resolved, any follow-up needed. "
                        "Be factual and brief."
                    ),
                },
                {"role": "user", "content": f"Summarize:\n\n{transcript}"},
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Summary generation failed: %s", exc)
        return ""