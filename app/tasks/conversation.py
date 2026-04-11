import asyncio
import logging
import re
from collections import Counter

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

    analysis = _analyze_conversation(conversation_history)

    # ── 4. UPDATE conversation row → completed ────────────────────────────
    await (
        supabase.table("conversations")
        .update({
            "status": "completed",
            "summary": summary,
            "ended_at": "now()",
            "message_count": len(conversation_history),
            "had_tool_call": analysis["had_tool_call"],
        })
        .eq("id", conversation_id)
        .execute()
    )

    # Outcome is best-effort so it never blocks status finalization.
    try:
        await (
            supabase.table("conversations")
            .update({"outcome": analysis["outcome"]})
            .eq("id", conversation_id)
            .execute()
        )
    except Exception as exc:
        logger.warning(
            "Could not set outcome for conv_id=%s: %s",
            conversation_id,
            exc,
        )

    # Persist V2 analytics row (upsert by conversation_id)
    try:
        await (
            supabase.table("conversation_analytics")
            .upsert(
                {
                    "conversation_id": conversation_id,
                    "overall_intent": analysis["overall_intent"],
                    "sentiment_start": analysis["sentiment_start"],
                    "sentiment_end": analysis["sentiment_end"],
                    "sentiment_arc": analysis["sentiment_arc"],
                    "topics": analysis["topics"],
                    "entities_mentioned": analysis["entities_mentioned"],
                    "outcome": analysis["outcome"],
                    "resolution_turns": analysis["resolution_turns"],
                    "key_phrases": analysis["key_phrases"],
                    "tool_calls_made": analysis["tool_calls_made"],
                },
                on_conflict="conversation_id",
            )
            .execute()
        )
    except Exception as exc:
        logger.warning(
            "Could not upsert conversation_analytics conv_id=%s: %s",
            conversation_id,
            exc,
        )

    logger.info(
        "save_conversation complete: call_sid=%s conv_id=%s", call_sid, conversation_id
    )
    return {
        "status": "ok",
        "conversation_id": conversation_id,
        "message_count": len(conversation_history),
    }


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


_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "have", "your", "from", "what",
    "would", "there", "about", "hello", "thanks", "thank", "please", "agent", "call",
    "just", "like", "need", "want", "you", "are", "was", "were", "will", "can",
}


def _analyze_conversation(conversation_history: list[dict[str, str]]) -> dict:
    user_turns = [t.get("content", "") for t in conversation_history if t.get("role") == "user"]
    assistant_turns = [t.get("content", "") for t in conversation_history if t.get("role") == "assistant"]
    all_text = " ".join([*user_turns, *assistant_turns]).lower()

    intent_keywords = {
        "booking": ["book", "appointment", "schedule", "slot"],
        "complaint": ["complaint", "issue", "problem", "angry", "bad"],
        "pricing": ["price", "pricing", "cost", "fee", "charge"],
        "support": ["help", "support", "assist", "guidance"],
        "cancellation": ["cancel", "refund", "stop"],
    }
    intent_scores = {
        intent: sum(all_text.count(token) for token in tokens)
        for intent, tokens in intent_keywords.items()
    }
    overall_intent = max(intent_scores, key=intent_scores.get) if any(intent_scores.values()) else "general"

    positive_tokens = ("thanks", "thank you", "great", "good", "perfect", "awesome", "resolved")
    negative_tokens = ("problem", "issue", "not working", "bad", "angry", "frustrated", "complaint")

    def _sentiment(text: str) -> str:
        text_l = text.lower()
        pos = sum(text_l.count(token) for token in positive_tokens)
        neg = sum(text_l.count(token) for token in negative_tokens)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    sentiment_start = _sentiment(user_turns[0]) if user_turns else "neutral"
    sentiment_end = _sentiment(user_turns[-1] if user_turns else all_text)
    sentiment_arc = [
        {"turn": index + 1, "sentiment": _sentiment(turn.get("content", ""))}
        for index, turn in enumerate(conversation_history)
    ]

    topic_keywords = {
        "pricing": ["price", "pricing", "cost"],
        "support": ["issue", "support", "help"],
        "booking": ["book", "appointment", "schedule"],
        "delivery": ["delivery", "shipping", "courier"],
    }
    topics = [
        topic for topic, tokens in topic_keywords.items()
        if any(token in all_text for token in tokens)
    ]

    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", all_text)
    times = re.findall(r"\b\d{1,2}:\d{2}\b", all_text)
    phone_numbers = re.findall(r"\+?\d{10,15}\b", all_text)
    entities_mentioned = {
        "dates": dates[:10],
        "times": times[:10],
        "phones": phone_numbers[:10],
    }

    tool_markers = ("booked", "appointment confirmed", "sms sent", "transferring", "transfer")
    had_tool_call = any(marker in all_text for marker in tool_markers)
    tool_calls_made = []
    if "booked" in all_text or "appointment" in all_text:
        tool_calls_made.append({"tool": "booking", "success": True})
    if "sms" in all_text:
        tool_calls_made.append({"tool": "send_sms", "success": True})
    if "transfer" in all_text:
        tool_calls_made.append({"tool": "call_transfer", "success": True})

    outcome = "hung_up"
    if any(token in all_text for token in ("appointment confirmed", "booked", "scheduled")):
        outcome = "booked"
    elif any(token in all_text for token in ("transferring", "transferred", "human agent")):
        outcome = "transferred"
    elif any(token in all_text for token in ("resolved", "fixed", "done", "sorted")):
        outcome = "resolved"
    elif any(token in all_text for token in ("not resolved", "could not", "can't help", "cannot help")):
        outcome = "unresolved"

    resolution_turns = None
    for idx, turn in enumerate(conversation_history, start=1):
        text = turn.get("content", "").lower()
        if any(token in text for token in ("resolved", "booked", "transferred", "appointment confirmed")):
            resolution_turns = idx
            break

    words = re.findall(r"\b[a-z]{4,}\b", all_text)
    phrase_counter = Counter(word for word in words if word not in _STOPWORDS)
    key_phrases = [word for word, _ in phrase_counter.most_common(8)]

    return {
        "overall_intent": overall_intent,
        "sentiment_start": sentiment_start,
        "sentiment_end": sentiment_end,
        "sentiment_arc": sentiment_arc,
        "topics": topics,
        "entities_mentioned": entities_mentioned,
        "outcome": outcome,
        "resolution_turns": resolution_turns,
        "key_phrases": key_phrases,
        "had_tool_call": had_tool_call,
        "tool_calls_made": tool_calls_made,
    }