"""
app/routers/query.py — POST /query

The shared text pipeline endpoint. Used by:
    1. Dashboard chat (Supabase JWT auth) - X-Source: dashboard_test (no persistence)
    2. External chatbot integrations (API key auth) - X-Source: widget|api (full persistence + analytics)

Pipeline: text -> KB + history -> build_prompt -> Groq -> SSE or JSON
No STT, no TTS.
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.clients.supabase import get_supabase
from app.middleware.auth import AuthContext, get_auth_context
from app.models.query import Channel, QueryPayload, TextQueryRequest, TextQueryResponse
from app.pipeline.llm import stream_llm_full, stream_llm_sentences
from app.pipeline.tts import synthesize_preview_wav
from app.pipeline.session import load_agent_config, load_kb_documents
from app.tasks.conversation import save_text_conversation

router = APIRouter(tags=["query"])
logger = logging.getLogger(__name__)

_OFF_TOPIC_PATTERNS = (
    "weather",
    "joke",
    "movie",
    "music",
    "song",
    "sports",
    "news",
    "capital of",
    "who are you",
    "what are you",
    "your name",
    "how old are you",
    "translate",
)


class SSEChunk(BaseModel):
    """A single Server-Sent Events chunk (stream=true)."""

    delta: str
    conversation_id: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"delta": "Hello! How can I help you today?", "conversation_id": "uuid-here"},
                {"delta": " Our opening hours are 9am to 6pm."},
            ]
        }
    }


class ErrorResponse(BaseModel):
    detail: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"detail": "Agent not found: abc-123"},
                {"detail": "Agent does not belong to your workspace"},
            ]
        }
    }


class TTSPreviewRequest(BaseModel):
    agent_id: str
    text: str
    voice: str | None = None


@router.post(
    "/query",
    summary="Send a text query to an agent",
    response_description="Streamed SSE response (stream=true) or full JSON (stream=false)",
    response_model=TextQueryResponse,
    responses={
        200: {
            "description": (
                "**stream=false:** Full JSON response.\n\n"
                "**stream=true:** `text/event-stream` - one sentence per chunk. "
                "First chunk includes `conversation_id`. Final message is `data: [DONE]`."
            ),
            "content": {
                "application/json": {
                    "example": {
                        "text": "Our opening hours are Monday to Friday, 9am to 6pm.",
                        "conversation_id": "3f6e8a21-...",
                        "agent_id": "b1c2d3e4-...",
                    }
                },
                "text/event-stream": {
                    "example": (
                        'data: {"delta": "Our opening hours", "conversation_id": "3f6e8a21-..."}\n\n'
                        'data: {"delta": " are Monday to Friday, 9am to 6pm."}\n\n'
                        "data: [DONE]\n\n"
                    )
                },
            },
        },
        401: {
            "description": "No credentials provided",
            "content": {"application/json": {"example": {"detail": "Authentication required."}}},
        },
        403: {
            "description": "Invalid credentials or agent does not belong to your workspace",
            "content": {
                "application/json": {"example": {"detail": "Agent does not belong to your workspace"}}
            },
        },
        404: {
            "description": "Agent not found",
            "content": {"application/json": {"example": {"detail": "Agent not found: abc-123"}}},
        },
    },
)
async def text_query(
    request: TextQueryRequest,
    http_request: Request,
    auth: AuthContext = Depends(get_auth_context),
):
    """
    Send a text message to an agent and receive a response.

    Auth options (either works):
    - Dashboard users -> Authorization: Bearer <supabase_jwt>
    - External developers -> Authorization: Bearer cm_live_xxx or X-API-Key: cm_live_xxx

    Source tracking via X-Source header:
    - X-Source: dashboard_test -> No DB persistence (for dashboard testing)
    - X-Source: widget|api -> Full persistence + analytics
    - No header -> Treated as external API usage
    """
    # Get source from header
    source = http_request.headers.get("X-Source", "api")
    is_test_mode = source == "dashboard_test"

    logger.info(
        "[QUERY] start agent=%s conv=%s visitor=%s stream=%s source=%s test=%s text=%r",
        request.agent_id,
        request.conversation_id or "new",
        request.visitor_id or "anonymous",
        request.stream,
        source,
        is_test_mode,
        request.text[:100] if len(request.text) > 100 else request.text,
    )

    agent_config = await load_agent_config(request.agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")

    if agent_config.get("workspace_id") != auth.workspace_id:
        logger.warning(
            "Ownership violation: agent=%s workspace=%s auth_workspace=%s",
            request.agent_id,
            agent_config.get("workspace_id"),
            auth.workspace_id,
        )
        raise HTTPException(status_code=403, detail="Agent does not belong to your workspace")

    # Chat should see newly attached/edited KB content immediately.
    kb_documents = await load_kb_documents(request.agent_id, force_refresh=True)
    logger.info(
        "[QUERY] kb_loaded agent=%s docs=%d total_chars=%d",
        request.agent_id,
        len(kb_documents),
        sum(len(doc) for doc in kb_documents),
    )

    # For test mode, use in-memory conversation tracking only
    if is_test_mode:
        conversation_id = request.conversation_id or str(uuid4())
        conversation_history = []

        # Load existing history if continuing a test conversation
        if request.conversation_id:
            # In test mode, we don't persist, so history is empty after page refresh
            # But we can support multi-turn within the same session
            pass

        logger.info(
            "[QUERY] test_mode_conv id=%s history_turns=%d",
            conversation_id,
            len(conversation_history),
        )
    else:
        # Full persistence mode
        conversation_id, conversation_history = await _get_or_create_conversation(
            agent_id=request.agent_id,
            conversation_id=request.conversation_id,
            auth=auth,
            visitor_id=request.visitor_id,
        )
        logger.info(
            "[QUERY] conversation_ready id=%s history_turns=%d existing=%s",
            conversation_id,
            len(conversation_history),
            bool(request.conversation_id),
        )

    visitor_history = await _load_visitor_history(
        visitor_id=request.visitor_id,
        agent_id=request.agent_id,
        current_conversation_id=conversation_id,
    )

    if _is_obviously_off_topic(request.text):
        refusal_text = _build_refusal_text(agent_config)

        if not is_test_mode:
            await _save_message(conversation_id, "user", request.text)
            await _save_message(conversation_id, "assistant", refusal_text)
            await _mark_conversation_completed(conversation_id)

        logger.info("[QUERY] off_topic_refusal agent=%s conv=%s", request.agent_id, conversation_id)
        return TextQueryResponse(
            text=refusal_text,
            conversation_id=conversation_id,
            agent_id=request.agent_id,
        )

    payload = QueryPayload(
        agent_id=request.agent_id,
        text=request.text,
        channel=Channel.TEXT_API,
        agent_config=agent_config,
        kb_documents=kb_documents,
        conversation_history=conversation_history,
        caller_history=visitor_history,
        conversation_id=conversation_id,
        visitor_id=request.visitor_id,
    )
    logger.info(
        "[QUERY] payload_ready provider=%s model=%s kb=%d history=%d visitor_history=%d",
        agent_config.get("llm_provider", "groq"),
        agent_config.get("llm_model", "default"),
        len(kb_documents),
        len(conversation_history),
        len(visitor_history),
    )

    if not is_test_mode:
        await _save_message(conversation_id, "user", request.text)

    if request.stream:
        return StreamingResponse(
            _sse_generator(payload, conversation_id, is_test_mode, source),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming mode
    full_text = await stream_llm_full(payload)

    if not is_test_mode:
        await _save_message(conversation_id, "assistant", full_text)
        await _mark_conversation_completed(conversation_id)

        # Fire analytics task for real usage
        save_text_conversation.delay(
            agent_id=request.agent_id,
            conversation_id=conversation_id,
            source=source,
        )

    return TextQueryResponse(
        text=full_text,
        conversation_id=conversation_id,
        agent_id=request.agent_id,
    )


@router.post(
    "/query/tts-preview",
    summary="Generate TTS preview audio for a selected voice",
    response_description="WAV audio bytes synthesized by Azure TTS",
    responses={
        200: {
            "description": "Audio preview generated",
            "content": {"audio/wav": {}},
        },
        401: {
            "description": "No credentials provided",
            "content": {"application/json": {"example": {"detail": "Authentication required."}}},
        },
        403: {
            "description": "Agent does not belong to your workspace",
            "content": {
                "application/json": {"example": {"detail": "Agent does not belong to your workspace"}}
            },
        },
        404: {
            "description": "Agent not found",
            "content": {"application/json": {"example": {"detail": "Agent not found: abc-123"}}},
        },
        502: {
            "description": "TTS synthesis failed",
            "content": {"application/json": {"example": {"detail": "TTS preview synthesis failed"}}},
        },
    },
)
async def tts_preview(
    request: TTSPreviewRequest,
    auth: AuthContext = Depends(get_auth_context),
):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Preview text is required")

    agent_config = await load_agent_config(request.agent_id)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent not found: {request.agent_id}")

    if agent_config.get("workspace_id") != auth.workspace_id:
        raise HTTPException(status_code=403, detail="Agent does not belong to your workspace")

    voice = request.voice or agent_config.get("tts_voice")
    audio_bytes = await synthesize_preview_wav(text=text, voice=voice)
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="TTS preview synthesis failed")

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Cache-Control": "no-store"},
    )


async def _get_or_create_conversation(
    agent_id: str,
    conversation_id: str | None,
    auth: AuthContext,
    visitor_id: str | None,
) -> tuple[str, list[dict[str, str]]]:
    supabase = get_supabase()

    if conversation_id:
        conv = (
            await supabase.table("conversations")
            .select("id, agent_id")
            .eq("id", conversation_id)
            .maybe_single()
            .execute()
        )
        if not conv.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if conv.data.get("agent_id") != agent_id:
            raise HTTPException(status_code=403, detail="Conversation does not belong to this agent")

        result = (
            await supabase.table("messages")
            .select("role, content")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        history = [
            {"role": row["role"], "content": row["content"]}
            for row in (result.data or [])
            if row.get("role") in ("user", "assistant") and row.get("content")
        ]
        logger.info("[QUERY] conversation_loaded id=%s turns=%d", conversation_id, len(history))
        return conversation_id, history

    conv_payload: dict = {
        "agent_id": agent_id,
        "channel": "text_api",
        "status": "active",
        "started_at": "now()",
        "session_id": str(uuid4()),
    }
    if visitor_id:
        conv_payload["visitor_id"] = visitor_id

    result = await supabase.table("conversations").insert(conv_payload).execute()
    new_id = result.data[0]["id"] if result.data else None
    if not new_id:
        raise HTTPException(status_code=500, detail="Failed to create conversation")

    logger.info("[QUERY] conversation_created id=%s agent=%s auth=%s", new_id, agent_id, auth.auth_type)
    return new_id, []


async def _load_visitor_history(
    visitor_id: str | None,
    agent_id: str,
    current_conversation_id: str,
) -> list[dict]:
    if not visitor_id:
        return []

    try:
        supabase = get_supabase()
        result = (
            await supabase.table("conversations")
            .select("started_at, summary")
            .eq("visitor_id", visitor_id)
            .eq("agent_id", agent_id)
            .eq("status", "completed")
            .neq("id", current_conversation_id)
            .not_.is_("summary", "null")
            .order("started_at", desc=True)
            .limit(5)
            .execute()
        )
        history = result.data or []
        logger.info("[QUERY] visitor_history_loaded visitor=%s count=%d", visitor_id, len(history))
        return history
    except Exception as exc:
        logger.warning("Failed to load visitor history for %s: %s", visitor_id, exc)
        return []


async def _save_message(conversation_id: str, role: str, content: str) -> None:
    try:
        supabase = get_supabase()
        await supabase.table("messages").insert(
            {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
            }
        ).execute()
    except Exception as exc:
        logger.error("Failed to save message conv=%s role=%s: %s", conversation_id, role, exc)


async def _mark_conversation_completed(conversation_id: str) -> None:
    try:
        supabase = get_supabase()
        await (
            supabase.table("conversations")
            .update({
                "status": "completed",
                "ended_at": "now()",
            })
            .eq("id", conversation_id)
            .execute()
        )
    except Exception as exc:
        logger.warning("Failed to mark conversation completed conv=%s: %s", conversation_id, exc)


def _is_obviously_off_topic(text: str) -> bool:
    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        return False

    if any(pattern in normalized for pattern in _OFF_TOPIC_PATTERNS):
        business_keywords = (
            "appointment",
            "booking",
            "book",
            "call",
            "support",
            "help",
            "price",
            "pricing",
            "hours",
            "timing",
            "service",
            "agent",
            "order",
            "refund",
            "shipping",
            "delivery",
            "status",
            "company",
            "business",
        )
        return not any(keyword in normalized for keyword in business_keywords)

    return False


def _build_refusal_text(agent_config: dict) -> str:
    preferred_language = str(agent_config.get("stt_language") or agent_config.get("stt_locale") or "").strip()
    if preferred_language.lower() == "hi-in":
        return "माफ़ कीजिए, मैं केवल इस व्यवसाय से जुड़ी मदद कर सकता हूं। कृपया अपना सवाल काम से संबंधित रखें।"
    if preferred_language.lower() == "mr-in":
        return "माफ करा, मी फक्त या व्यवसायाशी संबंधित मदत करू शकतो. कृपया तुमचा प्रश्न कामाशी संबंधित ठेवा."

    return "Sorry, I can only help with questions related to this business. Please keep your question work-related."


async def _sse_generator(
    payload: QueryPayload,
    conversation_id: str,
    is_test_mode: bool = False,
    source: str = "api",
) -> AsyncGenerator[str, None]:
    """Generate SSE stream. Optionally skip persistence for test mode."""
    response_parts: list[str] = []
    first_chunk = True
    chunk_count = 0

    try:
        async for sentence in stream_llm_sentences(payload):
            response_parts.append(sentence)
            chunk_count += 1

            if first_chunk:
                data = json.dumps({"delta": sentence, "conversation_id": conversation_id})
                first_chunk = False
            else:
                data = json.dumps({"delta": sentence})

            yield f"data: {data}\n\n"
            # Give the event loop a chance to flush the chunk to the client.
            # This does not slow generation; it only avoids buffering multiple
            # deltas into one network write under tight async loops.
            await asyncio.sleep(0)

        yield "data: [DONE]\n\n"

        full = " ".join(response_parts).strip()
        logger.info("[QUERY] response_complete conv=%s chunks=%d total_chars=%d test=%s", conversation_id, chunk_count, len(full), is_test_mode)

        if full and not is_test_mode:
            # Save message and trigger analytics for real usage
            await _save_message(conversation_id, "assistant", full)
            await _mark_conversation_completed(conversation_id)

            # Fire Celery task for analytics (summary, outcome, sentiment)
            save_text_conversation.delay(
                agent_id=payload.agent_id,
                conversation_id=conversation_id,
                source=source,
            )

    except Exception as exc:
        logger.error("SSE error conv=%s: %s", conversation_id, exc, exc_info=True)
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
