import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from app.clients.supabase import get_supabase
from app.config import get_settings
from app.models.query import Channel, QueryPayload
from app.models.session import SessionState
from app.pipeline.llm import stream_llm_sentences
from app.pipeline.session import (
    create_session,
    delete_session,
    get_session,
    load_caller_history,
    populate_session,
)
from app.pipeline.stt import AzureSTT
from app.pipeline.tts import synthesize_sentence
from app.tasks.conversation import save_conversation
from app.utils.audio import calculate_volume, mulaw_to_pcm16

settings = get_settings()
logger = logging.getLogger(__name__)

_VOICE_ACTIVITY_THRESHOLD = 8.0
_PARTIAL_COMMIT_SILENCE_CHUNKS = 25  # ~500ms at 20ms Twilio media frames


async def handle_call(websocket: WebSocket, agent_id: str | None) -> None:
    """
    Main WebSocket handler for a Twilio Media Stream call.

    Each call runs as an independent coroutine — concurrent calls are fully
    isolated by their call_sid. The sessions Dict is the only shared state,
    and each call only touches its own entry.

    Lifecycle:
        accept WS → start event → media stream → (LLM/TTS per transcript) → stop
    """
    await websocket.accept()
    call_sid: str | None = None
    resolved_agent_id = agent_id

    try:
        async for raw in websocket.iter_text():
            msg = _parse(raw)
            if not msg:
                continue

            event = msg.get("event")

            if event == "start":
                start = msg.get("start", {})
                custom = start.get("customParameters") or {}
                resolved_agent_id = resolved_agent_id or custom.get("agent_id")

                if not resolved_agent_id:
                    logger.warning("Call start missing agent_id; closing websocket")
                    await websocket.close(code=1008)
                    break

                call_sid = await _handle_start(websocket, msg, resolved_agent_id)

            elif event == "media":
                if call_sid:
                    session = get_session(call_sid)
                    if session:
                        await _handle_media(websocket, msg, session)

            elif event == "mark":
                if call_sid:
                    session = get_session(call_sid)
                    if session:
                        _handle_mark(msg, session)

            elif event in ("stop", "close", "disconnect"):
                if call_sid:
                    await _handle_stop(call_sid)
                break

    except WebSocketDisconnect:
        logger.info("WS disconnected: call_sid=%s", call_sid)
    except Exception as exc:
        logger.error("call_handler error: call_sid=%s %s", call_sid, exc, exc_info=True)
    finally:
        # BUG FIX: Only call _handle_stop if not already stopped.
        # Without the stopped guard, _handle_stop runs twice on clean disconnect:
        # once from the event branch + break, once from this finally block.
        # The second run was harmless but could queue a duplicate Celery task.
        if call_sid:
            await _handle_stop(call_sid)


# ── Event handlers ─────────────────────────────────────────────────────────────

async def _handle_start(websocket: WebSocket, msg: dict, agent_id: str) -> str:
    """
    Process Twilio 'start' event.

    - Creates SessionState (call isolated by call_sid)
    - Loads agent config + KB docs from Redis/Supabase cache
    - Creates conversations row with status=active so dashboard sees live call
    - Starts Azure STT push stream
    """
    start = msg.get("start", {})
    call_sid = msg.get("callSid") or start.get("callSid", "unknown")
    stream_sid = start.get("streamSid", "")
    caller_phone = _extract_caller_phone(start)

    logger.info("Call started: call_sid=%s agent_id=%s caller=%s", call_sid, agent_id, caller_phone)

    session = create_session(call_sid, agent_id)
    session.stream_sid = stream_sid
    session.websocket = websocket

    # Load all context — runs agent config + KB fetches in parallel
    await populate_session(session, caller_phone)

    if not session.agent_config:
        logger.error("Agent not found or inactive: %s — rejecting call", agent_id)
        await websocket.close()
        return call_sid

    # BUG FIX + GAP FIX: Insert active conversation row NOW so the dashboard
    # can display "call in progress". Previously this was only done at call end
    # (INSERT completed) which meant the dashboard had no visibility during a call.
    session.conversation_id = await _create_active_conversation(session)

    # Wire up STT callbacks — each call gets its own AzureSTT instance
    loop = asyncio.get_running_loop()

    def on_partial(text: str) -> None:
        session.stt_partial_text = text.strip()
        logger.debug("STT partial [%s]: %s", call_sid, text)

    def on_final(text: str) -> None:
        if session.stopped:
            logger.debug("STT on_final ignored — session already stopped [%s]", session.call_sid)
            return

        # Azure STT callbacks fire on a background thread.
        # Schedule the async pipeline back onto the event loop safely.
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(
                _handle_final_transcript(websocket, session, text)
            )
        )

    def on_error(err: Exception) -> None:
        logger.error("STT error [%s]: %s", call_sid, err)

    session.stt_recognizer = AzureSTT(
        on_partial=on_partial,
        on_final=on_final,
        on_error=on_error,
        language=_resolve_stt_language(session.agent_config),
    )
    session.stt_recognizer.start()

    return call_sid


async def _handle_media(websocket: WebSocket, msg: dict, session: SessionState) -> None:
    """
    Process Twilio 'media' event — one 20ms mulaw audio chunk.

    Two modes:
    - is_speaking=False: decode + push to Azure STT
    - is_speaking=True:  check barge-in threshold, cancel if triggered
    """
    media = msg.get("media", {})
    payload_b64 = media.get("payload", "")
    if not payload_b64:
        return

    session.latest_media_timestamp = int(media.get("timestamp", 0))
    session.media_packet_count += 1
    mulaw_bytes = base64.b64decode(payload_b64)

    if session.media_packet_count % 50 == 0:
        logger.debug(
            "Twilio media received: call_sid=%s packets=%d ts=%s bytes=%d speaking=%s",
            session.call_sid,
            session.media_packet_count,
            session.latest_media_timestamp,
            len(mulaw_bytes),
            session.is_speaking,
        )

    if not session.is_speaking:
        volume = calculate_volume(mulaw_bytes)

        if volume > _VOICE_ACTIVITY_THRESHOLD:
            session.stt_silence_chunks = 0
        else:
            session.stt_silence_chunks += 1

        pcm16 = mulaw_to_pcm16(mulaw_bytes)
        if session.stt_recognizer:
            session.stt_recognizer.push(pcm16)

        # Azure can delay final transcripts until stream stop in some PSTN paths.
        # Fallback: if we have a partial and then sustained silence, commit it now.
        if (
            session.stt_partial_text
            and session.stt_silence_chunks >= _PARTIAL_COMMIT_SILENCE_CHUNKS
            and not session.stopped
        ):
            partial_text = session.stt_partial_text
            session.stt_partial_text = ""
            session.stt_silence_chunks = 0
            logger.info(
                "STT partial committed on silence [%s]: %s",
                session.call_sid,
                partial_text,
            )
            asyncio.create_task(_handle_final_transcript(websocket, session, partial_text))
    else:
        volume = calculate_volume(mulaw_bytes)
        if volume > settings.barge_in_threshold:
            session.barge_in_loud_samples += 1
        else:
            session.barge_in_loud_samples = 0

        if session.barge_in_loud_samples >= settings.barge_in_min_chunks:
            logger.info("Barge-in: call_sid=%s volume=%.1f", session.call_sid, volume)
            await _cancel_tts(websocket, session)


def _handle_mark(msg: dict, session: SessionState) -> None:
    """
    Process Twilio 'mark' event — audio playback confirmation.
    When confirmed mark == last sent mark, all queued audio has finished playing.
    """
    mark_name = msg.get("mark", {}).get("name", "")
    session.latest_mark_confirmed = mark_name
    logger.debug("Mark confirmed: %s (sent=%s)", mark_name, session.latest_mark_sent)

    if (
        session.is_speaking
        and mark_name == session.latest_mark_sent
        and session.current_tts_cancel is None
    ):
        session.is_speaking = False
        logger.debug("Playback complete — is_speaking=False [%s]", session.call_sid)


async def _handle_stop(call_sid: str) -> None:
    """
    Clean up a call. Idempotent — safe to call from both event branch and finally.

    BUG FIX: session.stopped guard ensures this runs exactly once per call
    regardless of whether disconnect comes from a stop event or a WebSocketDisconnect.
    """
    session = get_session(call_sid)
    if not session:
        return  # Already cleaned up
    if session.stopped:
        return  # Already running — do not double-execute
    session.stopped = True

    logger.info("Call ending: call_sid=%s turns=%d", call_sid, len(session.conversation_history))

    # Stop STT
    if session.stt_recognizer:
        session.stt_recognizer.stop()
        session.stt_recognizer = None

    # Cancel any in-flight TTS
    if session.current_tts_cancel:
        try:
            session.current_tts_cancel()
        except Exception:
            pass
        session.current_tts_cancel = None

    # Fire Celery task — persist messages + summary + update conversation row
    try:
        save_conversation.delay(
            call_sid=session.call_sid,
            agent_id=session.agent_id,
            caller_id=session.caller_id,
            caller_phone=session.caller_phone,
            conversation_history=session.conversation_history,
            conversation_id=session.conversation_id,  # UPDATE existing row, not INSERT
        )
        logger.info("Celery task queued: call_sid=%s", call_sid)
    except Exception as exc:
        logger.error("Failed to queue Celery task for call_sid=%s: %s", call_sid, exc)

    delete_session(call_sid)


# ── LLM → TTS pipeline ────────────────────────────────────────────────────────

async def _handle_final_transcript(
    websocket: WebSocket, session: SessionState, text: str
) -> None:
    """
    Triggered by Azure STT on_final callback (runs on asyncio event loop).

    Runs the full pipeline for one user turn:
        text → build_prompt → Groq sentence stream → TTS → Twilio audio

    Concurrent call safety:
        Each call has its own session with its own is_speaking, conversation_history,
        and TTS cancel handle. There is no shared state touched here.
    """
    if session.stopped:
        logger.debug("_handle_final_transcript ignored — session stopped [%s]", session.call_sid)
        return

    text = text.strip()
    if not text:
        return

    # Ignore duplicate user transcript that can happen when a fallback partial
    # commit is followed by a late Azure final callback with the same text.
    if session.conversation_history:
        last_turn = session.conversation_history[-1]
        if last_turn.get("role") == "user":
            last_text = " ".join(last_turn.get("content", "").strip().lower().split())
            curr_text = " ".join(text.lower().split())
            if last_text and curr_text and last_text == curr_text:
                logger.debug("Duplicate transcript ignored [%s]: %s", session.call_sid, text)
                return

    # Guard: if already speaking (very fast back-to-back utterances), ignore
    if session.is_speaking:
        logger.debug("Ignoring transcript while speaking [%s]: %s", session.call_sid, text)
        return

    logger.info("Transcript [%s]: %s", session.call_sid, text)

    session.add_user_turn(text)
    session.is_speaking = True
    session.barge_in_loud_samples = 0

    # Load caller history (Redis-cached, fast)
    caller_history: list[dict] = []
    if session.caller_id:
        caller_history = await load_caller_history(session.caller_id)

    payload = QueryPayload(
        agent_id=session.agent_id,
        text=text,
        channel=Channel.TWILIO,
        agent_config=session.agent_config,
        kb_documents=session.kb_documents,
        # Pass history EXCLUDING the current user turn we just added
        # (it's already in payload.text — adding it again would duplicate it)
        conversation_history=session.conversation_history[:-1],
        caller_history=caller_history,
        call_sid=session.call_sid,
        conversation_id=session.conversation_id,
        caller_id=session.caller_id,
    )

    full_response = ""
    cancelled = False

    try:
        async for sentence in stream_llm_sentences(payload):
            if cancelled or not session.is_speaking:
                logger.info("LLM stream aborted (barge-in) [%s]", session.call_sid)
                break

            full_response += sentence + " "
            cancelled = await _stream_sentence_to_twilio(websocket, session, sentence)
            if cancelled:
                break

    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", session.call_sid, exc, exc_info=True)
    finally:
        session.is_speaking = False
        session.current_tts_cancel = None
        if full_response.strip():
            session.add_assistant_turn(full_response.strip())


async def _stream_sentence_to_twilio(
    websocket: WebSocket,
    session: SessionState,
    sentence: str,
) -> bool:
    """
    Synthesize one sentence via Azure TTS and stream mulaw chunks to Twilio.
    Returns True if cancelled mid-stream (barge-in), False on completion.
    """
    voice = session.agent_config.get("tts_voice") or settings.tts_default_voice
    cancelled = False
    cancel_event = asyncio.Event()

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True
        cancel_event.set()

    session.current_tts_cancel = cancel

    try:
        async for mulaw_chunk in synthesize_sentence(sentence, voice):
            if cancel_event.is_set():
                break

            # Check WS still open (connection might drop mid-sentence)
            if websocket.client_state.value != 1:
                logger.warning("WS closed during TTS [%s]", session.call_sid)
                cancelled = True
                break

            b64 = base64.b64encode(mulaw_chunk).decode()
            await _ws_send(websocket, {
                "event": "media",
                "streamSid": session.stream_sid,
                "media": {"payload": b64},
            })

        if not cancelled:
            mark_name = session.next_mark_name()
            session.latest_mark_sent = mark_name
            await _ws_send(websocket, {
                "event": "mark",
                "streamSid": session.stream_sid,
                "mark": {"name": mark_name},
            })

    except Exception as exc:
        logger.error("TTS stream error [%s]: %s", session.call_sid, exc, exc_info=True)
    finally:
        session.current_tts_cancel = None

    return cancelled


async def _cancel_tts(websocket: WebSocket, session: SessionState) -> None:
    """Cancel active TTS + tell Twilio to discard buffered audio."""
    if session.current_tts_cancel:
        try:
            session.current_tts_cancel()
        except Exception:
            pass
        session.current_tts_cancel = None

    session.is_speaking = False
    session.barge_in_loud_samples = 0

    await _ws_send(websocket, {
        "event": "clear",
        "streamSid": session.stream_sid,
    })
    logger.info("Barge-in: TTS cancelled + Twilio cleared [%s]", session.call_sid)


# ── Supabase helpers ──────────────────────────────────────────────────────────

async def _create_active_conversation(session: SessionState) -> str | None:
    """
    INSERT a conversations row with status=active at call start.

    This is the fix for Gap 1: the dashboard can now see live calls in real time
    by querying conversations WHERE status='active'.

    Returns the new conversations.id so the Celery task can UPDATE it on call end.
    """
    try:
        supabase = get_supabase()
        payload: dict[str, Any] = {
            "agent_id": session.agent_id,
            "session_id": session.call_sid,
            "channel": "twilio",
            "status": "active",
            "started_at": "now()",
        }
        if session.caller_id:
            payload["caller_id"] = session.caller_id

        result = await supabase.table("conversations").insert(payload).execute()
        conv_id = result.data[0]["id"] if result.data else None
        logger.info(
            "Active conversation created: conv_id=%s call_sid=%s",
            conv_id, session.call_sid,
        )
        return conv_id
    except Exception as exc:
        logger.error(
            "Failed to create active conversation for call_sid=%s: %s",
            session.call_sid, exc,
        )
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict[str, Any] | None:
    try:
        return json.loads(raw)
    except Exception:
        return None


async def _ws_send(websocket: WebSocket, obj: dict) -> None:
    try:
        await websocket.send_text(json.dumps(obj))
    except Exception as exc:
        logger.debug("ws_send failed (likely closed): %s", exc)


def _extract_caller_phone(start: dict) -> str:
    """
    Extract caller's E.164 phone number from Twilio start event.
    Twilio places it in different fields depending on account/stream config.
    """
    # customParameters — set when using <Parameter> in TwiML
    custom = start.get("customParameters") or {}
    if custom.get("From"):
        return custom["From"]
    # Standard Twilio call metadata fields
    for key in ("from", "From", "callerNumber"):
        if start.get(key):
            return start[key]
    return ""


def _resolve_stt_language(agent_config: dict[str, Any]) -> str:
    """Resolve the Azure speech recognition locale from agent config."""
    for key in ("stt_language", "stt_locale", "stt_model"):
        value = str(agent_config.get(key) or "").strip()
        if value and value.lower() != "default":
            return value

    return settings.stt_language