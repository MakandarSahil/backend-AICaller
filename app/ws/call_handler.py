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
from app.pipeline.llm import stream_llm_sentences, get_groq_client
from app.pipeline.session import (
    create_session,
    delete_session,
    get_session,
    load_caller_history,
    populate_session,
    preload_caller_data,
)
from app.pipeline.stt import AzureSTT
from app.pipeline.tts import synthesize_sentence
from app.tasks.conversation import save_conversation
from app.utils.audio import calculate_volume, mulaw_to_pcm16

settings = get_settings()
logger = logging.getLogger(__name__)

_VOICE_ACTIVITY_THRESHOLD = 500.0
_PARTIAL_COMMIT_SILENCE_CHUNKS = 10  # ~200ms — commits on sustained silence
_PARTIAL_COMMIT_MAX_PACKETS = 200   # ~4 seconds — force commit if partial stuck (noise)
_POST_CLEAR_STT_SUPPRESS_MS = 200


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

    # Run database writes in the background to prevent blocking media reception
    async def _background_db_setup():
        # 1. Resolve caller
        workspace_id = session.agent_config.get("workspace_id", "")
        if workspace_id and caller_phone:
            from app.pipeline.session import resolve_or_create_caller
            session.caller_id = await resolve_or_create_caller(workspace_id, caller_phone)

        # 2. Preload caller history into session (avoids Supabase query on critical LLM path)
        if session.caller_id:
            await preload_caller_data(session)

        # 3. Insert active conversation row NOW so the dashboard
        # can display "call in progress".
        session.conversation_id = await _create_active_conversation(session)

    asyncio.create_task(_background_db_setup())

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

    # Trigger the initial greeting — the agent speaks first so the caller
    # doesn't hear silence. Runs as a background task; barge-in works
    # naturally if the caller speaks over the greeting.
    asyncio.create_task(_trigger_greeting(websocket, session))

    return call_sid


async def _handle_media(websocket: WebSocket, msg: dict, session: SessionState) -> None:
    """
    Process Twilio 'media' event — one 20ms mulaw audio chunk.

    Two modes:
    - is_speaking=False: decode + push to Azure STT
    - is_speaking=True:  buffer audio for barge-in recovery, detect interruptions
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

    # Always convert — needed for both STT push and barge-in buffer
    pcm16 = mulaw_to_pcm16(mulaw_bytes)

    if not session.is_speaking:
        if session.latest_media_timestamp < session.stt_resume_after_ts:
            session.stt_partial_text = ""
            session.stt_silence_chunks = 0
            session.stt_partial_pending_since_packet = 0
            return

        volume = calculate_volume(mulaw_bytes)

        if volume > _VOICE_ACTIVITY_THRESHOLD:
            session.stt_silence_chunks = 0
        else:
            session.stt_silence_chunks += 1

        if session.stt_recognizer:
            session.stt_recognizer.push(pcm16)

        # Track when partial first appeared (for time-based force commit)
        if session.stt_partial_text and session.stt_partial_pending_since_packet == 0:
            session.stt_partial_pending_since_packet = session.media_packet_count

        # Azure can delay final transcripts indefinitely in some PSTN paths.
        # Two fallbacks to detect end-of-utterance:
        #
        # 1. Silence-based: user stopped speaking → sustained quiet audio
        # 2. Time-based:  partial stuck too long → force commit (noisy environment)
        if session.stt_partial_text and not session.stopped:

            packets_since_partial = (
                session.media_packet_count - session.stt_partial_pending_since_packet
                if session.stt_partial_pending_since_packet > 0
                else 0
            )

            should_commit = (
                session.stt_silence_chunks >= _PARTIAL_COMMIT_SILENCE_CHUNKS
                or packets_since_partial >= _PARTIAL_COMMIT_MAX_PACKETS
            )

            if should_commit:
                partial_text = session.stt_partial_text
                reason = "silence" if session.stt_silence_chunks >= _PARTIAL_COMMIT_SILENCE_CHUNKS else "timeout"
                session.stt_partial_text = ""
                session.stt_silence_chunks = 0
                session.stt_partial_pending_since_packet = 0
                logger.info(
                    "STT partial committed (%s) [%s]: %s",
                    reason,
                    session.call_sid,
                    partial_text,
                )
                asyncio.create_task(_handle_final_transcript(websocket, session, partial_text))
    else:
        session.speaking_chunks += 1

        # Ring buffer of recent audio while agent speaks.
        # When barge-in fires, this buffer is flushed to STT so no words are lost.
        session.stt_audio_buffer.append(pcm16)
        if len(session.stt_audio_buffer) > 10:
            session.stt_audio_buffer.pop(0)

        volume = calculate_volume(mulaw_bytes)
        if volume > settings.barge_in_threshold:
            session.barge_in_loud_samples += 1
        else:
            session.barge_in_loud_samples = 0

        # Startup guard: skip barge-in for first 25 chunks (500ms).
        # Phone-line echo cancellers need ~300-500ms to stabilise.
        # Without this guard the initial TTS echo burst falsely
        # triggers barge-in, cutting off the first sentence and
        # creating an interrupt loop.
        if (
            session.speaking_chunks >= 25
            and session.barge_in_loud_samples >= settings.barge_in_min_chunks
        ):
            logger.info(
                "Barge-in: call_sid=%s speaking_chunks=%d volume=%.1f",
                session.call_sid,
                session.speaking_chunks,
                volume,
            )
            await _cancel_tts(websocket, session)


def _handle_mark(msg: dict, session: SessionState) -> None:
    """
    Process Twilio 'mark' event — audio playback confirmation.
    When confirmed mark == last sent mark, all queued audio has finished playing.
    """
    mark_name = msg.get("mark", {}).get("name", "")
    session.latest_mark_confirmed = mark_name
    logger.debug("Mark confirmed: %s (sent=%s)", mark_name, session.latest_mark_sent)

    # Only flip is_speaking=False when the ENTIRE TTS batch is done.
    # tts_active=True means the TTS worker is still alive and more
    # sentences may follow — don't flip between sentences or the
    # worker will skip the remaining sentences.
    if (
        session.is_speaking
        and not session.tts_active
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
    if session.stopped or session.shutdown_in_progress:
        return  # Already running — do not double-execute
    session.shutdown_in_progress = True

    logger.info("Call ending: call_sid=%s turns=%d", call_sid, len(session.conversation_history))

    # Stop STT
    if session.stt_recognizer:
        session.stt_recognizer.stop()
        session.stt_recognizer = None

    # Give any late Azure final callback a chance to append the last turn.
    await asyncio.sleep(0)

    if session.stt_partial_text:
        partial_text = session.stt_partial_text.strip()
        session.stt_partial_text = ""
        session.stt_partial_pending_since_packet = 0
        if partial_text:
            _append_transcript_if_new(session, partial_text)

    await asyncio.sleep(0.05)

    # Cancel any in-flight TTS
    if session.current_tts_cancel:
        try:
            session.current_tts_cancel()
        except Exception:
            pass
        session.current_tts_cancel = None

    # Mark as not speaking so any in-flight LLM/TTS loops detect the hang-up
    session.is_speaking = False
    session.tts_active = False

    await _mark_conversation_completed(session)

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

    session.stopped = True
    delete_session(call_sid)


# ── Sentence boundary helpers ─────────────────────────────────────────────────

_SENTENCE_ENDINGS = {".", "!", "?", ",", ":", ";"}
_MIN_SENTENCE_LEN = 2
_MAX_BUFFER_CHARS = 50


def _find_sentence_boundary(text: str) -> int:
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDINGS:
            return i
    return -1


# ── Initial greeting ──────────────────────────────────────────────────────────

async def _trigger_greeting(websocket: WebSocket, session: SessionState) -> None:
    """
    Generate the agent's initial greeting when a call connects.
    Runs the LLM → TTS pipeline without waiting for user input.
    The system prompt instructs the LLM to greet the caller.
    Supports barge-in (user interrupts greeting).
    """
    if session.stopped or session.is_speaking:
        return

    gen = session.response_generation + 1
    session.response_generation = gen
    session.is_speaking = True
    session.speaking_chunks = 0
    session.barge_in_loud_samples = 0

    system_prompt = session.cached_system_prompt or ""
    greeting_messages = [
        {"role": "system", "content": system_prompt},
    ]

    spoken_response_parts: list[str] = []
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _greeting_tts_worker() -> None:
        while True:
            sentence = await tts_queue.get()
            if sentence is None:
                break
            if not session.is_speaking or session.response_generation != gen:
                break
            was_cancelled = await _stream_sentence_to_twilio(
                websocket, session, sentence
            )
            if was_cancelled or not session.is_speaking or session.response_generation != gen:
                break
            spoken_response_parts.append(sentence)

    session.tts_active = True
    worker = asyncio.create_task(_greeting_tts_worker())

    try:
        client = get_groq_client()
        model = session.agent_config.get("llm_model") or settings.groq_model
        stream = await client.chat.completions.create(
            model=model,
            messages=greeting_messages,
            max_tokens=settings.groq_max_tokens,
            temperature=settings.groq_temperature,
            stream=True,
        )

        buffer = ""
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue
            if not session.is_speaking or session.response_generation != gen:
                logger.info("Greeting aborted (barge-in) [%s]", session.call_sid)
                break

            buffer += delta

            while True:
                idx = _find_sentence_boundary(buffer)
                if idx == -1:
                    break
                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1 :]
                if len(sentence) >= _MIN_SENTENCE_LEN:
                    await tts_queue.put(sentence)

            if len(buffer) >= _MAX_BUFFER_CHARS:
                flush = buffer.strip()
                buffer = ""
                if len(flush) >= _MIN_SENTENCE_LEN:
                    await tts_queue.put(flush)

        final = buffer.strip()
        if len(final) >= _MIN_SENTENCE_LEN:
            await tts_queue.put(final)

    except Exception as exc:
        logger.error("Greeting error [%s]: %s", session.call_sid, exc, exc_info=True)
    finally:
        await tts_queue.put(None)
        await worker

        if session.response_generation == gen:
            session.tts_active = False
            session.current_tts_cancel = None

        if session.response_generation == gen:
            if session.latest_mark_sent is None:
                session.is_speaking = False
            elif session.latest_mark_sent == session.latest_mark_confirmed:
                session.is_speaking = False

        if session.response_generation == gen and session.is_speaking:
            _guard_speaking_chunks = session.speaking_chunks

            async def _greeting_mark_timeout() -> None:
                await asyncio.sleep(2.0)
                if not session.is_speaking or session.stopped:
                    return
                if session.speaking_chunks < _guard_speaking_chunks:
                    return
                if session.stt_recognizer and session.stt_audio_buffer:
                    for chunk in session.stt_audio_buffer:
                        session.stt_recognizer.push(chunk)
                    session.stt_audio_buffer.clear()
                session.is_speaking = False

            asyncio.create_task(_greeting_mark_timeout())

        if spoken_response_parts:
            response_text = " ".join(part.strip() for part in spoken_response_parts if part.strip()).strip()
            session.add_assistant_turn(response_text)

        logger.info("Greeting complete [%s]: %s", session.call_sid, spoken_response_parts)


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

    normalized_text = " ".join(text.lower().split())
    if not normalized_text:
        return

    # Guard: if already speaking (very fast back-to-back utterances), ignore.
    # Must come BEFORE _append_transcript_if_new to prevent spurious duplicate
    # turns from being added to conversation_history while is_speaking is True
    # (e.g. Azure STT on_final arriving for the same utterance that was already
    # committed via the silence fallback).
    if session.is_speaking:
        logger.debug("Ignoring transcript while speaking [%s]: %s", session.call_sid, text)
        return

    if (
        session.pending_user_transcript_norm
        and (
            normalized_text == session.pending_user_transcript_norm
            or normalized_text.startswith(session.pending_user_transcript_norm)
            or session.pending_user_transcript_norm.startswith(normalized_text)
        )
    ):
        logger.debug(
            "Duplicate transcript (overlap) [%s]: pending=%r new=%r",
            session.call_sid,
            session.pending_user_transcript_norm,
            normalized_text,
        )
        return

    if not _append_transcript_if_new(session, text):
        return

    session.pending_user_transcript_norm = normalized_text

    # If shutdown is in progress, record the transcript (for persistence) but
    # skip response generation.
    if session.shutdown_in_progress:
        logger.debug(
            "Transcript recorded during shutdown — skipping response generation [%s]",
            session.call_sid,
        )
        if session.pending_user_transcript_norm == normalized_text:
            session.pending_user_transcript_norm = ""
        return

    logger.info("Transcript [%s]: %s", session.call_sid, text)

    session.response_generation += 1
    response_generation = session.response_generation
    session.is_speaking = True
    session.stt_partial_text = ""
    session.stt_silence_chunks = 0
    session.stt_partial_pending_since_packet = 0
    session.stt_audio_buffer.clear()
    session.barge_in_loud_samples = 0
    session.speaking_chunks = 0

    # Use preloaded caller history if available (loaded in background task),
    # otherwise fall back to inline load.
    caller_history = session.caller_history or []
    if not caller_history and session.caller_id:
        caller_history = await load_caller_history(session.caller_id)

    payload = QueryPayload(
        agent_id=session.agent_id,
        text=text,
        channel=Channel.TWILIO,
        agent_config=session.agent_config,
        kb_documents=session.kb_documents,
        conversation_history=session.conversation_history[:-1],
        caller_history=caller_history,
        cached_system_prompt=session.cached_system_prompt,
        groq_messages=session.groq_messages,
        call_sid=session.call_sid,
        conversation_id=session.conversation_id,
        caller_id=session.caller_id,
    )

    spoken_response_parts: list[str] = []

    # TTS queue: LLM streams sentences into the queue, the TTS worker
    # consumes them in order and streams audio to Twilio in parallel.
    # This decouples LLM generation from TTS playback — the LLM is no
    # longer blocked while TTS synthesises and plays the current sentence.
    tts_queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _tts_worker() -> None:
        """Consume sentences from the queue and stream to Twilio in order."""
        while True:
            sentence = await tts_queue.get()
            if sentence is None:
                break
            if not session.is_speaking or session.response_generation != response_generation:
                break
            was_cancelled = await _stream_sentence_to_twilio(
                websocket, session, sentence
            )
            if was_cancelled or not session.is_speaking or session.response_generation != response_generation:
                break
            spoken_response_parts.append(sentence)

    session.tts_active = True
    worker = asyncio.create_task(_tts_worker())

    try:
        async for sentence in stream_llm_sentences(payload):
            if not session.is_speaking or session.response_generation != response_generation:
                logger.info("LLM stream aborted (barge-in) [%s]", session.call_sid)
                break

            await tts_queue.put(sentence)

    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", session.call_sid, exc, exc_info=True)
    finally:
        # Signal the TTS worker to stop (None sentinel).
        # If the worker already stopped (barge-in), this is a no-op.
        await tts_queue.put(None)
        await worker  # Wait for worker to finish its current TTS

        if session.response_generation == response_generation:
            session.tts_active = False
            session.current_tts_cancel = None

        # Never set is_speaking=False while TTS audio might still be in
        # the Twilio pipe — the TTS echo would loop back through the phone
        # line and trigger a false STT final. Only the mark handler flips
        # the flag when all audio has confirmed as played.
        #
        # Exception: if NO mark was ever sent (TTS cancelled before the
        # first sentence's mark), we can reset safely — no audio played.
        if session.response_generation == response_generation:
            if session.latest_mark_sent is None:
                session.is_speaking = False
            elif session.latest_mark_sent == session.latest_mark_confirmed:
                session.is_speaking = False

        if session.response_generation == response_generation and session.is_speaking:

            # Capture speaking_chunks at timeout creation. A new turn resets
            # speaking_chunks to 0, so detecting a LOWER value means the old
            # timeout is stale and must not cancel the active turn.
            _guard_speaking_chunks = session.speaking_chunks

            async def _mark_timeout() -> None:
                await asyncio.sleep(2.0)
                if not session.is_speaking or session.stopped:
                    return
                if session.speaking_chunks < _guard_speaking_chunks:
                    return  # Reset detected — new turn in progress
                logger.warning(
                    "Mark timeout — force is_speaking=False [%s]",
                    session.call_sid,
                )
                # Flush any buffered audio to STT before re-enabling input
                if session.stt_recognizer and session.stt_audio_buffer:
                    for chunk in session.stt_audio_buffer:
                        session.stt_recognizer.push(chunk)
                    session.stt_audio_buffer.clear()
                session.is_speaking = False

            asyncio.create_task(_mark_timeout())

        # Persist the Groq messages list for next turn's fast path
        if session.response_generation == response_generation:
            session.groq_messages = payload.groq_messages

        if spoken_response_parts:
            response_text = " ".join(part.strip() for part in spoken_response_parts if part.strip()).strip()
            session.add_assistant_turn(response_text)
            # Append assistant response to the persisted message list
            # so the next turn includes full conversation context
            if session.groq_messages is not None:
                session.groq_messages.append({"role": "assistant", "content": response_text})

        if session.pending_user_transcript_norm == normalized_text:
            session.pending_user_transcript_norm = ""


def _append_transcript_if_new(session: SessionState, text: str) -> bool:
    cleaned_text = text.strip()
    if not cleaned_text:
        return False

    if session.conversation_history:
        last_turn = session.conversation_history[-1]
        if last_turn.get("role") == "user":
            last_text = " ".join(last_turn.get("content", "").strip().lower().split())
            curr_text = " ".join(cleaned_text.lower().split())
            if last_text and curr_text and last_text == curr_text:
                logger.debug("Duplicate transcript ignored [%s]: %s", session.call_sid, cleaned_text)
                return False

    session.add_user_turn(cleaned_text)
    return True


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
    chunks_sent = 0

    def cancel() -> None:
        nonlocal cancelled
        cancelled = True
        cancel_event.set()

    session.current_tts_cancel = cancel

    try:
        logger.info("TTS streaming started [%s]: sentence=%r voice=%s", session.call_sid, sentence[:60], voice)
        async for mulaw_chunk in synthesize_sentence(sentence, voice):
            if cancel_event.is_set():
                logger.debug("TTS cancelled mid-stream [%s]", session.call_sid)
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
            chunks_sent += 1

        logger.info("TTS streaming complete [%s]: chunks_sent=%d cancelled=%s", session.call_sid, chunks_sent, cancelled)

        if not cancelled:
            mark_name = session.next_mark_name()
            session.latest_mark_sent = mark_name
            await _ws_send(websocket, {
                "event": "mark",
                "streamSid": session.stream_sid,
                "mark": {"name": mark_name},
            })
            logger.debug("TTS mark sent [%s]: %s", session.call_sid, mark_name)

    except Exception as exc:
        logger.error("TTS stream error [%s]: %s", session.call_sid, exc, exc_info=True)
    finally:
        session.current_tts_cancel = None

    return cancelled


async def _cancel_tts(websocket: WebSocket, session: SessionState) -> None:
    """Cancel active TTS + flush buffered audio + tell Twilio to discard buffered audio."""
    session.response_generation += 1

    if session.current_tts_cancel:
        try:
            session.current_tts_cancel()
        except Exception:
            pass
        session.current_tts_cancel = None

    # Flush buffered audio to STT before re-enabling push.
    # This preserves the first ~200ms of the user's interruption.
    if session.stt_recognizer and session.stt_audio_buffer:
        for chunk in session.stt_audio_buffer:
            session.stt_recognizer.push(chunk)
        logger.debug(
            "Flushed %d buffered chunks to STT [%s]",
            len(session.stt_audio_buffer),
            session.call_sid,
        )
        session.stt_audio_buffer.clear()

    session.stt_partial_text = ""
    session.stt_silence_chunks = 0
    session.stt_partial_pending_since_packet = 0
    session.stt_resume_after_ts = session.latest_media_timestamp + _POST_CLEAR_STT_SUPPRESS_MS
    session.is_speaking = False
    session.tts_active = False
    session.barge_in_loud_samples = 0
    session.speaking_chunks = 0
    session.latest_mark_sent = None
    session.latest_mark_confirmed = None

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


async def _mark_conversation_completed(session: SessionState) -> None:
    if not session.conversation_id:
        return

    try:
        supabase = get_supabase()
        await (
            supabase.table("conversations")
            .update(
                {
                    "status": "completed",
                    "ended_at": "now()",
                    "message_count": len(session.conversation_history),
                }
            )
            .eq("id", session.conversation_id)
            .execute()
        )

        try:
            await (
                supabase.table("conversations")
                .update({"outcome": "hung_up"})
                .eq("id", session.conversation_id)
                .execute()
            )
        except Exception as exc:
            logger.warning(
                "Could not set fallback outcome: conv_id=%s err=%s",
                session.conversation_id,
                exc,
            )

        logger.info(
            "Conversation marked completed: conv_id=%s turns=%d",
            session.conversation_id,
            len(session.conversation_history),
        )
    except Exception as exc:
        logger.error(
            "Failed to mark conversation completed: conv_id=%s err=%s",
            session.conversation_id,
            exc,
        )


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
    for key in ("From", "from", "caller", "callerNumber"):
        value = custom.get(key)
        if value:
            value_str = str(value)
            # Twilio can send values like "client:alice" for client calls.
            if value_str.startswith("client:"):
                return ""
            return value_str
    # Standard Twilio call metadata fields
    for key in ("from", "From", "caller", "callerNumber"):
        if start.get(key):
            value_str = str(start[key])
            if value_str.startswith("client:"):
                return ""
            return value_str
    return ""


def _resolve_stt_language(agent_config: dict[str, Any]) -> str:
    """Resolve the Azure speech recognition locale from agent config."""
    for key in ("stt_language", "stt_locale", "stt_model"):
        value = str(agent_config.get(key) or "").strip()
        if value and value.lower() != "default":
            return value

    return settings.stt_language
