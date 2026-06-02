from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class SessionState:
    """
    Holds all live state for a single active Twilio call.
    One instance per call, keyed by call_sid in pipeline/session.py.

    Concurrency:
        Multiple simultaneous calls each have their own isolated SessionState.
        Dict[call_sid, SessionState] — no shared mutable state between sessions.

    Lifecycle:
        created   → WS connect, start event received
        active    → audio streaming, STT → LLM → TTS running
        stopped   → stop event or disconnect — Celery task fired, session deleted
    """

    # ── Identity ───────────────────────────────────────────────────────────
    call_sid: str                             # Twilio CallSid — primary key
    agent_id: str                             # From WS ?agent_id= query param
    stream_sid: str = ""                      # Twilio StreamSid — set on start event
    caller_phone: str = ""                    # Caller E.164 e.g. +919876543210
    caller_id: str | None = None              # callers.id resolved from Supabase

    # ── Supabase conversation tracking ────────────────────────────────────
    conversation_id: str | None = None
    # conversations.id — INSERT with status=active on call start so
    # the dashboard can show a live call indicator.
    # Celery task on call end UPDATEs this row: status=completed + summary.

    # ── Loaded config (Redis cache → Supabase fallback) ───────────────────
    agent_config: dict[str, Any] = field(default_factory=dict)
    # Full agents row. Keys used by pipeline:
    #   system_prompt, tts_voice, tts_provider, tts_model,
    #   stt_provider, stt_model, llm_provider, llm_model, workspace_id

    kb_documents: list[str] = field(default_factory=list)
    # Raw text of every KB document attached to this agent.
    # Injected into system prompt (full-context dump, v1).

    caller_history: list[dict] = field(default_factory=list)
    # Past conversation summaries for this caller.
    # Preloaded in background task after caller_id is resolved.

    cached_system_prompt: str | None = None
    # Pre-built system prompt (system + KB + caller history) cached to avoid
    # rebuilding every LLM turn. Only conversation turns change per turn.

    groq_messages: list[dict[str, str]] | None = None
    # Running Groq messages list reused across turns.
    # Built once on first turn, then only the new user/assistant messages are
    # appended each turn — eliminates full prompt rebuild latency.

    # ── In-call conversation history ───────────────────────────────────────
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    # [{role: "user"|"assistant", content: "..."}]
    # Accumulated turn by turn. Persisted to Supabase on call end via Celery.

    # ── WebSocket handle ───────────────────────────────────────────────────
    websocket: Any = field(default=None)
    # FastAPI WebSocket object. Stored so STT thread callbacks can safely
    # schedule coroutines back onto the event loop without passing ws around.

    # ── TTS / playback state ───────────────────────────────────────────────
    is_speaking: bool = False
    # True while TTS audio is actively being streamed to Twilio.
    # Incoming audio is NOT pushed to STT while this is True.

    tts_active: bool = False
    # True while the TTS worker task is alive (between the start of the first
    # sentence and the end of the last). Prevents _handle_mark from setting
    # is_speaking=False between individual sentences — only the final worker
    # completion should allow the is_speaking flip via marks.

    current_tts_cancel: Callable[[], None] | None = None
    # Cancel function for the current in-flight TTS sentence stream.
    # Set before each sentence starts, cleared when it finishes or is cancelled.

    latest_mark_sent: str | None = None
    # Last mark name sent to Twilio (e.g. "sentence_3").

    latest_mark_confirmed: str | None = None
    # Last mark name Twilio confirmed has finished playing.
    # When confirmed == sent → all queued audio has played.

    latest_media_timestamp: int = 0
    # Twilio media event timestamp (ms). Used to calculate response latency.

    media_packet_count: int = 0
    # Counts Twilio media packets received for this call.

    # ── Barge-in detection ─────────────────────────────────────────────────
    barge_in_loud_samples: int = 0
    # Count of consecutive loud mulaw chunks received while is_speaking.
    # Resets to 0 on any quiet chunk.
    # Fires barge-in when >= settings.barge_in_min_chunks (default 15 ≈ 300ms).

    # ── Azure STT handle ───────────────────────────────────────────────────
    stt_recognizer: Any | None = None
    # AzureSTT instance. Created on call start, stopped on call end.

    stt_partial_text: str = ""
    # Latest partial transcript text from Azure STT.

    stt_silence_chunks: int = 0
    # Consecutive low-volume chunks while not speaking.

    stt_resume_after_ts: int = 0
    # Ignore live inbound audio until this Twilio media timestamp after a
    # Twilio clear. This drops the tail of the agent's own audio echo while
    # still allowing buffered barge-in audio to be flushed into STT.

    pending_user_transcript_norm: str = ""
    # Normalized user transcript currently being handled.
    # Prevents duplicate silence-commit / Azure-final races from creating
    # multiple turns for the same utterance.

    stt_audio_buffer: list[bytes] = field(default_factory=list)
    # Ring buffer of recent PCM16 chunks kept while agent speaks.
    # On barge-in these are flushed to Azure STT so the first ~200ms
    # of the user's interruption is not lost.

    speaking_chunks: int = 0
    # Chunks elapsed since agent started speaking. Barge-in is disabled
    # for the first N chunks to prevent the initial echo burst from
    # falsely cancelling TTS. Reset to 0 when is_speaking becomes True.

    # ── Sentence counter ───────────────────────────────────────────────────
    sentence_count: int = 0
    # Monotonically incremented per TTS sentence.
    # Generates unique Twilio mark names: "sentence_0", "sentence_1", …

    response_generation: int = 0
    # Monotonically increasing response turn id.
    # Incremented on each new response start and on barge-in cancellation so
    # stale LLM/TTS tasks cannot mutate the active turn's state.

    # ── Lifecycle guard ────────────────────────────────────────────────────
    shutdown_in_progress: bool = False
    # Set True as soon as _handle_stop() starts.
    # Lets late STT final callbacks record the last transcript without
    # kicking off a new LLM/TTS response during teardown.

    stopped: bool = False
    # Set True the first time _handle_stop() runs.
    # Prevents double-execution when both the event branch (stop event) and
    # the finally block both call _handle_stop on a clean disconnect.

    # ── Helpers ────────────────────────────────────────────────────────────

    def next_mark_name(self) -> str:
        name = f"sentence_{self.sentence_count}"
        self.sentence_count += 1
        return name

    def add_user_turn(self, text: str) -> None:
        self.conversation_history.append({"role": "user", "content": text})

    def add_assistant_turn(self, text: str) -> None:
        self.conversation_history.append({"role": "assistant", "content": text})
