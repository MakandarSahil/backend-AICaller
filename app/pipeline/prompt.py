import logging

from app.config import get_settings
from app.models.query import Channel, QueryPayload

settings = get_settings()
logger = logging.getLogger(__name__)


def build_prompt(payload: QueryPayload) -> list[dict[str, str]]:
    """
    Assemble the full Groq message list for one LLM turn.

    Structure (v1 - full context dump, no RAG):
        1. System prompt       from agent config
        2. Knowledge base      all attached KB docs, concatenated, hard-capped
        3. Caller history      past conversation summaries for this caller
        4. Current turns       this call/session conversation so far
        5. Current user turn   the message we're responding to right now

    Returns: list[dict] ready to pass to groq.chat.completions.create(messages=...)
    """
    messages: list[dict[str, str]] = []

    # 1) System prompt
    system = payload.agent_config.get(
        "system_prompt",
        "You are a helpful, concise voice assistant. Keep responses short and natural.",
    )

    if payload.channel == Channel.TWILIO:
        system += (
            "\n\nIMPORTANT: You are speaking on a phone call. "
            "Respond in plain spoken sentences only. "
            "No bullet points, no markdown, no lists. "
            "Keep each response under 3 sentences unless the caller asks for detail."
        )

    preferred_language = _resolve_preferred_language(payload)
    if preferred_language:
        language_label = _language_display_name(preferred_language)
        system += (
            "\n\nLANGUAGE RULES: "
            f"Prefer responding in {language_label} ({preferred_language}). "
            "If the user clearly switches languages mid-conversation, mirror the user's language. "
            "Keep wording natural and concise for spoken conversation."
        )

    system += (
        "\n\nDOMAIN BOUNDARY: Answer only questions related to the agent's business context "
        "and the provided knowledge base. If a question is unrelated, politely refuse and "
        "ask the caller to continue with business-relevant questions."
    )

    kb_docs_count = len(payload.kb_documents)
    kb_chars = 0

    # 2) Knowledge base
    if payload.kb_documents:
        kb_text = "\n\n---\n\n".join(doc.strip() for doc in payload.kb_documents if doc.strip())
        kb_chars = len(kb_text)
        if len(kb_text) > settings.kb_max_chars:
            kb_text = kb_text[: settings.kb_max_chars]
            kb_text += "\n\n[Knowledge base truncated to fit context window]"

        system += f"\n\n## KNOWLEDGE BASE\n{kb_text}"

    # 3) Caller history
    if payload.caller_history:
        history_block = _format_caller_history(payload.caller_history)
        system += f"\n\n## CALLER HISTORY\n{history_block}"

    messages.append({"role": "system", "content": system})

    # 4) Current conversation turns
    valid_turns = 0
    for turn in payload.conversation_history:
        role = turn.get("role", "")
        content = turn.get("content", "").strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
            valid_turns += 1

    # 5) Current user message
    user_text = payload.text.strip()
    messages.append({"role": "user", "content": user_text})

    logger.info(
        "[PROMPT] built agent=%s messages=%d valid_turns=%d kb_docs=%d kb_chars=%d user_chars=%d",
        payload.agent_id,
        len(messages),
        valid_turns,
        kb_docs_count,
        kb_chars,
        len(user_text),
    )

    return messages


def _resolve_preferred_language(payload: QueryPayload) -> str | None:
    """
    Resolve preferred response language from configured agent locale fields.

    Priority:
      1) Explicit language/locale fields in agent config (stt_language, stt_locale, stt_model)
      2) TTS voice locale prefix (e.g. en-IN from en-IN-PrabhatNeural)
    """
    agent_config = payload.agent_config or {}

    for key in ("stt_language", "stt_locale", "stt_model"):
        value = str(agent_config.get(key) or "").strip()
        if value and value.lower() != "default":
            return value

    voice = str(agent_config.get("tts_voice") or "").strip()
    if voice:
        parts = voice.split("-", 2)
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"

    return None


def _language_display_name(locale: str) -> str:
    normalized = locale.strip().lower()
    if normalized == "hi-in":
        return "Hindi"
    if normalized == "mr-in":
        return "Marathi"
    if normalized == "en-in":
        return "English (India)"
    if normalized == "en-us":
        return "English (US)"
    if normalized == "en-gb":
        return "English (UK)"
    return locale


def _format_caller_history(caller_history: list[dict]) -> str:
    """
    Format past conversation summaries into a concise block.
    Each item expected to have: started_at (ISO string), summary (text).
    Most recent first (Supabase returns them desc by started_at).
    """
    lines = []
    for i, conv in enumerate(caller_history[: settings.caller_history_max_turns], start=1):
        date = conv.get("started_at", "unknown date")
        if "T" in str(date):
            date = str(date).split("T")[0]
        summary = conv.get("summary", "").strip()
        if summary:
            lines.append(f"{i}. [{date}] {summary}")

    return "\n".join(lines) if lines else "No previous conversations on record."
