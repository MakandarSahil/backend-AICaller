import logging
from collections.abc import AsyncGenerator

from groq import AsyncGroq

from app.config import get_settings
from app.models.query import Channel, LLMProvider, QueryPayload
from app.pipeline.prompt import build_prompt

settings = get_settings()
logger = logging.getLogger(__name__)

_groq_client: AsyncGroq | None = None

_SENTENCE_ENDINGS = {".", "!", "?", ",", ":", ";"}
_MIN_SENTENCE_LEN = 2
_MAX_BUFFER_CHARS = 50


def get_groq_client() -> AsyncGroq:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def stream_llm_sentences(payload: QueryPayload) -> AsyncGenerator[str, None]:
    """
    Core pipeline entry point - yields complete sentences from the LLM.
    """
    provider_str = (
        payload.agent_config.get("llm_provider")
        or payload.llm_provider.value
        or LLMProvider.GROQ.value
    )

    try:
        provider = LLMProvider(provider_str)
    except ValueError:
        logger.warning("Unknown llm_provider '%s' - falling back to groq", provider_str)
        provider = LLMProvider.GROQ

    logger.info("[LLM] provider=%s agent=%s", provider.value, payload.agent_id)

    if provider == LLMProvider.GROQ:
        async for sentence in _stream_groq(payload):
            yield sentence
        return

    if provider == LLMProvider.RAGFLOW:
        raise NotImplementedError("RAGFlow provider not yet implemented (Phase 4)")

    if provider == LLMProvider.PGVECTOR:
        raise NotImplementedError("pgvector provider not yet implemented (Phase 4)")

    logger.error("Unhandled provider: %s", provider)
    raise ValueError(f"Unhandled LLM provider: {provider}")


async def stream_llm_full(payload: QueryPayload) -> str:
    """
    Non-streaming variant - returns the complete LLM response as one string.
    """
    parts: list[str] = []
    async for sentence in stream_llm_sentences(payload):
        parts.append(sentence)
    return " ".join(parts).strip()


async def _stream_groq(payload: QueryPayload) -> AsyncGenerator[str, None]:
    """
    Stream from Groq and yield complete sentences one at a time.
    """
    messages = build_prompt(payload)
    model = payload.agent_config.get("llm_model") or settings.groq_model

    logger.info(
        "[LLM] request provider=groq agent=%s model=%s messages=%d max_tokens=%d temp=%.2f",
        payload.agent_id,
        model,
        len(messages),
        settings.groq_max_tokens,
        settings.groq_temperature,
    )

    client = get_groq_client()
    buffer = ""
    chunk_count = 0
    is_text_stream = payload.channel == Channel.TEXT_API

    try:
        stream = await client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.groq_max_tokens,
            temperature=settings.groq_temperature,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if not delta:
                continue

            chunk_count += 1

            # Text chat UX should feel truly live: stream each provider delta
            # instead of waiting for sentence boundaries.
            if is_text_stream:
                logger.debug("[LLM] delta chars=%d text=%r", len(delta), delta[:120])
                yield delta
                continue

            buffer += delta

            while True:
                idx = _sentence_boundary(buffer)
                if idx == -1:
                    break
                sentence = buffer[: idx + 1].strip()
                buffer = buffer[idx + 1 :]
                if len(sentence) >= _MIN_SENTENCE_LEN:
                    logger.debug("[LLM] sentence chars=%d text=%r", len(sentence), sentence[:120])
                    yield sentence

            if len(buffer) >= _MAX_BUFFER_CHARS:
                flush = buffer.strip()
                buffer = ""
                if len(flush) >= _MIN_SENTENCE_LEN:
                    logger.debug("[LLM] force_flush chars=%d text=%r", len(flush), flush[:120])
                    yield flush

        if not is_text_stream:
            final = buffer.strip()
            if len(final) >= _MIN_SENTENCE_LEN:
                logger.debug("[LLM] final_fragment chars=%d text=%r", len(final), final[:120])
                yield final

        logger.info(
            "[LLM] response_complete provider=groq agent=%s chunks=%d",
            payload.agent_id,
            chunk_count,
        )

    except Exception as exc:
        logger.error("Groq streaming error (agent=%s): %s", payload.agent_id, exc, exc_info=True)
        raise


def _sentence_boundary(text: str) -> int:
    """
    Return index of first sentence-ending character (. ! ? : ; ,).
    No guard for trailing content — yields even on the last char
    so TTS starts as soon as the LLM emits terminal punctuation.
    """
    for i, ch in enumerate(text):
        if ch in _SENTENCE_ENDINGS:
            return i
    return -1
