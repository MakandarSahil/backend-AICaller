import asyncio
import logging
from collections.abc import AsyncGenerator

import azure.cognitiveservices.speech as speechsdk

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Twilio expects mulaw chunks in 160-byte pieces (20ms at 8kHz)
_CHUNK_SIZE = 160


async def synthesize_sentence(
    text: str,
    voice: str | None = None,
) -> AsyncGenerator[bytes, None]:
    """
    Synthesize one sentence and yield Twilio-sized mulaw chunks.

    This deliberately uses Azure's blocking synthesis call in a worker thread.
    The direct pull-stream approach was hanging in the live Docker container,
    which meant Twilio never received audio completion marks and calls stalled.
    Sentence-level chunking still gives us streamed playback because the LLM
    hands off sentences incrementally.
    """
    voice = voice or settings.tts_default_voice
    text = text.strip()
    if not text:
        return

    raw_audio = await asyncio.get_running_loop().run_in_executor(
        None,
        _synthesize_blocking,
        text,
        voice,
    )

    if not raw_audio:
        logger.warning("TTS returned empty audio for: %s", text[:60])
        return

    total_chunks = 0
    for i in range(0, len(raw_audio), _CHUNK_SIZE):
        yield raw_audio[i : i + _CHUNK_SIZE]
        total_chunks += 1

    logger.debug("TTS streamed %d chunks for: %s", total_chunks, text[:60])


def _synthesize_blocking(text: str, voice: str) -> bytes | None:
    """
    Synchronous Azure TTS call — runs in a thread via run_in_executor.

    Uses Azure's pull-stream output so we get the full audio buffer
    without needing a speaker device or file write.
    Returns raw mulaw bytes (or None on failure).
    """
    return _synthesize_blocking_with_format(
        text=text,
        voice=voice,
        output_format=speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw,
    )


async def synthesize_preview_wav(text: str, voice: str | None = None) -> bytes | None:
    """
    Synthesize browser-playable WAV preview audio for dashboard voice testing.
    """
    voice = voice or settings.tts_default_voice
    text = text.strip()
    if not text:
        return None

    return await asyncio.get_event_loop().run_in_executor(
        None,
        _synthesize_preview_wav_blocking,
        text,
        voice,
    )


def _synthesize_preview_wav_blocking(text: str, voice: str) -> bytes | None:
    return _synthesize_blocking_with_format(
        text=text,
        voice=voice,
        output_format=speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm,
    )


def _synthesize_blocking_with_format(
    text: str,
    voice: str,
    output_format: speechsdk.SpeechSynthesisOutputFormat,
) -> bytes | None:
    speech_config = speechsdk.SpeechConfig(
        subscription=settings.azure_speech_key,
        region=settings.azure_speech_region,
    )

    speech_config.set_speech_synthesis_output_format(output_format)

    # Use a pull-stream to capture audio in memory
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)

    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config,
        audio_config=audio_config,
    )

    # Build SSML for the specified voice
    ssml = _build_ssml(text, voice)

    result = synthesizer.speak_ssml(ssml)

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        audio_data = result.audio_data
        logger.debug("TTS synthesized %d bytes for: %s", len(audio_data), text[:60])
        return audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = speechsdk.SpeechSynthesisCancellationDetails(result)
        logger.error(
            "TTS canceled: %s — %s", cancellation.reason, cancellation.error_details
        )
        return None
    else:
        logger.error("TTS unexpected result reason: %s", result.reason)
        return None


def _build_ssml(text: str, voice: str) -> str:
    """
    Minimal SSML wrapper.
    Azure requires SSML to specify voice — plain text synthesis
    ignores the voice config in some SDK versions.
    """
    # Escape XML special chars in the text
    safe_text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}"><prosody rate="+15%">{safe_text}</prosody></voice>'
        f"</speak>"
    )
