import asyncio
import base64
import logging
from collections.abc import AsyncGenerator
import azure.cognitiveservices.speech as speechsdk
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger("app.pipeline.tts")

_CHUNK_SIZE = 160

async def synthesize_sentence(text: str, voice: str | None = None) -> AsyncGenerator[bytes, None]:
    voice = voice or settings.tts_default_voice
    text = text.strip()
    if not text:
        return

    speech_config = speechsdk.SpeechConfig(
        subscription=settings.azure_speech_key,
        region=settings.azure_speech_region,
    )
    # Use RAW format, no RIFF header to strip!
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw8Khz8BitMonoMULaw
    )

    pull_stream = speechsdk.audio.PullAudioOutputStream()
    audio_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    synthesizer = speechsdk.SpeechSynthesizer(
        speech_config=speech_config, audio_config=audio_config
    )

    ssml = _build_ssml(text, voice)
    
    # Start synthesis asynchronously
    result_future = synthesizer.start_speaking_ssml_async(ssml)
    
    # Read from the pull stream in chunks as they become available
    total_chunks = 0
    buffer = bytearray(_CHUNK_SIZE)
    
    loop = asyncio.get_running_loop()
    
    while True:
        # Run blocking read in thread
        bytes_read = await loop.run_in_executor(None, pull_stream.read, buffer)
        if bytes_read == 0:
            break
            
        yield bytes(buffer[:bytes_read])
        total_chunks += 1
        
    logger.debug("TTS streamed %d chunks for: %s", total_chunks, text[:60])
    
def _build_ssml(text: str, voice: str) -> str:
    safe_text = (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
    )
    return (
        f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
        f'<voice name="{voice}">{safe_text}</voice>'
        f"</speak>"
    )
