import logging
from typing import Callable

import azure.cognitiveservices.speech as speechsdk

from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class AzureSTT:
    """
    Wraps Azure Cognitive Speech SDK in push-stream mode.

    Twilio sends mulaw audio → we convert to PCM16 → push here.
    Azure processes continuously and fires callbacks on partial/final results.

    Usage:
        stt = AzureSTT(
            on_partial=lambda text: ...,
            on_final=lambda text: ...,
            on_error=lambda err: ...,
        )
        stt.start()
        stt.push(pcm16_bytes)   # call for every audio chunk
        stt.stop()
    """

    def __init__(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
        on_error: Callable[[Exception], None],
        language: str | None = None,
    ) -> None:
        self._on_partial = on_partial
        self._on_final = on_final
        self._on_error = on_error
        self._language = language or settings.stt_language

        # Azure SDK objects — created in start()
        self._speech_config: speechsdk.SpeechConfig | None = None
        self._push_stream: speechsdk.audio.PushAudioInputStream | None = None
        self._audio_config: speechsdk.audio.AudioConfig | None = None
        self._recognizer: speechsdk.SpeechRecognizer | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """
        Initialise the Azure push stream and start continuous recognition.
        Call once after receiving the Twilio 'start' event.
        """
        if self._running:
            logger.warning("STT.start() called but already running — ignoring")
            return

        # Speech config
        self._speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        self._speech_config.speech_recognition_language = self._language

        self._speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "3000",
        )
        self._speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "150",
        )
        if hasattr(speechsdk.PropertyId, "Speech_SegmentationSilenceTimeoutMs"):
            self._speech_config.set_property(
                speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs,
                "150",
            )

        # Push stream — PCM16 8kHz mono (Twilio's native format after mulaw decode)
        audio_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=8000,
            bits_per_sample=16,
            channels=1,
        )
        self._push_stream = speechsdk.audio.PushAudioInputStream(audio_format)
        self._audio_config = speechsdk.audio.AudioConfig(stream=self._push_stream)

        # Recognizer
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=self._speech_config,
            audio_config=self._audio_config,
        )

        # ── Wire up callbacks ──────────────────────────────────────────────
        self._recognizer.recognizing.connect(self._handle_recognizing)
        self._recognizer.recognized.connect(self._handle_recognized)
        self._recognizer.canceled.connect(self._handle_canceled)
        self._recognizer.session_stopped.connect(self._handle_session_stopped)

        self._recognizer.start_continuous_recognition()
        self._running = True
        logger.info("Azure STT started (language=%s)", self._language)

    def stop(self) -> None:
        """
        Close the push stream and stop recognition.
        Call on call end or when audio should stop being processed.
        """
        if not self._running:
            return
        self._running = False
        try:
            if self._push_stream:
                self._push_stream.close()
            if self._recognizer:
                self._recognizer.stop_continuous_recognition()
        except Exception as exc:
            logger.warning("STT stop error (non-fatal): %s", exc)
        finally:
            self._recognizer = None
            self._push_stream = None
            self._audio_config = None
            self._speech_config = None
            logger.info("Azure STT stopped")

    # ── Audio input ────────────────────────────────────────────────────────

    def push(self, pcm16_bytes: bytes) -> None:
        """
        Push a PCM16 chunk into the Azure recognition stream.
        Called from the WebSocket media handler for every Twilio audio chunk.
        Only pushes if STT is running — safe to call anytime.
        """
        if self._running and self._push_stream:
            try:
                self._push_stream.write(pcm16_bytes)
            except Exception as exc:
                logger.warning("STT push error: %s", exc)

    # ── Callbacks (called by Azure SDK on its own thread) ──────────────────

    def _handle_recognizing(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Partial result — fires while the user is still speaking."""
        text = evt.result.text
        if text:
            try:
                self._on_partial(text)
            except Exception as exc:
                logger.error("STT on_partial callback error: %s", exc)

    def _handle_recognized(self, evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        """Final result — fires after the user finishes a utterance."""
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text.strip()
            if text:
                logger.info("STT final: %s", text)
                try:
                    self._on_final(text)
                except Exception as exc:
                    logger.error("STT on_final callback error: %s", exc)
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            logger.debug("STT no match: %s", evt.result.no_match_details)
        else:
            logger.debug("STT recognized event with reason=%s", evt.result.reason)

    def _handle_canceled(self, evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        """Called when recognition is canceled (error or explicit stop)."""
        if evt.reason == speechsdk.CancellationReason.Error:
            err = Exception(
                f"Azure STT canceled: {evt.error_details} "
                f"(code={evt.error_code})"
            )
            logger.error("STT canceled with error: %s", err)
            try:
                self._on_error(err)
            except Exception as cb_exc:
                logger.error("STT on_error callback error: %s", cb_exc)
        else:
            # Normal stop — not an error
            logger.debug("STT recognition canceled (normal): reason=%s", evt.reason)

    def _handle_session_stopped(self, evt: speechsdk.SessionEventArgs) -> None:
        logger.debug("STT session stopped: %s", evt)