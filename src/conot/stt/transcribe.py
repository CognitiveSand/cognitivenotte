"""High-level transcription orchestrator."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from conot.stt.diarization import (
    create_diarizer,
    merge_transcription_with_diarization,
)
from conot.stt.exceptions import TranscriptionError
from conot.stt.models import StreamingSegment, TranscriptionResult
from conot.stt.registry import get_provider
from conot.stt.streaming import StreamingOrchestrator, create_streaming_transcriber

if TYPE_CHECKING:
    from conot.stt.protocol import STTProvider

logger = logging.getLogger(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[str, float], None]


def _get_configured_provider(
    provider: str | None,
    device: str | None,
    model_size: str | None,
    language: str | None = None,
) -> "STTProvider":
    """Get a provider with optional device/model/language configuration.

    Args:
        provider: Provider name or None for auto.
        device: Device ("cuda", "cpu") or None for auto.
        model_size: Model size or None for auto.
        language: Language code ("fr", "en") or None for auto-detect.

    Returns:
        Configured STTProvider instance.
    """
    # If any setting specified, we need to configure the provider directly
    if device is not None or model_size is not None or language is not None:
        # Determine which provider to use
        if provider is None or provider == "auto":
            # Try faster-whisper first, then whisper-cpp
            try:
                from conot.stt.providers.faster_whisper import FasterWhisperProvider

                return FasterWhisperProvider(
                    model_size=model_size or "auto",
                    device=device or "auto",
                    language=language,
                )
            except ImportError:
                pass

            try:
                from conot.stt.providers.whisper_cpp import WhisperCppProvider

                return WhisperCppProvider(model_size=model_size or "auto")
            except ImportError:
                pass

        elif provider == "faster-whisper":
            from conot.stt.providers.faster_whisper import FasterWhisperProvider

            return FasterWhisperProvider(
                model_size=model_size or "auto",
                device=device or "auto",
                language=language,
            )

        elif provider == "whisper-cpp":
            from conot.stt.providers.whisper_cpp import WhisperCppProvider

            return WhisperCppProvider(model_size=model_size or "auto")

    # Use registry for auto-selection
    return get_provider(provider)


def transcribe_audio(
    audio_path: Path | str,
    provider: str | None = None,
    device: str | None = None,
    model_size: str | None = None,
    language: str | None = None,
    enable_diarization: bool = True,
    huggingface_token: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TranscriptionResult:
    """High-level transcription with auto provider selection.

    This is the main entry point for batch transcription. It handles:
    - Provider selection based on hardware or explicit choice
    - Transcription with the selected provider
    - Optional speaker diarization
    - Progress reporting

    Args:
        audio_path: Path to the audio file to transcribe.
        provider: Provider name ("faster-whisper", "whisper-cpp") or None for auto.
        device: Compute device ("cuda", "cpu") or None for auto.
        model_size: Model size ("large-v3", "medium", "small", "tiny") or None for auto.
        language: Language code ("fr", "en") or None for auto-detect.
        enable_diarization: Whether to perform speaker diarization.
        huggingface_token: HuggingFace token for Pyannote (diarization).
        progress_callback: Callback for progress updates (stage, progress 0-1).

    Returns:
        TranscriptionResult with segments, speakers, and timing.

    Raises:
        TranscriptionError: If transcription fails.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise TranscriptionError(f"Audio file not found: {audio_path}")

    def report_progress(stage: str, progress: float) -> None:
        if progress_callback:
            progress_callback(stage, progress)
        logger.debug(f"{stage}: {progress * 100:.0f}%")

    # Step 1: Get provider
    report_progress("Loading model", 0.0)
    stt_provider = _get_configured_provider(provider, device, model_size, language)
    report_progress("Loading model", 0.5)

    # Step 2: Transcribe
    report_progress("Transcribing", 0.0)
    result = stt_provider.transcribe(audio_path)
    report_progress("Transcribing", 1.0)

    # Step 3: Diarization (if enabled)
    if enable_diarization:
        diarizer = create_diarizer(huggingface_token=huggingface_token)
        if diarizer.is_available():
            report_progress("Diarizing", 0.0)
            try:
                speaker_segments = diarizer.diarize(audio_path)
                result = merge_transcription_with_diarization(result, speaker_segments)
                report_progress("Diarizing", 1.0)
            except Exception as e:
                logger.warning(f"Diarization failed, continuing without speakers: {e}")
        else:
            logger.info("Diarization not available (missing HF_TOKEN or pyannote-audio)")

    report_progress("Complete", 1.0)
    return result


def transcribe_stream(
    audio_stream: Iterator[bytes],
    provider: str | None = None,
    sample_rate: int = 16000,
) -> Iterator[StreamingSegment]:
    """Stream transcription from audio iterator.

    Args:
        audio_stream: Iterator yielding audio chunks as bytes (int16, mono).
        provider: Provider name or None for auto-selection.
        sample_rate: Audio sample rate (default 16kHz).

    Yields:
        StreamingSegment with incremental transcription results.
    """
    stt_provider = get_provider(provider)
    yield from stt_provider.transcribe_stream(audio_stream)


class LiveTranscriber:
    """Live transcription manager for microphone input.

    Provides a high-level interface for live transcription with:
    - Audio recording from microphone
    - Streaming transcription
    - Progress callbacks
    - Graceful shutdown
    """

    def __init__(
        self,
        provider: str | None = None,
        sample_rate: int = 16000,
        device_id: int | None = None,
        callback: Callable[[StreamingSegment], None] | None = None,
    ) -> None:
        """Initialize the live transcriber.

        Args:
            provider: Provider name or None for auto.
            sample_rate: Audio sample rate.
            device_id: Audio input device ID (None for default).
            callback: Callback for segment updates.
        """
        self._provider_name = provider
        self._sample_rate = sample_rate
        self._device_id = device_id
        self._callback = callback

        self._provider: STTProvider | None = None
        self._orchestrator: StreamingOrchestrator | None = None
        self._stream = None
        self._running = False

    def start(self) -> None:
        """Start live transcription from microphone."""
        if self._running:
            return

        # Get provider
        self._provider = get_provider(self._provider_name)

        # Create streaming orchestrator
        self._orchestrator = create_streaming_transcriber(
            provider=self._provider,
            sample_rate=self._sample_rate,
            callback=self._callback,
        )

        # Start audio input
        try:
            import sounddevice as sd

            stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                device=self._device_id,
                callback=self._audio_callback,
                blocksize=int(self._sample_rate * 0.1),  # 100ms blocks
            )
            self._stream = stream

            orchestrator = self._orchestrator
            orchestrator.start()
            stream.start()
            self._running = True
            logger.info("Live transcription started")

        except Exception as e:
            raise TranscriptionError(f"Failed to start audio input: {e}") from e

    def stop(self) -> list[StreamingSegment]:
        """Stop live transcription and return final segments.

        Returns:
            List of all finalized segments.
        """
        if not self._running:
            return []

        self._running = False

        # Stop audio stream
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        # Stop orchestrator and get final segments
        final_segments: list[StreamingSegment] = []
        if self._orchestrator:
            final_segments = self._orchestrator.stop()
            self._orchestrator = None

        logger.info("Live transcription stopped")
        return final_segments

    def _audio_callback(
        self,
        indata: object,
        frames: int,  # noqa: ARG002
        time_info: object,  # noqa: ARG002
        status: object,
    ) -> None:
        """Sounddevice audio callback."""
        if status:
            logger.warning(f"Audio status: {status}")

        if self._orchestrator and self._running:
            # indata is NDArray[np.float32] from sounddevice
            self._orchestrator.feed_audio_callback(indata)  # type: ignore[arg-type]

    @property
    def is_running(self) -> bool:
        """Whether live transcription is running."""
        return self._running


def create_live_transcriber(
    provider: str | None = None,
    sample_rate: int = 16000,
    device_id: int | None = None,
    callback: Callable[[StreamingSegment], None] | None = None,
) -> LiveTranscriber:
    """Create a live transcriber for microphone input.

    Args:
        provider: Provider name or None for auto.
        sample_rate: Audio sample rate.
        device_id: Audio input device ID.
        callback: Callback for segment updates.

    Returns:
        Configured LiveTranscriber instance.
    """
    return LiveTranscriber(
        provider=provider,
        sample_rate=sample_rate,
        device_id=device_id,
        callback=callback,
    )
