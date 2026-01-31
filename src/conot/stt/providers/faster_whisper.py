"""Faster-whisper STT provider for GPU-accelerated transcription."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conot.stt.exceptions import ModelNotFoundError, TranscriptionError
from conot.stt.hardware import detect_hardware, get_recommended_model_size
from conot.stt.models import (
    ProviderCapabilities,
    ProviderTier,
    Segment,
    StreamingSegment,
    TranscriptionResult,
    Word,
)
from conot.stt.providers.base import BaseSTTProvider

if TYPE_CHECKING:
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import TranscriptionInfo

logger = logging.getLogger(__name__)

# Default model for each tier
DEFAULT_MODELS = {
    ProviderTier.ENTERPRISE: "large-v3",
    ProviderTier.STANDARD: "medium",
    ProviderTier.EDGE: "small",
}

# Compute types for each tier
COMPUTE_TYPES = {
    ProviderTier.ENTERPRISE: "float16",
    ProviderTier.STANDARD: "float16",
    ProviderTier.EDGE: "int8",
}


_cuda_available: bool | None = None


def _is_cuda_available() -> bool:
    """Check if CUDA is actually available and functional.

    Returns:
        True if CUDA can be used, False otherwise.
    """
    global _cuda_available
    if _cuda_available is not None:
        return _cuda_available

    # Try to actually use CUDA to verify it works
    try:
        import ctranslate2
        import numpy as np

        # Create a minimal model to test CUDA
        # This will fail if CUDA libraries are missing
        test_array = np.zeros((1, 80, 100), dtype=np.float32)
        storage = ctranslate2.StorageView.from_array(test_array)
        # If we get here without error, CUDA libraries are loadable
        # But we still need to test actual CUDA device
        del storage

        # Try torch for actual CUDA test
        try:
            import torch

            if torch.cuda.is_available():
                x = torch.zeros(1, device="cuda")
                del x
                _cuda_available = True
                logger.debug("CUDA is available (verified with torch)")
                return True
        except Exception as e:
            logger.debug(f"CUDA torch test failed: {e}")

    except Exception as e:
        error_msg = str(e).lower()
        if any(x in error_msg for x in ["cuda", "cublas", "cudnn", "nvrtc"]):
            logger.info(f"CUDA libraries not available: {e}")
            _cuda_available = False
            return False
        logger.debug(f"CUDA check error: {e}")

    # Default to False if we couldn't verify CUDA works
    _cuda_available = False
    return False


class FasterWhisperProvider(BaseSTTProvider):
    """STT provider using faster-whisper for GPU-accelerated transcription.

    This provider uses CTranslate2 for efficient GPU inference with
    support for word-level timestamps and language detection.
    """

    def __init__(
        self,
        model_size: str = "auto",
        device: str = "auto",
        compute_type: str = "auto",
        language: str | None = None,
    ) -> None:
        """Initialize the faster-whisper provider.

        Args:
            model_size: Model size (large-v3, medium, small, tiny) or "auto".
            device: Device to use (cuda, cpu) or "auto".
            compute_type: Compute precision (float16, int8, float32) or "auto".
            language: Language code (fr, en) or None for auto-detect.
        """
        super().__init__()
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language  # None = auto-detect
        self._model: WhisperModel | None = None
        self._hardware = detect_hardware()

    def _resolve_settings(self) -> tuple[str, str, str]:
        """Resolve auto settings based on hardware.

        Returns:
            Tuple of (model_size, device, compute_type).
        """
        # Resolve model size
        if self._model_size == "auto":
            model_size = get_recommended_model_size(self._hardware)
        else:
            model_size = self._model_size

        # Resolve device
        if self._device == "auto":
            if self._hardware.has_gpu and _is_cuda_available():
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = self._device

        # Resolve compute type
        if self._compute_type == "auto":
            if device == "cuda":
                compute_type = COMPUTE_TYPES.get(
                    self._hardware.recommended_tier, "float16"
                )
            else:
                compute_type = "int8"
        else:
            compute_type = self._compute_type

        return model_size, device, compute_type

    def _ensure_model_loaded(self) -> None:
        """Load the Whisper model if not already loaded."""
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ModelNotFoundError(
                "faster-whisper not installed. Install with: pip install faster-whisper"
            ) from e

        model_size, device, compute_type = self._resolve_settings()
        logger.info(
            f"Loading faster-whisper model: {model_size} on {device} ({compute_type})"
        )

        try:
            self._model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
            )
            self._model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            # Check if this is a CUDA library error and we can fall back to CPU
            error_msg = str(e).lower()
            cuda_errors = ["cuda", "cublas", "cudnn", "nvrtc", "gpu", "nvidia"]
            is_cuda_error = any(err in error_msg for err in cuda_errors)

            if is_cuda_error and device == "cuda" and self._device == "auto":
                logger.warning(
                    f"CUDA initialization failed: {e}. Falling back to CPU mode."
                )
                # Retry with CPU
                device = "cpu"
                compute_type = "int8"
                logger.info(
                    f"Loading faster-whisper model: {model_size} on {device} ({compute_type})"
                )
                try:
                    self._model = WhisperModel(
                        model_size,
                        device=device,
                        compute_type=compute_type,
                    )
                    self._model_loaded = True
                    logger.info("Model loaded successfully (CPU fallback)")
                    return
                except Exception as e2:
                    raise ModelNotFoundError(
                        f"Failed to load model '{model_size}' (CPU fallback): {e2}"
                    ) from e2

            raise ModelNotFoundError(f"Failed to load model '{model_size}': {e}") from e

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file to structured result.

        Args:
            audio_path: Path to the audio file.

        Returns:
            TranscriptionResult with segments and timing.
        """
        self.validate_audio_file(audio_path)
        self._ensure_model_loaded()

        if self._model is None:
            raise TranscriptionError("Model not loaded")

        try:
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                word_timestamps=True,
                language=self._language,  # None = auto-detect
                vad_filter=True,
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
            )

            segments = self._process_segments(segments_iter, info)
            languages = self._extract_languages(segments)

            return TranscriptionResult(
                audio_file=str(audio_path),
                duration_s=info.duration,
                languages_detected=languages,
                speakers=[],  # Diarization handled separately
                segments=segments,
            )

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _process_segments(
        self,
        segments_iter: Iterator[Any],
        info: TranscriptionInfo,
    ) -> list[Segment]:
        """Process transcription segments into structured format.

        Args:
            segments_iter: Iterator of faster-whisper segments.
            info: Transcription info with detected language.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        for seg in segments_iter:
            words: list[Word] = []
            if seg.words:
                for w in seg.words:
                    words.append(
                        Word(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            confidence=w.probability,
                        )
                    )

            # Use detected language from segment if available
            lang = getattr(seg, "language", None) or info.language

            segments.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    speaker="",  # Diarization handled separately
                    language=lang,
                    text=seg.text.strip(),
                    confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else 0.0,
                    words=words,
                )
            )

        return segments

    def _extract_languages(self, segments: list[Segment]) -> list[str]:
        """Extract unique languages from segments.

        Args:
            segments: List of transcription segments.

        Returns:
            List of unique language codes.
        """
        languages: list[str] = []
        for seg in segments:
            if seg.language and seg.language not in languages:
                languages.append(seg.language)
        return languages

    def transcribe_stream(
        self,
        audio_stream: Iterator[bytes],
    ) -> Iterator[StreamingSegment]:
        """Stream transcription with incremental results.

        Args:
            audio_stream: Iterator yielding audio chunks as bytes (16kHz, mono, int16).

        Yields:
            StreamingSegment with partial or final transcription.
        """
        self._ensure_model_loaded()

        if self._model is None:
            raise TranscriptionError("Model not loaded")

        # Accumulate audio chunks and process periodically
        audio_buffer = bytearray()
        chunk_size = 16000 * 2 * 2  # 2 seconds of 16kHz mono int16 audio
        segment_counter = 0
        time_offset = 0.0

        for chunk in audio_stream:
            audio_buffer.extend(chunk)

            # Process when we have enough audio
            while len(audio_buffer) >= chunk_size:
                # Extract chunk to process
                process_chunk = bytes(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]

                # Convert bytes to numpy array
                import numpy as np

                audio_array = np.frombuffer(process_chunk, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Transcribe chunk
                try:
                    segments_iter, info = self._model.transcribe(
                        audio_float,
                        word_timestamps=False,
                        language=self._language,
                        vad_filter=True,
                    )

                    for seg in segments_iter:
                        segment_counter += 1
                        yield StreamingSegment(
                            segment_id=f"seg_{segment_counter}_{uuid.uuid4().hex[:8]}",
                            start=time_offset + seg.start,
                            end=time_offset + seg.end,
                            text=seg.text.strip(),
                            language=info.language,
                            language_probability=info.language_probability,
                            is_final=True,
                            confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else 0.0,
                        )

                except Exception as e:
                    logger.warning(f"Streaming transcription error: {e}")

                # Update time offset (2 seconds per chunk)
                time_offset += 2.0

        # Process remaining audio
        if len(audio_buffer) > 1600:  # At least 0.1s of audio
            import numpy as np

            audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            try:
                segments_iter, info = self._model.transcribe(
                    audio_float,
                    word_timestamps=False,
                    language=self._language,
                    vad_filter=True,
                )

                for seg in segments_iter:
                    segment_counter += 1
                    yield StreamingSegment(
                        segment_id=f"seg_{segment_counter}_{uuid.uuid4().hex[:8]}",
                        start=time_offset + seg.start,
                        end=time_offset + seg.end,
                        text=seg.text.strip(),
                        language=info.language,
                        language_probability=info.language_probability,
                        is_final=True,
                        confidence=seg.avg_logprob if hasattr(seg, "avg_logprob") else 0.0,
                    )

            except Exception as e:
                logger.warning(f"Final streaming transcription error: {e}")

    def detect_language(self, audio_path: Path) -> str:
        """Detect primary language in audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ISO 639-1 language code.
        """
        self.validate_audio_file(audio_path)
        self._ensure_model_loaded()

        if self._model is None:
            raise TranscriptionError("Model not loaded")

        try:
            _, info = self._model.transcribe(
                str(audio_path),
                word_timestamps=False,
                language=self._language,
            )
            return str(info.language)
        except Exception as e:
            raise TranscriptionError(f"Language detection failed: {e}") from e

    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            languages=["fr", "en", "de", "es", "it", "pt", "nl", "ja", "zh", "ko"],
            supports_diarization=False,  # Handled by separate diarization module
            supports_word_timestamps=True,
            supports_streaming=True,
            streaming_latency_ms=1500,  # ~1.5s on GPU
            min_memory_gb=4.0,
            requires_gpu=False,  # Can run on CPU too
        )

    def is_available(self) -> bool:
        """Check if faster-whisper is installed and functional."""
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def get_model_info(self) -> dict[str, str]:
        """Get information about the current model."""
        model_size, device, compute_type = self._resolve_settings()
        return {
            "name": model_size,
            "provider": "faster-whisper",
            "device": device,
            "compute_type": compute_type,
        }
