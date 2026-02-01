"""Qwen3-ASR STT provider for high-accuracy transcription."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from conot.stt.exceptions import ModelNotFoundError, TranscriptionError
from conot.stt.hardware import detect_hardware
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
    from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)

# Model selection by tier
DEFAULT_MODELS = {
    ProviderTier.ENTERPRISE: "Qwen/Qwen3-ASR-1.7B",
    ProviderTier.STANDARD: "Qwen/Qwen3-ASR-1.7B",
    ProviderTier.EDGE: "Qwen/Qwen3-ASR-0.6B",
}


class QwenASRProvider(BaseSTTProvider):
    """STT provider using Qwen3-ASR for high-accuracy transcription.

    Qwen3-ASR achieves ~30% better WER than Whisper Large V3 while
    supporting 52 languages including French and English.

    All settings can be configured via settings.yml:
        stt.qwen.use_vllm
        stt.qwen.use_forced_aligner
        stt.qwen.gpu_memory_utilization
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        use_vllm: bool | None = None,
        language: str | None = None,
        use_forced_aligner: bool | None = None,
        gpu_memory_utilization: float | None = None,
    ) -> None:
        """Initialize the Qwen3-ASR provider.

        Args are optional - if None, reads from settings.yml.

        Args:
            model_name: Model name or "auto" for hardware-based selection.
            device: Device ("cuda", "cpu") or "auto".
            use_vllm: Use vLLM backend for faster inference.
            language: Language hint or None for auto-detect.
            use_forced_aligner: Enable word-level timestamps (batch only).
            gpu_memory_utilization: vLLM GPU memory fraction (0.0-1.0).
        """
        super().__init__()

        # Load settings with fallbacks
        from conot.config import get_settings

        settings = get_settings()

        # Read from settings.yml, override with explicit args
        stt = settings.stt if hasattr(settings, "stt") else None
        qwen = getattr(stt, "qwen", None) if stt else None

        self._model_name = model_name or getattr(stt, "model_size", "auto") or "auto"
        self._device = device or getattr(stt, "device", "auto") or "auto"
        self._language = language or getattr(stt, "language", None)
        if self._language == "auto":
            self._language = None  # None = auto-detect

        # Qwen-specific settings
        self._use_vllm = (
            use_vllm if use_vllm is not None else getattr(qwen, "use_vllm", False)
        )
        self._use_forced_aligner = (
            use_forced_aligner
            if use_forced_aligner is not None
            else getattr(qwen, "use_forced_aligner", False)
        )
        self._gpu_memory_utilization = gpu_memory_utilization or getattr(
            qwen, "gpu_memory_utilization", 0.7
        )

        self._model: Qwen3ASRModel | None = None
        self._hardware = detect_hardware()

    def _resolve_settings(self) -> tuple[str, str]:
        """Resolve auto settings based on hardware.

        Returns:
            Tuple of (model_name, device).
        """
        # Model selection
        if self._model_name == "auto":
            model_name = DEFAULT_MODELS.get(
                self._hardware.recommended_tier, "Qwen/Qwen3-ASR-0.6B"
            )
        elif "/" not in self._model_name:
            # Allow shorthand like "1.7B" or "0.6B"
            if "1.7" in self._model_name.upper():
                model_name = "Qwen/Qwen3-ASR-1.7B"
            elif "0.6" in self._model_name.upper():
                model_name = "Qwen/Qwen3-ASR-0.6B"
            else:
                model_name = f"Qwen/Qwen3-ASR-{self._model_name}"
        else:
            model_name = self._model_name

        # Device selection
        if self._device == "auto":
            device = "cuda:0" if self._hardware.has_gpu else "cpu"
        else:
            device = self._device

        return model_name, device

    def _ensure_model_loaded(self) -> None:
        """Load the Qwen3-ASR model if not already loaded."""
        if self._model is not None:
            return

        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            raise ModelNotFoundError(
                "qwen-asr not installed. Install with: pip install qwen-asr"
            ) from e

        model_name, device = self._resolve_settings()
        logger.info(f"Loading Qwen3-ASR model: {model_name} on {device}")

        try:
            if self._use_vllm:
                # vLLM backend (faster, but limited features)
                self._model = Qwen3ASRModel.LLM(
                    model=model_name,
                    gpu_memory_utilization=self._gpu_memory_utilization,
                )
            else:
                # Transformers backend
                import torch

                kwargs: dict[str, Any] = {
                    "dtype": torch.bfloat16,
                    "device_map": device,
                }
                if self._use_forced_aligner:
                    kwargs["forced_aligner"] = "Qwen/Qwen3-ForcedAligner-0.6B"

                self._model = Qwen3ASRModel.from_pretrained(
                    model_name,
                    **kwargs,
                )

            self._model_loaded = True
            logger.info("Qwen3-ASR model loaded successfully")

        except Exception as e:
            # Check if this is a CUDA error and we can fall back to CPU
            error_msg = str(e).lower()
            cuda_errors = ["cuda", "cublas", "cudnn", "nvrtc", "gpu", "nvidia"]
            is_cuda_error = any(err in error_msg for err in cuda_errors)

            if is_cuda_error and "cuda" in device and self._device == "auto":
                logger.warning(
                    f"CUDA initialization failed: {e}. Falling back to CPU mode."
                )
                # Retry with CPU
                device = "cpu"
                logger.info(f"Loading Qwen3-ASR model: {model_name} on {device}")
                try:
                    import torch

                    self._model = Qwen3ASRModel.from_pretrained(
                        model_name,
                        dtype=torch.float32,  # Use float32 on CPU
                        device_map=device,
                    )
                    self._model_loaded = True
                    logger.info("Qwen3-ASR model loaded successfully (CPU fallback)")
                    return
                except Exception as e2:
                    raise ModelNotFoundError(
                        f"Failed to load Qwen3-ASR model '{model_name}' (CPU fallback): {e2}"
                    ) from e2

            raise ModelNotFoundError(
                f"Failed to load Qwen3-ASR model '{model_name}': {e}"
            ) from e

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
            results = self._model.transcribe(
                audio=str(audio_path),
                language=self._language,
                return_time_stamps=self._use_forced_aligner,
            )

            segments = self._process_results(results, audio_path)
            languages = list({s.language for s in segments if s.language})

            # Get duration from audio file
            duration = self._get_audio_duration(audio_path)

            return TranscriptionResult(
                audio_file=str(audio_path),
                duration_s=duration,
                languages_detected=languages,
                speakers=[],  # Diarization handled separately
                segments=segments,
            )

        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio file duration in seconds."""
        import wave

        suffix = audio_path.suffix.lower()
        if suffix == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    return wf.getnframes() / wf.getframerate()
            except Exception:
                pass

        # Fallback: use scipy for other formats
        try:
            from scipy.io import wavfile

            sample_rate, data = wavfile.read(str(audio_path))
            return len(data) / sample_rate
        except Exception:
            # Last resort: estimate from file size
            return 0.0

    def _process_results(
        self,
        results: list[Any],
        audio_path: Path,
    ) -> list[Segment]:
        """Process Qwen3-ASR results into Segment objects.

        Args:
            results: List of transcription results from Qwen3-ASR.
            audio_path: Path to the audio file.

        Returns:
            List of Segment objects.
        """
        segments: list[Segment] = []

        for result in results:
            words: list[Word] = []

            # Extract word timestamps if available
            if hasattr(result, "timestamps") and result.timestamps:
                for ts in result.timestamps:
                    words.append(
                        Word(
                            word=ts.text,
                            start=ts.start_time,
                            end=ts.end_time,
                            confidence=getattr(ts, "confidence", 0.9),
                        )
                    )

            # Estimate segment timing from words or use defaults
            start = words[0].start if words else 0.0
            end = words[-1].end if words else 0.0

            segments.append(
                Segment(
                    start=start,
                    end=end,
                    speaker="",  # Diarization handled separately
                    language=getattr(result, "language", "") or self._language or "",
                    text=result.text.strip(),
                    confidence=getattr(result, "confidence", 0.9),
                    words=words,
                )
            )

        return segments

    def transcribe_stream(
        self,
        audio_stream: Iterator[bytes],
    ) -> Iterator[StreamingSegment]:
        """Stream transcription with incremental results.

        Note: Qwen3-ASR streaming works best with vLLM backend but
        this implementation uses chunk-based processing for the
        transformers backend.

        Args:
            audio_stream: Iterator yielding audio chunks as bytes (16kHz, mono, int16).

        Yields:
            StreamingSegment with transcription results.
        """
        self._ensure_model_loaded()

        if self._model is None:
            raise TranscriptionError("Model not loaded")

        # Accumulate audio chunks (Qwen3-ASR processes larger chunks)
        audio_buffer = bytearray()
        chunk_size = 16000 * 2 * 3  # 3 seconds of 16kHz mono int16
        segment_counter = 0
        time_offset = 0.0

        for chunk in audio_stream:
            audio_buffer.extend(chunk)

            while len(audio_buffer) >= chunk_size:
                process_chunk = bytes(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]

                # Convert to numpy array
                audio_array = np.frombuffer(process_chunk, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                try:
                    results = self._model.transcribe(
                        audio=(audio_float, 16000),
                        language=self._language,
                    )

                    for result in results:
                        if result.text.strip():
                            segment_counter += 1
                            yield StreamingSegment(
                                segment_id=f"qwen_{segment_counter}_{uuid.uuid4().hex[:8]}",
                                start=time_offset,
                                end=time_offset + 3.0,
                                text=result.text.strip(),
                                language=getattr(result, "language", "") or "",
                                language_probability=0.9,
                                is_final=True,
                                confidence=0.9,
                            )
                except Exception as e:
                    logger.warning(f"Streaming transcription error: {e}")

                time_offset += 3.0

        # Process remaining audio
        if len(audio_buffer) > 3200:  # >0.2s
            audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            try:
                results = self._model.transcribe(
                    audio=(audio_float, 16000),
                    language=self._language,
                )

                for result in results:
                    if result.text.strip():
                        segment_counter += 1
                        remaining_duration = len(audio_buffer) / (16000 * 2)
                        yield StreamingSegment(
                            segment_id=f"qwen_{segment_counter}_{uuid.uuid4().hex[:8]}",
                            start=time_offset,
                            end=time_offset + remaining_duration,
                            text=result.text.strip(),
                            language=getattr(result, "language", "") or "",
                            language_probability=0.9,
                            is_final=True,
                            confidence=0.9,
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
            results = self._model.transcribe(
                audio=str(audio_path),
                language=None,  # Auto-detect
            )
            if results and hasattr(results[0], "language"):
                return results[0].language or "en"
            return "en"
        except Exception as e:
            raise TranscriptionError(f"Language detection failed: {e}") from e

    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            languages=[
                "fr",
                "en",
                "zh",
                "de",
                "es",
                "pt",
                "it",
                "ja",
                "ko",
                "ru",
                "ar",
                "th",
                "vi",
                "tr",
                "hi",
                "id",
                "ms",
                "nl",
                "sv",
                "da",
                "fi",
                "pl",
                "cs",
                "fil",
                "fa",
                "el",
                "hu",
                "mk",
                "ro",
            ],
            supports_diarization=False,  # Handled by separate diarization module
            supports_word_timestamps=self._use_forced_aligner,
            supports_streaming=True,
            streaming_latency_ms=2000,  # ~2s chunks
            min_memory_gb=4.0,
            requires_gpu=False,  # Can run on CPU too
        )

    def is_available(self) -> bool:
        """Check if qwen-asr is installed and functional."""
        try:
            import qwen_asr  # noqa: F401

            return True
        except ImportError:
            return False

    def get_model_info(self) -> dict[str, str]:
        """Get information about the current model."""
        model_name, device = self._resolve_settings()
        return {
            "name": model_name.split("/")[-1],
            "provider": "qwen-asr",
            "device": device,
            "backend": "vllm" if self._use_vllm else "transformers",
        }
