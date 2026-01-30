"""Whisper.cpp STT provider for CPU-optimized transcription."""

from __future__ import annotations

import logging
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conot.stt.exceptions import ModelNotFoundError, TranscriptionError
from conot.stt.hardware import detect_hardware, get_recommended_model_size
from conot.stt.models import (
    ProviderCapabilities,
    Segment,
    StreamingSegment,
    TranscriptionResult,
)
from conot.stt.providers.base import BaseSTTProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Model file names for whisper.cpp
MODEL_FILES = {
    "tiny": "ggml-tiny.bin",
    "base": "ggml-base.bin",
    "small": "ggml-small.bin",
    "medium": "ggml-medium.bin",
    "large-v3": "ggml-large-v3.bin",
}

# Quantized model variants
QUANTIZED_MODEL_FILES = {
    "tiny": "ggml-tiny.en-q5_1.bin",
    "small": "ggml-small-q5_1.bin",
    "medium": "ggml-medium-q5_0.bin",
}


class WhisperCppProvider(BaseSTTProvider):
    """STT provider using whisper.cpp for CPU-optimized transcription.

    This provider uses whisper.cpp bindings for efficient CPU inference
    with support for quantized models and real-time streaming.
    """

    def __init__(
        self,
        model_size: str = "auto",
        model_path: str | Path | None = None,
        use_quantized: bool = True,
    ) -> None:
        """Initialize the whisper.cpp provider.

        Args:
            model_size: Model size (large-v3, medium, small, tiny) or "auto".
            model_path: Optional explicit path to model file.
            use_quantized: Use quantized models when available for faster CPU inference.
        """
        super().__init__()
        self._model_size = model_size
        self._model_path = Path(model_path) if model_path else None
        self._use_quantized = use_quantized
        self._model: Any = None
        self._hardware = detect_hardware()

    def _resolve_model_path(self) -> Path:
        """Resolve the model file path.

        Returns:
            Path to the model file.

        Raises:
            ModelNotFoundError: If model file cannot be found.
        """
        if self._model_path and self._model_path.exists():
            return self._model_path

        # Resolve model size
        if self._model_size == "auto":
            model_size = get_recommended_model_size(self._hardware)
        else:
            model_size = self._model_size

        # Determine model filename
        if self._use_quantized and model_size in QUANTIZED_MODEL_FILES:
            filename = QUANTIZED_MODEL_FILES[model_size]
        elif model_size in MODEL_FILES:
            filename = MODEL_FILES[model_size]
        else:
            raise ModelNotFoundError(f"Unknown model size: {model_size}")

        # Search for model in common locations
        search_paths = [
            Path.home() / ".cache" / "whisper" / filename,
            Path.home() / ".local" / "share" / "whisper" / filename,
            Path("/usr/local/share/whisper") / filename,
            Path("./models") / filename,
        ]

        # Check XDG_DATA_HOME
        xdg_data = os.environ.get("XDG_DATA_HOME")
        if xdg_data:
            search_paths.insert(0, Path(xdg_data) / "whisper" / filename)

        for path in search_paths:
            if path.exists():
                logger.debug(f"Found model at: {path}")
                return path

        raise ModelNotFoundError(
            f"Model file '{filename}' not found. "
            f"Download from: https://huggingface.co/ggerganov/whisper.cpp"
        )

    def _ensure_model_loaded(self) -> None:
        """Load the whisper.cpp model if not already loaded."""
        if self._model is not None:
            return

        model_path = self._resolve_model_path()

        # Try different whisper.cpp Python bindings
        try:
            self._load_with_pywhispercpp(model_path)
            return
        except ImportError:
            logger.debug("pywhispercpp not available")

        try:
            self._load_with_whispercpp(model_path)
            return
        except ImportError:
            logger.debug("whispercpp not available")

        raise ModelNotFoundError(
            "No whisper.cpp Python bindings found. "
            "Install with: pip install pywhispercpp"
        )

    def _load_with_pywhispercpp(self, model_path: Path) -> None:
        """Load model using pywhispercpp."""
        from pywhispercpp.model import Model

        logger.info(f"Loading whisper.cpp model: {model_path}")
        self._model = Model(str(model_path))
        self._model_type = "pywhispercpp"
        self._model_loaded = True
        logger.info("Model loaded successfully")

    def _load_with_whispercpp(self, model_path: Path) -> None:
        """Load model using whispercpp binding."""
        import whispercpp

        logger.info(f"Loading whisper.cpp model: {model_path}")
        self._model = whispercpp.Whisper.from_pretrained(str(model_path))
        self._model_type = "whispercpp"
        self._model_loaded = True
        logger.info("Model loaded successfully")

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
            if self._model_type == "pywhispercpp":
                return self._transcribe_pywhispercpp(audio_path)
            else:
                return self._transcribe_whispercpp(audio_path)
        except Exception as e:
            raise TranscriptionError(f"Transcription failed: {e}") from e

    def _transcribe_pywhispercpp(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using pywhispercpp."""
        # Load audio and get duration
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        # Transcribe
        result = self._model.transcribe(str(audio_path))

        # Process segments
        segments: list[Segment] = []
        detected_language = "unknown"

        for seg in result:
            # pywhispercpp segment format
            text = getattr(seg, "text", str(seg))
            start = getattr(seg, "t0", 0) / 100.0 if hasattr(seg, "t0") else 0.0
            end = getattr(seg, "t1", 0) / 100.0 if hasattr(seg, "t1") else start + 1.0

            if text.strip():
                segments.append(
                    Segment(
                        start=start,
                        end=end,
                        speaker="",
                        language=detected_language,
                        text=text.strip(),
                        confidence=0.0,
                        words=[],
                    )
                )

        return TranscriptionResult(
            audio_file=str(audio_path),
            duration_s=duration,
            languages_detected=[detected_language] if detected_language != "unknown" else [],
            speakers=[],
            segments=segments,
        )

    def _transcribe_whispercpp(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe using whispercpp binding."""
        # Load audio
        import wave

        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)
            audio_data = wf.readframes(frames)

        import numpy as np

        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        audio_float = audio_array.astype(np.float32) / 32768.0

        # Transcribe
        result = self._model.transcribe(audio_float)

        # Process segments
        segments: list[Segment] = []

        if isinstance(result, str):
            # Simple string result
            segments.append(
                Segment(
                    start=0.0,
                    end=duration,
                    speaker="",
                    language="unknown",
                    text=result.strip(),
                    confidence=0.0,
                    words=[],
                )
            )
        else:
            # Structured result
            for seg in result.get("segments", []):
                segments.append(
                    Segment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        speaker="",
                        language=result.get("language", "unknown"),
                        text=seg.get("text", "").strip(),
                        confidence=seg.get("confidence", 0.0),
                        words=[],
                    )
                )

        return TranscriptionResult(
            audio_file=str(audio_path),
            duration_s=duration,
            languages_detected=[],
            speakers=[],
            segments=segments,
        )

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

        # Accumulate audio and process in chunks
        import numpy as np

        audio_buffer = bytearray()
        chunk_duration_s = 3.0  # Process every 3 seconds
        chunk_size = int(16000 * 2 * chunk_duration_s)  # 16kHz, int16
        segment_counter = 0
        time_offset = 0.0

        for chunk in audio_stream:
            audio_buffer.extend(chunk)

            while len(audio_buffer) >= chunk_size:
                # Extract chunk
                process_chunk = bytes(audio_buffer[:chunk_size])
                audio_buffer = audio_buffer[chunk_size:]

                # Convert to float
                audio_array = np.frombuffer(process_chunk, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                try:
                    if self._model_type == "pywhispercpp":
                        # pywhispercpp doesn't support direct array input well
                        # We need to save to temp file
                        import tempfile
                        import wave

                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                            temp_path = f.name

                        with wave.open(temp_path, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(process_chunk)

                        result = self._model.transcribe(temp_path)
                        os.unlink(temp_path)

                        for seg in result:
                            text = getattr(seg, "text", str(seg))
                            if text.strip():
                                segment_counter += 1
                                t0 = getattr(seg, "t0", 0)
                                start = t0 / 100.0 if hasattr(seg, "t0") else 0.0
                                t1 = getattr(seg, "t1", 0)
                                end = t1 / 100.0 if hasattr(seg, "t1") else start + 1.0

                                yield StreamingSegment(
                                    segment_id=f"seg_{segment_counter}_{uuid.uuid4().hex[:8]}",
                                    start=time_offset + start,
                                    end=time_offset + end,
                                    text=text.strip(),
                                    language="unknown",
                                    is_final=True,
                                    confidence=0.0,
                                )
                    else:
                        result = self._model.transcribe(audio_float)
                        if isinstance(result, str) and result.strip():
                            segment_counter += 1
                            yield StreamingSegment(
                                segment_id=f"seg_{segment_counter}_{uuid.uuid4().hex[:8]}",
                                start=time_offset,
                                end=time_offset + chunk_duration_s,
                                text=result.strip(),
                                language="unknown",
                                is_final=True,
                                confidence=0.0,
                            )

                except Exception as e:
                    logger.warning(f"Streaming transcription error: {e}")

                time_offset += chunk_duration_s

        # Process remaining audio
        if len(audio_buffer) > 3200:  # At least 0.1s
            import tempfile
            import wave

            audio_array = np.frombuffer(bytes(audio_buffer), dtype=np.int16)

            try:
                if self._model_type == "pywhispercpp":
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        temp_path = f.name

                    with wave.open(temp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(16000)
                        wf.writeframes(bytes(audio_buffer))

                    result = self._model.transcribe(temp_path)
                    os.unlink(temp_path)

                    for seg in result:
                        text = getattr(seg, "text", str(seg))
                        if text.strip():
                            segment_counter += 1
                            yield StreamingSegment(
                                segment_id=f"seg_{segment_counter}_{uuid.uuid4().hex[:8]}",
                                start=time_offset,
                                end=time_offset + len(audio_buffer) / (16000 * 2),
                                text=text.strip(),
                                language="unknown",
                                is_final=True,
                                confidence=0.0,
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

        # whisper.cpp doesn't have explicit language detection API
        # We transcribe a short segment and infer from the result
        result = self.transcribe(audio_path)
        if result.languages_detected:
            return result.languages_detected[0]
        return "unknown"

    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities."""
        return ProviderCapabilities(
            languages=["fr", "en", "de", "es", "it", "pt", "nl", "ja", "zh", "ko"],
            supports_diarization=False,
            supports_word_timestamps=False,  # Limited support in whisper.cpp
            supports_streaming=True,
            streaming_latency_ms=3500,  # ~3.5s on CPU
            min_memory_gb=2.0,
            requires_gpu=False,
        )

    def is_available(self) -> bool:
        """Check if whisper.cpp bindings are installed."""
        try:
            import pywhispercpp  # noqa: F401

            return True
        except ImportError:
            pass

        try:
            import whispercpp  # noqa: F401

            return True
        except ImportError:
            pass

        return False
