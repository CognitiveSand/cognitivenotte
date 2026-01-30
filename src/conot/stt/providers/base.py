"""Base class for STT providers with common utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from conot.stt.exceptions import AudioFormatError
from conot.stt.models import (
    ProviderCapabilities,
    StreamingSegment,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Supported audio formats
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".webm"}


class BaseSTTProvider(ABC):
    """Abstract base class for STT providers with common utilities."""

    def __init__(self) -> None:
        """Initialize the provider."""
        self._model_loaded = False

    def validate_audio_file(self, audio_path: Path) -> None:
        """Validate that audio file exists and has supported format.

        Args:
            audio_path: Path to audio file.

        Raises:
            AudioFormatError: If file doesn't exist or has unsupported format.
        """
        if not audio_path.exists():
            raise AudioFormatError(f"Audio file not found: {audio_path}")

        suffix = audio_path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise AudioFormatError(
                f"Unsupported audio format: {suffix}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

    @abstractmethod
    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file to structured result (batch mode).

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            TranscriptionResult with segments, speakers, and timing.
        """
        ...

    @abstractmethod
    def transcribe_stream(
        self,
        audio_stream: Iterator[bytes],
    ) -> Iterator[StreamingSegment]:
        """Stream transcription with incremental results.

        Args:
            audio_stream: Iterator yielding audio chunks as bytes.

        Yields:
            StreamingSegment with partial or final transcription.
        """
        ...

    @abstractmethod
    def detect_language(self, audio_path: Path) -> str:
        """Detect primary language in audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ISO 639-1 language code (e.g., "fr", "en").
        """
        ...

    @abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities.

        Returns:
            ProviderCapabilities describing what this provider supports.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider dependencies are installed and functional.

        Returns:
            True if the provider can be used, False otherwise.
        """
        ...

    def _ensure_model_loaded(self) -> None:  # noqa: B027
        """Ensure the model is loaded before transcription.

        Subclasses should override this to implement lazy model loading.
        Default implementation does nothing (model loaded at init).
        """
