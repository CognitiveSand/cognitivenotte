"""Protocol definition for STT providers."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Protocol, runtime_checkable

from .models import ProviderCapabilities, StreamingSegment, TranscriptionResult


@runtime_checkable
class STTProvider(Protocol):
    """Generic interface for STT backends.

    All STT providers must implement this protocol to be usable
    with the transcription system. The protocol supports both
    batch and streaming transcription modes.
    """

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file to structured result (batch mode).

        Args:
            audio_path: Path to the audio file to transcribe.

        Returns:
            TranscriptionResult with segments, speakers, and timing.
        """
        ...

    def transcribe_stream(
        self,
        audio_stream: Iterator[bytes],
    ) -> Iterator[StreamingSegment]:
        """Stream transcription with incremental results.

        Yields partial segments as audio is processed.
        Each segment may be refined in subsequent yields
        (tracked by segment_id).

        Args:
            audio_stream: Iterator yielding audio chunks as bytes.

        Yields:
            StreamingSegment with partial or final transcription.
        """
        ...

    def detect_language(self, audio_path: Path) -> str:
        """Detect primary language in audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ISO 639-1 language code (e.g., "fr", "en").
        """
        ...

    def get_capabilities(self) -> ProviderCapabilities:
        """Return provider capabilities.

        Returns:
            ProviderCapabilities describing what this provider supports.
        """
        ...

    def is_available(self) -> bool:
        """Check if provider dependencies are installed and functional.

        Returns:
            True if the provider can be used, False otherwise.
        """
        ...
