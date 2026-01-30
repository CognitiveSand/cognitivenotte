"""Speech-to-text transcription module.

This module provides a generic interface for STT backends with:
- Streaming/quasi real-time transcription
- Hardware auto-detection for provider selection
- French and English language support with per-segment detection
- Speaker diarization
- Word-level timestamps
"""

from conot.stt.exceptions import (
    AudioFormatError,
    DiarizationError,
    HardwareDetectionError,
    ModelNotFoundError,
    NoProviderAvailableError,
    ProviderNotAvailableError,
    ProviderNotFoundError,
    STTError,
    TranscriptionError,
    VADError,
)
from conot.stt.hardware import detect_hardware
from conot.stt.models import (
    HardwareProfile,
    Language,
    ProviderCapabilities,
    ProviderTier,
    Segment,
    StreamingSegment,
    TranscriptionResult,
    Word,
)
from conot.stt.protocol import STTProvider
from conot.stt.registry import get_provider, get_registered_providers
from conot.stt.transcribe import (
    LiveTranscriber,
    create_live_transcriber,
    transcribe_audio,
    transcribe_stream,
)

__all__ = [
    # Protocol
    "STTProvider",
    # Models
    "HardwareProfile",
    "Language",
    "ProviderCapabilities",
    "ProviderTier",
    "Segment",
    "StreamingSegment",
    "TranscriptionResult",
    "Word",
    # Exceptions
    "AudioFormatError",
    "DiarizationError",
    "HardwareDetectionError",
    "ModelNotFoundError",
    "NoProviderAvailableError",
    "ProviderNotAvailableError",
    "ProviderNotFoundError",
    "STTError",
    "TranscriptionError",
    "VADError",
    # Functions
    "detect_hardware",
    "get_provider",
    "get_registered_providers",
    "transcribe_audio",
    "transcribe_stream",
    "create_live_transcriber",
    "LiveTranscriber",
]
