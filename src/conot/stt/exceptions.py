"""STT-specific exceptions."""

from conot.exceptions import ConotError


class STTError(ConotError):
    """Base exception for STT-related errors."""


class ProviderNotFoundError(STTError):
    """Requested STT provider not found."""


class ProviderNotAvailableError(STTError):
    """STT provider dependencies not available."""


class NoProviderAvailableError(STTError):
    """No STT provider available for current hardware."""


class TranscriptionError(STTError):
    """Error during transcription."""


class DiarizationError(STTError):
    """Error during speaker diarization."""


class VADError(STTError):
    """Error during voice activity detection."""


class AudioFormatError(STTError):
    """Unsupported or invalid audio format."""


class ModelNotFoundError(STTError):
    """Required model file not found."""


class HardwareDetectionError(STTError):
    """Error detecting hardware capabilities."""
