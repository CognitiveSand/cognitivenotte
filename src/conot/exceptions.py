"""Custom exceptions for Conot."""


class ConotError(Exception):
    """Base exception for all Conot errors."""


class ConfigError(ConotError):
    """Configuration-related errors."""


class DeviceError(ConotError):
    """Audio device-related errors."""


class DeviceNotFoundError(DeviceError):
    """No suitable audio input device found."""


class DeviceOpenError(DeviceError):
    """Failed to open audio device."""


class RecordingError(ConotError):
    """Recording-related errors."""


class RecordingNotStartedError(RecordingError):
    """Attempted to stop recording that was not started."""


class RecordingAlreadyStartedError(RecordingError):
    """Attempted to start recording that is already in progress."""


# STT-related exceptions are in conot.stt.exceptions
