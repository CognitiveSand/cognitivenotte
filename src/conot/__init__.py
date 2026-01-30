"""Conot - Sound acquisition for cognitive note-taking."""

from conot.audio_meter import AudioMeter, calculate_peak, calculate_rms
from conot.config import Settings, get_settings
from conot.devices import AudioDevice, list_input_devices, select_best_device
from conot.exceptions import (
    ConfigError,
    ConotError,
    DeviceError,
    DeviceNotFoundError,
    RecordingError,
)
from conot.recorder import AudioRecorder

__version__ = "0.1.0"

__all__ = [
    "AudioDevice",
    "AudioMeter",
    "AudioRecorder",
    "ConfigError",
    "ConotError",
    "DeviceError",
    "DeviceNotFoundError",
    "RecordingError",
    "Settings",
    "calculate_peak",
    "calculate_rms",
    "get_settings",
    "list_input_devices",
    "select_best_device",
]


# Lazy imports for STT module to avoid loading heavy dependencies
def __getattr__(name: str):
    """Lazy import for STT components."""
    if name == "stt":
        from conot import stt

        return stt
    if name == "transcribe_audio":
        from conot.stt.transcribe import transcribe_audio

        return transcribe_audio
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
