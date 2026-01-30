"""Audio recording functionality."""

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import sounddevice as sd
from numpy.typing import NDArray
from scipy.io import wavfile

from conot.config import Settings, get_settings
from conot.devices import AudioDevice, get_device_by_id, list_input_devices, select_best_device
from conot.exceptions import (
    DeviceOpenError,
    RecordingAlreadyStartedError,
    RecordingError,
    RecordingNotStartedError,
)

AudioCallback = Callable[[NDArray[np.float32]], None]


def _find_pulse_device_id() -> int | None:
    """Find the sounddevice ID for pulse or pipewire."""
    devices = list_input_devices(include_bluetooth=False)
    for dev in devices:
        if dev.name in ("pipewire", "pulse"):
            return dev.id
    return None


class AudioRecorder:
    """Records audio from an input device to WAV files."""

    def __init__(
        self,
        device: AudioDevice | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the recorder.

        Args:
            device: Audio device to use. Auto-selects if None.
            settings: Application settings. Uses defaults if None.
        """
        self._settings = settings or get_settings()
        self._device = device
        self._stream: sd.InputStream | None = None
        self._chunks: list[NDArray[np.float32]] = []
        self._lock = Lock()
        self._recording = False
        self._start_time: datetime | None = None
        self._audio_callback: AudioCallback | None = None

    @property
    def device(self) -> AudioDevice:
        """Get the audio device (auto-select if needed)."""
        if self._device is None:
            device_id = self._settings.audio.device_id
            if device_id is not None:
                self._device = get_device_by_id(device_id)
            else:
                self._device = select_best_device()
        return self._device

    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._recording

    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def set_audio_callback(self, callback: AudioCallback | None) -> None:
        """Set callback for audio data (for level metering).

        Args:
            callback: Function called with each audio chunk
        """
        self._audio_callback = callback

    def _audio_handler(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Handle incoming audio data."""
        if status:
            pass  # Could log status flags if needed

        # Make a copy of the data
        chunk = indata.copy()

        with self._lock:
            if self._recording:
                self._chunks.append(chunk)

        # Call external callback for metering
        if self._audio_callback is not None:
            self._audio_callback(chunk)

    def start(self) -> None:
        """Start recording.

        Raises:
            RecordingAlreadyStartedError: If recording is already in progress
            DeviceOpenError: If device cannot be opened
        """
        if self._recording:
            raise RecordingAlreadyStartedError("Recording already in progress")

        device = self.device
        audio_config = self._settings.audio

        # Use device's native sample rate if it matches common rates
        sample_rate = audio_config.sample_rate
        channels = min(audio_config.channels, device.channels)

        # Handle Bluetooth/PulseAudio-only devices
        actual_device_id = device.id
        old_pulse_source = os.environ.get("PULSE_SOURCE")

        if device.pulse_source:
            # For Bluetooth devices, use pulse/pipewire and set PULSE_SOURCE
            pulse_id = _find_pulse_device_id()
            if pulse_id is not None:
                actual_device_id = pulse_id
                os.environ["PULSE_SOURCE"] = device.pulse_source

        try:
            self._stream = sd.InputStream(
                device=actual_device_id,
                samplerate=sample_rate,
                channels=channels,
                dtype=np.float32,
                callback=self._audio_handler,
            )
        except sd.PortAudioError as e:
            # Restore PULSE_SOURCE
            if old_pulse_source is not None:
                os.environ["PULSE_SOURCE"] = old_pulse_source
            elif "PULSE_SOURCE" in os.environ:
                del os.environ["PULSE_SOURCE"]
            raise DeviceOpenError(f"Failed to open device: {e}") from e

        with self._lock:
            self._chunks = []
            self._recording = True
            self._start_time = datetime.now()

        self._stream.start()

    def stop(self) -> Path:
        """Stop recording and save to file.

        Returns:
            Path to the saved WAV file

        Raises:
            RecordingNotStartedError: If recording was not started
            RecordingError: If saving fails
        """
        if not self._recording or self._stream is None:
            raise RecordingNotStartedError("No recording in progress")

        self._stream.stop()
        self._stream.close()

        with self._lock:
            self._recording = False
            chunks = self._chunks.copy()
            self._chunks = []
            start_time = self._start_time
            self._start_time = None

        self._stream = None

        if not chunks:
            raise RecordingError("No audio data recorded")

        # Concatenate all chunks
        audio_data = np.concatenate(chunks, axis=0)

        # Generate filename
        recording_config = self._settings.recording
        output_dir = recording_config.output_dir

        if start_time is None:
            start_time = datetime.now()

        filename = start_time.strftime(recording_config.filename_format)
        output_path = output_dir / filename

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to int16 for WAV file
        audio_int16 = (audio_data * 32767).astype(np.int16)

        try:
            wavfile.write(output_path, self._settings.audio.sample_rate, audio_int16)
        except Exception as e:
            raise RecordingError(f"Failed to save recording: {e}") from e

        return output_path
