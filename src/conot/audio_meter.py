"""Audio level metering calculations."""

import numpy as np
from numpy.typing import NDArray


def calculate_rms(audio_data: NDArray[np.float32]) -> float:
    """Calculate RMS (Root Mean Square) level.

    Args:
        audio_data: Audio samples as float32 array

    Returns:
        RMS value (0.0 to 1.0 range for normalized audio)
    """
    if audio_data.size == 0:
        return 0.0

    # Flatten to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    return float(np.sqrt(np.mean(audio_data**2)))


def calculate_peak(audio_data: NDArray[np.float32]) -> float:
    """Calculate peak level.

    Args:
        audio_data: Audio samples as float32 array

    Returns:
        Peak absolute value (0.0 to 1.0 range for normalized audio)
    """
    if audio_data.size == 0:
        return 0.0

    # Flatten to mono if stereo
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    return float(np.max(np.abs(audio_data)))


def linear_to_db(linear: float, reference_db: float = -60.0) -> float:
    """Convert linear amplitude to decibels.

    Args:
        linear: Linear amplitude value
        reference_db: Minimum dB value for silence

    Returns:
        Value in decibels (reference_db for silence, 0 for max)
    """
    if linear <= 0:
        return reference_db

    db = 20 * np.log10(linear)
    return float(max(db, reference_db))


def db_to_normalized(db: float, reference_db: float = -60.0) -> float:
    """Convert dB to normalized 0-1 range for display.

    Args:
        db: Decibel value
        reference_db: Minimum dB (maps to 0.0)

    Returns:
        Normalized value (0.0 to 1.0)
    """
    if db <= reference_db:
        return 0.0
    if db >= 0:
        return 1.0
    return (db - reference_db) / (-reference_db)


class AudioMeter:
    """Calculates and tracks audio levels."""

    def __init__(self, reference_db: float = -60.0) -> None:
        """Initialize meter.

        Args:
            reference_db: Reference level for silence
        """
        self.reference_db = reference_db
        self._rms_db: float = reference_db
        self._peak_db: float = reference_db

    @property
    def rms_db(self) -> float:
        """Current RMS level in dB."""
        return self._rms_db

    @property
    def peak_db(self) -> float:
        """Current peak level in dB."""
        return self._peak_db

    @property
    def rms_normalized(self) -> float:
        """RMS level normalized to 0-1 range."""
        return db_to_normalized(self._rms_db, self.reference_db)

    @property
    def peak_normalized(self) -> float:
        """Peak level normalized to 0-1 range."""
        return db_to_normalized(self._peak_db, self.reference_db)

    def update(self, audio_data: NDArray[np.float32]) -> None:
        """Update meter with new audio data.

        Args:
            audio_data: Audio samples as float32 array
        """
        rms = calculate_rms(audio_data)
        peak = calculate_peak(audio_data)

        self._rms_db = linear_to_db(rms, self.reference_db)
        self._peak_db = linear_to_db(peak, self.reference_db)

    def reset(self) -> None:
        """Reset meter to silence levels."""
        self._rms_db = self.reference_db
        self._peak_db = self.reference_db
