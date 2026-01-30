"""Tests for audio meter module."""

import numpy as np
import pytest

from conot.audio_meter import (
    AudioMeter,
    calculate_peak,
    calculate_rms,
    db_to_normalized,
    linear_to_db,
)


class TestCalculateRms:
    """Tests for RMS calculation."""

    def test_silence(self):
        """Test RMS of silence is zero."""
        silence = np.zeros(1024, dtype=np.float32)
        assert calculate_rms(silence) == 0.0

    def test_full_scale_sine(self):
        """Test RMS of full-scale sine wave."""
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        rms = calculate_rms(sine)
        # RMS of sine wave is 1/sqrt(2) ≈ 0.707
        assert abs(rms - 0.707) < 0.01

    def test_full_scale_dc(self):
        """Test RMS of full-scale DC signal."""
        dc = np.ones(1024, dtype=np.float32)
        assert abs(calculate_rms(dc) - 1.0) < 0.001

    def test_stereo_input(self):
        """Test RMS handles stereo input."""
        stereo = np.ones((1024, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        stereo[:, 1] = 1.0
        rms = calculate_rms(stereo)
        # Average of channels: (0.5 + 1.0) / 2 = 0.75
        assert abs(rms - 0.75) < 0.01

    def test_empty_array(self):
        """Test RMS of empty array is zero."""
        empty = np.array([], dtype=np.float32)
        assert calculate_rms(empty) == 0.0


class TestCalculatePeak:
    """Tests for peak calculation."""

    def test_silence(self):
        """Test peak of silence is zero."""
        silence = np.zeros(1024, dtype=np.float32)
        assert calculate_peak(silence) == 0.0

    def test_full_scale(self):
        """Test peak of full-scale signal."""
        signal = np.ones(1024, dtype=np.float32)
        assert calculate_peak(signal) == 1.0

    def test_negative_peak(self):
        """Test peak detection of negative values."""
        signal = np.zeros(1024, dtype=np.float32)
        signal[512] = -0.8
        assert abs(calculate_peak(signal) - 0.8) < 0.001


class TestLinearToDb:
    """Tests for linear to dB conversion."""

    def test_unity_is_zero_db(self):
        """Test unity (1.0) converts to 0 dB."""
        assert abs(linear_to_db(1.0) - 0.0) < 0.001

    def test_half_is_minus_six_db(self):
        """Test 0.5 converts to approximately -6 dB."""
        assert abs(linear_to_db(0.5) - (-6.02)) < 0.1

    def test_zero_is_reference(self):
        """Test zero converts to reference dB."""
        assert linear_to_db(0.0, reference_db=-60.0) == -60.0

    def test_negative_is_reference(self):
        """Test negative values convert to reference dB."""
        assert linear_to_db(-0.5, reference_db=-60.0) == -60.0


class TestDbToNormalized:
    """Tests for dB to normalized conversion."""

    def test_zero_db_is_one(self):
        """Test 0 dB converts to 1.0."""
        assert db_to_normalized(0.0) == 1.0

    def test_reference_is_zero(self):
        """Test reference dB converts to 0.0."""
        assert db_to_normalized(-60.0, reference_db=-60.0) == 0.0

    def test_midpoint(self):
        """Test midpoint conversion."""
        normalized = db_to_normalized(-30.0, reference_db=-60.0)
        assert abs(normalized - 0.5) < 0.001


class TestAudioMeter:
    """Tests for AudioMeter class."""

    def test_initial_state(self):
        """Test meter starts at silence."""
        meter = AudioMeter(reference_db=-60.0)
        assert meter.rms_db == -60.0
        assert meter.peak_db == -60.0

    def test_update_with_signal(self):
        """Test meter updates with signal."""
        meter = AudioMeter(reference_db=-60.0)
        signal = np.ones(1024, dtype=np.float32) * 0.5
        meter.update(signal)
        assert meter.rms_db > -60.0
        assert meter.peak_db > -60.0

    def test_reset(self):
        """Test meter reset."""
        meter = AudioMeter(reference_db=-60.0)
        signal = np.ones(1024, dtype=np.float32)
        meter.update(signal)
        meter.reset()
        assert meter.rms_db == -60.0
        assert meter.peak_db == -60.0
