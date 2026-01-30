"""Tests for device detection module."""

import pytest

from conot.devices import (
    AudioDevice,
    _calculate_priority,
    _should_exclude,
    select_best_device,
)
from conot.exceptions import DeviceNotFoundError


class TestPriorityCalculation:
    """Tests for device priority calculation."""

    def test_usb_microphone_high_priority(self):
        """Test USB microphones get high priority."""
        priority = _calculate_priority("Blue Yeti USB Microphone", False, 48000.0)
        assert priority >= 100

    def test_builtin_mic_medium_priority(self):
        """Test built-in mics get medium priority."""
        priority = _calculate_priority("Built-in Microphone", False, 48000.0)
        assert 50 <= priority < 100

    def test_default_device_bonus(self):
        """Test default device gets priority bonus."""
        priority_default = _calculate_priority("Generic Mic", True, 48000.0)
        priority_non_default = _calculate_priority("Generic Mic", False, 48000.0)
        assert priority_default > priority_non_default

    def test_standard_sample_rate_bonus(self):
        """Test standard sample rates get bonus."""
        priority_48k = _calculate_priority("Mic", False, 48000.0)
        priority_44k = _calculate_priority("Mic", False, 44100.0)
        priority_96k = _calculate_priority("Mic", False, 96000.0)
        assert priority_48k == priority_44k
        assert priority_48k > priority_96k

    def test_bluetooth_device_priority(self):
        """Test Bluetooth devices get good priority."""
        priority = _calculate_priority("OpenRun Pro by Shokz", False, 48000.0)
        assert priority >= 80  # Bluetooth bonus

    def test_bluez_source_priority(self):
        """Test bluez sources get Bluetooth priority."""
        priority = _calculate_priority("bluez_input.A8:F5:E1:0B:25:3C", False, 48000.0)
        assert priority >= 80


class TestDeviceExclusion:
    """Tests for device exclusion."""

    @pytest.mark.parametrize(
        "name",
        [
            "HDMI Output",
            "Monitor of Built-in Audio",
            "Loopback Device",
            "Digital Output",
            "SPDIF Output",
        ],
    )
    def test_excluded_devices(self, name: str):
        """Test that output/monitor devices are excluded."""
        assert _should_exclude(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "Blue Yeti",
            "Built-in Microphone",
            "USB Audio Device",
        ],
    )
    def test_included_devices(self, name: str):
        """Test that input devices are not excluded."""
        assert _should_exclude(name) is False


class TestDeviceSelection:
    """Tests for best device selection."""

    def test_select_usb_over_builtin(self):
        """Test USB device selected over built-in."""
        devices = [
            AudioDevice(0, "Built-in Microphone", 2, 48000.0, True, 50),
            AudioDevice(1, "Blue Yeti USB", 2, 48000.0, False, 100),
        ]
        best = select_best_device(devices)
        assert best.id == 1

    def test_select_default_when_equal_priority(self):
        """Test default device selected when priorities equal."""
        devices = [
            AudioDevice(0, "Generic Mic A", 2, 48000.0, True, 40),
            AudioDevice(1, "Generic Mic B", 2, 48000.0, False, 10),
        ]
        best = select_best_device(devices)
        assert best.id == 0

    def test_no_devices_raises_error(self):
        """Test error raised when no devices available."""
        with pytest.raises(DeviceNotFoundError):
            select_best_device([])

    def test_select_usb_over_bluetooth(self):
        """Test USB device selected over Bluetooth."""
        devices = [
            AudioDevice(-1, "OpenRun Pro by Shokz", 1, 48000.0, False, 90),
            AudioDevice(1, "Blue Yeti USB", 2, 48000.0, False, 110),
        ]
        best = select_best_device(devices)
        assert best.id == 1

    def test_select_bluetooth_over_builtin(self):
        """Test Bluetooth device selected over built-in."""
        devices = [
            AudioDevice(0, "Built-in Microphone", 2, 48000.0, True, 60),
            AudioDevice(-1, "OpenRun Pro by Shokz", 1, 48000.0, False, 90),
        ]
        best = select_best_device(devices)
        assert best.id == -1
