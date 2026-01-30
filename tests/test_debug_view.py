"""Tests for debug view component."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conot.debug_view import DebugView
from conot.devices import AudioDevice


@pytest.fixture
def mock_device() -> AudioDevice:
    """Create a mock audio device."""
    return AudioDevice(
        id=1,
        name="Test Microphone",
        channels=2,
        sample_rate=48000.0,
        is_default=True,
        priority=100,
    )


class TestDebugViewCreation:
    """Tests for DebugView initialization."""

    def test_creation_minimal(self, mock_device):
        """Test DebugView can be created with minimal params."""
        view = DebugView(device=mock_device)
        assert view._device == mock_device

    def test_creation_with_reference_db(self, mock_device):
        """Test DebugView with custom reference dB."""
        view = DebugView(device=mock_device, reference_db=-50.0)
        assert view._meter.reference_db == -50.0

    def test_creation_with_sample_rate(self, mock_device):
        """Test DebugView with custom sample rate."""
        view = DebugView(device=mock_device, sample_rate=44100)
        assert view._sample_rate == 44100

    def test_creation_with_chunk_size(self, mock_device):
        """Test DebugView with custom chunk size."""
        view = DebugView(device=mock_device, chunk_size=2048)
        assert view._chunk_size == 2048

    def test_creation_all_params(self, mock_device):
        """Test DebugView with all parameters."""
        view = DebugView(
            device=mock_device,
            reference_db=-55.0,
            sample_rate=44100,
            chunk_size=1024,
        )
        assert view._sample_rate == 44100
        assert view._chunk_size == 1024
        assert view._meter.reference_db == -55.0


class TestDebugViewUpdate:
    """Tests for DebugView update method."""

    def test_update_audio_data(self, mock_device):
        """Test update with audio data."""
        view = DebugView(device=mock_device)
        audio = np.random.randn(1024).astype(np.float32) * 0.1

        view.update(audio_data=audio)
        # Meter should have been updated
        assert view._meter.rms_db != float("-inf")

    def test_update_duration(self, mock_device):
        """Test update with duration."""
        view = DebugView(device=mock_device)
        view.update(duration=123.5)
        assert view._duration == 123.5

    def test_update_is_recording(self, mock_device):
        """Test update with recording status."""
        view = DebugView(device=mock_device)
        assert view._is_recording is False

        view.update(is_recording=True)
        assert view._is_recording is True

    def test_update_extra_info(self, mock_device):
        """Test update with extra info (new feature)."""
        view = DebugView(device=mock_device)
        assert view._extra_info == ""

        view.update(extra_info="RMS=0.01 | Speech=yes")
        assert view._extra_info == "RMS=0.01 | Speech=yes"

    def test_update_multiple_fields(self, mock_device):
        """Test update with multiple fields at once."""
        view = DebugView(device=mock_device)
        audio = np.random.randn(1024).astype(np.float32) * 0.1

        view.update(
            audio_data=audio,
            duration=10.5,
            is_recording=True,
            extra_info="Testing",
        )

        assert view._duration == 10.5
        assert view._is_recording is True
        assert view._extra_info == "Testing"


class TestDebugViewRender:
    """Tests for DebugView rendering."""

    def test_render_contains_device_name(self, mock_device):
        """Test render output contains device name."""
        view = DebugView(device=mock_device)
        panel = view._render()

        # Panel content should contain device name
        content_str = str(panel.renderable)
        assert "Test Microphone" in content_str

    def test_render_contains_audio_settings(self, mock_device):
        """Test render output contains audio settings (sample rate, chunk size)."""
        view = DebugView(
            device=mock_device,
            sample_rate=16000,
            chunk_size=1600,
        )
        panel = view._render()

        content_str = str(panel.renderable)
        # Should show sample rate
        assert "16000" in content_str
        # Should show chunk size
        assert "1600" in content_str
        # Should show chunk duration in ms
        assert "100ms" in content_str

    def test_render_contains_status(self, mock_device):
        """Test render output contains recording status."""
        view = DebugView(device=mock_device)

        # Not recording
        panel = view._render()
        content_str = str(panel.renderable)
        assert "Stopped" in content_str

        # Recording
        view.update(is_recording=True)
        panel = view._render()
        content_str = str(panel.renderable)
        assert "Recording" in content_str

    def test_render_contains_extra_info(self, mock_device):
        """Test render output contains extra info when set."""
        view = DebugView(device=mock_device)
        view.update(extra_info="5 segments detected")

        panel = view._render()
        content_str = str(panel.renderable)
        assert "5 segments detected" in content_str

    def test_render_contains_meters(self, mock_device):
        """Test render output contains VU meters."""
        view = DebugView(device=mock_device)
        panel = view._render()

        content_str = str(panel.renderable)
        assert "RMS" in content_str
        assert "Peak" in content_str
        assert "dB" in content_str

    def test_render_default_marker(self, mock_device):
        """Test render shows default marker for default device."""
        view = DebugView(device=mock_device)  # mock_device.is_default = True
        panel = view._render()

        content_str = str(panel.renderable)
        assert "(default)" in content_str


class TestDebugViewFormatting:
    """Tests for DebugView formatting utilities."""

    def test_format_duration_zero(self, mock_device):
        """Test duration formatting for zero."""
        view = DebugView(device=mock_device)
        assert view._format_duration(0) == "00:00:00"

    def test_format_duration_seconds(self, mock_device):
        """Test duration formatting for seconds."""
        view = DebugView(device=mock_device)
        assert view._format_duration(45) == "00:00:45"

    def test_format_duration_minutes(self, mock_device):
        """Test duration formatting for minutes."""
        view = DebugView(device=mock_device)
        assert view._format_duration(125) == "00:02:05"

    def test_format_duration_hours(self, mock_device):
        """Test duration formatting for hours."""
        view = DebugView(device=mock_device)
        assert view._format_duration(3661) == "01:01:01"


class TestDebugViewContextManager:
    """Tests for DebugView context manager."""

    @patch("conot.debug_view.Live")
    def test_context_manager_start_stop(self, mock_live_class, mock_device):
        """Test context manager starts and stops Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        view = DebugView(device=mock_device)

        with view:
            mock_live.start.assert_called_once()

        mock_live.stop.assert_called_once()

    @patch("conot.debug_view.Live")
    def test_start_creates_live(self, mock_live_class, mock_device):
        """Test start() creates Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        view = DebugView(device=mock_device)
        view.start()

        assert view._live is not None
        mock_live.start.assert_called_once()

        view.stop()

    @patch("conot.debug_view.Live")
    def test_stop_clears_live(self, mock_live_class, mock_device):
        """Test stop() clears Live display."""
        mock_live = MagicMock()
        mock_live_class.return_value = mock_live

        view = DebugView(device=mock_device)
        view.start()
        view.stop()

        assert view._live is None
        mock_live.stop.assert_called_once()


class TestDebugViewMeterBar:
    """Tests for meter bar creation."""

    def test_create_meter_bar_low_level(self, mock_device):
        """Test meter bar for low audio level."""
        view = DebugView(device=mock_device)
        bar = view._create_meter_bar(0.2, -40.0, "RMS")

        bar_str = str(bar)
        assert "RMS" in bar_str
        assert "-40.0" in bar_str

    def test_create_meter_bar_high_level(self, mock_device):
        """Test meter bar for high audio level."""
        view = DebugView(device=mock_device)
        bar = view._create_meter_bar(0.95, -3.0, "Peak")

        bar_str = str(bar)
        assert "Peak" in bar_str
        assert "-3.0" in bar_str

    def test_create_meter_bar_custom_width(self, mock_device):
        """Test meter bar with custom width."""
        view = DebugView(device=mock_device)
        bar = view._create_meter_bar(0.5, -20.0, "Test", width=50)

        # Bar should have approximately 50 characters of blocks
        bar_str = str(bar)
        assert len(bar_str) > 50  # Includes label and dB value


class TestDebugViewChunkSizeDisplay:
    """Tests for chunk size display in debug view."""

    def test_chunk_duration_calculation_16khz(self, mock_device):
        """Test chunk duration calculation at 16kHz."""
        view = DebugView(
            device=mock_device,
            sample_rate=16000,
            chunk_size=1600,  # 100ms at 16kHz
        )
        panel = view._render()
        content_str = str(panel.renderable)
        assert "100ms" in content_str

    def test_chunk_duration_calculation_48khz(self, mock_device):
        """Test chunk duration calculation at 48kHz."""
        view = DebugView(
            device=mock_device,
            sample_rate=48000,
            chunk_size=4800,  # 100ms at 48kHz
        )
        panel = view._render()
        content_str = str(panel.renderable)
        assert "100ms" in content_str

    def test_chunk_duration_50ms(self, mock_device):
        """Test chunk duration for 50ms chunks."""
        view = DebugView(
            device=mock_device,
            sample_rate=16000,
            chunk_size=800,  # 50ms at 16kHz
        )
        panel = view._render()
        content_str = str(panel.renderable)
        assert "50ms" in content_str
