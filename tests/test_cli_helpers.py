"""Tests for CLI helper functions."""

import signal
from unittest.mock import MagicMock, patch

import pytest

from conot.cli import SignalHandler, _get_audio_device
from conot.devices import AudioDevice
from conot.exceptions import DeviceNotFoundError


class TestGetAudioDevice:
    """Tests for _get_audio_device helper."""

    @patch("conot.cli.get_device_by_id")
    def test_get_device_by_id_success(self, mock_get_device):
        """Test getting device by ID."""
        mock_device = AudioDevice(
            id=5,
            name="Test Mic",
            channels=2,
            sample_rate=48000.0,
            is_default=False,
            priority=50,
        )
        mock_get_device.return_value = mock_device

        result = _get_audio_device(5)

        assert result == mock_device
        mock_get_device.assert_called_once_with(5)

    @patch("conot.cli.get_device_by_id")
    @patch("conot.cli.console")
    def test_get_device_by_id_not_found(self, mock_console, mock_get_device):
        """Test getting device by ID when not found."""
        mock_get_device.side_effect = DeviceNotFoundError("Device 99 not found")

        result = _get_audio_device(99)

        assert result is None
        mock_console.print.assert_called()

    @patch("conot.cli.select_best_device")
    def test_get_device_auto_select(self, mock_select):
        """Test auto-selecting device when ID is None."""
        mock_device = AudioDevice(
            id=1,
            name="Default Mic",
            channels=1,
            sample_rate=44100.0,
            is_default=True,
            priority=100,
        )
        mock_select.return_value = mock_device

        result = _get_audio_device(None)

        assert result == mock_device
        mock_select.assert_called_once()

    @patch("conot.cli.select_best_device")
    @patch("conot.cli.console")
    def test_get_device_auto_select_no_devices(self, mock_console, mock_select):
        """Test auto-selection when no devices available."""
        mock_select.side_effect = DeviceNotFoundError("No audio devices found")

        result = _get_audio_device(None)

        assert result is None
        mock_console.print.assert_called()


class TestSignalHandler:
    """Tests for SignalHandler class."""

    def test_initial_state(self):
        """Test SignalHandler initial state."""
        handler = SignalHandler()
        assert handler.stop_requested is False
        assert handler.should_stop() is False

    def test_call_sets_stop_requested(self):
        """Test calling handler sets stop_requested."""
        handler = SignalHandler()
        handler(signal.SIGINT, None)
        assert handler.stop_requested is True
        assert handler.should_stop() is True

    def test_multiple_calls(self):
        """Test handler can be called multiple times."""
        handler = SignalHandler()
        handler(signal.SIGINT, None)
        handler(signal.SIGTERM, None)
        assert handler.stop_requested is True

    @patch("conot.cli.signal.signal")
    def test_install(self, mock_signal):
        """Test install registers signal handlers."""
        handler = SignalHandler()
        handler.install()

        # Should register for both SIGINT and SIGTERM
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list
        signals_registered = {call[0][0] for call in calls}
        assert signal.SIGINT in signals_registered
        assert signal.SIGTERM in signals_registered

    def test_should_stop_returns_bool(self):
        """Test should_stop returns boolean."""
        handler = SignalHandler()

        result = handler.should_stop()
        assert isinstance(result, bool)
        assert result is False

        handler.stop_requested = True
        result = handler.should_stop()
        assert isinstance(result, bool)
        assert result is True


class TestSignalHandlerIntegration:
    """Integration tests for SignalHandler with DRY pattern."""

    def test_handler_as_callable(self):
        """Test handler can be used as a callable predicate."""
        handler = SignalHandler()

        # Simulate a loop that checks should_stop
        iterations = 0
        while not handler.should_stop() and iterations < 5:
            iterations += 1
            if iterations == 3:
                handler(signal.SIGINT, None)

        # Should have stopped at iteration 3
        assert iterations == 3

    def test_multiple_handlers_independent(self):
        """Test multiple handlers are independent."""
        handler1 = SignalHandler()
        handler2 = SignalHandler()

        handler1(signal.SIGINT, None)

        assert handler1.stop_requested is True
        assert handler2.stop_requested is False
