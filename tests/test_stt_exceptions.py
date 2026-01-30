"""Tests for STT exceptions."""

import pytest

from conot.exceptions import ConotError
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


class TestSTTExceptionHierarchy:
    """Tests for STT exception class hierarchy."""

    def test_stt_error_inherits_from_conot_error(self):
        assert issubclass(STTError, ConotError)

    def test_provider_errors_inherit_from_stt_error(self):
        assert issubclass(ProviderNotFoundError, STTError)
        assert issubclass(ProviderNotAvailableError, STTError)
        assert issubclass(NoProviderAvailableError, STTError)

    def test_transcription_error_inherits_from_stt_error(self):
        assert issubclass(TranscriptionError, STTError)

    def test_diarization_error_inherits_from_stt_error(self):
        assert issubclass(DiarizationError, STTError)

    def test_vad_error_inherits_from_stt_error(self):
        assert issubclass(VADError, STTError)

    def test_audio_format_error_inherits_from_stt_error(self):
        assert issubclass(AudioFormatError, STTError)

    def test_model_not_found_error_inherits_from_stt_error(self):
        assert issubclass(ModelNotFoundError, STTError)

    def test_hardware_detection_error_inherits_from_stt_error(self):
        assert issubclass(HardwareDetectionError, STTError)


class TestSTTExceptionMessages:
    """Tests for STT exception message handling."""

    def test_stt_error_message(self):
        error = STTError("Test error message")
        assert str(error) == "Test error message"

    def test_provider_not_found_message(self):
        error = ProviderNotFoundError("Provider 'xyz' not found")
        assert "xyz" in str(error)

    def test_transcription_error_message(self):
        error = TranscriptionError("Transcription failed: timeout")
        assert "timeout" in str(error)

    def test_exception_can_be_raised_and_caught(self):
        with pytest.raises(STTError):
            raise TranscriptionError("Test")

        with pytest.raises(ConotError):
            raise STTError("Test")

    def test_exception_chaining(self):
        original = ValueError("Original error")
        error = TranscriptionError("Wrapper error")
        error.__cause__ = original

        assert error.__cause__ is original
