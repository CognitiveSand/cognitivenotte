"""Tests for STT provider base class and utilities."""

from pathlib import Path

import pytest

from conot.stt.exceptions import AudioFormatError
from conot.stt.providers.base import SUPPORTED_FORMATS, BaseSTTProvider


class ConcreteProvider(BaseSTTProvider):
    """Concrete implementation of BaseSTTProvider for testing."""

    def transcribe(self, audio_path):
        pass

    def transcribe_stream(self, audio_stream):
        yield from []

    def detect_language(self, audio_path):
        return "en"

    def get_capabilities(self):
        pass

    def is_available(self):
        return True


class TestBaseSTTProvider:
    """Tests for BaseSTTProvider base class."""

    def test_supported_formats(self):
        assert ".wav" in SUPPORTED_FORMATS
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS
        assert ".ogg" in SUPPORTED_FORMATS
        assert ".m4a" in SUPPORTED_FORMATS
        assert ".webm" in SUPPORTED_FORMATS

    def test_validate_audio_file_not_found(self):
        provider = ConcreteProvider()
        with pytest.raises(AudioFormatError) as exc_info:
            provider.validate_audio_file(Path("/nonexistent/audio.wav"))
        assert "not found" in str(exc_info.value)

    def test_validate_audio_file_unsupported_format(self, tmp_path):
        provider = ConcreteProvider()
        # Create a file with unsupported extension
        audio_file = tmp_path / "audio.xyz"
        audio_file.write_bytes(b"fake audio data")

        with pytest.raises(AudioFormatError) as exc_info:
            provider.validate_audio_file(audio_file)
        assert "Unsupported audio format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)

    def test_validate_audio_file_success(self, tmp_path):
        provider = ConcreteProvider()
        # Create valid audio files
        for ext in [".wav", ".mp3", ".flac"]:
            audio_file = tmp_path / f"audio{ext}"
            audio_file.write_bytes(b"fake audio data")
            # Should not raise
            provider.validate_audio_file(audio_file)

    def test_validate_audio_file_case_insensitive(self, tmp_path):
        provider = ConcreteProvider()
        audio_file = tmp_path / "audio.WAV"
        audio_file.write_bytes(b"fake audio data")
        # Should not raise
        provider.validate_audio_file(audio_file)

    def test_ensure_model_loaded_default(self):
        provider = ConcreteProvider()
        # Default implementation does nothing
        provider._ensure_model_loaded()
        # Should not raise


class TestProviderProtocolCompliance:
    """Tests to verify providers comply with the protocol."""

    def test_concrete_provider_has_required_methods(self):
        provider = ConcreteProvider()
        assert hasattr(provider, "transcribe")
        assert hasattr(provider, "transcribe_stream")
        assert hasattr(provider, "detect_language")
        assert hasattr(provider, "get_capabilities")
        assert hasattr(provider, "is_available")
        assert callable(provider.transcribe)
        assert callable(provider.transcribe_stream)
        assert callable(provider.detect_language)
        assert callable(provider.get_capabilities)
        assert callable(provider.is_available)
