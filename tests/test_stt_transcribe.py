"""Tests for STT transcription orchestrator."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conot.stt.exceptions import TranscriptionError
from conot.stt.models import (
    ProviderCapabilities,
    Segment,
    StreamingSegment,
    TranscriptionResult,
)
from conot.stt.transcribe import (
    LiveTranscriber,
    create_live_transcriber,
    transcribe_audio,
    transcribe_stream,
)


class MockProvider:
    """Mock STT provider for testing."""

    def __init__(self):
        self.transcribe_called = False
        self.transcribe_stream_called = False

    def transcribe(self, audio_path):
        self.transcribe_called = True
        return TranscriptionResult(
            audio_file=str(audio_path),
            duration_s=10.0,
            languages_detected=["en"],
            speakers=[],
            segments=[
                Segment(
                    start=0.0,
                    end=5.0,
                    speaker="",
                    language="en",
                    text="Hello world",
                    confidence=0.9,
                ),
            ],
        )

    def transcribe_stream(self, audio_stream):
        self.transcribe_stream_called = True
        # Consume the stream
        for _ in audio_stream:
            pass
        yield StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Streaming text",
            language="en",
            is_final=True,
            confidence=0.85,
        )

    def get_capabilities(self):
        return ProviderCapabilities(
            languages=["en", "fr"],
            supports_diarization=False,
            supports_word_timestamps=True,
            supports_streaming=True,
            streaming_latency_ms=1500,
            min_memory_gb=4.0,
            requires_gpu=False,
        )

    def is_available(self):
        return True


class TestTranscribeAudio:
    """Tests for transcribe_audio function."""

    def test_file_not_found(self):
        with pytest.raises(TranscriptionError) as exc_info:
            transcribe_audio(Path("/nonexistent/audio.wav"))
        assert "not found" in str(exc_info.value)

    @patch("conot.stt.transcribe.get_provider")
    def test_transcribe_with_provider(self, mock_get_provider, tmp_path):
        # Create a dummy audio file
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        provider = MockProvider()
        mock_get_provider.return_value = provider

        result = transcribe_audio(audio_file, enable_diarization=False)

        assert provider.transcribe_called
        assert isinstance(result, TranscriptionResult)
        assert result.audio_file == str(audio_file)

    @patch("conot.stt.transcribe.get_provider")
    def test_transcribe_with_specific_provider(self, mock_get_provider, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        provider = MockProvider()
        mock_get_provider.return_value = provider

        transcribe_audio(
            audio_file,
            provider="faster-whisper",
            enable_diarization=False,
        )

        mock_get_provider.assert_called_with("faster-whisper")

    @patch("conot.stt.transcribe.create_diarizer")
    @patch("conot.stt.transcribe.get_provider")
    def test_transcribe_with_diarization(
        self, mock_get_provider, mock_create_diarizer, tmp_path
    ):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        provider = MockProvider()
        mock_get_provider.return_value = provider

        mock_diarizer = MagicMock()
        mock_diarizer.is_available.return_value = True
        mock_diarizer.diarize.return_value = []
        mock_create_diarizer.return_value = mock_diarizer

        transcribe_audio(audio_file, enable_diarization=True)

        mock_diarizer.diarize.assert_called_once()

    @patch("conot.stt.transcribe.create_diarizer")
    @patch("conot.stt.transcribe.get_provider")
    def test_transcribe_diarization_not_available(
        self, mock_get_provider, mock_create_diarizer, tmp_path
    ):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        provider = MockProvider()
        mock_get_provider.return_value = provider

        mock_diarizer = MagicMock()
        mock_diarizer.is_available.return_value = False
        mock_create_diarizer.return_value = mock_diarizer

        result = transcribe_audio(audio_file, enable_diarization=True)

        # Should still succeed, just without diarization
        assert isinstance(result, TranscriptionResult)
        mock_diarizer.diarize.assert_not_called()

    @patch("conot.stt.transcribe.get_provider")
    def test_progress_callback(self, mock_get_provider, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        provider = MockProvider()
        mock_get_provider.return_value = provider

        progress_calls = []

        def on_progress(stage, pct):
            progress_calls.append((stage, pct))

        transcribe_audio(
            audio_file,
            enable_diarization=False,
            progress_callback=on_progress,
        )

        assert len(progress_calls) > 0
        stages = [call[0] for call in progress_calls]
        assert "Loading model" in stages
        assert "Transcribing" in stages


class TestTranscribeStream:
    """Tests for transcribe_stream function."""

    @patch("conot.stt.transcribe.get_provider")
    def test_stream_transcription(self, mock_get_provider):
        provider = MockProvider()
        mock_get_provider.return_value = provider

        def audio_generator():
            yield b"\x00" * 3200
            yield b"\x00" * 3200

        segments = list(transcribe_stream(audio_generator()))

        assert provider.transcribe_stream_called
        assert len(segments) == 1
        assert segments[0].text == "Streaming text"


class TestLiveTranscriber:
    """Tests for LiveTranscriber class."""

    def test_creation(self):
        transcriber = LiveTranscriber()
        assert transcriber._running is False
        assert transcriber._provider is None

    def test_creation_with_options(self):
        callback = MagicMock()
        transcriber = LiveTranscriber(
            provider="faster-whisper",
            sample_rate=48000,
            device_id=1,
            callback=callback,
        )
        assert transcriber._provider_name == "faster-whisper"
        assert transcriber._sample_rate == 48000
        assert transcriber._device_id == 1
        assert transcriber._callback == callback

    def test_is_running_property(self):
        transcriber = LiveTranscriber()
        assert transcriber.is_running is False

    @patch("conot.stt.transcribe.get_provider")
    @patch("conot.stt.transcribe.create_streaming_transcriber")
    def test_start_without_sounddevice(
        self, mock_create_transcriber, mock_get_provider
    ):
        provider = MockProvider()
        mock_get_provider.return_value = provider

        mock_orchestrator = MagicMock()
        mock_create_transcriber.return_value = mock_orchestrator

        transcriber = LiveTranscriber()

        # Starting will fail without sounddevice, but we can test the setup
        with patch("sounddevice.InputStream") as mock_input:
            mock_stream = MagicMock()
            mock_input.return_value = mock_stream

            transcriber.start()

            assert transcriber._running is True
            mock_orchestrator.start.assert_called_once()
            mock_stream.start.assert_called_once()

            transcriber.stop()

    def test_stop_when_not_running(self):
        transcriber = LiveTranscriber()
        segments = transcriber.stop()
        assert segments == []


class TestCreateLiveTranscriber:
    """Tests for create_live_transcriber factory."""

    def test_create_default(self):
        transcriber = create_live_transcriber()
        assert isinstance(transcriber, LiveTranscriber)

    def test_create_with_options(self):
        callback = MagicMock()
        transcriber = create_live_transcriber(
            provider="whisper-cpp",
            sample_rate=44100,
            device_id=2,
            callback=callback,
        )
        assert transcriber._provider_name == "whisper-cpp"
        assert transcriber._sample_rate == 44100
        assert transcriber._device_id == 2
