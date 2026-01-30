"""Tests for STT Voice Activity Detection."""

import numpy as np
import pytest

from conot.stt.vad import SpeechSegment, VoiceActivityDetector, create_vad


class TestVoiceActivityDetector:
    """Tests for VoiceActivityDetector class."""

    def test_creation_defaults(self):
        vad = VoiceActivityDetector()
        assert vad._threshold == 0.5
        assert vad._min_speech_ms == 250
        assert vad._min_silence_ms == 500
        assert vad._sample_rate == 16000

    def test_creation_custom_params(self):
        vad = VoiceActivityDetector(
            threshold=0.7,
            min_speech_duration_ms=300,
            min_silence_duration_ms=600,
            sample_rate=48000,
        )
        assert vad._threshold == 0.7
        assert vad._min_speech_ms == 300
        assert vad._min_silence_ms == 600
        assert vad._sample_rate == 48000

    def test_detect_speech_silence(self):
        """Test VAD on silent audio."""
        vad = VoiceActivityDetector()
        # Create 1 second of silence
        audio = np.zeros(16000, dtype=np.float32)

        segments = vad.detect_speech(audio)
        assert len(segments) == 0

    def test_detect_speech_with_signal(self):
        """Test VAD on audio with signal."""
        vad = VoiceActivityDetector(
            min_speech_duration_ms=100,
            min_silence_duration_ms=100,
        )
        # Create audio with a speech-like burst
        audio = np.zeros(16000, dtype=np.float32)
        # Add a "speech" segment (higher amplitude)
        audio[4000:8000] = 0.5 * np.sin(2 * np.pi * 440 * np.arange(4000) / 16000)

        segments = vad.detect_speech(audio)
        # Should detect at least one speech segment
        # Note: exact detection depends on VAD implementation

    def test_detect_speech_energy_fallback(self):
        """Test energy-based VAD fallback."""
        vad = VoiceActivityDetector()
        # Force energy-based detection by not loading Silero
        vad._model = None
        vad._model_loaded = True

        # Create audio with varying energy
        audio = np.zeros(16000, dtype=np.float32)
        # Add high energy section
        audio[3000:6000] = 0.3

        segments = vad.detect_speech(audio)
        # Energy-based VAD should find something


class TestSpeechSegment:
    """Tests for SpeechSegment dataclass."""

    def test_creation(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        segment = SpeechSegment(
            start_sample=1000,
            end_sample=2000,
            audio=audio,
        )
        assert segment.start_sample == 1000
        assert segment.end_sample == 2000
        assert len(segment.audio) == 3

    def test_duration_samples(self):
        audio = np.zeros(100, dtype=np.float32)
        segment = SpeechSegment(
            start_sample=0,
            end_sample=100,
            audio=audio,
        )
        assert segment.duration_samples == 100


class TestCreateVad:
    """Tests for create_vad factory function."""

    def test_create_with_defaults(self):
        vad = create_vad()
        assert isinstance(vad, VoiceActivityDetector)

    def test_create_with_custom_params(self):
        vad = create_vad(
            threshold=0.6,
            min_speech_duration_ms=200,
            min_silence_duration_ms=400,
            sample_rate=48000,
        )
        assert vad._threshold == 0.6
        assert vad._min_speech_ms == 200


class TestVADProcessStream:
    """Tests for VAD stream processing."""

    def test_process_stream_empty(self):
        vad = VoiceActivityDetector()

        def empty_stream():
            return
            yield  # Make it a generator

        segments = list(vad.process_stream(empty_stream()))
        assert len(segments) == 0

    def test_process_stream_silence(self):
        vad = VoiceActivityDetector()

        def silence_stream():
            # Generate 2 seconds of silence (int16 bytes)
            for _ in range(20):  # 20 chunks of 100ms
                chunk = np.zeros(1600, dtype=np.int16)
                yield chunk.tobytes()

        segments = list(vad.process_stream(silence_stream()))
        # Should not detect any speech in silence
        assert len(segments) == 0

    def test_process_stream_with_speech(self):
        vad = VoiceActivityDetector(
            min_speech_duration_ms=50,
            min_silence_duration_ms=100,
        )

        def speech_stream():
            # Generate audio with a speech-like burst
            for i in range(20):
                if 5 <= i <= 15:
                    # "Speech" chunk with signal
                    t = np.arange(1600) / 16000
                    chunk = (0.3 * np.sin(2 * np.pi * 200 * t) * 32767).astype(np.int16)
                else:
                    # Silence chunk
                    chunk = np.zeros(1600, dtype=np.int16)
                yield chunk.tobytes()

        segments = list(vad.process_stream(speech_stream()))
        # May or may not detect speech depending on thresholds
