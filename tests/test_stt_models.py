"""Tests for STT data models."""

import json

import pytest

from conot.stt.models import (
    HardwareProfile,
    Language,
    ProviderCapabilities,
    ProviderTier,
    Segment,
    StreamingSegment,
    TranscriptionResult,
    Word,
    _format_srt_time,
)


class TestLanguage:
    """Tests for Language enum."""

    def test_values(self):
        assert Language.FRENCH.value == "fr"
        assert Language.ENGLISH.value == "en"
        assert Language.AUTO.value == "auto"

    def test_string_comparison(self):
        assert Language.FRENCH == "fr"
        assert Language.ENGLISH == "en"


class TestProviderTier:
    """Tests for ProviderTier enum."""

    def test_values(self):
        assert ProviderTier.ENTERPRISE.value == "enterprise"
        assert ProviderTier.STANDARD.value == "standard"
        assert ProviderTier.EDGE.value == "edge"


class TestWord:
    """Tests for Word dataclass."""

    def test_creation(self):
        word = Word(word="hello", start=0.0, end=0.5, confidence=0.95)
        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.95


class TestSegment:
    """Tests for Segment dataclass."""

    def test_creation_minimal(self):
        segment = Segment(
            start=0.0,
            end=5.0,
            speaker="SPEAKER_00",
            language="fr",
            text="Bonjour le monde",
            confidence=0.9,
        )
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.speaker == "SPEAKER_00"
        assert segment.language == "fr"
        assert segment.text == "Bonjour le monde"
        assert segment.confidence == 0.9
        assert segment.words == []

    def test_creation_with_words(self):
        words = [
            Word(word="Hello", start=0.0, end=0.3, confidence=0.95),
            Word(word="world", start=0.4, end=0.8, confidence=0.92),
        ]
        segment = Segment(
            start=0.0,
            end=1.0,
            speaker="SPEAKER_01",
            language="en",
            text="Hello world",
            confidence=0.93,
            words=words,
        )
        assert len(segment.words) == 2
        assert segment.words[0].word == "Hello"

    def test_to_dict(self):
        words = [Word(word="Test", start=0.0, end=0.5, confidence=0.9)]
        segment = Segment(
            start=0.0,
            end=1.0,
            speaker="SPEAKER_00",
            language="en",
            text="Test",
            confidence=0.85,
            words=words,
        )
        result = segment.to_dict()
        assert result["start"] == 0.0
        assert result["end"] == 1.0
        assert result["speaker"] == "SPEAKER_00"
        assert result["language"] == "en"
        assert result["text"] == "Test"
        assert result["confidence"] == 0.85
        assert len(result["words"]) == 1
        assert result["words"][0]["word"] == "Test"


class TestStreamingSegment:
    """Tests for StreamingSegment dataclass."""

    def test_creation(self):
        segment = StreamingSegment(
            segment_id="seg_1_abc123",
            start=0.0,
            end=2.0,
            text="Hello",
            language="en",
            is_final=False,
            confidence=0.8,
        )
        assert segment.segment_id == "seg_1_abc123"
        assert segment.is_final is False

    def test_to_dict(self):
        segment = StreamingSegment(
            segment_id="seg_2",
            start=1.0,
            end=3.0,
            text="World",
            language="fr",
            is_final=True,
            confidence=0.95,
        )
        result = segment.to_dict()
        assert result["segment_id"] == "seg_2"
        assert result["is_final"] is True
        assert result["language"] == "fr"


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    @pytest.fixture
    def sample_result(self):
        segments = [
            Segment(
                start=0.0,
                end=2.0,
                speaker="SPEAKER_00",
                language="fr",
                text="Bonjour",
                confidence=0.9,
            ),
            Segment(
                start=2.5,
                end=5.0,
                speaker="SPEAKER_01",
                language="en",
                text="Hello there",
                confidence=0.85,
            ),
            Segment(
                start=5.5,
                end=8.0,
                speaker="SPEAKER_00",
                language="fr",
                text="Comment ça va?",
                confidence=0.92,
            ),
        ]
        return TranscriptionResult(
            audio_file="test.wav",
            duration_s=10.0,
            languages_detected=["fr", "en"],
            speakers=["SPEAKER_00", "SPEAKER_01"],
            segments=segments,
        )

    def test_creation(self, sample_result):
        assert sample_result.audio_file == "test.wav"
        assert sample_result.duration_s == 10.0
        assert len(sample_result.languages_detected) == 2
        assert len(sample_result.speakers) == 2
        assert len(sample_result.segments) == 3

    def test_to_dict(self, sample_result):
        result = sample_result.to_dict()
        assert result["audio_file"] == "test.wav"
        assert result["duration_s"] == 10.0
        assert result["languages_detected"] == ["fr", "en"]
        assert result["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
        assert len(result["segments"]) == 3

    def test_to_json(self, sample_result):
        json_str = sample_result.to_json()
        parsed = json.loads(json_str)
        assert parsed["audio_file"] == "test.wav"
        assert len(parsed["segments"]) == 3

    def test_to_json_ensures_unicode(self, sample_result):
        json_str = sample_result.to_json()
        assert "Comment ça va?" in json_str  # Non-ASCII preserved

    def test_to_text(self, sample_result):
        text = sample_result.to_text()
        assert "[SPEAKER_00]" in text
        assert "[SPEAKER_01]" in text
        assert "Bonjour" in text
        assert "Hello there" in text
        assert "Comment ça va?" in text

    def test_to_text_groups_same_speaker(self):
        segments = [
            Segment(start=0.0, end=1.0, speaker="A", language="en", text="First", confidence=0.9),
            Segment(start=1.0, end=2.0, speaker="A", language="en", text="Second", confidence=0.9),
            Segment(start=2.0, end=3.0, speaker="B", language="en", text="Third", confidence=0.9),
        ]
        result = TranscriptionResult(
            audio_file="test.wav",
            duration_s=3.0,
            languages_detected=["en"],
            speakers=["A", "B"],
            segments=segments,
        )
        text = result.to_text()
        # Should only have [A] once before First and Second
        assert text.count("[A]") == 1
        assert text.count("[B]") == 1

    def test_to_srt(self, sample_result):
        srt = sample_result.to_srt()
        lines = srt.strip().split("\n")
        # SRT format: index, timestamp, text, blank line
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "Bonjour" in lines[2]

    def test_to_srt_format(self):
        segments = [
            Segment(
                start=0.0,
                end=2.5,
                speaker="SPEAKER_00",
                language="en",
                text="Hello",
                confidence=0.9,
            ),
        ]
        result = TranscriptionResult(
            audio_file="test.wav",
            duration_s=3.0,
            languages_detected=["en"],
            speakers=["SPEAKER_00"],
            segments=segments,
        )
        srt = result.to_srt()
        assert "00:00:00,000 --> 00:00:02,500" in srt
        assert "[SPEAKER_00] Hello" in srt


class TestFormatSrtTime:
    """Tests for SRT time formatting."""

    def test_zero(self):
        assert _format_srt_time(0.0) == "00:00:00,000"

    def test_seconds(self):
        assert _format_srt_time(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert _format_srt_time(65.123) == "00:01:05,123"

    def test_hours(self):
        # Use exact value to avoid floating-point precision issues
        assert _format_srt_time(3661.5) == "01:01:01,500"


class TestProviderCapabilities:
    """Tests for ProviderCapabilities dataclass."""

    def test_creation(self):
        caps = ProviderCapabilities(
            languages=["fr", "en"],
            supports_diarization=False,
            supports_word_timestamps=True,
            supports_streaming=True,
            streaming_latency_ms=1500,
            min_memory_gb=4.0,
            requires_gpu=False,
        )
        assert caps.languages == ["fr", "en"]
        assert caps.supports_diarization is False
        assert caps.supports_word_timestamps is True
        assert caps.supports_streaming is True
        assert caps.streaming_latency_ms == 1500
        assert caps.min_memory_gb == 4.0
        assert caps.requires_gpu is False


class TestHardwareProfile:
    """Tests for HardwareProfile dataclass."""

    def test_creation_with_gpu(self):
        profile = HardwareProfile(
            has_gpu=True,
            gpu_name="NVIDIA RTX 4090",
            vram_gb=24.0,
            ram_gb=64.0,
            recommended_tier=ProviderTier.ENTERPRISE,
        )
        assert profile.has_gpu is True
        assert profile.gpu_name == "NVIDIA RTX 4090"
        assert profile.vram_gb == 24.0
        assert profile.ram_gb == 64.0
        assert profile.recommended_tier == ProviderTier.ENTERPRISE

    def test_creation_without_gpu(self):
        profile = HardwareProfile(
            has_gpu=False,
            gpu_name=None,
            vram_gb=0.0,
            ram_gb=8.0,
            recommended_tier=ProviderTier.EDGE,
        )
        assert profile.has_gpu is False
        assert profile.gpu_name is None
        assert profile.recommended_tier == ProviderTier.EDGE

    def test_to_dict(self):
        profile = HardwareProfile(
            has_gpu=True,
            gpu_name="RTX 3080",
            vram_gb=10.0,
            ram_gb=32.0,
            recommended_tier=ProviderTier.STANDARD,
        )
        result = profile.to_dict()
        assert result["has_gpu"] is True
        assert result["gpu_name"] == "RTX 3080"
        assert result["vram_gb"] == 10.0
        assert result["ram_gb"] == 32.0
        assert result["recommended_tier"] == "standard"
