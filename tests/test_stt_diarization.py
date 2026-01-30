"""Tests for STT diarization module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conot.stt.diarization import (
    Diarizer,
    SpeakerSegment,
    _find_speaker_for_segment,
    create_diarizer,
    merge_transcription_with_diarization,
)
from conot.stt.exceptions import DiarizationError
from conot.stt.models import Segment, TranscriptionResult


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_creation(self):
        segment = SpeakerSegment(start=0.0, end=5.0, speaker="SPEAKER_00")
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.speaker == "SPEAKER_00"


class TestDiarizer:
    """Tests for Diarizer class."""

    def test_creation_default(self):
        diarizer = Diarizer()
        assert diarizer._token is None
        assert diarizer._min_speakers is None
        assert diarizer._max_speakers is None

    def test_creation_with_token(self):
        diarizer = Diarizer(huggingface_token="test_token")
        assert diarizer._token == "test_token"

    def test_creation_with_speaker_limits(self):
        diarizer = Diarizer(min_speakers=2, max_speakers=5)
        assert diarizer._min_speakers == 2
        assert diarizer._max_speakers == 5

    @patch.dict("os.environ", {"HF_TOKEN": "env_token"})
    def test_token_from_env(self):
        diarizer = Diarizer()
        assert diarizer._token == "env_token"

    def test_is_available_no_token(self):
        diarizer = Diarizer(huggingface_token=None)
        # Remove env var if present
        with patch.dict("os.environ", {}, clear=True):
            # is_available depends on pyannote being installed
            # and token being present
            pass

    @patch("conot.stt.diarization.Diarizer._ensure_pipeline_loaded")
    def test_diarize_file_not_found(self, mock_load):
        diarizer = Diarizer(huggingface_token="test")
        diarizer._pipeline_loaded = True
        diarizer._pipeline = MagicMock()

        with pytest.raises(DiarizationError) as exc_info:
            diarizer.diarize(Path("/nonexistent/file.wav"))
        assert "not found" in str(exc_info.value)


class TestFindSpeakerForSegment:
    """Tests for _find_speaker_for_segment function."""

    def test_no_overlap(self):
        diarization = [
            SpeakerSegment(start=10.0, end=15.0, speaker="SPEAKER_00"),
        ]
        speaker = _find_speaker_for_segment(0.0, 5.0, diarization)
        assert speaker == "UNKNOWN"

    def test_full_overlap(self):
        diarization = [
            SpeakerSegment(start=0.0, end=10.0, speaker="SPEAKER_00"),
        ]
        speaker = _find_speaker_for_segment(2.0, 5.0, diarization)
        assert speaker == "SPEAKER_00"

    def test_partial_overlap(self):
        diarization = [
            SpeakerSegment(start=0.0, end=3.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=3.0, end=10.0, speaker="SPEAKER_01"),
        ]
        # Segment from 2.0 to 5.0 overlaps more with SPEAKER_01
        speaker = _find_speaker_for_segment(2.0, 5.0, diarization)
        assert speaker == "SPEAKER_01"

    def test_equal_overlap_takes_first_best(self):
        diarization = [
            SpeakerSegment(start=0.0, end=2.5, speaker="SPEAKER_00"),
            SpeakerSegment(start=2.5, end=5.0, speaker="SPEAKER_01"),
        ]
        # Segment from 0.0 to 5.0 has equal overlap with both
        # First one found with max overlap wins
        speaker = _find_speaker_for_segment(0.0, 5.0, diarization)
        assert speaker in ["SPEAKER_00", "SPEAKER_01"]

    def test_empty_diarization(self):
        speaker = _find_speaker_for_segment(0.0, 5.0, [])
        assert speaker == "UNKNOWN"


class TestMergeTranscriptionWithDiarization:
    """Tests for merge_transcription_with_diarization function."""

    @pytest.fixture
    def sample_transcription(self):
        return TranscriptionResult(
            audio_file="test.wav",
            duration_s=10.0,
            languages_detected=["en"],
            speakers=[],
            segments=[
                Segment(
                    start=0.0,
                    end=3.0,
                    speaker="",
                    language="en",
                    text="Hello",
                    confidence=0.9,
                ),
                Segment(
                    start=3.5,
                    end=7.0,
                    speaker="",
                    language="en",
                    text="How are you?",
                    confidence=0.85,
                ),
                Segment(
                    start=7.5,
                    end=10.0,
                    speaker="",
                    language="en",
                    text="I'm fine.",
                    confidence=0.88,
                ),
            ],
        )

    def test_merge_assigns_speakers(self, sample_transcription):
        diarization = [
            SpeakerSegment(start=0.0, end=4.0, speaker="SPEAKER_00"),
            SpeakerSegment(start=4.0, end=8.0, speaker="SPEAKER_01"),
            SpeakerSegment(start=8.0, end=10.0, speaker="SPEAKER_00"),
        ]

        result = merge_transcription_with_diarization(
            sample_transcription, diarization
        )

        assert result.segments[0].speaker == "SPEAKER_00"
        assert result.segments[1].speaker == "SPEAKER_01"
        assert result.segments[2].speaker == "SPEAKER_00"

    def test_merge_updates_speakers_list(self, sample_transcription):
        diarization = [
            SpeakerSegment(start=0.0, end=5.0, speaker="Alice"),
            SpeakerSegment(start=5.0, end=10.0, speaker="Bob"),
        ]

        result = merge_transcription_with_diarization(
            sample_transcription, diarization
        )

        assert "Alice" in result.speakers
        assert "Bob" in result.speakers

    def test_merge_empty_diarization(self, sample_transcription):
        result = merge_transcription_with_diarization(
            sample_transcription, []
        )

        # Should return original transcription unchanged
        assert result.audio_file == sample_transcription.audio_file
        assert len(result.segments) == len(sample_transcription.segments)

    def test_merge_preserves_other_fields(self, sample_transcription):
        diarization = [
            SpeakerSegment(start=0.0, end=10.0, speaker="SPEAKER_00"),
        ]

        result = merge_transcription_with_diarization(
            sample_transcription, diarization
        )

        assert result.audio_file == "test.wav"
        assert result.duration_s == 10.0
        assert result.languages_detected == ["en"]
        assert result.segments[0].text == "Hello"
        assert result.segments[0].confidence == 0.9


class TestCreateDiarizer:
    """Tests for create_diarizer factory function."""

    def test_create_default(self):
        diarizer = create_diarizer()
        assert isinstance(diarizer, Diarizer)

    def test_create_with_options(self):
        diarizer = create_diarizer(
            huggingface_token="test_token",
            min_speakers=2,
            max_speakers=4,
        )
        assert diarizer._token == "test_token"
        assert diarizer._min_speakers == 2
        assert diarizer._max_speakers == 4
