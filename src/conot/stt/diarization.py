"""Speaker diarization using Pyannote."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from conot.stt.exceptions import DiarizationError
from conot.stt.models import Segment, TranscriptionResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """A speaker segment from diarization."""

    start: float
    end: float
    speaker: str


class Diarizer:
    """Speaker diarization using Pyannote Audio.

    Identifies different speakers in audio and assigns
    speaker labels to transcription segments.
    """

    def __init__(
        self,
        huggingface_token: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        """Initialize the diarizer.

        Args:
            huggingface_token: HuggingFace token for accessing Pyannote models.
                Falls back to HF_TOKEN environment variable.
            min_speakers: Minimum expected speakers (None for auto-detect).
            max_speakers: Maximum expected speakers (None for auto-detect).
        """
        self._token = huggingface_token or os.environ.get("HF_TOKEN")
        self._min_speakers = min_speakers
        self._max_speakers = max_speakers
        self._pipeline: Any = None
        self._pipeline_loaded = False

    def _ensure_pipeline_loaded(self) -> None:
        """Load Pyannote pipeline if not already loaded."""
        if self._pipeline_loaded:
            return

        if not self._token:
            raise DiarizationError(
                "HuggingFace token required for Pyannote diarization. "
                "Set HF_TOKEN environment variable or pass huggingface_token parameter. "
                "Get token at: https://huggingface.co/settings/tokens"
            )

        try:
            from pyannote.audio import Pipeline

            logger.info("Loading Pyannote speaker diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._token,
            )
            self._pipeline_loaded = True
            logger.info("Pyannote pipeline loaded successfully")

        except ImportError as e:
            raise DiarizationError(
                "pyannote-audio not installed. Install with: pip install pyannote-audio"
            ) from e
        except Exception as e:
            raise DiarizationError(f"Failed to load Pyannote pipeline: {e}") from e

    def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        """Run speaker diarization on audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of SpeakerSegment with speaker labels.

        Raises:
            DiarizationError: If diarization fails.
        """
        self._ensure_pipeline_loaded()

        if not audio_path.exists():
            raise DiarizationError(f"Audio file not found: {audio_path}")

        try:
            # Run diarization
            kwargs = {}
            if self._min_speakers is not None:
                kwargs["min_speakers"] = self._min_speakers
            if self._max_speakers is not None:
                kwargs["max_speakers"] = self._max_speakers

            diarization = self._pipeline(str(audio_path), **kwargs)

            # Convert to SpeakerSegment list
            segments: list[SpeakerSegment] = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        start=turn.start,
                        end=turn.end,
                        speaker=speaker,
                    )
                )

            # Sort by start time
            segments.sort(key=lambda s: s.start)

            logger.info(f"Diarization found {len({s.speaker for s in segments})} speakers")
            return segments

        except Exception as e:
            raise DiarizationError(f"Diarization failed: {e}") from e

    def is_available(self) -> bool:
        """Check if Pyannote is available and configured.

        Returns:
            True if diarization can be performed.
        """
        try:
            import pyannote.audio  # noqa: F401

            return bool(self._token or os.environ.get("HF_TOKEN"))
        except ImportError:
            return False


def merge_transcription_with_diarization(
    transcription: TranscriptionResult,
    diarization: list[SpeakerSegment],
) -> TranscriptionResult:
    """Assign speaker labels to transcription segments.

    Uses temporal overlap to match transcription segments with
    speaker diarization results.

    Args:
        transcription: Transcription result to add speakers to.
        diarization: Speaker segments from diarization.

    Returns:
        TranscriptionResult with speaker labels assigned.
    """
    if not diarization:
        return transcription

    # Create new segments with speaker labels
    new_segments: list[Segment] = []
    speakers_found: set[str] = set()

    for seg in transcription.segments:
        # Find overlapping speaker segment
        speaker = _find_speaker_for_segment(seg.start, seg.end, diarization)
        speakers_found.add(speaker)

        new_segments.append(
            Segment(
                start=seg.start,
                end=seg.end,
                speaker=speaker,
                language=seg.language,
                text=seg.text,
                confidence=seg.confidence,
                words=seg.words,
            )
        )

    return TranscriptionResult(
        audio_file=transcription.audio_file,
        duration_s=transcription.duration_s,
        languages_detected=transcription.languages_detected,
        speakers=sorted(speakers_found),
        segments=new_segments,
    )


def _find_speaker_for_segment(
    start: float,
    end: float,
    diarization: list[SpeakerSegment],
) -> str:
    """Find the speaker with most overlap for a time segment.

    Args:
        start: Segment start time.
        end: Segment end time.
        diarization: Speaker segments to search.

    Returns:
        Speaker label with most overlap, or "UNKNOWN" if no overlap.
    """
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for speaker_seg in diarization:
        # Calculate overlap
        overlap_start = max(start, speaker_seg.start)
        overlap_end = min(end, speaker_seg.end)
        overlap = max(0.0, overlap_end - overlap_start)

        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker_seg.speaker

    return best_speaker


def create_diarizer(
    huggingface_token: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> Diarizer:
    """Create a diarizer instance.

    Args:
        huggingface_token: HuggingFace token for Pyannote.
        min_speakers: Minimum expected speakers.
        max_speakers: Maximum expected speakers.

    Returns:
        Configured Diarizer instance.
    """
    return Diarizer(
        huggingface_token=huggingface_token,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
