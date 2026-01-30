"""Data models for speech-to-text transcription."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Language(str, Enum):
    """Supported languages for transcription."""

    FRENCH = "fr"
    ENGLISH = "en"
    AUTO = "auto"


class ProviderTier(str, Enum):
    """Hardware tier for provider selection."""

    ENTERPRISE = "enterprise"  # 12GB+ VRAM
    STANDARD = "standard"  # 4-12GB VRAM
    EDGE = "edge"  # CPU-only, <4GB RAM


@dataclass
class Word:
    """A single word with timing information."""

    word: str
    start: float
    end: float
    confidence: float


@dataclass
class Segment:
    """A transcription segment with speaker and timing."""

    start: float
    end: float
    speaker: str
    language: str
    text: str
    confidence: float
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert segment to dictionary."""
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "language": self.language,
            "text": self.text,
            "confidence": self.confidence,
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in self.words
            ],
        }


@dataclass
class StreamingSegment:
    """Incremental segment for streaming transcription."""

    segment_id: str  # Unique ID to track refinements
    start: float
    end: float
    text: str
    language: str
    is_final: bool  # True when segment won't change
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        """Convert streaming segment to dictionary."""
        return {
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "language": self.language,
            "is_final": self.is_final,
            "confidence": self.confidence,
        }


@dataclass
class TranscriptionResult:
    """Complete transcription result."""

    audio_file: str
    duration_s: float
    languages_detected: list[str]
    speakers: list[str]
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "audio_file": self.audio_file,
            "duration_s": self.duration_s,
            "languages_detected": self.languages_detected,
            "speakers": self.speakers,
            "segments": [s.to_dict() for s in self.segments],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert result to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def to_text(self) -> str:
        """Convert result to plain text with speakers."""
        lines: list[str] = []
        current_speaker = ""
        for segment in self.segments:
            if segment.speaker != current_speaker:
                current_speaker = segment.speaker
                lines.append(f"\n[{current_speaker}]")
            lines.append(segment.text)
        return "\n".join(lines).strip()

    def to_srt(self) -> str:
        """Convert result to SRT subtitle format."""
        lines: list[str] = []
        for i, segment in enumerate(self.segments, 1):
            start_time = _format_srt_time(segment.start)
            end_time = _format_srt_time(segment.end)
            speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
            lines.append(str(i))
            lines.append(f"{start_time} --> {end_time}")
            lines.append(f"{speaker_prefix}{segment.text}")
            lines.append("")
        return "\n".join(lines)


def _format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


@dataclass
class ProviderCapabilities:
    """Capabilities of an STT provider."""

    languages: list[str]
    supports_diarization: bool
    supports_word_timestamps: bool
    supports_streaming: bool
    streaming_latency_ms: int
    min_memory_gb: float
    requires_gpu: bool


@dataclass
class HardwareProfile:
    """Detected hardware profile for provider selection."""

    has_gpu: bool
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    recommended_tier: ProviderTier

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        return {
            "has_gpu": self.has_gpu,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "recommended_tier": self.recommended_tier.value,
        }
