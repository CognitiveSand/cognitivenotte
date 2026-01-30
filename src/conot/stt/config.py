"""STT-specific configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from conot.stt.models import Language


class STTConfig(BaseModel):
    """Speech-to-text configuration."""

    provider: str = Field(
        default="auto",
        description="STT provider: auto | faster-whisper | whisper-cpp",
    )
    device: str = Field(
        default="auto",
        description="Compute device: auto | cuda | cpu",
    )
    language: str = Field(
        default=Language.AUTO.value,
        description="Primary language: auto | fr | en",
    )
    diarization: bool = Field(
        default=True,
        description="Enable speaker diarization",
    )
    model_size: str = Field(
        default="auto",
        description="Model size: auto | large-v3 | medium | small | tiny",
    )
    compute_type: str = Field(
        default="auto",
        description="Compute precision: auto | float16 | int8 | float32",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Voice activity detection threshold",
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=50,
        le=5000,
        description="Minimum speech duration in milliseconds",
    )
    max_speech_duration_s: float = Field(
        default=30.0,
        ge=1.0,
        le=300.0,
        description="Maximum speech segment duration in seconds",
    )
    huggingface_token: str | None = Field(
        default=None,
        description="HuggingFace token for pyannote diarization models",
    )
