"""Configuration management using Pydantic settings."""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, Field, model_validator

from conot.exceptions import ConfigError


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = Field(default=48000, ge=8000, le=192000)
    channels: int = Field(default=1, ge=1, le=2)
    device_id: int | None = Field(default=None)


class RecordingConfig(BaseModel):
    """Recording output configuration."""

    output_dir: Path = Field(default=Path("./recordings"))
    filename_format: str = Field(default="recording_%Y%m%d_%H%M%S.wav")

    @model_validator(mode="after")
    def validate_filename_format(self) -> Self:
        if not self.filename_format.endswith(".wav"):
            raise ValueError("filename_format must end with .wav")
        return self


class DebugConfig(BaseModel):
    """Debug view configuration."""

    meter_update_interval: float = Field(default=0.05, ge=0.01, le=1.0)
    reference_db: float = Field(default=-60.0, ge=-100.0, le=0.0)


class Settings(BaseModel):
    """Application settings loaded from settings.yml."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    recording: RecordingConfig = Field(default_factory=RecordingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        """Load settings from YAML file.

        Args:
            config_path: Path to settings file. Defaults to ./settings.yml

        Returns:
            Loaded Settings instance

        Raises:
            ConfigError: If config file exists but is invalid
        """
        if config_path is None:
            config_path = Path("./settings.yml")

        if not config_path.exists():
            return cls()

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Failed to parse config file: {e}") from e

        if data is None:
            return cls()

        try:
            return cls.model_validate(data)
        except Exception as e:
            raise ConfigError(f"Invalid configuration: {e}") from e


_settings: Settings | None = None


def get_settings(config_path: Path | None = None) -> Settings:
    """Get application settings (singleton).

    Args:
        config_path: Optional path to config file

    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings.load(config_path)
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)."""
    global _settings
    _settings = None
