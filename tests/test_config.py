"""Tests for configuration module."""

from pathlib import Path

import pytest
import yaml

from conot.config import Settings, reset_settings
from conot.exceptions import ConfigError


@pytest.fixture(autouse=True)
def reset():
    """Reset settings singleton before each test."""
    reset_settings()
    yield
    reset_settings()


class TestSettings:
    """Tests for Settings class."""

    def test_default_settings(self):
        """Test default settings values."""
        settings = Settings()
        assert settings.audio.sample_rate == 48000
        assert settings.audio.channels == 1
        assert settings.audio.device_id is None
        assert settings.recording.output_dir == Path("./recordings")
        assert settings.recording.filename_format == "recording_%Y%m%d_%H%M%S.wav"
        assert settings.debug.meter_update_interval == 0.05
        assert settings.debug.reference_db == -60.0

    def test_load_from_file(self, tmp_path: Path):
        """Test loading settings from YAML file."""
        config_path = tmp_path / "settings.yml"
        config_path.write_text(
            yaml.dump(
                {
                    "audio": {
                        "sample_rate": 44100,
                        "channels": 2,
                    },
                    "recording": {
                        "output_dir": "/tmp/recordings",
                    },
                }
            )
        )

        settings = Settings.load(config_path)
        assert settings.audio.sample_rate == 44100
        assert settings.audio.channels == 2
        assert settings.recording.output_dir == Path("/tmp/recordings")

    def test_load_missing_file_uses_defaults(self, tmp_path: Path):
        """Test that missing config file uses defaults."""
        settings = Settings.load(tmp_path / "nonexistent.yml")
        assert settings.audio.sample_rate == 48000

    def test_load_empty_file_uses_defaults(self, tmp_path: Path):
        """Test that empty config file uses defaults."""
        config_path = tmp_path / "settings.yml"
        config_path.write_text("")

        settings = Settings.load(config_path)
        assert settings.audio.sample_rate == 48000

    def test_invalid_yaml_raises_error(self, tmp_path: Path):
        """Test that invalid YAML raises ConfigError."""
        config_path = tmp_path / "settings.yml"
        config_path.write_text("invalid: yaml: content:")

        with pytest.raises(ConfigError):
            Settings.load(config_path)

    def test_invalid_values_raise_error(self, tmp_path: Path):
        """Test that invalid config values raise ConfigError."""
        config_path = tmp_path / "settings.yml"
        config_path.write_text(
            yaml.dump(
                {
                    "audio": {
                        "sample_rate": -1,  # Invalid
                    }
                }
            )
        )

        with pytest.raises(ConfigError):
            Settings.load(config_path)

    def test_filename_format_must_end_with_wav(self):
        """Test that filename format must end with .wav."""
        with pytest.raises(ValueError, match="must end with .wav"):
            Settings(
                recording={"filename_format": "recording.mp3"}  # type: ignore
            )
