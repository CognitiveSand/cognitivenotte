"""Tests for language detection with smoothing."""

import pytest

from conot.stt.language_detector import (
    LanguageDetection,
    LanguageDetector,
    LanguageDetectorConfig,
    create_language_detector,
)


class TestLanguageDetection:
    """Tests for LanguageDetection dataclass."""

    def test_creation(self):
        det = LanguageDetection(language="fr", probability=0.92)
        assert det.language == "fr"
        assert det.probability == 0.92
        assert det.timestamp == 0.0

    def test_creation_with_timestamp(self):
        det = LanguageDetection(language="en", probability=0.85, timestamp=5.0)
        assert det.timestamp == 5.0


class TestLanguageDetectorConfig:
    """Tests for LanguageDetectorConfig."""

    def test_defaults(self):
        config = LanguageDetectorConfig()
        assert config.min_confidence == 0.6
        assert config.high_confidence == 0.75
        assert config.voting_window == 3
        assert config.change_threshold == 0.5
        assert config.allowed_languages is None
        assert config.initial_detection_duration == 5.0

    def test_custom_values(self):
        config = LanguageDetectorConfig(
            min_confidence=0.8,
            allowed_languages=["fr", "en"],
        )
        assert config.min_confidence == 0.8
        assert config.allowed_languages == ["fr", "en"]


class TestLanguageDetector:
    """Tests for LanguageDetector."""

    def test_initial_state(self):
        detector = LanguageDetector()
        assert detector.current_language is None
        assert not detector.initial_detection_done

    def test_high_confidence_detection(self):
        detector = LanguageDetector()
        result = detector.update("fr", 0.92, audio_duration=2.0)
        assert result == "fr"
        assert detector.current_language == "fr"

    def test_low_confidence_ignored(self):
        """Low confidence detection should use previous language."""
        detector = LanguageDetector()
        # First: high confidence French
        detector.update("fr", 0.92, audio_duration=2.0)
        # Second: low confidence German - should be ignored
        result = detector.update("de", 0.45, audio_duration=2.0)
        assert result == "fr"  # Keep previous
        assert detector.current_language == "fr"

    def test_allowed_languages_filter(self):
        """Detection outside allowed languages should be filtered."""
        config = LanguageDetectorConfig(allowed_languages=["fr", "en"])
        detector = LanguageDetector(config)

        # German is not allowed, should return None or first allowed
        result = detector.update("de", 0.95, audio_duration=2.0)
        assert result == "fr"  # First allowed language

    def test_allowed_languages_accepts_valid(self):
        """Detection within allowed languages should be accepted."""
        config = LanguageDetectorConfig(allowed_languages=["fr", "en"])
        detector = LanguageDetector(config)

        result = detector.update("en", 0.88, audio_duration=2.0)
        assert result == "en"

    def test_voting_prevents_flip_flop(self):
        """Medium confidence detection shouldn't flip-flop without consensus."""
        config = LanguageDetectorConfig(
            voting_window=4,  # Need more votes to see effect
            initial_detection_duration=0,
        )
        detector = LanguageDetector(config)
        # Build up French consensus
        detector.update("fr", 0.80, audio_duration=2.0)
        detector.update("fr", 0.78, audio_duration=2.0)
        detector.update("fr", 0.82, audio_duration=2.0)
        # Single MEDIUM-confidence English (below high_confidence=0.75) shouldn't change
        result = detector.update("en", 0.68, audio_duration=2.0)
        # Need 50% consensus to change, only have 1/4 = 25%
        assert result == "fr"

    def test_language_change_with_consensus(self):
        """Language should change when there's enough consensus."""
        config = LanguageDetectorConfig(
            voting_window=3,
            change_threshold=0.6,
            initial_detection_duration=0,  # Skip initial phase
        )
        detector = LanguageDetector(config)
        # Start with French
        detector.update("fr", 0.90, audio_duration=1.0)
        # Build English consensus (2/3 = 67% > 60%)
        detector.update("en", 0.85, audio_duration=1.0)
        result = detector.update("en", 0.88, audio_duration=1.0)
        assert result == "en"

    def test_high_confidence_immediate_switch(self):
        """High confidence detection should switch immediately."""
        config = LanguageDetectorConfig(
            high_confidence=0.85,
            initial_detection_duration=0,
        )
        detector = LanguageDetector(config)
        # Start with French
        detector.update("fr", 0.90, audio_duration=2.0)
        assert detector.current_language == "fr"
        # Single high-confidence English should switch immediately
        result = detector.update("en", 0.92, audio_duration=2.0)
        assert result == "en"  # Immediate switch due to high confidence

    def test_high_confidence_bilingual_switching(self):
        """Bilingual speech should switch back and forth with high confidence."""
        config = LanguageDetectorConfig(
            allowed_languages=["fr", "en"],
            high_confidence=0.85,
            initial_detection_duration=0,
        )
        detector = LanguageDetector(config)
        # French
        result = detector.update("fr", 0.91, audio_duration=2.0)
        assert result == "fr"
        # Switch to English (high confidence)
        result = detector.update("en", 0.88, audio_duration=2.0)
        assert result == "en"
        # Back to French (high confidence)
        result = detector.update("fr", 0.90, audio_duration=2.0)
        assert result == "fr"
        # Back to English
        result = detector.update("en", 0.86, audio_duration=2.0)
        assert result == "en"

    def test_reset(self):
        """Reset should clear all state."""
        detector = LanguageDetector()
        detector.update("fr", 0.92, audio_duration=5.0)
        detector.reset()
        assert detector.current_language is None
        assert not detector.initial_detection_done

    def test_set_language_manual(self):
        """Manual language setting should work."""
        detector = LanguageDetector()
        detector.set_language("en")
        assert detector.current_language == "en"
        assert detector.initial_detection_done
        # Subsequent low-confidence detections shouldn't override manual setting
        result = detector.update("fr", 0.50, audio_duration=2.0)
        assert result == "en"  # Low confidence (<0.6), keep manual

    def test_stats(self):
        """Stats should return useful debug info."""
        detector = LanguageDetector()
        detector.update("fr", 0.92, audio_duration=2.0)
        detector.update("fr", 0.88, audio_duration=2.0)
        stats = detector.stats
        assert stats["current_language"] == "fr"
        assert stats["accumulated_duration"] == 4.0
        assert stats["detection_count"] == 2
        assert stats["recent_votes"] == {"fr": 2}


class TestCreateLanguageDetector:
    """Tests for create_language_detector factory."""

    def test_create_default(self):
        detector = create_language_detector()
        assert detector._config.min_confidence == 0.7
        assert detector._config.allowed_languages is None

    def test_create_with_allowed_languages(self):
        detector = create_language_detector(allowed_languages=["fr", "en"])
        assert detector._config.allowed_languages == ["fr", "en"]

    def test_create_with_min_confidence(self):
        detector = create_language_detector(min_confidence=0.8)
        assert detector._config.min_confidence == 0.8


class TestLanguageDetectorIntegration:
    """Integration tests for language detection scenarios."""

    def test_realistic_french_english_session(self):
        """Simulate a realistic bilingual session."""
        config = LanguageDetectorConfig(
            allowed_languages=["fr", "en"],
            min_confidence=0.7,
            initial_detection_duration=4.0,
        )
        detector = LanguageDetector(config)

        # Initial French speech (building up confidence)
        detector.update("fr", 0.91, audio_duration=2.0)
        detector.update("fr", 0.89, audio_duration=2.0)
        assert detector.current_language == "fr"
        assert detector.initial_detection_done

        # Brief noise detected as German (should be ignored - not in allowed)
        result = detector.update("de", 0.55, audio_duration=1.0)
        assert result == "fr"  # Stays French (German not allowed)

        # Speaker switches to English
        detector.update("en", 0.82, audio_duration=2.0)
        detector.update("en", 0.87, audio_duration=2.0)
        detector.update("en", 0.85, audio_duration=2.0)
        assert detector.current_language == "en"

    def test_noisy_detection_with_low_confidence(self):
        """Low confidence detections should be ignored."""
        detector = create_language_detector(min_confidence=0.7)

        # High confidence French
        detector.update("fr", 0.92, audio_duration=3.0)
        detector.update("fr", 0.88, audio_duration=2.0)

        # Series of low-confidence noise detections
        for lang in ["de", "ru", "ja", "zh"]:
            result = detector.update(lang, 0.45, audio_duration=1.0)
            assert result == "fr"  # All ignored due to low confidence

        # French continues
        result = detector.update("fr", 0.90, audio_duration=2.0)
        assert result == "fr"
