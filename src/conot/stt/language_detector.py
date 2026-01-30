"""Language detection with confidence tracking and smoothing."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class LanguageDetection:
    """A single language detection result."""

    language: str
    probability: float
    timestamp: float = 0.0


@dataclass
class LanguageDetectorConfig:
    """Configuration for language detection behavior."""

    # Minimum probability to accept a detection
    min_confidence: float = 0.7

    # High confidence threshold - immediately switch if above this
    high_confidence: float = 0.85

    # Number of recent detections to consider for voting
    voting_window: int = 5

    # Minimum ratio of votes needed to change language (e.g., 0.6 = 60%)
    change_threshold: float = 0.6

    # If set, only these languages are allowed (e.g., ["fr", "en"])
    allowed_languages: list[str] | None = None

    # Duration (seconds) of initial audio to use for first detection
    initial_detection_duration: float = 5.0


class LanguageDetector:
    """Tracks language detections and provides smoothed/voted results.

    This class improves language detection accuracy by:
    1. Filtering out low-confidence detections
    2. Using voting across multiple chunks for consistency
    3. Constraining to allowed languages if specified
    4. Providing stable language output that doesn't flip-flop
    """

    def __init__(self, config: LanguageDetectorConfig | None = None) -> None:
        """Initialize the language detector.

        Args:
            config: Configuration options. Uses defaults if None.
        """
        self._config = config or LanguageDetectorConfig()
        self._lock = Lock()

        # Recent detections for voting
        self._recent_detections: deque[LanguageDetection] = deque(
            maxlen=self._config.voting_window
        )

        # Current stable language
        self._current_language: str | None = None

        # Initial detection state
        self._initial_detection_done = False
        self._accumulated_audio_duration = 0.0

    @property
    def current_language(self) -> str | None:
        """Get the current stable detected language."""
        with self._lock:
            return self._current_language

    @property
    def initial_detection_done(self) -> bool:
        """Whether initial language detection is complete."""
        with self._lock:
            return self._initial_detection_done

    def update(
        self,
        language: str,
        probability: float,
        audio_duration: float = 0.0,
    ) -> str | None:
        """Update with a new language detection.

        Args:
            language: Detected language code (e.g., "fr", "en").
            probability: Detection probability (0-1).
            audio_duration: Duration of audio used for this detection.

        Returns:
            The stable language to use, or None if no reliable detection yet.
        """
        with self._lock:
            self._accumulated_audio_duration += audio_duration

            # Filter by allowed languages
            if self._config.allowed_languages:
                if language not in self._config.allowed_languages:
                    logger.debug(
                        f"Language {language} not in allowed list "
                        f"{self._config.allowed_languages}, ignoring"
                    )
                    # Return current language or first allowed
                    return self._current_language or self._config.allowed_languages[0]

            # Check confidence threshold
            if probability < self._config.min_confidence:
                logger.debug(
                    f"Language {language} probability {probability:.2f} "
                    f"below threshold {self._config.min_confidence}, ignoring"
                )
                return self._current_language

            # Add to recent detections
            detection = LanguageDetection(
                language=language,
                probability=probability,
                timestamp=self._accumulated_audio_duration,
            )
            self._recent_detections.append(detection)

            # Initial detection: wait for enough audio
            if not self._initial_detection_done:
                if self._accumulated_audio_duration >= self._config.initial_detection_duration:
                    # Use voting on initial detections
                    self._current_language = self._get_voted_language()
                    self._initial_detection_done = True
                    logger.info(
                        f"Initial language detection: {self._current_language} "
                        f"(after {self._accumulated_audio_duration:.1f}s)"
                    )
                else:
                    # Not enough audio yet, use best so far
                    self._current_language = self._get_voted_language()
                return self._current_language

            # High-confidence fast path: if detection is very confident AND
            # within allowed languages, switch immediately (for bilingual speech)
            if probability >= self._config.high_confidence:
                if self._config.allowed_languages is None or language in self._config.allowed_languages:
                    if language != self._current_language:
                        logger.info(
                            f"Language changed (high confidence): "
                            f"{self._current_language} → {language} ({probability:.0%})"
                        )
                        self._current_language = language
                    return self._current_language

            # Ongoing detection: use voting with change threshold
            voted_language = self._get_voted_language()

            if voted_language != self._current_language:
                # Check if there's enough consensus to change
                vote_ratio = self._get_vote_ratio(voted_language)
                if vote_ratio >= self._config.change_threshold:
                    logger.info(
                        f"Language changed: {self._current_language} → {voted_language} "
                        f"(vote ratio: {vote_ratio:.0%})"
                    )
                    self._current_language = voted_language
                else:
                    logger.debug(
                        f"Language vote for {voted_language} ({vote_ratio:.0%}) "
                        f"below threshold, keeping {self._current_language}"
                    )

            return self._current_language

    def _get_voted_language(self) -> str | None:
        """Get the language with most votes in recent detections.

        Returns:
            Language code with highest weighted vote count.
        """
        if not self._recent_detections:
            return None

        # Weight votes by probability
        votes: dict[str, float] = {}
        for det in self._recent_detections:
            votes[det.language] = votes.get(det.language, 0) + det.probability

        # Return language with highest weighted votes
        return max(votes, key=lambda k: votes[k])

    def _get_vote_ratio(self, language: str) -> float:
        """Get the ratio of votes for a specific language.

        Args:
            language: Language code to check.

        Returns:
            Ratio of votes for this language (0-1).
        """
        if not self._recent_detections:
            return 0.0

        total_votes = len(self._recent_detections)
        lang_votes = sum(1 for d in self._recent_detections if d.language == language)
        return lang_votes / total_votes

    def reset(self) -> None:
        """Reset the detector state."""
        with self._lock:
            self._recent_detections.clear()
            self._current_language = None
            self._initial_detection_done = False
            self._accumulated_audio_duration = 0.0

    def set_language(self, language: str) -> None:
        """Manually set the language (disables auto-detection).

        Args:
            language: Language code to use.
        """
        with self._lock:
            self._current_language = language
            self._initial_detection_done = True
            logger.info(f"Language manually set to: {language}")

    @property
    def stats(self) -> dict:
        """Get detection statistics for debugging."""
        with self._lock:
            votes: dict[str, int] = {}
            for det in self._recent_detections:
                votes[det.language] = votes.get(det.language, 0) + 1

            return {
                "current_language": self._current_language,
                "initial_detection_done": self._initial_detection_done,
                "accumulated_duration": self._accumulated_audio_duration,
                "recent_votes": votes,
                "detection_count": len(self._recent_detections),
            }


def create_language_detector(
    allowed_languages: list[str] | None = None,
    min_confidence: float = 0.7,
) -> LanguageDetector:
    """Create a language detector with common settings.

    Args:
        allowed_languages: List of allowed language codes (e.g., ["fr", "en"]).
        min_confidence: Minimum probability to accept a detection.

    Returns:
        Configured LanguageDetector instance.
    """
    config = LanguageDetectorConfig(
        allowed_languages=allowed_languages,
        min_confidence=min_confidence,
    )
    return LanguageDetector(config)
