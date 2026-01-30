"""Voice Activity Detection for streaming transcription."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from conot.stt.exceptions import VADError

logger = logging.getLogger(__name__)


@dataclass
class SpeechSegment:
    """A detected speech segment."""

    start_sample: int
    end_sample: int
    audio: NDArray[np.float32]

    @property
    def duration_samples(self) -> int:
        """Duration in samples."""
        return self.end_sample - self.start_sample


class VoiceActivityDetector:
    """Voice Activity Detection using Silero VAD.

    Detects speech/silence boundaries in audio streams for
    efficient chunking during streaming transcription.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 200,
        sample_rate: int = 16000,
    ) -> None:
        """Initialize the VAD.

        Args:
            threshold: Speech probability threshold (0-1).
            min_speech_duration_ms: Minimum speech duration to detect.
            min_silence_duration_ms: Minimum silence duration to split segments.
            speech_pad_ms: Padding around speech segments.
            sample_rate: Audio sample rate.
        """
        self._threshold = threshold
        self._min_speech_ms = min_speech_duration_ms
        self._min_silence_ms = min_silence_duration_ms
        self._speech_pad_ms = speech_pad_ms
        self._sample_rate = sample_rate

        self._model: Any = None
        self._model_loaded = False

    def _ensure_model_loaded(self) -> None:
        """Load Silero VAD model if not already loaded."""
        if self._model_loaded:
            return

        try:
            self._load_silero_vad()
        except Exception:
            logger.warning("Silero VAD not available, using energy-based VAD")
            self._model = None
            self._model_loaded = True

    def _load_silero_vad(self) -> None:
        """Load Silero VAD model."""
        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]  # get_speech_timestamps function
            self._model_loaded = True
            logger.info("Loaded Silero VAD model")
        except ImportError as e:
            raise VADError("torch not installed for Silero VAD") from e
        except Exception as e:
            raise VADError(f"Failed to load Silero VAD: {e}") from e

    def detect_speech(
        self,
        audio: NDArray[np.float32],
    ) -> list[tuple[int, int]]:
        """Detect speech segments in audio.

        Args:
            audio: Audio data as float32 numpy array (normalized to -1 to 1).

        Returns:
            List of (start_sample, end_sample) tuples for speech segments.
        """
        self._ensure_model_loaded()

        if self._model is not None:
            return self._detect_with_silero(audio)
        else:
            return self._detect_with_energy(audio)

    def _detect_with_silero(
        self,
        audio: NDArray[np.float32],
    ) -> list[tuple[int, int]]:
        """Detect speech using Silero VAD."""
        import torch

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)

        # Get speech timestamps
        speech_timestamps = self._get_speech_timestamps(
            audio_tensor,
            self._model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
            min_speech_duration_ms=self._min_speech_ms,
            min_silence_duration_ms=self._min_silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )

        # Convert to sample indices
        segments = []
        for ts in speech_timestamps:
            segments.append((ts["start"], ts["end"]))

        return segments

    def _detect_with_energy(
        self,
        audio: NDArray[np.float32],
    ) -> list[tuple[int, int]]:
        """Fallback energy-based VAD when Silero is not available."""
        # Frame parameters
        frame_size = int(0.03 * self._sample_rate)  # 30ms frames
        hop_size = int(0.01 * self._sample_rate)  # 10ms hop

        # Calculate RMS energy per frame
        n_frames = (len(audio) - frame_size) // hop_size + 1
        energies = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            frame = audio[start:end]
            energies[i] = np.sqrt(np.mean(frame**2))

        # Adaptive threshold
        noise_floor = np.percentile(energies, 10)
        speech_threshold = noise_floor * 3

        # Find speech frames
        is_speech = energies > speech_threshold

        # Smooth with minimum durations
        min_speech_frames = int(self._min_speech_ms / 10)
        min_silence_frames = int(self._min_silence_ms / 10)

        # Find speech segment boundaries
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    speech_start = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        speech_end = i - silence_count
                        if speech_end - speech_start >= min_speech_frames:
                            # Convert frame indices to sample indices
                            start_sample = speech_start * hop_size
                            end_sample = min(speech_end * hop_size + frame_size, len(audio))
                            # Add padding
                            pad_samples = int(self._speech_pad_ms * self._sample_rate / 1000)
                            start_sample = max(0, start_sample - pad_samples)
                            end_sample = min(len(audio), end_sample + pad_samples)
                            segments.append((start_sample, end_sample))
                        in_speech = False

        # Handle segment at end
        if in_speech:
            speech_end = len(is_speech)
            if speech_end - speech_start >= min_speech_frames:
                start_sample = speech_start * hop_size
                end_sample = len(audio)
                pad_samples = int(self._speech_pad_ms * self._sample_rate / 1000)
                start_sample = max(0, start_sample - pad_samples)
                segments.append((start_sample, end_sample))

        return segments

    def process_stream(
        self,
        audio_stream: Iterator[bytes],
        bytes_per_sample: int = 2,
    ) -> Iterator[SpeechSegment]:
        """Process audio stream and yield speech segments.

        Args:
            audio_stream: Iterator yielding audio chunks as bytes (int16).
            bytes_per_sample: Bytes per audio sample (2 for int16).

        Yields:
            SpeechSegment objects containing detected speech.
        """
        self._ensure_model_loaded()

        # Accumulate audio for VAD processing
        buffer = bytearray()
        buffer_samples = 0
        total_samples = 0

        # Process in chunks of ~1 second
        chunk_duration_s = 1.0
        chunk_samples = int(chunk_duration_s * self._sample_rate)
        chunk_bytes = chunk_samples * bytes_per_sample

        for chunk in audio_stream:
            buffer.extend(chunk)
            buffer_samples = len(buffer) // bytes_per_sample

            # Process when we have enough audio
            while buffer_samples >= chunk_samples:
                # Extract chunk
                process_bytes = bytes(buffer[:chunk_bytes])
                buffer = buffer[chunk_bytes:]
                buffer_samples = len(buffer) // bytes_per_sample

                # Convert to float32
                audio_array = np.frombuffer(process_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Detect speech in this chunk
                speech_segments = self.detect_speech(audio_float)

                # Yield speech segments
                for start, end in speech_segments:
                    yield SpeechSegment(
                        start_sample=total_samples + start,
                        end_sample=total_samples + end,
                        audio=audio_float[start:end],
                    )

                total_samples += chunk_samples

        # Process remaining audio
        if len(buffer) > 0:
            audio_array = np.frombuffer(bytes(buffer), dtype=np.int16)
            audio_float = audio_array.astype(np.float32) / 32768.0

            speech_segments = self.detect_speech(audio_float)

            for start, end in speech_segments:
                yield SpeechSegment(
                    start_sample=total_samples + start,
                    end_sample=total_samples + end,
                    audio=audio_float[start:end],
                )


def create_vad(
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 500,
    sample_rate: int = 16000,
) -> VoiceActivityDetector:
    """Create a VAD instance with the specified parameters.

    Args:
        threshold: Speech probability threshold.
        min_speech_duration_ms: Minimum speech duration.
        min_silence_duration_ms: Minimum silence to split segments.
        sample_rate: Audio sample rate.

    Returns:
        Configured VoiceActivityDetector instance.
    """
    return VoiceActivityDetector(
        threshold=threshold,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        sample_rate=sample_rate,
    )
