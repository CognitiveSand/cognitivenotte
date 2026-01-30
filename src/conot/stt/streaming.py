"""Streaming transcription orchestrator."""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from conot.stt.models import StreamingSegment
from conot.stt.vad import VoiceActivityDetector, create_vad

if TYPE_CHECKING:
    from conot.stt.protocol import STTProvider

logger = logging.getLogger(__name__)

# Type alias for streaming callbacks
StreamingCallback = Callable[[StreamingSegment], None]


@dataclass
class AudioBuffer:
    """Ring buffer for accumulating audio data."""

    max_duration_s: float = 30.0
    sample_rate: int = 16000
    _buffer: bytearray = field(default_factory=bytearray)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def max_bytes(self) -> int:
        """Maximum buffer size in bytes (int16 audio)."""
        return int(self.max_duration_s * self.sample_rate * 2)

    def add(self, audio_bytes: bytes) -> None:
        """Add audio data to buffer.

        Args:
            audio_bytes: Audio data as bytes (int16).
        """
        with self._lock:
            self._buffer.extend(audio_bytes)
            # Trim if exceeds max
            if len(self._buffer) > self.max_bytes:
                excess = len(self._buffer) - self.max_bytes
                self._buffer = self._buffer[excess:]

    def get_all(self) -> bytes:
        """Get all buffered audio and clear buffer.

        Returns:
            All buffered audio as bytes.
        """
        with self._lock:
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

    def peek(self, duration_s: float | None = None) -> bytes:
        """Peek at buffered audio without clearing.

        Args:
            duration_s: Duration to peek (None for all).

        Returns:
            Audio data as bytes.
        """
        with self._lock:
            if duration_s is None:
                return bytes(self._buffer)
            num_bytes = int(duration_s * self.sample_rate * 2)
            return bytes(self._buffer[:num_bytes])

    @property
    def duration_s(self) -> float:
        """Current buffer duration in seconds."""
        with self._lock:
            return len(self._buffer) / (self.sample_rate * 2)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._buffer.clear()


@dataclass
class SegmentTracker:
    """Tracks streaming segments and their refinements."""

    _segments: dict[str, StreamingSegment] = field(default_factory=dict)
    _final_segments: list[StreamingSegment] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, segment: StreamingSegment) -> StreamingSegment | None:
        """Update or add a segment.

        Args:
            segment: The streaming segment to track.

        Returns:
            The segment if it's new or updated, None if unchanged.
        """
        with self._lock:
            existing = self._segments.get(segment.segment_id)

            if existing is None:
                # New segment
                self._segments[segment.segment_id] = segment
                if segment.is_final:
                    self._final_segments.append(segment)
                return segment

            # Check if segment changed
            if existing.text != segment.text or existing.is_final != segment.is_final:
                self._segments[segment.segment_id] = segment
                if segment.is_final and not existing.is_final:
                    self._final_segments.append(segment)
                return segment

            return None  # No change

    def get_final_segments(self) -> list[StreamingSegment]:
        """Get all finalized segments.

        Returns:
            List of final segments in order.
        """
        with self._lock:
            return list(self._final_segments)

    def clear(self) -> None:
        """Clear all tracked segments."""
        with self._lock:
            self._segments.clear()
            self._final_segments.clear()


class StreamingOrchestrator:
    """Orchestrates streaming transcription with VAD and segment tracking.

    This class coordinates:
    - Audio buffering from microphone callbacks
    - Voice activity detection for segment boundaries
    - Transcription via the STT provider
    - Segment refinement tracking
    - Callback notifications for UI updates
    """

    def __init__(
        self,
        provider: STTProvider,
        vad: VoiceActivityDetector | None = None,
        sample_rate: int = 16000,
        callback: StreamingCallback | None = None,
    ) -> None:
        """Initialize the streaming orchestrator.

        Args:
            provider: STT provider for transcription.
            vad: Voice activity detector (created if None).
            sample_rate: Audio sample rate.
            callback: Callback for streaming segment updates.
        """
        self._provider = provider
        self._vad = vad or create_vad(sample_rate=sample_rate)
        self._sample_rate = sample_rate
        self._callback = callback

        self._buffer = AudioBuffer(sample_rate=sample_rate)
        self._tracker = SegmentTracker()

        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

        # Accumulator for speech segments
        self._speech_accumulator = bytearray()
        self._speech_start_time = 0.0
        self._last_speech_end = 0.0
        self._total_audio_time = 0.0

    def start(self) -> None:
        """Start the streaming transcription pipeline."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        logger.info("Streaming transcription started")

    def stop(self) -> list[StreamingSegment]:
        """Stop the streaming pipeline and return final segments.

        Returns:
            List of all finalized segments.
        """
        self._running = False
        self._audio_queue.put(None)  # Signal to stop

        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Process any remaining audio
        self._process_remaining()

        logger.info("Streaming transcription stopped")
        return self._tracker.get_final_segments()

    def feed_audio(self, audio_bytes: bytes) -> None:
        """Feed audio data to the pipeline.

        Args:
            audio_bytes: Audio data as bytes (int16, mono).
        """
        if self._running:
            self._audio_queue.put(audio_bytes)

    def feed_audio_callback(
        self,
        audio_data: NDArray[np.float32],
        *args: object,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> None:
        """Audio callback compatible with sounddevice.

        Args:
            audio_data: Audio data as float32 numpy array.
        """
        # Convert float32 to int16 bytes
        audio_int16 = (audio_data * 32767).astype(np.int16)
        if audio_int16.ndim > 1:
            audio_int16 = audio_int16[:, 0]  # Take first channel
        self.feed_audio(audio_int16.tobytes())

    def _process_loop(self) -> None:
        """Main processing loop running in background thread."""
        silence_threshold_s = 0.8  # Silence duration to trigger transcription
        min_speech_s = 0.3  # Minimum speech to transcribe

        while self._running:
            try:
                # Get audio with timeout
                audio_bytes = self._audio_queue.get(timeout=0.1)

                if audio_bytes is None:
                    break

                # Add to buffer
                self._buffer.add(audio_bytes)

                # Update total audio time
                chunk_duration = len(audio_bytes) / (self._sample_rate * 2)
                self._total_audio_time += chunk_duration

                # Convert to float for VAD
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_array.astype(np.float32) / 32768.0

                # Check for speech
                speech_segments = self._vad.detect_speech(audio_float)

                if speech_segments:
                    # Speech detected - accumulate
                    self._speech_accumulator.extend(audio_bytes)

                    if not self._speech_start_time:
                        self._speech_start_time = self._total_audio_time - chunk_duration

                    self._last_speech_end = self._total_audio_time

                else:
                    # Silence - check if we should transcribe accumulated speech
                    silence_duration = self._total_audio_time - self._last_speech_end

                    if (
                        len(self._speech_accumulator) > 0
                        and silence_duration >= silence_threshold_s
                    ):
                        speech_duration = len(self._speech_accumulator) / (
                            self._sample_rate * 2
                        )

                        if speech_duration >= min_speech_s:
                            self._transcribe_accumulated()

            except queue.Empty:
                # Check for pending speech on timeout
                if len(self._speech_accumulator) > 0:
                    silence_duration = self._total_audio_time - self._last_speech_end
                    if silence_duration >= silence_threshold_s:
                        self._transcribe_accumulated()
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")

    def _transcribe_accumulated(self) -> None:
        """Transcribe accumulated speech."""
        if not self._speech_accumulator:
            return

        speech_bytes = bytes(self._speech_accumulator)
        speech_start = self._speech_start_time

        # Clear accumulator
        self._speech_accumulator.clear()
        self._speech_start_time = 0.0

        # Create audio stream for provider
        def audio_generator() -> Iterator[bytes]:
            yield speech_bytes

        try:
            # Get transcription
            for segment in self._provider.transcribe_stream(audio_generator()):
                # Adjust timestamps
                adjusted_segment = StreamingSegment(
                    segment_id=segment.segment_id,
                    start=speech_start + segment.start,
                    end=speech_start + segment.end,
                    text=segment.text,
                    language=segment.language,
                    is_final=segment.is_final,
                    confidence=segment.confidence,
                )

                # Track and notify
                updated = self._tracker.update(adjusted_segment)
                if updated and self._callback:
                    self._callback(updated)

        except Exception as e:
            logger.error(f"Transcription error: {e}")

    def _process_remaining(self) -> None:
        """Process any remaining audio in buffers."""
        # Transcribe any accumulated speech
        if self._speech_accumulator:
            self._transcribe_accumulated()


def create_streaming_transcriber(
    provider: STTProvider,
    sample_rate: int = 16000,
    callback: StreamingCallback | None = None,
) -> StreamingOrchestrator:
    """Create a streaming transcription orchestrator.

    Args:
        provider: STT provider for transcription.
        sample_rate: Audio sample rate.
        callback: Callback for segment updates.

    Returns:
        Configured StreamingOrchestrator instance.
    """
    return StreamingOrchestrator(
        provider=provider,
        sample_rate=sample_rate,
        callback=callback,
    )
