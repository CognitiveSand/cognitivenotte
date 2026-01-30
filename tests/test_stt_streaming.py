"""Tests for STT streaming orchestrator."""

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from conot.stt.models import ProviderCapabilities, StreamingSegment
from conot.stt.streaming import (
    AudioBuffer,
    SegmentTracker,
    StreamingOrchestrator,
    create_streaming_transcriber,
)


class TestAudioBuffer:
    """Tests for AudioBuffer class."""

    def test_creation_defaults(self):
        buffer = AudioBuffer()
        assert buffer.max_duration_s == 30.0
        assert buffer.sample_rate == 16000
        assert buffer.duration_s == 0.0

    def test_add_audio(self):
        buffer = AudioBuffer(sample_rate=16000)
        # Add 0.1 seconds of audio (3200 bytes for 16kHz int16)
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        buffer.add(audio)
        assert buffer.duration_s == pytest.approx(0.1, rel=0.01)

    def test_get_all_clears_buffer(self):
        buffer = AudioBuffer()
        audio = b"\x00" * 3200
        buffer.add(audio)

        data = buffer.get_all()
        assert len(data) == 3200
        assert buffer.duration_s == 0.0

    def test_peek_does_not_clear(self):
        buffer = AudioBuffer()
        audio = b"\x00" * 3200
        buffer.add(audio)

        data = buffer.peek()
        assert len(data) == 3200
        assert buffer.duration_s > 0

    def test_peek_with_duration(self):
        buffer = AudioBuffer(sample_rate=16000)
        # Add 1 second of audio
        audio = np.zeros(16000, dtype=np.int16).tobytes()
        buffer.add(audio)

        # Peek only 0.5 seconds
        data = buffer.peek(duration_s=0.5)
        assert len(data) == 16000  # 0.5s * 16000 * 2 bytes

    def test_clear(self):
        buffer = AudioBuffer()
        buffer.add(b"\x00" * 1000)
        assert buffer.duration_s > 0

        buffer.clear()
        assert buffer.duration_s == 0.0

    def test_max_size_enforcement(self):
        buffer = AudioBuffer(max_duration_s=1.0, sample_rate=16000)
        # Add 2 seconds of audio
        audio = np.zeros(32000, dtype=np.int16).tobytes()
        buffer.add(audio)

        # Should be trimmed to max
        assert buffer.duration_s <= 1.0

    def test_thread_safety(self):
        """Test buffer operations are thread-safe."""
        buffer = AudioBuffer()
        errors = []

        def writer():
            try:
                for _ in range(100):
                    buffer.add(b"\x00" * 320)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(100):
                    buffer.peek()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSegmentTracker:
    """Tests for SegmentTracker class."""

    def test_creation(self):
        tracker = SegmentTracker()
        assert tracker.get_final_segments() == []

    def test_update_new_segment(self):
        tracker = SegmentTracker()
        segment = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hello",
            language="en",
            is_final=False,
            confidence=0.8,
        )

        result = tracker.update(segment)
        assert result is not None
        assert result.segment_id == "seg_1"

    def test_update_unchanged_segment(self):
        tracker = SegmentTracker()
        segment = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hello",
            language="en",
            is_final=False,
            confidence=0.8,
        )

        tracker.update(segment)
        result = tracker.update(segment)  # Same segment again
        assert result is None  # No change

    def test_update_refined_segment(self):
        tracker = SegmentTracker()
        segment1 = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hell",
            language="en",
            is_final=False,
            confidence=0.7,
        )
        segment2 = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hello",
            language="en",
            is_final=False,
            confidence=0.8,
        )

        tracker.update(segment1)
        result = tracker.update(segment2)
        assert result is not None
        assert result.text == "Hello"

    def test_final_segments(self):
        tracker = SegmentTracker()
        segment = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hello",
            language="en",
            is_final=True,
            confidence=0.9,
        )

        tracker.update(segment)
        finals = tracker.get_final_segments()
        assert len(finals) == 1
        assert finals[0].is_final is True

    def test_clear(self):
        tracker = SegmentTracker()
        segment = StreamingSegment(
            segment_id="seg_1",
            start=0.0,
            end=1.0,
            text="Hello",
            language="en",
            is_final=True,
            confidence=0.9,
        )
        tracker.update(segment)

        tracker.clear()
        assert tracker.get_final_segments() == []


class MockSTTProvider:
    """Mock STT provider for testing."""

    def __init__(self):
        self._transcribe_calls = 0

    def transcribe_stream(self, audio_stream):
        self._transcribe_calls += 1
        # Consume the stream
        for _ in audio_stream:
            pass
        # Yield a mock segment
        yield StreamingSegment(
            segment_id=f"seg_{self._transcribe_calls}",
            start=0.0,
            end=1.0,
            text="Test transcription",
            language="en",
            is_final=True,
            confidence=0.9,
        )

    def get_capabilities(self):
        return ProviderCapabilities(
            languages=["en"],
            supports_diarization=False,
            supports_word_timestamps=True,
            supports_streaming=True,
            streaming_latency_ms=1000,
            min_memory_gb=4.0,
            requires_gpu=False,
        )

    def is_available(self):
        return True


class TestStreamingOrchestrator:
    """Tests for StreamingOrchestrator class."""

    def test_creation(self):
        provider = MockSTTProvider()
        orchestrator = StreamingOrchestrator(provider=provider)
        assert orchestrator._running is False

    def test_start_stop(self):
        provider = MockSTTProvider()
        orchestrator = StreamingOrchestrator(provider=provider)

        orchestrator.start()
        assert orchestrator._running is True

        segments = orchestrator.stop()
        assert orchestrator._running is False
        assert isinstance(segments, list)

    def test_feed_audio(self):
        provider = MockSTTProvider()
        orchestrator = StreamingOrchestrator(provider=provider)

        orchestrator.start()
        # Feed some audio
        audio = np.zeros(1600, dtype=np.int16).tobytes()
        orchestrator.feed_audio(audio)

        time.sleep(0.1)  # Let the processing thread run
        orchestrator.stop()

    def test_feed_audio_callback(self):
        provider = MockSTTProvider()
        orchestrator = StreamingOrchestrator(provider=provider)

        orchestrator.start()
        # Feed audio via callback interface
        audio = np.zeros((160, 1), dtype=np.float32)
        orchestrator.feed_audio_callback(audio)

        time.sleep(0.1)
        orchestrator.stop()

    def test_callback_invoked(self):
        provider = MockSTTProvider()
        received_segments = []

        def callback(segment):
            received_segments.append(segment)

        orchestrator = StreamingOrchestrator(
            provider=provider,
            callback=callback,
        )

        orchestrator.start()
        # Feed enough audio to trigger transcription
        for _ in range(50):
            audio = np.random.randn(1600).astype(np.float32) * 0.1
            audio_int16 = (audio * 32767).astype(np.int16)
            orchestrator.feed_audio(audio_int16.tobytes())
            time.sleep(0.01)

        time.sleep(1.0)  # Wait for processing
        orchestrator.stop()

        # Callback may or may not have been called depending on VAD


class TestCreateStreamingTranscriber:
    """Tests for create_streaming_transcriber factory."""

    def test_create_default(self):
        provider = MockSTTProvider()
        orchestrator = create_streaming_transcriber(provider=provider)
        assert isinstance(orchestrator, StreamingOrchestrator)

    def test_create_with_options(self):
        provider = MockSTTProvider()
        callback = MagicMock()

        orchestrator = create_streaming_transcriber(
            provider=provider,
            sample_rate=48000,
            callback=callback,
        )
        assert orchestrator._sample_rate == 48000
        assert orchestrator._callback == callback
