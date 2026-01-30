"""Debug terminal UI with live audio meters."""

from threading import Lock

import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from conot.audio_meter import AudioMeter
from conot.devices import AudioDevice


class DebugView:
    """Terminal UI for debugging audio capture."""

    def __init__(
        self,
        device: AudioDevice,
        reference_db: float = -60.0,
        sample_rate: int = 16000,
        chunk_size: int = 1600,
    ) -> None:
        """Initialize debug view.

        Args:
            device: Audio device being used
            reference_db: Reference dB level for meters
            sample_rate: Audio sample rate in Hz
            chunk_size: Audio chunk size in samples
        """
        self._device = device
        self._meter = AudioMeter(reference_db=reference_db)
        self._lock = Lock()
        self._duration: float = 0.0
        self._is_recording: bool = False
        self._console = Console()
        self._live: Live | None = None
        self._sample_rate = sample_rate
        self._chunk_size = chunk_size
        self._extra_info: str = ""  # For additional status info
        self._transcription_lines: list[tuple[str, str, str]] = []  # (speaker, lang, text)
        self._max_transcription_lines = 8  # Max lines to show

    def _format_duration(self, seconds: float) -> str:
        """Format duration as HH:MM:SS."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _create_meter_bar(self, normalized: float, db: float, label: str, width: int = 30) -> Text:
        """Create a text-based meter bar.

        Args:
            normalized: 0-1 normalized level
            db: dB value for display
            label: Meter label (e.g., "RMS", "Peak")
            width: Bar width in characters

        Returns:
            Rich Text object with the meter
        """
        filled = int(normalized * width)
        empty = width - filled

        # Color based on level
        if normalized > 0.9:
            color = "red"
        elif normalized > 0.7:
            color = "yellow"
        else:
            color = "green"

        bar = Text()
        bar.append(f"  {label}:  [")
        bar.append("█" * filled, style=color)
        bar.append("░" * empty, style="dim")
        bar.append(f"] {db:+6.1f} dB")

        return bar

    def _render(self) -> Panel:
        """Render the debug view panel."""
        with self._lock:
            rms_norm = self._meter.rms_normalized
            peak_norm = self._meter.peak_normalized
            rms_db = self._meter.rms_db
            peak_db = self._meter.peak_db
            duration = self._duration
            is_recording = self._is_recording
            extra_info = self._extra_info

        # Status
        status = "Recording" if is_recording else "Stopped"
        status_color = "green" if is_recording else "red"

        # Build content
        lines: list[Text] = []

        # Device info
        device_text = Text()
        device_text.append("  Device: ", style="bold")
        device_text.append(self._device.name)
        if self._device.is_default:
            device_text.append(" (default)", style="dim")
        lines.append(device_text)

        # Audio settings
        settings_text = Text()
        settings_text.append("  Audio:  ", style="bold")
        chunk_ms = int(self._chunk_size * 1000 / self._sample_rate)
        settings_text.append(f"{self._sample_rate} Hz, {self._chunk_size} samples ({chunk_ms}ms)", style="dim")
        lines.append(settings_text)

        # Status
        status_text = Text()
        status_text.append("  Status: ", style="bold")
        status_text.append(status, style=status_color)
        lines.append(status_text)

        # Duration
        duration_text = Text()
        duration_text.append("  Duration: ", style="bold")
        duration_text.append(self._format_duration(duration))
        lines.append(duration_text)

        # Extra info (if any)
        if extra_info:
            info_text = Text()
            info_text.append("  Info: ", style="bold")
            info_text.append(extra_info, style="cyan")
            lines.append(info_text)

        # Empty line
        lines.append(Text())

        # Meters
        lines.append(self._create_meter_bar(rms_norm, rms_db, "RMS "))
        lines.append(self._create_meter_bar(peak_norm, peak_db, "Peak"))

        # Transcription section (if any)
        with self._lock:
            transcription_lines = list(self._transcription_lines)

        if transcription_lines:
            lines.append(Text())
            lines.append(Text("  ─── Transcription ───", style="dim"))
            for speaker, lang, text in transcription_lines:
                line = Text()
                line.append("  ")
                if speaker:
                    line.append(f"[{speaker}] ", style="cyan")
                if lang:
                    line.append(f"({lang}) ", style="dim")
                # Truncate long lines
                display_text = text[:70] + "..." if len(text) > 70 else text
                line.append(display_text)
                lines.append(line)

        # Combine all lines
        content = Text("\n").join(lines)

        return Panel(
            content,
            title="conot debug",
            border_style="blue",
            padding=(0, 1),
        )

    def start(self) -> None:
        """Start the live display."""
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=20,
            transient=True,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def update(
        self,
        audio_data: NDArray[np.float32] | None = None,
        duration: float | None = None,
        is_recording: bool | None = None,
        extra_info: str | None = None,
    ) -> None:
        """Update the display with new data.

        Args:
            audio_data: New audio chunk for meter update
            duration: Current recording duration
            is_recording: Recording status
            extra_info: Additional status info to display
        """
        with self._lock:
            if audio_data is not None:
                self._meter.update(audio_data)
            if duration is not None:
                self._duration = duration
            if is_recording is not None:
                self._is_recording = is_recording
            if extra_info is not None:
                self._extra_info = extra_info

        if self._live is not None:
            self._live.update(self._render())

    def add_transcription(self, text: str, language: str = "", speaker: str = "") -> None:
        """Add a transcription line to display.

        Args:
            text: Transcribed text
            language: Language code (e.g., "fr", "en")
            speaker: Speaker ID (e.g., "SPEAKER_00")
        """
        with self._lock:
            self._transcription_lines.append((speaker, language, text))
            # Keep only the last N lines
            if len(self._transcription_lines) > self._max_transcription_lines:
                self._transcription_lines = self._transcription_lines[-self._max_transcription_lines:]

        if self._live is not None:
            self._live.update(self._render())

    def __enter__(self) -> "DebugView":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.stop()
