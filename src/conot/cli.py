"""Command-line interface for Conot."""

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from conot.config import Settings, get_settings
from conot.debug_view import DebugView
from conot.devices import AudioDevice, get_device_by_id, list_input_devices, select_best_device
from conot.exceptions import ConotError, DeviceNotFoundError
from conot.recorder import AudioRecorder

console = Console()

T = TypeVar("T")


def _get_audio_device(device_id: int | None) -> AudioDevice | None:
    """Get audio device by ID or auto-select.

    Args:
        device_id: Device ID or None for auto-selection.

    Returns:
        AudioDevice if found, None on error (error already printed).
    """
    if device_id is not None:
        try:
            return get_device_by_id(device_id)
        except DeviceNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return None
    else:
        try:
            return select_best_device()
        except DeviceNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return None


class SignalHandler:
    """Reusable signal handler for graceful stop."""

    def __init__(self) -> None:
        self.stop_requested = False

    def __call__(self, signum: int, frame: object) -> None:
        self.stop_requested = True

    def install(self) -> None:
        """Install signal handlers for SIGINT and SIGTERM."""
        signal.signal(signal.SIGINT, self)
        signal.signal(signal.SIGTERM, self)

    def should_stop(self) -> bool:
        """Check if stop was requested."""
        return self.stop_requested


def cmd_list_devices(args: argparse.Namespace) -> int:
    """List available audio input devices."""
    try:
        devices = list_input_devices()
    except Exception as e:
        console.print(f"[red]Error listing devices:[/red] {e}")
        return 1

    if not devices:
        console.print("[yellow]No audio input devices found.[/yellow]")
        return 0

    table = Table(title="Audio Input Devices")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Channels", justify="center")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Priority", justify="right")
    table.add_column("", justify="center")

    for dev in sorted(devices, key=lambda d: -d.priority):
        default_marker = "★" if dev.is_default else ""
        table.add_row(
            str(dev.id),
            dev.name,
            str(dev.channels),
            f"{int(dev.sample_rate)} Hz",
            str(dev.priority),
            default_marker,
        )

    console.print(table)
    console.print("\n[dim]★ = system default[/dim]")

    # Show auto-selected device
    try:
        best = select_best_device(devices)
        console.print(f"\n[green]Auto-selected:[/green] [{best.id}] {best.name}")
    except DeviceNotFoundError:
        pass

    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record audio from microphone."""
    settings = get_settings(Path(args.config) if args.config else None)

    # Get device
    device = _get_audio_device(args.device)
    if device is None:
        return 1

    recorder = AudioRecorder(device=device, settings=settings)

    # Setup signal handler for graceful stop
    sig_handler = SignalHandler()
    sig_handler.install()

    if args.debug:
        return _record_with_debug(recorder, device, settings, sig_handler.should_stop)
    else:
        return _record_simple(recorder, device, sig_handler.should_stop)


def _record_simple(
    recorder: AudioRecorder,
    device: AudioDevice,
    should_stop: Callable[[], bool],
) -> int:
    """Record without debug UI."""
    console.print(f"[bold]Recording from:[/bold] {device.name}")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        recorder.start()
    except ConotError as e:
        console.print(f"[red]Error starting recording:[/red] {e}")
        return 1

    console.print("[green]● Recording...[/green]")

    # Wait for stop signal
    while not should_stop():
        time.sleep(0.1)

    console.print("\n[yellow]Stopping...[/yellow]")

    try:
        output_path = recorder.stop()
        console.print(f"\n[green]✓ Saved:[/green] {output_path}")
        return 0
    except ConotError as e:
        console.print(f"[red]Error saving recording:[/red] {e}")
        return 1


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe audio from file or microphone."""
    if args.live:
        return _transcribe_live(args)
    else:
        return _transcribe_file(args)


def _transcribe_file(args: argparse.Namespace) -> int:
    """Transcribe an audio file."""
    from conot.stt.exceptions import STTError
    from conot.stt.transcribe import transcribe_audio

    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        console.print(f"[red]Error:[/red] Audio file not found: {audio_path}")
        return 1

    output_path = Path(args.output) if args.output else None
    output_format = args.format or "json"
    enable_diarization = not args.no_diarization

    console.print(f"[bold]Transcribing:[/bold] {audio_path}")
    if enable_diarization:
        console.print("[dim]Speaker diarization enabled[/dim]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task_id = progress.add_task("Loading model...", total=None)

            def on_progress(stage: str, pct: float) -> None:
                progress.update(task_id, description=f"{stage}...")

            # Handle device/model args
            compute_device = getattr(args, "compute_device", "auto")
            if compute_device == "auto":
                compute_device = None
            model_size = getattr(args, "model_size", "auto")
            if model_size == "auto":
                model_size = None

            result = transcribe_audio(
                audio_path=audio_path,
                provider=args.provider if hasattr(args, "provider") else None,
                device=compute_device,
                model_size=model_size,
                enable_diarization=enable_diarization,
                progress_callback=on_progress,
            )

        # Format output
        if output_format == "json":
            output = result.to_json()
        elif output_format == "txt":
            output = result.to_text()
        elif output_format == "srt":
            output = result.to_srt()
        else:
            console.print(f"[red]Error:[/red] Unknown format: {output_format}")
            return 1

        # Write or print output
        if output_path:
            output_path.write_text(output, encoding="utf-8")
            console.print(f"[green]✓ Saved:[/green] {output_path}")
        else:
            console.print(output)

        # Print summary
        if output_path or output_format != "json":
            n_segments = len(result.segments)
            n_speakers = len(result.speakers) if result.speakers else 0
            if result.languages_detected:
                languages = ", ".join(result.languages_detected)
            else:
                languages = "unknown"
            console.print(f"\n[dim]Duration: {result.duration_s:.1f}s | "
                          f"Segments: {n_segments} | "
                          f"Speakers: {n_speakers} | "
                          f"Languages: {languages}[/dim]")

        return 0

    except STTError as e:
        console.print(f"[red]Transcription error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


def _transcribe_live(args: argparse.Namespace) -> int:
    """Live transcription from microphone."""
    from conot.stt.exceptions import STTError
    from conot.stt.models import StreamingSegment
    from conot.stt.registry import get_provider
    from conot.stt.streaming import create_streaming_transcriber

    output_path = Path(args.output) if args.output else None
    device_id = args.device if hasattr(args, "device") else None
    use_debug = args.debug if hasattr(args, "debug") else False

    # Get audio device
    device = _get_audio_device(device_id)
    if device is None:
        return 1

    # Storage for segments
    all_segments: list[StreamingSegment] = []
    segment_count = 0

    # Setup signal handler
    sig_handler = SignalHandler()
    sig_handler.install()

    # Audio settings - use device's native sample rate
    device_sample_rate = int(device.sample_rate)
    stt_sample_rate = 16000  # STT providers expect 16kHz
    chunk_duration_s = 0.1  # 100ms blocks
    chunk_size = int(device_sample_rate * chunk_duration_s)

    try:
        import sounddevice as sd

        # Get STT provider
        provider_name = args.provider if hasattr(args, "provider") else None
        provider = get_provider(provider_name)

        # Setup debug view if requested
        debug_view: DebugView | None = None
        if use_debug:
            settings = get_settings(Path(args.config) if args.config else None)
            debug_view = DebugView(
                device=device,
                reference_db=settings.debug.reference_db,
                sample_rate=device_sample_rate,
                chunk_size=chunk_size,
            )

        # Track languages detected
        languages_seen: set[str] = set()

        def on_segment(segment: StreamingSegment) -> None:
            nonlocal segment_count
            all_segments.append(segment)
            if segment.language:
                languages_seen.add(segment.language)
            if segment.is_final:
                segment_count += 1
                if debug_view:
                    lang_info = ",".join(sorted(languages_seen)) if languages_seen else "?"
                    debug_view.update(extra_info=f"{segment_count} segs | Lang: {lang_info}")
                else:
                    # Print inline when not in debug mode with clear formatting
                    # Format: [speaker] [lang] text
                    speaker_prefix = f"[cyan][{segment.speaker}][/cyan] " if segment.speaker else ""
                    lang_tag = f"[dim]({segment.language})[/dim] " if segment.language else ""
                    console.print(f"{speaker_prefix}{lang_tag}{segment.text}")

        def on_audio(audio_data: object) -> None:
            if debug_view:
                debug_view.update(audio_data=audio_data)  # type: ignore[arg-type]

        # Create streaming orchestrator (expects 16kHz audio)
        orchestrator = create_streaming_transcriber(
            provider=provider,
            sample_rate=stt_sample_rate,
            callback=on_segment,
            audio_callback=on_audio if debug_view else None,
        )

        # Resampler for converting device sample rate to STT sample rate
        import numpy as np
        from scipy import signal as scipy_signal

        resample_ratio = stt_sample_rate / device_sample_rate

        def audio_callback_with_resample(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: object,
        ) -> None:
            """Resample audio and feed to orchestrator."""
            if status:
                pass  # Could log status

            # Flatten to mono if needed
            audio = indata[:, 0] if indata.ndim > 1 else indata.flatten()

            # Resample to 16kHz if needed
            if device_sample_rate != stt_sample_rate:
                num_samples = int(len(audio) * resample_ratio)
                audio_resampled = scipy_signal.resample(audio, num_samples).astype(np.float32)
            else:
                audio_resampled = audio

            # Feed to orchestrator
            orchestrator.feed_audio_callback(audio_resampled)

        # Create audio stream at device's native sample rate
        stream = sd.InputStream(
            samplerate=device_sample_rate,
            channels=1,
            dtype="float32",
            device=device.id,
            callback=audio_callback_with_resample,
            blocksize=chunk_size,
        )

        console.print(f"[bold]Live transcription[/bold]")
        console.print(f"[dim]Device: {device.name}[/dim]")
        console.print("[dim]Press Ctrl+C to stop[/dim]\n")

        # Start everything
        orchestrator.start()
        stream.start()

        if debug_view:
            debug_view.update(is_recording=True)
            with debug_view:
                start_time = time.time()
                while not sig_handler.stop_requested:
                    time.sleep(0.1)
                    # Get debug stats from orchestrator
                    stats = orchestrator.debug_stats
                    lang_info = ",".join(sorted(languages_seen)) if languages_seen else "?"
                    info = (
                        f"RMS={stats['last_rms']} | "
                        f"Speech={'yes' if stats['speech'] else 'no'} | "
                        f"Lang: {lang_info} | "
                        f"{segment_count} segs"
                    )
                    debug_view.update(duration=time.time() - start_time, extra_info=info)
        else:
            console.print("[green]● Listening...[/green]\n")
            while not sig_handler.stop_requested:
                time.sleep(0.1)

        console.print("\n[yellow]Stopping...[/yellow]")

        # Stop audio stream
        stream.stop()
        stream.close()

        # Stop orchestrator and get final segments
        final_segments = orchestrator.stop()

        # Save output if requested
        if output_path:
            output_data = {
                "languages_detected": sorted(languages_seen),
                "segments": [
                    {
                        "segment_id": s.segment_id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "language": s.language,
                        "speaker": s.speaker,
                        "confidence": s.confidence,
                    }
                    for s in final_segments
                ]
            }
            output_path.write_text(
                json.dumps(output_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            console.print(f"\n[green]✓ Saved:[/green] {output_path}")

        # Summary
        lang_summary = ", ".join(sorted(languages_seen)) if languages_seen else "unknown"
        console.print(f"\n[dim]Total segments: {len(final_segments)} | Languages: {lang_summary}[/dim]")
        return 0

    except STTError as e:
        console.print(f"[red]Transcription error:[/red] {e}")
        return 1
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 1


def _record_with_debug(
    recorder: AudioRecorder,
    device: AudioDevice,
    settings: Settings,
    should_stop: Callable[[], bool],
) -> int:
    """Record with debug UI showing meters."""
    # Get audio settings
    sample_rate = settings.audio.sample_rate
    # Default chunk size (sounddevice default is ~100ms)
    chunk_samples = int(sample_rate * 0.1)

    debug_view = DebugView(
        device=device,
        reference_db=settings.debug.reference_db,
        sample_rate=sample_rate,
        chunk_size=chunk_samples,
    )

    # Connect meter callback
    def on_audio(data: object) -> None:
        debug_view.update(
            audio_data=data,  # type: ignore[arg-type]
            duration=recorder.duration,
            is_recording=recorder.is_recording,
        )

    recorder.set_audio_callback(on_audio)

    try:
        recorder.start()
    except ConotError as e:
        console.print(f"[red]Error starting recording:[/red] {e}")
        return 1

    debug_view.update(is_recording=True)

    with debug_view:
        while not should_stop():
            time.sleep(settings.debug.meter_update_interval)
            debug_view.update(duration=recorder.duration)

    debug_view.update(is_recording=False)

    try:
        output_path = recorder.stop()
        console.print(f"\n[green]✓ Saved:[/green] {output_path}")
        return 0
    except ConotError as e:
        console.print(f"[red]Error saving recording:[/red] {e}")
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="conot",
        description="Sound acquisition for cognitive note-taking",
    )
    parser.add_argument(
        "--config",
        "-c",
        help="Path to settings.yml config file",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list-devices command
    list_parser = subparsers.add_parser(
        "list-devices",
        help="List available audio input devices",
    )
    list_parser.set_defaults(func=cmd_list_devices)

    # record command
    record_parser = subparsers.add_parser(
        "record",
        help="Record audio from microphone",
    )
    record_parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show debug view with audio level meters",
    )
    record_parser.add_argument(
        "--device",
        type=int,
        help="Device ID to use (from list-devices)",
    )
    record_parser.set_defaults(func=cmd_record)

    # transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Transcribe audio from file or microphone",
    )
    transcribe_parser.add_argument(
        "audio_file",
        nargs="?",
        help="Audio file to transcribe (omit for --live mode)",
    )
    transcribe_parser.add_argument(
        "--live",
        action="store_true",
        help="Live transcription from microphone",
    )
    transcribe_parser.add_argument(
        "--output",
        "-o",
        help="Output file path (default: stdout)",
    )
    transcribe_parser.add_argument(
        "--format",
        "-f",
        choices=["json", "txt", "srt"],
        default="json",
        help="Output format (default: json)",
    )
    transcribe_parser.add_argument(
        "--no-diarization",
        action="store_true",
        help="Disable speaker diarization",
    )
    transcribe_parser.add_argument(
        "--provider",
        choices=["auto", "faster-whisper", "whisper-cpp"],
        default="auto",
        help="STT provider (default: auto)",
    )
    transcribe_parser.add_argument(
        "--compute-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device for transcription (default: auto)",
    )
    transcribe_parser.add_argument(
        "--model-size",
        choices=["auto", "large-v3", "medium", "small", "tiny"],
        default="auto",
        help="Whisper model size (default: auto based on hardware)",
    )
    transcribe_parser.add_argument(
        "--device",
        type=int,
        help="Audio input device ID (for --live mode)",
    )
    transcribe_parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Show debug information",
    )
    transcribe_parser.set_defaults(func=cmd_transcribe)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
