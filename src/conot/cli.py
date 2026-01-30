"""Command-line interface for Conot."""

import argparse
import json
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

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
    device: AudioDevice
    if args.device is not None:
        try:
            device = get_device_by_id(args.device)
        except DeviceNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
    else:
        try:
            device = select_best_device()
        except DeviceNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1

    recorder = AudioRecorder(device=device, settings=settings)

    # Setup signal handler for graceful stop
    stop_requested = False

    def signal_handler(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.debug:
        return _record_with_debug(recorder, device, settings, lambda: stop_requested)
    else:
        return _record_simple(recorder, device, lambda: stop_requested)


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

            result = transcribe_audio(
                audio_path=audio_path,
                provider=args.provider if hasattr(args, "provider") else None,
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
    from conot.stt.transcribe import create_live_transcriber

    output_path = Path(args.output) if args.output else None
    device_id = args.device if hasattr(args, "device") else None

    # Storage for segments
    all_segments: list[StreamingSegment] = []
    display_text = Text()

    def on_segment(segment: StreamingSegment) -> None:
        all_segments.append(segment)
        # Update display
        if segment.is_final:
            display_text.append(f"[{segment.language}] ", style="dim")
            display_text.append(segment.text + "\n")

    # Setup signal handler
    stop_requested = False

    def signal_handler(signum: int, frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print("[bold]Live transcription[/bold]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        transcriber = create_live_transcriber(
            provider=args.provider if hasattr(args, "provider") else None,
            device_id=device_id,
            callback=on_segment,
        )

        transcriber.start()
        console.print("[green]● Listening...[/green]\n")

        # Use Live display for streaming output
        panel = Panel(display_text, title="Transcription")
        with Live(panel, refresh_per_second=4, console=console) as live:
            while not stop_requested:
                time.sleep(0.1)
                live.update(Panel(display_text, title="Transcription"))

        console.print("\n[yellow]Stopping...[/yellow]")
        final_segments = transcriber.stop()

        # Save output if requested
        if output_path:
            # Convert to JSON
            output_data = {
                "segments": [
                    {
                        "segment_id": s.segment_id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "language": s.language,
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

        console.print(f"\n[dim]Total segments: {len(final_segments)}[/dim]")
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
    debug_view = DebugView(
        device=device,
        reference_db=settings.debug.reference_db,
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
