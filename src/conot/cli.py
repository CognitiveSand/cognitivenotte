"""Command-line interface for Conot."""

import argparse
import signal
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from rich.table import Table

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
