"""Audio device detection and selection."""

import re
import subprocess
from dataclasses import dataclass, field

import sounddevice as sd

from conot.exceptions import DeviceNotFoundError

# Patterns for devices to exclude
EXCLUDE_PATTERNS = [
    r"hdmi",
    r"loopback",
    r"monitor",
    r"digital\s*output",
    r"spdif",
    r"\.monitor$",  # PulseAudio monitor sources
]

# Patterns indicating USB devices (higher quality)
USB_PATTERNS = [
    r"usb",
    r"blue\s*(yeti|snowball)",
    r"rode",
    r"shure",
    r"audio.technica",
    r"focusrite",
    r"scarlett",
    r"behringer",
    r"zoom\s*h[0-9]",
]

# Patterns indicating Bluetooth devices
BLUETOOTH_PATTERNS = [
    r"bluez",
    r"bluetooth",
    r"shokz",
    r"airpods",
    r"jabra",
    r"sony\s*wh",
    r"bose",
]

# Patterns indicating built-in microphones
BUILTIN_PATTERNS = [
    r"built.?in",
    r"internal",
    r"integrated",
    r"laptop",
    r"webcam",
]


@dataclass
class AudioDevice:
    """Represents an audio input device."""

    id: int
    name: str
    channels: int
    sample_rate: float
    is_default: bool
    priority: int = 0
    pulse_source: str | None = field(default=None)  # PulseAudio source name for BT devices

    def __str__(self) -> str:
        default_marker = " (default)" if self.is_default else ""
        bt_marker = " [BT]" if self.pulse_source and "bluez" in self.pulse_source else ""
        rate = int(self.sample_rate)
        return f"[{self.id}] {self.name}{default_marker}{bt_marker} - {self.channels}ch @ {rate}Hz"


def _calculate_priority(device_name: str, is_default: bool, sample_rate: float) -> int:
    """Calculate device priority score.

    Higher score = better device.

    Priority factors:
    - USB microphones: +100 (better quality)
    - Bluetooth devices: +80 (good for mobility)
    - Built-in mics: +50
    - System default: +30 bonus
    - Standard sample rate (44100/48000): +10
    """
    name_lower = device_name.lower()
    priority = 0

    # Check for USB/external microphones
    for pattern in USB_PATTERNS:
        if re.search(pattern, name_lower):
            priority += 100
            break

    # Check for Bluetooth devices
    if priority == 0:
        for pattern in BLUETOOTH_PATTERNS:
            if re.search(pattern, name_lower):
                priority += 80
                break

    # Check for built-in microphones
    if priority == 0:  # Only if not already identified as USB or Bluetooth
        for pattern in BUILTIN_PATTERNS:
            if re.search(pattern, name_lower):
                priority += 50
                break

    # Default device bonus
    if is_default:
        priority += 30

    # Standard sample rate bonus
    if sample_rate in (44100.0, 48000.0):
        priority += 10

    return priority


def _should_exclude(device_name: str) -> bool:
    """Check if device should be excluded from selection."""
    name_lower = device_name.lower()
    return any(re.search(pattern, name_lower) for pattern in EXCLUDE_PATTERNS)


@dataclass
class PulseSource:
    """PulseAudio/PipeWire source info."""

    name: str
    description: str
    sample_rate: int
    channels: int


def _query_pulse_sources() -> list[PulseSource]:
    """Query PulseAudio/PipeWire for available input sources.

    Returns:
        List of PulseSource objects, or empty list if pactl unavailable
    """
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    sources: list[PulseSource] = []
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue

        name = parts[1]
        # Parse format like "s16le 1ch 16000Hz" or "float32le 1ch 48000Hz"
        fmt = parts[4]
        channels = 1
        sample_rate = 48000

        ch_match = re.search(r"(\d+)ch", fmt)
        if ch_match:
            channels = int(ch_match.group(1))

        rate_match = re.search(r"(\d+)Hz", fmt)
        if rate_match:
            sample_rate = int(rate_match.group(1))

        # Get description via pactl
        desc = name
        try:
            desc_result = subprocess.run(
                ["pactl", "list", "sources"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            # Find description for this source
            in_source = False
            for desc_line in desc_result.stdout.split("\n"):
                if f"Name: {name}" in desc_line:
                    in_source = True
                elif in_source and "Description:" in desc_line:
                    desc = desc_line.split("Description:", 1)[1].strip()
                    break
                elif in_source and desc_line.startswith("Source #"):
                    break
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        sources.append(
            PulseSource(name=name, description=desc, sample_rate=sample_rate, channels=channels)
        )

    return sources


def list_input_devices(include_bluetooth: bool = True) -> list[AudioDevice]:
    """List all available audio input devices.

    Args:
        include_bluetooth: If True, also query PulseAudio for Bluetooth devices

    Returns:
        List of AudioDevice objects for all input devices
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]

    input_devices: list[AudioDevice] = []
    seen_names: set[str] = set()

    for idx, dev in enumerate(devices):  # type: ignore[arg-type]
        # Skip output-only devices
        if dev["max_input_channels"] <= 0:  # type: ignore[index]
            continue

        name: str = dev["name"]  # type: ignore[index]
        channels: int = dev["max_input_channels"]  # type: ignore[index]
        sample_rate: float = dev["default_samplerate"]  # type: ignore[index]
        is_default = idx == default_input

        priority = _calculate_priority(name, is_default, sample_rate)
        seen_names.add(name.lower())

        input_devices.append(
            AudioDevice(
                id=idx,
                name=name,
                channels=channels,
                sample_rate=sample_rate,
                is_default=is_default,
                priority=priority,
            )
        )

    # Query PulseAudio for Bluetooth and other sources not visible to sounddevice
    if include_bluetooth:
        pulse_sources = _query_pulse_sources()

        # Use negative IDs for pulse-only sources (starting from -1)
        pulse_id = -1
        for src in pulse_sources:
            # Skip if already seen or is a monitor
            if _should_exclude(src.name):
                continue

            # Check if this is a Bluetooth source
            is_bluetooth = "bluez" in src.name.lower()
            if not is_bluetooth:
                continue

            # Create a friendly name from the description
            name = src.description
            if not name or name == src.name:
                # Extract device name from bluez source name
                # e.g., "bluez_input.A8:F5:E1:0B:25:3C" -> use description
                name = f"Bluetooth: {src.name.split('.')[-1]}"

            priority = _calculate_priority(name, False, float(src.sample_rate))

            input_devices.append(
                AudioDevice(
                    id=pulse_id,
                    name=name,
                    channels=src.channels,
                    sample_rate=float(src.sample_rate),
                    is_default=False,
                    priority=priority,
                    pulse_source=src.name,
                )
            )
            pulse_id -= 1

    return input_devices


def select_best_device(devices: list[AudioDevice] | None = None) -> AudioDevice:
    """Select the best available input device.

    Uses priority scoring to select the most suitable device:
    - USB microphones preferred over built-in
    - System default gets a bonus
    - Standard sample rates preferred

    Args:
        devices: Optional list of devices to choose from.
                 If None, queries system devices.

    Returns:
        Best available AudioDevice

    Raises:
        DeviceNotFoundError: If no suitable input device found
    """
    if devices is None:
        devices = list_input_devices()

    # Filter out excluded devices
    suitable = [d for d in devices if not _should_exclude(d.name)]

    if not suitable:
        # Fall back to all devices if filtering removed everything
        suitable = devices

    if not suitable:
        raise DeviceNotFoundError("No audio input devices found")

    # Sort by priority (highest first), then by ID (lower is usually better)
    suitable.sort(key=lambda d: (-d.priority, d.id))

    return suitable[0]


def get_device_by_id(device_id: int) -> AudioDevice:
    """Get a specific device by ID.

    Args:
        device_id: Device index (negative IDs are Bluetooth/PulseAudio sources)

    Returns:
        AudioDevice for the specified ID

    Raises:
        DeviceNotFoundError: If device ID is invalid or not an input device
    """
    # Negative IDs are PulseAudio-only sources (like Bluetooth)
    if device_id < 0:
        all_devices = list_input_devices(include_bluetooth=True)
        for dev in all_devices:
            if dev.id == device_id:
                return dev
        raise DeviceNotFoundError(f"Bluetooth device {device_id} not found")

    try:
        dev = sd.query_devices(device_id)
    except sd.PortAudioError as e:
        raise DeviceNotFoundError(f"Device {device_id} not found: {e}") from e

    if dev["max_input_channels"] <= 0:  # type: ignore[index]
        raise DeviceNotFoundError(f"Device {device_id} is not an input device")

    default_input = sd.default.device[0]
    is_default = device_id == default_input

    name: str = dev["name"]  # type: ignore[index]
    channels: int = dev["max_input_channels"]  # type: ignore[index]
    sample_rate: float = dev["default_samplerate"]  # type: ignore[index]

    return AudioDevice(
        id=device_id,
        name=name,
        channels=channels,
        sample_rate=sample_rate,
        is_default=is_default,
        priority=_calculate_priority(name, is_default, sample_rate),
    )
