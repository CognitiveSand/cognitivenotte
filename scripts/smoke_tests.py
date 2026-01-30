#!/usr/bin/env python3
"""Smoke tests for Conot."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and check for success."""
    print(f"\n{'='*60}")
    print(f"TEST: {description}")
    print(f"CMD:  {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    if result.returncode == 0:
        print(f"✓ PASSED: {description}")
        return True
    else:
        print(f"✗ FAILED: {description} (exit code: {result.returncode})")
        return False


def test_import() -> bool:
    """Test that conot can be imported."""
    try:
        import conot

        print(f"✓ Import successful, version: {conot.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_config_loading() -> bool:
    """Test configuration loading."""
    try:
        from conot.config import Settings

        settings = Settings()
        assert settings.audio.sample_rate == 48000
        assert settings.audio.channels == 1
        print("✓ Config loading successful")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_device_listing() -> bool:
    """Test device enumeration."""
    try:
        from conot.devices import list_input_devices

        devices = list_input_devices()
        print(f"✓ Found {len(devices)} input device(s)")
        for dev in devices:
            print(f"  - {dev}")
        return True
    except Exception as e:
        print(f"✗ Device listing failed: {e}")
        return False


def test_audio_meter() -> bool:
    """Test audio meter calculations."""
    try:
        import numpy as np

        from conot.audio_meter import AudioMeter

        meter = AudioMeter(reference_db=-60.0)

        # Test with silence
        silence = np.zeros(1024, dtype=np.float32)
        meter.update(silence)
        assert meter.rms_db == -60.0
        assert meter.peak_db == -60.0

        # Test with full scale
        full_scale = np.ones(1024, dtype=np.float32)
        meter.update(full_scale)
        assert abs(meter.rms_db - 0.0) < 0.1
        assert abs(meter.peak_db - 0.0) < 0.1

        # Test with half scale
        half_scale = np.ones(1024, dtype=np.float32) * 0.5
        meter.update(half_scale)
        assert abs(meter.rms_db - (-6.0)) < 0.5  # -6 dB for half amplitude

        print("✓ Audio meter calculations correct")
        return True
    except Exception as e:
        print(f"✗ Audio meter test failed: {e}")
        return False


def test_cli_help() -> bool:
    """Test CLI help output."""
    return run_command(
        [sys.executable, "-m", "conot", "--help"],
        "CLI help output",
    )


def test_cli_list_devices() -> bool:
    """Test CLI list-devices command."""
    return run_command(
        [sys.executable, "-m", "conot", "list-devices"],
        "CLI list-devices command",
    )


def main() -> int:
    """Run all smoke tests."""
    print("\n" + "=" * 60)
    print("CONOT SMOKE TESTS")
    print("=" * 60)

    tests = [
        test_import,
        test_config_loading,
        test_device_listing,
        test_audio_meter,
        test_cli_help,
        test_cli_list_devices,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
