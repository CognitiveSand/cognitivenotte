# Conot

Sound acquisition pipeline for cognitive note-taking.

## Installation

```bash
uv sync
```

## Usage

```bash
# List available audio devices
uv run conot list-devices

# Record audio (Ctrl+C to stop)
uv run conot record

# Record with debug view (live audio meters)
uv run conot record --debug

# Record from specific device
uv run conot record --device 1
```

## Configuration

Copy `settings.yml` to customize:

- `audio.sample_rate`: Sample rate in Hz (default: 48000)
- `audio.channels`: Number of channels (default: 1)
- `audio.device_id`: Preferred device ID (default: auto-select)
- `recording.output_dir`: Output directory (default: ./recordings)
- `recording.filename_format`: Filename template with strftime placeholders

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run smoke tests
uv run python scripts/smoke_tests.py

# Lint
uv run ruff check src/
uv run ruff format src/
```
