# Conot

**Cognitive Note-Taking** - A local, privacy-first audio recording and speech transcription tool.

Conot captures audio from your microphone and transcribes it to text in real-time using state-of-the-art open-source speech recognition models. All processing happens locally on your machine - no cloud services, no API keys, no data leaving your computer.

## Features

- **Zero-configuration audio capture** - Automatically selects the best microphone
- **Real-time transcription** - Live speech-to-text with ~2s latency (GPU) or ~5s (CPU)
- **Multi-language support** - French and English with automatic language detection
- **Speaker diarization** - Identifies different speakers (Speaker 1, Speaker 2, etc.)
- **Word-level timestamps** - Precise timing for each word
- **Multiple output formats** - JSON, plain text, SRT subtitles
- **LLM integration** - Stream JSONL output to stdout for piping to AI tools
- **100% open-source** - Uses Whisper models, runs entirely offline

## Requirements

- Linux (tested on Ubuntu 22.04+)
- Python 3.12+
- Audio input device (USB microphone recommended)
- For GPU acceleration: NVIDIA GPU with CUDA support (optional but recommended)

## Installation

### Quick Start (CPU only)

```bash
# Clone the repository
git clone https://github.com/yourusername/cognitivenotte.git
cd cognitivenotte

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### With GPU Acceleration (Recommended)

For faster transcription with NVIDIA GPUs:

```bash
# Install with GPU support
uv sync --extra stt-gpu

# Or with pip
pip install -e ".[stt-gpu]"
```

### With Speaker Diarization

To identify different speakers in recordings:

```bash
# Install with diarization support
uv sync --extra stt-diarization

# Or with pip
pip install -e ".[stt-diarization]"
```

**Note:** Pyannote diarization models require a free HuggingFace account. Set your token:
```bash
export HF_TOKEN=your_huggingface_token
```
Or add it to `settings.yml`:
```yaml
stt:
  huggingface_token: your_token_here
```

### Full Installation (All Features)

```bash
# Install everything
uv sync --extra stt-full

# Or with pip
pip install -e ".[stt-full]"
```

### System Dependencies

On Debian/Ubuntu, install audio libraries:
```bash
sudo apt install portaudio19-dev python3-dev
```

On Fedora:
```bash
sudo dnf install portaudio-devel python3-devel
```

## Usage

### List Audio Devices

```bash
conot list-devices
```

Shows all available microphones with their IDs, sample rates, and auto-selection priority.

### Record Audio

```bash
# Basic recording (auto-selects best microphone)
conot record

# Record with live audio meters
conot record --debug

# Record from a specific device
conot record --device 3
```

Press `Ctrl+C` to stop recording. Files are saved to `./recordings/` by default.

### Transcribe Audio Files

```bash
# Transcribe a file (outputs JSON)
conot transcribe recording.wav

# Save to a specific file
conot transcribe recording.wav -o transcript.json

# Output as plain text
conot transcribe recording.wav --format txt

# Output as SRT subtitles
conot transcribe recording.wav --format srt

# Disable speaker diarization
conot transcribe recording.wav --no-diarization

# Force specific language (skip auto-detection)
conot transcribe recording.wav --language fr
```

### Live Transcription

Real-time transcription from your microphone:

```bash
# Basic live transcription
conot transcribe --live

# With debug view (audio meters + transcription)
conot transcribe --live --debug

# Save session to file
conot transcribe --live -o session.json

# Specify allowed languages (improves accuracy)
conot transcribe --live --languages fr,en

# Stream JSONL to stdout (for LLM integration)
conot transcribe --live --stdout
```

### LLM Integration

Stream transcription output to other tools:

```bash
# Pipe to an LLM tool
conot transcribe --live --stdout | your_llm_tool

# Save while processing
conot transcribe --live --stdout | tee transcript.jsonl | your_llm_tool

# Process with jq
conot transcribe --live --stdout | jq -r '.text'
```

Each line is a JSON object:
```json
{"segment_id":"seg_1_abc123","start":0.0,"end":2.5,"text":"Hello world","language":"en","speaker":"Speaker 1","confidence":-0.3}
```

### Model and Hardware Options

```bash
# Force CPU mode (useful if GPU has issues)
conot transcribe --live --compute-device cpu

# Use a specific model size
conot transcribe --live --model-size medium

# Available model sizes: tiny, small, medium, large-v3 (default: auto based on hardware)
```

## Configuration

Create a `settings.yml` file to customize behavior:

```yaml
audio:
  sample_rate: 48000          # Hz (44100 or 48000 recommended)
  channels: 1                 # 1 = mono, 2 = stereo
  device_id: null             # null = auto-select, or specify device ID

recording:
  output_dir: "./recordings"
  filename_format: "recording_%Y%m%d_%H%M%S.wav"

stt:
  provider: auto              # auto | faster-whisper | whisper-cpp
  device: auto                # auto | cuda | cpu
  language: auto              # auto | fr | en
  diarization: true           # Enable speaker identification
  model_size: auto            # auto | large-v3 | medium | small | tiny
  huggingface_token: null     # For pyannote diarization

debug:
  meter_update_interval: 0.05 # Seconds between meter updates
  reference_db: -60.0         # Reference level for dB display
```

## Output Formats

### JSON (default)

```json
{
  "audio_file": "recording.wav",
  "duration_s": 45.2,
  "languages_detected": ["fr", "en"],
  "speakers": ["Speaker 1", "Speaker 2"],
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "speaker": "Speaker 1",
      "language": "fr",
      "text": "Bonjour, comment allez-vous?",
      "confidence": 0.92,
      "words": [...]
    }
  ]
}
```

### Plain Text

```
[Speaker 1]
Bonjour, comment allez-vous?

[Speaker 2]
I'm doing well, thank you!
```

### SRT Subtitles

```
1
00:00:00,000 --> 00:00:03,500
[Speaker 1] Bonjour, comment allez-vous?

2
00:00:04,200 --> 00:00:06,800
[Speaker 2] I'm doing well, thank you!
```

## Transcripts Directory

Live transcriptions are automatically saved to `./transcripts/` with timestamps:
```
transcripts/
  transcript_20260131_143022.json
  transcript_20260131_151847.json
```

Files are saved incrementally after each segment, so you won't lose data if the session is interrupted.

## Hardware Requirements

| Mode | RAM | GPU VRAM | Model | Latency |
|------|-----|----------|-------|---------|
| Edge (CPU) | 4GB+ | - | small/tiny | ~5-10s |
| Standard | 8GB+ | 4-8GB | medium | ~2-3s |
| Enterprise | 16GB+ | 12GB+ | large-v3 | ~1-2s |

The system automatically detects your hardware and selects the appropriate model.

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=conot --cov-report=html

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run mypy src/
```

## Troubleshooting

### No audio devices found

```bash
# Check if PulseAudio/PipeWire is running
pactl list sources short

# Restart audio service
systemctl --user restart pipewire  # or pulseaudio
```

### CUDA errors on GPU system

```bash
# Force CPU mode
conot transcribe --live --compute-device cpu

# Or set in settings.yml
stt:
  device: cpu
```

### Model download issues

Models are downloaded automatically on first use. If you have network issues:

```bash
# Pre-download models manually
python -c "from faster_whisper import WhisperModel; WhisperModel('medium')"
```

### HuggingFace token for diarization

Pyannote models require accepting the license on HuggingFace:
1. Create account at https://huggingface.co
2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Create token at https://huggingface.co/settings/tokens
4. Set `export HF_TOKEN=your_token`

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 acceleration
- [pyannote-audio](https://github.com/pyannote/pyannote-audio) - Speaker diarization
- [sounddevice](https://python-sounddevice.readthedocs.io/) - Audio I/O
