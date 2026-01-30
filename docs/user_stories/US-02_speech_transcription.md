# US-02: Speech Transcription

| Field | Value |
|-------|-------|
| **ID** | US-02 |
| **Title** | Speech Transcription |
| **Status** | Draft |
| **Priority** | Must |
| **Created** | 2026-01-30 |

---

## User Story

**As a** user taking notes during meetings,
**I want** my recorded audio to be automatically transcribed to text,
**So that** I can search, review, and share my notes without manual transcription.

---

## Scenario: Alice's Two Meetings

### Meeting 1: Desktop PC with NVIDIA GPU

Alice has a meeting with Bob. The conversation switches between **French and English** because Bob doesn't speak English well. Alice records the meeting on her **desktop PC** with a strong NVIDIA GPU (RTX 4090, 24GB VRAM).

```bash
# Alice records the meeting
conot record
# [Ctrl+C to stop]
# Saved: recordings/recording_20260130_143022.wav

# Alice transcribes - NO CONFIG NEEDED
conot transcribe recordings/recording_20260130_143022.wav
```

**What happens automatically:**
1. System detects NVIDIA GPU with 24GB VRAM
2. Selects **enterprise-tier provider** for best accuracy
3. Detects mixed French/English speech
4. Identifies speakers (Alice, Bob)
5. Produces full diarized transcript

**Expected Output:**
```json
{
  "audio_file": "recording_20260130_143022.wav",
  "duration_s": 1847.5,
  "languages_detected": ["fr", "en"],
  "speakers": ["Speaker_1", "Speaker_2"],
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "speaker": "Speaker_1",
      "language": "fr",
      "text": "Bonjour Bob, merci d'avoir pris le temps de me rencontrer.",
      "words": [
        {"word": "Bonjour", "start": 0.0, "end": 0.45, "confidence": 0.98},
        {"word": "Bob,", "start": 0.50, "end": 0.72, "confidence": 0.99},
        ...
      ]
    },
    {
      "start": 3.5,
      "end": 7.8,
      "speaker": "Speaker_2",
      "language": "fr",
      "text": "Bonjour Alice. Oui, c'est important ce projet.",
      "words": [...]
    },
    {
      "start": 8.1,
      "end": 12.4,
      "speaker": "Speaker_1",
      "language": "en",
      "text": "Let me explain the technical requirements.",
      "words": [...]
    },
    {
      "start": 12.8,
      "end": 18.2,
      "speaker": "Speaker_2",
      "language": "fr",
      "text": "Pardon, je ne comprends pas bien l'anglais. En français ?",
      "words": [...]
    },
    {
      "start": 18.5,
      "end": 24.1,
      "speaker": "Speaker_1",
      "language": "fr",
      "text": "Bien sûr. Les exigences techniques sont les suivantes...",
      "words": [...]
    }
  ]
}
```

### Meeting 2: Laptop without GPU

Later, Alice has another meeting on her **laptop** (8GB RAM, no GPU, Intel CPU only). She records and transcribes the same way:

```bash
conot record
conot transcribe recordings/recording_20260130_163045.wav
```

**What happens automatically:**
1. System detects no GPU, 8GB RAM available
2. Selects **edge/nano provider** optimized for CPU
3. Uses smaller model that fits in memory
4. Still produces full diarized transcript with speakers and languages
5. Processing takes longer but output format is identical

### Key Point: Zero Configuration

Alice never touches `settings.yml`. The system adapts to her hardware automatically.
Same command, same output format, different hardware = different provider selected transparently.

---

## Acceptance Criteria

### AC-01: Basic Transcription
- [ ] Given a recorded audio file, when I run transcription, then I receive a text transcript
- [ ] Transcription works offline without internet connection
- [ ] Output includes full text and structured segments with timestamps

### AC-02: French Language Support
- [ ] Given a French audio recording, when transcribed, then the output is accurate French text
- [ ] French accents and special characters are correctly rendered (é, è, ê, ç, etc.)
- [ ] Common French expressions and vocabulary are recognized

### AC-03: English Language Support
- [ ] Given an English audio recording, when transcribed, then the output is accurate English text
- [ ] Works with various English accents (US, UK, etc.)

### AC-04: Automatic Language Detection
- [ ] Given an audio file without language specification, when transcribed, then the correct language is detected
- [ ] Language is identified per-segment (not just per-file)
- [ ] Mixed French/English recordings are handled gracefully (code-switching)
- [ ] Each segment includes its detected language code (fr/en)

### AC-05: Speaker Diarization
- [ ] Given a multi-speaker recording, when transcribed, then speakers are identified
- [ ] Each segment is attributed to a speaker (Speaker_1, Speaker_2, etc.)
- [ ] Speaker identification is consistent throughout the transcript
- [ ] Works with 2+ speakers

### AC-06: Timestamps
- [ ] Transcription includes segment-level timestamps (start/end times)
- [ ] Word-level timestamps are available for precise alignment
- [ ] Timestamps are in seconds with millisecond precision
- [ ] Timestamps enable jumping to specific parts of the recording

### AC-07: Zero-Configuration Hardware Detection
- [ ] Given a PC with NVIDIA GPU, when transcribing, then GPU is automatically used
- [ ] Given a PC without GPU, when transcribing, then CPU-optimized model is used
- [ ] Given limited RAM (8GB), when transcribing, then edge/nano model is selected
- [ ] No manual configuration required for hardware adaptation

### AC-08: CLI Interface
- [ ] `conot transcribe <audio_file>` produces full diarized transcript
- [ ] `conot transcribe --output <file>` saves to specified file
- [ ] `conot transcribe --format json|txt|srt` supports multiple output formats
- [ ] Progress indication during transcription

---

## Technical Notes

### Architecture: Generic STT Provider Interface

The system uses an abstract provider interface (Protocol) to enable easy switching between STT backends based on:
- Host hardware (GPU vs CPU-only)
- Model evolution and new capabilities
- User preferences

```
STTProvider (Protocol)
    ├── transcribe(audio_path) -> TranscriptionResult
    ├── detect_language(audio_path) -> str
    ├── get_capabilities() -> ProviderCapabilities
    └── is_available() -> bool
```

### Provider Categories

| Category | Hardware Target | Memory | Latency |
|----------|-----------------|--------|---------|
| Edge/Nano | CPU, mobile | < 4GB | Higher |
| Standard | Consumer GPU | 4-12GB | Medium |
| Enterprise | Server GPU | 12GB+ | Low |

All providers must support:
- French and English transcription
- Automatic language detection
- Segment-level timestamps
- Word-level timestamps (if capability available)

### Configuration Example

```yaml
stt:
  provider: <provider-id>   # From provider registry
  model: <model-id>         # Provider-specific model
  device: auto              # or: cuda, cpu
  language: auto            # or: fr, en
```

### Provider Selection Logic

1. If `device: auto`, detect available hardware
2. Query provider registry for compatible providers
3. Select provider based on hardware tier and user preference
4. Fall back to CPU-compatible provider if GPU unavailable

### Speaker Diarization

All providers must support diarization to identify:
- Speaker identification ("Speaker_1", "Speaker_2", etc.)
- Per-segment speaker attribution
- Consistent speaker labels throughout transcript

From research: **Pyannote 3.1** is the recommended diarization engine (MIT license, 11-19% DER).

---

## Implementation Plan

### Phase 1: Provider Interface & Data Models
1. Define `STTProvider` protocol in `stt/protocol.py`
2. Define `TranscriptionResult`, `Segment`, `Word` data classes
3. Define `ProviderCapabilities` for feature detection
4. Create provider registry for dynamic loading

### Phase 2: Hardware Detection
1. Implement GPU detection (CUDA availability, VRAM)
2. Implement RAM detection
3. Auto-select provider tier based on hardware

### Phase 3: First Provider (GPU tier)
1. Implement enterprise/standard provider for GPU hosts
2. Transcription with language detection per-segment
3. Speaker diarization integration
4. Word-level timestamps

### Phase 4: Edge Provider (CPU tier)
1. Implement edge/nano provider for CPU-only hosts
2. Same interface, same output format
3. Optimized for low memory

### Phase 5: Output Formats
1. JSON output (structured, with speakers + languages)
2. Plain text output (speaker-attributed)
3. SRT subtitle format

### Phase 6: CLI Integration
1. Add `conot transcribe` command
2. Progress reporting
3. Batch transcription support

---

## Dependencies

### Core (always required)
| Category | Purpose |
|----------|---------|
| Data validation | Pydantic models for results |
| Audio I/O | Read audio files for processing |

### Provider Dependencies
Each provider has its own dependencies, installed separately:
- Providers must declare their dependencies
- Installation is based on user's selected provider
- CPU-only providers must not require GPU libraries

### License Requirements
- All provider dependencies must use permissive licenses (MIT, Apache 2.0, BSD)
- CC-BY-4.0 acceptable for model weights

---

## Related Requirements

- SYS-STT-001 through SYS-STT-010
- STK-STT-001 through STK-STT-007

---

## Open Questions

1. ~~Which STT library to use?~~ → **Generic provider interface** with pluggable backends
2. ~~GPU required?~~ → **No**, CPU-only hosts supported via edge/nano providers
3. ~~Model names in config?~~ → **No**, provider-specific identifiers only
4. Default model size? → Provider decides based on hardware detection
5. How to handle very long recordings? → Chunked processing with overlap
6. Real-time streaming transcription? → Deferred to future US
7. Auto-detect best provider based on hardware? → Yes, `device: auto` in config
