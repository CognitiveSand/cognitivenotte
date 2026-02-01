# Project Requirements: cognitive notte (conot)

| Field | Value |
|-------|-------|
| **Project** | conot |
| **Version** | 0.1 |
| **Date** | 2026-02-01 |
| **Status** | Implemented |
| **Reference** | |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Stakeholder Requirements (STK)](#2-stakeholder-requirements-stk)
3. [System Requirements (SYS)](#3-system-requirements-sys)
4. [Configuration Parameters](#4-configuration-parameters)
5. [Data Structures](#5-data-structures)
6. [Traceability Matrix](#6-traceability-matrix)
7. [Constraints](#7-constraints)
8. [Dependencies](#8-dependencies)

---

## 1. Introduction

### 1.1 Purpose

This document defines the requirements for implementing a notte taking library

### 1.2 Scope

The conot system encompasses:
- sound acsuisition
- timestamping
- sound transcription
- speaked diarization (post mvp)


### 1.3 References

| Document | Description |
|----------|-------------|
| US-01 | User Story: sound acquisition |
| US-02 | User Story: speech transcription |


### 1.4 Terminology

| Term | Definition |
|------|------------|


---

## 2. Stakeholder Requirements (STK)

High-level requirements expressing stakeholder needs.

### 2.0 General

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| STK-GEN-001 | System shall only use open-source libraries, tools, and models | Must | - |
| STK-GEN-002 | All dependencies shall be freely available without commercial licensing | Must | STK-GEN-001 |
| STK-GEN-003 | Models shall be usable without API keys or paid services | Must | STK-GEN-001 |

### 2.1 Sound Acquisition

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| STK-ACQ-001 | User shall be able to record audio on a Linux computer | Must | US-01 |
| STK-ACQ-002 | Recording shall work without any configuration | Must | US-01 |
| STK-ACQ-003 | Recording shall start with 1 click and stop with 1 click | Must | US-01 |
| STK-ACQ-004 | Recording shall work across different devices (desktop, laptop) without adjustment | Must | US-01 |
| STK-ACQ-005 | Recording shall work in noisy environments without manual adjustment | Must | US-01 |
| STK-ACQ-006 | System shall always record from the right microphone for optimal speaker sound quality | Must | US-01 |

### 2.2 Speech Transcription

| ID | Requirement | Priority | Source |
|----|-------------|----------|--------|
| STK-STT-001 | User shall be able to transcribe recorded audio to text | Must | US-02 |
| STK-STT-002 | Transcription shall support French language (primary) | Must | US-02 |
| STK-STT-003 | Transcription shall support English language | Must | US-02 |
| STK-STT-004 | System shall automatically detect the spoken language | Must | US-02 |
| STK-STT-005 | Transcription shall include word-level timestamps | Should | US-02 |
| STK-STT-006 | Transcription shall work without internet connection | Should | US-02 |
| STK-STT-007 | System shall identify different speakers in a recording (diarization) | Must | US-02 |
| STK-STT-008 | System shall display the model currently in use in the terminal | Must | - |
| STK-STT-009 | Transcript files shall be stored in ./transcripts directory | Must | - |
| STK-STT-010 | Transcripts shall be saved incrementally after each segment is added | Must | - |
| STK-STT-011 | System shall support streaming JSON output to stdout for LLM integration | Must | - |
| STK-STT-012 | Default mode shall capture audio from microphone and perform real-time streaming transcription | Must | - |
| STK-STT-013 | Real-time transcription shall display each line with timestamp, language, and speaker prefix | Must | STK-STT-012 |
| STK-STT-014 | Transcription shall appear fluidly in real-time (<2s latency from speech start) | Must | STK-STT-012 |

---

## 3. System Requirements (SYS)

Technical requirements derived from stakeholder needs.

### 3.1 Sound Acquisition

| ID | Requirement | Priority | Derives From |
|----|-------------|----------|--------------|
| SYS-ACQ-001 | System shall automatically detect available audio input devices | Must | STK-ACQ-002 |
| SYS-ACQ-002 | System shall automatically select the best available input device for speech capture | Must | STK-ACQ-002, STK-ACQ-004, STK-ACQ-006 |
| SYS-ACQ-003 | System shall support USB microphones | Must | STK-ACQ-004 |
| SYS-ACQ-004 | System shall support built-in laptop microphones | Must | STK-ACQ-004 |
| SYS-ACQ-005 | System shall capture audio and save to file | Must | STK-ACQ-001 |
| SYS-ACQ-006 | System shall timestamp recordings in the filename | Must | STK-ACQ-001 |
| SYS-ACQ-007 | System shall provide a single action to start recording | Must | STK-ACQ-003 |
| SYS-ACQ-008 | System shall provide a single action to stop recording | Must | STK-ACQ-003 |
| SYS-CFG-001 | System shall read configuration from ./settings.yml | Must | - |

### 3.2 Speech Transcription

| ID | Requirement | Priority | Derives From |
|----|-------------|----------|--------------|
| SYS-STT-001 | System shall transcribe audio files to text using a local STT engine | Must | STK-STT-001, STK-STT-006 |
| SYS-STT-002 | System shall support French language transcription | Must | STK-STT-002 |
| SYS-STT-003 | System shall support English language transcription | Must | STK-STT-003 |
| SYS-STT-004 | System shall automatically detect language per-segment | Must | STK-STT-004 |
| SYS-STT-005 | System shall provide word-level timestamps in transcription output | Should | STK-STT-005 |
| SYS-STT-006 | System shall output transcription in structured format (JSON) | Must | STK-STT-001 |
| SYS-STT-007 | System shall work on CPU-only hosts without GPU | Must | STK-STT-006 |
| SYS-STT-008 | System shall utilize GPU acceleration when available | Should | - |
| SYS-STT-009 | System shall perform speaker diarization on multi-speaker recordings | Must | STK-STT-007 |
| SYS-STT-010 | System shall provide confidence scores for transcriptions | Should | - |
| SYS-STT-011 | System shall provide a generic STT provider interface (protocol) | Must | - |
| SYS-STT-012 | System shall allow switching STT backend via configuration | Must | SYS-STT-011 |
| SYS-STT-013 | System shall support edge/nano models (< 1B params, < 4GB memory) | Must | SYS-STT-007 |
| SYS-STT-014 | System shall support larger models when hardware permits | Should | SYS-STT-008 |
| SYS-STT-015 | System shall auto-detect available GPU and VRAM | Must | SYS-STT-007, SYS-STT-008 |
| SYS-STT-016 | System shall auto-detect available system RAM | Must | SYS-STT-007 |
| SYS-STT-017 | System shall auto-select best provider based on detected hardware | Must | SYS-STT-011 |
| SYS-STT-018 | System shall work without any user configuration | Must | STK-STT-006 |
| SYS-STT-019 | System shall display the active model name in terminal/debug view | Must | STK-STT-008 |
| SYS-STT-020 | System shall store transcript files in ./transcripts directory | Must | STK-STT-009 |
| SYS-STT-021 | System shall create ./transcripts directory if it does not exist | Must | STK-STT-009 |
| SYS-STT-022 | System shall save transcript file after each new segment is transcribed | Must | STK-STT-010 |
| SYS-STT-023 | Transcript filename shall include timestamp for uniqueness | Must | STK-STT-009 |
| SYS-STT-024 | System shall support --stdout flag for streaming JSON to stdout | Must | STK-STT-011 |
| SYS-STT-025 | Each segment shall be output as a single JSON line (JSONL format) | Must | STK-STT-011 |
| SYS-STT-026 | Stdout streaming shall be pipe-friendly (no decorations/colors) | Must | STK-STT-011 |
| SYS-STT-027 | System shall auto-select the most accurate provider that hardware supports | Must | SYS-STT-017 |
| SYS-STT-028 | Default CLI command shall start live transcription from microphone | Must | STK-STT-012 |
| SYS-STT-029 | Each transcribed segment shall display format: [HH:MM:SS] (lang) [Speaker] text | Must | STK-STT-013 |
| SYS-STT-030 | Transcription shall stream to display within 2 seconds of speech start (fluid real-time) | Must | STK-STT-014 |
| SYS-STT-031 | Display shall show timestamp at start of each segment | Must | STK-STT-013 |
| SYS-STT-032 | Display shall show detected language code for each segment | Must | STK-STT-013 |
| SYS-STT-033 | Display shall show speaker identifier when diarization is available | Must | STK-STT-013 |

### 3.3 Development and Debugging

| ID | Requirement | Priority | Derives From |
|----|-------------|----------|--------------|
| SYS-DEV-001 | System shall use standardized logging adapted for debugging | Must | - |
| SYS-DEV-002 | System shall provide a debug terminal view with audio level meters | Must | - |

---

## 4. Configuration Parameters

**Configuration file:** `./settings.yml`

| Parameter | Description | Default | Source |
|-----------|-------------|---------|--------|
| output_dir | Directory where recordings are saved | ~/conot/recordings | SYS-ACQ-005 |
| audio_format | Format for saved recordings | WAV | SYS-ACQ-005 |
| sample_rate_hz | Audio sample rate | 44100 | SYS-ACQ-005 |
| stt.provider | STT backend provider identifier | (see provider registry) | SYS-STT-011, SYS-STT-012 |
| stt.model | Model identifier within provider | (provider default) | SYS-STT-001 |
| stt.device | Compute device (auto/cuda/cpu) | auto | SYS-STT-007, SYS-STT-008 |
| stt.language | Language hint (auto/fr/en) | auto | SYS-STT-004 |
| stt.diarization | Enable speaker diarization | false | SYS-STT-009 |
| transcripts_dir | Directory where transcripts are saved | ./transcripts | SYS-STT-020 |

---

## 5. Data Structures

### 5.1 Recording Metadata

| Field | Type | Description |
|-------|------|-------------|
| filename | string | Path to the recording file |
| timestamp | datetime | When the recording started |
| duration_s | float | Length of recording in seconds |
| device_name | string | Name of the input device used |

### 5.2 Transcription Output

| Field | Type | Description |
|-------|------|-------------|
| audio_file | string | Path to source audio file |
| duration_s | float | Total audio duration in seconds |
| languages_detected | array | List of detected language codes |
| speakers | array | List of speaker identifiers |
| segments | array | List of transcription segments |
| segments[].start | float | Segment start time in seconds |
| segments[].end | float | Segment end time in seconds |
| segments[].speaker | string | Speaker identifier (Speaker_1, Speaker_2, etc.) |
| segments[].language | string | Language code for this segment (fr, en) |
| segments[].text | string | Segment text |
| segments[].confidence | float | Transcription confidence (0-1) |
| segments[].words | array | Word-level details |
| segments[].words[].word | string | Individual word |
| segments[].words[].start | float | Word start time |
| segments[].words[].end | float | Word end time |
| segments[].words[].confidence | float | Word confidence |

---

## 6. Traceability Matrix

| STK ID | SYS ID | Status |
|--------|--------|--------|
| STK-ACQ-001 | SYS-ACQ-005, SYS-ACQ-006 | Implemented |
| STK-ACQ-002 | SYS-ACQ-001, SYS-ACQ-002 | Implemented |
| STK-ACQ-003 | SYS-ACQ-007, SYS-ACQ-008 | Implemented |
| STK-ACQ-004 | SYS-ACQ-002, SYS-ACQ-003, SYS-ACQ-004 | Implemented |
| STK-ACQ-005 | - | Implemented |
| STK-ACQ-006 | SYS-ACQ-002 | Implemented |
| STK-STT-001 | SYS-STT-001, SYS-STT-006 | Implemented |
| STK-STT-002 | SYS-STT-002 | Implemented |
| STK-STT-003 | SYS-STT-003 | Implemented |
| STK-STT-004 | SYS-STT-004 | Implemented |
| STK-STT-005 | SYS-STT-005 | Implemented |
| STK-STT-006 | SYS-STT-001, SYS-STT-007, SYS-STT-018 | Implemented |
| STK-STT-007 | SYS-STT-009 | Implemented |
| STK-STT-008 | SYS-STT-019 | Implemented |
| STK-STT-009 | SYS-STT-020, SYS-STT-021, SYS-STT-023 | Implemented |
| STK-STT-010 | SYS-STT-022 | Implemented |
| STK-STT-011 | SYS-STT-024, SYS-STT-025, SYS-STT-026 | Implemented |
| STK-STT-012 | SYS-STT-028, SYS-STT-030 | Planned |
| STK-STT-013 | SYS-STT-029, SYS-STT-031, SYS-STT-032, SYS-STT-033 | Planned |
| STK-STT-014 | SYS-STT-030 | Planned |
| STK-GEN-001 | CON-003, CON-004 | Implemented |
| STK-GEN-002 | CON-006 | Implemented |
| STK-GEN-003 | CON-005 | Implemented |

---

## 7. Constraints

| ID | Constraint | Rationale |
|----|------------|-----------|
| CON-001 | Linux operating system only | Target user environment |
| CON-002 | Python 3.12 or newer | Per conventions |
| CON-003 | Open-source libraries only (OSI-approved licenses) | STK-GEN-001: No vendor lock-in, transparency |
| CON-004 | Open-source models only (Apache 2.0, MIT, or equivalent) | STK-GEN-001: Free redistribution |
| CON-005 | No cloud API dependencies for core functionality | STK-GEN-003: Offline operation, privacy |
| CON-006 | No paid services or API keys required | STK-GEN-002: Free to use |

---

## 8. Dependencies

**All dependencies must be open-source with OSI-approved licenses (CON-003).**

### 8.1 Core Dependencies

| Dependency | Purpose | Version | License |
|------------|---------|---------|---------|
| sounddevice | Audio device interface | >=0.5.0 | MIT |
| numpy | Audio data processing | >=2.0.0 | BSD |
| scipy | WAV file I/O | >=1.14.0 | BSD |

### 8.2 STT Dependencies

| Dependency | Purpose | License |
|------------|---------|---------|
| faster-whisper | GPU-accelerated transcription | MIT |
| whisper.cpp | CPU-optimized transcription | MIT |
| qwen-asr | High-accuracy transcription (~30% better WER) | Apache 2.0 |
| pyannote-audio | Speaker diarization | MIT |
| Whisper models | Speech recognition | MIT (OpenAI) |
| Qwen3-ASR models | High-accuracy speech recognition | Apache 2.0 (Alibaba) |

### 8.3 STT Provider Categories

Providers are installed based on user configuration. The system supports:

| Category | Hardware Target | Memory Budget | Use Case |
|----------|-----------------|---------------|----------|
| Edge/Nano | CPU, low-end GPU | < 4GB | Laptops, edge devices |
| Standard | Consumer GPU | 4-12GB | Desktop workstations |
| Enterprise | High-end GPU | 12GB+ | Servers, batch processing |

Provider implementations are maintained separately from requirements to allow evolution.

### 8.4 Model Requirements

All models must be:
- Open-source with permissive license (MIT, Apache 2.0, BSD)
- Freely downloadable without API keys
- Runnable locally without cloud dependencies

| Model | Purpose | License | Source |
|-------|---------|---------|--------|
| Whisper (large-v3, medium, small, tiny) | Speech recognition | MIT | OpenAI/HuggingFace |
| Qwen3-ASR (1.7B, 0.6B) | High-accuracy speech recognition | Apache 2.0 | Alibaba/HuggingFace |
| Pyannote speaker-diarization-3.1 | Speaker identification | MIT | HuggingFace |

**Note**: HuggingFace token is required to download Pyannote models (free account), but models run locally after download.
