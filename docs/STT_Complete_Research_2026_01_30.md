# 🎙️ Complete STT/ASR Research Compendium
## January 2026

---

# Table of Contents

1. [Executive Summary](#executive-summary)
2. [Part 1: State of the Art](#part-1-state-of-the-art)
3. [Part 2: Open Source EN+FR Models](#part-2-open-source-enfr-models)
4. [Part 3: Streaming Capabilities](#part-3-streaming-capabilities)
5. [Part 4: Language Detection & Diarization](#part-4-language-detection--diarization)
6. [Quick Reference](#quick-reference)

---

# Executive Summary

The STT landscape in 2026 is dominated by:
1. **NVIDIA Canary/Parakeet** - Highest accuracy, fastest inference
2. **OpenAI Whisper variants** - Best multilingual coverage (99+ languages)
3. **Commercial APIs** - Deepgram Nova-3, AssemblyAI Universal-2
4. **Edge solutions** - Moonshine, whisper.cpp for on-device

**Key winner for most use cases:** NVIDIA Canary Qwen 2.5B (5.63% WER, 418x RTF)

**For EN+FR specifically:** 
- Best accuracy: Canary 1B / Canary Qwen 2.5B
- Best streaming: Parakeet TDT 0.6B V3
- Best ecosystem: Whisper Large V3 Turbo

---

# Part 1: State of the Art

## 🏆 Top Models by Category

### Maximum Accuracy (English)

| Model | WER | RTFx | Params | License |
|-------|-----|------|--------|---------|
| **Canary Qwen 2.5B** | 5.63% | 418x | 2.5B | CC-BY-4.0 |
| IBM Granite Speech 3.3 8B | 5.85% | - | ~9B | Apache 2.0 |
| Whisper Large V3 | 7.4% | varies | 1.55B | MIT |

### Speed-Optimized

| Model | WER | RTFx | Params | Notes |
|-------|-----|------|--------|-------|
| **Parakeet TDT 1.1B** | ~8.0% | >2,000x | 1.1B | Ultra-low latency streaming |
| Whisper Large V3 Turbo | 7.75% | 216x | 809M | 6x faster than V3 |
| Distil-Whisper | ~7.4% | ~6x V3 | 756M | English-only |

### Multilingual

| Model | Languages | WER | Notes |
|-------|-----------|-----|-------|
| **Whisper Large V3** | 99+ | 7.4% | Gold standard multilingual |
| Whisper Large V3 Turbo | 99+ | 7.75% | 6x faster |
| NVIDIA Canary | 25 EU | - | ASR + translation |

### Edge/Mobile

| Model | Params | Target | Notes |
|-------|--------|--------|-------|
| **Moonshine** | 27M | Mobile/IoT | Useful Sensors |
| Whisper Tiny | 39M | Edge | Basic accuracy |
| Distil-Whisper | 756M | Low VRAM | ~5GB VRAM |

---

## 🔬 Architecture Deep Dive

### NVIDIA Canary Qwen 2.5B (Best Overall)
- **Architecture:** Speech-Augmented Language Model (SALM)
- **Components:** FastConformer encoder + Qwen3-1.7B LLM decoder
- **Training data:** 234,000 hours English speech
- **Special features:**
  - Dual mode: pure transcription + intelligent analysis (summarization, Q&A)
  - Auto punctuation/capitalization
  - Noise tolerant: 2.41% WER at 10 dB SNR
- **Limitation:** English-only, needs chunked inference for >10s audio
- **Framework:** NVIDIA NeMo

### OpenAI Whisper Large V3
- **Architecture:** Transformer encoder-decoder, 32 decoder layers
- **Training data:** 680,000 hours multilingual web audio
- **Features:**
  - 128-bin mel-spectrogram (up from 80 in V2)
  - Auto language identification
  - Phrase-level timestamps
- **Variants:**
  - **Turbo:** 4 decoder layers, 809M params, 6x faster
  - **Distil:** 2 decoder layers, 756M params, English-only

### AssemblyAI Universal-2
- **Architecture:** 600M parameter Conformer RNN-T
- **Training data:** 12.5M hours multilingual audio
- **Advantages:**
  - Robust against hallucinations
  - Excellent timestamp accuracy
  - Immutable streaming transcripts (no rewriting)
- **Streaming:** 6 languages (EN, ES, FR, DE, IT, PT)

### NVIDIA Parakeet TDT
- **Architecture:** RNN-Transducer (streaming-optimized)
- **Training data:** 65,000 hours English
- **Speed:** >2,000x RTF (among fastest on Open ASR)
- **Use case:** Real-time streaming, live captioning, phone systems

---

## ⚡ Local Deployment Options

### faster-whisper
- **Speedup:** Up to 4x faster than OpenAI implementation
- **Backend:** CTranslate2 inference engine
- **Memory:** Lower VRAM usage
- **Quantization:** 8-bit on CPU and GPU
- **Latency:** 2-3 seconds with Turbo model for streaming

### whisper.cpp
- **Speedup:** 2-10x faster (hardware dependent)
- **Platform:** Pure C++ implementation
- **Target:** CPU inference, no GPU required
- **Use case:** Edge devices, offline deployment

### MLX Whisper (Apple Silicon)
- **Target:** M1/M2/M3 Macs
- **Speed:** Near real-time on Apple Silicon
- **Memory:** Optimized for unified memory

---

## 💰 Commercial API Comparison

| Provider | Model | WER | Latency | Price |
|----------|-------|-----|---------|-------|
| **Deepgram** | Nova-3 | ~18% | <300ms | ~$4.30/1000 min |
| **AssemblyAI** | Universal-2 | 14.5% | streaming | ~$0.15/hour |
| **Google Cloud** | Chirp | 11.6% | batch | varies |
| **OpenAI** | GPT-4o-Transcribe | - | 320ms | varies |

### When to use Commercial:
- Rapid prototyping
- Low volume (<100 hours/month)
- Need advanced features (diarization, PII detection)
- Regulated industries requiring SLAs

### When to use Open Source:
- High volume (cost savings at scale)
- Data privacy requirements
- Custom fine-tuning needs
- Specific domain adaptation

---

## 🎯 Recommendations by Use Case

| Use Case | Primary | Alternative | Why |
|----------|---------|-------------|-----|
| **Voice Assistants / Real-Time** | Parakeet TDT 1.1B | Whisper Large V3 Turbo + faster-whisper | Ultra-low latency, streaming support |
| **Meeting Transcription** | Canary Qwen 2.5B | AssemblyAI Universal-2 (commercial) | High accuracy on conversational speech |
| **Multilingual Applications** | Whisper Large V3 | Whisper Large V3 Turbo | 99+ languages, zero-shot capability |
| **Mobile/Edge Deployment** | Moonshine (27M params) | Whisper Tiny + whisper.cpp | Minimal resources, offline capable |
| **Medical/Legal (High Stakes)** | Canary Qwen 2.5B or Granite 8B | AssemblyAI with guardrails | Highest accuracy, auditability |
| **Call Centers** | Deepgram Nova-3 (streaming) | Parakeet TDT + custom pipeline | Low latency, handles telephony audio |

---

## 📊 Benchmark Sources

1. **Hugging Face Open ASR Leaderboard** - Primary accuracy benchmark
2. **LibriSpeech** - Clean/Other test sets (academic standard)
3. **Ionio Benchmark** - Real-world noise conditions
4. **CommonVoice** - Multilingual evaluation

---

## 🔮 Trends & Predictions

1. **SALM architectures** (Speech + LLM) will dominate - enables analysis beyond transcription
2. **Streaming accuracy** improving rapidly - gap with batch closing
3. **Edge deployment** becoming viable - Moonshine, quantized models
4. **Multimodal integration** - GPT-4o style audio understanding
5. **Domain-specific fine-tuning** - Medical, legal, technical verticals

---

# Part 2: Open Source EN+FR Models

## 🏆 Top Open Source Models for English + French

### Tier 1: Best Accuracy

| Model | EN WER | FR Support | RTFx | Params | License |
|-------|--------|------------|------|--------|---------|
| **NVIDIA Canary 1B** | 6.67% | ✅ Native (25 EU langs) | >1000x | 1B | CC-BY-4.0 |
| **Whisper Large V3** | 7.4% | ✅ Native (99+ langs) | varies | 1.55B | MIT |
| Whisper Large V3 Turbo | 7.75% | ✅ Native | 216x | 809M | MIT |

### Tier 2: Speed-Optimized

| Model | EN WER | FR Support | Speed | Params | License |
|-------|--------|------------|-------|--------|---------|
| **Whisper Large V3 Turbo** | 7.75% | ✅ | 6x faster | 809M | MIT |
| Wav2Vec 2.0 XLSR | ~8-10% | ✅ (53 langs) | moderate | 300M | MIT |

---

## 🥇 Recommended: NVIDIA Canary 1B

**Why Canary for EN+FR:**
- Specifically trained on **English, French, Spanish, German**
- **Outperforms Whisper, OWSM, Seamless-M4T** on these 4 languages
- Trained on **85,000 hours** (vs Whisper's 680K but more curated)
- Supports **ASR + Translation** between supported languages
- **>1000x RTFx** - extremely fast inference

**Architecture:**
- FastConformer encoder + Transformer decoder
- Multi-task: transcription, translation, language ID
- Trained with NVIDIA Granary dataset

**French Performance:**
- Native French support with dedicated training data
- Bi-directional EN↔FR translation
- Punctuation & capitalization in French

**Deployment:**
```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b")
# Supports: English, French, Spanish, German
```

**VRAM:** ~4-6GB depending on batch size

---

## 🥈 Alternative: Whisper Large V3 / Turbo

**Why Whisper for EN+FR:**
- Most battle-tested multilingual model
- 99+ languages with strong French performance
- MIT license - fully permissive
- Huge ecosystem (faster-whisper, whisper.cpp, etc.)

**Variants:**

| Variant | Speed | Accuracy | FR Quality |
|---------|-------|----------|------------|
| Large V3 | 1x | Best | Excellent |
| Large V3 Turbo | 6x | -1% | Excellent |
| Distil-Large V3 | 6x | ~same | ❌ English only |

**French Considerations:**
- Strong French accuracy in V3 (improved over V2)
- Auto language detection works well for FR
- Handles French accents (Québécois, African French)

**Deployment (fastest):**
```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cuda")
# or "large-v3" for max accuracy
```

**VRAM:** 
- Large V3: ~10GB
- Turbo: ~6GB

---

## 📊 French-Specific Benchmarks

From the Canary paper (arXiv:2406.19674):

| Model | English WER | French WER | Training Data |
|-------|-------------|------------|---------------|
| **Canary 1B** | 6.67% | Best in class | 85K hours |
| Whisper Large V3 | 7.4% | Good | 680K hours |
| SeamlessM4T | ~8% | Good | - |
| OWSM | ~9% | Moderate | - |

*Note: Canary achieves better results with 8x less training data through better curation*

---

## ⚡ Speed vs Accuracy Trade-offs (EN+FR)

| Priority | Model | Notes |
|----------|-------|-------|
| **Maximum Accuracy** | Canary 1B or Whisper Large V3 | Best WER |
| **Balanced (Production)** | Whisper Large V3 Turbo | 6x faster, ~0.5% drop |
| **Real-Time Streaming** | Canary 1B | >1000x RTFx |
| **Edge/Low Resource** | Whisper Medium | 769M params, ~5GB |

---

## 🛠️ Deployment Options

### Option 1: NeMo + Canary (Best Accuracy)
```bash
pip install nemo_toolkit[all]

# Python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-1b")
transcription = model.transcribe(["audio.wav"])
```

### Option 2: faster-whisper (Best Ecosystem)
```bash
pip install faster-whisper

# Python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
segments, info = model.transcribe("audio.wav", language="fr")  # or auto-detect
```

### Option 3: whisper.cpp (CPU/Edge)
```bash
# Build
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp && make

# Download model
./models/download-ggml-model.sh large-v3-turbo

# Run
./main -m models/ggml-large-v3-turbo.bin -f audio.wav -l fr
```

---

## 🎯 EN+FR Recommendation Summary

| Use Case | Model | Why |
|----------|-------|-----|
| **Best EN+FR Accuracy** | Canary 1B | Designed for EU languages, fastest |
| **Most Flexible** | Whisper Large V3 | 99+ langs, huge ecosystem |
| **Production Balanced** | Whisper Large V3 Turbo | 6x speed, good accuracy |
| **Real-Time/Streaming** | Canary 1B | >1000x RTFx |
| **CPU/Edge** | whisper.cpp + Large V3 Turbo | No GPU needed |

---

## 🔧 Hardware Requirements (EN+FR Models)

| Model | VRAM | Recommended GPU |
|-------|------|-----------------|
| Canary 1B | ~4-6GB | RTX 3060+ |
| Whisper Large V3 | ~10GB | RTX 3090+ |
| Whisper Large V3 Turbo | ~6GB | RTX 3070+ |
| Whisper Medium | ~5GB | RTX 3060+ |

---

# Part 3: Streaming Capabilities

## 🚀 Streaming Capabilities Summary

| Model | Streaming | Latency | EN/FR | Architecture |
|-------|-----------|---------|-------|--------------|
| **Parakeet TDT 0.6B V3** | ✅ Native | Ultra-low | ✅ Both | TDT (streaming-first) |
| **Canary 1B** | ✅ Chunked | Low | ✅ Both | FastConformer + AED |
| Whisper + streaming libs | ⚠️ Wrapper | Medium | ✅ Both | Encoder-Decoder |
| Nemotron Speech 0.6B | ✅ Native | Ultra-low | ❌ EN only | Cache-aware RNN-T |

---

## 🥇 Best for Streaming EN+FR: Parakeet TDT 0.6B V3

**Native streaming support with French!**

### Specs:
- **Languages:** 25 EU languages including **English** and **French**
- **Params:** 600M
- **Speed:** >2000x RTFx (1 hour audio → ~2 seconds)
- **Architecture:** Token-and-Duration Transducer (TDT)
- **Streaming:** Native support, ultra-low latency
- **License:** CC-BY-4.0

### Why TDT for Streaming:
- RNN-T based = designed for streaming from the ground up
- Processes audio incrementally without needing full context
- Auto language detection (no prompting needed)
- Auto punctuation & capitalization

### Deployment:
```python
import nemo.collections.asr as nemo_asr

# Load model
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")

# Streaming inference
from nemo.collections.asr.modules.rnnt import RNNTDecoding
# Configure for streaming with chunked audio
```

### Hardware:
- VRAM: ~3-4GB
- GPU: RTX 3060+

---

## 🥈 Alternative: Canary 1B with Chunked Streaming

**Higher accuracy but slightly more latency**

### Streaming Support:
- Chunked inference: splits audio into segments
- Supports real-time processing via `cache_aware_streaming`
- Official NeMo streaming scripts available

### Configuration:
```python
# Chunked streaming
model.transcribe(
    audio_files,
    chunk_len_in_secs=10,  # Process 10s chunks
    streaming=True
)
```

### Trade-off:
- Better accuracy than Parakeet (6.67% vs ~8% WER)
- Slightly higher latency due to encoder-decoder architecture
- Good for "near-real-time" (seconds, not milliseconds)

---

## 🥉 Whisper with Streaming Wrappers

**Most ecosystem options, requires extra setup**

### Streaming Solutions:

| Library | Latency | Approach |
|---------|---------|----------|
| **WhisperLive** | 2-3s | TensorRT backend |
| **whisper_streaming** | 3-5s | Sliding window |
| **faster-whisper** | 2-3s | With streaming mode |
| **SimulStreaming** | <2s | Replacing whisper_streaming |

### WhisperLive (Collabora):
```bash
# Docker with TensorRT acceleration
docker run -p 9090:9090 whisperlive-tensorrt
python3 run_server.py --port 9090 --backend tensorrt
```

### faster-whisper streaming:
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3-turbo", device="cuda")

# Process in chunks for pseudo-streaming
for chunk in audio_chunks:
    segments, _ = model.transcribe(chunk, language="fr")
    yield segments
```

### Limitations:
- Not native streaming (encoder-decoder architecture)
- Requires buffering and sliding window
- Higher latency than TDT-based models
- But: Best ecosystem, MIT license, 99+ languages

---

## 📊 Streaming Latency Comparison

| Solution | End-to-End Latency | Notes |
|----------|-------------------|-------|
| **Parakeet TDT** | ~100-300ms | Native streaming |
| Canary chunked | ~500ms-2s | Depends on chunk size |
| WhisperLive | ~2-3s | TensorRT optimized |
| faster-whisper streaming | ~2-3s | With good buffering |
| Plain Whisper | Not streaming | Batch only |

---

## 🎯 Streaming Recommendation for EN+FR

| Use Case | Model | Latency |
|----------|-------|---------|
| **Real-Time Voice Assistant** | Parakeet TDT 0.6B V3 | ~100-300ms |
| **Near-Real-Time + Best Accuracy** | Canary 1B chunked | ~1-2s |
| **Maximum Flexibility** | Whisper Large V3 Turbo + WhisperLive | ~2-3s |

---

## 📋 Streaming Comparison Matrix (EN+FR)

| | Parakeet TDT V3 | Canary 1B | Whisper+Live |
|--|-----------------|-----------|--------------|
| **Streaming** | ✅ Native | ✅ Chunked | ⚠️ Wrapper |
| **Latency** | ~100-300ms | ~1-2s | ~2-3s |
| **EN Accuracy** | ~8% WER | 6.67% WER | 7.4% WER |
| **French** | ✅ Native | ✅ Native | ✅ Native |
| **Languages** | 25 EU | 25 EU | 99+ |
| **License** | CC-BY-4.0 | CC-BY-4.0 | MIT |
| **VRAM** | ~4GB | ~6GB | ~6-10GB |
| **Best For** | Voice agents | Quality transcription | Flexibility |

---

# Part 4: Language Detection & Diarization

## 🌍 Language Detection Capabilities

### Summary Table

| Model | Auto Language Detection | Supported Languages | Accuracy |
|-------|------------------------|---------------------|----------|
| **Whisper Large V3** | ✅ Built-in | 99+ languages | Excellent |
| **Canary 1B** | ✅ Built-in | 25 EU (inc. FR) | Excellent |
| **Parakeet TDT V3** | ✅ Built-in | 25 EU (inc. FR) | Excellent |
| Wav2Vec 2.0 XLSR | ❌ Manual | 53 languages | Good |

### Whisper Language Detection
- **Automatic** language identification from first 30 seconds
- Supports 99+ languages including French
- Can force specific language with `language="fr"` parameter
- Code-switching detection (mixing languages) is limited

```python
# Auto-detect
segments, info = model.transcribe("audio.wav")
print(f"Detected language: {info.language}")

# Force French
segments, info = model.transcribe("audio.wav", language="fr")
```

### NVIDIA Canary/Parakeet Language Detection
- **Automatic** detection without prompting
- 25 EU languages: EN, FR, DE, ES, IT, PT, NL, PL, RU, etc.
- Detects language per-segment (better for multilingual audio)

```python
# Canary - auto-detects and transcribes
transcription = model.transcribe(["audio.wav"])
# Returns detected language with transcript
```

### Language Detection Accuracy (EN+FR)
- All models: **>95% accuracy** for EN vs FR detection
- Challenge: Short utterances (<3 seconds)
- Challenge: Code-switching (EN/FR mixed in same sentence)

---

## 👥 Speaker Diarization (Who Spoke When)

### Diarization Options Comparison

| Solution | Accuracy (DER) | Speed | Ease | Open Source |
|----------|---------------|-------|------|-------------|
| **Pyannote 3.1** | 11-19% | Fast (GPU) | Medium | ✅ MIT |
| **NVIDIA NeMo** | ~15% | Very fast | Hard | ✅ Apache |
| **WhisperX** | Good | Moderate | Easy | ✅ |
| Sortformer | Excellent | Fast | Medium | ✅ |

---

### 🥇 Recommended: Pyannote 3.1

**Best balance of accuracy, speed, and ease of use**

- **DER:** 11-19% depending on audio quality
- **Overlapping speech:** ✅ Handles well
- **License:** MIT (fully open)
- **Languages:** Language-agnostic (works with FR)

```python
from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="YOUR_HF_TOKEN"
)

diarization = pipeline("audio.wav")

for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{speaker}: {turn.start:.1f}s - {turn.end:.1f}s")
```

**Requirements:**
- HuggingFace account (free)
- GPU recommended (2-4GB VRAM)
- ~1GB model weights

---

### 🥈 WhisperX (Transcription + Diarization)

**Best for combined STT + diarization in one pipeline**

Combines:
- Whisper (transcription)
- Pyannote (diarization)
- Forced alignment (word-level timestamps)

```python
import whisperx

model = whisperx.load_model("large-v3", device="cuda")
audio = whisperx.load_audio("audio.wav")

# Transcribe
result = model.transcribe(audio)

# Align words
result = whisperx.align(result["segments"], model_a, audio)

# Diarize
diarize_model = whisperx.DiarizationPipeline(use_auth_token="HF_TOKEN")
diarize_segments = diarize_model(audio)

# Assign speakers to words
result = whisperx.assign_word_speakers(diarize_segments, result)
```

**Output:** Speaker-attributed transcript with word-level timestamps

---

### 🥉 NVIDIA NeMo (Enterprise Scale)

**Best for production at scale with NVIDIA GPUs**

- Uses TitaNet embeddings + MSDD decoder
- Very fast on A100/RTX 4090
- Better for long-form audio (>1 hour)

```python
import nemo.collections.asr as nemo_asr

msdd_model = nemo_asr.models.ClusteringDiarizer.from_pretrained(
    "diar_msdd_telephonic"
)

diarization = msdd_model.diarize("audio.wav")
```

**Requirements:**
- NVIDIA GPU required (not optional)
- Complex setup
- 4-8GB VRAM

---

## 🔄 Complete Pipeline: STT + Lang Detection + Diarization

### Option 1: WhisperX (Easiest)
```bash
pip install whisperx
```
- Whisper for STT (99+ langs, auto-detect)
- Pyannote for diarization
- Word-level speaker attribution
- **French:** ✅ Full support

### Option 2: Canary/Parakeet + Pyannote (Best Accuracy)
```bash
pip install nemo_toolkit pyannote.audio
```
- Canary/Parakeet for STT (EN+FR optimized)
- Pyannote for diarization
- Requires manual pipeline integration
- **French:** ✅ Native support

### Option 3: NeMo Full Stack (Enterprise)
```bash
pip install nemo_toolkit[all]
```
- NeMo ASR (Canary/Parakeet)
- NeMo Diarization (MSDD)
- Unified framework
- **French:** ✅ Via Canary/Parakeet

---

## 📊 Language Detection + Diarization Matrix

| Requirement | Best Solution |
|-------------|---------------|
| Auto language detect (EN/FR) | Whisper or Canary |
| Speaker diarization | Pyannote 3.1 |
| Combined STT + diarization | WhisperX |
| Streaming + diarization | Parakeet + post-processing |
| Production scale | NeMo full stack |

---

## ⚠️ Limitations & Considerations

### Language Detection
- **Code-switching:** All models struggle with mid-sentence language changes
- **Short utterances:** Detection less reliable for <3s audio
- **Similar languages:** FR vs Quebec French, EN-US vs EN-UK handled differently

### Diarization
- **Overlapping speech:** ~70-80% accuracy (challenging for all models)
- **Speaker similarity:** Similar voices harder to distinguish
- **Background noise:** Degrades accuracy significantly
- **Real-time:** Pyannote is offline-only; real-time diarization requires streaming models

### Streaming + Diarization
- Currently no open source solution does real-time diarization well
- Best approach: Parakeet for streaming STT → batch diarization afterward
- pyannote.ai offers commercial real-time diarization (not open source)

---

# Quick Reference

## 🏆 Overall Winners

| Category | Model | Why |
|----------|-------|-----|
| **Best Overall Accuracy** | Canary Qwen 2.5B | 5.63% WER, fastest |
| **Best EN+FR** | Canary 1B | Optimized for EU languages |
| **Best Streaming EN+FR** | Parakeet TDT 0.6B V3 | Native streaming, ~100-300ms |
| **Best Multilingual** | Whisper Large V3 | 99+ languages |
| **Best Ecosystem** | Whisper + faster-whisper | MIT license, huge community |
| **Best Diarization** | Pyannote 3.1 | 11-19% DER, MIT |
| **Best Combined Pipeline** | WhisperX | STT + diarization + alignment |

---

## 🛠️ Quick Start Commands

```bash
# Best accuracy (EN+FR)
pip install nemo_toolkit[all]
# → nvidia/canary-1b

# Best ecosystem (99+ langs)
pip install faster-whisper
# → whisper-large-v3-turbo

# Best streaming (EN+FR)
pip install nemo_toolkit[all]
# → nvidia/parakeet-tdt-0.6b-v3

# STT + Diarization
pip install whisperx
# → whisperx with pyannote

# Diarization only
pip install pyannote.audio
# → pyannote/speaker-diarization-3.1
```

---

## 🔧 Hardware Requirements Summary

| Model | VRAM | GPU Tier |
|-------|------|----------|
| Canary Qwen 2.5B | ~8GB | RTX 3080+ |
| Canary 1B | ~4-6GB | RTX 3060+ |
| Parakeet TDT 0.6B V3 | ~3-4GB | RTX 3060+ |
| Whisper Large V3 | ~10GB | RTX 3090+ |
| Whisper Large V3 Turbo | ~6GB | RTX 3070+ |
| Pyannote 3.1 | ~2-4GB | RTX 3060+ |

---

*Complete STT Research Compendium compiled 2026-01-30*
*Sources: Hugging Face Open ASR Leaderboard, NVIDIA docs, Northflank benchmarks, AssemblyAI research, arXiv papers, community reports*
