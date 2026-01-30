#!/bin/bash
# Conot Installation Script
# Installs conot with optional STT providers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Conot Installation Script ===${NC}"
echo

# Detect if uv is installed
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed.${NC}"
    echo "Please install uv first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Detect hardware
echo "Detecting hardware..."
HAS_NVIDIA=false
VRAM_GB=0

if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    if [ -n "$VRAM_MB" ] && [ "$VRAM_MB" != "0" ]; then
        HAS_NVIDIA=true
        VRAM_GB=$((VRAM_MB / 1024))
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo -e "  ${GREEN}✓${NC} NVIDIA GPU detected: $GPU_NAME (${VRAM_GB}GB VRAM)"
    fi
fi

if [ "$HAS_NVIDIA" = false ]; then
    echo -e "  ${YELLOW}!${NC} No NVIDIA GPU detected - will use CPU mode"
fi

RAM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
RAM_GB=$((RAM_KB / 1024 / 1024))
echo -e "  ${GREEN}✓${NC} System RAM: ${RAM_GB}GB"

echo

# Determine recommended installation
echo "Recommended installation:"
if [ "$HAS_NVIDIA" = true ] && [ "$VRAM_GB" -ge 8 ]; then
    RECOMMENDED="gpu"
    echo -e "  ${GREEN}→${NC} GPU mode (faster-whisper with CUDA)"
else
    RECOMMENDED="cpu"
    echo -e "  ${GREEN}→${NC} CPU mode (faster-whisper on CPU)"
fi

echo

# Ask user for installation type
echo "Select installation type:"
echo "  1) Minimal    - Core audio recording only"
echo "  2) CPU        - Add STT with CPU support (recommended for laptops)"
echo "  3) GPU        - Add STT with GPU support (requires NVIDIA + CUDA)"
echo "  4) Full       - All features including diarization"
echo

read -p "Enter choice [1-4] (default: 2 for CPU): " choice
choice=${choice:-2}

echo

case $choice in
    1)
        echo "Installing minimal (core only)..."
        uv pip install -e .
        ;;
    2)
        echo "Installing with CPU STT support..."
        uv pip install -e .
        uv pip install faster-whisper
        # Force CPU-only torch to avoid CUDA issues
        uv pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
    3)
        echo "Installing with GPU STT support..."
        if [ "$HAS_NVIDIA" = false ]; then
            echo -e "${YELLOW}Warning: No NVIDIA GPU detected. GPU mode may not work.${NC}"
            read -p "Continue anyway? [y/N]: " confirm
            if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
                echo "Aborted."
                exit 0
            fi
        fi
        uv pip install -e .
        uv pip install faster-whisper
        # Install CUDA-enabled torch
        uv pip install torch --index-url https://download.pytorch.org/whl/cu121
        ;;
    4)
        echo "Installing full (all features)..."
        uv pip install -e .
        uv pip install faster-whisper
        if [ "$HAS_NVIDIA" = true ]; then
            uv pip install torch --index-url https://download.pytorch.org/whl/cu121
        else
            uv pip install torch --index-url https://download.pytorch.org/whl/cpu
        fi
        echo
        echo -e "${YELLOW}Note: Speaker diarization requires a HuggingFace token.${NC}"
        echo "Get one at: https://huggingface.co/settings/tokens"
        echo "Then set: export HF_TOKEN=your_token"
        echo
        uv pip install pyannote-audio
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo
echo "Quick start:"
echo "  conot list-devices           # List audio input devices"
echo "  conot record                 # Record audio"
echo "  conot transcribe audio.wav   # Transcribe audio file"
echo "  conot transcribe --live      # Live transcription"
echo
echo "For CPU mode (if GPU has issues):"
echo "  conot transcribe audio.wav --compute-device cpu"
echo
