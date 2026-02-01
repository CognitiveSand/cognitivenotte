#!/bin/bash
# Conot Installation Script
# Installs conot with optional STT providers
# Idempotent: safe to run multiple times

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Conot Installation Script ===${NC}"
echo

# Find uv binary - check common locations
find_uv() {
    # Check if uv is in PATH
    if command -v uv &> /dev/null; then
        echo "$(command -v uv)"
        return 0
    fi

    # Check common install locations
    local locations=(
        "$HOME/.local/bin/uv"
        "$HOME/.cargo/bin/uv"
        "/usr/local/bin/uv"
        "/usr/bin/uv"
    )

    for loc in "${locations[@]}"; do
        if [ -x "$loc" ]; then
            echo "$loc"
            return 0
        fi
    done

    return 1
}

UV_BIN=$(find_uv) || {
    echo -e "${RED}Error: uv is not installed.${NC}"
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo
    echo "Or see: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

echo -e "${GREEN}✓${NC} Found uv: $UV_BIN"

# Detect hardware
echo
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
    RECOMMENDED="3"
    echo -e "  ${GREEN}→${NC} GPU mode (faster-whisper with CUDA)"
else
    RECOMMENDED="2"
    echo -e "  ${GREEN}→${NC} CPU mode (faster-whisper on CPU)"
fi

echo

# Ask user for installation type
echo "Select installation type:"
echo "  1) Minimal    - Core audio recording only"
echo "  2) CPU        - Add STT with CPU support (recommended for laptops)"
echo "  3) GPU        - Add STT with GPU support (requires NVIDIA + CUDA)"
echo "  4) Qwen       - Add Qwen3-ASR (~30% better accuracy than Whisper)"
echo "  5) Full       - All features including diarization + Qwen3-ASR"
echo

read -p "Enter choice [1-5] (default: $RECOMMENDED): " choice
choice=${choice:-$RECOMMENDED}

echo

# Change to script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${BLUE}Installing from: $PROJECT_DIR${NC}"
echo

case $choice in
    1)
        echo "Installing minimal (core only)..."
        "$UV_BIN" sync
        ;;
    2)
        echo "Installing with CPU STT support..."
        "$UV_BIN" sync --extra stt-gpu --extra stt-vad
        echo
        echo -e "${YELLOW}Note: If you get CUDA errors, the system will auto-fallback to CPU.${NC}"
        echo "You can also force CPU mode with: --compute-device cpu"
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
        "$UV_BIN" sync --extra stt-gpu --extra stt-vad
        ;;
    4)
        echo "Installing with Qwen3-ASR support..."
        "$UV_BIN" sync --extra stt-qwen --extra stt-vad
        echo
        echo -e "${GREEN}Qwen3-ASR installed.${NC} It offers ~30% better accuracy than Whisper."
        echo "Models will be downloaded on first use (~3GB for 1.7B model)."
        ;;
    5)
        echo "Installing full (all features)..."
        "$UV_BIN" sync --extra stt-full
        echo
        echo -e "${YELLOW}Note: Speaker diarization requires a HuggingFace token.${NC}"
        echo "Get one at: https://huggingface.co/settings/tokens"
        echo "Then set: export HF_TOKEN=your_token"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo

# Verify installation
echo "Verifying installation..."
if "$UV_BIN" run python -c "from conot.stt.registry import get_registered_providers; providers = get_registered_providers(); print(f'STT providers available: {providers}')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} STT providers loaded successfully"
else
    echo -e "${YELLOW}!${NC} No STT providers detected (install option 2-5 for transcription)"
fi

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
