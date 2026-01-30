"""Hardware detection for STT provider selection."""

from __future__ import annotations

import logging
import shutil
import subprocess

from conot.stt.exceptions import HardwareDetectionError
from conot.stt.models import HardwareProfile, ProviderTier

logger = logging.getLogger(__name__)

# Tier thresholds
ENTERPRISE_VRAM_GB = 12.0
STANDARD_VRAM_GB = 4.0


def detect_hardware() -> HardwareProfile:
    """Detect GPU, VRAM, and RAM. Return recommended tier.

    Returns:
        HardwareProfile with detected hardware info and recommended tier.

    Raises:
        HardwareDetectionError: If hardware detection fails critically.
    """
    has_gpu = False
    gpu_name: str | None = None
    vram_gb = 0.0
    ram_gb = _detect_system_ram()

    # Try to detect NVIDIA GPU
    nvidia_info = _detect_nvidia_gpu()
    if nvidia_info:
        has_gpu = True
        gpu_name = str(nvidia_info["name"])
        vram_gb = float(nvidia_info["vram_gb"])
        logger.info(f"Detected NVIDIA GPU: {gpu_name} with {vram_gb:.1f}GB VRAM")

    # Determine recommended tier
    recommended_tier = _determine_tier(has_gpu, vram_gb, ram_gb)
    logger.info(f"Recommended provider tier: {recommended_tier.value}")

    return HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        recommended_tier=recommended_tier,
    )


def _detect_nvidia_gpu() -> dict[str, str | float] | None:
    """Detect NVIDIA GPU using nvidia-smi.

    Returns:
        Dict with 'name' and 'vram_gb' if GPU found, None otherwise.
    """
    # Check if nvidia-smi is available
    if not shutil.which("nvidia-smi"):
        logger.debug("nvidia-smi not found, no NVIDIA GPU detected")
        return None

    try:
        # Query GPU name and memory
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
            logger.debug(f"nvidia-smi failed: {result.stderr}")
            return None

        output = result.stdout.strip()
        if not output:
            return None

        # Parse first GPU (in case of multi-GPU)
        first_line = output.split("\n")[0]
        parts = first_line.split(", ")
        if len(parts) != 2:
            logger.warning(f"Unexpected nvidia-smi output: {first_line}")
            return None

        gpu_name = parts[0].strip()
        vram_mb = float(parts[1].strip())
        vram_gb = vram_mb / 1024.0

        return {"name": gpu_name, "vram_gb": vram_gb}

    except subprocess.TimeoutExpired:
        logger.warning("nvidia-smi timed out")
        return None
    except (ValueError, IndexError) as e:
        logger.warning(f"Failed to parse nvidia-smi output: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error detecting NVIDIA GPU: {e}")
        return None


def _detect_system_ram() -> float:
    """Detect system RAM in GB.

    Returns:
        System RAM in GB.

    Raises:
        HardwareDetectionError: If RAM detection fails.
    """
    try:
        import psutil

        ram_bytes: int = psutil.virtual_memory().total
        ram_gb: float = ram_bytes / (1024**3)
        logger.debug(f"Detected {ram_gb:.1f}GB system RAM")
        return ram_gb
    except ImportError:
        logger.warning("psutil not available, using fallback RAM detection")
        return _detect_ram_fallback()
    except Exception as e:
        raise HardwareDetectionError(f"Failed to detect system RAM: {e}") from e


def _detect_ram_fallback() -> float:
    """Fallback RAM detection using /proc/meminfo on Linux.

    Returns:
        System RAM in GB, or 8.0 as conservative default.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    # Format: "MemTotal:       16384000 kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb / (1024**2)
    except Exception as e:
        logger.warning(f"Fallback RAM detection failed: {e}")

    # Conservative default
    logger.warning("Using default RAM value of 8GB")
    return 8.0


def _determine_tier(has_gpu: bool, vram_gb: float, ram_gb: float) -> ProviderTier:
    """Determine recommended provider tier based on hardware.

    Args:
        has_gpu: Whether a GPU was detected.
        vram_gb: GPU VRAM in GB.
        ram_gb: System RAM in GB.

    Returns:
        Recommended ProviderTier.
    """
    if has_gpu:
        if vram_gb >= ENTERPRISE_VRAM_GB:
            return ProviderTier.ENTERPRISE
        elif vram_gb >= STANDARD_VRAM_GB:
            return ProviderTier.STANDARD
        else:
            # GPU with less than 4GB VRAM - use CPU path
            logger.info(
                f"GPU has only {vram_gb:.1f}GB VRAM, recommending CPU-based tier"
            )
            return ProviderTier.EDGE
    else:
        # No GPU - always use edge tier
        return ProviderTier.EDGE


def get_recommended_model_size(profile: HardwareProfile) -> str:
    """Get recommended Whisper model size for hardware profile.

    Args:
        profile: Hardware profile from detect_hardware().

    Returns:
        Recommended model size string (e.g., "large-v3", "medium", "small").
    """
    if profile.recommended_tier == ProviderTier.ENTERPRISE:
        return "large-v3"
    elif profile.recommended_tier == ProviderTier.STANDARD:
        # Check if we can fit medium or need small
        if profile.vram_gb >= 6.0:
            return "medium"
        else:
            return "small"
    else:
        # Edge tier - CPU-optimized
        if profile.ram_gb >= 8.0:
            return "small"
        else:
            return "tiny"
