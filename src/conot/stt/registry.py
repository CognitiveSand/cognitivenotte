"""Provider registry and auto-selection logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from conot.stt.exceptions import (
    NoProviderAvailableError,
    ProviderNotAvailableError,
    ProviderNotFoundError,
)
from conot.stt.hardware import detect_hardware
from conot.stt.models import HardwareProfile, ProviderTier

if TYPE_CHECKING:
    from conot.stt.protocol import STTProvider

logger = logging.getLogger(__name__)

# Provider accuracy ranking (lower WER = higher priority)
# qwen-asr: ~5-6% WER, faster-whisper/whisper-cpp: ~7.4% WER
PROVIDER_ACCURACY_RANK = {
    "qwen-asr": 1,  # ~5-6% WER - most accurate
    "faster-whisper": 2,  # ~7.4% WER
    "whisper-cpp": 3,  # ~7.4% WER (same model, slower)
}

# Registry of provider classes by name
_PROVIDERS: dict[str, type[STTProvider]] = {}

# Provider tier compatibility
_PROVIDER_TIERS: dict[str, list[ProviderTier]] = {}


def register_provider(
    name: str,
    provider_class: type[STTProvider],
    tiers: list[ProviderTier] | None = None,
) -> None:
    """Register a provider implementation.

    Args:
        name: Provider name (e.g., "faster-whisper").
        provider_class: Provider class implementing STTProvider protocol.
        tiers: List of ProviderTier values this provider supports.
            Defaults to all tiers.
    """
    _PROVIDERS[name] = provider_class
    _PROVIDER_TIERS[name] = tiers or list(ProviderTier)
    logger.debug(f"Registered STT provider: {name}")


def unregister_provider(name: str) -> None:
    """Unregister a provider (mainly for testing).

    Args:
        name: Provider name to unregister.
    """
    _PROVIDERS.pop(name, None)
    _PROVIDER_TIERS.pop(name, None)


def get_registered_providers() -> list[str]:
    """Get list of registered provider names.

    Returns:
        List of provider names.
    """
    return list(_PROVIDERS.keys())


def get_provider(name: str | None = None) -> STTProvider:
    """Get provider by name, or auto-select based on hardware.

    Args:
        name: Provider name, or None for auto-selection.

    Returns:
        Instantiated STTProvider.

    Raises:
        ProviderNotFoundError: If named provider is not registered.
        ProviderNotAvailableError: If named provider is not available.
        NoProviderAvailableError: If no provider is available for auto-selection.
    """
    if name is None or name == "auto":
        return auto_select_provider()

    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS.keys()) or "none"
        raise ProviderNotFoundError(
            f"Provider '{name}' not found. Available: {available}"
        )

    provider_class = _PROVIDERS[name]
    provider = provider_class()

    if not provider.is_available():
        raise ProviderNotAvailableError(
            f"Provider '{name}' is not available. "
            "Check that required dependencies are installed."
        )

    return provider


def auto_select_provider(
    hardware: HardwareProfile | None = None,
) -> STTProvider:
    """Select best available provider for current hardware.

    Args:
        hardware: Hardware profile. If None, will be detected.

    Returns:
        Instantiated STTProvider.

    Raises:
        NoProviderAvailableError: If no provider is available.
    """
    if hardware is None:
        hardware = detect_hardware()

    tier = hardware.recommended_tier
    logger.info(f"Auto-selecting provider for tier: {tier.value}")

    # Get providers compatible with this tier, ordered by preference
    compatible = _get_providers_for_tier(tier)

    if not compatible:
        raise NoProviderAvailableError(
            f"No providers registered for tier: {tier.value}"
        )

    # Try each compatible provider
    for provider_name in compatible:
        provider_class = _PROVIDERS[provider_name]
        try:
            provider = provider_class()
            if provider.is_available():
                logger.info(f"Selected provider: {provider_name}")
                return provider
            else:
                logger.debug(f"Provider {provider_name} not available")
        except Exception as e:
            logger.warning(f"Failed to instantiate {provider_name}: {e}")

    # Try fallback to edge tier if we haven't already
    if tier != ProviderTier.EDGE:
        logger.info("Falling back to edge tier providers")
        return auto_select_provider(
            HardwareProfile(
                has_gpu=False,
                gpu_name=None,
                vram_gb=0.0,
                ram_gb=hardware.ram_gb,
                recommended_tier=ProviderTier.EDGE,
            )
        )

    raise NoProviderAvailableError(
        "No STT provider available. Install faster-whisper or whisper-cpp."
    )


def _get_providers_for_tier(tier: ProviderTier) -> list[str]:
    """Get provider names compatible with a tier, ordered by accuracy preference.

    Selection priority (SYS-STT-027):
    1. Filter providers compatible with hardware tier
    2. Sort by accuracy rank (qwen-asr > faster-whisper > whisper-cpp)
    3. Return in accuracy order

    This ensures users always get the most accurate transcription
    their hardware can support.

    Args:
        tier: The hardware tier.

    Returns:
        List of provider names, most accurate first.
    """
    compatible: list[str] = []

    # Get all providers compatible with this tier
    for name in _PROVIDERS:
        if tier in _PROVIDER_TIERS.get(name, []):
            compatible.append(name)

    # Sort by accuracy rank (lower rank = better accuracy)
    compatible.sort(key=lambda p: PROVIDER_ACCURACY_RANK.get(p, 99))

    return compatible


def _register_builtin_providers() -> None:
    """Register built-in providers. Called on module import."""
    # Import and register qwen-asr provider (highest accuracy)
    try:
        from conot.stt.providers.qwen_asr import QwenASRProvider

        register_provider(
            "qwen-asr",
            QwenASRProvider,
            [ProviderTier.ENTERPRISE, ProviderTier.STANDARD, ProviderTier.EDGE],
        )
    except ImportError:
        logger.debug("qwen-asr provider not available")

    # Import and register faster-whisper provider
    try:
        from conot.stt.providers.faster_whisper import FasterWhisperProvider

        register_provider(
            "faster-whisper",
            FasterWhisperProvider,
            [ProviderTier.ENTERPRISE, ProviderTier.STANDARD, ProviderTier.EDGE],
        )
    except ImportError:
        logger.debug("faster-whisper provider not available")

    # Import and register whisper-cpp provider
    try:
        from conot.stt.providers.whisper_cpp import WhisperCppProvider

        register_provider(
            "whisper-cpp",
            WhisperCppProvider,
            [ProviderTier.ENTERPRISE, ProviderTier.STANDARD, ProviderTier.EDGE],
        )
    except ImportError:
        logger.debug("whisper-cpp provider not available")


# Register built-in providers on module import
_register_builtin_providers()
