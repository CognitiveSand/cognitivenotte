"""Tests for Qwen3-ASR STT provider."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conot.stt.models import HardwareProfile, ProviderCapabilities, ProviderTier


class TestQwenASRProvider:
    """Tests for QwenASRProvider class."""

    def test_import_provider(self):
        """Test that provider can be imported."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        assert QwenASRProvider is not None

    def test_provider_instantiation(self):
        """Test provider can be instantiated."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider()
        assert provider is not None
        assert provider._model_name == "auto"
        assert provider._device == "auto"

    def test_provider_with_explicit_args(self):
        """Test provider with explicit arguments."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(
            model_name="Qwen/Qwen3-ASR-0.6B",
            device="cpu",
            language="fr",
            use_vllm=False,
            use_forced_aligner=True,
        )
        assert provider._model_name == "Qwen/Qwen3-ASR-0.6B"
        assert provider._device == "cpu"
        assert provider._language == "fr"
        assert provider._use_vllm is False
        assert provider._use_forced_aligner is True

    def test_is_available_without_qwen_asr(self):
        """Test availability check when qwen-asr not installed."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider()
        # Will be False if qwen-asr not installed
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_get_capabilities(self):
        """Test capabilities declaration."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider()
        caps = provider.get_capabilities()

        assert isinstance(caps, ProviderCapabilities)
        assert "fr" in caps.languages
        assert "en" in caps.languages
        assert caps.supports_streaming is True
        assert caps.supports_diarization is False
        assert caps.requires_gpu is False
        assert caps.min_memory_gb == 4.0

    def test_get_capabilities_with_forced_aligner(self):
        """Test capabilities with forced aligner enabled."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(use_forced_aligner=True)
        caps = provider.get_capabilities()
        assert caps.supports_word_timestamps is True

    def test_get_capabilities_without_forced_aligner(self):
        """Test capabilities without forced aligner."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(use_forced_aligner=False)
        caps = provider.get_capabilities()
        assert caps.supports_word_timestamps is False

    def test_get_model_info(self):
        """Test model info resolution."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(model_name="Qwen/Qwen3-ASR-0.6B", device="cpu")
        info = provider.get_model_info()

        assert info["name"] == "Qwen3-ASR-0.6B"
        assert info["provider"] == "qwen-asr"
        assert info["device"] == "cpu"
        assert info["backend"] == "transformers"

    def test_get_model_info_vllm_backend(self):
        """Test model info with vLLM backend."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(
            model_name="Qwen/Qwen3-ASR-1.7B", device="cuda:0", use_vllm=True
        )
        info = provider.get_model_info()

        assert info["name"] == "Qwen3-ASR-1.7B"
        assert info["backend"] == "vllm"

    def test_resolve_settings_auto_model_enterprise(self):
        """Test auto model selection for enterprise tier."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        with patch(
            "conot.stt.providers.qwen_asr.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value = HardwareProfile(
                has_gpu=True,
                gpu_name="RTX 4090",
                vram_gb=24.0,
                ram_gb=64.0,
                recommended_tier=ProviderTier.ENTERPRISE,
            )
            provider = QwenASRProvider(model_name="auto", device="auto")
            model_name, device = provider._resolve_settings()

            assert model_name == "Qwen/Qwen3-ASR-1.7B"
            assert device == "cuda:0"

    def test_resolve_settings_auto_model_edge(self):
        """Test auto model selection for edge tier."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        with patch(
            "conot.stt.providers.qwen_asr.detect_hardware"
        ) as mock_detect:
            mock_detect.return_value = HardwareProfile(
                has_gpu=False,
                gpu_name=None,
                vram_gb=0.0,
                ram_gb=8.0,
                recommended_tier=ProviderTier.EDGE,
            )
            provider = QwenASRProvider(model_name="auto", device="auto")
            model_name, device = provider._resolve_settings()

            assert model_name == "Qwen/Qwen3-ASR-0.6B"
            assert device == "cpu"

    def test_resolve_settings_shorthand_model_names(self):
        """Test shorthand model name resolution."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        # Test 1.7B shorthand
        provider = QwenASRProvider(model_name="1.7B")
        model_name, _ = provider._resolve_settings()
        assert model_name == "Qwen/Qwen3-ASR-1.7B"

        # Test 0.6B shorthand
        provider = QwenASRProvider(model_name="0.6B")
        model_name, _ = provider._resolve_settings()
        assert model_name == "Qwen/Qwen3-ASR-0.6B"

    def test_language_auto_converts_to_none(self):
        """Test that language='auto' is converted to None for auto-detection."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(language="auto")
        assert provider._language is None

    def test_language_explicit_preserved(self):
        """Test that explicit language is preserved."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider(language="fr")
        assert provider._language == "fr"


class TestQwenASRProviderProtocolCompliance:
    """Tests to verify QwenASRProvider complies with the STTProvider protocol."""

    def test_has_required_methods(self):
        """Test provider has all required protocol methods."""
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider()
        assert hasattr(provider, "transcribe")
        assert hasattr(provider, "transcribe_stream")
        assert hasattr(provider, "detect_language")
        assert hasattr(provider, "get_capabilities")
        assert hasattr(provider, "is_available")
        assert callable(provider.transcribe)
        assert callable(provider.transcribe_stream)
        assert callable(provider.detect_language)
        assert callable(provider.get_capabilities)
        assert callable(provider.is_available)

    def test_inherits_from_base_provider(self):
        """Test provider inherits from BaseSTTProvider."""
        from conot.stt.providers.base import BaseSTTProvider
        from conot.stt.providers.qwen_asr import QwenASRProvider

        provider = QwenASRProvider()
        assert isinstance(provider, BaseSTTProvider)


class TestQwenASRRegistration:
    """Tests for Qwen3-ASR provider registration."""

    def test_provider_registered(self):
        """Test that qwen-asr is registered in the provider registry."""
        from conot.stt.registry import get_registered_providers

        providers = get_registered_providers()
        assert "qwen-asr" in providers

    def test_provider_preference_rank(self):
        """Test that faster-whisper is preferred by default."""
        from conot.stt.registry import PROVIDER_PREFERENCE_RANK

        assert "qwen-asr" in PROVIDER_PREFERENCE_RANK
        assert "faster-whisper" in PROVIDER_PREFERENCE_RANK
        # faster-whisper should be preferred (lower rank = more preferred)
        assert PROVIDER_PREFERENCE_RANK["faster-whisper"] < PROVIDER_PREFERENCE_RANK["qwen-asr"]

    def test_providers_sorted_by_preference(self):
        """Test that providers are sorted by preference in tier selection."""
        from conot.stt.registry import _get_providers_for_tier
        from conot.stt.models import ProviderTier

        providers = _get_providers_for_tier(ProviderTier.ENTERPRISE)
        # faster-whisper should be first (preferred by default)
        if "faster-whisper" in providers:
            assert providers[0] == "faster-whisper"
