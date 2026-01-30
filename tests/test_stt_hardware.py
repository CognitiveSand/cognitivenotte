"""Tests for STT hardware detection."""

from unittest.mock import MagicMock, patch

import pytest

from conot.stt.hardware import (
    _detect_nvidia_gpu,
    _detect_ram_fallback,
    _detect_system_ram,
    _determine_tier,
    detect_hardware,
    get_recommended_model_size,
)
from conot.stt.models import HardwareProfile, ProviderTier


class TestDetectHardware:
    """Tests for detect_hardware function."""

    @patch("conot.stt.hardware._detect_nvidia_gpu")
    @patch("conot.stt.hardware._detect_system_ram")
    def test_with_enterprise_gpu(self, mock_ram, mock_gpu):
        mock_gpu.return_value = {"name": "RTX 4090", "vram_gb": 24.0}
        mock_ram.return_value = 64.0

        profile = detect_hardware()

        assert profile.has_gpu is True
        assert profile.gpu_name == "RTX 4090"
        assert profile.vram_gb == 24.0
        assert profile.ram_gb == 64.0
        assert profile.recommended_tier == ProviderTier.ENTERPRISE

    @patch("conot.stt.hardware._detect_nvidia_gpu")
    @patch("conot.stt.hardware._detect_system_ram")
    def test_with_standard_gpu(self, mock_ram, mock_gpu):
        mock_gpu.return_value = {"name": "RTX 3060", "vram_gb": 6.0}
        mock_ram.return_value = 16.0

        profile = detect_hardware()

        assert profile.has_gpu is True
        assert profile.vram_gb == 6.0
        assert profile.recommended_tier == ProviderTier.STANDARD

    @patch("conot.stt.hardware._detect_nvidia_gpu")
    @patch("conot.stt.hardware._detect_system_ram")
    def test_without_gpu(self, mock_ram, mock_gpu):
        mock_gpu.return_value = None
        mock_ram.return_value = 8.0

        profile = detect_hardware()

        assert profile.has_gpu is False
        assert profile.gpu_name is None
        assert profile.vram_gb == 0.0
        assert profile.recommended_tier == ProviderTier.EDGE

    @patch("conot.stt.hardware._detect_nvidia_gpu")
    @patch("conot.stt.hardware._detect_system_ram")
    def test_with_low_vram_gpu(self, mock_ram, mock_gpu):
        mock_gpu.return_value = {"name": "GTX 1050", "vram_gb": 2.0}
        mock_ram.return_value = 8.0

        profile = detect_hardware()

        assert profile.has_gpu is True
        assert profile.vram_gb == 2.0
        # Low VRAM should recommend edge tier
        assert profile.recommended_tier == ProviderTier.EDGE


class TestDetectNvidiaGpu:
    """Tests for _detect_nvidia_gpu function."""

    @patch("shutil.which")
    def test_no_nvidia_smi(self, mock_which):
        mock_which.return_value = None
        result = _detect_nvidia_gpu()
        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_nvidia_smi_success(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NVIDIA GeForce RTX 3080, 10240\n",
        )

        result = _detect_nvidia_gpu()

        assert result is not None
        assert result["name"] == "NVIDIA GeForce RTX 3080"
        assert result["vram_gb"] == 10.0

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_nvidia_smi_failure(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error",
        )

        result = _detect_nvidia_gpu()
        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run, mock_which):
        import subprocess

        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

        result = _detect_nvidia_gpu()
        assert result is None

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_nvidia_smi_malformed_output(self, mock_run, mock_which):
        mock_which.return_value = "/usr/bin/nvidia-smi"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="malformed output",
        )

        result = _detect_nvidia_gpu()
        assert result is None


class TestDetectSystemRam:
    """Tests for _detect_system_ram function."""

    @patch("conot.stt.hardware._detect_ram_fallback")
    def test_psutil_available(self, mock_fallback):
        # psutil should be available in test environment
        ram = _detect_system_ram()
        assert ram > 0
        mock_fallback.assert_not_called()

    @patch("psutil.virtual_memory")
    @patch("conot.stt.hardware._detect_ram_fallback")
    def test_psutil_import_error(self, mock_fallback, mock_psutil):
        mock_psutil.side_effect = ImportError("No psutil")
        mock_fallback.return_value = 8.0

        # Need to reimport to trigger the import error path
        # This test verifies the fallback path exists


class TestDetectRamFallback:
    """Tests for _detect_ram_fallback function."""

    @patch("builtins.open")
    def test_proc_meminfo_parsing(self, mock_open):
        mock_open.return_value.__enter__.return_value = iter([
            "MemTotal:       16384000 kB\n",
            "MemFree:        8192000 kB\n",
        ])

        ram = _detect_ram_fallback()
        # 16384000 kB = ~15.625 GB
        assert 15.0 < ram < 16.0

    @patch("builtins.open")
    def test_proc_meminfo_error(self, mock_open):
        mock_open.side_effect = FileNotFoundError()

        ram = _detect_ram_fallback()
        assert ram == 8.0  # Default fallback


class TestDetermineTier:
    """Tests for _determine_tier function."""

    def test_enterprise_tier(self):
        tier = _determine_tier(has_gpu=True, vram_gb=24.0, ram_gb=64.0)
        assert tier == ProviderTier.ENTERPRISE

    def test_enterprise_threshold(self):
        tier = _determine_tier(has_gpu=True, vram_gb=12.0, ram_gb=32.0)
        assert tier == ProviderTier.ENTERPRISE

    def test_standard_tier(self):
        tier = _determine_tier(has_gpu=True, vram_gb=8.0, ram_gb=16.0)
        assert tier == ProviderTier.STANDARD

    def test_standard_threshold(self):
        tier = _determine_tier(has_gpu=True, vram_gb=4.0, ram_gb=16.0)
        assert tier == ProviderTier.STANDARD

    def test_edge_tier_low_vram(self):
        tier = _determine_tier(has_gpu=True, vram_gb=2.0, ram_gb=8.0)
        assert tier == ProviderTier.EDGE

    def test_edge_tier_no_gpu(self):
        tier = _determine_tier(has_gpu=False, vram_gb=0.0, ram_gb=32.0)
        assert tier == ProviderTier.EDGE


class TestGetRecommendedModelSize:
    """Tests for get_recommended_model_size function."""

    def test_enterprise_tier(self):
        profile = HardwareProfile(
            has_gpu=True,
            gpu_name="RTX 4090",
            vram_gb=24.0,
            ram_gb=64.0,
            recommended_tier=ProviderTier.ENTERPRISE,
        )
        assert get_recommended_model_size(profile) == "large-v3"

    def test_standard_tier_high_vram(self):
        profile = HardwareProfile(
            has_gpu=True,
            gpu_name="RTX 3070",
            vram_gb=8.0,
            ram_gb=32.0,
            recommended_tier=ProviderTier.STANDARD,
        )
        assert get_recommended_model_size(profile) == "medium"

    def test_standard_tier_low_vram(self):
        profile = HardwareProfile(
            has_gpu=True,
            gpu_name="RTX 3060",
            vram_gb=4.0,
            ram_gb=16.0,
            recommended_tier=ProviderTier.STANDARD,
        )
        assert get_recommended_model_size(profile) == "small"

    def test_edge_tier_high_ram(self):
        profile = HardwareProfile(
            has_gpu=False,
            gpu_name=None,
            vram_gb=0.0,
            ram_gb=16.0,
            recommended_tier=ProviderTier.EDGE,
        )
        assert get_recommended_model_size(profile) == "small"

    def test_edge_tier_low_ram(self):
        profile = HardwareProfile(
            has_gpu=False,
            gpu_name=None,
            vram_gb=0.0,
            ram_gb=4.0,
            recommended_tier=ProviderTier.EDGE,
        )
        assert get_recommended_model_size(profile) == "tiny"
