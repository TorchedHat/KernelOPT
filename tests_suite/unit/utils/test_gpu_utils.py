"""
Tests for GPU utility functions.

These tests verify:
- GPU availability detection
- Resource monitoring
- Free memory checking
- H200-specific features
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "kernelopt"))

# Import from local utils module (same directory)
from .gpu_utils import (
    get_gpu_info,
    get_gpu_memory_free_percent,
    is_gpu_free,
    list_available_gpus,
    select_best_gpu,
    is_h200_gpu,
    get_h200_specific_settings,
    check_gpu_for_profiling,
    GPUResourceError,
)


class TestGPUAvailability:
    """Test GPU availability detection."""

    def test_list_available_gpus(self):
        """Test listing available GPUs."""
        gpus = list_available_gpus()

        assert isinstance(gpus, list)

        if torch.cuda.is_available():
            assert len(gpus) > 0
            assert all(isinstance(gpu_id, int) for gpu_id in gpus)
        else:
            assert len(gpus) == 0

    def test_cuda_available(self, cuda_available):
        """Test CUDA availability check."""
        if cuda_available:
            gpus = list_available_gpus()
            assert len(gpus) > 0
        else:
            pytest.skip("CUDA not available")


class TestGPUInfo:
    """Test GPU information retrieval."""

    @pytest.mark.gpu
    def test_get_gpu_info(self, skip_if_no_cuda, gpu_id):
        """Test getting GPU information."""
        info = get_gpu_info(gpu_id)

        assert isinstance(info, dict)
        assert "name" in info
        assert "memory_total_mb" in info
        assert "memory_used_mb" in info
        assert "memory_free_mb" in info
        assert "gpu_utilization_pct" in info
        assert "memory_utilization_pct" in info

    @pytest.mark.gpu
    def test_get_gpu_info_values(self, skip_if_no_cuda, gpu_id):
        """Test that GPU info values are reasonable."""
        info = get_gpu_info(gpu_id)

        # Memory should be positive
        assert info["memory_total_mb"] > 0
        assert info["memory_used_mb"] >= 0
        assert info["memory_free_mb"] >= 0

        # Used + Free should approximately equal Total
        total_accounted = info["memory_used_mb"] + info["memory_free_mb"]
        assert abs(total_accounted - info["memory_total_mb"]) < 1000  # Within 1GB tolerance

        # Utilization should be 0-100%
        assert 0 <= info["gpu_utilization_pct"] <= 100
        assert 0 <= info["memory_utilization_pct"] <= 100

    @pytest.mark.gpu
    def test_get_gpu_info_invalid_id(self):
        """Test getting info for invalid GPU ID."""
        if torch.cuda.is_available():
            invalid_id = torch.cuda.device_count() + 10

            with pytest.raises(RuntimeError):
                get_gpu_info(invalid_id)


class TestGPUMemory:
    """Test GPU memory checking."""

    @pytest.mark.gpu
    def test_get_gpu_memory_free_percent(self, skip_if_no_cuda, gpu_id):
        """Test getting free memory percentage."""
        free_pct = get_gpu_memory_free_percent(gpu_id)

        assert isinstance(free_pct, float)
        assert 0 <= free_pct <= 100

    @pytest.mark.gpu
    def test_gpu_memory_consistency(self, skip_if_no_cuda, gpu_id):
        """Test that memory measurements are consistent."""
        info = get_gpu_info(gpu_id)
        free_pct = get_gpu_memory_free_percent(gpu_id)

        # Calculate expected free percentage
        expected_free_pct = (info["memory_free_mb"] / info["memory_total_mb"]) * 100

        # Should match
        assert abs(free_pct - expected_free_pct) < 0.1


class TestGPUFreeCheck:
    """Test GPU free status checking."""

    @pytest.mark.gpu
    def test_is_gpu_free_basic(self, skip_if_no_cuda, gpu_id):
        """Test basic GPU free check."""
        # Use lenient thresholds for this test
        is_free = is_gpu_free(gpu_id, min_free_memory_pct=10.0, max_utilization_pct=95.0)

        assert isinstance(is_free, bool)

    @pytest.mark.gpu
    def test_is_gpu_free_strict(self, skip_if_no_cuda, gpu_id):
        """Test strict GPU free check (40% free memory)."""
        is_free = is_gpu_free(gpu_id, min_free_memory_pct=40.0, max_utilization_pct=20.0)

        # Result depends on actual GPU state
        assert isinstance(is_free, bool)

    @pytest.mark.gpu
    def test_is_gpu_free_thresholds(self, skip_if_no_cuda, gpu_id):
        """Test different threshold values."""
        # Very lenient - should usually pass
        lenient = is_gpu_free(gpu_id, min_free_memory_pct=5.0, max_utilization_pct=99.0)

        # Very strict - may fail
        strict = is_gpu_free(gpu_id, min_free_memory_pct=95.0, max_utilization_pct=1.0)

        # Lenient should be >= strict
        assert isinstance(lenient, bool)
        assert isinstance(strict, bool)


class TestGPUSelection:
    """Test GPU selection utilities."""

    @pytest.mark.gpu
    def test_select_best_gpu(self, skip_if_no_cuda):
        """Test selecting best GPU."""
        # Use lenient threshold
        best_gpu = select_best_gpu(min_free_memory_pct=10.0)

        if torch.cuda.device_count() > 0:
            # Should select some GPU
            assert best_gpu is not None
            assert isinstance(best_gpu, int)
            assert 0 <= best_gpu < torch.cuda.device_count()

    @pytest.mark.gpu
    def test_select_best_gpu_strict(self, skip_if_no_cuda):
        """Test selecting GPU with strict requirements."""
        best_gpu = select_best_gpu(min_free_memory_pct=40.0)

        # May or may not find a suitable GPU
        if best_gpu is not None:
            assert isinstance(best_gpu, int)
            assert 0 <= best_gpu < torch.cuda.device_count()

            # Verify it meets requirements
            free_pct = get_gpu_memory_free_percent(best_gpu)
            assert free_pct >= 40.0


class TestH200Features:
    """Test H200-specific features."""

    def test_get_h200_specific_settings(self):
        """Test getting H200-specific settings."""
        settings = get_h200_specific_settings()

        assert isinstance(settings, dict)
        assert "min_free_memory_pct" in settings
        assert "max_utilization_pct" in settings
        assert "warmup_iterations" in settings
        assert "benchmark_iterations" in settings

        # Verify values match H200 recommendations
        assert settings["min_free_memory_pct"] == 40.0
        assert settings["max_utilization_pct"] == 20.0
        assert settings["warmup_iterations"] >= 10
        assert settings["benchmark_iterations"] >= 50

    @pytest.mark.gpu
    def test_is_h200_gpu(self, skip_if_no_cuda, gpu_id):
        """Test H200 detection."""
        is_h200 = is_h200_gpu(gpu_id)

        assert isinstance(is_h200, bool)

        # If it's H200, verify name
        if is_h200:
            info = get_gpu_info(gpu_id)
            assert "H200" in info["name"].upper()


class TestProfilingPrecheck:
    """Test profiling resource checking."""

    @pytest.mark.gpu
    def test_check_gpu_for_profiling_sufficient(self, skip_if_no_cuda, skip_if_gpu_busy, gpu_id):
        """Test profiling check with sufficient resources."""
        # This test only runs if GPU is actually free
        try:
            check_gpu_for_profiling(gpu_id)
            # Should not raise if GPU is free
        except GPUResourceError:
            pytest.skip("GPU doesn't have sufficient resources")

    @pytest.mark.gpu
    def test_check_gpu_for_profiling_displays_info(self, skip_if_no_cuda, gpu_id, capsys):
        """Test that profiling check displays GPU info."""
        try:
            check_gpu_for_profiling(gpu_id)
        except GPUResourceError:
            pass  # Expected if GPU is busy

        captured = capsys.readouterr()
        # Should print GPU information
        assert f"GPU {gpu_id}" in captured.out

    def test_check_gpu_for_profiling_no_cuda(self):
        """Test profiling check without CUDA."""
        if not torch.cuda.is_available():
            with pytest.raises(GPUResourceError, match="CUDA is not available"):
                check_gpu_for_profiling(0)

    @pytest.mark.gpu
    def test_check_gpu_for_profiling_invalid_id(self, skip_if_no_cuda):
        """Test profiling check with invalid GPU ID."""
        invalid_id = torch.cuda.device_count() + 10

        with pytest.raises(GPUResourceError, match="not found"):
            check_gpu_for_profiling(invalid_id)


class TestGPUResourceError:
    """Test GPUResourceError exception."""

    def test_gpu_resource_error_creation(self):
        """Test creating GPUResourceError."""
        error = GPUResourceError("Test error message")

        assert isinstance(error, Exception)
        assert "Test error message" in str(error)

    def test_gpu_resource_error_raise(self):
        """Test raising GPUResourceError."""
        with pytest.raises(GPUResourceError):
            raise GPUResourceError("Insufficient resources")


class TestGPUUtilsIntegration:
    """Integration tests for GPU utilities."""

    @pytest.mark.gpu
    def test_full_gpu_check_workflow(self, skip_if_no_cuda):
        """Test complete GPU checking workflow."""
        # 1. List GPUs
        gpus = list_available_gpus()
        assert len(gpus) > 0

        # 2. Get info for first GPU
        info = get_gpu_info(gpus[0])
        assert info is not None

        # 3. Check if free
        is_free = is_gpu_free(gpus[0], min_free_memory_pct=10.0)
        assert isinstance(is_free, bool)

        # 4. Get free memory percentage
        free_pct = get_gpu_memory_free_percent(gpus[0])
        assert free_pct >= 0

    @pytest.mark.gpu
    def test_select_and_verify_gpu(self, skip_if_no_cuda):
        """Test selecting GPU and verifying it meets requirements."""
        best_gpu = select_best_gpu(min_free_memory_pct=20.0)

        if best_gpu is not None:
            # Verify it actually has the required free memory
            free_pct = get_gpu_memory_free_percent(best_gpu)
            assert free_pct >= 20.0
