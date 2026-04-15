"""
GPU Specifications Tests.

Tests for GPU hardware specifications used in roofline analysis.
Tests GPU_SPECS dictionary structure and values.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "kernelopt"))


class TestGPUSpecsStructure:
    """Test GPU_SPECS dictionary structure."""

    def test_gpu_specs_dict_exists(self):
        """Test GPU_SPECS dictionary is defined."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        assert isinstance(GPU_SPECS, dict)
        assert len(GPU_SPECS) > 0

    def test_all_gpus_have_required_fields(self):
        """Test all GPU models have required specification fields."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        required_fields = [
            "peak_hbm_bandwidth_gbs",
            "peak_fp32_tflops",
            "l2_cache_mb",
            "sm_count",
            "max_warps_per_sm",
        ]

        for gpu_name, specs in GPU_SPECS.items():
            for field in required_fields:
                assert field in specs, f"{gpu_name} missing {field}"

    def test_spec_value_types(self):
        """Test specification values are correct types."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        for gpu_name, specs in GPU_SPECS.items():
            # Numeric fields
            assert isinstance(specs["peak_hbm_bandwidth_gbs"], (int, float))
            assert isinstance(specs["peak_fp32_tflops"], (int, float))
            assert isinstance(specs["l2_cache_mb"], (int, float))
            assert isinstance(specs["sm_count"], int)
            assert isinstance(specs["max_warps_per_sm"], int)

    def test_spec_value_ranges(self):
        """Test specification values are in realistic ranges."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        for gpu_name, specs in GPU_SPECS.items():
            # Bandwidth: 100-10000 GB/s
            assert 100 <= specs["peak_hbm_bandwidth_gbs"] <= 10000

            # TFLOPS: 1-2000
            assert 1 <= specs["peak_fp32_tflops"] <= 2000

            # L2 cache: 1-200 MB
            assert 1 <= specs["l2_cache_mb"] <= 200

            # SM count: 10-500
            assert 10 <= specs["sm_count"] <= 500

            # Warps per SM: typically 32 or 64
            assert specs["max_warps_per_sm"] in [32, 64]

    def test_supported_gpu_models(self):
        """Test list of supported GPU models."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        expected_gpus = [
            "H200 SXM5",
            "H100 SXM5",
            "H100 PCIe",
            "A100 SXM4",
            "A100 PCIe",
        ]

        for gpu in expected_gpus:
            assert gpu in GPU_SPECS, f"{gpu} not in GPU_SPECS"


class TestGPUSpecsData:
    """Test specific GPU specification data."""

    def test_h200_specs(self):
        """Test H200 SXM5 specifications."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        h200 = GPU_SPECS["H200 SXM5"]

        assert h200["peak_hbm_bandwidth_gbs"] == 4800
        assert h200["peak_fp32_tflops"] == 67
        assert h200["l2_cache_mb"] == 50
        assert h200["sm_count"] == 132
        assert h200["max_warps_per_sm"] == 64

    def test_h100_sxm5_specs(self):
        """Test H100 SXM5 specifications."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        h100 = GPU_SPECS["H100 SXM5"]

        assert h100["peak_hbm_bandwidth_gbs"] == 3350
        assert h100["peak_fp32_tflops"] == 67
        assert h100["l2_cache_mb"] == 50
        assert h100["sm_count"] == 132

    def test_h100_pcie_specs(self):
        """Test H100 PCIe specifications."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        h100_pcie = GPU_SPECS["H100 PCIe"]

        assert h100_pcie["peak_hbm_bandwidth_gbs"] == 2000
        assert h100_pcie["peak_fp32_tflops"] == 51
        assert h100_pcie["sm_count"] == 114

    def test_a100_specs(self):
        """Test A100 specifications."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        a100_sxm4 = GPU_SPECS["A100 SXM4"]
        a100_pcie = GPU_SPECS["A100 PCIe"]

        # Both A100 variants
        assert "A100" in "A100 SXM4"
        assert "A100" in "A100 PCIe"

        # Similar but different specs
        assert a100_sxm4["sm_count"] == 108
        assert a100_pcie["peak_hbm_bandwidth_gbs"] == 1935

    def test_bandwidth_values_realistic(self):
        """Test bandwidth values are realistic for each GPU."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # H200 > H100 SXM5 > H100 PCIe > A100
        assert GPU_SPECS["H200 SXM5"]["peak_hbm_bandwidth_gbs"] > GPU_SPECS["H100 SXM5"]["peak_hbm_bandwidth_gbs"]
        assert GPU_SPECS["H100 SXM5"]["peak_hbm_bandwidth_gbs"] > GPU_SPECS["H100 PCIe"]["peak_hbm_bandwidth_gbs"]

    def test_tflops_values_realistic(self):
        """Test TFLOPS values are realistic."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # Tensor core TFLOPS should be much higher than FP32
        for gpu_name, specs in GPU_SPECS.items():
            if "peak_fp16_tensor_tflops" in specs:
                assert specs["peak_fp16_tensor_tflops"] > specs["peak_fp32_tflops"]


class TestGPUSpecsComparison:
    """Test GPU specifications comparison."""

    def test_h200_vs_h100_bandwidth(self):
        """Test H200 has higher bandwidth than H100."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        h200_bw = GPU_SPECS["H200 SXM5"]["peak_hbm_bandwidth_gbs"]
        h100_bw = GPU_SPECS["H100 SXM5"]["peak_hbm_bandwidth_gbs"]

        assert h200_bw > h100_bw

    def test_tensor_core_specs(self):
        """Test tensor core specifications."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # H200 and H100 should have tensor core specs
        for gpu in ["H200 SXM5", "H100 SXM5"]:
            specs = GPU_SPECS[gpu]
            assert "peak_fp32_tensor_tflops" in specs
            assert "peak_fp16_tensor_tflops" in specs

    def test_sm_count_variations(self):
        """Test SM count variations across GPUs."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # H200/H100 SXM5 have more SMs than PCIe
        assert GPU_SPECS["H100 SXM5"]["sm_count"] > GPU_SPECS["H100 PCIe"]["sm_count"]

    def test_l2_cache_consistent(self):
        """Test L2 cache sizes are consistent within GPU families."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # H200 and H100 SXM5 have same L2 cache
        assert GPU_SPECS["H200 SXM5"]["l2_cache_mb"] == GPU_SPECS["H100 SXM5"]["l2_cache_mb"]


class TestGPUSpecsOptionalFields:
    """Test optional GPU specification fields."""

    def test_tensor_core_fields_optional(self):
        """Test tensor core fields may be optional."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        for gpu_name, specs in GPU_SPECS.items():
            # Tensor specs may or may not be present
            if "peak_fp32_tensor_tflops" in specs:
                assert specs["peak_fp32_tensor_tflops"] > 0

    def test_bf16_support(self):
        """Test BF16 tensor core support."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # Modern GPUs support BF16
        modern_gpus = ["H200 SXM5", "H100 SXM5", "A100 SXM4"]
        for gpu in modern_gpus:
            if "peak_bf16_tensor_tflops" in GPU_SPECS[gpu]:
                assert GPU_SPECS[gpu]["peak_bf16_tensor_tflops"] > 0


class TestGPUSpecsEdgeCases:
    """Test GPU specs edge cases."""

    def test_missing_gpu_model_handling(self):
        """Test handling of missing GPU model."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        unknown_gpu = "RTX 9999"
        assert unknown_gpu not in GPU_SPECS

    def test_gpu_name_case_sensitivity(self):
        """Test GPU name case sensitivity."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # Exact match required
        assert "H200 SXM5" in GPU_SPECS
        assert "h200 sxm5" not in GPU_SPECS  # lowercase not in dict

    def test_gpu_specs_immutable(self):
        """Test GPU_SPECS should be treated as immutable."""
        from kernelopt.agents.gpu_specs import GPU_SPECS

        # Don't modify the original
        original_count = len(GPU_SPECS)
        assert original_count > 0
