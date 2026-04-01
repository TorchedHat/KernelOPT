"""GPU hardware specifications for roofline analysis."""

GPU_SPECS = {
    "H100 SXM5": {
        "peak_hbm_bandwidth_gbs": 3350,
        "peak_fp32_tflops": 67,
        "peak_fp32_tensor_tflops": 989,
        "peak_fp16_tensor_tflops": 1979,
        "peak_bf16_tensor_tflops": 1979,
        "l2_cache_mb": 50,
        "sm_count": 132,
        "max_warps_per_sm": 64,
    },
    "H100 PCIe": {
        "peak_hbm_bandwidth_gbs": 2000,
        "peak_fp32_tensor_tflops": 756,
        "peak_fp16_tensor_tflops": 1513,
        "sm_count": 114,
        "max_warps_per_sm": 64,
    },
    "A100 SXM4": {
        "peak_hbm_bandwidth_gbs": 2000,
        "peak_fp32_tflops": 19.5,
        "peak_fp32_tensor_tflops": 312,
        "peak_fp16_tensor_tflops": 624,
        "l2_cache_mb": 40,
        "sm_count": 108,
        "max_warps_per_sm": 64,
    },
    "RTX 4090": {
        "peak_hbm_bandwidth_gbs": 1008,
        "peak_fp32_tflops": 82.6,
        "peak_fp16_tensor_tflops": 330,
        "l2_cache_mb": 72,
        "sm_count": 128,
        "max_warps_per_sm": 64,
    },
    "RTX 3090": {
        "peak_hbm_bandwidth_gbs": 936,
        "peak_fp32_tflops": 35.6,
        "sm_count": 82,
        "max_warps_per_sm": 64,
    },
}


def detect_gpu_model(sm_count: int | None, compute_capability: str) -> str:
    """Detect GPU model from SM count and compute capability.

    Returns a key from GPU_SPECS or 'unknown'.
    """
    if not sm_count:
        return "unknown"

    # CC 9.0 = H100
    if compute_capability.startswith("9.0"):
        if sm_count == 132:
            return "H100 SXM5"
        elif sm_count == 114:
            return "H100 PCIe"

    # CC 8.0 = A100
    elif compute_capability.startswith("8.0"):
        if sm_count == 108:
            return "A100 SXM4"

    # CC 8.6 = RTX 30xx/40xx
    elif compute_capability.startswith("8.6"):
        if sm_count == 128:
            return "RTX 4090"
        elif sm_count == 82:
            return "RTX 3090"

    return "unknown"


def get_gpu_specs(sm_count: int | None, compute_capability: str) -> dict:
    """Get GPU specs for roofline analysis.

    Returns GPU specs dict or default A100 specs if unknown.
    """
    model = detect_gpu_model(sm_count, compute_capability)

    if model in GPU_SPECS:
        return {
            "gpu_model": model,
            "gpu_model_detected": True,
            **GPU_SPECS[model]
        }
    else:
        # Default to A100 if unknown
        return {
            "gpu_model": "A100 SXM4 (assumed)",
            "gpu_model_detected": False,
            **GPU_SPECS["A100 SXM4"]
        }
