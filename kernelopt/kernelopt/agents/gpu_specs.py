"""GPU hardware specifications for roofline analysis."""

GPU_SPECS: dict[str, dict] = {
    "H200 SXM5": {
        "peak_hbm_bandwidth_gbs": 4800,
        "peak_fp32_tflops": 67,
        "peak_fp32_tensor_tflops": 989,
        "peak_fp16_tensor_tflops": 1979,
        "peak_bf16_tensor_tflops": 1979,
        "l2_cache_mb": 50,
        "sm_count": 132,
        "max_warps_per_sm": 64,
    },
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
        "peak_fp32_tflops": 51,
        "peak_fp32_tensor_tflops": 756,
        "peak_fp16_tensor_tflops": 1513,
        "peak_bf16_tensor_tflops": 1513,
        "l2_cache_mb": 50,
        "sm_count": 114,
        "max_warps_per_sm": 64,
    },
    "A100 SXM4": {
        "peak_hbm_bandwidth_gbs": 2000,
        "peak_fp32_tflops": 19.5,
        "peak_fp32_tensor_tflops": 312,
        "peak_fp16_tensor_tflops": 624,
        "peak_bf16_tensor_tflops": 312,
        "l2_cache_mb": 40,
        "sm_count": 108,
        "max_warps_per_sm": 64,
    },
    "A100 PCIe": {
        "peak_hbm_bandwidth_gbs": 1935,
        "peak_fp32_tflops": 19.5,
        "peak_fp32_tensor_tflops": 312,
        "peak_fp16_tensor_tflops": 624,
        "peak_bf16_tensor_tflops": 312,
        "l2_cache_mb": 40,
        "sm_count": 108,
        "max_warps_per_sm": 64,
    },
    "RTX 4090": {
        "peak_hbm_bandwidth_gbs": 1008,
        "peak_fp32_tflops": 82.6,
        "peak_fp16_tensor_tflops": 330,
        "peak_bf16_tensor_tflops": 330,
        "l2_cache_mb": 72,
        "sm_count": 128,
        "max_warps_per_sm": 64,
    },
    "RTX 3090": {
        "peak_hbm_bandwidth_gbs": 936,
        "peak_fp32_tflops": 35.6,
        "peak_fp16_tensor_tflops": 142,
        "l2_cache_mb": 6,
        "sm_count": 82,
        "max_warps_per_sm": 64,
    },
    "L40S": {
        "peak_hbm_bandwidth_gbs": 864,
        "peak_fp32_tflops": 91.6,
        "peak_fp16_tensor_tflops": 733,
        "peak_bf16_tensor_tflops": 733,
        "l2_cache_mb": 96,
        "sm_count": 142,
        "max_warps_per_sm": 64,
    },
}

# Priority-ordered list of (substring_in_device_name, GPU_SPECS_key).
# Checked case-insensitively against torch.cuda.get_device_name().
# More specific entries must come before less specific ones.
_DEVICE_NAME_PATTERNS: list[tuple[str, str]] = [
    ("H200",    "H200 SXM5"),
    ("H100 SXM","H100 SXM5"),
    ("H100",    "H100 PCIe"),
    ("A100-SXM","A100 SXM4"),
    ("A100 SXM","A100 SXM4"),
    ("A100",    "A100 SXM4"),   # default A100 to SXM4 (most common)
    ("L40S",    "L40S"),
    ("4090",    "RTX 4090"),
    ("3090",    "RTX 3090"),
]


def _detect_from_device_name(device_name: str) -> str:
    """Match torch.cuda.get_device_name() against known patterns."""
    name_upper = device_name.upper()
    for substr, key in _DEVICE_NAME_PATTERNS:
        if substr.upper() in name_upper:
            return key
    return "unknown"


def _detect_from_cc_and_sm(sm_count: int | None, compute_capability: str) -> str:
    """Fallback: match by compute capability + SM count."""
    if not sm_count:
        return "unknown"
    cc = compute_capability
    if cc.startswith("9.0"):
        if sm_count == 132: return "H100 SXM5"   # conservative — may be H200
        if sm_count == 114: return "H100 PCIe"
    elif cc.startswith("8.0"):
        if sm_count == 108: return "A100 SXM4"
    elif cc.startswith("8.9"):
        if sm_count == 142: return "L40S"
    elif cc.startswith("8.6"):
        if sm_count == 128: return "RTX 4090"
        if sm_count == 82:  return "RTX 3090"
    return "unknown"


def get_gpu_specs(sm_count: int | None, compute_capability: str) -> dict:
    """
    Return GPU hardware specs for roofline analysis.

    Detection order:
      1. torch.cuda.get_device_name() substring match  — most accurate,
         correctly distinguishes H200 from H100 (same CC and SM count).
      2. Compute capability + SM count lookup           — fallback when
         torch is unavailable or returns an unrecognised string.
      3. If still unknown, query torch.cuda.get_device_properties() for
         what is actually measurable (SM count, memory) and return the
         real device name so the LLM can use its own knowledge.
         No bandwidth or TFLOPS are fabricated from a wrong preset.
    """
    cuda_device_name: str | None = None
    cuda_props: dict = {}

    try:
        import torch
        if torch.cuda.is_available():
            cuda_device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            cuda_props = {
                "sm_count": props.multi_processor_count,
                "max_threads_per_sm": props.max_threads_per_multi_processor,
                "total_memory_gb": round(props.total_memory / 1024**3, 1),
            }
    except Exception:
        pass

    # Step 1: match device name against known patterns
    model = "unknown"
    if cuda_device_name:
        model = _detect_from_device_name(cuda_device_name)

    # Step 2: fallback to CC + SM count
    if model == "unknown":
        model = _detect_from_cc_and_sm(sm_count, compute_capability)

    # Step 3a: known GPU — return full preset specs
    if model in GPU_SPECS:
        return {
            "gpu_model": model,
            "gpu_model_detected": True,
            **GPU_SPECS[model],
            **cuda_props,   # override sm_count with measured value if available
        }

    # Step 3b: unknown GPU — return real device name + measured properties only.
    # Do NOT substitute another GPU's bandwidth/TFLOPS — those numbers would
    # silently corrupt the planner's roofline arithmetic.
    return {
        "gpu_model": cuda_device_name or "unknown",
        "gpu_model_detected": False,
        **cuda_props,
    }
