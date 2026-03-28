# GPU Hardware Specs for Roofline Analysis

Use these specs when computing peak throughput % in the roofline model.

## NVIDIA H100 SXM5
| Metric | Value |
|---|---|
| Peak HBM bandwidth | 3,350 GB/s |
| Peak FP32 TFLOPS (non-tensor) | 67 TFLOPS |
| Peak FP32 TFLOPS (tensor, TF32) | 989 TFLOPS |
| Peak FP16 TFLOPS (tensor) | 1,979 TFLOPS |
| Peak BF16 TFLOPS (tensor) | 1,979 TFLOPS |
| L2 cache | 50 MB |
| SM count | 132 |
| Max warps per SM | 64 |

## NVIDIA H100 PCIe
| Metric | Value |
|---|---|
| Peak HBM bandwidth | 2,000 GB/s |
| Peak FP32 TFLOPS (tensor, TF32) | 756 TFLOPS |
| Peak FP16 TFLOPS (tensor) | 1,513 TFLOPS |
| SM count | 114 |

## NVIDIA A100 SXM4 (80GB)
| Metric | Value |
|---|---|
| Peak HBM bandwidth | 2,000 GB/s |
| Peak FP32 TFLOPS (non-tensor) | 19.5 TFLOPS |
| Peak FP32 TFLOPS (tensor, TF32) | 312 TFLOPS |
| Peak FP16 TFLOPS (tensor) | 624 TFLOPS |
| L2 cache | 40 MB |
| SM count | 108 |
| Max warps per SM | 64 |

## NVIDIA RTX 4090
| Metric | Value |
|---|---|
| Peak memory bandwidth | 1,008 GB/s |
| Peak FP32 TFLOPS (non-tensor) | 82.6 TFLOPS |
| Peak FP16 TFLOPS (tensor) | 330 TFLOPS |
| L2 cache | 72 MB |
| SM count | 128 |

## NVIDIA RTX 3090
| Metric | Value |
|---|---|
| Peak memory bandwidth | 936 GB/s |
| Peak FP32 TFLOPS | 35.6 TFLOPS |
| SM count | 82 |

## Roofline formula

```
# Determine which bound applies
arithmetic_intensity = flops / (hbm_read_bytes + hbm_write_bytes)
ridge_point = peak_flops / peak_bandwidth  # in FLOPs/byte

if arithmetic_intensity < ridge_point:
    bound = "memory"
    peak_perf = arithmetic_intensity * peak_bandwidth  # achievable FLOPS
else:
    bound = "compute"
    peak_perf = peak_flops

# Convert to time-based % of peak
ideal_time_s = total_flops / peak_perf   # or total_bytes / peak_bandwidth
actual_time_s = measured_latency_ms / 1000
peak_throughput_pct = (ideal_time_s / actual_time_s) * 100
```

## If GPU model is unknown
Ask the user which GPU they are running on. If unavailable, default to A100 SXM4 and note the assumption in the summary.
