# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285120 ns — **Best latency**: 285120 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_001/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285120 ns
- **SM Throughput**: 19.72 % | **Memory**: 79.06 % | **DRAM**: 72.39 %
- **Achieved Occupancy**: 94.03 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 80.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_yblock128_contiguous_hints_correct_indexing | 0.999× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285120 ns | 286720 ns | +1600 ns (+0.6%) |
| SM Throughput (%) | 19.72 % | 34.32 % | +14.60 pp |
| Memory Throughput (%) | 79.06 % | 78.58 % | -0.48 pp |
| DRAM Throughput (%) | 72.39 % | 71.98 % | -0.41 pp |
| Achieved Occupancy (%) | 94.03 % | 94.65 % | +0.62 pp |
| Registers/Thread | 25.00 register/thread | 26.00 register/thread | +1 |
| Waves/SM | 59.88 | 59.88 | +0.00 |
| No Eligible (%) | 80.21 % | 65.35 % | -14.86 pp |
| IPC Active | 0.79 inst/cycle | 1.39 inst/cycle | +0.60 |

---

## Cumulative Progress

```
Baseline:  285120 ns  (1.000×)

Final best: 285120 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
