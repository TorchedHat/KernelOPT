# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 918430 ns — **Best latency**: 918430 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_5_triton_poi_fused_convolution_r/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1492992,1,1), Waves/SM: 706.91
- **Registers/thread**: 16
- **Duration**: 918430 ns
- **SM Throughput**: 23.89 % | **Memory**: 33.28 % | **DRAM**: 33.28 %
- **Achieved Occupancy**: 41.23 %
- **IPC Active**: 0.96 inst/cycle | **No Eligible**: 76.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_larger_xblock | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 918430 ns | 918660 ns | +230 ns (+0.0%) |
| SM Throughput (%) | 23.89 % | 23.92 % | +0.03 pp |
| Memory Throughput (%) | 33.28 % | 33.27 % | -0.01 pp |
| DRAM Throughput (%) | 33.28 % | 33.27 % | -0.01 pp |
| Achieved Occupancy (%) | 41.23 % | 41.14 % | -0.09 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 |
| Waves/SM | 706.91 | 706.91 | +0.00 |
| No Eligible (%) | 76.13 % | 76.14 % | +0.01 pp |
| IPC Active | 0.96 inst/cycle | 0.95 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  918430 ns  (1.000×)

Final best: 918430 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
