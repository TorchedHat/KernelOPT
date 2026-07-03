# Kernel Optimization Log

## Summary

- **Kernel**: `cumsum_kernel_v2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2940 ns — **Best latency**: 2940 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `cumsum_kernel_v2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_089/synthesis/kernel_0_triton_red_fused_cumsum_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4,1,1), Waves/SM: 0.00
- **Registers/thread**: 40
- **Duration**: 2940 ns
- **SM Throughput**: 0.21 % | **Memory**: 0.42 % | **DRAM**: 0.05 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.53 inst/cycle | **No Eligible**: 85.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | vectorized_loads_stores_with_grid_expansion | 0.997× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2940 ns | 3070 ns | +130 ns (+4.4%) |
| SM Throughput (%) | 0.21 % | 0.20 % | -0.01 pp |
| Memory Throughput (%) | 0.42 % | 0.39 % | -0.03 pp |
| DRAM Throughput (%) | 0.05 % | 0.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 85.74 % | 88.30 % | +2.56 pp |
| IPC Active | 0.53 inst/cycle | 0.51 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  2940 ns  (1.000×)

Final best: 2940 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
