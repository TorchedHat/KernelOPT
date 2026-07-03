# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6140 ns — **Best latency**: 6140 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_069/kernel_0_triton_poi_fused_convolution_0/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 26
- **Duration**: 6140 ns
- **SM Throughput**: 16.00 % | **Memory**: 41.25 % | **DRAM**: 20.14 %
- **Achieved Occupancy**: 56.04 %
- **IPC Active**: 1.10 inst/cycle | **No Eligible**: 71.69 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_alignment_hints | 0.895× | 1.037× | ✓ |  |
| B | autotune_tile_sizes_and_fix_uncoalesced_stores | 0.990× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 6140 ns | 5920 ns | -220 ns (-3.6%) | 5950 ns | -190 ns (-3.1%) |
| SM Throughput (%) | 16.00 % | 16.67 % | +0.67 pp | 16.46 % | +0.46 pp |
| Memory Throughput (%) | 41.25 % | 42.33 % | +1.08 pp | 42.11 % | +0.86 pp |
| DRAM Throughput (%) | 20.14 % | 20.82 % | +0.68 pp | 20.77 % | +0.63 pp |
| Achieved Occupancy (%) | 56.04 % | 54.92 % | -1.12 pp | 55.95 % | -0.09 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 71.69 % | 72.22 % | +0.53 pp | 70.50 % | -1.19 pp |
| IPC Active | 1.10 inst/cycle | 1.14 inst/cycle | +0.04 | 1.11 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  6140 ns  (1.000×)

Final best: 6140 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
