# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_45` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4610 ns — **Best latency**: 4610 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_45`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_45_triton_poi_fused_convolution_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3136,1,1), Waves/SM: 1.48
- **Registers/thread**: 28
- **Duration**: 4610 ns
- **SM Throughput**: 36.61 % | **Memory**: 32.28 % | **DRAM**: 4.05 %
- **Achieved Occupancy**: 68.96 %
- **IPC Active**: 2.68 inst/cycle | **No Eligible**: 30.09 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_sizes_to_eliminate_partial_wave | 0.987× | 1.043× | ✓ |  |
| B | fix_tile_sizes_for_small_xdim_permutation_kernel | 0.986× | 1.060× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_to... | Δ | fix_tile_sizes_fo... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4610 ns | 4420 ns | -190 ns (-4.1%) | 4350 ns | -260 ns (-5.6%) |
| SM Throughput (%) | 36.61 % | 30.21 % | -6.40 pp | 15.36 % | -21.25 pp |
| Memory Throughput (%) | 32.28 % | 15.79 % | -16.49 pp | 11.34 % | -20.94 pp |
| DRAM Throughput (%) | 4.05 % | 4.22 % | +0.17 pp | 4.26 % | +0.21 pp |
| Achieved Occupancy (%) | 68.96 % | 65.73 % | -3.23 pp | 47.73 % | -21.23 pp |
| Registers/Thread | 28.00 register/thread | 21.00 register/thread | -7 | 22.00 register/thread | -6 |
| Waves/SM | 1.48 | 1.48 | +0.00 | 0.74 | -0.74 |
| No Eligible (%) | 30.09 % | 27.18 % | -2.91 pp | 42.99 % | +12.90 pp |
| IPC Active | 2.68 inst/cycle | 2.79 inst/cycle | +0.11 | 2.05 inst/cycle | -0.63 |

---

## Cumulative Progress

```
Baseline:  4610 ns  (1.000×)

Final best: 4610 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
