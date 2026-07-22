# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 42980 ns — **Best latency**: 42980 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_addmm_mul_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_039/kernel_1_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 22
- **Duration**: 42980 ns
- **SM Throughput**: 61.73 % | **Memory**: 31.17 % | **DRAM**: 19.36 %
- **Achieved Occupancy**: 57.71 %
- **IPC Active**: 2.61 inst/cycle | **No Eligible**: 34.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse | 0.629× | 0.600× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.627× | 0.599× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 42980 ns | 71680 ns | +28700 ns (+66.8%) | 71710 ns | +28730 ns (+66.8%) |
| SM Throughput (%) | 61.73 % | 83.22 % | +21.49 pp | 83.24 % | +21.51 pp |
| Memory Throughput (%) | 31.17 % | 28.53 % | -2.64 pp | 28.53 % | -2.64 pp |
| DRAM Throughput (%) | 19.36 % | 11.66 % | -7.70 pp | 11.64 % | -7.72 pp |
| Achieved Occupancy (%) | 57.71 % | 88.97 % | +31.26 pp | 89.16 % | +31.45 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 34.44 % | 14.54 % | -19.90 pp | 14.30 % | -20.14 pp |
| IPC Active | 2.61 inst/cycle | 3.42 inst/cycle | +0.81 | 3.41 inst/cycle | +0.80 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_with_autotune_to_hide_l1tex_latency` (0.599×)

Increasing XBLOCK via autotune raised achieved occupancy from 57.7% to 89.2% and eligible warps per scheduler from 1.37 to 7.44, successfully hiding L1TEX stalls. However, the larger tile sizes caused register pressure to jump from 22 to 32 registers/thread, and the total executed instruction count grew 2.24x (20.97M → 46.92M), resulting in a 1.67x longer kernel duration (42.98 µs → 71.71 µs). The autotune overhead combined with the instruction count explosion from larger XBLOCK tiles produced a net 0.6x regression despite improved occupancy. Additionally, the grid was not updated to use a lambda, so the fixed grid of cdiv(xnumel, 128) was used regardless of the autotuned XBLOCK, causing incorrect tile coverage.

> **Applicable when**: Kernel has low achieved occupancy (<60%), high L1TEX stall fraction (>40% of CPI), and few eligible warps per scheduler (<2), suggesting the warp pool is too shallow to hide memory latency.

> **Avoid when**: The kernel already executes a large instruction count per thread; increasing XBLOCK will multiply instruction count proportionally and cause regression even if occupancy improves. Also avoid when the launch grid is not updated to a lambda — a fixed grid with larger XBLOCK causes incorrect execution. Avoid when register pressure increase from larger tiles reduces theoretical occupancy enough to negate the benefit.

---

## Cumulative Progress

```
Baseline:  42980 ns  (1.000×)

Final best: 42980 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
