# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7420 ns — **Best latency**: 7420 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_44`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_44_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 18
- **Duration**: 7420 ns
- **SM Throughput**: 50.08 % | **Memory**: 24.98 % | **DRAM**: 24.98 %
- **Achieved Occupancy**: 76.75 %
- **IPC Active**: 2.77 inst/cycle | **No Eligible**: 30.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_autotune_xblock | 0.672× | 0.575× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave_tail | 0.677× | 0.573× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7420 ns | 12900 ns | +5480 ns (+73.9%) | 12960 ns | +5540 ns (+74.7%) |
| SM Throughput (%) | 50.08 % | 65.69 % | +15.61 pp | 66.41 % | +16.33 pp |
| Memory Throughput (%) | 24.98 % | 18.42 % | -6.56 pp | 18.61 % | -6.37 pp |
| DRAM Throughput (%) | 24.98 % | 14.29 % | -10.69 pp | 14.26 % | -10.72 pp |
| Achieved Occupancy (%) | 76.75 % | 80.74 % | +3.99 pp | 80.60 % | +3.85 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 2.78 | +0.00 |
| No Eligible (%) | 30.59 % | 22.02 % | -8.57 pp | 22.12 % | -8.47 pp |
| IPC Active | 2.77 inst/cycle | 3.08 inst/cycle | +0.31 | 3.10 inst/cycle | +0.33 |


### Key Learnings

#### ❌ Plan A: `eliminate_partial_wave_via_autotune_xblock` (0.575×)

Adding a @triton.autotune decorator with larger XBLOCK configs (512, 1024, 2048) to a pointwise kernel that had a 33% partial-wave tail penalty caused a 74% regression (7.42us → 12.9us). The executed instruction count more than doubled (2.94M → 6.67M) and registers per thread jumped from 18 to 32, indicating the autotuner selected a config that dramatically increased per-thread register pressure and instruction count rather than reducing grid overhead. The partial-wave penalty (LaunchConfiguration rule, 33.3% estimated speedup) was real, but the autotune overhead and register spill cost from larger XBLOCK outweighed the benefit for this short-duration, low-register-count kernel. Do not apply autotune-based XBLOCK enlargement to kernels with very low register usage (≤18 registers/thread) and short duration (≤10us), as the autotuner may select configs that inflate register pressure and instruction count beyond the partial-wave savings.

> **Applicable when**: Kernel has a large partial-wave tail (LaunchConfiguration rule, >20% estimated speedup), high register headroom (registers_per_thread well below 32), and long enough duration that autotune overhead amortizes.

> **Avoid when**: Kernel already uses few registers (≤18/thread) and is short-duration (≤10us); autotune with large XBLOCK configs causes register pressure to jump (18→32) and instruction count to more than double, producing a net regression despite reducing partial-wave overhead. Also avoid when the kernel body has complex index arithmetic that scales super-linearly with XBLOCK.

---

## Cumulative Progress

```
Baseline:  7420 ns  (1.000×)

Final best: 7420 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
