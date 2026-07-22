# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7460 ns — **Best latency**: 7460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_31_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 17
- **Duration**: 7460 ns
- **SM Throughput**: 39.06 % | **Memory**: 17.13 % | **DRAM**: 11.03 %
- **Achieved Occupancy**: 57.99 %
- **IPC Active**: 2.28 inst/cycle | **No Eligible**: 42.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.930× | 0.772× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.929× | 0.775× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7460 ns | 9660 ns | +2200 ns (+29.5%) | 9630 ns | +2170 ns (+29.1%) |
| SM Throughput (%) | 39.06 % | 66.44 % | +27.38 pp | 66.37 % | +27.31 pp |
| Memory Throughput (%) | 17.13 % | 17.99 % | +0.86 pp | 17.98 % | +0.85 pp |
| DRAM Throughput (%) | 11.03 % | 8.52 % | -2.51 pp | 8.54 % | -2.49 pp |
| Achieved Occupancy (%) | 57.99 % | 81.40 % | +23.41 pp | 81.48 % | +23.49 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 42.63 % | 15.79 % | -26.84 pp | 15.83 % | -26.80 pp |
| IPC Active | 2.28 inst/cycle | 3.34 inst/cycle | +1.06 | 3.34 inst/cycle | +1.06 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.772×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) caused registers per thread to double from 17 to 32, and executed instructions to increase by 2.2x (2.29M → 5.03M), resulting in a 29% longer kernel duration (7.46 µs → 9.66 µs) despite improved occupancy (58% → 81%) and compute throughput (39% → 66%). The root cause is that the autotune decorator itself, combined with the introduction of a bounds-check mask (xmask = xindex < xnumel) on a kernel where xnumel was previously known to be exactly divisible by XBLOCK, forced the compiler to emit additional predicated instructions and increased register pressure. When the original kernel had only 17 registers/thread and used tl.full([XBLOCK], True) as a no-op mask, replacing it with a real bounds check adds per-element comparison and predication overhead that dominates at small workloads. Avoid adding real xmask bounds checks to kernels where the problem size is guaranteed divisible by all candidate tile sizes, as the extra instructions outweigh any tile-size benefit.

> **Applicable when**: Never — this specific combination of autotune + real mask introduction on a divisible-size kernel caused regression. The strategy of increasing XBLOCK via autotune is theoretically sound but the implementation introduced unnecessary masking overhead.

> **Avoid when**: Do not add real bounds-check masks (xindex < xnumel) when the problem size is guaranteed divisible by all candidate XBLOCK values. Do not add @triton.autotune when the resulting register pressure increase (e.g., 17→32 regs/thread) will cause instruction count to more than double, as the compute overhead exceeds the latency-hiding benefit. Avoid this pattern when the kernel is already short (<10 µs) and instruction count is the primary cost driver.

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.775×)

Adding @triton.autotune with larger XBLOCK configs (512, 1024) and more warps improved scheduler eligibility (No Eligible dropped from 42.6% to 15.8%) and raised achieved occupancy (58% to 81%), but the selected config doubled registers per thread from 17 to 32, which more than doubled total executed instructions (2.29M to 5.03M). The extra instruction count overwhelmed the latency-hiding benefit, increasing kernel duration from 7.46 µs to 9.63 µs (0.77x regression). The implementation also introduced a bug: the launch grid was left hardcoded at cdiv(xnumel, 128) instead of being updated to use the autotuned XBLOCK via a lambda, meaning the grid size did not match the tile size selected by autotune.

> **Applicable when**: Kernel has low achieved occupancy (<60%), high No Eligible scheduler cycles (>40%), and L1TEX scoreboard stalls dominate; registers per thread are low (<20) leaving headroom for larger blocks.

> **Avoid when**: The launch grid is not updated to a lambda that reads meta['XBLOCK'] — mismatched grid/tile causes incorrect execution and instruction bloat. Also avoid if the autotune search selects a config that doubles register usage, as the resulting instruction count increase can negate latency-hiding gains. Do not apply when the kernel is already compute-bound or when instruction count is the primary cost.

---

## Cumulative Progress

```
Baseline:  7460 ns  (1.000×)

Final best: 7460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
