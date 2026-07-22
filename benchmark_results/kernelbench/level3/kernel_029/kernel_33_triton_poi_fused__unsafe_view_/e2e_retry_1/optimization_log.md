# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6750 ns — **Best latency**: 6750 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__unsafe_view_clone_convolution_native_layer_norm_permute_transpose_view_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_33_triton_poi_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5880,1,1), Waves/SM: 2.78
- **Registers/thread**: 20
- **Duration**: 6750 ns
- **SM Throughput**: 45.10 % | **Memory**: 14.14 % | **DRAM**: 9.19 %
- **Achieved Occupancy**: 71.01 %
- **IPC Active**: 2.67 inst/cycle | **No Eligible**: 30.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_autotune_xblock | 0.977× | 1.038× | ✓ |  |
| B | increase_xblock_to_eliminate_partial_wave | 0.987× | 0.545× | ✓ |  |
| C | add_triton_autotune | 0.978× | 1.038× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | increase_xblock_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6750 ns | 6500 ns | -250 ns (-3.7%) | 12380 ns | +5630 ns (+83.4%) | 6500 ns | -250 ns (-3.7%) |
| SM Throughput (%) | 45.10 % | 53.89 % | +8.79 pp | 70.29 % | +25.19 pp | 47.75 % | +2.65 pp |
| Memory Throughput (%) | 14.14 % | 21.48 % | +7.34 pp | 27.72 % | +13.58 pp | 14.77 % | +0.63 pp |
| DRAM Throughput (%) | 9.19 % | 9.55 % | +0.36 pp | 4.99 % | -4.20 pp | 9.58 % | +0.39 pp |
| Achieved Occupancy (%) | 71.01 % | 71.32 % | +0.31 pp | 83.85 % | +12.84 pp | 71.45 % | +0.44 pp |
| Registers/Thread | 20.00 register/thread | 24.00 register/thread | +4 | 31.00 register/thread | +11 | 20.00 register/thread | +0 |
| Waves/SM | 2.78 | 2.78 | +0.00 | 5.57 | +2.79 | 2.78 | +0.00 |
| No Eligible (%) | 30.86 % | 21.19 % | -9.67 pp | 15.89 % | -14.97 pp | 31.29 % | +0.43 pp |
| IPC Active | 2.67 inst/cycle | 3.14 inst/cycle | +0.47 | 3.28 inst/cycle | +0.61 | 2.72 inst/cycle | +0.05 |


### Key Learnings

#### ❌ Plan B: `increase_xblock_to_eliminate_partial_wave` (0.545×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) to a pointwise kernel that had a large partial-wave tail (LaunchConfiguration rule, 33% estimated speedup) caused a severe regression (0.55x). The autotune selected XBLOCK=256 (not the intended 512), which kept the grid size at 5880 (unchanged) while increasing registers per thread from 20 to 31 — this halved the block limit from registers (21→8 blocks/SM), dramatically increasing waves_per_sm from 2.78 to 5.57 and nearly doubling executed instructions (2.4M→6.8M). The NCU LaunchConfiguration rule's estimated speedup from eliminating the partial wave does not account for the register pressure increase that results from larger tile sizes, which can negate or reverse the benefit by reducing occupancy and increasing total work.

> **Applicable when**: Never: this specific pattern (adding autotune to increase XBLOCK for partial-wave elimination) caused regression when register count per thread increased significantly with larger tiles.

> **Avoid when**: Do not add autotune with larger XBLOCK configs when the kernel already uses close to 20+ registers per thread at the baseline block size. Larger XBLOCK increases register pressure, which reduces block-limit-registers and can increase waves_per_sm rather than decrease it, negating the partial-wave benefit. Also avoid when the grid size does not actually decrease (autotune may select a config that keeps grid size the same but increases register pressure). The LaunchConfiguration NCU rule's estimated speedup assumes the grid shrinks proportionally, but autotune may select a suboptimal config.

---

## Cumulative Progress

```
Baseline:  6750 ns  (1.000×)

Final best: 6750 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
