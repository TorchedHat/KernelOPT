# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3460 ns — **Best latency**: 3460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_43`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_43_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 0.58
- **Registers/thread**: 18
- **Duration**: 3460 ns
- **SM Throughput**: 15.29 % | **Memory**: 5.65 % | **DRAM**: 3.77 %
- **Achieved Occupancy**: 44.30 %
- **IPC Active**: 1.52 inst/cycle | **No Eligible**: 56.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size_for_underutilized_pointwise_kernel | 1.000× | 0.554× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 0.963× | 1.039× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 6240 ns | +2780 ns (+80.3%) | 3330 ns | -130 ns (-3.8%) |
| SM Throughput (%) | 15.29 % | 55.94 % | +40.65 pp | 14.72 % | -0.57 pp |
| Memory Throughput (%) | 5.65 % | 11.21 % | +5.56 pp | 5.96 % | +0.31 pp |
| DRAM Throughput (%) | 3.77 % | 2.10 % | -1.67 pp | 3.93 % | +0.16 pp |
| Achieved Occupancy (%) | 44.30 % | 79.60 % | +35.30 pp | 49.58 % | +5.28 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 0.58 | 2.32 | +1.74 | 0.58 | +0.00 |
| No Eligible (%) | 56.58 % | 18.53 % | -38.05 pp | 58.34 % | +1.76 pp |
| IPC Active | 1.52 inst/cycle | 3.17 inst/cycle | +1.65 | 1.36 inst/cycle | -0.16 |


### Key Learnings

#### ❌ Plan A: `autotune_xblock_increase_tile_size_for_underutilized_pointwise_kernel` (0.554×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) to a small pointwise kernel caused a 1.8x slowdown despite improving occupancy (44.3%→79.6%) and compute throughput (15.3%→55.9%). The autotune selected XBLOCK=512 with 32 registers/thread (up from 18), which reduced Block Limit Registers from 21 to 4 blocks/SM, capping actual parallelism. Critically, the grid remained fixed at 1225 (the lambda was not applied — the diff shows the old fixed grid was kept), so the kernel executed 2.32 waves instead of fewer larger waves, multiplying total instruction count by ~6.8x (411K→2.8M instructions) with no proportional throughput gain.

> **Applicable when**: Never apply this exact pattern (autotune without updating grid to lambda). The general direction of increasing XBLOCK may still be valid if the grid lambda is correctly wired.

> **Avoid when**: When adding @triton.autotune with variable XBLOCK configs, if the launch grid is not simultaneously changed to a lambda that reads meta['XBLOCK'], the grid size stays fixed at the old small-block count. This causes the kernel to run many more (smaller) CTAs than intended, multiplying total instruction count and increasing duration. Also avoid large XBLOCK configs that push registers/thread high enough to reduce Block Limit Registers below 4, as this caps SM occupancy regardless of warp count.

---

## Cumulative Progress

```
Baseline:  3460 ns  (1.000×)

Final best: 3460 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
