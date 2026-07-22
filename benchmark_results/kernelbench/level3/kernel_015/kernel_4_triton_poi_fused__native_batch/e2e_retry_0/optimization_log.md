# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41060 ns — **Best latency**: 41060 ns (**1.0000×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62720,1,1), Waves/SM: 29.70
- **Registers/thread**: 17
- **Duration**: 41060 ns
- **SM Throughput**: 56.79 % | **Memory**: 24.67 % | **DRAM**: 18.51 %
- **Achieved Occupancy**: 55.01 %
- **IPC Active**: 2.40 inst/cycle | **No Eligible**: 39.70 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_occupancy_and_cache_reuse | 0.861× | 0.688× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.996× | 0.283× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41060 ns | 59680 ns | +18620 ns (+45.3%) | 145150 ns | +104090 ns (+253.5%) |
| SM Throughput (%) | 56.79 % | 86.01 % | +29.22 pp | 86.93 % | +30.14 pp |
| Memory Throughput (%) | 24.67 % | 25.29 % | +0.62 pp | 24.95 % | +0.28 pp |
| DRAM Throughput (%) | 18.51 % | 12.79 % | -5.72 pp | 5.26 % | -13.25 pp |
| Achieved Occupancy (%) | 55.01 % | 88.02 % | +33.01 pp | 82.75 % | +27.74 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 24.00 register/thread | +7 |
| Waves/SM | 29.70 | 29.70 | +0.00 | 118.79 | +89.09 |
| No Eligible (%) | 39.70 % | 10.96 % | -28.74 pp | 11.93 % | -27.77 pp |
| IPC Active | 2.40 inst/cycle | 3.56 inst/cycle | +1.16 | 3.52 inst/cycle | +1.12 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_for_occupancy_and_cache_reuse` (0.688×)

Adding @triton.autotune with larger XBLOCK configs (512/1024/2048) caused the compiler to allocate significantly more registers per thread (17 → 32), which reduced Block_Limit_Registers from 21 to 16 and increased total executed instructions by 2.2x (18M → 40M). Despite occupancy improving from 55% to 88%, the kernel duration increased by 45% (41us → 60us) because the instruction count explosion dominated. The autotune decorator itself, or the larger tile sizes it selected, triggered a different register allocation path that inflated compute work far beyond the memory latency savings.

> **Applicable when**: Never — this is a negative result. Do not apply autotune with large XBLOCK configs when the kernel currently uses low register counts (≤20 regs/thread) and the xnumel is evenly divisible by the current XBLOCK (making masking unnecessary).

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs to kernels where: (1) current register usage is already low (≤20 regs/thread) and headroom appears available but the compiler may allocate more registers for larger tiles; (2) xnumel is exactly divisible by current XBLOCK so xmask=True and adding conditional masking adds overhead; (3) the call-site grid is not updated to match the autotuned XBLOCK (grid remained cdiv(xnumel,128) while autotune selected larger blocks, causing incorrect tiling). Also avoid when instruction count is already low — larger tiles can cause 2x+ instruction inflation.

#### ❌ Plan B: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.283×)

Adding @triton.autotune to explore larger XBLOCK values caused a severe regression (0.28x) because the launch-site grid was hardcoded to `triton.cdiv(xnumel, 128)` instead of being updated to a lambda `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`. As a result, the autotuner selected XBLOCK=512 but the grid remained sized for XBLOCK=128, causing each thread to process 4x more work than intended and inflating executed instructions from 18M to 99M (5.4x increase). The duration increased from 41 µs to 145 µs despite occupancy improving from 55% to 83%, because the instruction count explosion dominated. When using @triton.autotune with variable XBLOCK, the grid must always be a lambda that reads `meta['XBLOCK']`; a static grid sized for the original XBLOCK is an implementation flaw that causes catastrophic instruction count inflation.

> **Applicable when**: Never apply this broken pattern. The correct pattern (lambda grid) is applicable whenever @triton.autotune is used with variable tile/block sizes.

> **Avoid when**: Never use a static grid expression when @triton.autotune varies XBLOCK or any tile dimension that determines the grid size. This causes instruction count to scale as (new_XBLOCK / old_XBLOCK) x original_work, producing severe regressions proportional to the XBLOCK ratio.

---

## Cumulative Progress

```
Baseline:  41060 ns  (1.000×)

Final best: 41060 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

_No textual differences between baseline and best kernel (or baseline source was unavailable)._ 

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
