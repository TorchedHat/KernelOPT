# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8830 ns — **Best latency**: 8860 ns (**0.9966×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (10125,1,1), Waves/SM: 4.79
- **Registers/thread**: 16
- **Duration**: 8830 ns
- **SM Throughput**: 30.02 % | **Memory**: 20.19 % | **DRAM**: 12.03 %
- **Achieved Occupancy**: 45.77 %
- **IPC Active**: 1.64 inst/cycle | **No Eligible**: 58.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_latency_hiding | 0.972× | 0.431× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_eliminate_partial_wave_and_hide_l1tex_latency | 0.927× | 0.935× | ✓ |  |
| C | increase_xblock_with_autotune_for_vectorized_loads | 0.927× | 0.929× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 0.989× | 0.432× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8830 ns | 20480 ns | +11650 ns (+131.9%) | 9440 ns | +610 ns (+6.9%) | 9500 ns | +670 ns (+7.6%) | 20450 ns | +11620 ns (+131.6%) |
| SM Throughput (%) | 30.02 % | 70.97 % | +40.95 pp | 39.49 % | +9.47 pp | 39.62 % | +9.60 pp | 70.23 % | +40.21 pp |
| Memory Throughput (%) | 20.19 % | 31.11 % | +10.92 pp | 28.10 % | +7.91 pp | 28.38 % | +8.19 pp | 30.80 % | +10.61 pp |
| DRAM Throughput (%) | 12.03 % | 5.17 % | -6.86 pp | 11.25 % | -0.78 pp | 11.18 % | -0.85 pp | 5.17 % | -6.86 pp |
| Achieved Occupancy (%) | 45.77 % | 78.16 % | +32.39 pp | 44.02 % | -1.75 pp | 43.95 % | -1.82 pp | 78.11 % | +32.34 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 4.79 | 19.18 | +14.39 | 4.79 | +0.00 | 4.79 | +0.00 | 19.18 | +14.39 |
| No Eligible (%) | 58.82 % | 21.39 % | -37.43 pp | 43.43 % | -15.39 pp | 43.40 % | -15.42 pp | 22.06 % | -36.76 pp |
| IPC Active | 1.64 inst/cycle | 3.14 inst/cycle | +1.50 | 2.24 inst/cycle | +0.60 | 2.25 inst/cycle | +0.61 | 3.13 inst/cycle | +1.49 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_occupancy` (0.432×)

Increasing XBLOCK from 128 to 512 via autotune improved occupancy (45.8%→78.1%) and compute throughput (30%→70%), but caused a 5.5x explosion in executed instructions (2.07M→11.4M) and doubled register pressure (16→32 registers/thread), resulting in 2.3x longer kernel duration. The autotune selected a larger config (XBLOCK=512) that did not reduce the grid size (still 10125 blocks, waves_per_sm jumped from 4.79 to 19.18), meaning the grid lambda was not updated to use meta['XBLOCK'] — the grid remained fixed at cdiv(xnumel, 128) while the kernel processed more elements per block, causing massive redundant work. When adding autotune to a pointwise kernel, the grid launch expression MUST be a lambda reading meta['XBLOCK']; a hardcoded grid with a larger XBLOCK causes each element to be processed multiple times.

> **Applicable when**: Pointwise Triton kernels with low achieved occupancy (<60%), high scheduler no-eligible rate (>50%), and low registers/thread (<24) suggesting headroom for larger tiles.

> **Avoid when**: When the grid launch expression is not updated to a lambda reading meta['XBLOCK']. A fixed grid with larger XBLOCK causes each output element to be computed multiple times, exploding instruction count and causing severe regression. Also avoid when registers/thread are already high (>24) as larger XBLOCK will cause register spilling.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_eliminate_redundant_work | 0.927× | 0.926× | ✓ |  |
| B | fix_autotune_grid_lambda_to_enable_larger_xblock_tiles | 0.927× | 0.941× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_redundant_work | 0.972× | 0.428× | ✓ |  |
| D | fix_autotune_grid_lambda_and_fuse_fp32_instructions | 1.000× | 0.430× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8830 ns | 9540 ns | +710 ns (+8.0%) | 9380 ns | +550 ns (+6.2%) | 20640 ns | +11810 ns (+133.7%) | 20510 ns | +11680 ns (+132.3%) |
| SM Throughput (%) | 30.02 % | 39.64 % | +9.62 pp | 39.52 % | +9.50 pp | 70.82 % | +40.80 pp | 70.90 % | +40.88 pp |
| Memory Throughput (%) | 20.19 % | 28.16 % | +7.97 pp | 28.14 % | +7.95 pp | 31.07 % | +10.88 pp | 31.09 % | +10.90 pp |
| DRAM Throughput (%) | 12.03 % | 11.12 % | -0.91 pp | 11.28 % | -0.75 pp | 5.13 % | -6.90 pp | 5.16 % | -6.87 pp |
| Achieved Occupancy (%) | 45.77 % | 44.36 % | -1.41 pp | 44.14 % | -1.63 pp | 78.08 % | +32.31 pp | 78.19 % | +32.42 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 4.79 | 4.79 | +0.00 | 4.79 | +0.00 | 19.18 | +14.39 | 19.18 | +14.39 |
| No Eligible (%) | 58.82 % | 43.37 % | -15.45 pp | 43.47 % | -15.35 pp | 22.29 % | -36.53 pp | 21.51 % | -37.31 pp |
| IPC Active | 1.64 inst/cycle | 2.26 inst/cycle | +0.62 | 2.26 inst/cycle | +0.62 | 3.13 inst/cycle | +1.49 | 3.13 inst/cycle | +1.49 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_autotune_xblock_tile_reduction | 0.942× | 0.910× | ✓ |  |
| B | fix_autotune_grid_lambda_to_match_xblock_and_eliminate_partial_wave | 0.941× | 0.938× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_redundant_work_and_reduce_instruction_count | 0.999× | 0.430× | ✓ |  |
| D | fix_autotune_grid_lambda_to_eliminate_redundant_work_and_fused_fp32 | 0.989× | 0.432× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8830 ns | 9700 ns | +870 ns (+9.9%) | 9410 ns | +580 ns (+6.6%) | 20510 ns | +11680 ns (+132.3%) | 20450 ns | +11620 ns (+131.6%) |
| SM Throughput (%) | 30.02 % | 39.54 % | +9.52 pp | 39.43 % | +9.41 pp | 70.47 % | +40.45 pp | 70.38 % | +40.36 pp |
| Memory Throughput (%) | 20.19 % | 28.36 % | +8.17 pp | 28.02 % | +7.83 pp | 30.76 % | +10.57 pp | 30.84 % | +10.65 pp |
| DRAM Throughput (%) | 12.03 % | 10.92 % | -1.11 pp | 11.25 % | -0.78 pp | 5.15 % | -6.88 pp | 5.17 % | -6.86 pp |
| Achieved Occupancy (%) | 45.77 % | 44.25 % | -1.52 pp | 43.57 % | -2.20 pp | 78.40 % | +32.63 pp | 78.03 % | +32.26 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 4.79 | 4.79 | +0.00 | 4.79 | +0.00 | 19.18 | +14.39 | 19.18 | +14.39 |
| No Eligible (%) | 58.82 % | 43.44 % | -15.38 pp | 44.43 % | -14.39 pp | 21.39 % | -37.43 pp | 21.82 % | -37.00 pp |
| IPC Active | 1.64 inst/cycle | 2.24 inst/cycle | +0.60 | 2.23 inst/cycle | +0.59 | 3.13 inst/cycle | +1.49 | 3.12 inst/cycle | +1.48 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_enable_autotune_and_fuse_fp32_fma | 0.936× | 0.935× | ✓ |  |
| B | fix_grid_lambda_to_use_meta_xblock_enabling_correct_autotune | 0.937× | 0.932× | ✓ |  |
| C | remove_broken_autotune_and_use_correct_fixed_xblock_launch | 0.995× | 1.010× | ✓ |  |
| D | fix_grid_lambda_to_match_autotune_xblock_and_apply_fma_fusion | 1.003× | 0.432× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_t... | Δ | remove_broken_aut... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8830 ns | 9440 ns | +610 ns (+6.9%) | 9470 ns | +640 ns (+7.2%) | 8740 ns | -90 ns (-1.0%) | 20450 ns | +11620 ns (+131.6%) |
| SM Throughput (%) | 30.02 % | 39.17 % | +9.15 pp | 39.46 % | +9.44 pp | 29.97 % | -0.05 pp | 70.47 % | +40.45 pp |
| Memory Throughput (%) | 20.19 % | 27.82 % | +7.63 pp | 28.09 % | +7.90 pp | 20.13 % | -0.06 pp | 30.95 % | +10.76 pp |
| DRAM Throughput (%) | 12.03 % | 11.22 % | -0.81 pp | 11.22 % | -0.81 pp | 12.14 % | +0.11 pp | 5.17 % | -6.86 pp |
| Achieved Occupancy (%) | 45.77 % | 45.90 % | +0.13 pp | 44.02 % | -1.75 pp | 46.90 % | +1.13 pp | 78.19 % | +32.42 pp |
| Registers/Thread | 16.00 register/thread | 32.00 register/thread | +16 | 32.00 register/thread | +16 | 16.00 register/thread | +0 | 32.00 register/thread | +16 |
| Waves/SM | 4.79 | 4.79 | +0.00 | 4.79 | +0.00 | 4.79 | +0.00 | 19.18 | +14.39 |
| No Eligible (%) | 58.82 % | 44.32 % | -14.50 pp | 45.22 % | -13.60 pp | 59.51 % | +0.69 pp | 21.39 % | -37.43 pp |
| IPC Active | 1.64 inst/cycle | 2.18 inst/cycle | +0.54 | 2.24 inst/cycle | +0.60 | 1.61 inst/cycle | -0.03 | 3.14 inst/cycle | +1.50 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_1024_with_matching_fixed_grid_and_more_warps | 1.004× | 0.997× | ✓ | **✓ best** |
| B | remove_broken_autotune_use_fixed_xblock1024_with_correct_grid | 0.995× | 0.997× | ✓ |  |
| C | fuse_fp32_multiply_add_to_fma_with_corrected_autotune_grid_lambda | 0.945× | 0.938× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_enabling_correct_autotune_tile_dispatch | 1.003× | 0.431× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | remove_broken_aut... | Δ | fuse_fp32_multipl... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8830 ns | 8860 ns | +30 ns (+0.3%) | 8860 ns | +30 ns (+0.3%) | 9410 ns | +580 ns (+6.6%) | 20480 ns | +11650 ns (+131.9%) |
| SM Throughput (%) | 30.02 % | 29.97 % | -0.05 pp | 29.53 % | -0.49 pp | 39.55 % | +9.53 pp | 71.09 % | +41.07 pp |
| Memory Throughput (%) | 20.19 % | 20.13 % | -0.06 pp | 19.85 % | -0.34 pp | 28.09 % | +7.90 pp | 31.17 % | +10.98 pp |
| DRAM Throughput (%) | 12.03 % | 11.93 % | -0.10 pp | 11.94 % | -0.09 pp | 11.24 % | -0.79 pp | 5.16 % | -6.87 pp |
| Achieved Occupancy (%) | 45.77 % | 45.91 % | +0.14 pp | 46.07 % | +0.30 pp | 44.14 % | -1.63 pp | 77.97 % | +32.20 pp |
| Registers/Thread | 16.00 register/thread | 16.00 register/thread | +0 | 16.00 register/thread | +0 | 32.00 register/thread | +16 | 32.00 register/thread | +16 |
| Waves/SM | 4.79 | 4.79 | +0.00 | 4.79 | +0.00 | 4.79 | +0.00 | 19.18 | +14.39 |
| No Eligible (%) | 58.82 % | 59.16 % | +0.34 pp | 59.10 % | +0.28 pp | 45.12 % | -13.70 pp | 22.18 % | -36.64 pp |
| IPC Active | 1.64 inst/cycle | 1.66 inst/cycle | +0.02 | 1.64 inst/cycle | +0.00 | 2.26 inst/cycle | +0.62 | 3.08 inst/cycle | +1.44 |

---

## Cumulative Progress

```
Baseline:  8830 ns  (1.000×)

Final best: 8860 ns  (0.9966×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
@@ -20,10 +21,10 @@

     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp_inv_std = tl.rsqrt(tmp3 + 1e-5)
     tmp_scale = tmp12 * tmp_inv_std
-    tmp_normed = tmp_scale * (tmp0 - tmp1) + tmp14
+    tmp_shifted = tmp0 - tmp1
+    tmp_normed = libdevice.fma(tmp_scale, tmp_shifted, tmp14)
     tmp_clamped = tl.minimum(tl.maximum(tmp_normed, 0.0), 6.0)
     tl.store(in_out_ptr0 + (x2), tmp_clamped, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
