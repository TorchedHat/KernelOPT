# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3200 ns — **Best latency**: 4030 ns (**0.7940×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (768,1,1), Waves/SM: 0.36
- **Registers/thread**: 28
- **Duration**: 3200 ns
- **SM Throughput**: 13.20 % | **Memory**: 12.18 % | **DRAM**: 1.46 %
- **Achieved Occupancy**: 30.22 %
- **IPC Active**: 1.44 inst/cycle | **No Eligible**: 64.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_tile_sizes_and_add_autotune_for_transpose_kernel | — | 0.794× | ✓ | **✓ best** |
| C | increase_yblock_reduce_xblock_for_better_occupancy | — | 1.135× | ✓ |  |
| D | fix_tile_sizes_and_autotune_for_underutilized_transpose_kernel | — | 1.088× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_sizes_an... | Δ | increase_yblock_r... | Δ | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 4030 ns | +830 ns (+25.9%) | 2820 ns | -380 ns (-11.9%) | 2940 ns | -260 ns (-8.1%) |
| SM Throughput (%) | 13.20 % | 5.35 % | -7.85 pp | 19.42 % | +6.22 pp | 5.11 % | -8.09 pp |
| Memory Throughput (%) | 12.18 % | 5.41 % | -6.77 pp | 12.84 % | +0.66 pp | 10.46 % | -1.72 pp |
| DRAM Throughput (%) | 1.46 % | 3.33 % | +1.87 pp | 1.66 % | +0.20 pp | 6.28 % | +4.82 pp |
| Achieved Occupancy (%) | 30.22 % | 14.75 % | -15.47 pp | 28.72 % | -1.50 pp | 15.52 % | -14.70 pp |
| Registers/Thread | 28.00 register/thread | 21.00 register/thread | -7 | 28.00 register/thread | +0 | 30.00 register/thread | +2 |
| Waves/SM | 0.36 | 0.18 | -0.18 | 0.36 | +0.00 | 0.18 | -0.18 |
| No Eligible (%) | 64.61 % | 81.29 % | +16.68 pp | 44.31 % | -20.30 pp | 76.14 % | +11.53 pp |
| IPC Active | 1.44 inst/cycle | 0.62 inst/cycle | -0.82 | 2.09 inst/cycle | +0.65 | 0.87 inst/cycle | -0.57 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_underutilized_transpose_kernel | — | 0.006× | ✓ |  |
| B | fix_adaptive_grid_lambda_for_autotuned_transpose_kernel | — | 1.088× | ✓ |  |
| C | persistent_tile_scheduling_for_underutilized_transpose_kernel | — | 0.006× | ✓ |  |
| D | add_triton_autotune | — | 0.982× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_adaptive_grid... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 541340 ns | +538140 ns (+16816.9%) | 2940 ns | -260 ns (-8.1%) | 530140 ns | +526940 ns (+16466.9%) | 3260 ns | +60 ns (+1.9%) |
| SM Throughput (%) | 13.20 % | 64.43 % | +51.23 pp | 8.27 % | -4.93 pp | 66.26 % | +53.06 pp | 28.36 % | +15.16 pp |
| Memory Throughput (%) | 12.18 % | 63.47 % | +51.29 pp | 10.36 % | -1.82 pp | 65.26 % | +53.08 pp | 23.51 % | +11.33 pp |
| DRAM Throughput (%) | 1.46 % | 0.01 % | -1.45 pp | 6.21 % | +4.75 pp | 0.18 % | -1.28 pp | 1.43 % | -0.03 pp |
| Achieved Occupancy (%) | 30.22 % | 36.33 % | +6.11 pp | 15.39 % | -14.83 pp | 36.32 % | +6.10 pp | 29.17 % | -1.05 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 0.36 | 0.36 | +0.00 | 0.18 | -0.18 | 0.36 | +0.00 | 0.36 | +0.00 |
| No Eligible (%) | 64.61 % | 35.09 % | -29.52 pp | 79.69 % | +15.08 pp | 32.65 % | -31.96 pp | 43.39 % | -21.22 pp |
| IPC Active | 1.44 inst/cycle | 2.62 inst/cycle | +1.18 | 0.88 inst/cycle | -0.56 | 2.71 inst/cycle | +1.27 | 2.50 inst/cycle | +1.06 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_for_underutilized_transpose_kernel` (0.006×)

The kernel body was correctly rewritten to use a persistent while-loop with `tl.num_programs(0)` as the tile-iteration stride, but the launch site grid was NOT updated — it remained `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 768)`, a 2D grid. With `tl.num_programs(0) = 1` (only 1 CTA in dim-0 of the 2D grid), each of the 768 CTAs iterated over all 96 tiles with stride=1, executing 96×768=73,728 tile iterations instead of 96, inflating executed instructions from 183K to 278M and duration from 2.94 µs to 549 µs (~187x regression). The critical rule is: any time the kernel body uses `tl.num_programs(0)` as a stride, the grid MUST simultaneously be changed to a 1D tuple `(min(sm_count, num_tiles),)` — failing to do so multiplies total work by the old grid's CTA count.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel concept is valid when waves_per_sm < 0.5 and num_tiles <= sm_count, but the grid launch MUST be updated to 1D simultaneously with the kernel body rewrite.

> **Avoid when**: Any time the kernel body uses tl.num_programs(0) as a tile-iteration stride but the grid launch remains a 2D expression from the original kernel. This causes O(old_grid_cta_count × num_tiles) work instead of O(num_tiles), producing catastrophic regressions (100x–1000x slowdown). Also do not apply persistent kernels when num_tiles >> sm_count.

#### ❌ Plan C: `persistent_tile_scheduling_for_underutilized_transpose_kernel` (0.006×)

The kernel body was correctly rewritten to use a persistent while-loop with `tl.num_programs(0)` as the tile-iteration stride, but the launch site grid was NOT updated — it remained `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 768)`, a 2D grid. With `tl.num_programs(0) = 1` (only 1 CTA in dim-0 of the 2D grid), each of the 768 CTAs iterated over all 96 tiles with stride=1, executing 96×768=73,728 tile iterations instead of 96, inflating executed instructions from 183K to 278M and duration from 2.94 µs to 549 µs (~187x regression). The critical rule is: any time the kernel body uses `tl.num_programs(0)` as a stride, the grid MUST simultaneously be changed to a 1D tuple `(min(sm_count, num_tiles),)` — failing to do so multiplies total work by the old grid's CTA count.

> **Applicable when**: Never apply this specific broken pattern. The persistent kernel concept is valid when waves_per_sm < 0.5 and num_tiles <= sm_count, but the grid launch MUST be updated to 1D simultaneously with the kernel body rewrite.

> **Avoid when**: Any time the kernel body uses tl.num_programs(0) as a tile-iteration stride but the grid launch remains a 2D expression from the original kernel. This causes O(old_grid_cta_count × num_tiles) work instead of O(num_tiles), producing catastrophic regressions (100x–1000x slowdown). Also do not apply persistent kernels when num_tiles >> sm_count.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_to_use_autotuned_meta_tile_sizes | — | 1.088× | ✓ |  |
| B | fix_grid_lambda_to_match_existing_autotune_configs | — | 0.026× | ✓ |  |
| C | fix_broken_persistent_loop_revert_to_static_2d_grid_with_correct_tile_sizes | — | 1.053× | ✓ |  |
| D | fix_broken_persistent_grid_and_right_size_tile_shape | — | 0.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_lambda_t... | Δ | fix_broken_persis... | Δ | fix_broken_persis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3200 ns | 2940 ns | -260 ns (-8.1%) | 123140 ns | +119940 ns (+3748.1%) | 3040 ns | -160 ns (-5.0%) | 549380 ns | +546180 ns (+17068.1%) |
| SM Throughput (%) | 13.20 % | 8.02 % | -5.18 pp | 35.81 % | +22.61 pp | 14.32 % | +1.12 pp | 63.97 % | +50.77 pp |
| Memory Throughput (%) | 12.18 % | 10.40 % | -1.78 pp | 39.45 % | +27.27 pp | 12.73 % | +0.55 pp | 63.01 % | +50.83 pp |
| DRAM Throughput (%) | 1.46 % | 6.21 % | +4.75 pp | 0.05 % | -1.41 pp | 1.53 % | +0.07 pp | 0.01 % | -1.45 pp |
| Achieved Occupancy (%) | 30.22 % | 15.23 % | -14.99 pp | 35.86 % | +5.64 pp | 28.91 % | -1.31 pp | 36.31 % | +6.09 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 72.00 register/thread | +44 | 30.00 register/thread | +2 | 32.00 register/thread | +4 |
| Waves/SM | 0.36 | 0.18 | -0.18 | 0.83 | +0.47 | 0.36 | +0.00 | 0.36 | +0.00 |
| No Eligible (%) | 64.61 % | 75.33 % | +10.72 pp | 59.49 % | -5.12 pp | 63.40 % | -1.21 pp | 35.59 % | -29.02 pp |
| IPC Active | 1.44 inst/cycle | 0.86 inst/cycle | -0.58 | 1.67 inst/cycle | +0.23 | 1.32 inst/cycle | -0.12 | 2.60 inst/cycle | +1.16 |

---

## Cumulative Progress

```
Baseline:  3200 ns  (1.000×)

Final best: 4030 ns  (0.7940×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,26 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
+        triton.Config({'YBLOCK': 64,  'XBLOCK': 8},  num_warps=4),
+        triton.Config({'YBLOCK': 128, 'XBLOCK': 8},  num_warps=8),
+        triton.Config({'YBLOCK': 32,  'XBLOCK': 8},  num_warps=2),
+        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 6144
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -22,15 +35,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 192*x2 + 1728*y1), tmp0, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 6144
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11[grid](arg_0, arg_1, 6144, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11[grid](arg_0, arg_1, 6144, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
