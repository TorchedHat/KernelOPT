# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3580 ns — **Best latency**: 3010 ns (**1.1894×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`flatten_2d_grid_to_1d_with_correct_autotune_lambda`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_26_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1280,1,1), Waves/SM: 0.61
- **Registers/thread**: 28
- **Duration**: 3580 ns
- **SM Throughput**: 20.11 % | **Memory**: 18.59 % | **DRAM**: 2.15 %
- **Achieved Occupancy**: 48.56 %
- **IPC Active**: 1.93 inst/cycle | **No Eligible**: 49.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 1.031× | 1.189× | ✓ |  |
| D | fix_tile_sizes_and_add_autotune_for_transpose_permutation | 1.069× | 1.178× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_tile_sizes_an... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 3010 ns | -570 ns (-15.9%) | 3040 ns | -540 ns (-15.1%) |
| SM Throughput (%) | 20.11 % | 13.25 % | -6.86 pp | 18.91 % | -1.20 pp |
| Memory Throughput (%) | 18.59 % | 12.10 % | -6.49 pp | 9.27 % | -9.32 pp |
| DRAM Throughput (%) | 2.15 % | 2.57 % | +0.42 pp | 2.54 % | +0.39 pp |
| Achieved Occupancy (%) | 48.56 % | 21.54 % | -27.02 pp | 45.22 % | -3.34 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 22.00 register/thread | -6 |
| Waves/SM | 0.61 | 0.30 | -0.31 | 0.61 | +0.00 |
| No Eligible (%) | 49.08 % | 55.34 % | +6.26 pp | 44.45 % | -4.63 pp |
| IPC Active | 1.93 inst/cycle | 1.60 inst/cycle | -0.33 | 2.06 inst/cycle | +0.13 |

---

## Iteration 2

**Best so far**: 1.1776× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_for_higher_cta_count_and_warp_coverage | 1.074× | 1.218× | ✓ | **✓ best** |
| B | fix_grid_launch_to_respect_autotune_tile_sizes | 1.049× | 1.218× | ✓ |  |
| C | fix_grid_lambda_to_use_autotune_tile_sizes | 1.071× | 1.155× | ✓ |  |
| D | fix_hardcoded_grid_to_use_autotune_meta_xblock_yblock | 1.047× | 1.178× | ✓ |  |
| E | add_triton_autotune | 1.072× | 1.218× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | fix_grid_launch_t... | Δ | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2940 ns | -640 ns (-17.9%) | 2940 ns | -640 ns (-17.9%) | 3100 ns | -480 ns (-13.4%) | 3040 ns | -540 ns (-15.1%) | 2940 ns | -640 ns (-17.9%) |
| SM Throughput (%) | 20.11 % | 13.27 % | -6.84 pp | 13.04 % | -7.07 pp | 13.25 % | -6.86 pp | 18.31 % | -1.80 pp | 13.00 % | -7.11 pp |
| Memory Throughput (%) | 18.59 % | 12.04 % | -6.55 pp | 11.83 % | -6.76 pp | 12.12 % | -6.47 pp | 8.95 % | -9.64 pp | 11.83 % | -6.76 pp |
| DRAM Throughput (%) | 2.15 % | 2.60 % | +0.45 pp | 2.59 % | +0.44 pp | 2.48 % | +0.33 pp | 2.53 % | +0.38 pp | 2.60 % | +0.45 pp |
| Achieved Occupancy (%) | 48.56 % | 22.86 % | -25.70 pp | 21.28 % | -27.28 pp | 21.93 % | -26.63 pp | 45.34 % | -3.22 pp | 22.65 % | -25.91 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 22.00 register/thread | -6 | 30.00 register/thread | +2 |
| Waves/SM | 0.61 | 0.30 | -0.31 | 0.30 | -0.31 | 0.30 | -0.31 | 0.61 | +0.00 | 0.30 | -0.31 |
| No Eligible (%) | 49.08 % | 55.36 % | +6.28 pp | 54.34 % | +5.26 pp | 55.28 % | +6.20 pp | 45.74 % | -3.34 pp | 56.36 % | +7.28 pp |
| IPC Active | 1.93 inst/cycle | 1.58 inst/cycle | -0.35 | 1.59 inst/cycle | -0.34 | 1.64 inst/cycle | -0.29 | 2.05 inst/cycle | +0.12 | 1.68 inst/cycle | -0.25 |

---

## Iteration 3

**Best so far**: 1.2177× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_loop_to_overcome_low_wave_count | 0.029× | 0.012× | ✓ |  |
| B | persistent_tile_scheduling_fixed_sm_grid | 0.006× | 0.003× | ✓ |  |
| C | fix_grid_lambda_and_expand_autotune_for_sm_utilization | 1.025× | 1.201× | ✓ |  |
| D | persistent_tile_scheduling_inner_loop | 0.007× | 0.003× | ✓ |  |
| E | add_triton_autotune | 1.071× | 1.201× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_l... | Δ | persistent_tile_s... | Δ | fix_grid_lambda_a... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 297600 ns | +294020 ns (+8212.8%) | 1350000 ns | +1346420 ns (+37609.5%) | 2980 ns | -600 ns (-16.8%) | 1060000 ns | +1056420 ns (+29508.9%) | 2980 ns | -600 ns (-16.8%) |
| SM Throughput (%) | 20.11 % | 43.17 % | +23.06 pp | 74.86 % | +54.75 pp | 13.02 % | -7.09 pp | 62.71 % | +42.60 pp | 13.01 % | -7.10 pp |
| Memory Throughput (%) | 18.59 % | 46.82 % | +28.23 pp | 73.99 % | +55.40 pp | 11.84 % | -6.75 pp | 75.56 % | +56.97 pp | 11.79 % | -6.80 pp |
| DRAM Throughput (%) | 2.15 % | 0.03 % | -2.12 pp | 0.01 % | -2.14 pp | 2.59 % | +0.44 pp | 0.01 % | -2.14 pp | 2.59 % | +0.44 pp |
| Achieved Occupancy (%) | 48.56 % | 60.41 % | +11.85 pp | 52.42 % | +3.86 pp | 18.16 % | -30.40 pp | 59.72 % | +11.16 pp | 22.93 % | -25.63 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 30.00 register/thread | +2 |
| Waves/SM | 0.61 | 0.61 | +0.00 | 0.61 | +0.00 | 0.30 | -0.31 | 0.61 | +0.00 | 0.30 | -0.31 |
| No Eligible (%) | 49.08 % | 52.10 % | +3.02 pp | 23.47 % | -25.61 pp | 59.92 % | +10.84 pp | 33.80 % | -15.28 pp | 53.67 % | +4.59 pp |
| IPC Active | 1.93 inst/cycle | 1.83 inst/cycle | -0.10 | 3.06 inst/cycle | +1.13 | 1.19 inst/cycle | -0.74 | 2.61 inst/cycle | +0.68 | 1.73 inst/cycle | -0.20 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_loop_to_overcome_low_wave_count` (0.012×)

Converting a single-pass 2D-tiled Triton kernel into a persistent kernel with a while-loop over Y-tiles caused a 100x regression (2.94 µs → 297.6 µs). The root cause is that the persistent loop was implemented incorrectly: instead of each CTA processing a fixed subset of tiles, the while-loop condition `tile_id < num_y_tiles` combined with `tile_id += num_ctas` using `tl.num_programs(0)` as the stride caused each CTA to iterate over all 320 Y-tiles rather than its assigned share, exploding executed instructions from 305K to 98.6M (a 323x increase). Warp cycles per issued instruction also doubled (8.19 → 20.08) due to L1TEX scoreboard stalls (49.1% of cycles), indicating the while-loop's repeated global loads saturated the memory pipeline. The persistent pattern is structurally sound but requires correct tile assignment logic; the implementation flaw here was that `tl.num_programs(0)` returned the wrong stride value, causing all CTAs to redundantly process all tiles.

> **Applicable when**: Never apply this specific persistent loop pattern when tl.num_programs(0) is used as the tile stride in a while-loop without verifying it equals the actual grid size at runtime. Avoid persistent loops for very small kernels (total work < 1M elements, duration < 10 µs) where the while-loop overhead and potential stride bugs dominate.

> **Avoid when**: Kernel has small total work (xnumel * ynumel < 1M elements); kernel duration is already < 5 µs; tl.num_programs(0) behavior is not verified to return the actual launched grid size; the while-loop tile assignment logic has not been validated to ensure each tile is processed exactly once across all CTAs.

#### ❌ Plan D: `persistent_tile_scheduling_inner_loop` (0.003×)

Converting a 2D-grid pointwise kernel to a persistent 1D-grid loop (where each CTA iterates over multiple tiles) caused a ~294x increase in executed instructions (305K → 524M) and duration (2.94 µs → 1.06 ms), a 0.0034x speedup. The root cause is an implementation flaw: the launcher grid was left as the original hardcoded 2D tuple `(cdiv(xnumel, 128), cdiv(ynumel, 8)) = (1, 1280)` instead of being changed to `(NUM_SMS,)`. With grid=(1, 1280) and the persistent loop using `tl.num_programs(0)=1` as the stride, every CTA iterated over all 160 tiles instead of its assigned subset, causing each of the 1280 CTAs to process all 160 tiles — a 1280x work multiplication. The L1/TEX hit rate jumped from 14% to 92% (data fits in cache from repeated access) and memory throughput collapsed from 127 GB/s to 0.44 MB/s, confirming the same data was read repeatedly from cache rather than streaming through HBM.

> **Applicable when**: Never apply persistent tile scheduling without simultaneously updating the grid to (NUM_SMS,); the two changes are inseparable

> **Avoid when**: When the grid is not updated atomically with the persistent loop body — leaving the old grid causes each CTA to process all tiles (work multiplication by grid_size), causing catastrophic slowdown proportional to the original grid size. Also avoid when total_tiles is only marginally larger than NUM_SMS (e.g., 160 tiles / 132 SMs = 1.2 tiles/SM), as persistent loop overhead may not be amortized.

---

## Iteration 4

**Best so far**: 1.2177× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | flatten_2d_grid_to_1d_with_correct_autotune_lambda | 1.076× | 1.189× | ✓ | **✓ best** |
| B | fix_broken_persistent_loop_revert_to_2d_grid_with_autotune | 1.072× | 1.201× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_xblock_yblock_and_add_evict_first_store | 1.049× | 1.201× | ✓ |  |
| D | revert_broken_persistent_loop_to_correct_single_pass_2d_tiled_kernel | 1.038× | 1.155× | ✓ |  |


### Metric Comparison

| Metric | Baseline | flatten_2d_grid_t... | Δ | fix_broken_persis... | Δ | fix_grid_lambda_t... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 3010 ns | -570 ns (-15.9%) | 2980 ns | -600 ns (-16.8%) | 2980 ns | -600 ns (-16.8%) | 3100 ns | -480 ns (-13.4%) |
| SM Throughput (%) | 20.11 % | 18.46 % | -1.65 pp | 13.46 % | -6.65 pp | 13.39 % | -6.72 pp | 19.74 % | -0.37 pp |
| Memory Throughput (%) | 18.59 % | 8.54 % | -10.05 pp | 11.73 % | -6.86 pp | 11.77 % | -6.82 pp | 9.12 % | -9.47 pp |
| DRAM Throughput (%) | 2.15 % | 2.56 % | +0.41 pp | 2.60 % | +0.45 pp | 2.59 % | +0.44 pp | 2.47 % | +0.32 pp |
| Achieved Occupancy (%) | 48.56 % | 45.81 % | -2.75 pp | 22.82 % | -25.74 pp | 21.01 % | -27.55 pp | 44.66 % | -3.90 pp |
| Registers/Thread | 28.00 register/thread | 22.00 register/thread | -6 | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 22.00 register/thread | -6 |
| Waves/SM | 0.61 | 0.61 | +0.00 | 0.30 | -0.31 | 0.30 | -0.31 | 0.61 | +0.00 |
| No Eligible (%) | 49.08 % | 39.41 % | -9.67 pp | 55.57 % | +6.49 pp | 61.01 % | +11.93 pp | 39.70 % | -9.38 pp |
| IPC Active | 1.93 inst/cycle | 2.26 inst/cycle | +0.33 | 1.77 inst/cycle | -0.16 | 1.58 inst/cycle | -0.35 | 2.18 inst/cycle | +0.25 |

---

## Iteration 5

**Best so far**: 1.2177× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_xblock9_configs_for_imc_reduction | 1.044× | 1.201× | ✓ |  |
| C | fix_grid_launcher_and_swap_eviction_policies_for_transpose | 1.027× | 1.108× | ✓ |  |
| D | add_xblock8_configs_and_low_warp_variants_to_reduce_masked_thread_waste | 1.058× | 1.201× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_launcher... | Δ | add_xblock8_confi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3580 ns | 2980 ns | -600 ns (-16.8%) | 3230 ns | -350 ns (-9.8%) | 2980 ns | -600 ns (-16.8%) |
| SM Throughput (%) | 20.11 % | 13.77 % | -6.34 pp | 19.85 % | -0.26 pp | 19.72 % | -0.39 pp |
| Memory Throughput (%) | 18.59 % | 12.01 % | -6.58 pp | 9.08 % | -9.51 pp | 9.14 % | -9.45 pp |
| DRAM Throughput (%) | 2.15 % | 2.57 % | +0.42 pp | 2.38 % | +0.23 pp | 2.57 % | +0.42 pp |
| Achieved Occupancy (%) | 48.56 % | 22.76 % | -25.80 pp | 45.80 % | -2.76 pp | 45.40 % | -3.16 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 22.00 register/thread | -6 | 22.00 register/thread | -6 |
| Waves/SM | 0.61 | 0.30 | -0.31 | 0.61 | +0.00 | 0.61 | +0.00 |
| No Eligible (%) | 49.08 % | 51.79 % | +2.71 pp | 46.72 % | -2.36 pp | 42.66 % | -6.42 pp |
| IPC Active | 1.93 inst/cycle | 1.74 inst/cycle | -0.19 | 2.28 inst/cycle | +0.35 | 2.07 inst/cycle | +0.14 |

---

## Cumulative Progress

```
Baseline:  3580 ns  (1.000×)
Iter 1     3040 ns  (1.178×)  — fix_tile_sizes_and_add_autotune_for_transpose_permutation
Iter 2     2940 ns  (1.218×)  — expand_autotune_configs_for_higher_cta_count_and_warp_coverage

Final best: 3010 ns  (1.1894×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,32 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},   num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},   num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 1024}, num_warps=16),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
-def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 10240
-    xnumel = 9
+def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
+    xoffset = tl.program_id(0) * XBLOCK
     yoffset = tl.program_id(1) * YBLOCK
+    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
+    ymask = yindex < ynumel
     x2 = xindex
     y3 = yindex
     y0 = (yindex % 320)
     y1 = yindex // 320
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 320*x2 + 2880*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     ynumel = 10240
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26[grid](arg_0, arg_1, 10240, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26[grid](arg_0, arg_1, 10240, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
