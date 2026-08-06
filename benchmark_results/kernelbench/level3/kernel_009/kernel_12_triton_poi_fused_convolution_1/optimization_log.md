# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3520 ns — **Best latency**: 2880 ns (**1.2222×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`persistent_tile_scheduling_via_lambda_grid`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_12`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_12_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1536,1,1), Waves/SM: 0.73
- **Registers/thread**: 28
- **Duration**: 3520 ns
- **SM Throughput**: 22.63 % | **Memory**: 20.86 % | **DRAM**: 2.62 %
- **Achieved Occupancy**: 51.67 %
- **IPC Active**: 1.88 inst/cycle | **No Eligible**: 47.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_tile_shape_and_autotune_for_underutilized_grid | 1.088× | 1.181× | ✓ |  |
| C | fix_tile_shape_for_small_xnumel_and_improve_coalescing | 1.094× | 1.222× | ✓ | **✓ best** |
| D | fix_tile_dimensions_and_vectorize_loads | 1.087× | 1.222× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_tile_shape_an... | Δ | fix_tile_shape_fo... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 2980 ns | -540 ns (-15.3%) | 2880 ns | -640 ns (-18.2%) | 2880 ns | -640 ns (-18.2%) |
| SM Throughput (%) | 22.63 % | 8.16 % | -14.47 pp | 7.98 % | -14.65 pp | 8.19 % | -14.44 pp |
| Memory Throughput (%) | 20.86 % | 3.11 % | -17.75 pp | 3.18 % | -17.68 pp | 3.19 % | -17.67 pp |
| DRAM Throughput (%) | 2.62 % | 3.11 % | +0.49 pp | 3.18 % | +0.56 pp | 3.19 % | +0.57 pp |
| Achieved Occupancy (%) | 51.67 % | 37.57 % | -14.10 pp | 37.35 % | -14.32 pp | 37.71 % | -13.96 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 47.07 % | 66.72 % | +19.65 pp | 66.44 % | +19.37 pp | 65.82 % | +18.75 pp |
| IPC Active | 1.88 inst/cycle | 1.10 inst/cycle | -0.78 | 1.10 inst/cycle | -0.78 | 1.12 inst/cycle | -0.76 |

---

## Iteration 2

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.022× | 0.012× | ✓ |  |
| B | persistent_tile_scheduling_for_underutilized_grid | 0.003× | 0.002× | ✓ |  |
| C | persistent_tile_scheduling_via_lambda_grid | 1.051× | 1.222× | ✓ |  |
| D | add_triton_autotune | 1.063× | 1.222× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 283300 ns | +279780 ns (+7948.3%) | 1860000 ns | +1856480 ns (+52740.9%) | 2880 ns | -640 ns (-18.2%) | 2880 ns | -640 ns (-18.2%) |
| SM Throughput (%) | 22.63 % | 56.57 % | +33.94 pp | 76.41 % | +53.78 pp | 8.10 % | -14.53 pp | 8.18 % | -14.45 pp |
| Memory Throughput (%) | 20.86 % | 37.13 % | +16.27 pp | 75.56 % | +54.70 pp | 3.19 % | -17.67 pp | 3.19 % | -17.67 pp |
| DRAM Throughput (%) | 2.62 % | 0.05 % | -2.57 pp | 0.01 % | -2.61 pp | 3.19 % | +0.57 pp | 3.19 % | +0.57 pp |
| Achieved Occupancy (%) | 51.67 % | 62.75 % | +11.08 pp | 58.52 % | +6.85 pp | 39.11 % | -12.56 pp | 37.95 % | -13.72 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 47.07 % | 40.72 % | -6.35 pp | 21.75 % | -25.32 pp | 65.21 % | +18.14 pp | 66.92 % | +19.85 pp |
| IPC Active | 1.88 inst/cycle | 2.40 inst/cycle | +0.52 | 3.14 inst/cycle | +1.26 | 1.15 inst/cycle | -0.73 | 1.10 inst/cycle | -0.78 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_for_underutilized_grid` (0.002×)

Converting a 2D grid kernel to a persistent 1D while-loop kernel while leaving the launcher grid unchanged (still using the old 2D formula `(cdiv(9,128), cdiv(12288,8)) = (1,1536)`) caused a catastrophic regression: executed instructions exploded from 173K to 1.11B (~6,400x) and duration increased from 2.88 µs to 1.86 ms (~645x slower). With a 2D grid of shape (1,1536), `tl.program_id(0)` is always 0 and `tl.num_programs(0)` is 1 for every CTA, so each of the 1536 CTAs has pid=0 and loop stride=1, causing every CTA to process all 864 tiles sequentially instead of once. The persistent while-loop pattern is only correct when the launcher grid is simultaneously changed to `(NUM_SMS,)` and tile counts are passed as kernel arguments rather than computed inside the kernel with `tl.cdiv`.

> **Applicable when**: Never apply this specific broken pattern. The persistent while-loop approach is theoretically applicable when waves_per_sm < 1.0 and the kernel is memory-bandwidth-bound, but ONLY when the launcher grid is simultaneously updated to (NUM_SMS,).

> **Avoid when**: Any time the launcher grid formula is not simultaneously updated to (NUM_SMS,) when converting to a persistent while-loop. With a 2D grid, tl.program_id(0) and tl.num_programs(0) return dimension-0 values only, causing every CTA to have pid=0 and stride=1 if dim-0 size is 1, making every CTA process all tiles serially. Also avoid persistent loops entirely for simple pointwise/transpose kernels with small tile counts — hardware dispatch is always more efficient.

---

## Iteration 3

**Best so far**: 1.2222× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_autotune_grid_lambda_to_use_meta_tile_sizes | 0.022× | 0.012× | ✓ |  |
| C | specialize_small_constexpr_dimension_to_eliminate_masked_threads | 0.003× | 0.002× | ✓ |  |
| D | shared_memory_staged_transpose_to_coalesce_stores | 1.051× | 1.147× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | specialize_small_... | Δ | shared_memory_sta... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3520 ns | 291460 ns | +287940 ns (+8180.1%) | 1860000 ns | +1856480 ns (+52740.9%) | 3070 ns | -450 ns (-12.8%) |
| SM Throughput (%) | 22.63 % | 55.84 % | +33.21 pp | 75.87 % | +53.24 pp | 14.03 % | -8.60 pp |
| Memory Throughput (%) | 20.86 % | 34.48 % | +13.62 pp | 75.03 % | +54.17 pp | 4.04 % | -16.82 pp |
| DRAM Throughput (%) | 2.62 % | 0.05 % | -2.57 pp | 0.01 % | -2.61 pp | 1.01 % | -1.61 pp |
| Achieved Occupancy (%) | 51.67 % | 65.84 % | +14.17 pp | 56.99 % | +5.32 pp | 51.15 % | -0.52 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 32.00 register/thread | +4 | 16.00 register/thread | -12 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 47.07 % | 40.28 % | -6.79 pp | 21.61 % | -25.46 pp | 53.73 % | +6.66 pp |
| IPC Active | 1.88 inst/cycle | 2.34 inst/cycle | +0.46 | 3.17 inst/cycle | +1.29 | 1.77 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  3520 ns  (1.000×)
Iter 1     2880 ns  (1.222×)  — fix_tile_shape_for_small_xnumel_and_improve_coalescing

Final best: 2880 ns  (1.2222×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 12288
@@ -22,14 +36,13 @@

     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 96*x2 + 864*y1), tmp0, xmask & ymask)
 
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     ynumel = 12288
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
