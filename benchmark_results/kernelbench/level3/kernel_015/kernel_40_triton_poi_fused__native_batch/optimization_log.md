# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3840 ns — **Best latency**: 3360 ns (**1.1429×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_and_add_num_stages3_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_40_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1920,1,1), Waves/SM: 0.91
- **Registers/thread**: 28
- **Duration**: 3840 ns
- **SM Throughput**: 25.76 % | **Memory**: 24.11 % | **DRAM**: 2.99 %
- **Achieved Occupancy**: 60.73 %
- **IPC Active**: 2.14 inst/cycle | **No Eligible**: 44.68 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_better_occupancy_and_coalescing | 1.055× | 1.091× | ✓ |  |
| D | fix_tile_dimensions_and_add_vectorized_loads | 1.118× | 1.239× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_dimensio... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3520 ns | -320 ns (-8.3%) | 3100 ns | -740 ns (-19.3%) |
| SM Throughput (%) | 25.76 % | 23.53 % | -2.23 pp | 9.46 % | -16.30 pp |
| Memory Throughput (%) | 24.11 % | 11.71 % | -12.40 pp | 3.69 % | -20.42 pp |
| DRAM Throughput (%) | 2.99 % | 3.26 % | +0.27 pp | 3.69 % | +0.70 pp |
| Achieved Occupancy (%) | 60.73 % | 64.22 % | +3.49 pp | 44.89 % | -15.84 pp |
| Registers/Thread | 28.00 register/thread | 22.00 register/thread | -6 | 16.00 register/thread | -12 |
| Waves/SM | 0.91 | 0.91 | +0.00 | 0.91 | +0.00 |
| No Eligible (%) | 44.68 % | 33.77 % | -10.91 pp | 60.98 % | +16.30 pp |
| IPC Active | 2.14 inst/cycle | 2.48 inst/cycle | +0.34 | 1.29 inst/cycle | -0.85 |

---

## Iteration 2

**Best so far**: 1.2387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_for_small_grid_permutation_kernel | 0.003× | 0.001× | ✓ |  |
| B | fix_grid_lambda_to_respect_autotune_tile_sizes | 1.117× | 1.239× | ✓ |  |
| C | persistent_tile_scheduling_flattened_1d_grid | 0.019× | 0.008× | ✓ |  |
| D | fix_grid_lambda_to_use_autotune_meta_and_expand_large_yblock_configs | 1.057× | 1.082× | ✓ |  |
| E | add_triton_autotune | 1.089× | 1.239× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_grid_lambda_t... | Δ | persistent_tile_s... | Δ | fix_grid_lambda_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 2870000 ns | +2866160 ns (+74639.6%) | 3100 ns | -740 ns (-19.3%) | 487300 ns | +483460 ns (+12590.1%) | 3550 ns | -290 ns (-7.6%) | 3100 ns | -740 ns (-19.3%) |
| SM Throughput (%) | 25.76 % | 80.34 % | +54.58 pp | 9.45 % | -16.31 pp | 65.47 % | +39.71 pp | 23.28 % | -2.48 pp | 9.25 % | -16.51 pp |
| Memory Throughput (%) | 24.11 % | 75.65 % | +51.54 pp | 3.71 % | -20.40 pp | 42.88 % | +18.77 pp | 11.43 % | -12.68 pp | 3.68 % | -20.43 pp |
| DRAM Throughput (%) | 2.99 % | 0.01 % | -2.98 pp | 3.71 % | +0.72 pp | 0.03 % | -2.96 pp | 3.25 % | +0.26 pp | 3.68 % | +0.69 pp |
| Achieved Occupancy (%) | 60.73 % | 65.04 % | +4.31 pp | 45.16 % | -15.57 pp | 67.69 % | +6.96 pp | 64.94 % | +4.21 pp | 59.86 % | -0.87 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 22.00 register/thread | -6 | 16.00 register/thread | -12 |
| Waves/SM | 0.91 | 0.91 | +0.00 | 0.91 | +0.00 | 0.91 | +0.00 | 0.91 | +0.00 | 0.91 | +0.00 |
| No Eligible (%) | 44.68 % | 16.85 % | -27.83 pp | 64.71 % | +20.03 pp | 31.12 % | -13.56 pp | 35.14 % | -9.54 pp | 62.80 % | +18.12 pp |
| IPC Active | 2.14 inst/cycle | 3.33 inst/cycle | +1.19 | 1.31 inst/cycle | -0.83 | 2.76 inst/cycle | +0.62 | 2.32 inst/cycle | +0.18 | 0.91 inst/cycle | -1.23 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_for_small_grid_permutation_kernel` (0.001×)

Applying a persistent tile-scheduling pattern (one CTA per SM, while-loop over tiles) to a small permutation kernel caused a catastrophic ~770x slowdown (3.1 µs → 2.87 ms). The root cause was an implementation flaw: the kernel_function still used the original hardcoded grid `(cdiv(9,128), cdiv(15360,8)) = (1, 1920)` instead of the intended `(132,)` persistent grid, while the kernel body was rewritten to use `tl.num_programs(0)` as the stride — so 1920 CTAs each ran a while-loop iterating over all 240 tiles with stride=1920, executing ~8,300x more instructions than the original (1.83B vs 218K). Occupancy and scheduler metrics improved (occupancy 44.9%→65%, No Eligible 61%→17%) because the kernel was now genuinely compute-bound, but the total work exploded. The persistent kernel pattern itself is not wrong, but the grid launch must be updated atomically with the kernel body rewrite.

> **Applicable when**: Never apply this specific pattern (persistent kernel for small grid) without simultaneously updating the grid launch to (NUM_SMS,) — the kernel body and grid launch are tightly coupled and must be changed together.

> **Avoid when**: When the grid launch cannot be atomically updated with the kernel body rewrite; when the kernel is already very short-lived (< 5 µs) and launch overhead dominates — persistent kernels add while-loop overhead that only pays off when total_tiles >> NUM_SMS and the kernel duration is long enough to amortize it; when total_tiles <= NUM_SMS (no benefit from persistence, each SM handles at most 1 tile anyway).

#### ❌ Plan C: `persistent_tile_scheduling_flattened_1d_grid` (0.008×)

Replacing a 2D grid launch with a persistent 1D grid where each CTA loops over multiple tiles via a software while-loop caused a ~157x slowdown (3.1 µs → 487.3 µs). The executed instruction count exploded from 218,880 to 248,954,880 (~1138x more instructions), because the while-loop body — including tl.constexpr tile index arithmetic (NUM_TILES_X, NUM_TILES_Y, TOTAL_TILES computed as constexpr from XBLOCK/YBLOCK) — was unrolled or repeated at massive scale inside the persistent loop. The GPU became compute-bound (65% SM throughput) rather than latency-bound, but the sheer volume of instructions dwarfed any wave-boundary savings. For small kernels with <2 waves, the overhead of a persistent software loop far exceeds the cost of the partial wave; the correct fix is to adjust tile sizes (XBLOCK/YBLOCK) to increase grid size, not to add a persistent loop.

> **Applicable when**: Never applicable for small pointwise kernels with <2 waves where the per-tile work is tiny (microsecond-scale). Persistent loops are only beneficial when per-tile work is large (e.g., matmul tiles with heavy compute) and the loop overhead is negligible relative to tile compute time.

> **Avoid when**: Kernel duration is already in the low-microsecond range (< 10 µs); per-tile work is a simple load-permute-store with no reduction or heavy compute; total tile count is small (< 10,000); the grid is already close to filling SMs (waves_per_sm ≥ 0.5). In these cases, the software loop overhead and instruction explosion will dominate any wave-boundary savings.

---

## Iteration 3

**Best so far**: 1.2387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_to_use_meta_and_add_xblock1_configs | 1.119× | 1.153× | ✓ | **✓ best** |
| C | coalesce_stores_by_covering_full_x_dimension_in_one_tile | 1.118× | 1.239× | ✓ |  |
| D | shared_memory_transpose_for_coalesced_permutation | 1.110× | 0.765× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | coalesce_stores_b... | Δ | shared_memory_tra... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3330 ns | -510 ns (-13.3%) | 3100 ns | -740 ns (-19.3%) | 5020 ns | +1180 ns (+30.7%) |
| SM Throughput (%) | 25.76 % | 9.83 % | -15.93 pp | 9.73 % | -16.03 pp | 46.98 % | +21.22 pp |
| Memory Throughput (%) | 24.11 % | 3.41 % | -20.70 pp | 3.68 % | -20.43 pp | 38.58 % | +14.47 pp |
| DRAM Throughput (%) | 2.99 % | 3.41 % | +0.42 pp | 3.68 % | +0.69 pp | 2.28 % | -0.71 pp |
| Achieved Occupancy (%) | 60.73 % | 46.24 % | -14.49 pp | 44.45 % | -16.28 pp | 59.31 % | -1.42 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.91 | 0.91 | +0.00 | 0.91 | +0.00 | 0.91 | +0.00 |
| No Eligible (%) | 44.68 % | 61.77 % | +17.09 pp | 68.51 % | +23.83 pp | 23.03 % | -21.65 pp |
| IPC Active | 2.14 inst/cycle | 1.33 inst/cycle | -0.81 | 1.30 inst/cycle | -0.84 | 3.03 inst/cycle | +0.89 |

---

## Iteration 4

**Best so far**: 1.2387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_num_stages_to_hide_store_latency | 1.079× | 1.223× | ✓ |  |
| B | expand_autotune_configs_large_yblock_with_lambda_grid | 1.090× | 1.223× | ✓ |  |
| C | fix_grid_lambda_and_cover_full_x_dimension_with_single_tile | 1.112× | 1.123× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | expand_autotune_c... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3140 ns | -700 ns (-18.2%) | 3140 ns | -700 ns (-18.2%) | 3420 ns | -420 ns (-10.9%) |
| SM Throughput (%) | 25.76 % | 9.12 % | -16.64 pp | 9.93 % | -15.83 pp | 17.34 % | -8.42 pp |
| Memory Throughput (%) | 24.11 % | 3.65 % | -20.46 pp | 3.66 % | -20.45 pp | 15.31 % | -8.80 pp |
| DRAM Throughput (%) | 2.99 % | 3.65 % | +0.66 pp | 3.66 % | +0.67 pp | 3.34 % | +0.35 pp |
| Achieved Occupancy (%) | 60.73 % | 43.04 % | -17.69 pp | 45.56 % | -15.17 pp | 34.22 % | -26.51 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 30.00 register/thread | +2 |
| Waves/SM | 0.91 | 0.91 | +0.00 | 0.91 | +0.00 | 0.45 | -0.46 |
| No Eligible (%) | 44.68 % | 62.14 % | +17.46 pp | 62.93 % | +18.25 pp | 49.38 % | +4.70 pp |
| IPC Active | 2.14 inst/cycle | 1.25 inst/cycle | -0.89 | 1.32 inst/cycle | -0.82 | 2.10 inst/cycle | -0.04 |

---

## Iteration 5

**Best so far**: 1.2387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_num_stages3_configs | 1.120× | 1.143× | ✓ | **✓ best** |
| B | add_num_stages2_to_autotune_configs_for_latency_hiding | 1.120× | 1.164× | ✓ |  |
| C | fix_hardcoded_grid_lambda_and_add_num_stages_1_fallback_configs | 1.112× | 1.052× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | add_num_stages2_t... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3840 ns | 3360 ns | -480 ns (-12.5%) | 3300 ns | -540 ns (-14.1%) | 3650 ns | -190 ns (-4.9%) |
| SM Throughput (%) | 25.76 % | 9.43 % | -16.33 pp | 9.91 % | -15.85 pp | 16.54 % | -9.22 pp |
| Memory Throughput (%) | 24.11 % | 3.41 % | -20.70 pp | 3.44 % | -20.67 pp | 14.58 % | -9.53 pp |
| DRAM Throughput (%) | 2.99 % | 3.41 % | +0.42 pp | 3.44 % | +0.45 pp | 3.14 % | +0.15 pp |
| Achieved Occupancy (%) | 60.73 % | 45.04 % | -15.69 pp | 49.47 % | -11.26 pp | 35.56 % | -25.17 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 30.00 register/thread | +2 |
| Waves/SM | 0.91 | 0.91 | +0.00 | 0.91 | +0.00 | 0.45 | -0.46 |
| No Eligible (%) | 44.68 % | 60.12 % | +15.44 pp | 61.79 % | +17.11 pp | 43.87 % | -0.81 pp |
| IPC Active | 2.14 inst/cycle | 1.32 inst/cycle | -0.82 | 1.30 inst/cycle | -0.84 | 1.86 inst/cycle | -0.28 |

---

## Cumulative Progress

```
Baseline:  3840 ns  (1.000×)
Iter 1     3100 ns  (1.239×)  — fix_tile_dimensions_and_add_vectorized_loads

Final best: 3360 ns  (1.1429×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,43 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK':  64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK':  32}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=2, num_stages=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=1),
+        # num_stages=3 configs for software pipelining
+        triton.Config({'XBLOCK': 1, 'YBLOCK':  32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 1, 'YBLOCK':  64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 1, 'YBLOCK': 512}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  32}, num_warps=2, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK':  64}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 15360
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +49,8 @@

     y3 = yindex
     y0 = (yindex % 480)
     y1 = yindex // 480
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 480*x2 + 4320*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 480*x2 + 4320*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +58,7 @@

     ynumel = 15360
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40[grid](arg_0, arg_1, 15360, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40[grid](arg_0, arg_1, 15360, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
