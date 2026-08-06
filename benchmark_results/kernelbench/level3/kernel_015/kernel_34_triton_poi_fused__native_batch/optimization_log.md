# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3620 ns — **Best latency**: 3300 ns (**1.0970×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_static_grid_to_enable_autotune_tile_selection`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_34_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1664,1,1), Waves/SM: 0.79
- **Registers/thread**: 28
- **Duration**: 3620 ns
- **SM Throughput**: 24.05 % | **Memory**: 22.47 % | **DRAM**: 2.74 %
- **Achieved Occupancy**: 53.95 %
- **IPC Active**: 2.01 inst/cycle | **No Eligible**: 49.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fill_gpu | 1.088× | 1.231× | ✓ | **✓ best** |
| C | fix_tile_shape_for_coalesced_access_and_occupancy | 1.042× | 1.077× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | fix_tile_shape_fo... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 2940 ns | -680 ns (-18.8%) | 3360 ns | -260 ns (-7.2%) |
| SM Throughput (%) | 24.05 % | 9.00 % | -15.05 pp | 21.89 % | -2.16 pp |
| Memory Throughput (%) | 22.47 % | 3.34 % | -19.13 pp | 10.45 % | -12.02 pp |
| DRAM Throughput (%) | 2.74 % | 3.34 % | +0.60 pp | 2.99 % | +0.25 pp |
| Achieved Occupancy (%) | 53.95 % | 49.13 % | -4.82 pp | 51.77 % | -2.18 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 28.00 register/thread | +0 |
| Waves/SM | 0.79 | 0.79 | +0.00 | 0.79 | +0.00 |
| No Eligible (%) | 49.81 % | 64.27 % | +14.46 pp | 34.34 % | -15.47 pp |
| IPC Active | 2.01 inst/cycle | 0.93 inst/cycle | -1.08 | 2.45 inst/cycle | +0.44 |

---

## Iteration 2

**Best so far**: 1.2313× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.004× | 0.002× | ✓ |  |
| B | fix_hardcoded_grid_to_use_meta_and_add_coalesced_configs | 1.064× | 1.020× | ✓ |  |
| C | persistent_tile_scheduling | 0.004× | 0.002× | ✓ |  |
| D | fix_static_grid_to_enable_autotune_tile_selection | 1.090× | 1.097× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.536× | 0.332× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_hardcoded_gri... | Δ | persistent_tile_s... | Δ | fix_static_grid_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 1770000 ns | +1766380 ns (+48795.0%) | 3550 ns | -70 ns (-1.9%) | 2160000 ns | +2156380 ns (+59568.5%) | 3300 ns | -320 ns (-8.8%) | 10910 ns | +7290 ns (+201.4%) |
| SM Throughput (%) | 24.05 % | 66.05 % | +42.00 pp | 22.62 % | -1.43 pp | 76.15 % | +52.10 pp | 21.96 % | -2.09 pp | 52.34 % | +28.29 pp |
| Memory Throughput (%) | 22.47 % | 79.88 % | +57.41 pp | 10.80 % | -11.67 pp | 75.32 % | +52.85 pp | 10.49 % | -11.98 pp | 66.66 % | +44.19 pp |
| DRAM Throughput (%) | 2.74 % | 0.01 % | -2.73 pp | 2.80 % | +0.06 pp | 0.01 % | -2.73 pp | 3.01 % | +0.27 pp | 0.92 % | -1.82 pp |
| Achieved Occupancy (%) | 53.95 % | 68.79 % | +14.84 pp | 51.57 % | -2.38 pp | 60.38 % | +6.43 pp | 51.52 % | -2.43 pp | 31.56 % | -22.39 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 80.00 register/thread | +52 |
| Waves/SM | 0.79 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 | 2.10 | +1.31 |
| No Eligible (%) | 49.81 % | 30.98 % | -18.83 pp | 34.97 % | -14.84 pp | 20.64 % | -29.17 pp | 34.14 % | -15.67 pp | 35.41 % | -14.40 pp |
| IPC Active | 2.01 inst/cycle | 2.76 inst/cycle | +0.75 | 2.45 inst/cycle | +0.44 | 3.17 inst/cycle | +1.16 | 2.44 inst/cycle | +0.43 | 2.57 inst/cycle | +0.56 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.011×)

Even with a correctly-formed 1D lambda grid (min(132, total_tiles),), applying a persistent tile scheduling pattern to a very short kernel (~3 µs) caused a catastrophic regression: executed instructions exploded from 599K to 143M (~239x), and kernel duration increased from 3.33 µs to 328.74 µs (~99x slower). The extremely high L1 hit rate (95.8%) and L2 hit rate (96.58%) after the change confirm that the same tiles are being re-processed many times from cache — indicating the persistent loop is iterating far more times than intended, likely because the grid was capped at min(132, total_tiles) but total_tiles was computed incorrectly inside the kernel (using compile-time constexpr XBLOCK/YBLOCK vs. runtime values), causing the loop bound to mismatch. For short-duration kernels (<10 µs) with small total work (ynumel=13312, xnumel=9), persistent kernel patterns add loop overhead and tile-recomputation cost that far exceeds any SM-utilization benefit, even when the grid formula is nominally correct.

> **Applicable when**: Never apply persistent tile scheduling to kernels with duration < 10 µs or total work < ~100K elements. The pattern is only beneficial when total_tiles >> NUM_SMS and the kernel is long-running enough that SM-utilization amortizes the loop overhead.

> **Avoid when**: Kernel duration is < 10 µs; total work is small (ynumel * xnumel < 1M); waves_per_sm is already near 1.0; the kernel is a simple pointwise or transpose with no reuse across tiles. Do NOT use persistent scheduling as a remedy for low waves_per_sm when the underlying cause is simply a small problem size — the loop overhead will dominate.

#### ❌ Plan C: `persistent_tile_scheduling` (0.011×)

Even with a correctly-formed 1D lambda grid (min(132, total_tiles),), applying a persistent tile scheduling pattern to a very short kernel (~3 µs) caused a catastrophic regression: executed instructions exploded from 599K to 143M (~239x), and kernel duration increased from 3.33 µs to 328.74 µs (~99x slower). The extremely high L1 hit rate (95.8%) and L2 hit rate (96.58%) after the change confirm that the same tiles are being re-processed many times from cache — indicating the persistent loop is iterating far more times than intended, likely because the grid was capped at min(132, total_tiles) but total_tiles was computed incorrectly inside the kernel (using compile-time constexpr XBLOCK/YBLOCK vs. runtime values), causing the loop bound to mismatch. For short-duration kernels (<10 µs) with small total work (ynumel=13312, xnumel=9), persistent kernel patterns add loop overhead and tile-recomputation cost that far exceeds any SM-utilization benefit, even when the grid formula is nominally correct.

> **Applicable when**: Never apply persistent tile scheduling to kernels with duration < 10 µs or total work < ~100K elements. The pattern is only beneficial when total_tiles >> NUM_SMS and the kernel is long-running enough that SM-utilization amortizes the loop overhead.

> **Avoid when**: Kernel duration is < 10 µs; total work is small (ynumel * xnumel < 1M); waves_per_sm is already near 1.0; the kernel is a simple pointwise or transpose with no reuse across tiles. Do NOT use persistent scheduling as a remedy for low waves_per_sm when the underlying cause is simply a small problem size — the loop overhead will dominate.

---

## Iteration 3

**Best so far**: 1.2313× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_eviction_policy_for_streaming_transpose_loads_and_stores | 1.067× | 1.087× | ✓ |  |
| B | fix_kernel_function_grid_to_use_lambda_for_autotune_coalesced_stores | 0.026× | 0.006× | ✓ |  |
| C | fix_autotune_xblock_register_spill_with_bounded_configs_and_lambda_grid | 0.023× | 0.006× | ✓ |  |
| D | fix_broken_persistent_kernel_grid_to_1d_sm_count | 0.004× | 0.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_eviction_poli... | Δ | fix_kernel_functi... | Δ | fix_autotune_xblo... | Δ | fix_broken_persis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 3330 ns | -290 ns (-8.0%) | 637890 ns | +634270 ns (+17521.3%) | 568290 ns | +564670 ns (+15598.6%) | 1770000 ns | +1766380 ns (+48795.0%) |
| SM Throughput (%) | 24.05 % | 23.75 % | -0.30 pp | 13.43 % | -10.62 pp | 66.19 % | +42.14 pp | 62.81 % | +38.76 pp |
| Memory Throughput (%) | 22.47 % | 10.91 % | -11.56 pp | 57.00 % | +34.53 pp | 30.76 % | +8.29 pp | 75.87 % | +53.40 pp |
| DRAM Throughput (%) | 2.74 % | 2.97 % | +0.23 pp | 0.02 % | -2.72 pp | 0.03 % | -2.71 pp | 0.01 % | -2.73 pp |
| Achieved Occupancy (%) | 53.95 % | 51.32 % | -2.63 pp | 57.33 % | +3.38 pp | 57.88 % | +3.93 pp | 69.92 % | +15.97 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 29.00 register/thread | +1 | 16.00 register/thread | -12 | 32.00 register/thread | +4 |
| Waves/SM | 0.79 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 |
| No Eligible (%) | 49.81 % | 33.94 % | -15.87 pp | 84.19 % | +34.38 pp | 29.89 % | -19.92 pp | 33.72 % | -16.09 pp |
| IPC Active | 2.01 inst/cycle | 2.47 inst/cycle | +0.46 | 0.63 inst/cycle | -1.38 | 2.79 inst/cycle | +0.78 | 2.74 inst/cycle | +0.73 |

---

## Iteration 4

**Best so far**: 1.2313× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | wire_autotune_grid_lambda_to_eliminate_static_tile_bypass | 1.047× | 1.077× | ✓ |  |
| B | fix_persistent_kernel_grid_to_sm_count_and_wire_autotune_lambda | 0.026× | 0.011× | ✓ |  |
| C | fix_persistent_kernel_grid_to_1d_sm_aware_lambda | 0.026× | 0.006× | ✓ |  |
| D | swap_tile_loop_order_for_coalesced_output_stores | 1.083× | 1.037× | ✓ |  |


### Metric Comparison

| Metric | Baseline | wire_autotune_gri... | Δ | fix_persistent_ke... | Δ | fix_persistent_ke... | Δ | swap_tile_loop_or... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3620 ns | 3360 ns | -260 ns (-7.2%) | 328740 ns | +325120 ns (+8981.2%) | 610560 ns | +606940 ns (+16766.3%) | 3490 ns | -130 ns (-3.6%) |
| SM Throughput (%) | 24.05 % | 23.28 % | -0.77 pp | 57.43 % | +33.38 pp | 13.12 % | -10.93 pp | 21.94 % | -2.11 pp |
| Memory Throughput (%) | 22.47 % | 10.64 % | -11.83 pp | 34.85 % | +12.38 pp | 61.03 % | +38.56 pp | 10.48 % | -11.99 pp |
| DRAM Throughput (%) | 2.74 % | 2.96 % | +0.22 pp | 0.04 % | -2.70 pp | 0.02 % | -2.72 pp | 2.86 % | +0.12 pp |
| Achieved Occupancy (%) | 53.95 % | 50.80 % | -3.15 pp | 70.91 % | +16.96 pp | 57.00 % | +3.05 pp | 51.54 % | -2.41 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 16.00 register/thread | -12 | 29.00 register/thread | +1 | 28.00 register/thread | +0 |
| Waves/SM | 0.79 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 | 0.79 | +0.00 |
| No Eligible (%) | 49.81 % | 36.45 % | -13.36 pp | 38.51 % | -11.30 pp | 83.91 % | +34.10 pp | 34.65 % | -15.16 pp |
| IPC Active | 2.01 inst/cycle | 2.46 inst/cycle | +0.45 | 2.41 inst/cycle | +0.40 | 0.63 inst/cycle | -1.38 | 2.45 inst/cycle | +0.44 |

---

## Cumulative Progress

```
Baseline:  3620 ns  (1.000×)
Iter 1     2940 ns  (1.231×)  — autotune_tile_sizes_to_fill_gpu

Final best: 3300 ns  (1.0970×)
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
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 1}, num_warps=4),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 1}, num_warps=4),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 1}, num_warps=4),
+        triton.Config({'YBLOCK': 1024, 'XBLOCK': 4}, num_warps=4),
+        triton.Config({'YBLOCK': 512,  'XBLOCK': 4}, num_warps=4),
+        triton.Config({'YBLOCK': 256,  'XBLOCK': 4}, num_warps=8),
+        triton.Config({'YBLOCK': 128,  'XBLOCK': 16}, num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 13312
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +32,8 @@

     y3 = yindex
     y0 = (yindex % 416)
     y1 = yindex // 416
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 416*x2 + 3744*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 416*x2 + 3744*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +41,7 @@

     ynumel = 13312
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34[grid](arg_0, arg_1, 13312, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34[grid](arg_0, arg_1, 13312, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
