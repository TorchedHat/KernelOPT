# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3170 ns — **Best latency**: 2590 ns (**1.2239×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_grid_lambda_and_expand_autotune_with_num_warps2_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_027/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 28
- **Duration**: 3170 ns
- **SM Throughput**: 16.57 % | **Memory**: 15.75 % | **DRAM**: 1.95 %
- **Achieved Occupancy**: 38.75 %
- **IPC Active**: 1.68 inst/cycle | **No Eligible**: 54.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_autotune | 0.044× | 0.018× | ✓ |  |
| B | fix_tile_sizing_and_autotune_for_small_xnumel | 1.121× | 1.253× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_tile_sizing_a... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3170 ns | 179390 ns | +176220 ns (+5559.0%) | 2530 ns | -640 ns (-20.2%) |
| SM Throughput (%) | 16.57 % | 34.26 % | +17.69 pp | 6.20 % | -10.37 pp |
| Memory Throughput (%) | 15.75 % | 43.53 % | +27.78 pp | 2.42 % | -13.33 pp |
| DRAM Throughput (%) | 1.95 % | 0.04 % | -1.91 pp | 2.42 % | +0.47 pp |
| Achieved Occupancy (%) | 38.75 % | 45.50 % | +6.75 pp | 30.67 % | -8.08 pp |
| Registers/Thread | 28.00 register/thread | 56.00 register/thread | +28 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.86 | +0.38 | 0.48 | +0.00 |
| No Eligible (%) | 54.60 % | 59.75 % | +5.15 pp | 71.49 % | +16.89 pp |
| IPC Active | 1.68 inst/cycle | 1.57 inst/cycle | -0.11 | 0.81 inst/cycle | -0.87 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_autotune` (0.021×)

Adding num_stages=3 to the autotune configs of a persistent-loop Triton kernel did not prevent the catastrophic regression seen in prior iterations: executed instructions exploded from 115K to 59M (+511x) and kernel duration increased from 2.66 µs to 148.32 µs (+56x). The waves_per_sm remained at 0.48 — the persistent loop did not increase parallelism. Although L1/TEX hit rate rose to 96.23% and L2 hit rate to 99.65% (data reused across loop iterations within the same CTA), the sequential tile processing completely dominates. This is the third consecutive iteration confirming that persistent tile scheduling is strictly harmful for sub-wave kernels regardless of grid formulation, autotune config set, or num_stages value.

> **Applicable when**: Never applicable as a replacement for a flat 2D grid on sub-wave kernels. Persistent loops are only beneficial when: (1) total work spans many waves (>>4 waves/SM), (2) there is significant per-tile setup cost to amortize, or (3) inter-tile communication requires persistence (e.g., cross-tile reductions).

> **Avoid when**: Kernel is wave-limited at < 1 wave/SM (grid_size ≤ SM_count); kernel is a simple elementwise or permutation operation with no inter-tile dependencies; instruction count is already low (< 200K); the bottleneck is occupancy/waves rather than per-tile overhead. Adding num_stages=3 to the persistent loop configs does NOT fix the regression — the loop serialization dominates regardless of pipelining depth.

---

## Iteration 2

**Best so far**: 1.253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_activate_existing_autotune_configs | 1.124× | 1.224× | ✓ | **✓ best** |
| B | shared_memory_transpose_to_fix_uncoalesced_global_access | 1.024× | 1.210× | ✓ |  |
| C | revert_persistent_loop_to_flat_2d_grid_with_small_xblock_autotune | 1.077× | 1.238× | ✓ |  |
| D | fix_grid_launch_to_match_persistent_tile_loop_and_expand_autotune | 0.048× | 0.021× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | shared_memory_tra... | Δ | revert_persistent... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3170 ns | 2590 ns | -580 ns (-18.3%) | 2620 ns | -550 ns (-17.4%) | 2560 ns | -610 ns (-19.2%) | 151460 ns | +148290 ns (+4677.9%) |
| SM Throughput (%) | 16.57 % | 6.18 % | -10.39 pp | 5.83 % | -10.74 pp | 5.66 % | -10.91 pp | 52.28 % | +35.71 pp |
| Memory Throughput (%) | 15.75 % | 2.36 % | -13.39 pp | 2.34 % | -13.41 pp | 2.41 % | -13.34 pp | 28.38 % | +12.63 pp |
| DRAM Throughput (%) | 1.95 % | 2.36 % | +0.41 pp | 2.34 % | +0.39 pp | 2.41 % | +0.46 pp | 0.06 % | -1.89 pp |
| Achieved Occupancy (%) | 38.75 % | 25.51 % | -13.24 pp | 25.87 % | -12.88 pp | 31.93 % | -6.82 pp | 44.98 % | +6.23 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 54.60 % | 81.95 % | +27.35 pp | 70.64 % | +16.04 pp | 69.68 % | +15.08 pp | 44.29 % | -10.31 pp |
| IPC Active | 1.68 inst/cycle | 0.89 inst/cycle | -0.79 | 0.91 inst/cycle | -0.77 | 0.68 inst/cycle | -1.00 | 2.24 inst/cycle | +0.56 |


### Key Learnings

#### ✅ Plan A: `fix_grid_lambda_to_activate_existing_autotune_configs` (1.224×)

When a Triton kernel has a @triton.autotune decorator but the grid in the launcher is hardcoded using fixed constants (e.g., cdiv(xnumel, 128)) instead of META['XBLOCK'], the autotune configs are selected but the grid never adapts — every config launches with the same block count, making autotune effectively inert. Replacing the hardcoded grid tuple with a lambda using META['XBLOCK'] and META['YBLOCK'] activates the autotune so the grid actually varies per config. In this case the profiling metrics (elapsed cycles, occupancy, scheduler eligibility) did not improve as predicted — the speedup of 1.22x appears to come from a different mechanism (possibly reduced L2 sector count: 1092→1039) rather than the expected wave-count improvement, as waves_per_sm remained at 0.48 and achieved occupancy actually dropped from 30.67% to 25.51%.

> **Applicable when**: A Triton kernel has @triton.autotune but the launcher grid uses hardcoded tile constants instead of META['XBLOCK']/META['YBLOCK']; autotune is present but effectively inert because the grid size is invariant across configs.

> **Avoid when**: The grid is already a lambda using META; autotune is not present; the kernel is not wave-limited and grid size variation would not help throughput.

---

## Iteration 3

**Best so far**: 1.253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_yblock_from_autotune | 0.048× | 0.021× | ✓ |  |
| B | fix_static_grid_to_use_autotune_meta_lambda | 1.052× | 1.124× | ✓ |  |
| C | shared_memory_tiled_transpose_to_fix_uncoalesced_stores | 1.020× | 1.192× | ✓ |  |
| D | fix_grid_to_1d_and_coalesce_store_via_xblock1_autotune | 0.048× | 0.021× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_static_grid_t... | Δ | shared_memory_til... | Δ | fix_grid_to_1d_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3170 ns | 149020 ns | +145850 ns (+4600.9%) | 2820 ns | -350 ns (-11.0%) | 2660 ns | -510 ns (-16.1%) | 150140 ns | +146970 ns (+4636.3%) |
| SM Throughput (%) | 16.57 % | 53.78 % | +37.21 pp | 6.38 % | -10.19 pp | 6.23 % | -10.34 pp | 52.16 % | +35.59 pp |
| Memory Throughput (%) | 15.75 % | 29.87 % | +14.12 pp | 2.19 % | -13.56 pp | 2.33 % | -13.42 pp | 28.98 % | +13.23 pp |
| DRAM Throughput (%) | 1.95 % | 0.06 % | -1.89 pp | 2.19 % | +0.24 pp | 2.33 % | +0.38 pp | 0.06 % | -1.89 pp |
| Achieved Occupancy (%) | 38.75 % | 46.20 % | +7.45 pp | 26.46 % | -12.29 pp | 26.10 % | -12.65 pp | 46.55 % | +7.80 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 54.60 % | 42.69 % | -11.91 pp | 74.39 % | +19.79 pp | 71.06 % | +16.46 pp | 43.82 % | -10.78 pp |
| IPC Active | 1.68 inst/cycle | 2.23 inst/cycle | +0.55 | 0.98 inst/cycle | -0.70 | 0.98 inst/cycle | -0.70 | 2.25 inst/cycle | +0.57 |

---

## Iteration 4

**Best so far**: 1.253× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_expand_autotune_with_num_warps2_configs | 1.026× | 1.224× | ✓ |  |
| C | fix_grid_lambda_to_use_meta_xblock_yblock_corrected_kernel_function | 1.021× | 1.210× | ✓ |  |
| D | fix_grid_to_1d_persistent_matching_tile_loop_with_expanded_autotune | 0.048× | 0.021× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_t... | Δ | fix_grid_to_1d_pe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3170 ns | 2590 ns | -580 ns (-18.3%) | 2620 ns | -550 ns (-17.4%) | 148320 ns | +145150 ns (+4578.9%) |
| SM Throughput (%) | 16.57 % | 6.35 % | -10.22 pp | 5.37 % | -11.20 pp | 51.03 % | +34.46 pp |
| Memory Throughput (%) | 15.75 % | 2.37 % | -13.38 pp | 2.35 % | -13.40 pp | 29.39 % | +13.64 pp |
| DRAM Throughput (%) | 1.95 % | 2.37 % | +0.42 pp | 2.35 % | +0.40 pp | 0.06 % | -1.89 pp |
| Achieved Occupancy (%) | 38.75 % | 25.66 % | -13.09 pp | 24.55 % | -14.20 pp | 45.17 % | +6.42 pp |
| Registers/Thread | 28.00 register/thread | 16.00 register/thread | -12 | 16.00 register/thread | -12 | 16.00 register/thread | -12 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 |
| No Eligible (%) | 54.60 % | 69.98 % | +15.38 pp | 78.70 % | +24.10 pp | 42.64 % | -11.96 pp |
| IPC Active | 1.68 inst/cycle | 0.97 inst/cycle | -0.71 | 0.91 inst/cycle | -0.77 | 2.30 inst/cycle | +0.62 |

---

## Cumulative Progress

```
Baseline:  3170 ns  (1.000×)
Iter 1     2530 ns  (1.253×)  — fix_tile_sizing_and_autotune_for_small_xnumel

Final best: 2590 ns  (1.2239×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 1,  'YBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 4,  'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 8192
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +33,24 @@

     y3 = yindex
     y0 = (yindex % 64)
     y1 = yindex // 64
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask)
 
+    # Load tile: shape [YBLOCK, XBLOCK], coalesced (stride-1 in x2 dim)
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_first', other=0.0)
 
+    # Transpose tile to [XBLOCK, YBLOCK] so that the store iterates over y (stride-1) in the fast dim
+    tmp0_t = tl.trans(tmp0)
+
+    # After transpose: shape is [XBLOCK, YBLOCK]
+    # x2_t: shape [XBLOCK, 1], y0_t: shape [1, YBLOCK], y1_t: shape [1, YBLOCK]
+    x2_t = xoffset + tl.arange(0, XBLOCK)[:, None]
+    xmask_t = x2_t < xnumel
+    y_t = yoffset + tl.arange(0, YBLOCK)[None, :]
+    ymask_t = y_t < ynumel
+    y0_t = (y_t % 64)
+    y1_t = y_t // 64
+
+    # Store with coalesced access: y_t varies in the fast (column) dimension with stride-1 in y0
+    tl.store(out_ptr0 + (y0_t + 64*x2_t + 576*y1_t), tmp0_t, xmask_t & ymask_t)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +58,7 @@

     ynumel = 8192
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5[grid](arg_0, arg_1, 8192, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_5[grid](arg_0, arg_1, 8192, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
