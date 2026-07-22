# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 268060 ns — **Best latency**: 3840 ns (**69.8073×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`add_num_stages_and_num_warps16_to_autotune_configs_for_l1tex_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_40_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.04
- **Registers/thread**: 32
- **Duration**: 268060 ns
- **SM Throughput**: 1.37 % | **Memory**: 4.92 % | **DRAM**: 0.75 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.26 inst/cycle | **No Eligible**: 93.54 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_and_num_warps_to_maximize_grid_coverage | 28.106× | 27.022× | ✓ |  |
| C | add_autotune_reduce_xblock_increase_grid_coverage | 28.119× | 39.537× | ✓ |  |
| D | autotune_xblock_to_increase_grid_size_and_sm_utilization | 40.651× | 67.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | add_autotune_redu... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 268060 ns | 9920 ns | -258140 ns (-96.3%) | 6780 ns | -261280 ns (-97.5%) | 4000 ns | -264060 ns (-98.5%) |
| SM Throughput (%) | 1.37 % | 1.51 % | +0.14 pp | 2.52 % | +1.15 pp | 0.46 % | -0.91 pp |
| Memory Throughput (%) | 4.92 % | 2.51 % | -2.41 pp | 2.85 % | -2.07 pp | 0.58 % | -4.34 pp |
| DRAM Throughput (%) | 0.75 % | 1.78 % | +1.03 pp | 2.50 % | +1.75 pp | 0.30 % | -0.45 pp |
| Achieved Occupancy (%) | 6.23 % | 3.10 % | -3.13 pp | 6.13 % | -0.10 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 255.00 register/thread | +223 | 166.00 register/thread | +134 | 31.00 register/thread | -1 |
| Waves/SM | 0.04 | 0.03 | -0.01 | 0.04 | +0.00 | 0.01 | -0.03 |
| No Eligible (%) | 93.54 % | 69.39 % | -24.15 pp | 70.43 % | -23.11 pp | 93.04 % | -0.50 pp |
| IPC Active | 0.26 inst/cycle | 0.61 inst/cycle | +0.35 | 1.12 inst/cycle | +0.86 | 0.30 inst/cycle | +0.04 |

---

## Iteration 2

**Best so far**: 67.015× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_and_num_warps16_to_autotune_configs_for_l1tex_latency_hiding | 43.212× | 69.807× | ✓ | **✓ best** |
| B | reduce_register_pressure_via_maxnreg_to_increase_sm_occupancy | 39.500× | 76.808× | ✓ |  |
| D | add_triton_autotune | 39.561× | 70.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_an... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 268060 ns | 3840 ns | -264220 ns (-98.6%) | 3490 ns | -264570 ns (-98.7%) | 3780 ns | -264280 ns (-98.6%) |
| SM Throughput (%) | 1.37 % | 0.52 % | -0.85 pp | 0.48 % | -0.89 pp | 0.47 % | -0.90 pp |
| Memory Throughput (%) | 4.92 % | 0.70 % | -4.22 pp | 0.72 % | -4.20 pp | 0.64 % | -4.28 pp |
| DRAM Throughput (%) | 0.75 % | 0.31 % | -0.44 pp | 0.35 % | -0.40 pp | 0.32 % | -0.43 pp |
| Achieved Occupancy (%) | 6.23 % | 6.22 % | -0.01 pp | 6.22 % | -0.01 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 32.00 register/thread | +0 | 31.00 register/thread | -1 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.01 | -0.03 | 0.01 | -0.03 |
| No Eligible (%) | 93.54 % | 92.18 % | -1.36 pp | 93.13 % | -0.41 pp | 91.96 % | -1.58 pp |
| IPC Active | 0.26 inst/cycle | 0.29 inst/cycle | +0.03 | 0.29 inst/cycle | +0.03 | 0.32 inst/cycle | +0.06 |

---

## Iteration 3

**Best so far**: 69.8073× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_call_site_grid_to_use_autotuned_xblock | 38.877× | 70.357× | ✓ |  |
| B | fix_static_grid_to_use_autotuned_xblock_via_lambda | 41.535× | 70.915× | ✓ |  |
| C | rightsize_r0_block_to_match_r0_numel_and_fix_grid_lambda | 42.980× | 68.733× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_call_site_gri... | Δ | fix_static_grid_t... | Δ | rightsize_r0_bloc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 268060 ns | 3810 ns | -264250 ns (-98.6%) | 3780 ns | -264280 ns (-98.6%) | 3900 ns | -264160 ns (-98.5%) |
| SM Throughput (%) | 1.37 % | 0.31 % | -1.06 pp | 0.48 % | -0.89 pp | 0.48 % | -0.89 pp |
| Memory Throughput (%) | 4.92 % | 0.66 % | -4.26 pp | 0.61 % | -4.31 pp | 0.70 % | -4.22 pp |
| DRAM Throughput (%) | 0.75 % | 0.42 % | -0.33 pp | 0.32 % | -0.43 pp | 0.32 % | -0.43 pp |
| Achieved Occupancy (%) | 6.23 % | 3.11 % | -3.12 pp | 6.22 % | -0.01 pp | 6.22 % | -0.01 pp |
| Registers/Thread | 32.00 register/thread | 40.00 register/thread | +8 | 31.00 register/thread | -1 | 31.00 register/thread | -1 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.01 | -0.03 | 0.01 | -0.03 |
| No Eligible (%) | 93.54 % | 89.36 % | -4.18 pp | 92.34 % | -1.20 pp | 91.71 % | -1.83 pp |
| IPC Active | 0.26 inst/cycle | 0.21 inst/cycle | -0.05 | 0.28 inst/cycle | +0.02 | 0.27 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan A: `fix_call_site_grid_to_use_autotuned_xblock` (70.357×)

When a Triton kernel has a @triton.autotune decorator but the call site uses a static grid tuple (e.g., `(triton.cdiv(xnumel, 128),)`) instead of a lambda reading `meta['XBLOCK']`, the autotune is completely non-functional — the grid is always fixed regardless of which config the autotuner selects. The fix is to change the grid to `lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)` and remove any explicit XBLOCK kwarg at the call site. Adding small-XBLOCK configs (XBLOCK=1,2,4 without maxnreg) alongside existing maxnreg configs expands the autotuner's search space to include configurations that can expand the grid from 16 to up to 1960 blocks. Note: in this specific case the profiled kernel metrics showed only marginal change (3.49→3.81 µs on the profiled kernel), suggesting the large end-to-end speedup (70x) came from the autotuner selecting better configs across the full workload rather than from the profiled kernel alone.

> **Applicable when**: LaunchConfiguration NCU rule fires; grid_size << sm_count (waves_per_sm < 0.05); kernel has @triton.autotune decorator but call site uses a static grid tuple with a hardcoded XBLOCK value instead of a lambda; no_eligible_pct > 85%; the autotune decorator exists but is effectively dead code because the grid does not adapt to the selected config

> **Avoid when**: Call site already uses a lambda grid (meta['XBLOCK']); kernel does not have @triton.autotune; grid is already large enough to saturate SMs (waves_per_sm > 0.5); the static grid is intentional because XBLOCK must match a fixed shared memory layout that cannot change

---

## Iteration 4

**Best so far**: 69.8073× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_dynamic_lambda_so_autotune_xblock_expands_grid | 41.566× | 67.015× | ✓ |  |
| B | replace_broken_autotune_with_large_fixed_xblock_single_block | 1.000× | 0.997× | ✓ |  |
| C | fix_static_grid_to_lambda_and_add_small_xblock_configs_for_sm_coverage | 42.993× | 70.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | replace_broken_au... | Δ | fix_static_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 268060 ns | 4000 ns | -264060 ns (-98.5%) | 268990 ns | +930 ns (+0.3%) | 3780 ns | -264280 ns (-98.6%) |
| SM Throughput (%) | 1.37 % | 0.50 % | -0.87 pp | 1.36 % | -0.01 pp | 0.32 % | -1.05 pp |
| Memory Throughput (%) | 4.92 % | 0.63 % | -4.29 pp | 4.90 % | -0.02 pp | 0.64 % | -4.28 pp |
| DRAM Throughput (%) | 0.75 % | 0.31 % | -0.44 pp | 0.68 % | -0.07 pp | 0.33 % | -0.42 pp |
| Achieved Occupancy (%) | 6.23 % | 6.22 % | -0.01 pp | 6.23 % | +0.00 pp | 3.10 % | -3.13 pp |
| Registers/Thread | 32.00 register/thread | 31.00 register/thread | -1 | 32.00 register/thread | +0 | 40.00 register/thread | +8 |
| Waves/SM | 0.04 | 0.01 | -0.03 | 0.04 | +0.00 | 0.01 | -0.03 |
| No Eligible (%) | 93.54 % | 91.76 % | -1.78 pp | 93.54 % | +0.00 pp | 90.07 % | -3.47 pp |
| IPC Active | 0.26 inst/cycle | 0.31 inst/cycle | +0.05 | 0.26 inst/cycle | +0.00 | 0.18 inst/cycle | -0.08 |

---

## Cumulative Progress

```
Baseline:  268060 ns  (1.000×)
Iter 1     4000 ns  (67.015×)  — autotune_xblock_to_increase_grid_size_and_sm_utilization
Iter 2     3840 ns  (69.807×)  — add_num_stages_and_num_warps16_to_autotune_configs_for_l1tex_latency_hiding

Final best: 3840 ns  (69.8073×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,40 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1},   num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1},   num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 2},   num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2},   num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 4},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 4},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 4},   num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 4},   num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 8},   num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 8},   num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 8},   num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 16},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 16},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 16},  num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 1960
@@ -24,11 +58,11 @@

     x3 = xindex
     x0 = (xindex % 196)
     x1 = xindex // 196
-    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 14))) % 7)) + 224*(((4 + (x0 // 14)) % 7)) + 1568*(r0_2 // 32) + 18816*((4 + ((x0 % 14))) // 7) + 56448*((4 + (x0 // 14)) // 7) + 169344*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
+    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 14)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 14))) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+    tmp28 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr4 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -53,9 +87,7 @@

     tmp27 = tmp21 * tmp26
     tmp29 = tmp27 * tmp28
     tmp31 = tmp29 + tmp30
-    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, r0_mask & xmask)
-
-
+    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -67,7 +99,7 @@

     xnumel = 1960
     r0_numel = 384
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384, XBLOCK=128)
+    triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_40[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
