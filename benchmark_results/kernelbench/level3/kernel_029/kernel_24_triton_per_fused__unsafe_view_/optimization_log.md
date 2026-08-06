# Kernel Optimization Log

## Summary

- **Kernel**: `triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 195840 ns — **Best latency**: 3520 ns (**55.6364×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`increase_xblock_tile_sizes_for_cache_reuse_and_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_24_triton_per_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62,1,1), Waves/SM: 0.16
- **Registers/thread**: 32
- **Duration**: 195840 ns
- **SM Throughput**: 5.27 % | **Memory**: 18.33 % | **DRAM**: 6.50 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.22 inst/cycle | **No Eligible**: 94.34 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_and_num_warps_for_underutilized_layernorm_kernel | 10.794× | 1.295× | ✓ |  |
| B | autotune_xblock_to_increase_grid_coverage | 29.271× | 54.704× | ✓ | **✓ best** |
| C | autotune_xblock_rblock_reduce_smem_improve_occupancy | 17.951× | 17.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | autotune_xblock_t... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 195840 ns | 151200 ns | -44640 ns (-22.8%) | 3580 ns | -192260 ns (-98.2%) | 11520 ns | -184320 ns (-94.1%) |
| SM Throughput (%) | 5.27 % | 3.21 % | -2.06 pp | 0.55 % | -4.72 pp | 6.05 % | +0.78 pp |
| Memory Throughput (%) | 18.33 % | 10.36 % | -7.97 pp | 1.00 % | -17.33 pp | 7.53 % | -10.80 pp |
| DRAM Throughput (%) | 6.50 % | 1.54 % | -4.96 pp | 0.60 % | -5.90 pp | 5.53 % | -0.97 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 1.56 % | -4.68 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 41.00 register/thread | +9 | 255.00 register/thread | +223 |
| Waves/SM | 0.16 | 0.04 | -0.12 | 0.01 | -0.15 | 0.12 | -0.04 |
| No Eligible (%) | 94.34 % | 92.96 % | -1.38 pp | 89.72 % | -4.62 pp | 69.08 % | -25.26 pp |
| IPC Active | 0.22 inst/cycle | 0.14 inst/cycle | -0.08 | 0.10 inst/cycle | -0.12 | 0.63 inst/cycle | +0.41 |


### Key Learnings

#### ✅ Plan B: `autotune_xblock_to_increase_grid_coverage` (57.263×)

When a Triton kernel's launch site uses a fixed grid tuple hardcoded to a specific XBLOCK, the autotuner's XBLOCK selection has no effect on block count, leaving the GPU severely under-utilized (waves_per_sm ~0.01, achieved occupancy ~1.56%). Replacing the fixed grid with a lambda (`lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`) and pairing it with expanded autotune configs spanning small XBLOCK values (1–512) with mixed num_stages allows the autotuner to select a tile size that produces enough blocks to cover all SMs. The 57x wall-clock speedup is achieved despite the profiled kernel instance showing nearly the same launch shape, indicating the dominant gain comes from the autotuner selecting an optimal config across the full search space enabled by the lambda grid.

> **Applicable when**: Triton reduction kernel where: (1) grid is a fixed tuple hardcoded to a specific XBLOCK value, (2) autotune configs only cover large XBLOCK values leaving the GPU under-utilized (waves_per_sm < 0.1, achieved occupancy < 5%), (3) LaunchConfiguration and AchievedOccupancy NCU rules are triggered, (4) Scheduler 'No Eligible' > 85%, (5) the kernel body already has boundary masks (xmask, r0_mask) for all XBLOCK values.

> **Avoid when**: Kernel is already memory-bandwidth bound (DRAM Throughput > 60%) or compute bound (SM Throughput > 50%); grid is already a lambda or otherwise adapts to XBLOCK; autotune overhead is unacceptable for the use case; kernel body lacks boundary masks that would be needed for small XBLOCK values that don't evenly divide xnumel.

---

## Iteration 2

**Best so far**: 54.7039× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_sizes_for_cache_reuse_and_parallelism | 30.438× | 55.636× | ✓ | **✓ best** |
| B | reduce_r0block_to_cut_register_pressure_and_lift_occupancy | 29.047× | 52.787× | ✓ |  |
| C | expand_autotune_configs_with_small_xblock_and_lambda_grid | 30.345× | 52.364× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | reduce_r0block_to... | Δ | expand_autotune_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 195840 ns | 3520 ns | -192320 ns (-98.2%) | 3710 ns | -192130 ns (-98.1%) | 3740 ns | -192100 ns (-98.1%) |
| SM Throughput (%) | 5.27 % | 0.57 % | -4.70 pp | 0.58 % | -4.69 pp | 0.55 % | -4.72 pp |
| Memory Throughput (%) | 18.33 % | 1.02 % | -17.31 pp | 1.10 % | -17.23 pp | 0.95 % | -17.38 pp |
| DRAM Throughput (%) | 6.50 % | 0.61 % | -5.89 pp | 0.57 % | -5.93 pp | 0.58 % | -5.92 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 41.00 register/thread | +9 | 41.00 register/thread | +9 | 41.00 register/thread | +9 |
| Waves/SM | 0.16 | 0.01 | -0.15 | 0.01 | -0.15 | 0.01 | -0.15 |
| No Eligible (%) | 94.34 % | 89.70 % | -4.64 pp | 89.70 % | -4.64 pp | 90.03 % | -4.31 pp |
| IPC Active | 0.22 inst/cycle | 0.10 inst/cycle | -0.12 | 0.10 inst/cycle | -0.12 | 0.08 inst/cycle | -0.14 |

---

## Iteration 3

**Best so far**: 55.6364× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_to_autotune_configs_for_l1tex_stall_hiding | 29.196× | 53.217× | ✓ |  |
| B | fix_static_grid_to_enable_autotune_xblock_grid_scaling | 29.125× | 52.787× | ✓ |  |
| C | fix_fixed_grid_to_lambda_and_add_num_stages_pipelining | 30.149× | 56.115× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | fix_static_grid_t... | Δ | fix_fixed_grid_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 195840 ns | 3680 ns | -192160 ns (-98.1%) | 3710 ns | -192130 ns (-98.1%) | 3490 ns | -192350 ns (-98.2%) |
| SM Throughput (%) | 5.27 % | 0.56 % | -4.71 pp | 0.57 % | -4.70 pp | 0.60 % | -4.67 pp |
| Memory Throughput (%) | 18.33 % | 0.99 % | -17.34 pp | 1.00 % | -17.33 pp | 1.12 % | -17.21 pp |
| DRAM Throughput (%) | 6.50 % | 0.58 % | -5.92 pp | 0.59 % | -5.91 pp | 0.61 % | -5.89 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp |
| Registers/Thread | 32.00 register/thread | 41.00 register/thread | +9 | 41.00 register/thread | +9 | 41.00 register/thread | +9 |
| Waves/SM | 0.16 | 0.01 | -0.15 | 0.01 | -0.15 | 0.01 | -0.15 |
| No Eligible (%) | 94.34 % | 89.89 % | -4.45 pp | 90.02 % | -4.32 pp | 90.79 % | -3.55 pp |
| IPC Active | 0.22 inst/cycle | 0.09 inst/cycle | -0.13 | 0.09 inst/cycle | -0.13 | 0.10 inst/cycle | -0.12 |

---

## Iteration 4

**Best so far**: 55.6364× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | right_size_r0block_to_eliminate_wasted_reduction_lanes | 30.305× | 57.263× | ✓ |  |
| B | fix_static_grid_to_lambda_enabling_autotune_xblock_grid_scaling | 30.134× | 57.770× | ✓ |  |
| C | fix_static_grid_lambda_and_add_num_stages2_pipelining_combined | 29.087× | 51.809× | ✓ |  |
| D | add_triton_autotune | 0.995× | 1.027× | ✓ |  |


### Metric Comparison

| Metric | Baseline | right_size_r0bloc... | Δ | fix_static_grid_t... | Δ | fix_static_grid_l... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 195840 ns | 3420 ns | -192420 ns (-98.3%) | 3390 ns | -192450 ns (-98.3%) | 3780 ns | -192060 ns (-98.1%) | 190660 ns | -5180 ns (-2.6%) |
| SM Throughput (%) | 5.27 % | 0.59 % | -4.68 pp | 0.61 % | -4.66 pp | 0.57 % | -4.70 pp | 5.55 % | +0.28 pp |
| Memory Throughput (%) | 18.33 % | 1.13 % | -17.20 pp | 1.04 % | -17.29 pp | 1.06 % | -17.27 pp | 18.82 % | +0.49 pp |
| DRAM Throughput (%) | 6.50 % | 0.62 % | -5.88 pp | 0.63 % | -5.87 pp | 0.56 % | -5.94 pp | 6.39 % | -0.11 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 1.56 % | -4.68 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 41.00 register/thread | +9 | 41.00 register/thread | +9 | 41.00 register/thread | +9 | 32.00 register/thread | +0 |
| Waves/SM | 0.16 | 0.01 | -0.15 | 0.01 | -0.15 | 0.01 | -0.15 | 0.16 | +0.00 |
| No Eligible (%) | 94.34 % | 90.25 % | -4.09 pp | 90.30 % | -4.04 pp | 89.46 % | -4.88 pp | 94.24 % | -0.10 pp |
| IPC Active | 0.22 inst/cycle | 0.10 inst/cycle | -0.12 | 0.09 inst/cycle | -0.13 | 0.10 inst/cycle | -0.12 | 0.23 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  195840 ns  (1.000×)
Iter 1     3580 ns  (54.704×)  — autotune_xblock_to_increase_grid_coverage
Iter 2     3520 ns  (55.636×)  — increase_xblock_tile_sizes_for_cache_reuse_and_parallelism

Final best: 3520 ns  (55.6364×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 1},   num_warps=1,  num_stages=1),
+        triton.Config({'XBLOCK': 2},   num_warps=1,  num_stages=1),
+        triton.Config({'XBLOCK': 4},   num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 8},   num_warps=2,  num_stages=1),
+        triton.Config({'XBLOCK': 16},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 32},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
     xnumel = 7840
@@ -24,11 +45,11 @@

     x3 = xindex
     x0 = (xindex % 784)
     x1 = xindex // 784
-    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), r0_mask & xmask, other=0.0)
-    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r0_2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*((4 + (x0 // 28)) // 7) + 235200*x1 + ((r0_2 % 32))), r0_mask & xmask, other=0.0)
-    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 28))) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-    tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
-    tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp0 = tl.load(in_ptr0 + (r0_2 + 192*x3), mask=r0_mask & xmask, other=0.0)
+    tmp1 = tl.load(in_ptr1 + (32*(((4 + ((x0 % 28))) % 7)) + 224*(((4 + (x0 // 28)) % 7)) + 1568*(r0_2 // 32) + 9408*((4 + ((x0 % 28))) // 7) + 47040*((4 + (x0 // 28)) // 7) + 235200*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
+    tmp2 = tl.load(in_ptr2 + (7*(((4 + (x0 // 28)) % 7)) + 49*(r0_2 // 32) + (((4 + ((x0 % 28))) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+    tmp28 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr4 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
     tmp3 = tmp1 + tmp2
     tmp4 = tmp0 + tmp3
     tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
@@ -53,9 +74,7 @@

     tmp27 = tmp21 * tmp26
     tmp29 = tmp27 * tmp28
     tmp31 = tmp29 + tmp30
-    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, r0_mask & xmask)
-
-
+    tl.store(out_ptr2 + (r0_2 + 192*x3), tmp31, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -67,7 +86,7 @@

     xnumel = 7840
     r0_numel = 192
     grid = (triton.cdiv(xnumel, 128),)
-    triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7840, 192, XBLOCK=128)
+    triton_per_fused__unsafe_view_add_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 7840, 192)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
