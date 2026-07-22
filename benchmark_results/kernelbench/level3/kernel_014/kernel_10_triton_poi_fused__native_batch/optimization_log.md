# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4000 ns — **Best latency**: 3040 ns (**1.3158×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`remove_unnecessary_debug_barrier_in_pointwise_kernel`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (24,1,1), Waves/SM: 0.02
- **Registers/thread**: 42
- **Duration**: 4000 ns
- **SM Throughput**: 0.97 % | **Memory**: 0.71 % | **DRAM**: 0.10 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.87 inst/cycle | **No Eligible**: 77.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_and_increase_tile_sizes | — | 1.262× | ✓ | **✓ best** |
| B | fix_2d_grid_launch_to_parallelize_y_dimension | — | 0.739× | ✓ |  |
| C | fix_grid_launch_and_add_autotune_for_yblock_vectorization | — | fail | ✓ |  |
| D | fix_grid_launch_add_autotune | — | 1.250× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ | fix_2d_grid_launc... | Δ | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3170 ns | -830 ns (-20.8%) | 5410 ns | +1410 ns (+35.2%) | 3200 ns | -800 ns (-20.0%) |
| SM Throughput (%) | 0.97 % | 0.88 % | -0.09 pp | 2.07 % | +1.10 pp | 0.90 % | -0.07 pp |
| Memory Throughput (%) | 0.71 % | 1.04 % | +0.33 pp | 4.75 % | +4.04 pp | 1.04 % | +0.33 pp |
| DRAM Throughput (%) | 0.10 % | 0.22 % | +0.12 pp | 1.57 % | +1.47 pp | 0.22 % | +0.12 pp |
| Achieved Occupancy (%) | 6.23 % | 6.19 % | -0.04 pp | 6.20 % | -0.03 pp | 6.20 % | -0.03 pp |
| Registers/Thread | 42.00 register/thread | 32.00 register/thread | -10 | 140.00 register/thread | +98 | 32.00 register/thread | -10 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.06 | +0.04 | 0.01 | -0.01 |
| No Eligible (%) | 77.80 % | 87.29 % | +9.49 pp | 79.94 % | +2.14 pp | 86.58 % | +8.78 pp |
| IPC Active | 0.87 inst/cycle | 0.49 inst/cycle | -0.38 | 0.77 inst/cycle | -0.10 | 0.46 inst/cycle | -0.41 |

---

## Iteration 2

**Best so far**: 1.2618× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_1d_grid_to_2d_grid_for_y_dimension_parallelism | — | 1.303× | ✓ | **✓ best** |
| B | fix_1d_grid_to_2d_lambda_preserving_existing_autotune | — | 1.274× | ✓ |  |
| C | reduce_register_pressure_via_small_yblock_configs_with_2d_grid | — | fail | ✓ |  |
| D | add_triton_autotune | — | 1.190× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_1d_grid_to_2d... | Δ | fix_1d_grid_to_2d... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3070 ns | -930 ns (-23.3%) | 3140 ns | -860 ns (-21.5%) | 3360 ns | -640 ns (-16.0%) |
| SM Throughput (%) | 0.97 % | 0.92 % | -0.05 pp | 0.90 % | -0.07 pp | 0.93 % | -0.04 pp |
| Memory Throughput (%) | 0.71 % | 1.08 % | +0.37 pp | 1.14 % | +0.43 pp | 0.98 % | +0.27 pp |
| DRAM Throughput (%) | 0.10 % | 0.23 % | +0.13 pp | 0.22 % | +0.12 pp | 0.21 % | +0.11 pp |
| Achieved Occupancy (%) | 6.23 % | 6.18 % | -0.05 pp | 6.19 % | -0.04 pp | 6.19 % | -0.04 pp |
| Registers/Thread | 42.00 register/thread | 32.00 register/thread | -10 | 32.00 register/thread | -10 | 32.00 register/thread | -10 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 77.80 % | 86.95 % | +9.15 pp | 87.13 % | +9.33 pp | 87.55 % | +9.75 pp |
| IPC Active | 0.87 inst/cycle | 0.49 inst/cycle | -0.38 | 0.49 inst/cycle | -0.38 | 0.45 inst/cycle | -0.42 |

---

## Iteration 3

**Best so far**: 1.3029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | remove_unnecessary_debug_barrier_in_pointwise_kernel | — | 1.316× | ✓ | **✓ best** |
| B | fix_grid_lambda_to_use_meta_xblock_yblock | — | fail | ✓ |  |
| C | flatten_y_to_1d_pid_with_full_xblock_in_registers | — | 1.127× | ✓ |  |
| D | add_triton_autotune | — | 1.290× | ✓ |  |


### Metric Comparison

| Metric | Baseline | remove_unnecessar... | Δ | flatten_y_to_1d_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3040 ns | -960 ns (-24.0%) | 3550 ns | -450 ns (-11.3%) | 3100 ns | -900 ns (-22.5%) |
| SM Throughput (%) | 0.97 % | 0.90 % | -0.07 pp | 0.74 % | -0.23 pp | 0.89 % | -0.08 pp |
| Memory Throughput (%) | 0.71 % | 1.09 % | +0.38 pp | 0.79 % | +0.08 pp | 1.06 % | +0.35 pp |
| DRAM Throughput (%) | 0.10 % | 0.23 % | +0.13 pp | 0.16 % | +0.06 pp | 0.22 % | +0.12 pp |
| Achieved Occupancy (%) | 6.23 % | 6.09 % | -0.14 pp | 6.22 % | -0.01 pp | 6.18 % | -0.05 pp |
| Registers/Thread | 42.00 register/thread | 32.00 register/thread | -10 | 32.00 register/thread | -10 | 32.00 register/thread | -10 |
| Waves/SM | 0.02 | 0.01 | -0.01 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 77.80 % | 88.45 % | +10.65 pp | 92.12 % | +14.32 pp | 88.05 % | +10.25 pp |
| IPC Active | 0.87 inst/cycle | 0.50 inst/cycle | -0.37 | 0.31 inst/cycle | -0.56 | 0.49 inst/cycle | -0.38 |


### Key Learnings

#### ✅ Plan A: `remove_unnecessary_debug_barrier_in_pointwise_kernel` (1.316×)

In a pointwise Triton kernel with no shared memory (static_smem=0), no inter-thread dependencies, and no reductions, a tl.debug_barrier() call before the final tl.store() adds a warp-level synchronization fence that contributes to issued-but-not-executed instruction overhead (22804 issued vs 22036 executed before removal). Removing the barrier eliminates this overhead, reduces issued instruction count, and allows the compiler backend to freely schedule the final ALU op and store without an artificial fence. The speedup is modest (1.32x) because the kernel remains severely launch-bound (24 blocks vs 132 SMs), but the barrier removal is always safe and correct for purely pointwise kernels with no shared state between threads.

> **Applicable when**: Triton pointwise kernel (no tl.shared memory allocations, static_smem=0) contains tl.debug_barrier() before a tl.store(); issued instructions significantly exceed executed instructions (gap > 500); IssueSlotUtilization NCU rule fires; kernel has no inter-thread data dependencies or reductions.

> **Avoid when**: Kernel uses tl.shared memory (static_smem > 0) where the barrier may guard a shared-memory write-then-read hazard; kernel performs reductions where barrier enforces ordering; kernel has explicit inter-warp synchronization patterns; the barrier was intentionally inserted for correctness in a fused kernel with shared state.

---

## Iteration 4

**Best so far**: 1.3158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_kernel_as_flat_1d_linear_index_over_full_element_space | — | 1.170× | ✓ |  |
| B | fix_grid_lambda_to_reference_meta_xblock_yblock_from_autotune | — | 1.180× | ✓ |  |
| C | fix_1d_grid_divisor_from_xnumel_to_ynumel_and_remove_debug_barrier | — | 1.227× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_kernel_as... | Δ | fix_grid_lambda_t... | Δ | fix_1d_grid_divis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3420 ns | -580 ns (-14.5%) | 3390 ns | -610 ns (-15.2%) | 3260 ns | -740 ns (-18.5%) |
| SM Throughput (%) | 0.97 % | 1.10 % | +0.13 pp | 0.73 % | -0.24 pp | 0.73 % | -0.24 pp |
| Memory Throughput (%) | 0.71 % | 1.07 % | +0.36 pp | 0.83 % | +0.12 pp | 0.85 % | +0.14 pp |
| DRAM Throughput (%) | 0.10 % | 0.36 % | +0.26 pp | 0.17 % | +0.07 pp | 0.18 % | +0.08 pp |
| Achieved Occupancy (%) | 6.23 % | 6.24 % | +0.01 pp | 6.22 % | -0.01 pp | 5.63 % | -0.60 pp |
| Registers/Thread | 42.00 register/thread | 34.00 register/thread | -8 | 32.00 register/thread | -10 | 32.00 register/thread | -10 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.01 | -0.01 | 0.01 | -0.01 |
| No Eligible (%) | 77.80 % | 88.93 % | +11.13 pp | 91.86 % | +14.06 pp | 90.76 % | +12.96 pp |
| IPC Active | 0.87 inst/cycle | 0.41 inst/cycle | -0.46 | 0.32 inst/cycle | -0.55 | 0.34 inst/cycle | -0.53 |

---

## Iteration 5

**Best so far**: 1.3158× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_formula_from_xnumel_to_ynumel_over_yblock | — | fail | ✓ |  |
| B | fix_grid_launch_to_use_ynumel_divided_by_yblock_lambda | — | 1.127× | ✓ |  |
| C | fix_grid_launch_to_cover_total_elements_with_flat_1d_index | — | 1.087× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_t... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4000 ns | 3550 ns | -450 ns (-11.3%) | 3680 ns | -320 ns (-8.0%) |
| SM Throughput (%) | 0.97 % | 1.08 % | +0.11 pp | 1.09 % | +0.12 pp |
| Memory Throughput (%) | 0.71 % | 0.95 % | +0.24 pp | 0.92 % | +0.21 pp |
| DRAM Throughput (%) | 0.10 % | 0.34 % | +0.24 pp | 0.33 % | +0.23 pp |
| Achieved Occupancy (%) | 6.23 % | 6.01 % | -0.22 pp | 6.24 % | +0.01 pp |
| Registers/Thread | 42.00 register/thread | 34.00 register/thread | -8 | 34.00 register/thread | -8 |
| Waves/SM | 0.02 | 0.02 | +0.00 | 0.02 | +0.00 |
| No Eligible (%) | 77.80 % | 88.04 % | +10.24 pp | 87.62 % | +9.82 pp |
| IPC Active | 0.87 inst/cycle | 0.45 inst/cycle | -0.42 | 0.42 inst/cycle | -0.45 |

---

## Cumulative Progress

```
Baseline:  4000 ns  (1.000×)
Iter 1     3170 ns  (1.262×)  — fix_grid_launch_and_increase_tile_sizes
Iter 2     3070 ns  (1.303×)  — fix_1d_grid_to_2d_grid_for_y_dimension_parallelism
Iter 3     3040 ns  (1.316×)  — remove_unnecessary_debug_barrier_in_pointwise_kernel

Final best: 3040 ns  (1.3158×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=8),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=8),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 501760
@@ -19,40 +37,40 @@

     y0 = (yindex % 50176)
     y1 = yindex // 50176
     y3 = yindex
-    tmp35 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
-    tmp37 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
-    tmp46 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
-    tmp48 = tl.load(in_ptr9 + (x2), xmask, eviction_policy='evict_last')
+    tmp35 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp37 = tl.load(in_ptr7 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp46 = tl.load(in_ptr8 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp48 = tl.load(in_ptr9 + (x2), mask=xmask, eviction_policy='evict_last')
     tmp0 = x2
     tmp1 = tl.full([1, 1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1, 1], 32, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), mask=tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1, 1], 64, tl.int64)
     tmp8 = tmp0 < tmp7
     tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), mask=tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp0 >= tmp7
     tmp12 = tl.full([1, 1], 96, tl.int64)
     tmp13 = tmp0 < tmp12
     tmp14 = tmp11 & tmp13
-    tmp15 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp15 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), mask=tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp16 = tmp0 >= tmp12
     tmp17 = tl.full([1, 1], 128, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr3 + (32*y3 + ((-96) + x2)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr3 + (32*y3 + ((-96) + x2)), mask=tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1, 1], 160, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), mask=tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1, 1], 192, tl.int64)
     tmp28 = tmp0 < tmp27
-    tmp29 = tl.load(in_ptr5 + (32*y3 + ((-160) + x2)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp29 = tl.load(in_ptr5 + (32*y3 + ((-160) + x2)), mask=tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp30 = tl.where(tmp24, tmp25, tmp29)
     tmp31 = tl.where(tmp19, tmp20, tmp30)
     tmp32 = tl.where(tmp14, tmp15, tmp31)
@@ -71,10 +89,7 @@

     tmp49 = tmp47 + tmp48
     tmp50 = tl.full([1, 1], 0, tl.int32)
     tmp51 = tl.maximum(tmp50, tmp49)
-    tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp51, xmask & ymask)
-
-
+    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp51, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -91,7 +106,7 @@

     ynumel = 501760
     xnumel = 192
     grid = (triton.cdiv(xnumel, 8),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 501760, 192, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 501760, 192)
     return arg_10
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
