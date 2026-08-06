# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4510 ns — **Best latency**: 3070 ns (**1.4691×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_grid_launch_to_parallelize_y_dimension`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_8_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (20,1,1), Waves/SM: 0.01
- **Registers/thread**: 40
- **Duration**: 4510 ns
- **SM Throughput**: 0.65 % | **Memory**: 0.52 % | **DRAM**: 0.08 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.76 inst/cycle | **No Eligible**: 80.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_and_increase_tile_sizes | 0.807× | 0.658× | ✓ |  |
| B | fix_grid_launch_and_autotune_tile_sizes | 1.009× | 1.216× | ✓ |  |
| C | fix_grid_dimensionality_and_vectorize_loads | 0.908× | 0.952× | ✓ |  |
| D | fix_grid_launch_to_parallelize_y_dimension | 1.122× | 1.469× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ | fix_grid_launch_a... | Δ | fix_grid_dimensio... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4510 ns | 6850 ns | +2340 ns (+51.9%) | 3710 ns | -800 ns (-17.7%) | 4740 ns | +230 ns (+5.1%) | 3070 ns | -1440 ns (-31.9%) |
| SM Throughput (%) | 0.65 % | 2.32 % | +1.67 pp | 1.08 % | +0.43 pp | 1.67 % | +1.02 pp | 0.69 % | +0.04 pp |
| Memory Throughput (%) | 0.52 % | 4.10 % | +3.58 pp | 0.50 % | -0.02 pp | 4.56 % | +4.04 pp | 0.87 % | +0.35 pp |
| DRAM Throughput (%) | 0.08 % | 1.07 % | +0.99 pp | 0.18 % | +0.10 pp | 1.49 % | +1.41 pp | 0.19 % | +0.11 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.22 % | -0.01 pp | 6.19 % | -0.04 pp | 6.18 % | -0.05 pp |
| Registers/Thread | 40.00 register/thread | 255.00 register/thread | +215 | 48.00 register/thread | +8 | 135.00 register/thread | +95 | 32.00 register/thread | -8 |
| Waves/SM | 0.01 | 0.08 | +0.07 | 0.02 | +0.01 | 0.05 | +0.04 | 0.01 | +0.00 |
| No Eligible (%) | 80.72 % | 73.84 % | -6.88 pp | 75.57 % | -5.15 pp | 81.06 % | +0.34 pp | 87.63 % | +6.91 pp |
| IPC Active | 0.76 inst/cycle | 1.05 inst/cycle | +0.29 | 0.94 inst/cycle | +0.18 | 0.74 inst/cycle | -0.02 | 0.48 inst/cycle | -0.28 |

---

## Iteration 2

**Best so far**: 1.4691× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_1d_grid_to_use_autotune_2d_lambda | 1.073× | 1.498× | ✓ |  |
| B | fix_static_grid_to_2d_lambda_with_existing_autotune_configs | 0.974× | 1.270× | ✓ |  |
| C | reduce_register_pressure_with_small_tile_autotune_and_correct_2d_grid | 1.054× | 1.383× | ✓ |  |
| D | fix_register_spill_via_small_tiles_and_correct_2d_grid_lambda | 1.112× | 1.622× | ✓ |  |
| E | add_triton_autotune | 0.751× | 0.629× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_1d_grid_to_us... | Δ | fix_static_grid_t... | Δ | reduce_register_p... | Δ | fix_register_spil... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4510 ns | 3010 ns | -1500 ns (-33.3%) | 3550 ns | -960 ns (-21.3%) | 3260 ns | -1250 ns (-27.7%) | 2780 ns | -1730 ns (-38.4%) | 7170 ns | +2660 ns (+59.0%) |
| SM Throughput (%) | 0.65 % | 0.69 % | +0.04 pp | 1.11 % | +0.46 pp | 0.34 % | -0.31 pp | 0.65 % | +0.00 pp | 1.06 % | +0.41 pp |
| Memory Throughput (%) | 0.52 % | 0.90 % | +0.38 pp | 0.49 % | -0.03 pp | 0.56 % | +0.04 pp | 0.64 % | +0.12 pp | 0.81 % | +0.29 pp |
| DRAM Throughput (%) | 0.08 % | 0.20 % | +0.12 pp | 0.19 % | +0.11 pp | 0.12 % | +0.04 pp | 0.14 % | +0.06 pp | 0.11 % | +0.03 pp |
| Achieved Occupancy (%) | 6.23 % | 6.18 % | -0.05 pp | 6.22 % | -0.01 pp | 3.12 % | -3.11 pp | 6.22 % | -0.01 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 48.00 register/thread | +8 | 32.00 register/thread | -8 | 26.00 register/thread | -14 | 72.00 register/thread | +32 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.02 | +0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.02 | +0.01 |
| No Eligible (%) | 80.72 % | 86.81 % | +6.09 pp | 75.30 % | -5.42 pp | 87.17 % | +6.45 pp | 88.61 % | +7.89 pp | 73.85 % | -6.87 pp |
| IPC Active | 0.76 inst/cycle | 0.48 inst/cycle | -0.28 | 0.95 inst/cycle | +0.19 | 0.26 inst/cycle | -0.50 | 0.48 inst/cycle | -0.28 | 0.98 inst/cycle | +0.22 |

---

## Iteration 3

**Best so far**: 1.4691× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_1d_grid_to_3d_lambda_with_register_safe_autotune_configs | 1.105× | 1.484× | ✓ |  |
| B | fix_1d_grid_to_3d_lambda_with_large_yblock_for_cache_reuse | 1.074× | 1.409× | ✓ |  |
| C | fix_static_1d_grid_to_2d_lambda_covering_y_dimension | 1.071× | 1.455× | ✓ |  |
| D | fix_static_1d_grid_to_2d_lambda_with_small_tile_configs | 1.115× | 1.367× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_1d_gri... | Δ | fix_1d_grid_to_3d... | Δ | fix_static_1d_gri... | Δ | fix_static_1d_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4510 ns | 3040 ns | -1470 ns (-32.6%) | 3200 ns | -1310 ns (-29.0%) | 3100 ns | -1410 ns (-31.3%) | 3300 ns | -1210 ns (-26.8%) |
| SM Throughput (%) | 0.65 % | 0.38 % | -0.27 pp | 0.73 % | +0.08 pp | 0.35 % | -0.30 pp | 0.38 % | -0.27 pp |
| Memory Throughput (%) | 0.52 % | 0.59 % | +0.07 pp | 0.83 % | +0.31 pp | 0.59 % | +0.07 pp | 0.55 % | +0.03 pp |
| DRAM Throughput (%) | 0.08 % | 0.13 % | +0.05 pp | 0.19 % | +0.11 pp | 0.12 % | +0.04 pp | 0.12 % | +0.04 pp |
| Achieved Occupancy (%) | 6.23 % | 3.12 % | -3.11 pp | 6.19 % | -0.04 pp | 3.12 % | -3.11 pp | 3.12 % | -3.11 pp |
| Registers/Thread | 40.00 register/thread | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 32.00 register/thread | -8 | 32.00 register/thread | -8 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 80.72 % | 86.52 % | +5.80 pp | 87.71 % | +6.99 pp | 86.30 % | +5.58 pp | 86.58 % | +5.86 pp |
| IPC Active | 0.76 inst/cycle | 0.22 inst/cycle | -0.54 | 0.42 inst/cycle | -0.34 | 0.25 inst/cycle | -0.51 | 0.23 inst/cycle | -0.53 |

---

## Cumulative Progress

```
Baseline:  4510 ns  (1.000×)
Iter 1     3070 ns  (1.469×)  — fix_grid_launch_to_parallelize_y_dimension

Final best: 3070 ns  (1.4691×)
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
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64}, num_warps=8),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 16}, num_warps=4),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 501760
@@ -19,35 +33,35 @@

     y0 = (yindex % 50176)
     y1 = yindex // 50176
     y3 = yindex
-    tmp29 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
-    tmp31 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
-    tmp40 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
-    tmp42 = tl.load(in_ptr8 + (x2), xmask, eviction_policy='evict_last')
+    tmp29 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp31 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp40 = tl.load(in_ptr7 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp42 = tl.load(in_ptr8 + (x2), mask=xmask, eviction_policy='evict_last')
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
-    tmp24 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp24 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), mask=tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp25 = tl.where(tmp19, tmp20, tmp24)
     tmp26 = tl.where(tmp14, tmp15, tmp25)
     tmp27 = tl.where(tmp9, tmp10, tmp26)
@@ -66,9 +80,7 @@

     tmp44 = tl.full([1, 1], 0, tl.int32)
     tmp45 = tl.maximum(tmp44, tmp43)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x2 + 160*y3), tmp45, xmask & ymask)
-
-
+    tl.store(in_out_ptr0 + (x2 + 160*y3), tmp45, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -84,7 +96,7 @@

     ynumel = 501760
     xnumel = 160
     grid = (triton.cdiv(xnumel, 8),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 501760, 160, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 501760, 160)
     return arg_9
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
