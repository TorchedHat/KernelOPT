# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3780 ns — **Best latency**: 2910 ns (**1.2990×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`fix_launch_grid_to_cover_full_2d_iteration_space`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_4_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (12,1,1), Waves/SM: 0.01
- **Registers/thread**: 34
- **Duration**: 3780 ns
- **SM Throughput**: 0.41 % | **Memory**: 0.37 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.72 inst/cycle | **No Eligible**: 81.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_sizes_and_fix_2d_grid_launch | 0.714× | 0.596× | ✓ |  |
| B | fix_launch_grid_to_cover_full_2d_iteration_space | 1.016× | 1.299× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3780 ns | 6340 ns | +2560 ns (+67.7%) | 2910 ns | -870 ns (-23.0%) |
| SM Throughput (%) | 0.41 % | 1.06 % | +0.65 pp | 0.40 % | -0.01 pp |
| Memory Throughput (%) | 0.37 % | 3.83 % | +3.46 pp | 0.71 % | +0.34 pp |
| DRAM Throughput (%) | 0.07 % | 1.34 % | +1.27 pp | 0.14 % | +0.07 pp |
| Achieved Occupancy (%) | 6.23 % | 6.21 % | -0.02 pp | 6.18 % | -0.05 pp |
| Registers/Thread | 34.00 register/thread | 148.00 register/thread | +114 | 32.00 register/thread | -2 |
| Waves/SM | 0.01 | 0.03 | +0.02 | 0.01 | +0.00 |
| No Eligible (%) | 81.48 % | 82.26 % | +0.78 pp | 88.67 % | +7.19 pp |
| IPC Active | 0.72 inst/cycle | 0.71 inst/cycle | -0.01 | 0.38 inst/cycle | -0.34 |

---

## Cumulative Progress

```
Baseline:  3780 ns  (1.000×)
Iter 1     2910 ns  (1.299×)  — fix_launch_grid_to_cover_full_2d_iteration_space

Final best: 2910 ns  (1.2990×)
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
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 16},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 8},   num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 8,  'YBLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 32, 'YBLOCK': 32},  num_warps=4, num_stages=1),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 501760
@@ -19,25 +33,25 @@

     y0 = (yindex % 50176)
     y1 = yindex // 50176
     y3 = yindex
-    tmp17 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
-    tmp28 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
-    tmp30 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr3 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr4 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp28 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
+    tmp30 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
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
-    tmp14 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), mask=tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp15 = tl.where(tmp9, tmp10, tmp14)
     tmp16 = tl.where(tmp4, tmp5, tmp15)
     tmp18 = tmp16 - tmp17
@@ -54,9 +68,7 @@

     tmp32 = tl.full([1, 1], 0, tl.int32)
     tmp33 = tl.maximum(tmp32, tmp31)
     tl.debug_barrier()
-    tl.store(in_out_ptr0 + (x2 + 96*y3), tmp33, xmask & ymask)
-
-
+    tl.store(in_out_ptr0 + (x2 + 96*y3), tmp33, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -70,7 +82,7 @@

     ynumel = 501760
     xnumel = 96
     grid = (triton.cdiv(xnumel, 8),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 501760, 96, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_4[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 501760, 96)
     return arg_7
 
 def get_inputs():
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
