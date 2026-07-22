# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_cat_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 257540 ns — **Best latency**: 221760 ns (**1.1613×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_reduce_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_cat_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_12_triton_poi_fused_cat_12/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (109760,1,1), Waves/SM: 69.29
- **Registers/thread**: 40
- **Duration**: 257540 ns
- **SM Throughput**: 61.90 % | **Memory**: 81.19 % | **DRAM**: 69.69 %
- **Achieved Occupancy**: 71.09 %
- **IPC Active**: 2.51 inst/cycle | **No Eligible**: 37.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_and_num_warps_for_strided_cat_kernel | — | 1.159× | ✓ |  |
| B | autotune_tile_sizes_reduce_register_pressure | — | 1.161× | ✓ | **✓ best** |
| C | add_autotune_vectorized_loads_larger_xblock | — | 0.596× | ✓ |  |
| D | add_autotune_reduce_register_pressure_tile_sizes | — | fail | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | add_autotune_vect... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 257540 ns | 222240 ns | -35300 ns (-13.7%) | 221760 ns | -35780 ns (-13.9%) | 432220 ns | +174680 ns (+67.8%) |
| SM Throughput (%) | 61.90 % | 79.61 % | +17.71 pp | 79.63 % | +17.73 pp | 86.39 % | +24.49 pp |
| Memory Throughput (%) | 81.19 % | 53.07 % | -28.12 pp | 53.09 % | -28.10 pp | 53.77 % | -27.42 pp |
| DRAM Throughput (%) | 69.69 % | 18.63 % | -51.06 pp | 18.70 % | -50.99 pp | 4.79 % | -64.90 pp |
| Achieved Occupancy (%) | 71.09 % | 52.57 % | -18.52 pp | 52.58 % | -18.51 pp | 71.20 % | +0.11 pp |
| Registers/Thread | 40.00 register/thread | 54.00 register/thread | +14 | 54.00 register/thread | +14 | 40.00 register/thread | +0 |
| Waves/SM | 69.29 | 92.39 | +23.10 | 92.39 | +23.10 | 138.59 | +69.30 |
| No Eligible (%) | 37.65 % | 22.46 % | -15.19 pp | 22.64 % | -15.01 pp | 13.07 % | -24.58 pp |
| IPC Active | 2.51 inst/cycle | 3.08 inst/cycle | +0.57 | 3.09 inst/cycle | +0.58 | 3.48 inst/cycle | +0.97 |

---

## Iteration 2

**Best so far**: 1.1613× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_yblock_reduce_warps_for_compute_bound_cat_kernel | — | 1.429× | ✓ |  |
| B | cap_register_usage_with_maxnreg_to_improve_occupancy | — | 1.163× | ✓ |  |
| C | fix_grid_lambda_and_retune_yblock_dominant_configs_for_strided_cat | — | 1.157× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_yblock_red... | Δ | cap_register_usag... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 257540 ns | 180290 ns | -77250 ns (-30.0%) | 221410 ns | -36130 ns (-14.0%) | 222660 ns | -34880 ns (-13.5%) |
| SM Throughput (%) | 61.90 % | 74.44 % | +12.54 pp | 79.45 % | +17.55 pp | 79.61 % | +17.71 pp |
| Memory Throughput (%) | 81.19 % | 66.26 % | -14.93 pp | 52.96 % | -28.23 pp | 53.07 % | -28.12 pp |
| DRAM Throughput (%) | 69.69 % | 4.30 % | -65.39 pp | 18.71 % | -50.98 pp | 18.62 % | -51.07 pp |
| Achieved Occupancy (%) | 71.09 % | 28.82 % | -42.27 pp | 52.56 % | -18.53 pp | 52.58 % | -18.51 pp |
| Registers/Thread | 40.00 register/thread | 96.00 register/thread | +56 | 54.00 register/thread | +14 | 54.00 register/thread | +14 |
| Waves/SM | 69.29 | 83.15 | +13.86 | 92.39 | +23.10 | 92.39 | +23.10 |
| No Eligible (%) | 37.65 % | 26.45 % | -11.20 pp | 22.65 % | -15.00 pp | 22.59 % | -15.06 pp |
| IPC Active | 2.51 inst/cycle | 2.94 inst/cycle | +0.43 | 3.10 inst/cycle | +0.59 | 3.09 inst/cycle | +0.58 |


### Key Learnings

#### ✅ Plan B: `cap_register_usage_with_maxnreg_to_improve_occupancy` (1.163×)

Adding maxnreg=40 and maxnreg=48 variants to a Triton @triton.autotune config list for a compute-bound kernel (ALU pipeline ~80%, 54 registers/thread, theoretical occupancy 56.25%) produced only a marginal 1.16x speedup. Critically, the profiling metrics after are nearly identical to before: registers/thread remains 54, theoretical occupancy stays at 56.25%, and Block Limit Registers stays at 9 — indicating the autotuner did NOT select any maxnreg-capped config as the winner, and the speedup likely came from the expanded tile-shape search space rather than register capping. When the kernel is already at a local optimum for tile sizes and the autotuner rejects the maxnreg configs (likely due to register spilling overhead), adding maxnreg variants to the config list provides negligible benefit beyond the tile-size search.

> **Applicable when**: Kernel is register-pressure limited (TheoreticalOccupancy rule triggered, Block Limit Registers < Block Limit Warps, registers/thread > 40) AND the kernel is not already at a compute-bound plateau. Best applied when the kernel has low waves/SM (<10) so occupancy improvement directly translates to more parallelism.

> **Avoid when**: Kernel already has high waves/SM (>50) where occupancy improvement has diminishing returns. Kernel is at a compute-bound plateau (ALU pipeline >78%) where more warps cannot hide latency further. The autotuner is likely to reject maxnreg configs if register spilling cost exceeds occupancy gain — in this case, profiling metrics will be unchanged post-optimization, indicating the strategy had no effect.

---

## Iteration 3

**Best so far**: 1.1613× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | expand_autotune_with_small_xblock_large_yblock_low_warps_to_cut_not_selected_stall | — | 1.422× | ✓ |  |
| C | expand_yblock_and_reduce_num_warps_to_cut_not_selected_cpi_stall | — | 1.427× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_w... | Δ | expand_yblock_and... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 257540 ns | 181120 ns | -76420 ns (-29.7%) | 180420 ns | -77120 ns (-29.9%) |
| SM Throughput (%) | 61.90 % | 74.45 % | +12.55 pp | 74.35 % | +12.45 pp |
| Memory Throughput (%) | 81.19 % | 66.27 % | -14.92 pp | 66.18 % | -15.01 pp |
| DRAM Throughput (%) | 69.69 % | 4.46 % | -65.23 pp | 4.48 % | -65.21 pp |
| Achieved Occupancy (%) | 71.09 % | 28.82 % | -42.27 pp | 28.84 % | -42.25 pp |
| Registers/Thread | 40.00 register/thread | 96.00 register/thread | +56 | 96.00 register/thread | +56 |
| Waves/SM | 69.29 | 83.15 | +13.86 | 83.15 | +13.86 |
| No Eligible (%) | 37.65 % | 26.47 % | -11.18 pp | 26.45 % | -11.20 pp |
| IPC Active | 2.51 inst/cycle | 2.94 inst/cycle | +0.43 | 2.94 inst/cycle | +0.43 |

---

## Cumulative Progress

```
Baseline:  257540 ns  (1.000×)
Iter 1     221760 ns  (1.161×)  — autotune_tile_sizes_reduce_register_pressure

Final best: 221760 ns  (1.1613×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 64},  num_warps=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 8,   'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},   num_warps=4),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_cat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 2240
@@ -24,45 +40,43 @@

     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1, 1], 32, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (x2 + 50176*(y0) + 1605632*y1), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (x2 + 50176*(y0) + 1605632*y1), mask=tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1, 1], 64, tl.int64)
     tmp8 = tmp0 < tmp7
     tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*x2 + 1605632*y1 + ((-32) + y0)), tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr1 + (32*x2 + 1605632*y1 + ((-32) + y0)), mask=tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp0 >= tmp7
     tmp12 = tl.full([1, 1], 96, tl.int64)
     tmp13 = tmp0 < tmp12
     tmp14 = tmp11 & tmp13
-    tmp15 = tl.load(in_ptr2 + (32*x2 + 1605632*y1 + ((-64) + y0)), tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp15 = tl.load(in_ptr2 + (32*x2 + 1605632*y1 + ((-64) + y0)), mask=tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp16 = tmp0 >= tmp12
     tmp17 = tl.full([1, 1], 128, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr3 + (32*x2 + 1605632*y1 + ((-96) + y0)), tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr3 + (32*x2 + 1605632*y1 + ((-96) + y0)), mask=tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1, 1], 160, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr4 + (32*x2 + 1605632*y1 + ((-128) + y0)), tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr4 + (32*x2 + 1605632*y1 + ((-128) + y0)), mask=tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1, 1], 192, tl.int64)
     tmp28 = tmp0 < tmp27
     tmp29 = tmp26 & tmp28
-    tmp30 = tl.load(in_ptr5 + (32*x2 + 1605632*y1 + ((-160) + y0)), tmp29 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp30 = tl.load(in_ptr5 + (32*x2 + 1605632*y1 + ((-160) + y0)), mask=tmp29 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp31 = tmp0 >= tmp27
     tmp32 = tl.full([1, 1], 224, tl.int64)
     tmp33 = tmp0 < tmp32
-    tmp34 = tl.load(in_ptr6 + (32*x2 + 1605632*y1 + ((-192) + y0)), tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
+    tmp34 = tl.load(in_ptr6 + (32*x2 + 1605632*y1 + ((-192) + y0)), mask=tmp31 & xmask & ymask, eviction_policy='evict_last', other=0.0)
     tmp35 = tl.where(tmp29, tmp30, tmp34)
     tmp36 = tl.where(tmp24, tmp25, tmp35)
     tmp37 = tl.where(tmp19, tmp20, tmp36)
     tmp38 = tl.where(tmp14, tmp15, tmp37)
     tmp39 = tl.where(tmp9, tmp10, tmp38)
     tmp40 = tl.where(tmp4, tmp5, tmp39)
-    tl.store(out_ptr0 + (x2 + 50176*y3), tmp40, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (x2 + 50176*y3), tmp40, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -76,7 +90,7 @@

     ynumel = 2240
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_cat_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 2240, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_cat_12[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 2240, 50176)
     return arg_7
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
