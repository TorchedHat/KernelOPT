# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_max_pool2d_with_indices_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2900000 ns — **Best latency**: 1170000 ns (**2.4786×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_streaming_load_eviction_policy_and_extend_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_max_pool2d_with_indices_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_2_triton_poi_fused_convolution_m/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (235200,1,1), Waves/SM: 356.36
- **Registers/thread**: 94
- **Duration**: 2900000 ns
- **SM Throughput**: 24.88 % | **Memory**: 70.12 % | **DRAM**: 33.70 %
- **Achieved Occupancy**: 30.35 %
- **IPC Active**: 0.99 inst/cycle | **No Eligible**: 75.08 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguous_hints | 1.064× | 0.539× | ✓ |  |
| C | fix_uncoalesced_stores_via_tile_rebalancing_and_autotune | 3.210× | 2.589× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_fix_uncoalesced_stores | 3.207× | 2.544× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | fix_uncoalesced_s... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2900000 ns | 5380000 ns | +2480000 ns (+85.5%) | 1120000 ns | -1780000 ns (-61.4%) | 1140000 ns | -1760000 ns (-60.7%) |
| SM Throughput (%) | 24.88 % | 13.78 % | -11.10 pp | 69.33 % | +44.45 pp | 70.59 % | +45.71 pp |
| Memory Throughput (%) | 70.12 % | 36.71 % | -33.41 pp | 66.64 % | -3.48 pp | 67.22 % | -2.90 pp |
| DRAM Throughput (%) | 33.70 % | 19.36 % | -14.34 pp | 10.55 % | -23.15 pp | 21.06 % | -12.64 pp |
| Achieved Occupancy (%) | 30.35 % | 30.34 % | -0.01 pp | 60.89 % | +30.54 pp | 60.84 % | +30.49 pp |
| Registers/Thread | 94.00 register/thread | 92.00 register/thread | -2 | 44.00 register/thread | -50 | 44.00 register/thread | -50 |
| Waves/SM | 356.36 | 356.36 | +0.00 | 178.18 | -178.18 | 178.18 | -178.18 |
| No Eligible (%) | 75.08 % | 86.18 % | +11.10 pp | 30.60 % | -44.48 pp | 28.89 % | -46.19 pp |
| IPC Active | 0.99 inst/cycle | 0.55 inst/cycle | -0.44 | 2.78 inst/cycle | +1.79 | 2.84 inst/cycle | +1.85 |

---

## Iteration 2

**Best so far**: 2.5893× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_streaming_load_eviction_policy_and_extend_autotune_configs | 13.782× | 2.479× | ✓ | **✓ best** |
| B | streaming_eviction_policy_and_pipelined_loads | 2.745× | 2.197× | ✓ |  |
| C | autotune_large_yblock_small_xblock_to_fix_register_pressure_and_coalescing | 3.209× | 2.589× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_streaming_loa... | Δ | streaming_evictio... | Δ | autotune_large_yb... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2900000 ns | 1170000 ns | -1730000 ns (-59.7%) | 1320000 ns | -1580000 ns (-54.5%) | 1120000 ns | -1780000 ns (-61.4%) |
| SM Throughput (%) | 24.88 % | 66.76 % | +41.88 pp | 61.40 % | +36.52 pp | 69.30 % | +44.42 pp |
| Memory Throughput (%) | 70.12 % | 64.50 % | -5.62 pp | 56.86 % | -13.26 pp | 66.59 % | -3.53 pp |
| DRAM Throughput (%) | 33.70 % | 18.26 % | -15.44 pp | 28.14 % | -5.56 pp | 10.53 % | -23.17 pp |
| Achieved Occupancy (%) | 30.35 % | 60.93 % | +30.58 pp | 61.01 % | +30.66 pp | 60.89 % | +30.54 pp |
| Registers/Thread | 94.00 register/thread | 44.00 register/thread | -50 | 44.00 register/thread | -50 | 44.00 register/thread | -50 |
| Waves/SM | 356.36 | 178.18 | -178.18 | 178.18 | -178.18 | 178.18 | -178.18 |
| No Eligible (%) | 75.08 % | 33.06 % | -42.02 pp | 38.57 % | -36.51 pp | 30.56 % | -44.52 pp |
| IPC Active | 0.99 inst/cycle | 2.67 inst/cycle | +1.68 | 2.47 inst/cycle | +1.48 | 2.78 inst/cycle | +1.79 |

---

## Iteration 3

**Best so far**: 2.5893× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_hardcoded_grid_to_enable_autotune_tile_selection | 2.744× | 2.197× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2900000 ns | 1320000 ns | -1580000 ns (-54.5%) |
| SM Throughput (%) | 24.88 % | 61.31 % | +36.43 pp |
| Memory Throughput (%) | 70.12 % | 56.78 % | -13.34 pp |
| DRAM Throughput (%) | 33.70 % | 28.22 % | -5.48 pp |
| Achieved Occupancy (%) | 30.35 % | 61.01 % | +30.66 pp |
| Registers/Thread | 94.00 register/thread | 44.00 register/thread | -50 |
| Waves/SM | 356.36 | 178.18 | -178.18 |
| No Eligible (%) | 75.08 % | 38.52 % | -36.56 pp |
| IPC Active | 0.99 inst/cycle | 2.46 inst/cycle | +1.47 |

---

## Iteration 4

**Best so far**: 2.5893× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_via_maxnreg_in_autotune_configs_to_boost_occupancy | 2.745× | 1.503× | ✓ |  |
| B | fix_hardcoded_launch_to_enable_autotune_coalesced_stores | 2.743× | 2.197× | ✓ |  |
| C | software_pipelining_via_fixed_autotune_grid_lambda | 2.744× | 2.197× | ✓ |  |
| D | fix_hardcoded_grid_bypass_and_add_large_yblock_configs | 9.264× | 2.197× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_via... | Δ | fix_hardcoded_lau... | Δ | software_pipelini... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2900000 ns | 1930000 ns | -970000 ns (-33.4%) | 1320000 ns | -1580000 ns (-54.5%) | 1320000 ns | -1580000 ns (-54.5%) | 1320000 ns | -1580000 ns (-54.5%) |
| SM Throughput (%) | 24.88 % | 46.85 % | +21.97 pp | 61.37 % | +36.49 pp | 61.48 % | +36.60 pp | 61.40 % | +36.52 pp |
| Memory Throughput (%) | 70.12 % | 66.75 % | -3.37 pp | 56.84 % | -13.28 pp | 56.94 % | -13.18 pp | 56.85 % | -13.27 pp |
| DRAM Throughput (%) | 33.70 % | 47.80 % | +14.10 pp | 28.18 % | -5.52 pp | 28.22 % | -5.48 pp | 28.02 % | -5.68 pp |
| Achieved Occupancy (%) | 30.35 % | 98.86 % | +68.51 pp | 61.01 % | +30.66 pp | 61.01 % | +30.66 pp | 61.01 % | +30.66 pp |
| Registers/Thread | 94.00 register/thread | 32.00 register/thread | -62 | 44.00 register/thread | -50 | 44.00 register/thread | -50 | 44.00 register/thread | -50 |
| Waves/SM | 356.36 | 111.36 | -245.00 | 178.18 | -178.18 | 178.18 | -178.18 | 178.18 | -178.18 |
| No Eligible (%) | 75.08 % | 53.22 % | -21.86 pp | 38.57 % | -36.51 pp | 38.47 % | -36.61 pp | 38.50 % | -36.58 pp |
| IPC Active | 0.99 inst/cycle | 1.87 inst/cycle | +0.88 | 2.47 inst/cycle | +1.48 | 2.46 inst/cycle | +1.47 | 2.46 inst/cycle | +1.47 |


### Key Learnings

#### ✅ Plan A: `cap_registers_via_maxnreg_in_autotune_configs_to_boost_occupancy` (1.503×)

When a Triton kernel is register-limited (e.g., 44 registers/thread capping Block Limit Registers at 10 and theoretical occupancy at 62.5%), adding maxnreg=32 or maxnreg=40 to @triton.autotune Config objects forces the compiler to use fewer registers per thread, allowing more blocks per SM (up to 16 at maxnreg=32 → 100% theoretical occupancy). In this case, the autotuner selected maxnreg=32, achieving 98.86% achieved occupancy vs 61.01% before. However, the register spilling introduced local memory traffic (local load/store MemoryCacheAccessPattern rules fired with only 1.0/32 bytes utilized per sector), increasing warp stall cycles from 15.8 to 33.8 cycles/instruction and kernel duration from 1.32ms to 1.93ms within the profiled kernel — yet the end-to-end speedup was 1.50x, suggesting the occupancy gain and pipeline effects outweigh the spill overhead at the benchmark level. This strategy is a net positive but the profiled kernel itself ran slower, indicating the benefit is partially from autotuner config selection effects.

> **Applicable when**: TheoreticalOccupancy NCU rule fires; registers_per_thread > 40; Block Limit Registers is the binding occupancy constraint (not shared memory or warps); No-Eligible scheduler cycles > 30%; kernel body does not have excessive register-dependent chains that would make spilling catastrophic; kernel is not already memory-bandwidth-bound at DRAM level.

> **Avoid when**: Kernel already has low register count (<32/thread); kernel is DRAM-bandwidth-bound (DRAM Throughput >60%) — spilling to local memory will add DRAM pressure; kernel has long register-dependent chains where spilling causes cascading stalls; the profiled kernel duration increase (due to local memory spill) is not offset by pipeline-level benefits. Note: in this case the profiled kernel duration increased from 1.32ms to 1.93ms despite the 1.50x end-to-end speedup, so the benefit is marginal and context-dependent.

---

## Cumulative Progress

```
Baseline:  2900000 ns  (1.000×)
Iter 1     1120000 ns  (2.589×)  — fix_uncoalesced_stores_via_tile_rebalancing_and_autotune

Final best: 1170000 ns  (2.4786×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original num_stages=1 configs (kept)
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 32},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=1),
+        # New pipelined configs with num_stages=2
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 128}, num_warps=8, num_stages=2),
+        # New pipelined configs with num_stages=3
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 32},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4800
@@ -21,7 +49,7 @@

     y5 = yindex
     y0 = (yindex % 480)
     y1 = yindex // 480
-    tmp52 = tl.load(in_ptr0 + (x4 + 50176*y5), xmask & ymask, eviction_policy='evict_last')
+    tmp52 = tl.load(in_ptr0 + (x4 + 50176*y5), mask=xmask & ymask, eviction_policy='evict_first')
     tmp0 = (-1) + x3
     tmp1 = tl.full([1, 1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
@@ -33,53 +61,51 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-225) + x4 + 50176*y5), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-225) + x4 + 50176*y5), mask=tmp10 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp12 = x2
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-224) + x4 + 50176*y5), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-224) + x4 + 50176*y5), mask=tmp16 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + x2
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-223) + x4 + 50176*y5), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-223) + x4 + 50176*y5), mask=tmp23 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = x3
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + 50176*y5), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + 50176*y5), mask=tmp30 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x4 + 50176*y5), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x4 + 50176*y5), mask=tmp33 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (1 + x4 + 50176*y5), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (1 + x4 + 50176*y5), mask=tmp36 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + x3
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (223 + x4 + 50176*y5), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (223 + x4 + 50176*y5), mask=tmp43 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (224 + x4 + 50176*y5), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (224 + x4 + 50176*y5), mask=tmp46 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (225 + x4 + 50176*y5), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (225 + x4 + 50176*y5), mask=tmp49 & xmask & ymask, eviction_policy='evict_first', other=float("-inf"))
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (y0 + 480*x4 + 24084480*y1), tmp51, xmask & ymask)
-    tl.store(out_ptr1 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-    tl.store(out_ptr2 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-    tl.store(out_ptr3 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
-
-
+    tl.store(out_ptr0 + (y0 + 480*x4 + 24084480*y1), tmp51, mask=xmask & ymask)
+    tl.store(out_ptr1 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
+    tl.store(out_ptr2 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
+    tl.store(out_ptr3 + (y0 + 480*x4 + 24084480*y1), tmp52, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -90,7 +116,7 @@

     ynumel = 4800
     xnumel = 50176
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_max_pool2d_with_indices_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4800, 50176, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_max_pool2d_with_indices_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4800, 50176)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
