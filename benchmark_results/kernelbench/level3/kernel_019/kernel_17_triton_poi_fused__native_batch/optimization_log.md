# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 3710 ns (**1.0431×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_to_use_autotuned_xblock_and_add_l2_grouping_for_workload_balance`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_17_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 0.93
- **Registers/thread**: 18
- **Duration**: 3870 ns
- **SM Throughput**: 21.96 % | **Memory**: 7.76 % | **DRAM**: 5.41 %
- **Achieved Occupancy**: 63.68 %
- **IPC Active**: 2.13 inst/cycle | **No Eligible**: 44.13 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_wave_utilization | 0.957× | 0.884× | ✓ |  |
| B | autotune_xblock_tile_size_for_better_occupancy_and_latency_hiding | 1.013× | 0.937× | ✓ |  |
| C | increase_xblock_vectorized_loads_with_autotune | 1.014× | 0.930× | ✓ | **✓ best** |
| D | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.955× | 0.884× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | increase_xblock_v... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 4380 ns | +510 ns (+13.2%) | 4130 ns | +260 ns (+6.7%) | 4160 ns | +290 ns (+7.5%) | 4380 ns | +510 ns (+13.2%) |
| SM Throughput (%) | 21.96 % | 37.86 % | +15.90 pp | 15.59 % | -6.37 pp | 37.65 % | +15.69 pp | 37.81 % | +15.85 pp |
| Memory Throughput (%) | 7.76 % | 10.14 % | +2.38 pp | 7.13 % | -0.63 pp | 10.06 % | +2.30 pp | 10.10 % | +2.34 pp |
| DRAM Throughput (%) | 5.41 % | 4.76 % | -0.65 pp | 5.06 % | -0.35 pp | 5.02 % | -0.39 pp | 4.77 % | -0.64 pp |
| Achieved Occupancy (%) | 63.68 % | 62.66 % | -1.02 pp | 37.99 % | -25.69 pp | 62.63 % | -1.05 pp | 62.82 % | -0.86 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.46 | -0.47 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 44.13 % | 31.21 % | -12.92 pp | 59.91 % | +15.78 pp | 25.02 % | -19.11 pp | 24.80 % | -19.33 pp |
| IPC Active | 2.13 inst/cycle | 2.93 inst/cycle | +0.80 | 1.55 inst/cycle | -0.58 | 2.94 inst/cycle | +0.81 | 2.92 inst/cycle | +0.79 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_and_fix_lambda_grid_for_latency_hiding | 0.975× | 0.930× | ✓ |  |
| B | fix_hardcoded_grid_to_enable_autotune_and_expand_configs | 1.001× | 0.876× | ✓ |  |
| C | fix_grid_formula_and_add_num_stages_to_autotune | 0.922× | 0.917× | ✓ |  |
| D | fix_grid_launch_to_use_autotuned_xblock_and_add_num_stages | 0.925× | 0.924× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_an... | Δ | fix_hardcoded_gri... | Δ | fix_grid_formula_... | Δ | fix_grid_launch_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 4160 ns | +290 ns (+7.5%) | 4420 ns | +550 ns (+14.2%) | 4220 ns | +350 ns (+9.0%) | 4190 ns | +320 ns (+8.3%) |
| SM Throughput (%) | 21.96 % | 14.92 % | -7.04 pp | 37.82 % | +15.86 pp | 37.80 % | +15.84 pp | 36.15 % | +14.19 pp |
| Memory Throughput (%) | 7.76 % | 7.15 % | -0.61 pp | 10.12 % | +2.36 pp | 10.13 % | +2.37 pp | 9.68 % | +1.92 pp |
| DRAM Throughput (%) | 5.41 % | 5.00 % | -0.41 pp | 4.73 % | -0.68 pp | 4.97 % | -0.44 pp | 4.98 % | -0.43 pp |
| Achieved Occupancy (%) | 63.68 % | 35.50 % | -28.18 pp | 65.59 % | +1.91 pp | 62.98 % | -0.70 pp | 64.85 % | +1.17 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 0.93 | 0.46 | -0.47 | 0.93 | +0.00 | 0.93 | +0.00 | 0.93 | +0.00 |
| No Eligible (%) | 44.13 % | 53.90 % | +9.77 pp | 25.20 % | -18.93 pp | 25.12 % | -19.01 pp | 24.55 % | -19.58 pp |
| IPC Active | 2.13 inst/cycle | 1.66 inst/cycle | -0.47 | 2.65 inst/cycle | +0.52 | 2.91 inst/cycle | +0.78 | 2.75 inst/cycle | +0.62 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_and_improve_occupancy | 1.014× | 0.908× | ✓ |  |
| B | rewrite_arithmetic_for_fma_fusion_and_fix_grid_launch | 1.042× | 1.090× | ✓ | **✓ best** |
| C | reduce_warp_contention_via_fewer_warps_and_correct_grid_formula | 0.964× | 1.016× | ✓ |  |
| D | replace_sqrt_reciprocal_with_rsqrt_and_eliminate_multiply_by_one | 1.029× | 1.099× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | rewrite_arithmeti... | Δ | reduce_warp_conte... | Δ | replace_sqrt_reci... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 4260 ns | +390 ns (+10.1%) | 3550 ns | -320 ns (-8.3%) | 3810 ns | -60 ns (-1.6%) | 3520 ns | -350 ns (-9.0%) |
| SM Throughput (%) | 21.96 % | 15.85 % | -6.11 pp | 20.40 % | -1.56 pp | 23.52 % | +1.56 pp | 21.10 % | -0.86 pp |
| Memory Throughput (%) | 7.76 % | 6.96 % | -0.80 pp | 11.93 % | +4.17 pp | 8.25 % | +0.49 pp | 12.99 % | +5.23 pp |
| DRAM Throughput (%) | 5.41 % | 4.93 % | -0.48 pp | 5.88 % | +0.47 pp | 5.51 % | +0.10 pp | 5.92 % | +0.51 pp |
| Achieved Occupancy (%) | 63.68 % | 35.54 % | -28.14 pp | 63.33 % | -0.35 pp | 32.70 % | -30.98 pp | 63.21 % | -0.47 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 0.93 | 0.46 | -0.47 | 0.93 | +0.00 | 0.46 | -0.47 | 0.93 | +0.00 |
| No Eligible (%) | 44.13 % | 57.28 % | +13.15 pp | 36.06 % | -8.07 pp | 44.94 % | +0.81 pp | 36.90 % | -7.23 pp |
| IPC Active | 2.13 inst/cycle | 1.69 inst/cycle | -0.44 | 2.40 inst/cycle | +0.27 | 1.90 inst/cycle | -0.23 | 2.40 inst/cycle | +0.27 |

---

## Iteration 4

**Best so far**: 1.0901× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_autotuned_xblock_and_add_l2_grouping_for_workload_balance | 1.068× | 1.043× | ✓ | **✓ best** |
| B | fix_hardcoded_grid_and_eliminate_relu_int32_cast_for_fma_fusion | 1.020× | 1.109× | ✓ |  |
| C | l2_swizzle_channel_aligned_tiling_to_fix_workload_imbalance | 1.010× | 1.090× | ✓ |  |
| D | fix_grid_to_use_autotuned_xblock_and_add_l2_grouping_for_channel_last_locality | 0.971× | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | l2_swizzle_channe... | Δ | fix_grid_to_use_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3710 ns | -160 ns (-4.1%) | 3490 ns | -380 ns (-9.8%) | 3550 ns | -320 ns (-8.3%) | 4030 ns | +160 ns (+4.1%) |
| SM Throughput (%) | 21.96 % | 20.90 % | -1.06 pp | 20.82 % | -1.14 pp | 20.82 % | -1.14 pp | 8.45 % | -13.51 pp |
| Memory Throughput (%) | 7.76 % | 12.74 % | +4.98 pp | 12.74 % | +4.98 pp | 12.74 % | +4.98 pp | 4.04 % | -3.72 pp |
| DRAM Throughput (%) | 5.41 % | 5.65 % | +0.24 pp | 5.95 % | +0.54 pp | 5.81 % | +0.40 pp | 2.63 % | -2.78 pp |
| Achieved Occupancy (%) | 63.68 % | 64.90 % | +1.22 pp | 64.53 % | +0.85 pp | 63.69 % | +0.01 pp | 19.22 % | -44.46 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 26.00 register/thread | +8 |
| Waves/SM | 0.93 | 0.93 | +0.00 | 0.93 | +0.00 | 0.93 | +0.00 | 0.46 | -0.47 |
| No Eligible (%) | 44.13 % | 36.45 % | -7.68 pp | 37.89 % | -6.24 pp | 36.56 % | -7.57 pp | 74.81 % | +30.68 pp |
| IPC Active | 2.13 inst/cycle | 2.25 inst/cycle | +0.12 | 2.24 inst/cycle | +0.11 | 2.39 inst/cycle | +0.26 | 1.06 inst/cycle | -1.07 |

---

## Iteration 5

**Best so far**: 1.0901× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swizzle_program_id_for_l2_locality_and_fix_grid_lambda | 1.007× | 1.052× | ✓ |  |
| B | fix_hardcoded_grid_lambda_and_simplify_swizzle_for_occupancy_gain | 1.044× | 1.099× | ✓ |  |
| C | fix_grid_lambda_and_force_large_xblock_via_config_pruning | 1.030× | 1.090× | ✓ |  |
| D | unify_rsqrt_float_relu_constant_and_correct_grid_lambda_for_fma_and_occupancy | 1.056× | 1.099× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swizzle_program_i... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ | unify_rsqrt_float... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3680 ns | -190 ns (-4.9%) | 3520 ns | -350 ns (-9.0%) | 3550 ns | -320 ns (-8.3%) | 3520 ns | -350 ns (-9.0%) |
| SM Throughput (%) | 21.96 % | 7.81 % | -14.15 pp | 7.50 % | -14.46 pp | 20.69 % | -1.27 pp | 4.44 % | -17.52 pp |
| Memory Throughput (%) | 7.76 % | 4.79 % | -2.97 pp | 4.54 % | -3.22 pp | 12.63 % | +4.87 pp | 4.54 % | -3.22 pp |
| DRAM Throughput (%) | 5.41 % | 2.88 % | -2.53 pp | 3.01 % | -2.40 pp | 5.88 % | +0.47 pp | 2.99 % | -2.42 pp |
| Achieved Occupancy (%) | 63.68 % | 32.94 % | -30.74 pp | 33.67 % | -30.01 pp | 62.68 % | -1.00 pp | 17.73 % | -45.95 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 22.00 register/thread | +4 |
| Waves/SM | 0.93 | 0.46 | -0.47 | 0.46 | -0.47 | 0.93 | +0.00 | 0.46 | -0.47 |
| No Eligible (%) | 44.13 % | 72.65 % | +28.52 pp | 72.31 % | +28.18 pp | 38.60 % | -5.53 pp | 84.78 % | +40.65 pp |
| IPC Active | 2.13 inst/cycle | 0.94 inst/cycle | -1.19 | 0.89 inst/cycle | -1.24 | 2.44 inst/cycle | +0.31 | 0.71 inst/cycle | -1.42 |


### Key Learnings

#### ✅ Plan A: `swizzle_program_id_for_l2_locality_and_fix_grid_lambda` (1.052×)

Adding a program-ID swizzle (reordering blocks to group same-channel-stripe work together) alongside expanding autotune configs to include smaller XBLOCK values (64, 128) caused the autotuner to select XBLOCK=64 instead of XBLOCK=128. This halved waves-per-SM from 0.93 to 0.46 and dropped achieved occupancy from 64.5% to 32.9%, increasing elapsed cycles from 5195 to 5439 despite a marginal L2 hit rate improvement (58.1% → 61.8%). The net wall-clock result was a modest 5.2% speedup, but the profiling data shows the kernel is now severely under-occupancy-limited: the occupancy regression from the smaller block size outweighs any L2 locality gain. The swizzle itself is not the primary cause of the speedup; rather, the register reduction (32 → 18 registers/thread) from XBLOCK=64 may have contributed, but the occupancy collapse is the dominant effect.

> **Applicable when**: WorkloadImbalance NCU rule fires on L2 slices (>20% above average); kernel has a channel-stride access pattern where consecutive blocks access different channel stripes; the autotuner is constrained to large XBLOCK values (>=512) so the swizzle does not interact with block-size selection; achieved occupancy is already high (>60%) and waves-per-SM is near-integer.

> **Avoid when**: Autotune config space includes small XBLOCK values (64, 128) that the autotuner may prefer over larger ones, because the swizzle logic at small XBLOCK (swizzle_width>1) adds overhead and the autotuner may select a block size that collapses occupancy; achieved occupancy is already below 50% (adding swizzle overhead worsens stall ratio); the kernel is already compute-bound rather than L2-imbalance-bound.

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 3     3550 ns  (1.090×)  — rewrite_arithmetic_for_fma_fusion_and_fix_grid_launch

Final best: 3710 ns  (1.0431×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 250880
@@ -21,19 +33,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    # tl.math.rsqrt maps to single MUFU.RSQ instruction; enables FMA chain
+    tmp6 = tl.math.rsqrt(tmp5)   # 1/sqrt(var+eps) in one instruction
+    tmp11 = tmp2 * tmp6           # (x - mean) * rsqrt(var+eps)
+    tmp13 = tmp11 * tmp12         # * weight — compiler emits FFMA
+    tmp15 = tmp13 + tmp14         # + bias   — compiler emits FFMA
+    tmp17 = tl.maximum(0.0, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +49,7 @@

     arg_4 = inputs[4]
     xnumel = 250880
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 250880, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 250880)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
