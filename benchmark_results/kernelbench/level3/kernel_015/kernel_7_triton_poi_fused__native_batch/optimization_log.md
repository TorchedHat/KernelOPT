# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 22140 ns — **Best latency**: 21820 ns (**1.0147×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_with_small_xblock_and_pipelined_stages`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_7_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 24
- **Duration**: 22140 ns
- **SM Throughput**: 71.49 % | **Memory**: 28.91 % | **DRAM**: 15.14 %
- **Achieved Occupancy**: 73.11 %
- **IPC Active**: 3.16 inst/cycle | **No Eligible**: 20.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | autotune_xblock_for_warp_occupancy_and_latency_hiding | 1.011× | 0.897× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 22140 ns | 24670 ns | +2530 ns (+11.4%) |
| SM Throughput (%) | 71.49 % | 75.66 % | +4.17 pp |
| Memory Throughput (%) | 28.91 % | 25.60 % | -3.31 pp |
| DRAM Throughput (%) | 15.14 % | 13.60 % | -1.54 pp |
| Achieved Occupancy (%) | 73.11 % | 82.55 % | +9.44 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 |
| No Eligible (%) | 20.46 % | 16.14 % | -4.32 pp |
| IPC Active | 3.16 inst/cycle | 3.33 inst/cycle | +0.17 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_to_enable_autotune_xblock_configs | 1.004× | 0.892× | ✓ |  |
| B | fix_grid_and_enable_fma_fusion_via_float_arithmetic | 1.019× | 1.015× | ✓ | **✓ best** |
| C | fix_grid_xblock_and_fuse_fp32_instructions | 1.018× | 1.018× | ✓ |  |
| D | fix_grid_hardcode_and_precompute_bn_scale_for_fma_fusion | 1.009× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_grid_and_enab... | Δ | fix_grid_xblock_a... | Δ | fix_grid_hardcode... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22140 ns | 24830 ns | +2690 ns (+12.1%) | 21820 ns | -320 ns (-1.4%) | 21760 ns | -380 ns (-1.7%) | 22050 ns | -90 ns (-0.4%) |
| SM Throughput (%) | 71.49 % | 76.31 % | +4.82 pp | 59.69 % | -11.80 pp | 60.07 % | -11.42 pp | 63.13 % | -8.36 pp |
| Memory Throughput (%) | 28.91 % | 25.83 % | -3.08 pp | 28.80 % | -0.11 pp | 29.05 % | +0.14 pp | 28.41 % | -0.50 pp |
| DRAM Throughput (%) | 15.14 % | 13.51 % | -1.63 pp | 15.37 % | +0.23 pp | 15.41 % | +0.27 pp | 15.18 % | +0.04 pp |
| Achieved Occupancy (%) | 73.11 % | 81.13 % | +8.02 pp | 48.71 % | -24.40 pp | 48.59 % | -24.52 pp | 51.16 % | -21.95 pp |
| Registers/Thread | 24.00 register/thread | 24.00 register/thread | +0 | 23.00 register/thread | -1 | 23.00 register/thread | -1 | 23.00 register/thread | -1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 20.46 % | 16.14 % | -4.32 pp | 32.81 % | +12.35 pp | 31.47 % | +11.01 pp | 28.16 % | +7.70 pp |
| IPC Active | 3.16 inst/cycle | 3.34 inst/cycle | +0.18 | 2.67 inst/cycle | -0.49 | 2.70 inst/cycle | -0.46 | 2.86 inst/cycle | -0.30 |

---

## Iteration 3

**Best so far**: 1.0147× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_autotuned_xblock | 1.012× | 1.015× | ✓ |  |
| B | fix_hardcoded_grid_and_increase_xblock_tile_size | 0.757× | 0.606× | ✓ |  |
| C | fix_hardcoded_grid_and_use_dynamic_xblock_in_launch | 1.011× | 1.013× | ✓ |  |
| D | fix_grid_lambda_and_fuse_bn_arithmetic_via_inv_std | 1.010× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_hardcoded_gri... | Δ | fix_hardcoded_gri... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22140 ns | 21820 ns | -320 ns (-1.4%) | 36540 ns | +14400 ns (+65.0%) | 21860 ns | -280 ns (-1.3%) | 21790 ns | -350 ns (-1.6%) |
| SM Throughput (%) | 71.49 % | 60.70 % | -10.79 pp | 78.82 % | +7.33 pp | 64.89 % | -6.60 pp | 57.97 % | -13.52 pp |
| Memory Throughput (%) | 28.91 % | 29.35 % | +0.44 pp | 36.00 % | +7.09 pp | 29.27 % | +0.36 pp | 28.78 % | -0.13 pp |
| DRAM Throughput (%) | 15.14 % | 15.38 % | +0.24 pp | 9.17 % | -5.97 pp | 15.36 % | +0.22 pp | 15.39 % | +0.25 pp |
| Achieved Occupancy (%) | 73.11 % | 49.62 % | -23.49 pp | 81.69 % | +8.58 pp | 51.46 % | -21.65 pp | 49.77 % | -23.34 pp |
| Registers/Thread | 24.00 register/thread | 23.00 register/thread | -1 | 30.00 register/thread | +6 | 23.00 register/thread | -1 | 25.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 29.70 | +14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 20.46 % | 31.49 % | +11.03 pp | 16.69 % | -3.77 pp | 26.83 % | +6.37 pp | 33.41 % | +12.95 pp |
| IPC Active | 3.16 inst/cycle | 2.68 inst/cycle | -0.48 | 3.30 inst/cycle | +0.14 | 2.89 inst/cycle | -0.27 | 2.65 inst/cycle | -0.51 |

---

## Iteration 4

**Best so far**: 1.0147× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_for_autotune | 1.008× | 1.009× | ✓ |  |
| B | fix_grid_lambda_with_small_xblock_and_pipelined_stages | 1.020× | 1.015× | ✓ | **✓ best** |
| C | fix_grid_lambda_with_small_xblock_configs_only | 1.011× | 1.015× | ✓ |  |
| D | restore_small_xblock_configs_with_corrected_grid_lambda | 1.018× | 1.018× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_w... | Δ | fix_grid_lambda_w... | Δ | restore_small_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22140 ns | 21950 ns | -190 ns (-0.9%) | 21820 ns | -320 ns (-1.4%) | 21820 ns | -320 ns (-1.4%) | 21760 ns | -380 ns (-1.7%) |
| SM Throughput (%) | 71.49 % | 59.45 % | -12.04 pp | 60.38 % | -11.11 pp | 64.65 % | -6.84 pp | 60.44 % | -11.05 pp |
| Memory Throughput (%) | 28.91 % | 29.04 % | +0.13 pp | 29.17 % | +0.26 pp | 29.12 % | +0.21 pp | 29.18 % | +0.27 pp |
| DRAM Throughput (%) | 15.14 % | 15.30 % | +0.16 pp | 15.40 % | +0.26 pp | 15.39 % | +0.25 pp | 15.40 % | +0.26 pp |
| Achieved Occupancy (%) | 73.11 % | 49.32 % | -23.79 pp | 49.10 % | -24.01 pp | 52.67 % | -20.44 pp | 49.37 % | -23.74 pp |
| Registers/Thread | 24.00 register/thread | 27.00 register/thread | +3 | 23.00 register/thread | -1 | 23.00 register/thread | -1 | 23.00 register/thread | -1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 20.46 % | 32.81 % | +12.35 pp | 32.04 % | +11.58 pp | 27.27 % | +6.81 pp | 32.20 % | +11.74 pp |
| IPC Active | 3.16 inst/cycle | 2.67 inst/cycle | -0.49 | 2.72 inst/cycle | -0.44 | 2.92 inst/cycle | -0.24 | 2.71 inst/cycle | -0.45 |

---

## Iteration 5

**Best so far**: 1.0147× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_to_use_meta_xblock_from_autotuner | 1.019× | 1.018× | ✓ |  |
| B | fix_grid_lambda_and_tune_num_warps_num_stages_at_xblock128 | 1.012× | 1.013× | ✓ |  |
| C | fix_grid_lambda_and_expand_xblock_configs_for_occupancy | 1.010× | 1.010× | ✓ |  |
| D | fix_grid_lambda_and_tune_num_warps_num_stages | 1.016× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 22140 ns | 21760 ns | -380 ns (-1.7%) | 21860 ns | -280 ns (-1.3%) | 21920 ns | -220 ns (-1.0%) | 21920 ns | -220 ns (-1.0%) |
| SM Throughput (%) | 71.49 % | 60.26 % | -11.23 pp | 60.52 % | -10.97 pp | 64.80 % | -6.69 pp | 59.37 % | -12.12 pp |
| Memory Throughput (%) | 28.91 % | 29.17 % | +0.26 pp | 29.24 % | +0.33 pp | 29.19 % | +0.28 pp | 28.97 % | +0.06 pp |
| DRAM Throughput (%) | 15.14 % | 15.43 % | +0.29 pp | 15.36 % | +0.22 pp | 15.35 % | +0.21 pp | 15.32 % | +0.18 pp |
| Achieved Occupancy (%) | 73.11 % | 49.32 % | -23.79 pp | 49.75 % | -23.36 pp | 52.47 % | -20.64 pp | 49.71 % | -23.40 pp |
| Registers/Thread | 24.00 register/thread | 23.00 register/thread | -1 | 23.00 register/thread | -1 | 23.00 register/thread | -1 | 27.00 register/thread | +3 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 20.46 % | 32.07 % | +11.61 pp | 31.48 % | +11.02 pp | 27.24 % | +6.78 pp | 32.65 % | +12.19 pp |
| IPC Active | 3.16 inst/cycle | 2.73 inst/cycle | -0.43 | 2.70 inst/cycle | -0.46 | 2.90 inst/cycle | -0.26 | 2.66 inst/cycle | -0.50 |

---

## Cumulative Progress

```
Baseline:  22140 ns  (1.000×)
Iter 2     21820 ns  (1.015×)  — fix_grid_and_enable_fma_fusion_via_float_arithmetic

Final best: 21820 ns  (1.0147×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,52 +5,61 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4014080
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 128)
     x1 = xindex // 128
     x2 = xindex
-    tmp17 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp28 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
-    tmp30 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
+    tmp17 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp28 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp30 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 64, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 96, tl.int64)
     tmp8 = tmp0 < tmp7
     tmp9 = tmp6 & tmp8
-    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), tmp9, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=tmp9 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp0 >= tmp7
     tmp12 = tl.full([1], 128, tl.int64)
     tmp13 = tmp0 < tmp12
-    tmp14 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), tmp11, eviction_policy='evict_last', other=0.0)
+    tmp14 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), mask=tmp11 & xmask, eviction_policy='evict_last', other=0.0)
     tmp15 = tl.where(tmp9, tmp10, tmp14)
     tmp16 = tl.where(tmp4, tmp5, tmp15)
     tmp18 = tmp16 - tmp17
     tmp20 = tl.full([1], 1e-05, tl.float32)
     tmp21 = tmp19 + tmp20
-    tmp22 = tl.sqrt_rn(tmp21)
-    tmp23 = tl.full([1], 1, tl.int32)
-    tmp24 = (tmp23 / tmp22)
-    tmp25 = tl.full([1], 1.0, tl.float32)
-    tmp26 = tmp24 * tmp25
-    tmp27 = tmp18 * tmp26
+    tmp24 = tl.math.rsqrt(tmp21)
+    tmp27 = tmp18 * tmp24
     tmp29 = tmp27 * tmp28
     tmp31 = tmp29 + tmp30
     tmp32 = tl.full([1], 0, tl.int32)
     tmp33 = tl.maximum(tmp32, tmp31)
-    tl.store(in_out_ptr0 + (x2), tmp33, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp33, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -63,7 +72,7 @@

     arg_7 = inputs[7]
     xnumel = 4014080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 4014080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 4014080)
     return arg_7
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
