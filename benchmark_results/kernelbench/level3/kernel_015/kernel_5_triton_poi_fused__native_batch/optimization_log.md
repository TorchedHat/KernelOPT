# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17340 ns — **Best latency**: 17090 ns (**1.0146×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`replace_manual_reciprocal_sqrt_with_rsqrt_for_fma_fusion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 22
- **Duration**: 17340 ns
- **SM Throughput**: 64.23 % | **Memory**: 24.46 % | **DRAM**: 14.19 %
- **Achieved Occupancy**: 64.96 %
- **IPC Active**: 2.99 inst/cycle | **No Eligible**: 25.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_xblock_and_num_warps_for_occupancy_and_latency_hiding | 0.801× | 0.624× | ✓ |  |
| C | increase_xblock_with_autotune_for_latency_hiding | 0.952× | 0.843× | ✓ |  |
| D | autotune_xblock_and_num_warps | 0.990× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_a... | Δ | increase_xblock_w... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17340 ns | 27780 ns | +10440 ns (+60.2%) | 20580 ns | +3240 ns (+18.7%) | 17630 ns | +290 ns (+1.7%) |
| SM Throughput (%) | 64.23 % | 78.87 % | +14.64 pp | 76.65 % | +12.42 pp | 71.99 % | +7.76 pp |
| Memory Throughput (%) | 24.46 % | 21.24 % | -3.22 pp | 28.64 % | +4.18 pp | 25.00 % | +0.54 pp |
| DRAM Throughput (%) | 14.19 % | 8.85 % | -5.34 pp | 11.95 % | -2.24 pp | 13.95 % | -0.24 pp |
| Achieved Occupancy (%) | 64.96 % | 88.33 % | +23.37 pp | 84.66 % | +19.70 pp | 78.46 % | +13.50 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 28.00 register/thread | +6 | 22.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 25.24 % | 14.03 % | -11.21 pp | 14.41 % | -10.83 pp | 17.48 % | -7.76 pp |
| IPC Active | 2.99 inst/cycle | 3.43 inst/cycle | +0.44 | 3.41 inst/cycle | +0.42 | 3.28 inst/cycle | +0.29 |


### Key Learnings

#### ❌ Plan C: `increase_xblock_with_autotune_for_latency_hiding` (0.843×)

Increasing XBLOCK via autotune to hide L1TEX scoreboard stalls (CPIStall, 38.5% of cycles) caused a 18.7% slowdown. Although achieved occupancy improved from 65% to 85% and compute SM throughput rose from 64% to 77%, executed instructions ballooned from 8.8M to 12.2M (+38%) and registers per thread increased from 22 to 28, indicating the autotune selected a larger-XBLOCK config that introduced more instruction overhead and register pressure than the latency hiding benefit could offset. The net effect was a longer kernel duration (17.3 µs → 20.6 µs). Additionally, the grid was not updated to use a lambda, so the static grid=(cdiv(xnumel,128),) was used regardless of the autotuned XBLOCK, meaning the kernel was launched with the wrong grid size — this is an implementation flaw that invalidates the experiment.

> **Applicable when**: CPIStall rule fires with L1TEX stall fraction >30%, achieved occupancy well below theoretical, low register pressure (<24/thread), and large grid (>10k blocks)

> **Avoid when**: The static launch grid is not updated to a lambda that reads meta['XBLOCK']; when the autotune selects a larger XBLOCK but the grid remains fixed at cdiv(xnumel, original_XBLOCK), the kernel is launched with too many blocks and each block processes a smaller-than-XBLOCK tile, causing instruction count inflation and register pressure increase without the intended latency hiding. Also avoid when the xmask is changed from tl.full([XBLOCK], True, tl.int1) to a bounds-check mask when xnumel is exactly divisible by XBLOCK, as this adds unnecessary predication overhead.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_manual_reciprocal_sqrt_with_rsqrt_for_fma_fusion | 1.006× | 1.015× | ✓ | **✓ best** |
| B | fuse_fp32_instructions_via_rsqrt_and_fma_restructuring | 0.996× | 0.996× | ✓ |  |
| C | replace_integer_reciprocal_with_fp32_rsqrt_to_enable_fma_fusion | 0.957× | 0.956× | ✓ |  |
| D | add_triton_autotune | 0.991× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_manual_re... | Δ | fuse_fp32_instruc... | Δ | replace_integer_r... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17340 ns | 17090 ns | -250 ns (-1.4%) | 17410 ns | +70 ns (+0.4%) | 18140 ns | +800 ns (+4.6%) | 17540 ns | +200 ns (+1.2%) |
| SM Throughput (%) | 64.23 % | 50.96 % | -13.27 pp | 54.52 % | -9.71 pp | 65.44 % | +1.21 pp | 71.24 % | +7.01 pp |
| Memory Throughput (%) | 24.46 % | 25.40 % | +0.94 pp | 35.04 % | +10.58 pp | 34.27 % | +9.81 pp | 24.74 % | +0.28 pp |
| DRAM Throughput (%) | 14.19 % | 14.39 % | +0.20 pp | 14.15 % | -0.04 pp | 13.56 % | -0.63 pp | 14.04 % | -0.15 pp |
| Achieved Occupancy (%) | 64.96 % | 46.06 % | -18.90 pp | 39.45 % | -25.51 pp | 63.34 % | -1.62 pp | 78.25 % | +13.29 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 28.00 register/thread | +6 | 32.00 register/thread | +10 | 22.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 25.24 % | 40.23 % | +14.99 pp | 34.06 % | +8.82 pp | 24.73 % | -0.51 pp | 18.88 % | -6.36 pp |
| IPC Active | 2.99 inst/cycle | 2.39 inst/cycle | -0.60 | 2.63 inst/cycle | -0.36 | 3.01 inst/cycle | +0.02 | 3.27 inst/cycle | +0.28 |

---

## Iteration 3

**Best so far**: 1.0146× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restructure_bn_arithmetic_for_fma_fusion | 0.993× | 0.989× | ✓ |  |
| B | fix_autotune_grid_lambda_and_reduce_register_pressure_via_fma | 0.991× | 0.993× | ✓ |  |
| C | fix_hardcoded_grid_to_enable_autotune_xblock_selection | 0.993× | 1.016× | ✓ |  |
| D | enable_fma_fusion_by_replacing_int_reciprocal_sqrt_with_libdevice_rsqrt | 1.002× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | restructure_bn_ar... | Δ | fix_autotune_grid... | Δ | fix_hardcoded_gri... | Δ | enable_fma_fusion... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17340 ns | 17540 ns | +200 ns (+1.2%) | 17470 ns | +130 ns (+0.7%) | 17060 ns | -280 ns (-1.6%) | 16960 ns | -380 ns (-2.2%) |
| SM Throughput (%) | 64.23 % | 71.81 % | +7.58 pp | 54.60 % | -9.63 pp | 51.00 % | -13.23 pp | 51.16 % | -13.07 pp |
| Memory Throughput (%) | 24.46 % | 24.93 % | +0.47 pp | 35.11 % | +10.65 pp | 25.47 % | +1.01 pp | 25.54 % | +1.08 pp |
| DRAM Throughput (%) | 14.19 % | 14.02 % | -0.17 pp | 14.09 % | -0.10 pp | 14.45 % | +0.26 pp | 14.50 % | +0.31 pp |
| Achieved Occupancy (%) | 64.96 % | 78.52 % | +13.56 pp | 40.28 % | -24.68 pp | 45.75 % | -19.21 pp | 46.18 % | -18.78 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 28.00 register/thread | +6 | 22.00 register/thread | +0 | 22.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 25.24 % | 18.42 % | -6.82 pp | 34.12 % | +8.88 pp | 40.51 % | +15.27 pp | 39.89 % | +14.65 pp |
| IPC Active | 2.99 inst/cycle | 3.28 inst/cycle | +0.29 | 2.57 inst/cycle | -0.42 | 2.39 inst/cycle | -0.60 | 2.34 inst/cycle | -0.65 |

---

## Iteration 4

**Best so far**: 1.0146× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_static_grid_to_lambda_so_autotune_xblock_takes_effect | 0.984× | 0.989× | ✓ |  |
| B | change_input_tensor_eviction_policy_to_evict_first_for_streaming_loads | 1.000× | 1.009× | ✓ |  |
| C | change_main_tensor_eviction_policy_to_evict_first_for_l1_cache_pressure_relief | 0.980× | 0.996× | ✓ |  |
| D | eliminate_nonfused_fp32_via_libdevice_rsqrt_direct_fma_chain | 0.993× | 1.019× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_static_grid_t... | Δ | change_input_tens... | Δ | change_main_tenso... | Δ | eliminate_nonfuse... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17340 ns | 17540 ns | +200 ns (+1.2%) | 17180 ns | -160 ns (-0.9%) | 17410 ns | +70 ns (+0.4%) | 17020 ns | -320 ns (-1.8%) |
| SM Throughput (%) | 64.23 % | 71.94 % | +7.71 pp | 50.51 % | -13.72 pp | 54.57 % | -9.66 pp | 49.81 % | -14.42 pp |
| Memory Throughput (%) | 24.46 % | 24.99 % | +0.53 pp | 25.23 % | +0.77 pp | 35.08 % | +10.62 pp | 25.00 % | +0.54 pp |
| DRAM Throughput (%) | 14.19 % | 14.03 % | -0.16 pp | 14.32 % | +0.13 pp | 14.14 % | -0.05 pp | 14.45 % | +0.26 pp |
| Achieved Occupancy (%) | 64.96 % | 77.96 % | +13.00 pp | 45.40 % | -19.56 pp | 40.67 % | -24.29 pp | 47.56 % | -17.40 pp |
| Registers/Thread | 22.00 register/thread | 23.00 register/thread | +1 | 22.00 register/thread | +0 | 28.00 register/thread | +6 | 20.00 register/thread | -2 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 25.24 % | 17.59 % | -7.65 pp | 40.77 % | +15.53 pp | 34.11 % | +8.87 pp | 40.95 % | +15.71 pp |
| IPC Active | 2.99 inst/cycle | 3.29 inst/cycle | +0.30 | 2.36 inst/cycle | -0.63 | 2.63 inst/cycle | -0.36 | 2.32 inst/cycle | -0.67 |

---

## Cumulative Progress

```
Baseline:  17340 ns  (1.000×)
Iter 2     17090 ns  (1.015×)  — replace_manual_reciprocal_sqrt_with_rsqrt_for_fma_fusion

Final best: 17090 ns  (1.0146×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,46 +5,49 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x0 = (xindex % 96)
     x1 = xindex // 96
     x2 = xindex
-    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp22 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
-    tmp24 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
+    tmp11 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp13 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp22 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp24 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 64, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), tmp4, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=xmask & tmp4, eviction_policy='evict_last', other=0.0)
     tmp6 = tmp0 >= tmp3
     tmp7 = tl.full([1], 96, tl.int64)
     tmp8 = tmp0 < tmp7
-    tmp9 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), tmp6, eviction_policy='evict_last', other=0.0)
+    tmp9 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=xmask & tmp6, eviction_policy='evict_last', other=0.0)
     tmp10 = tl.where(tmp4, tmp5, tmp9)
     tmp12 = tmp10 - tmp11
-    tmp14 = tl.full([1], 1e-05, tl.float32)
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.sqrt_rn(tmp15)
-    tmp17 = tl.full([1], 1, tl.int32)
-    tmp18 = (tmp17 / tmp16)
-    tmp19 = tl.full([1], 1.0, tl.float32)
-    tmp20 = tmp18 * tmp19
-    tmp21 = tmp12 * tmp20
+    tmp15 = tmp13 + tl.full([1], 1e-05, tl.float32)
+    tmp_inv_std = libdevice.rsqrt(tmp15)
+    tmp21 = tmp12 * tmp_inv_std
     tmp23 = tmp21 * tmp22
     tmp25 = tmp23 + tmp24
     tmp26 = tl.full([1], 0, tl.int32)
     tmp27 = tl.maximum(tmp26, tmp25)
-    tl.store(out_ptr0 + (x2), tmp27, None)
-
-
+    tl.store(out_ptr0 + (x2), tmp27, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -56,7 +59,7 @@

     arg_6 = inputs[6]
     xnumel = 3010560
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 3010560, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 3010560)
     return arg_6
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
