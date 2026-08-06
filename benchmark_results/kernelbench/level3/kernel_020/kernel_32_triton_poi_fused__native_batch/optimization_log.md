# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6210 ns — **Best latency**: 5980 ns (**1.0385×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_tl_full_scalar_constants_with_python_float_literals_to_enable_fma_fusion`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_32_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5760,1,1), Waves/SM: 2.73
- **Registers/thread**: 17
- **Duration**: 6210 ns
- **SM Throughput**: 33.28 % | **Memory**: 14.26 % | **DRAM**: 9.78 %
- **Achieved Occupancy**: 58.46 %
- **IPC Active**: 2.10 inst/cycle | **No Eligible**: 46.04 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave_and_improve_cache_reuse | 0.886× | 0.823× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_hide_l1tex_latency | 1.023× | 1.000× | ✓ |  |
| C | vectorized_loads_larger_xblock_autotune | 1.027× | 0.290× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 7550 ns | +1340 ns (+21.6%) | 6210 ns | +0 ns (+0.0%) | 21440 ns | +15230 ns (+245.2%) |
| SM Throughput (%) | 33.28 % | 60.90 % | +27.62 pp | 34.06 % | +0.78 pp | 79.05 % | +45.77 pp |
| Memory Throughput (%) | 14.26 % | 16.77 % | +2.51 pp | 22.00 % | +7.74 pp | 15.78 % | +1.52 pp |
| DRAM Throughput (%) | 9.78 % | 8.04 % | -1.74 pp | 9.78 % | +0.00 pp | 2.83 % | -6.95 pp |
| Achieved Occupancy (%) | 58.46 % | 78.55 % | +20.09 pp | 49.84 % | -8.62 pp | 84.39 % | +25.93 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 2.73 | 2.73 | +0.00 | 2.73 | +0.00 | 10.91 | +8.18 |
| No Eligible (%) | 46.04 % | 17.41 % | -28.63 pp | 38.76 % | -7.28 pp | 12.16 % | -33.88 pp |
| IPC Active | 2.10 inst/cycle | 3.27 inst/cycle | +1.17 | 2.46 inst/cycle | +0.36 | 3.51 inst/cycle | +1.41 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_mismatch_and_tune_num_warps_at_fixed_xblock | 1.041× | 1.026× | ✓ |  |
| B | revert_autotune_and_fuse_bn_arithmetic_to_fma | 1.042× | 1.011× | ✓ | **✓ best** |
| C | fix_grid_lambda_and_enable_fma_fusion_via_fp32_reciprocal | 1.004× | 0.291× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | revert_autotune_a... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 6050 ns | -160 ns (-2.6%) | 6140 ns | -70 ns (-1.1%) | 21340 ns | +15130 ns (+243.6%) |
| SM Throughput (%) | 33.28 % | 25.44 % | -7.84 pp | 24.34 % | -8.94 pp | 78.38 % | +45.10 pp |
| Memory Throughput (%) | 14.26 % | 15.66 % | +1.40 pp | 15.41 % | +1.15 pp | 15.65 % | +1.39 pp |
| DRAM Throughput (%) | 9.78 % | 10.05 % | +0.27 pp | 9.91 % | +0.13 pp | 2.84 % | -6.94 pp |
| Achieved Occupancy (%) | 58.46 % | 51.94 % | -6.52 pp | 52.22 % | -6.24 pp | 84.31 % | +25.85 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 2.73 | 2.73 | +0.00 | 2.73 | +0.00 | 10.91 | +8.18 |
| No Eligible (%) | 46.04 % | 57.15 % | +11.11 pp | 59.00 % | +12.96 pp | 12.08 % | -33.96 pp |
| IPC Active | 2.10 inst/cycle | 1.70 inst/cycle | -0.40 | 1.56 inst/cycle | -0.54 | 3.50 inst/cycle | +1.40 |

---

## Iteration 3

**Best so far**: 1.0114× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_xblock_256_to_autotune_configs | 1.015× | 1.026× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.017× | 1.011× | ✓ |  |
| C | fix_autotune_grid_lambda_and_restore_rsqrt_to_eliminate_8x_instruction_inflation | 1.058× | 1.038× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 6050 ns | -160 ns (-2.6%) | 6140 ns | -70 ns (-1.1%) | 5980 ns | -230 ns (-3.7%) |
| SM Throughput (%) | 33.28 % | 24.64 % | -8.64 pp | 24.47 % | -8.81 pp | 24.96 % | -8.32 pp |
| Memory Throughput (%) | 14.26 % | 15.18 % | +0.92 pp | 15.33 % | +1.07 pp | 15.55 % | +1.29 pp |
| DRAM Throughput (%) | 9.78 % | 10.08 % | +0.30 pp | 9.87 % | +0.09 pp | 10.11 % | +0.33 pp |
| Achieved Occupancy (%) | 58.46 % | 52.01 % | -6.45 pp | 49.73 % | -8.73 pp | 52.04 % | -6.42 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 2.73 | 2.73 | +0.00 | 2.73 | +0.00 | 2.73 | +0.00 |
| No Eligible (%) | 46.04 % | 56.65 % | +10.61 pp | 58.89 % | +12.85 pp | 58.17 % | +12.13 pp |
| IPC Active | 2.10 inst/cycle | 1.71 inst/cycle | -0.39 | 1.63 inst/cycle | -0.47 | 1.64 inst/cycle | -0.46 |

---

## Iteration 4

**Best so far**: 1.0385× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_full_scalar_constants_with_python_float_literals_to_enable_fma_fusion | 1.023× | 1.038× | ✓ |  |
| B | replace_tl_full_scalar_constants_with_python_float_literals_to_enable_fma_fusion | 1.015× | 1.000× | ✓ |  |
| C | evict_first_on_streaming_tensor_to_improve_broadcast_cache_hit_rate | 1.047× | 0.995× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_tl_full_s... | Δ | replace_tl_full_s... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 5980 ns | -230 ns (-3.7%) | 6210 ns | +0 ns (+0.0%) | 6240 ns | +30 ns (+0.5%) |
| SM Throughput (%) | 33.28 % | 24.96 % | -8.32 pp | 25.41 % | -7.87 pp | 26.69 % | -6.59 pp |
| Memory Throughput (%) | 14.26 % | 15.65 % | +1.39 pp | 15.63 % | +1.37 pp | 15.43 % | +1.17 pp |
| DRAM Throughput (%) | 9.78 % | 10.15 % | +0.37 pp | 9.79 % | +0.01 pp | 9.73 % | -0.05 pp |
| Achieved Occupancy (%) | 58.46 % | 49.97 % | -8.49 pp | 52.55 % | -5.91 pp | 50.33 % | -8.13 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 2.73 | 2.73 | +0.00 | 2.73 | +0.00 | 2.73 | +0.00 |
| No Eligible (%) | 46.04 % | 57.90 % | +11.86 pp | 58.14 % | +12.10 pp | 54.17 % | +8.13 pp |
| IPC Active | 2.10 inst/cycle | 1.66 inst/cycle | -0.44 | 1.69 inst/cycle | -0.41 | 1.76 inst/cycle | -0.34 |

---

## Iteration 5

**Best so far**: 1.0385× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency_with_correct_grid | 1.018× | 1.016× | ✓ |  |
| B | eliminate_modulo_channel_index_with_2d_pid_decomposition | 1.021× | 1.005× | ✓ |  |
| C | precompute_bn_scale_and_bias_to_eliminate_three_broadcast_loads_and_reduce_l1tex_stalls | 1.046× | 1.032× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | eliminate_modulo_... | Δ | precompute_bn_sca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6210 ns | 6110 ns | -100 ns (-1.6%) | 6180 ns | -30 ns (-0.5%) | 6020 ns | -190 ns (-3.1%) |
| SM Throughput (%) | 33.28 % | 24.96 % | -8.32 pp | 14.58 % | -18.70 pp | 26.55 % | -6.73 pp |
| Memory Throughput (%) | 14.26 % | 15.68 % | +1.42 pp | 14.32 % | +0.06 pp | 15.31 % | +1.05 pp |
| DRAM Throughput (%) | 9.78 % | 9.97 % | +0.19 pp | 9.94 % | +0.16 pp | 10.08 % | +0.30 pp |
| Achieved Occupancy (%) | 58.46 % | 49.31 % | -9.15 pp | 43.05 % | -15.41 pp | 54.10 % | -4.36 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 2.73 | 2.73 | +0.00 | 1.36 | -1.37 | 2.73 | +0.00 |
| No Eligible (%) | 46.04 % | 57.19 % | +11.15 pp | 65.03 % | +18.99 pp | 55.03 % | +8.99 pp |
| IPC Active | 2.10 inst/cycle | 1.60 inst/cycle | -0.50 | 1.29 inst/cycle | -0.81 | 1.69 inst/cycle | -0.41 |

---

## Cumulative Progress

```
Baseline:  6210 ns  (1.000×)
Iter 2     6140 ns  (1.011×)  — revert_autotune_and_fuse_bn_arithmetic_to_fma
Iter 3     5980 ns  (1.038×)  — fix_autotune_grid_lambda_and_restore_rsqrt_to_eliminate_8x_instruction_inflation

Final best: 5980 ns  (1.0385×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,37 +5,37 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 32},  num_warps=4),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 737280
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 1152)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp5 = tmp3 + 1e-5
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, None)
-
-
+    tmp17 = tl.maximum(tmp15, 0.0)
+    tmp19 = tl.minimum(tmp17, 6.0)
+    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +45,7 @@

     arg_4 = inputs[4]
     xnumel = 737280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 737280, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_32[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 737280)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
