# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3070 ns — **Best latency**: 2980 ns (**1.0302×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_bn_normalization_with_rsqrt_and_restore_small_xblock_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_37_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (735,1,1), Waves/SM: 0.35
- **Registers/thread**: 18
- **Duration**: 3070 ns
- **SM Throughput**: 10.05 % | **Memory**: 3.99 % | **DRAM**: 2.58 %
- **Achieved Occupancy**: 30.25 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 70.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 1.018× | 0.880× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_occupancy | 1.038× | 1.020× | ✓ | **✓ best** |
| C | vectorized_loads_larger_xblock_autotune | 0.996× | 0.773× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 3490 ns | +420 ns (+13.7%) | 3010 ns | -60 ns (-2.0%) | 3970 ns | +900 ns (+29.3%) |
| SM Throughput (%) | 10.05 % | 16.48 % | +6.43 pp | 10.34 % | +0.29 pp | 29.66 % | +19.61 pp |
| Memory Throughput (%) | 3.99 % | 4.46 % | +0.47 pp | 4.25 % | +0.26 pp | 6.64 % | +2.65 pp |
| DRAM Throughput (%) | 2.58 % | 2.26 % | -0.32 pp | 2.63 % | +0.05 pp | 2.00 % | -0.58 pp |
| Achieved Occupancy (%) | 30.25 % | 21.90 % | -8.35 pp | 29.97 % | -0.28 pp | 50.84 % | +20.59 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 32.00 register/thread | +14 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.70 | +0.35 |
| No Eligible (%) | 70.33 % | 58.21 % | -12.12 pp | 70.95 % | +0.62 pp | 29.66 % | -40.67 pp |
| IPC Active | 1.13 inst/cycle | 1.59 inst/cycle | +0.46 | 1.16 inst/cycle | +0.03 | 2.66 inst/cycle | +1.53 |

---

## Iteration 2

**Best so far**: 1.0199× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_policy_for_main_tensor_to_protect_channel_param_l1_residency | 1.038× | 0.990× | ✓ |  |
| B | fix_grid_lambda_to_honor_autotuned_xblock_and_simplify_fma | 0.981× | 0.841× | ✓ |  |
| C | reduce_register_pressure_via_rsqrt_and_smaller_xblock_num_warps | 1.064× | 1.055× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | fix_grid_lambda_t... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 3100 ns | +30 ns (+1.0%) | 3650 ns | +580 ns (+18.9%) | 2910 ns | -160 ns (-5.2%) |
| SM Throughput (%) | 10.05 % | 10.68 % | +0.63 pp | 16.44 % | +6.39 pp | 7.25 % | -2.80 pp |
| Memory Throughput (%) | 3.99 % | 3.92 % | -0.07 pp | 4.45 % | +0.46 pp | 4.22 % | +0.23 pp |
| DRAM Throughput (%) | 2.58 % | 2.54 % | -0.04 pp | 2.17 % | -0.41 pp | 2.70 % | +0.12 pp |
| Achieved Occupancy (%) | 30.25 % | 29.82 % | -0.43 pp | 22.27 % | -7.98 pp | 28.60 % | -1.65 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 70.33 % | 69.01 % | -1.32 pp | 59.96 % | -10.37 pp | 79.22 % | +8.89 pp |
| IPC Active | 1.13 inst/cycle | 1.14 inst/cycle | +0.01 | 1.63 inst/cycle | +0.50 | 0.76 inst/cycle | -0.37 |

---

## Iteration 3

**Best so far**: 1.055× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_to_reduce_non_fused_fp32_instructions_and_fix_grid_lambda | 1.064× | 1.066× | ✓ |  |
| B | replace_sqrt_div_sequence_with_rsqrt_and_eliminate_redundant_fma | 1.019× | 1.044× | ✓ |  |
| C | fuse_bn_normalization_with_rsqrt_and_restore_small_xblock_configs | 1.069× | 1.030× | ✓ | **✓ best** |
| D | add_triton_autotune | 1.055× | 1.055× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_to_red... | Δ | replace_sqrt_div_... | Δ | fuse_bn_normaliza... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2880 ns | -190 ns (-6.2%) | 2940 ns | -130 ns (-4.2%) | 2980 ns | -90 ns (-2.9%) | 2910 ns | -160 ns (-5.2%) |
| SM Throughput (%) | 10.05 % | 6.93 % | -3.12 pp | 7.44 % | -2.61 pp | 7.14 % | -2.91 pp | 7.08 % | -2.97 pp |
| Memory Throughput (%) | 3.99 % | 4.37 % | +0.38 pp | 4.10 % | +0.11 pp | 4.18 % | +0.19 pp | 4.37 % | +0.38 pp |
| DRAM Throughput (%) | 2.58 % | 2.74 % | +0.16 pp | 2.67 % | +0.09 pp | 2.64 % | +0.06 pp | 2.69 % | +0.11 pp |
| Achieved Occupancy (%) | 30.25 % | 29.76 % | -0.49 pp | 29.05 % | -1.20 pp | 26.89 % | -3.36 pp | 28.65 % | -1.60 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 | 0.35 | +0.00 |
| No Eligible (%) | 70.33 % | 77.44 % | +7.11 pp | 75.59 % | +5.26 pp | 78.58 % | +8.25 pp | 79.94 % | +9.61 pp |
| IPC Active | 1.13 inst/cycle | 0.71 inst/cycle | -0.42 | 0.83 inst/cycle | -0.30 | 0.75 inst/cycle | -0.38 | 0.82 inst/cycle | -0.31 |

---

## Iteration 4

**Best so far**: 1.055× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_main_tensor_load_to_protect_channel_param_l1_residency_and_fix_grid_lambda | 1.014× | 1.066× | ✓ |  |
| B | fma_fusion_and_fix_broken_autotune_grid_lambda | 1.012× | 1.010× | ✓ |  |
| C | fma_fusion_and_fix_grid_lambda_for_autotune_correctness | 1.066× | 1.077× | ✓ |  |
| D | fma_fusion_to_reduce_nonfused_fp32_instructions_in_bn_hardtanh_kernel | 1.020× | 0.951× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_ma... | Δ | fma_fusion_and_fi... | Δ | fma_fusion_and_fi... | Δ | fma_fusion_to_red... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3070 ns | 2880 ns | -190 ns (-6.2%) | 3040 ns | -30 ns (-1.0%) | 2850 ns | -220 ns (-7.2%) | 3230 ns | +160 ns (+5.2%) |
| SM Throughput (%) | 10.05 % | 7.24 % | -2.81 pp | 7.24 % | -2.81 pp | 4.88 % | -5.17 pp | 7.04 % | -3.01 pp |
| Memory Throughput (%) | 3.99 % | 4.43 % | +0.44 pp | 4.02 % | +0.03 pp | 4.25 % | +0.26 pp | 3.77 % | -0.22 pp |
| DRAM Throughput (%) | 2.58 % | 2.74 % | +0.16 pp | 2.57 % | -0.01 pp | 2.75 % | +0.17 pp | 2.46 % | -0.12 pp |
| Achieved Occupancy (%) | 30.25 % | 27.85 % | -2.40 pp | 28.50 % | -1.75 pp | 14.98 % | -15.27 pp | 28.59 % | -1.66 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 |
| Waves/SM | 0.35 | 0.35 | +0.00 | 0.35 | +0.00 | 0.17 | -0.18 | 0.35 | +0.00 |
| No Eligible (%) | 70.33 % | 75.91 % | +5.58 pp | 77.43 % | +7.10 pp | 87.63 % | +17.30 pp | 79.55 % | +9.22 pp |
| IPC Active | 1.13 inst/cycle | 0.86 inst/cycle | -0.27 | 0.89 inst/cycle | -0.24 | 0.50 inst/cycle | -0.63 | 0.77 inst/cycle | -0.36 |

---

## Iteration 5

**Best so far**: 1.055× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  3070 ns  (1.000×)
Iter 1     3010 ns  (1.020×)  — autotune_xblock_and_num_warps_for_occupancy
Iter 2     2910 ns  (1.055×)  — reduce_register_pressure_via_rsqrt_and_smaller_xblock_num_warps

Final best: 2980 ns  (1.0302×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,16 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 94080
@@ -21,18 +31,12 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = tl.math.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
-    tmp16 = tl.maximum(tl.full([XBLOCK], 0.0, tl.float32), tmp15)
+    tmp13 = tl.math.fma(tmp11, tmp12, tmp14)
+    tmp16 = tl.maximum(tl.full([XBLOCK], 0.0, tl.float32), tmp13)
     tmp17 = tl.minimum(tl.full([XBLOCK], 6.0, tl.float32), tmp16)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -42,7 +46,7 @@

     arg_4 = inputs[4]
     xnumel = 94080
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 94080, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_37[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 94080)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
