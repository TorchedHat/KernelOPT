# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3970 ns — **Best latency**: 3870 ns (**1.0258×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`evict_first_on_streaming_tensor_loads_to_reduce_l1tex_stalls`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_10_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1568,1,1), Waves/SM: 0.74
- **Registers/thread**: 28
- **Duration**: 3970 ns
- **SM Throughput**: 24.82 % | **Memory**: 9.71 % | **DRAM**: 8.38 %
- **Achieved Occupancy**: 58.33 %
- **IPC Active**: 2.20 inst/cycle | **No Eligible**: 45.33 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_underutilized_pointwise_kernel | 0.838× | 0.730× | ✓ |  |
| B | autotune_xblock_increase_tile_size | 0.979× | 0.913× | ✓ |  |
| C | vectorized_loads_larger_xblock_autotune | 0.846× | 0.738× | ✓ |  |
| D | autotune_xblock_for_better_wave_utilization | 1.006× | 0.880× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_i... | Δ | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3970 ns | 5440 ns | +1470 ns (+37.0%) | 4350 ns | +380 ns (+9.6%) | 5380 ns | +1410 ns (+35.5%) | 4510 ns | +540 ns (+13.6%) |
| SM Throughput (%) | 24.82 % | 42.65 % | +17.83 pp | 36.72 % | +11.90 pp | 44.55 % | +19.73 pp | 29.00 % | +4.18 pp |
| Memory Throughput (%) | 9.71 % | 11.00 % | +1.29 pp | 12.53 % | +2.82 pp | 11.62 % | +1.91 pp | 9.47 % | -0.24 pp |
| DRAM Throughput (%) | 8.38 % | 6.07 % | -2.31 pp | 7.64 % | -0.74 pp | 6.17 % | -2.21 pp | 7.34 % | -1.04 pp |
| Achieved Occupancy (%) | 58.33 % | 44.87 % | -13.46 pp | 58.17 % | -0.16 pp | 44.47 % | -13.86 pp | 62.03 % | +3.70 pp |
| Registers/Thread | 28.00 register/thread | 54.00 register/thread | +26 | 32.00 register/thread | +4 | 54.00 register/thread | +26 | 30.00 register/thread | +2 |
| Waves/SM | 0.74 | 1.32 | +0.58 | 0.74 | +0.00 | 1.32 | +0.58 | 0.74 | +0.00 |
| No Eligible (%) | 45.33 % | 25.96 % | -19.37 pp | 29.24 % | -16.09 pp | 23.12 % | -22.21 pp | 38.43 % | -6.90 pp |
| IPC Active | 2.20 inst/cycle | 2.91 inst/cycle | +0.71 | 2.77 inst/cycle | +0.57 | 3.05 inst/cycle | +0.85 | 2.17 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_launch_grid_and_enable_fma_fusion | 0.975× | 0.906× | ✓ |  |
| B | fix_autotune_grid_lambda_and_cap_num_warps_for_correct_wave_utilization | 0.980× | 0.919× | ✓ |  |
| C | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.970× | 0.961× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_restore_occupancy | 0.957× | 0.906× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_lau... | Δ | fix_autotune_grid... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3970 ns | 4380 ns | +410 ns (+10.3%) | 4320 ns | +350 ns (+8.8%) | 4130 ns | +160 ns (+4.0%) | 4380 ns | +410 ns (+10.3%) |
| SM Throughput (%) | 24.82 % | 28.53 % | +3.71 pp | 20.90 % | -3.92 pp | 27.81 % | +2.99 pp | 28.08 % | +3.26 pp |
| Memory Throughput (%) | 9.71 % | 9.30 % | -0.41 pp | 8.36 % | -1.35 pp | 9.07 % | -0.64 pp | 9.05 % | -0.66 pp |
| DRAM Throughput (%) | 8.38 % | 7.59 % | -0.79 pp | 7.64 % | -0.74 pp | 8.02 % | -0.36 pp | 7.56 % | -0.82 pp |
| Achieved Occupancy (%) | 58.33 % | 60.03 % | +1.70 pp | 31.86 % | -26.47 pp | 57.94 % | -0.39 pp | 55.22 % | -3.11 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 30.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 45.33 % | 39.20 % | -6.13 pp | 52.71 % | +7.38 pp | 38.24 % | -7.09 pp | 42.85 % | -2.48 pp |
| IPC Active | 2.20 inst/cycle | 2.39 inst/cycle | +0.19 | 1.84 inst/cycle | -0.36 | 2.36 inst/cycle | +0.16 | 2.21 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_rsqrt_and_explicit_fma_in_batchnorm_pointwise | 1.024× | 1.137× | ✓ |  |
| B | fma_fusion_to_reduce_non_fused_fp32_instructions | 1.002× | 0.993× | ✓ |  |
| C | fma_fusion_via_batch_norm_algebraic_precompute | 1.095× | 1.026× | ✓ | **✓ best** |
| D | enable_fma_fusion_by_removing_maxnreg_cap_on_small_xblock_configs | 0.962× | 0.925× | ✓ |  |
| E | add_triton_autotune | 0.982× | 0.961× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_rs... | Δ | fma_fusion_to_red... | Δ | fma_fusion_via_ba... | Δ | enable_fma_fusion... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3970 ns | 3490 ns | -480 ns (-12.1%) | 4000 ns | +30 ns (+0.8%) | 3870 ns | -100 ns (-2.5%) | 4290 ns | +320 ns (+8.1%) | 4130 ns | +160 ns (+4.0%) |
| SM Throughput (%) | 24.82 % | 22.03 % | -2.79 pp | 11.25 % | -13.57 pp | 22.78 % | -2.04 pp | 28.19 % | +3.37 pp | 29.12 % | +4.30 pp |
| Memory Throughput (%) | 9.71 % | 11.29 % | +1.58 pp | 9.68 % | -0.03 pp | 11.45 % | +1.74 pp | 9.20 % | -0.51 pp | 9.52 % | -0.19 pp |
| DRAM Throughput (%) | 8.38 % | 9.47 % | +1.09 pp | 8.32 % | -0.06 pp | 8.57 % | +0.19 pp | 7.69 % | -0.69 pp | 8.01 % | -0.37 pp |
| Achieved Occupancy (%) | 58.33 % | 49.76 % | -8.57 pp | 29.40 % | -28.93 pp | 53.67 % | -4.66 pp | 54.00 % | -4.33 pp | 56.77 % | -1.56 pp |
| Registers/Thread | 28.00 register/thread | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 30.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.37 | -0.37 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 45.33 % | 57.12 % | +11.79 pp | 72.48 % | +27.15 pp | 63.74 % | +18.41 pp | 42.17 % | -3.16 pp | 39.36 % | -5.97 pp |
| IPC Active | 2.20 inst/cycle | 1.62 inst/cycle | -0.58 | 1.01 inst/cycle | -1.19 | 1.45 inst/cycle | -0.75 | 2.16 inst/cycle | -0.04 | 2.36 inst/cycle | +0.16 |


### Key Learnings

#### ✅ Plan A: `fma_fusion_via_rsqrt_and_explicit_fma_in_batchnorm_pointwise` (1.137×)

In fused batch-norm pointwise kernels, computing the reciprocal standard deviation via `(tl.full([1], 1, tl.int32) / tl.sqrt_rn(var_eps))` introduces an int-to-float cast that breaks the compiler's FMA pattern matcher, producing a large fraction of non-fused FP32 instructions (76% non-fused in this case). Replacing this with `libdevice.rsqrt(var_eps)` and restructuring the affine transform as `tl.math.fma(x_norm, weight, bias)` eliminates 8 intermediate temporaries per BN branch, reduces total executed instructions by 44% (959K→533K), and eliminates warp divergence entirely (branch efficiency 83%→0% divergent branches, avg active threads/warp 27.77→32.0). The net effect is a 15% reduction in kernel duration (4.13→3.49 µs) despite a slight drop in achieved occupancy (57.9%→49.8%), because the instruction-count reduction dominates.

> **Applicable when**: NCU FPInstructions rule fires with >50% non-fused FP32 instructions; kernel contains batch-norm normalization arithmetic using `tl.full([1], 1, tl.int32)` as the numerator of a reciprocal; warp divergence is present (Branch_Efficiency < 100%, Avg_Active_Threads_Per_Warp < 32); the bottleneck is instruction count rather than memory bandwidth.

> **Avoid when**: The kernel is already memory-bandwidth bound (DRAM Throughput >80%); the BN reciprocal is already computed via a pure float path (no int32 literals); numerical precision requirements forbid the single-rounding FMA semantics of `tl.math.fma`; `libdevice` is unavailable in the target Triton version.

---

## Iteration 4

**Best so far**: 1.0258× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_streaming_tensor_loads_to_reduce_l1tex_stalls | 1.050× | 1.118× | ✓ |  |
| B | evict_first_on_streaming_tensor_loads_to_reduce_l1tex_stalls | 1.094× | 1.026× | ✓ |  |
| C | increase_num_warps_per_block_to_hide_l1tex_latency | 1.053× | 1.097× | ✓ |  |
| D | collapse_bn_reciprocal_chain_via_rsqrt_and_fma | 0.989× | 1.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | evict_first_on_st... | Δ | increase_num_warp... | Δ | collapse_bn_recip... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3970 ns | 3550 ns | -420 ns (-10.6%) | 3870 ns | -100 ns (-2.5%) | 3620 ns | -350 ns (-8.8%) | 3780 ns | -190 ns (-4.8%) |
| SM Throughput (%) | 24.82 % | 22.18 % | -2.64 pp | 19.55 % | -5.27 pp | 23.51 % | -1.31 pp | 22.11 % | -2.71 pp |
| Memory Throughput (%) | 9.71 % | 11.27 % | +1.56 pp | 9.76 % | +0.05 pp | 11.04 % | +1.33 pp | 11.29 % | +1.58 pp |
| DRAM Throughput (%) | 8.38 % | 9.35 % | +0.97 pp | 8.55 % | +0.17 pp | 4.63 % | -3.75 pp | 8.72 % | +0.34 pp |
| Achieved Occupancy (%) | 58.33 % | 49.16 % | -9.17 pp | 58.44 % | +0.11 pp | 54.89 % | -3.44 pp | 52.96 % | -5.37 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 30.00 register/thread | +2 | 32.00 register/thread | +4 | 30.00 register/thread | +2 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 45.33 % | 54.22 % | +8.89 pp | 54.58 % | +9.25 pp | 54.80 % | +9.47 pp | 58.79 % | +13.46 pp |
| IPC Active | 2.20 inst/cycle | 1.64 inst/cycle | -0.56 | 1.57 inst/cycle | -0.63 | 1.65 inst/cycle | -0.55 | 1.65 inst/cycle | -0.55 |

---

## Iteration 5

**Best so far**: 1.0258× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_slice_rebalancing_via_program_id_swizzling | 1.070× | 1.062× | ✓ |  |
| B | fold_bn_centering_into_fma_to_eliminate_residual_non_fused_fp32 | 1.061× | 1.161× | ✓ |  |
| C | fix_launch_grid_and_add_num_stages_pipelining_to_autotune | 0.982× | 1.097× | ✓ |  |
| D | swizzle_program_id_for_l2_slice_balance | 1.006× | 0.978× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_slice_rebalanc... | Δ | fold_bn_centering... | Δ | fix_launch_grid_a... | Δ | swizzle_program_i... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3970 ns | 3740 ns | -230 ns (-5.8%) | 3420 ns | -550 ns (-13.9%) | 3620 ns | -350 ns (-8.8%) | 4060 ns | +90 ns (+2.3%) |
| SM Throughput (%) | 24.82 % | 22.63 % | -2.19 pp | 21.15 % | -3.67 pp | 20.91 % | -3.91 pp | 20.27 % | -4.55 pp |
| Memory Throughput (%) | 9.71 % | 11.80 % | +2.09 pp | 9.87 % | +0.16 pp | 10.63 % | +0.92 pp | 10.33 % | +0.62 pp |
| DRAM Throughput (%) | 8.38 % | 8.84 % | +0.46 pp | 4.83 % | -3.55 pp | 9.20 % | +0.82 pp | 8.10 % | -0.28 pp |
| Achieved Occupancy (%) | 58.33 % | 52.94 % | -5.39 pp | 54.15 % | -4.18 pp | 53.85 % | -4.48 pp | 54.67 % | -3.66 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 30.00 register/thread | +2 | 30.00 register/thread | +2 | 28.00 register/thread | +0 |
| Waves/SM | 0.74 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 | 0.74 | +0.00 |
| No Eligible (%) | 45.33 % | 65.22 % | +19.89 pp | 55.47 % | +10.14 pp | 58.09 % | +12.76 pp | 67.47 % | +22.14 pp |
| IPC Active | 2.20 inst/cycle | 1.35 inst/cycle | -0.85 | 1.76 inst/cycle | -0.44 | 1.63 inst/cycle | -0.57 | 1.24 inst/cycle | -0.96 |

---

## Cumulative Progress

```
Baseline:  3970 ns  (1.000×)
Iter 3     3870 ns  (1.026×)  — fma_fusion_via_batch_norm_algebraic_precompute

Final best: 3870 ns  (1.0258×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,49 +5,51 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4, num_stages=2, maxnreg=32),  # known-good baseline
+        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2, maxnreg=32),
+        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2, maxnreg=32),
+        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2, maxnreg=32),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2, maxnreg=32),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2, maxnreg=32),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
     xnumel = 200704
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 128)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp16 = tl.load(in_ptr4 + (x2), None)
-    tmp17 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
-    tmp19 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
-    tmp25 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
-    tmp27 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
+    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
+    tmp16 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_first')
+    tmp17 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
+    tmp19 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
+    tmp25 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
+    tmp27 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
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
+    inv_std1 = libdevice.rsqrt(tmp5)
+    x_norm1 = tmp2 * inv_std1
+    tmp15 = tl.math.fma(x_norm1, tmp12, tmp14)
     tmp18 = tmp16 - tmp17
     tmp20 = tmp19 + tmp4
-    tmp21 = tl.sqrt_rn(tmp20)
-    tmp22 = (tmp7 / tmp21)
-    tmp23 = tmp22 * tmp9
-    tmp24 = tmp18 * tmp23
-    tmp26 = tmp24 * tmp25
-    tmp28 = tmp26 + tmp27
+    inv_std2 = libdevice.rsqrt(tmp20)
+    x_norm2 = tmp18 * inv_std2
+    tmp28 = tl.math.fma(x_norm2, tmp25, tmp27)
     tmp29 = tmp15 + tmp28
     tmp30 = tl.full([1], 0, tl.int32)
     tmp31 = tl.maximum(tmp30, tmp29)
-    tl.store(in_out_ptr0 + (x2), tmp31, None)
-
-
+    tl.store(in_out_ptr0 + (x2), tmp31, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -62,7 +64,7 @@

     arg_9 = inputs[9]
     xnumel = 200704
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 200704, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_add_relu_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 200704)
     return arg_9
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
