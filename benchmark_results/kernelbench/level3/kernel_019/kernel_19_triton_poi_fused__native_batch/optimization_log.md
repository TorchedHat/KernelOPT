# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5090 ns — **Best latency**: 4830 ns (**1.0538×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_callsite_grid_lambda_and_add_l2_eviction_hint_for_broadcast_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_19_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 1.86
- **Registers/thread**: 18
- **Duration**: 5090 ns
- **SM Throughput**: 32.03 % | **Memory**: 12.17 % | **DRAM**: 8.14 %
- **Achieved Occupancy**: 61.11 %
- **IPC Active**: 2.34 inst/cycle | **No Eligible**: 39.76 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_to_eliminate_partial_wave | 0.915× | 0.879× | ✓ |  |
| B | autotune_xblock_tile_size_to_eliminate_partial_wave | 1.009× | 0.964× | ✓ | **✓ best** |
| C | increase_xblock_for_vectorized_loads_and_eliminate_partial_wave | 0.933× | 0.860× | ✓ |  |
| D | eliminate_partial_wave_via_autotune_xblock | 0.984× | 0.964× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_t... | Δ | increase_xblock_f... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5090 ns | 5790 ns | +700 ns (+13.8%) | 5280 ns | +190 ns (+3.7%) | 5920 ns | +830 ns (+16.3%) | 5280 ns | +190 ns (+3.7%) |
| SM Throughput (%) | 32.03 % | 54.20 % | +22.17 pp | 41.39 % | +9.36 pp | 53.83 % | +21.80 pp | 41.33 % | +9.30 pp |
| Memory Throughput (%) | 12.17 % | 14.66 % | +2.49 pp | 14.54 % | +2.37 pp | 14.47 % | +2.30 pp | 14.60 % | +2.43 pp |
| DRAM Throughput (%) | 8.14 % | 7.17 % | -0.97 pp | 7.90 % | -0.24 pp | 7.02 % | -1.12 pp | 7.91 % | -0.23 pp |
| Achieved Occupancy (%) | 61.11 % | 74.80 % | +13.69 pp | 69.20 % | +8.09 pp | 74.45 % | +13.34 pp | 69.39 % | +8.28 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 39.76 % | 18.17 % | -21.59 pp | 25.64 % | -14.12 pp | 19.30 % | -20.46 pp | 21.68 % | -18.08 pp |
| IPC Active | 2.34 inst/cycle | 3.16 inst/cycle | +0.82 | 3.06 inst/cycle | +0.72 | 3.18 inst/cycle | +0.84 | 3.07 inst/cycle | +0.73 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_simplify_batchnorm_arithmetic | 1.060× | 1.026× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_to_activate_xblock_selection | 0.980× | 0.941× | ✓ |  |
| C | fix_autotune_callsite_grid_to_activate_existing_configs | 0.943× | 0.875× | ✓ |  |
| D | reduce_non_fused_fp32_instructions_via_fma_rewrite | 1.048× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_call... | Δ | reduce_non_fused_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5090 ns | 4960 ns | -130 ns (-2.6%) | 5410 ns | +320 ns (+6.3%) | 5820 ns | +730 ns (+14.3%) | 4990 ns | -100 ns (-2.0%) |
| SM Throughput (%) | 32.03 % | 23.59 % | -8.44 pp | 41.60 % | +9.57 pp | 54.30 % | +22.27 pp | 30.26 % | -1.77 pp |
| Memory Throughput (%) | 12.17 % | 17.19 % | +5.02 pp | 14.70 % | +2.53 pp | 14.64 % | +2.47 pp | 18.80 % | +6.63 pp |
| DRAM Throughput (%) | 8.14 % | 8.40 % | +0.26 pp | 7.66 % | -0.48 pp | 7.11 % | -1.03 pp | 8.35 % | +0.21 pp |
| Achieved Occupancy (%) | 61.11 % | 53.13 % | -7.98 pp | 69.45 % | +8.34 pp | 73.82 % | +12.71 pp | 60.59 % | -0.52 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 39.76 % | 46.21 % | +6.45 pp | 24.12 % | -15.64 pp | 18.34 % | -21.42 pp | 32.26 % | -7.50 pp |
| IPC Active | 2.34 inst/cycle | 2.11 inst/cycle | -0.23 | 3.06 inst/cycle | +0.72 | 3.23 inst/cycle | +0.89 | 2.51 inst/cycle | +0.17 |

---

## Iteration 3

**Best so far**: 1.0262× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_to_use_autotuned_xblock_and_add_num_stages | 1.021× | 1.039× | ✓ |  |
| B | fix_grid_lambda_and_add_num_stages_to_autotune_configs | 1.021× | 1.032× | ✓ |  |
| C | persistent_kernel_sm_aligned_grid_to_eliminate_partial_wave | 0.406× | 0.292× | ✓ |  |
| D | replace_sqrt_div_with_rsqrt_and_fix_grid_lambda | 1.060× | 1.054× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_to_use_a... | Δ | fix_grid_lambda_a... | Δ | persistent_kernel... | Δ | replace_sqrt_div_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5090 ns | 4900 ns | -190 ns (-3.7%) | 4930 ns | -160 ns (-3.1%) | 17410 ns | +12320 ns (+242.0%) | 4830 ns | -260 ns (-5.1%) |
| SM Throughput (%) | 32.03 % | 23.58 % | -8.45 pp | 30.56 % | -1.47 pp | 67.01 % | +34.98 pp | 30.89 % | -1.14 pp |
| Memory Throughput (%) | 12.17 % | 17.27 % | +5.10 pp | 18.96 % | +6.79 pp | 26.41 % | +14.24 pp | 19.48 % | +7.31 pp |
| DRAM Throughput (%) | 8.14 % | 8.48 % | +0.34 pp | 8.45 % | +0.31 pp | 2.37 % | -5.77 pp | 8.62 % | +0.48 pp |
| Achieved Occupancy (%) | 61.11 % | 53.06 % | -8.05 pp | 58.33 % | -2.78 pp | 91.49 % | +30.38 pp | 56.20 % | -4.91 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 26.00 register/thread | +8 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 |
| No Eligible (%) | 39.76 % | 46.22 % | +6.46 pp | 32.18 % | -7.58 pp | 19.74 % | -20.02 pp | 31.78 % | -7.98 pp |
| IPC Active | 2.34 inst/cycle | 2.12 inst/cycle | -0.22 | 2.68 inst/cycle | +0.34 | 3.11 inst/cycle | +0.77 | 2.71 inst/cycle | +0.37 |


### Key Learnings

#### ❌ Plan C: `persistent_kernel_sm_aligned_grid_to_eliminate_partial_wave` (0.292×)

Converting a simple elementwise kernel to a persistent while-loop with a reduced grid (264 CTAs instead of 3920) caused a 9.8x increase in executed instructions (925K → 9.09M) and a 3.5x increase in kernel duration (4.96µs → 17.41µs), despite improving occupancy from 53% to 91%. The root cause is an implementation flaw: the callsite grid was left as `triton.cdiv(xnumel, 128) = 3920` instead of being changed to 264, so each of the 3920 CTAs still executed the persistent while-loop body (iterating ~2 times each), multiplying total work by ~10x. The persistent kernel pattern is only beneficial when the callsite grid is actually reduced to match the SM-aligned count; leaving the original large grid intact turns a loop-once kernel into a loop-twice kernel with no benefit.

> **Applicable when**: Kernel has a partial wave (LaunchConfiguration NCU rule fires with >20% estimated speedup), grid is large relative to SM count, and the persistent loop pattern is being applied.

> **Avoid when**: The callsite grid cannot be simultaneously reduced to match GRID_SIZE — if the grid remains at the original large value while the kernel body contains a persistent while-loop, total instruction count multiplies by the loop iteration count, causing severe regression. Also avoid when XBLOCK in the kernel body does not match the block_size used in the callsite grid calculation.

---

## Iteration 4

**Best so far**: 1.0538× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_callsite_to_use_xnumel_div_xblock_not_hardcoded_128 | 1.028× | 1.020× | ✓ |  |
| B | fuse_multiply_add_to_fma_in_batchnorm_relu_epilogue | 1.060× | 1.039× | ✓ |  |
| C | fix_callsite_grid_to_use_xblock_meta_and_add_eviction_policy | 1.018× | 1.026× | ✓ |  |
| D | fix_broken_persistent_loop_and_apply_rsqrt_fma_fusion | 0.832× | 0.856× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_callsite... | Δ | fuse_multiply_add... | Δ | fix_callsite_grid... | Δ | fix_broken_persis... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5090 ns | 4990 ns | -100 ns (-2.0%) | 4900 ns | -190 ns (-3.7%) | 4960 ns | -130 ns (-2.6%) | 5950 ns | +860 ns (+16.9%) |
| SM Throughput (%) | 32.03 % | 30.38 % | -1.65 pp | 23.62 % | -8.41 pp | 31.27 % | -0.76 pp | 51.36 % | +19.33 pp |
| Memory Throughput (%) | 12.17 % | 19.11 % | +6.94 pp | 17.29 % | +5.12 pp | 18.41 % | +6.24 pp | 24.16 % | +11.99 pp |
| DRAM Throughput (%) | 8.14 % | 8.37 % | +0.23 pp | 8.47 % | +0.33 pp | 8.34 % | +0.20 pp | 6.99 % | -1.15 pp |
| Achieved Occupancy (%) | 61.11 % | 56.47 % | -4.64 pp | 53.93 % | -7.18 pp | 59.93 % | -1.18 pp | 75.99 % | +14.88 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 3.71 | +1.85 |
| No Eligible (%) | 39.76 % | 30.88 % | -8.88 pp | 45.81 % | +6.05 pp | 32.72 % | -7.04 pp | 23.81 % | -15.95 pp |
| IPC Active | 2.34 inst/cycle | 2.71 inst/cycle | +0.37 | 2.08 inst/cycle | -0.26 | 2.84 inst/cycle | +0.50 | 2.99 inst/cycle | +0.65 |

---

## Iteration 5

**Best so far**: 1.0538× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_first_to_streaming_tensor_load_to_reduce_l1tex_stall | 1.043× | 1.006× | ✓ |  |
| B | fix_callsite_grid_lambda_to_activate_autotuned_xblock_selection | 1.052× | 1.039× | ✓ |  |
| C | fix_callsite_grid_lambda_and_add_l2_eviction_hint_for_broadcast_loads | 1.028× | 1.054× | ✓ |  |
| D | fix_callsite_grid_to_use_xblock_meta_and_reduce_num_warps_for_cpi_stall | 1.029× | 0.869× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_first_t... | Δ | fix_callsite_grid... | Δ | fix_callsite_grid... | Δ | fix_callsite_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5090 ns | 5060 ns | -30 ns (-0.6%) | 4900 ns | -190 ns (-3.7%) | 4830 ns | -260 ns (-5.1%) | 5860 ns | +770 ns (+15.1%) |
| SM Throughput (%) | 32.03 % | 25.26 % | -6.77 pp | 32.32 % | +0.29 pp | 30.97 % | -1.06 pp | 46.92 % | +14.89 pp |
| Memory Throughput (%) | 12.17 % | 17.30 % | +5.13 pp | 19.03 % | +6.86 pp | 18.45 % | +6.28 pp | 23.81 % | +11.64 pp |
| DRAM Throughput (%) | 8.14 % | 8.23 % | +0.09 pp | 8.51 % | +0.37 pp | 8.57 % | +0.43 pp | 7.11 % | -1.03 pp |
| Achieved Occupancy (%) | 61.11 % | 54.18 % | -6.93 pp | 60.42 % | -0.69 pp | 59.76 % | -1.35 pp | 54.16 % | -6.95 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 34.00 register/thread | +16 |
| Waves/SM | 1.86 | 1.86 | +0.00 | 1.86 | +0.00 | 1.86 | +0.00 | 2.47 | +0.61 |
| No Eligible (%) | 39.76 % | 43.06 % | +3.30 pp | 30.91 % | -8.85 pp | 27.01 % | -12.75 pp | 25.80 % | -13.96 pp |
| IPC Active | 2.34 inst/cycle | 2.16 inst/cycle | -0.18 | 2.82 inst/cycle | +0.48 | 2.86 inst/cycle | +0.52 | 2.94 inst/cycle | +0.60 |

---

## Cumulative Progress

```
Baseline:  5090 ns  (1.000×)
Iter 2     4960 ns  (1.026×)  — fix_autotune_grid_lambda_and_simplify_batchnorm_arithmetic
Iter 3     4830 ns  (1.054×)  — replace_sqrt_div_with_rsqrt_and_fix_grid_lambda

Final best: 4830 ns  (1.0538×)
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
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 501760
@@ -19,21 +33,11 @@

     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp11 = tmp2 * tl.math.rsqrt(tmp3 + 1e-5)
+    tmp15 = tl.math.fma(tmp11, tmp12, tmp14)
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(in_out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +47,7 @@

     arg_4 = inputs[4]
     xnumel = 501760
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 501760)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
