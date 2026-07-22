# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_216` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6690 ns — **Best latency**: 6590 ns (**1.0152×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_at_fixed_xblock256_to_double_active_warps_per_sm`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_216`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_216_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6370,1,1), Waves/SM: 3.02
- **Registers/thread**: 22
- **Duration**: 6690 ns
- **SM Throughput**: 41.67 % | **Memory**: 15.88 % | **DRAM**: 10.07 %
- **Achieved Occupancy**: 61.52 %
- **IPC Active**: 2.52 inst/cycle | **No Eligible**: 37.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.972× | 0.964× | ✓ |  |
| B | autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy | 1.022× | 1.011× | ✓ | **✓ best** |
| C | vectorized_loads_stores_with_larger_xblock | 0.994× | 0.954× | ✓ |  |
| D | autotune_xblock_and_num_warps_for_occupancy | 0.936× | 0.807× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_t... | Δ | vectorized_loads_... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6940 ns | +250 ns (+3.7%) | 6620 ns | -70 ns (-1.0%) | 7010 ns | +320 ns (+4.8%) | 8290 ns | +1600 ns (+23.9%) |
| SM Throughput (%) | 41.67 % | 51.74 % | +10.07 pp | 41.86 % | +0.19 pp | 51.90 % | +10.23 pp | 62.81 % | +21.14 pp |
| Memory Throughput (%) | 15.88 % | 19.08 % | +3.20 pp | 15.96 % | +0.08 pp | 19.17 % | +3.29 pp | 17.12 % | +1.24 pp |
| DRAM Throughput (%) | 10.07 % | 9.64 % | -0.43 pp | 10.17 % | +0.10 pp | 9.64 % | -0.43 pp | 8.12 % | -1.95 pp |
| Achieved Occupancy (%) | 61.52 % | 71.96 % | +10.44 pp | 60.38 % | -1.14 pp | 72.80 % | +11.28 pp | 79.70 % | +18.18 pp |
| Registers/Thread | 22.00 register/thread | 24.00 register/thread | +2 | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 32.00 register/thread | +10 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 |
| No Eligible (%) | 37.10 % | 21.30 % | -15.80 pp | 35.69 % | -1.41 pp | 21.97 % | -15.13 pp | 16.94 % | -20.16 pp |
| IPC Active | 2.52 inst/cycle | 3.09 inst/cycle | +0.57 | 2.48 inst/cycle | -0.04 | 3.13 inst/cycle | +0.61 | 3.30 inst/cycle | +0.78 |


### Key Learnings

#### ❌ Plan D: `autotune_xblock_and_num_warps_for_occupancy` (0.807×)

Increasing XBLOCK from 128 to larger values (512–2048) via @triton.autotune raised achieved occupancy from 61.5% to 79.7% and reduced warp stall cycles slightly, but caused a 1.88x increase in total executed instructions (2.17M → 4.08M) and a jump in registers per thread from 22 to 32. The net effect was a 24% longer kernel duration (6.69 µs → 8.29 µs), indicating the compute overhead of processing more elements per block outweighed the latency-hiding benefit. When a kernel already has moderate occupancy and the bottleneck is compute (SM throughput 41.67%), enlarging tiles increases instruction count without proportionally reducing memory latency exposure, causing regression.

> **Applicable when**: Kernel has very low occupancy (<40%) AND is clearly memory-bandwidth bound (DRAM throughput >60%) AND instruction count is low — meaning larger tiles would amortize memory latency without proportionally increasing compute work.

> **Avoid when**: Kernel is already compute-bound or mixed (SM throughput >40%), registers per thread are already moderate (>20), and the primary stall is L1TEX scoreboard rather than pure memory bandwidth. In these cases, larger XBLOCK increases instruction count and register pressure, causing regression even if occupancy improves.

---

## Iteration 2

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_to_pipeline_l1tex_loads | 1.025× | 1.004× | ✓ |  |
| B | fix_grid_lambda_and_reduce_num_warps_to_eliminate_partial_wave | 1.014× | 0.981× | ✓ |  |
| C | eliminate_partial_wave_via_grid_alignment | 0.994× | 0.959× | ✓ |  |
| D | replace_integer_reciprocal_with_fp32_rsqrt_for_fma_fusion | 1.031× | 1.011× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | fix_grid_lambda_a... | Δ | eliminate_partial... | Δ | replace_integer_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6660 ns | -30 ns (-0.4%) | 6820 ns | +130 ns (+1.9%) | 6980 ns | +290 ns (+4.3%) | 6620 ns | -70 ns (-1.0%) |
| SM Throughput (%) | 41.67 % | 41.87 % | +0.20 pp | 30.11 % | -11.56 pp | 51.81 % | +10.14 pp | 35.76 % | -5.91 pp |
| Memory Throughput (%) | 15.88 % | 16.00 % | +0.12 pp | 14.15 % | -1.73 pp | 19.14 % | +3.26 pp | 23.14 % | +7.26 pp |
| DRAM Throughput (%) | 10.07 % | 10.10 % | +0.03 pp | 9.89 % | -0.18 pp | 9.66 % | -0.41 pp | 10.13 % | +0.06 pp |
| Achieved Occupancy (%) | 61.52 % | 63.71 % | +2.19 pp | 50.51 % | -11.01 pp | 72.52 % | +11.00 pp | 50.11 % | -11.41 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 24.00 register/thread | +2 | 32.00 register/thread | +10 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 1.51 | -1.51 | 3.02 | +0.00 | 3.02 | +0.00 |
| No Eligible (%) | 37.10 % | 35.63 % | -1.47 pp | 41.08 % | +3.98 pp | 22.17 % | -14.93 pp | 39.91 % | +2.81 pp |
| IPC Active | 2.52 inst/cycle | 2.43 inst/cycle | -0.09 | 2.28 inst/cycle | -0.24 | 3.13 inst/cycle | +0.61 | 2.43 inst/cycle | -0.09 |

---

## Iteration 3

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_to_enable_tile_selection | 1.032× | 1.020× | ✓ |  |
| B | fuse_fp32_instructions_via_libdevice_rsqrt_and_fma_rewrite | 1.034× | 1.011× | ✓ | **✓ best** |
| C | replace_integer_div_reciprocal_with_fp32_rsqrt_for_fma_fusion | 1.033× | 1.000× | ✓ |  |
| D | fuse_non_fused_fp32_instructions_via_rsqrt_and_fma_rewrite | 1.030× | 1.020× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | fuse_fp32_instruc... | Δ | replace_integer_d... | Δ | fuse_non_fused_fp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6560 ns | -130 ns (-1.9%) | 6620 ns | -70 ns (-1.0%) | 6690 ns | +0 ns (+0.0%) | 6560 ns | -130 ns (-1.9%) |
| SM Throughput (%) | 41.67 % | 31.70 % | -9.97 pp | 31.17 % | -10.50 pp | 16.55 % | -25.12 pp | 31.43 % | -10.24 pp |
| Memory Throughput (%) | 15.88 % | 16.07 % | +0.19 pp | 15.86 % | -0.02 pp | 15.97 % | +0.09 pp | 21.51 % | +5.63 pp |
| DRAM Throughput (%) | 10.07 % | 10.23 % | +0.16 pp | 10.20 % | +0.13 pp | 10.05 % | -0.02 pp | 10.30 % | +0.23 pp |
| Achieved Occupancy (%) | 61.52 % | 47.95 % | -13.57 pp | 48.78 % | -12.74 pp | 41.18 % | -20.34 pp | 45.37 % | -16.15 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 23.00 register/thread | +1 | 23.00 register/thread | +1 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 3.02 | +0.00 | 1.51 | -1.51 | 3.02 | +0.00 |
| No Eligible (%) | 37.10 % | 61.70 % | +24.60 pp | 60.01 % | +22.91 pp | 65.94 % | +28.84 pp | 46.85 % | +9.75 pp |
| IPC Active | 2.52 inst/cycle | 1.56 inst/cycle | -0.96 | 1.58 inst/cycle | -0.94 | 1.22 inst/cycle | -1.30 | 2.10 inst/cycle | -0.42 |

---

## Iteration 4

**Best so far**: 1.0106× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_via_precomputed_bn_scale_and_float_relu | 1.023× | 1.015× | ✓ |  |
| B | align_grid_to_full_waves_and_fix_autotune_launch_lambda | 1.034× | 1.004× | ✓ |  |
| C | evict_first_on_streaming_input_to_improve_bn_param_l1_residency | 1.024× | 1.011× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_scoreboard_stall | 1.034× | 1.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | align_grid_to_ful... | Δ | evict_first_on_st... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6590 ns | -100 ns (-1.5%) | 6660 ns | -30 ns (-0.4%) | 6620 ns | -70 ns (-1.0%) | 6590 ns | -100 ns (-1.5%) |
| SM Throughput (%) | 41.67 % | 32.67 % | -9.00 pp | 31.88 % | -9.79 pp | 31.51 % | -10.16 pp | 31.76 % | -9.91 pp |
| Memory Throughput (%) | 15.88 % | 16.52 % | +0.64 pp | 16.01 % | +0.13 pp | 15.87 % | -0.01 pp | 16.04 % | +0.16 pp |
| DRAM Throughput (%) | 10.07 % | 10.17 % | +0.10 pp | 10.15 % | +0.08 pp | 10.17 % | +0.10 pp | 10.22 % | +0.15 pp |
| Achieved Occupancy (%) | 61.52 % | 49.42 % | -12.10 pp | 48.69 % | -12.83 pp | 49.82 % | -11.70 pp | 47.56 % | -13.96 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 18.00 register/thread | -4 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 |
| No Eligible (%) | 37.10 % | 56.94 % | +19.84 pp | 59.77 % | +22.67 pp | 56.29 % | +19.19 pp | 59.70 % | +22.60 pp |
| IPC Active | 2.52 inst/cycle | 1.69 inst/cycle | -0.83 | 1.59 inst/cycle | -0.93 | 1.71 inst/cycle | -0.81 | 1.58 inst/cycle | -0.94 |

---

## Iteration 5

**Best so far**: 1.0152× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock256_to_double_active_warps_per_sm | 1.037× | 1.015× | ✓ | **✓ best** |
| B | precompute_bn_scale_shift_to_expose_fma_and_reduce_instruction_count | 1.011× | 1.029× | ✓ |  |
| D | evict_first_on_streaming_activation_to_improve_bn_param_l1_residency | 1.008× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | precompute_bn_sca... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6690 ns | 6590 ns | -100 ns (-1.5%) | 6500 ns | -190 ns (-2.8%) | 6620 ns | -70 ns (-1.0%) |
| SM Throughput (%) | 41.67 % | 32.06 % | -9.61 pp | 32.71 % | -8.96 pp | 31.47 % | -10.20 pp |
| Memory Throughput (%) | 15.88 % | 16.23 % | +0.35 pp | 16.54 % | +0.66 pp | 15.90 % | +0.02 pp |
| DRAM Throughput (%) | 10.07 % | 10.26 % | +0.19 pp | 10.30 % | +0.23 pp | 10.15 % | +0.08 pp |
| Achieved Occupancy (%) | 61.52 % | 49.59 % | -11.93 pp | 50.27 % | -11.25 pp | 49.08 % | -12.44 pp |
| Registers/Thread | 22.00 register/thread | 20.00 register/thread | -2 | 21.00 register/thread | -1 | 18.00 register/thread | -4 |
| Waves/SM | 3.02 | 3.02 | +0.00 | 3.02 | +0.00 | 3.02 | +0.00 |
| No Eligible (%) | 37.10 % | 57.13 % | +20.03 pp | 59.99 % | +22.89 pp | 56.73 % | +19.63 pp |
| IPC Active | 2.52 inst/cycle | 1.72 inst/cycle | -0.80 | 1.66 inst/cycle | -0.86 | 1.71 inst/cycle | -0.81 |

---

## Cumulative Progress

```
Baseline:  6690 ns  (1.000×)
Iter 1     6620 ns  (1.011×)  — autotune_xblock_to_eliminate_partial_wave_and_improve_occupancy
Iter 4     6590 ns  (1.015×)  — increase_num_warps_to_hide_l1tex_scoreboard_stall

Final best: 6590 ns  (1.0152×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_216(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 815360
@@ -21,19 +36,11 @@

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
+    tmp10 = tl.math.rsqrt(tmp5)
+    tmp_scale = tmp10 * tmp12
+    tmp15 = tl.math.fma(tmp2, tmp_scale, tmp14)
+    tmp17 = tl.maximum(tl.full([1], 0.0, tl.float32), tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +51,7 @@

     arg_5 = inputs[5]
     xnumel = 815360
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_216[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 815360, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_216[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 815360)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
