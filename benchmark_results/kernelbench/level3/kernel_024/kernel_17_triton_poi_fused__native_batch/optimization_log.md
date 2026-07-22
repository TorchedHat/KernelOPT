# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2590 ns — **Best latency**: 2620 ns (**0.9885×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_17_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14,1,1), Waves/SM: 0.01
- **Registers/thread**: 18
- **Duration**: 2590 ns
- **SM Throughput**: 0.22 % | **Memory**: 0.51 % | **DRAM**: 0.20 %
- **Achieved Occupancy**: 6.07 %
- **IPC Active**: 0.24 inst/cycle | **No Eligible**: 92.62 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_grid_coverage_via_autotune_smaller_xblock | 1.048× | 0.974× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_tiny_pointwise_kernel | 1.022× | 0.932× | ✓ |  |
| C | vectorized_loads_stores_with_larger_xblock | 0.972× | 0.881× | ✓ |  |
| D | autotune_xblock_for_sm_coverage | 1.051× | 1.036× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_grid_cov... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | autotune_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2660 ns | +70 ns (+2.7%) | 2780 ns | +190 ns (+7.3%) | 2940 ns | +350 ns (+13.5%) | 2500 ns | -90 ns (-3.5%) |
| SM Throughput (%) | 0.22 % | 0.12 % | -0.10 pp | 0.27 % | +0.05 pp | 0.20 % | -0.02 pp | 0.06 % | -0.16 pp |
| Memory Throughput (%) | 0.51 % | 0.42 % | -0.09 pp | 0.46 % | -0.05 pp | 0.54 % | +0.03 pp | 0.45 % | -0.06 pp |
| DRAM Throughput (%) | 0.20 % | 0.07 % | -0.13 pp | 0.18 % | -0.02 pp | 0.18 % | -0.02 pp | 0.07 % | -0.13 pp |
| Achieved Occupancy (%) | 6.07 % | 3.08 % | -2.99 pp | 6.05 % | -0.02 pp | 3.11 % | -2.96 pp | 1.56 % | -4.51 pp |
| Registers/Thread | 18.00 register/thread | 21.00 register/thread | +3 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 19.00 register/thread | +1 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 92.62 % | 92.34 % | -0.28 pp | 91.16 % | -1.46 pp | 86.64 % | -5.98 pp | 93.42 % | +0.80 pp |
| IPC Active | 0.24 inst/cycle | 0.13 inst/cycle | -0.11 | 0.43 inst/cycle | +0.19 | 0.26 inst/cycle | +0.02 | 0.06 inst/cycle | -0.18 |

---

## Iteration 2

**Best so far**: 1.036× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_sm_coverage_with_smaller_xblock_and_more_warps | 1.015× | 0.932× | ✓ |  |
| B | increase_xblock_to_single_wave_full_tile | 0.336× | 1.024× | ✓ |  |
| C | fix_launch_grid_to_respect_autotune_xblock_and_cover_full_workload | 0.970× | 0.932× | ✓ |  |
| D | increase_sm_coverage_via_smaller_xblock_and_fewer_warps | 1.052× | 0.989× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_sm_cover... | Δ | increase_xblock_t... | Δ | fix_launch_grid_t... | Δ | increase_sm_cover... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2780 ns | +190 ns (+7.3%) | 2530 ns | -60 ns (-2.3%) | 2780 ns | +190 ns (+7.3%) | 2620 ns | +30 ns (+1.2%) |
| SM Throughput (%) | 0.22 % | 0.06 % | -0.16 pp | 0.12 % | -0.10 pp | 0.29 % | +0.07 pp | 0.06 % | -0.16 pp |
| Memory Throughput (%) | 0.51 % | 0.35 % | -0.16 pp | 0.40 % | -0.11 pp | 0.47 % | -0.04 pp | 0.39 % | -0.12 pp |
| DRAM Throughput (%) | 0.20 % | 0.05 % | -0.15 pp | 0.07 % | -0.13 pp | 0.19 % | -0.01 pp | 0.07 % | -0.13 pp |
| Achieved Occupancy (%) | 6.07 % | 1.56 % | -4.51 pp | 3.09 % | -2.98 pp | 6.07 % | +0.00 pp | 1.56 % | -4.51 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 21.00 register/thread | +3 | 24.00 register/thread | +6 | 19.00 register/thread | +1 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 92.62 % | 92.40 % | -0.22 pp | 92.27 % | -0.35 pp | 89.13 % | -3.49 pp | 93.57 % | +0.95 pp |
| IPC Active | 0.24 inst/cycle | 0.06 inst/cycle | -0.18 | 0.12 inst/cycle | -0.12 | 0.41 inst/cycle | +0.17 | 0.07 inst/cycle | -0.17 |

---

## Iteration 3

**Best so far**: 1.036× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_launch_grid_to_use_autotuned_xblock_and_single_tile_config | 1.052× | 1.093× | ✓ |  |
| B | fix_dynamic_grid_and_add_full_tile_autotune_config | 1.019× | 1.066× | ✓ |  |
| C | fix_launch_grid_and_add_small_xblock_configs_for_sm_coverage | 1.032× | 0.974× | ✓ |  |
| D | fix_launch_grid_and_add_small_xblock_configs_for_sm_coverage | 1.014× | 1.066× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_t... | Δ | fix_dynamic_grid_... | Δ | fix_launch_grid_a... | Δ | fix_launch_grid_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2370 ns | -220 ns (-8.5%) | 2430 ns | -160 ns (-6.2%) | 2660 ns | +70 ns (+2.7%) | 2430 ns | -160 ns (-6.2%) |
| SM Throughput (%) | 0.22 % | 0.12 % | -0.10 pp | 0.06 % | -0.16 pp | 0.07 % | -0.15 pp | 0.06 % | -0.16 pp |
| Memory Throughput (%) | 0.51 % | 0.43 % | -0.08 pp | 0.48 % | -0.03 pp | 0.39 % | -0.12 pp | 0.48 % | -0.03 pp |
| DRAM Throughput (%) | 0.20 % | 0.07 % | -0.13 pp | 0.07 % | -0.13 pp | 0.04 % | -0.16 pp | 0.11 % | -0.09 pp |
| Achieved Occupancy (%) | 6.07 % | 3.09 % | -2.98 pp | 1.56 % | -4.51 pp | 1.56 % | -4.51 pp | 1.56 % | -4.51 pp |
| Registers/Thread | 18.00 register/thread | 21.00 register/thread | +3 | 19.00 register/thread | +1 | 20.00 register/thread | +2 | 18.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 | 0.00 | -0.01 |
| No Eligible (%) | 92.62 % | 94.13 % | +1.51 pp | 92.30 % | -0.32 pp | 95.58 % | +2.96 pp | 94.32 % | +1.70 pp |
| IPC Active | 0.24 inst/cycle | 0.11 inst/cycle | -0.13 | 0.07 inst/cycle | -0.17 | 0.06 inst/cycle | -0.18 | 0.06 inst/cycle | -0.18 |

---

## Iteration 4

**Best so far**: 1.036× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_arithmetic_and_fix_grid_lambda_with_lower_num_warps | 1.011× | 1.066× | ✓ |  |
| B | static_single_tile_with_instruction_simplification | 1.031× | 1.093× | ✓ |  |
| C | fix_grid_lambda_and_simplify_kernel_instructions | 1.016× | 1.079× | ✓ |  |
| D | increase_tile_to_single_cta_and_eliminate_redundant_instructions | 1.011× | 0.503× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | static_single_til... | Δ | fix_grid_lambda_a... | Δ | increase_tile_to_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2590 ns | 2430 ns | -160 ns (-6.2%) | 2370 ns | -220 ns (-8.5%) | 2400 ns | -190 ns (-7.3%) | 5150 ns | +2560 ns (+98.8%) |
| SM Throughput (%) | 0.22 % | 0.06 % | -0.16 pp | 0.25 % | +0.03 pp | 0.06 % | -0.16 pp | 0.79 % | +0.57 pp |
| Memory Throughput (%) | 0.51 % | 0.46 % | -0.05 pp | 0.57 % | +0.06 pp | 0.55 % | +0.04 pp | 0.35 % | -0.16 pp |
| DRAM Throughput (%) | 0.20 % | 0.07 % | -0.13 pp | 0.22 % | +0.02 pp | 0.11 % | -0.09 pp | 0.13 % | -0.07 pp |
| Achieved Occupancy (%) | 6.07 % | 1.56 % | -4.51 pp | 6.04 % | -0.03 pp | 1.56 % | -4.51 pp | 6.17 % | +0.10 pp |
| Registers/Thread | 18.00 register/thread | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 84.00 register/thread | +66 |
| Waves/SM | 0.01 | 0.00 | -0.01 | 0.01 | +0.00 | 0.00 | -0.01 | 0.02 | +0.01 |
| No Eligible (%) | 92.62 % | 94.25 % | +1.63 pp | 93.24 % | +0.62 pp | 94.32 % | +1.70 pp | 75.33 % | -17.29 pp |
| IPC Active | 0.24 inst/cycle | 0.07 inst/cycle | -0.17 | 0.25 inst/cycle | +0.01 | 0.06 inst/cycle | -0.18 | 0.99 inst/cycle | +0.75 |


### Key Learnings

#### ❌ Plan D: `increase_tile_to_single_cta_and_eliminate_redundant_instructions` (0.503×)

Replacing small-XBLOCK autotune configs with large-XBLOCK configs (e.g., XBLOCK=2048, num_warps=4) on a tiny kernel (xnumel=1728) caused registers per thread to explode from 21 to 84, collapsing theoretical occupancy from 100% to 31.25% and Block Limit Registers from 42 to 5. Despite the intent to collapse 14 blocks into 1 CTA, the massive register pressure caused the autotuner to select XBLOCK=128 (not 2048), keeping grid_size=14 while dramatically increasing instruction count (2408→31084) and duration (2.37→5.15 µs). For tiny kernels where the bottleneck is launch configuration rather than compute, forcing large XBLOCK configs with reduced num_warps does not guarantee single-CTA execution and risks severe register pressure regression if the compiler cannot keep all live values in registers.

> **Applicable when**: Kernel has tiny xnumel (<2048), registers per thread are already low (~21), theoretical occupancy is near 100%, and you are considering adding very large XBLOCK configs (>=512) with reduced num_warps to force single-CTA execution.

> **Avoid when**: Kernel already has high register pressure (>40 regs/thread) or low theoretical occupancy; kernel has large xnumel where large XBLOCK is needed for tiling efficiency; the register count per thread is known to stay low regardless of XBLOCK (e.g., simple elementwise ops with no accumulation).

---

## Cumulative Progress

```
Baseline:  2590 ns  (1.000×)
Iter 1     2500 ns  (1.036×)  — autotune_xblock_for_sm_coverage

Final best: 2620 ns  (0.9885×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,19 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16},   num_warps=2),
+        triton.Config({'XBLOCK': 32},   num_warps=4),
+        triton.Config({'XBLOCK': 64},   num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+        triton.Config({'XBLOCK': 2048}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1728
@@ -34,8 +47,6 @@

     tmp18 = (tmp17 / tmp9)
     tl.store(in_out_ptr0 + (x2), tmp18, xmask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
@@ -44,7 +55,7 @@

     arg_4 = inputs[4]
     xnumel = 1728
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1728, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_17[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 1728)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
