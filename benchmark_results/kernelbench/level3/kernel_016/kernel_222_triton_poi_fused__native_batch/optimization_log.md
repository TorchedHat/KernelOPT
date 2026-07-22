# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_222` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6820 ns — **Best latency**: 6880 ns (**0.9913×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_222`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_016/kernel_222_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (6738,1,1), Waves/SM: 3.19
- **Registers/thread**: 22
- **Duration**: 6820 ns
- **SM Throughput**: 42.76 % | **Memory**: 16.28 % | **DRAM**: 10.42 %
- **Achieved Occupancy**: 61.58 %
- **IPC Active**: 2.50 inst/cycle | **No Eligible**: 35.78 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave | 0.931× | 0.789× | ✓ |  |
| B | autotune_xblock_num_warps_for_occupancy_and_wave_efficiency | 1.020× | 0.977× | ✓ | **✓ best** |
| C | vectorized_loads_larger_xblock_autotune | 0.932× | 0.787× | ✓ |  |
| D | increase_xblock_autotune_eliminate_partial_wave | 0.954× | 0.787× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_n... | Δ | vectorized_loads_... | Δ | increase_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6820 ns | 8640 ns | +1820 ns (+26.7%) | 6980 ns | +160 ns (+2.3%) | 8670 ns | +1850 ns (+27.1%) | 8670 ns | +1850 ns (+27.1%) |
| SM Throughput (%) | 42.76 % | 63.63 % | +20.87 pp | 41.93 % | -0.83 pp | 63.11 % | +20.35 pp | 62.40 % | +19.64 pp |
| Memory Throughput (%) | 16.28 % | 17.34 % | +1.06 pp | 15.75 % | -0.53 pp | 17.18 % | +0.90 pp | 17.00 % | +0.72 pp |
| DRAM Throughput (%) | 10.42 % | 8.25 % | -2.17 pp | 10.21 % | -0.21 pp | 8.22 % | -2.20 pp | 8.20 % | -2.22 pp |
| Achieved Occupancy (%) | 61.58 % | 80.32 % | +18.74 pp | 61.71 % | +0.13 pp | 80.77 % | +19.19 pp | 80.33 % | +18.75 pp |
| Registers/Thread | 22.00 register/thread | 32.00 register/thread | +10 | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 |
| Waves/SM | 3.19 | 3.19 | +0.00 | 3.19 | +0.00 | 3.19 | +0.00 | 3.19 | +0.00 |
| No Eligible (%) | 35.78 % | 16.93 % | -18.85 pp | 35.49 % | -0.29 pp | 17.19 % | -18.59 pp | 17.14 % | -18.64 pp |
| IPC Active | 2.50 inst/cycle | 3.30 inst/cycle | +0.80 | 2.56 inst/cycle | +0.06 | 3.29 inst/cycle | +0.79 | 3.31 inst/cycle | +0.81 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_fp32_instructions_via_reciprocal_simplification | 1.020× | 0.969× | ✓ | **✓ best** |
| B | fix_autotune_grid_lambda_and_fuse_fp32_instructions | 1.017× | 0.983× | ✓ |  |
| C | fix_autotune_grid_lambda_to_enable_correct_tile_exploration | 0.950× | 0.787× | ✓ |  |
| D | fix_autotune_grid_lambda_with_register_safe_configs | 0.997× | 0.943× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_instruc... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6820 ns | 7040 ns | +220 ns (+3.2%) | 6940 ns | +120 ns (+1.8%) | 8670 ns | +1850 ns (+27.1%) | 7230 ns | +410 ns (+6.0%) |
| SM Throughput (%) | 42.76 % | 42.54 % | -0.22 pp | 37.14 % | -5.62 pp | 63.28 % | +20.52 pp | 30.64 % | -12.12 pp |
| Memory Throughput (%) | 16.28 % | 16.19 % | -0.09 pp | 24.06 % | +7.78 pp | 17.21 % | +0.93 pp | 14.23 % | -2.05 pp |
| DRAM Throughput (%) | 10.42 % | 10.12 % | -0.30 pp | 10.28 % | -0.14 pp | 8.23 % | -2.19 pp | 9.86 % | -0.56 pp |
| Achieved Occupancy (%) | 61.58 % | 61.93 % | +0.35 pp | 47.33 % | -14.25 pp | 80.39 % | +18.81 pp | 50.76 % | -10.82 pp |
| Registers/Thread | 22.00 register/thread | 22.00 register/thread | +0 | 32.00 register/thread | +10 | 32.00 register/thread | +10 | 24.00 register/thread | +2 |
| Waves/SM | 3.19 | 3.19 | +0.00 | 3.19 | +0.00 | 3.19 | +0.00 | 1.60 | -1.59 |
| No Eligible (%) | 35.78 % | 35.39 % | -0.39 pp | 37.12 % | +1.34 pp | 17.34 % | -18.44 pp | 40.37 % | +4.59 pp |
| IPC Active | 2.50 inst/cycle | 2.57 inst/cycle | +0.07 | 2.51 inst/cycle | +0.01 | 3.29 inst/cycle | +0.79 | 2.30 inst/cycle | -0.20 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_add_small_xblock_configs | 1.031× | 0.991× | ✓ | **✓ best** |
| B | evict_first_on_streaming_input_to_improve_l1_hit_rate | 1.015× | 0.987× | ✓ |  |
| C | fix_grid_lambda_and_increase_num_warps_for_warp_slot_utilization | 0.983× | 0.969× | ✓ |  |
| D | fuse_fp32_instructions_via_rsqrt_and_float_literals | 0.998× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | evict_first_on_st... | Δ | fix_grid_lambda_a... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6820 ns | 6880 ns | +60 ns (+0.9%) | 6910 ns | +90 ns (+1.3%) | 7040 ns | +220 ns (+3.2%) | 6940 ns | +120 ns (+1.8%) |
| SM Throughput (%) | 42.76 % | 32.44 % | -10.32 pp | 45.08 % | +2.32 pp | 30.67 % | -12.09 pp | 36.43 % | -6.33 pp |
| Memory Throughput (%) | 16.28 % | 16.37 % | +0.09 pp | 16.23 % | -0.05 pp | 14.45 % | -1.83 pp | 23.90 % | +7.62 pp |
| DRAM Throughput (%) | 10.42 % | 10.36 % | -0.06 pp | 10.31 % | -0.11 pp | 10.12 % | -0.30 pp | 10.28 % | -0.14 pp |
| Achieved Occupancy (%) | 61.58 % | 48.52 % | -13.06 pp | 63.78 % | +2.20 pp | 50.26 % | -11.32 pp | 49.49 % | -12.09 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 22.00 register/thread | +0 | 24.00 register/thread | +2 | 32.00 register/thread | +10 |
| Waves/SM | 3.19 | 3.19 | +0.00 | 3.19 | +0.00 | 1.60 | -1.59 | 3.19 | +0.00 |
| No Eligible (%) | 35.78 % | 61.28 % | +25.50 pp | 33.02 % | -2.76 pp | 40.71 % | +4.93 pp | 36.96 % | +1.18 pp |
| IPC Active | 2.50 inst/cycle | 1.58 inst/cycle | -0.92 | 2.71 inst/cycle | +0.21 | 2.30 inst/cycle | -0.20 | 2.41 inst/cycle | -0.09 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_bn_scale_bias_to_fuse_fp32_instructions_and_fix_grid_lambda | 1.027× | 1.010× | ✓ |  |
| C | fix_grid_lambda_and_add_high_num_warps_configs_for_latency_hiding | 0.952× | 0.927× | ✓ |  |
| D | eliminate_partial_wave_via_wave_aligned_xblock_and_correct_grid_lambda | 0.994× | 0.914× | ✓ |  |
| E | add_triton_autotune | 1.031× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_bn_sca... | Δ | fix_grid_lambda_a... | Δ | eliminate_partial... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6820 ns | 6750 ns | -70 ns (-1.0%) | 7360 ns | +540 ns (+7.9%) | 7460 ns | +640 ns (+9.4%) | 6910 ns | +90 ns (+1.3%) |
| SM Throughput (%) | 42.76 % | 33.69 % | -9.07 pp | 60.32 % | +17.56 pp | 52.17 % | +9.41 pp | 32.44 % | -10.32 pp |
| Memory Throughput (%) | 16.28 % | 17.01 % | +0.73 pp | 27.30 % | +11.02 pp | 19.28 % | +3.00 pp | 16.38 % | +0.10 pp |
| DRAM Throughput (%) | 10.42 % | 10.54 % | +0.12 pp | 9.68 % | -0.74 pp | 9.52 % | -0.90 pp | 10.31 % | -0.11 pp |
| Achieved Occupancy (%) | 61.58 % | 50.44 % | -11.14 pp | 69.82 % | +8.24 pp | 72.15 % | +10.57 pp | 47.27 % | -14.31 pp |
| Registers/Thread | 22.00 register/thread | 19.00 register/thread | -3 | 18.00 register/thread | -4 | 24.00 register/thread | +2 | 18.00 register/thread | -4 |
| Waves/SM | 3.19 | 3.19 | +0.00 | 6.38 | +3.19 | 3.19 | +0.00 | 3.19 | +0.00 |
| No Eligible (%) | 35.78 % | 59.20 % | +23.42 pp | 36.55 % | +0.77 pp | 21.91 % | -13.87 pp | 60.54 % | +24.76 pp |
| IPC Active | 2.50 inst/cycle | 1.66 inst/cycle | -0.84 | 2.46 inst/cycle | -0.04 | 3.14 inst/cycle | +0.64 | 1.59 inst/cycle | -0.91 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_maximize_warps_for_l1tex_latency_hiding | 1.031× | 0.991× | ✓ | **✓ best** |
| B | fix_grid_lambda_and_precompute_reciprocal_std_to_fuse_fp32 | 1.023× | 1.000× | ✓ |  |
| C | replace_integer_reciprocal_with_rsqrt_to_fuse_fp32_instructions | 1.026× | 1.010× | ✓ |  |
| D | channel_aligned_xblock_eliminate_modulo_and_partial_wave | 0.978× | 0.477× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_grid_lambda_a... | Δ | replace_integer_r... | Δ | channel_aligned_x... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6820 ns | 6880 ns | +60 ns (+0.9%) | 6820 ns | +0 ns (+0.0%) | 6750 ns | -70 ns (-1.0%) | 14300 ns | +7480 ns (+109.7%) |
| SM Throughput (%) | 42.76 % | 32.41 % | -10.35 pp | 16.50 % | -26.26 pp | 32.18 % | -10.58 pp | 64.90 % | +22.14 pp |
| Memory Throughput (%) | 16.28 % | 16.38 % | +0.10 pp | 8.56 % | -7.72 pp | 22.15 % | +5.87 pp | 28.43 % | +12.15 pp |
| DRAM Throughput (%) | 10.42 % | 10.34 % | -0.08 pp | 5.28 % | -5.14 pp | 10.57 % | +0.15 pp | 4.96 % | -5.46 pp |
| Achieved Occupancy (%) | 61.58 % | 46.63 % | -14.95 pp | 37.66 % | -23.92 pp | 47.65 % | -13.93 pp | 62.43 % | +0.85 pp |
| Registers/Thread | 22.00 register/thread | 18.00 register/thread | -4 | 18.00 register/thread | -4 | 23.00 register/thread | +1 | 38.00 register/thread | +16 |
| Waves/SM | 3.19 | 3.19 | +0.00 | 1.60 | -1.59 | 3.19 | +0.00 | 8.51 | +5.32 |
| No Eligible (%) | 35.78 % | 60.50 % | +24.72 pp | 72.71 % | +36.93 pp | 46.66 % | +10.88 pp | 24.05 % | -11.73 pp |
| IPC Active | 2.50 inst/cycle | 1.57 inst/cycle | -0.93 | 1.03 inst/cycle | -1.47 | 1.99 inst/cycle | -0.51 | 3.00 inst/cycle | +0.50 |

---

## Cumulative Progress

```
Baseline:  6820 ns  (1.000×)

Final best: 6880 ns  (0.9913×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,22 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 128},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=2),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 1024}, num_warps=32, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=2),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_222(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 862400
@@ -21,19 +37,13 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0, tl.int32)
     tmp17 = tl.maximum(tmp16, tmp15)
     tl.store(out_ptr0 + (x2), tmp17, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -44,7 +54,7 @@

     arg_5 = inputs[5]
     xnumel = 862400
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_relu_222[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 862400, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_relu_222[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 862400)
     return arg_5
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
