# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12260 ns — **Best latency**: 12160 ns (**1.0082×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`increase_num_stages_for_async_prefetch_to_hide_l1tex_latency`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_22_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12260 ns
- **SM Throughput**: 47.59 % | **Memory**: 21.04 % | **DRAM**: 13.37 %
- **Achieved Occupancy**: 55.46 %
- **IPC Active**: 2.34 inst/cycle | **No Eligible**: 40.45 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.914× | 0.724× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 1.001× | 0.982× | ✓ | **✓ best** |
| C | vectorized_loads_stores_via_larger_xblock_autotune | 0.992× | 0.919× | ✓ |  |
| D | increase_xblock_for_latency_hiding_via_autotune | 0.982× | 0.916× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | vectorized_loads_... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12260 ns | 16930 ns | +4670 ns (+38.1%) | 12480 ns | +220 ns (+1.8%) | 13340 ns | +1080 ns (+8.8%) | 13380 ns | +1120 ns (+9.1%) |
| SM Throughput (%) | 47.59 % | 76.19 % | +28.60 pp | 56.05 % | +8.46 pp | 66.21 % | +18.62 pp | 66.73 % | +19.14 pp |
| Memory Throughput (%) | 21.04 % | 21.84 % | +0.80 pp | 21.18 % | +0.14 pp | 25.33 % | +4.29 pp | 25.51 % | +4.47 pp |
| DRAM Throughput (%) | 13.37 % | 9.69 % | -3.68 pp | 13.14 % | -0.23 pp | 12.29 % | -1.08 pp | 12.25 % | -1.12 pp |
| Achieved Occupancy (%) | 55.46 % | 85.36 % | +29.90 pp | 58.95 % | +3.49 pp | 69.97 % | +14.51 pp | 69.59 % | +14.13 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 18.00 register/thread | +1 | 24.00 register/thread | +7 | 24.00 register/thread | +7 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.45 % | 13.21 % | -27.24 pp | 30.68 % | -9.77 pp | 16.31 % | -24.14 pp | 16.93 % | -23.52 pp |
| IPC Active | 2.34 inst/cycle | 3.45 inst/cycle | +1.11 | 2.73 inst/cycle | +0.39 | 3.37 inst/cycle | +1.03 | 3.36 inst/cycle | +1.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_and_constrain_safe_xblock_configs | 1.012× | 0.995× | ✓ |  |
| B | fix_autotune_grid_lambda_and_eliminate_redundant_fp_ops | 0.980× | 0.919× | ✓ |  |
| C | fix_hardcoded_grid_and_fuse_fp32_instructions | 0.992× | 0.919× | ✓ |  |
| D | reduce_register_pressure_and_enable_fma_fusion_via_small_xblock_arithmetic_rewrite | 1.025× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_hardcoded_gri... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12260 ns | 12320 ns | +60 ns (+0.5%) | 13340 ns | +1080 ns (+8.8%) | 13340 ns | +1080 ns (+8.8%) | 12160 ns | -100 ns (-0.8%) |
| SM Throughput (%) | 47.59 % | 56.50 % | +8.91 pp | 66.35 % | +18.76 pp | 66.87 % | +19.28 pp | 19.53 % | -28.06 pp |
| Memory Throughput (%) | 21.04 % | 21.35 % | +0.31 pp | 25.16 % | +4.12 pp | 25.56 % | +4.52 pp | 19.37 % | -1.67 pp |
| DRAM Throughput (%) | 13.37 % | 13.32 % | -0.05 pp | 12.29 % | -1.08 pp | 12.28 % | -1.09 pp | 13.49 % | +0.12 pp |
| Achieved Occupancy (%) | 55.46 % | 60.00 % | +4.54 pp | 71.57 % | +16.11 pp | 69.13 % | +13.67 pp | 29.59 % | -25.87 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 24.00 register/thread | +7 | 24.00 register/thread | +7 | 22.00 register/thread | +5 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 3.71 | -3.71 |
| No Eligible (%) | 40.45 % | 30.44 % | -10.01 pp | 15.68 % | -24.77 pp | 16.39 % | -24.06 pp | 71.73 % | +31.28 pp |
| IPC Active | 2.34 inst/cycle | 2.77 inst/cycle | +0.43 | 3.30 inst/cycle | +0.96 | 3.33 inst/cycle | +0.99 | 1.09 inst/cycle | -1.25 |

---

## Iteration 3

**Best so far**: 1.0082× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_for_latency_hiding | 1.021× | 1.013× | ✓ |  |
| B | eliminate_redundant_fp_ops_to_enable_fma_fusion | 1.021× | 1.000× | ✓ |  |
| C | fix_hardcoded_grid_and_replace_sqrt_div_with_rsqrt_for_fma_fusion | 1.001× | 0.998× | ✓ |  |
| D | eliminate_redundant_fp32_multiply_to_enable_fma_fusion_and_reduce_register_pressure | 1.002× | 0.998× | ✓ |  |
| E | add_triton_autotune | 1.022× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | eliminate_redunda... | Δ | fix_hardcoded_gri... | Δ | eliminate_redunda... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12260 ns | 12100 ns | -160 ns (-1.3%) | 12260 ns | +0 ns (+0.0%) | 12290 ns | +30 ns (+0.2%) | 12290 ns | +30 ns (+0.2%) | 12100 ns | -160 ns (-1.3%) |
| SM Throughput (%) | 47.59 % | 34.05 % | -13.54 pp | 33.89 % | -13.70 pp | 38.46 % | -9.13 pp | 49.76 % | +2.17 pp | 34.35 % | -13.24 pp |
| Memory Throughput (%) | 21.04 % | 21.15 % | +0.11 pp | 21.04 % | +0.00 pp | 28.59 % | +7.55 pp | 28.68 % | +7.64 pp | 21.32 % | +0.28 pp |
| DRAM Throughput (%) | 13.37 % | 13.57 % | +0.20 pp | 13.38 % | +0.01 pp | 13.37 % | +0.00 pp | 13.35 % | -0.02 pp | 13.56 % | +0.19 pp |
| Achieved Occupancy (%) | 55.46 % | 42.21 % | -13.25 pp | 42.31 % | -13.15 pp | 35.28 % | -20.18 pp | 38.89 % | -16.57 pp | 43.20 % | -12.26 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 22.00 register/thread | +5 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.45 % | 57.06 % | +16.61 pp | 57.84 % | +17.39 pp | 48.31 % | +7.86 pp | 34.87 % | -5.58 pp | 57.54 % | +17.09 pp |
| IPC Active | 2.34 inst/cycle | 1.69 inst/cycle | -0.65 | 1.70 inst/cycle | -0.64 | 2.01 inst/cycle | -0.33 | 2.60 inst/cycle | +0.26 | 1.71 inst/cycle | -0.63 |

---

## Iteration 4

**Best so far**: 1.0082× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_num_stages_for_l1tex_latency_hiding | 1.011× | 1.017× | ✓ |  |
| B | increase_num_stages_for_async_prefetch_to_hide_l1tex_latency | 1.022× | 1.008× | ✓ |  |
| C | fix_grid_lambda_and_add_num_stages_pipelining | 1.001× | 0.990× | ✓ |  |
| D | fix_grid_to_respect_autotune_xblock_and_use_rsqrt_for_fma | 1.001× | 0.980× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | increase_num_stag... | Δ | fix_grid_lambda_a... | Δ | fix_grid_to_respe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12260 ns | 12060 ns | -200 ns (-1.6%) | 12160 ns | -100 ns (-0.8%) | 12380 ns | +120 ns (+1.0%) | 12510 ns | +250 ns (+2.0%) |
| SM Throughput (%) | 47.59 % | 33.83 % | -13.76 pp | 34.28 % | -13.31 pp | 38.39 % | -9.20 pp | 38.44 % | -9.15 pp |
| Memory Throughput (%) | 21.04 % | 21.01 % | -0.03 pp | 21.33 % | +0.29 pp | 28.58 % | +7.54 pp | 28.55 % | +7.51 pp |
| DRAM Throughput (%) | 13.37 % | 13.57 % | +0.20 pp | 13.49 % | +0.12 pp | 13.24 % | -0.13 pp | 13.11 % | -0.26 pp |
| Achieved Occupancy (%) | 55.46 % | 43.46 % | -12.00 pp | 44.18 % | -11.28 pp | 34.71 % | -20.75 pp | 35.36 % | -20.10 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 22.00 register/thread | +5 | 22.00 register/thread | +5 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.45 % | 57.58 % | +17.13 pp | 56.68 % | +16.23 pp | 48.41 % | +7.96 pp | 48.80 % | +8.35 pp |
| IPC Active | 2.34 inst/cycle | 1.71 inst/cycle | -0.63 | 1.69 inst/cycle | -0.65 | 2.03 inst/cycle | -0.31 | 2.01 inst/cycle | -0.33 |

---

## Cumulative Progress

```
Baseline:  12260 ns  (1.000×)
Iter 2     12160 ns  (1.008×)  — reduce_register_pressure_and_enable_fma_fusion_via_small_xblock_arithmetic_rewrite

Final best: 12160 ns  (1.0082×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=8),
+        triton.Config({'XBLOCK': 256}, num_warps=16),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 256)
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
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    inv_std = libdevice.rsqrt(tmp5)
+    normed = tmp2 * inv_std
+    scaled = normed * tmp12
+    shifted = scaled + tmp14
+    zero = tl.full([1], 0.0, tl.float32)
+    result = tl.maximum(zero, shifted)
+    tl.store(in_out_ptr0 + (x2), result, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -43,7 +49,7 @@

     arg_4 = inputs[4]
     xnumel = 2007040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2007040)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
