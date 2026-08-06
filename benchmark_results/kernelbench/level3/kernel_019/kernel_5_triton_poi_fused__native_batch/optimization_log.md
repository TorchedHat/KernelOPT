# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12420 ns — **Best latency**: 12130 ns (**1.0239×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_num_warps_to_hide_l1tex_latency_without_register_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_019/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12420 ns
- **SM Throughput**: 47.21 % | **Memory**: 20.89 % | **DRAM**: 13.30 %
- **Achieved Occupancy**: 55.94 %
- **IPC Active**: 2.36 inst/cycle | **No Eligible**: 40.47 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_latency_hiding | 0.935× | 0.935× | ✓ |  |
| B | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 0.776× | 0.738× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.775× | 0.738× | ✓ |  |
| D | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.786× | 0.739× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12420 ns | 13280 ns | +860 ns (+6.9%) | 16830 ns | +4410 ns (+35.5%) | 16830 ns | +4410 ns (+35.5%) | 16800 ns | +4380 ns (+35.3%) |
| SM Throughput (%) | 47.21 % | 67.37 % | +20.16 pp | 76.52 % | +29.31 pp | 76.50 % | +29.29 pp | 76.67 % | +29.46 pp |
| Memory Throughput (%) | 20.89 % | 25.53 % | +4.64 pp | 21.97 % | +1.08 pp | 21.98 % | +1.09 pp | 22.02 % | +1.13 pp |
| DRAM Throughput (%) | 13.30 % | 12.34 % | -0.96 pp | 9.73 % | -3.57 pp | 9.73 % | -3.57 pp | 9.75 % | -3.55 pp |
| Achieved Occupancy (%) | 55.94 % | 69.12 % | +13.18 pp | 84.94 % | +29.00 pp | 85.18 % | +29.24 pp | 85.30 % | +29.36 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.47 % | 15.55 % | -24.92 pp | 13.19 % | -27.28 pp | 13.22 % | -27.25 pp | 13.08 % | -27.39 pp |
| IPC Active | 2.36 inst/cycle | 3.36 inst/cycle | +1.00 | 3.45 inst/cycle | +1.09 | 3.45 inst/cycle | +1.09 | 3.43 inst/cycle | +1.07 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_redundant_fp32_ops_and_promote_fma_fusion | 1.004× | 1.003× | ✓ |  |
| B | eliminate_non_fused_fp32_instructions_via_rsqrt_and_fma | 0.971× | 0.978× | ✓ |  |
| C | algebraic_simplification_to_reduce_non_fused_fp32_instructions | 1.010× | 1.026× | ✓ |  |
| D | fuse_fp32_instructions_via_fma_and_rsqrt_remove_autotune_overhead | 1.011× | 1.013× | ✓ |  |
| E | add_triton_autotune | 1.012× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eliminate_redunda... | Δ | eliminate_non_fus... | Δ | algebraic_simplif... | Δ | fuse_fp32_instruc... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12420 ns | 12380 ns | -40 ns (-0.3%) | 12700 ns | +280 ns (+2.3%) | 12100 ns | -320 ns (-2.6%) | 12260 ns | -160 ns (-1.3%) | 12320 ns | -100 ns (-0.8%) |
| SM Throughput (%) | 47.21 % | 38.52 % | -8.69 pp | 46.83 % | -0.38 pp | 32.84 % | -14.37 pp | 31.70 % | -15.51 pp | 56.31 % | +9.10 pp |
| Memory Throughput (%) | 20.89 % | 28.72 % | +7.83 pp | 30.53 % | +9.64 pp | 21.06 % | +0.17 pp | 20.87 % | -0.02 pp | 21.25 % | +0.36 pp |
| DRAM Throughput (%) | 13.30 % | 13.25 % | -0.05 pp | 13.01 % | -0.29 pp | 13.53 % | +0.23 pp | 13.36 % | +0.06 pp | 13.29 % | -0.01 pp |
| Achieved Occupancy (%) | 55.94 % | 35.38 % | -20.56 pp | 41.52 % | -14.42 pp | 44.39 % | -11.55 pp | 42.76 % | -13.18 pp | 59.88 % | +3.94 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 32.00 register/thread | +15 | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.47 % | 48.36 % | +7.89 pp | 39.08 % | -1.39 pp | 58.55 % | +18.08 pp | 65.98 % | +25.51 pp | 30.39 % | -10.08 pp |
| IPC Active | 2.36 inst/cycle | 2.04 inst/cycle | -0.32 | 2.41 inst/cycle | +0.05 | 1.64 inst/cycle | -0.72 | 1.33 inst/cycle | -1.03 | 2.77 inst/cycle | +0.41 |

---

## Iteration 3

**Best so far**: 1.0081× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_without_register_pressure | 1.026× | 1.024× | ✓ | **✓ best** |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 1.014× | 1.026× | ✓ |  |
| C | replace_integer_div_sqrt_with_libdevice_rsqrt_and_remove_redundant_multiply | 1.005× | 1.013× | ✓ |  |
| D | fix_autotune_grid_and_add_streaming_eviction_policy | 0.957× | 0.975× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | replace_integer_d... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12420 ns | 12130 ns | -290 ns (-2.3%) | 12100 ns | -320 ns (-2.6%) | 12260 ns | -160 ns (-1.3%) | 12740 ns | +320 ns (+2.6%) |
| SM Throughput (%) | 47.21 % | 32.48 % | -14.73 pp | 31.98 % | -15.23 pp | 34.24 % | -12.97 pp | 49.29 % | +2.08 pp |
| Memory Throughput (%) | 20.89 % | 20.82 % | -0.07 pp | 21.09 % | +0.20 pp | 21.32 % | +0.43 pp | 30.55 % | +9.66 pp |
| DRAM Throughput (%) | 13.30 % | 13.53 % | +0.23 pp | 13.55 % | +0.25 pp | 13.38 % | +0.08 pp | 12.88 % | -0.42 pp |
| Achieved Occupancy (%) | 55.94 % | 45.14 % | -10.80 pp | 41.74 % | -14.20 pp | 43.24 % | -12.70 pp | 43.27 % | -12.67 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.47 % | 58.66 % | +18.19 pp | 65.78 % | +25.31 pp | 56.93 % | +16.46 pp | 37.61 % | -2.86 pp |
| IPC Active | 2.36 inst/cycle | 1.63 inst/cycle | -0.73 | 1.35 inst/cycle | -1.01 | 1.73 inst/cycle | -0.63 | 2.50 inst/cycle | +0.14 |

---

## Iteration 4

**Best so far**: 1.0239× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_for_main_tensor_to_preserve_l1_for_channel_params | 1.026× | 1.006× | ✓ |  |
| B | replace_sqrt_rcp_with_rsqrt_and_fuse_multiply_add_to_fma | 1.013× | 1.030× | ✓ |  |
| C | preload_channel_params_into_shared_memory_to_eliminate_l1tex_stalls | 0.991× | 1.003× | ✓ |  |
| D | fix_autotune_grid_lambda_to_enable_correct_xblock_exploration | 0.971× | 0.966× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | replace_sqrt_rcp_... | Δ | preload_channel_p... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12420 ns | 12350 ns | -70 ns (-0.6%) | 12060 ns | -360 ns (-2.9%) | 12380 ns | -40 ns (-0.3%) | 12860 ns | +440 ns (+3.5%) |
| SM Throughput (%) | 47.21 % | 31.35 % | -15.86 pp | 31.96 % | -15.25 pp | 78.26 % | +31.05 pp | 48.95 % | +1.74 pp |
| Memory Throughput (%) | 20.89 % | 20.79 % | -0.10 pp | 21.08 % | +0.19 pp | 50.89 % | +30.00 pp | 30.54 % | +9.65 pp |
| DRAM Throughput (%) | 13.30 % | 13.28 % | -0.02 pp | 13.59 % | +0.29 pp | 13.26 % | -0.04 pp | 12.74 % | -0.56 pp |
| Achieved Occupancy (%) | 55.94 % | 42.59 % | -13.35 pp | 41.61 % | -14.33 pp | 59.34 % | +3.40 pp | 42.66 % | -13.28 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.47 % | 66.09 % | +25.62 pp | 65.92 % | +25.45 pp | 52.74 % | +12.27 pp | 35.89 % | -4.58 pp |
| IPC Active | 2.36 inst/cycle | 1.36 inst/cycle | -1.00 | 1.35 inst/cycle | -1.01 | 1.94 inst/cycle | -0.42 | 2.49 inst/cycle | +0.13 |

---

## Iteration 5

**Best so far**: 1.0239× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | double_warps_per_block_with_fixed_xblock256_to_hide_l1tex_latency | 1.014× | 1.026× | ✓ |  |
| B | 2d_tiling_hoist_channel_rsqrt_to_reduce_redundant_fp32_and_l1tex_stalls | 0.948× | 0.951× | ✓ |  |
| C | replace_tl_gather_with_direct_indexed_global_load_to_eliminate_l1tex_stalls | 1.026× | 1.013× | ✓ |  |
| D | fix_grid_lambda_to_match_autotune_xblock_selection | 0.970× | 0.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | double_warps_per_... | Δ | 2d_tiling_hoist_c... | Δ | replace_tl_gather... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12420 ns | 12100 ns | -320 ns (-2.6%) | 13060 ns | +640 ns (+5.2%) | 12260 ns | -160 ns (-1.3%) | 12830 ns | +410 ns (+3.3%) |
| SM Throughput (%) | 47.21 % | 31.78 % | -15.43 pp | 26.23 % | -20.98 pp | 31.83 % | -15.38 pp | 49.84 % | +2.63 pp |
| Memory Throughput (%) | 20.89 % | 20.89 % | +0.00 pp | 15.85 % | -5.04 pp | 21.18 % | +0.29 pp | 30.89 % | +10.00 pp |
| DRAM Throughput (%) | 13.30 % | 13.55 % | +0.25 pp | 12.55 % | -0.75 pp | 13.36 % | +0.06 pp | 12.78 % | -0.52 pp |
| Achieved Occupancy (%) | 55.94 % | 41.80 % | -14.14 pp | 45.18 % | -10.76 pp | 42.28 % | -13.66 pp | 41.07 % | -14.87 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 20.00 register/thread | +3 | 18.00 register/thread | +1 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 3.71 | -3.71 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.47 % | 66.32 % | +25.85 pp | 60.41 % | +19.94 pp | 66.01 % | +25.54 pp | 35.93 % | -4.54 pp |
| IPC Active | 2.36 inst/cycle | 1.36 inst/cycle | -1.00 | 1.51 inst/cycle | -0.85 | 1.35 inst/cycle | -1.01 | 2.55 inst/cycle | +0.19 |

---

## Cumulative Progress

```
Baseline:  12420 ns  (1.000×)
Iter 2     12320 ns  (1.008×)  — add_triton_autotune
Iter 3     12130 ns  (1.024×)  — increase_num_warps_to_hide_l1tex_latency_without_register_pressure

Final best: 12130 ns  (1.0239×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,25 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
     x2 = xindex
     x0 = (xindex % 64)
-    tmp0 = tl.load(in_out_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_out_ptr0 + (x2))
+    tmp1 = tl.load(in_ptr0 + (x0), eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr1 + (x0), eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr2 + (x0), eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr3 + (x0), eviction_policy='evict_last')
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
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
+    inv_std = libdevice.rsqrt(tmp3 + 1e-5)
+    normalized = tmp2 * inv_std
+    result = tl.fma(normalized, tmp12, tmp14)
+    out = tl.maximum(0.0, result)
+    tl.store(in_out_ptr0 + (x2), out)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
