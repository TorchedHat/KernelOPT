# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4260 ns — **Best latency**: 3900 ns (**1.0923×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`precompute_bn_fused_params_with_larger_block_for_wave_doubling`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_35_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2573,1,1), Waves/SM: 1.22
- **Registers/thread**: 18
- **Duration**: 4260 ns
- **SM Throughput**: 26.13 % | **Memory**: 9.63 % | **DRAM**: 6.43 %
- **Achieved Occupancy**: 61.03 %
- **IPC Active**: 2.09 inst/cycle | **No Eligible**: 47.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_eliminate_partial_wave | 0.994× | 0.964× | ✓ |  |
| B | autotune_xblock_eliminate_partial_wave | 0.995× | 1.010× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads_and_full_waves | 0.960× | 0.394× | ✓ |  |
| D | eliminate_partial_wave_via_larger_xblock | 0.956× | 0.869× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_e... | Δ | autotune_xblock_e... | Δ | add_autotune_with... | Δ | eliminate_partial... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260 ns | 4420 ns | +160 ns (+3.8%) | 4220 ns | -40 ns (-0.9%) | 10820 ns | +6560 ns (+154.0%) | 4900 ns | +640 ns (+15.0%) |
| SM Throughput (%) | 26.13 % | 32.77 % | +6.64 pp | 25.94 % | -0.19 pp | 68.13 % | +42.00 pp | 44.36 % | +18.23 pp |
| Memory Throughput (%) | 9.63 % | 11.44 % | +1.81 pp | 9.63 % | +0.00 pp | 13.75 % | +4.12 pp | 12.15 % | +2.52 pp |
| DRAM Throughput (%) | 6.43 % | 6.20 % | -0.23 pp | 6.45 % | +0.02 pp | 2.52 % | -3.91 pp | 5.62 % | -0.81 pp |
| Achieved Occupancy (%) | 61.03 % | 71.42 % | +10.39 pp | 60.36 % | -0.67 pp | 52.84 % | -8.19 pp | 71.52 % | +10.49 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 18.00 register/thread | +0 | 48.00 register/thread | +30 | 32.00 register/thread | +14 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 1.22 | +0.00 | 3.90 | +2.68 | 1.22 | +0.00 |
| No Eligible (%) | 47.30 % | 27.12 % | -20.18 pp | 51.77 % | +4.47 pp | 17.24 % | -30.06 pp | 23.65 % | -23.65 pp |
| IPC Active | 2.09 inst/cycle | 2.84 inst/cycle | +0.75 | 1.96 inst/cycle | -0.13 | 3.35 inst/cycle | +1.26 | 2.93 inst/cycle | +0.84 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_for_l1tex_latency_hiding | 0.789× | 1.000× | ✓ |  |
| B | replace_integer_division_with_rsqrt_to_enable_fma_fusion | 1.030× | 0.986× | ✓ | **✓ best** |
| C | fix_autotune_grid_hardcode_implementation_flaw | 0.999× | 1.000× | ✓ |  |
| D | simplify_arithmetic_to_reduce_register_pressure_and_restore_occupancy | 0.993× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_integer_d... | Δ | fix_autotune_grid... | Δ | simplify_arithmet... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260 ns | 4260 ns | +0 ns (+0.0%) | 4320 ns | +60 ns (+1.4%) | 4260 ns | +0 ns (+0.0%) | 4220 ns | -40 ns (-0.9%) |
| SM Throughput (%) | 26.13 % | 25.18 % | -0.95 pp | 18.66 % | -7.47 pp | 26.15 % | +0.02 pp | 9.48 % | -16.65 pp |
| Memory Throughput (%) | 9.63 % | 9.52 % | -0.11 pp | 13.49 % | +3.86 pp | 9.69 % | +0.06 pp | 9.51 % | -0.12 pp |
| DRAM Throughput (%) | 6.43 % | 6.40 % | -0.03 pp | 6.32 % | -0.11 pp | 6.45 % | +0.02 pp | 6.50 % | +0.07 pp |
| Achieved Occupancy (%) | 61.03 % | 62.59 % | +1.56 pp | 66.73 % | +5.70 pp | 62.52 % | +1.49 pp | 41.67 % | -19.36 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 22.00 register/thread | +4 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 | 0.61 | -0.61 |
| No Eligible (%) | 47.30 % | 44.60 % | -2.70 pp | 43.98 % | -3.32 pp | 45.93 % | -1.37 pp | 65.92 % | +18.62 pp |
| IPC Active | 2.09 inst/cycle | 2.01 inst/cycle | -0.08 | 2.19 inst/cycle | +0.10 | 2.15 inst/cycle | +0.06 | 1.19 inst/cycle | -0.90 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_increase_num_warps_to_hide_l1tex_latency | 0.987× | 1.010× | ✓ |  |
| B | replace_division_chain_with_rsqrt_to_enable_fma_fusion_and_reduce_l1tex_stalls | 1.027× | 1.049× | ✓ |  |
| C | enable_fma_fusion_by_replacing_integer_reciprocal_with_float_rsqrt | 1.025× | 1.017× | ✓ |  |
| D | precompute_bn_scale_bias_to_eliminate_rsqrt_and_enable_fma_fusion | 1.033× | 1.101× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | replace_division_... | Δ | enable_fma_fusion... | Δ | precompute_bn_sca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260 ns | 4220 ns | -40 ns (-0.9%) | 4060 ns | -200 ns (-4.7%) | 4190 ns | -70 ns (-1.6%) | 3870 ns | -390 ns (-9.2%) |
| SM Throughput (%) | 26.13 % | 9.72 % | -16.41 pp | 16.11 % | -10.02 pp | 15.76 % | -10.37 pp | 13.35 % | -12.78 pp |
| Memory Throughput (%) | 9.63 % | 9.75 % | +0.12 pp | 10.19 % | +0.56 pp | 10.04 % | +0.41 pp | 10.76 % | +1.13 pp |
| DRAM Throughput (%) | 6.43 % | 6.46 % | +0.03 pp | 6.74 % | +0.31 pp | 6.54 % | +0.11 pp | 7.10 % | +0.67 pp |
| Achieved Occupancy (%) | 61.03 % | 40.46 % | -20.57 pp | 61.40 % | +0.37 pp | 59.39 % | -1.64 pp | 63.04 % | +2.01 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 1.22 | 0.61 | -0.61 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 |
| No Eligible (%) | 47.30 % | 65.73 % | +18.43 pp | 61.96 % | +14.66 pp | 66.11 % | +18.81 pp | 62.06 % | +14.76 pp |
| IPC Active | 2.09 inst/cycle | 1.21 inst/cycle | -0.88 | 1.63 inst/cycle | -0.46 | 1.55 inst/cycle | -0.54 | 1.67 inst/cycle | -0.42 |


### Key Learnings

#### ✅ Plan D: `precompute_bn_scale_bias_to_eliminate_rsqrt_and_enable_fma_fusion` (1.101×)

When a pointwise kernel applies batch normalization as `out = clamp((x - mean) * rsqrt(var + eps) * weight + bias, lo, hi)`, the mean, var, weight, and bias are channel-indexed constants reused across all spatial elements. Precomputing `scale = weight * rsqrt(var + eps)` and `bias_fused = bias - mean * scale` on the host (Python/PyTorch) before kernel launch collapses 5 loads + rsqrt + 8 FP ops into 3 loads + 1 FMA + 2 clamps. This eliminates the libdevice.rsqrt() transcendental (MUFU instruction) from the hot path, reduces registers/thread (22→16), improves achieved occupancy (41.7%→63.0%), and enables the compiler to emit a single FFMA instruction per element. The speedup is moderate (~10%) because the kernel remains latency-bound by L1TEX stalls, but the register reduction and occupancy improvement are the primary drivers.

> **Applicable when**: Kernel applies batch normalization (or any affine normalization) pointwise where mean, var, weight, bias are channel-indexed constants (not computed in-kernel). NCU FPInstructions rule fires with high non-fused FP32 instruction count. Registers/thread ≥20 and achieved occupancy <50%. The rsqrt/transcendental is the only non-FMA operation in the compute chain.

> **Avoid when**: Mean and variance are computed dynamically within the kernel (e.g., layer norm, group norm with in-kernel reduction) — precomputation is not possible. Also avoid when the two extra host-side tensor ops (scale, bias_fused) cannot be amortized (e.g., called in a tight loop with tiny batch sizes). Do not apply when the kernel is already FMA-fused and the bottleneck is DRAM bandwidth rather than compute or L1TEX stalls.

---

## Iteration 4

**Best so far**: 1.1008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_add_high_warp_configs_at_fixed_xblock_to_hide_l1tex_latency | 0.976× | 1.065× | ✓ |  |
| B | precompute_bn_scale_bias_to_eliminate_rsqrt_and_enable_fma_fusion | 1.036× | 1.065× | ✓ |  |
| C | apply_evict_first_on_streaming_store_to_improve_l1tex_hit_rate_for_channel_constants | 0.992× | 1.057× | ✓ |  |
| D | precompute_bn_fused_params_with_larger_block_for_wave_doubling | 1.081× | 1.092× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | precompute_bn_sca... | Δ | apply_evict_first... | Δ | precompute_bn_fus... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260 ns | 4000 ns | -260 ns (-6.1%) | 4000 ns | -260 ns (-6.1%) | 4030 ns | -230 ns (-5.4%) | 3900 ns | -360 ns (-8.5%) |
| SM Throughput (%) | 26.13 % | 11.31 % | -14.82 pp | 11.08 % | -15.05 pp | 17.66 % | -8.47 pp | 12.68 % | -13.45 pp |
| Memory Throughput (%) | 9.63 % | 9.62 % | -0.01 pp | 9.61 % | -0.02 pp | 10.41 % | +0.78 pp | 10.23 % | +0.60 pp |
| DRAM Throughput (%) | 6.43 % | 6.76 % | +0.33 pp | 6.84 % | +0.41 pp | 6.78 % | +0.35 pp | 7.02 % | +0.59 pp |
| Achieved Occupancy (%) | 61.03 % | 58.75 % | -2.28 pp | 56.96 % | -4.07 pp | 60.57 % | -0.46 pp | 69.67 % | +8.64 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 |
| No Eligible (%) | 47.30 % | 73.20 % | +25.90 pp | 70.60 % | +23.30 pp | 56.39 % | +9.09 pp | 55.57 % | +8.27 pp |
| IPC Active | 2.09 inst/cycle | 1.06 inst/cycle | -1.03 | 1.12 inst/cycle | -0.97 | 1.49 inst/cycle | -0.60 | 1.50 inst/cycle | -0.59 |


### Key Learnings

#### ✅ Plan B: `precompute_bn_scale_bias_to_eliminate_rsqrt_and_enable_fma_fusion` (1.101×)

When a pointwise kernel applies batch normalization as `out = clamp((x - mean) * rsqrt(var + eps) * weight + bias, lo, hi)`, the mean, var, weight, and bias are channel-indexed constants reused across all spatial elements. Precomputing `scale = weight * rsqrt(var + eps)` and `bias_fused = bias - mean * scale` on the host (Python/PyTorch) before kernel launch collapses 5 loads + rsqrt + 8 FP ops into 3 loads + 1 FMA + 2 clamps. This eliminates the libdevice.rsqrt() transcendental (MUFU instruction) from the hot path, reduces registers/thread (22→16), improves achieved occupancy (41.7%→63.0%), and enables the compiler to emit a single FFMA instruction per element. The speedup is moderate (~10%) because the kernel remains latency-bound by L1TEX stalls, but the register reduction and occupancy improvement are the primary drivers.

> **Applicable when**: Kernel applies batch normalization (or any affine normalization) pointwise where mean, var, weight, bias are channel-indexed constants (not computed in-kernel). NCU FPInstructions rule fires with high non-fused FP32 instruction count. Registers/thread ≥20 and achieved occupancy <50%. The rsqrt/transcendental is the only non-FMA operation in the compute chain.

> **Avoid when**: Mean and variance are computed dynamically within the kernel (e.g., layer norm, group norm with in-kernel reduction) — precomputation is not possible. Also avoid when the two extra host-side tensor ops (scale, bias_fused) cannot be amortized (e.g., called in a tight loop with tiny batch sizes). Do not apply when the kernel is already FMA-fused and the bottleneck is DRAM bandwidth rather than compute or L1TEX stalls.

---

## Iteration 5

**Best so far**: 1.1008× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_streaming_inout_tensor_to_reduce_l1tex_cache_pollution | 1.038× | 1.081× | ✓ |  |
| B | evict_first_on_main_tensor_to_improve_channel_constant_l1tex_hit_rate_and_fix_grid | 1.053× | 1.032× | ✓ |  |
| C | expand_autotune_config_space_with_xblock256_512_to_hide_l1tex_latency | 1.064× | 1.065× | ✓ |  |
| D | combine_precompute_bn_evict_first_and_fixed_optimal_block_config | 1.001× | 1.065× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ | evict_first_on_ma... | Δ | expand_autotune_c... | Δ | combine_precomput... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4260 ns | 3940 ns | -320 ns (-7.5%) | 4130 ns | -130 ns (-3.1%) | 4000 ns | -260 ns (-6.1%) | 4000 ns | -260 ns (-6.1%) |
| SM Throughput (%) | 26.13 % | 13.86 % | -12.27 pp | 12.57 % | -13.56 pp | 11.39 % | -14.74 pp | 12.60 % | -13.53 pp |
| Memory Throughput (%) | 9.63 % | 9.61 % | -0.02 pp | 9.36 % | -0.27 pp | 9.56 % | -0.07 pp | 9.66 % | +0.03 pp |
| DRAM Throughput (%) | 6.43 % | 6.87 % | +0.44 pp | 6.58 % | +0.15 pp | 6.80 % | +0.37 pp | 6.80 % | +0.37 pp |
| Achieved Occupancy (%) | 61.03 % | 65.22 % | +4.19 pp | 56.43 % | -4.60 pp | 55.51 % | -5.52 pp | 57.00 % | -4.03 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 16.00 register/thread | -2 | 16.00 register/thread | -2 | 16.00 register/thread | -2 |
| Waves/SM | 1.22 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 | 1.22 | +0.00 |
| No Eligible (%) | 47.30 % | 63.95 % | +16.65 pp | 69.81 % | +22.51 pp | 71.80 % | +24.50 pp | 70.39 % | +23.09 pp |
| IPC Active | 2.09 inst/cycle | 1.76 inst/cycle | -0.33 | 1.22 inst/cycle | -0.87 | 1.10 inst/cycle | -0.99 | 1.26 inst/cycle | -0.83 |

---

## Cumulative Progress

```
Baseline:  4260 ns  (1.000×)
Iter 3     3870 ns  (1.101×)  — precompute_bn_scale_bias_to_eliminate_rsqrt_and_enable_fma_fusion

Final best: 3900 ns  (1.0923×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 329280
@@ -13,29 +24,13 @@

     xmask = xindex < xnumel
     x2 = xindex
     x0 = (xindex % 672)
-    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
-    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
-    tmp2 = tmp0 - tmp1
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
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
-    tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
+    tmp0  = tl.load(in_out_ptr0 + (x2), xmask)
+    scale = tl.load(in_ptr0     + (x0), xmask, eviction_policy='evict_last')
+    bf    = tl.load(in_ptr1     + (x0), xmask, eviction_policy='evict_last')
+    out   = tmp0 * scale + bf
+    out   = tl.maximum(out, 0.0)
+    out   = tl.minimum(out, 6.0)
+    tl.store(in_out_ptr0 + (x2), out, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +40,7 @@

     arg_4 = inputs[4]
     xnumel = 329280
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 329280, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_35[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 329280)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
