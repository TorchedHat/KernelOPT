# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24450 ns — **Best latency**: 24480 ns (**0.9988×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_11_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (35280,1,1), Waves/SM: 16.70
- **Registers/thread**: 18
- **Duration**: 24450 ns
- **SM Throughput**: 63.46 % | **Memory**: 25.43 % | **DRAM**: 15.12 %
- **Achieved Occupancy**: 61.24 %
- **IPC Active**: 2.80 inst/cycle | **No Eligible**: 29.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune | 1.001× | 0.207× | ✓ |  |
| B | autotune_xblock_tile_size_for_latency_hiding | 1.000× | 0.998× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 1.004× | 1.001× | ✓ | **✓ best** |
| D | increase_xblock_with_autotune_for_occupancy_and_fma | 0.910× | 0.900× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | autotune_xblock_t... | Δ | add_autotune_with... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24450 ns | 118240 ns | +93790 ns (+383.6%) | 24510 ns | +60 ns (+0.2%) | 24420 ns | -30 ns (-0.1%) | 27170 ns | +2720 ns (+11.1%) |
| SM Throughput (%) | 63.46 % | 85.36 % | +21.90 pp | 63.30 % | -0.16 pp | 63.72 % | +0.26 pp | 74.07 % | +10.61 pp |
| Memory Throughput (%) | 25.43 % | 18.18 % | -7.25 pp | 25.40 % | -0.03 pp | 25.54 % | +0.11 pp | 29.98 % | +4.55 pp |
| DRAM Throughput (%) | 15.12 % | 3.12 % | -12.00 pp | 15.07 % | -0.05 pp | 15.14 % | +0.02 pp | 13.59 % | -1.53 pp |
| Achieved Occupancy (%) | 61.24 % | 55.85 % | -5.39 pp | 60.54 % | -0.70 pp | 60.67 % | -0.57 pp | 66.21 % | +4.97 pp |
| Registers/Thread | 18.00 register/thread | 48.00 register/thread | +30 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 24.00 register/thread | +6 |
| Waves/SM | 16.70 | 53.45 | +36.75 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 29.81 % | 13.10 % | -16.71 pp | 29.78 % | -0.03 pp | 29.52 % | -0.29 pp | 17.49 % | -12.32 pp |
| IPC Active | 2.80 inst/cycle | 3.47 inst/cycle | +0.67 | 2.81 inst/cycle | +0.01 | 2.81 inst/cycle | +0.01 | 3.29 inst/cycle | +0.49 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune` (0.207×)

Adding @triton.autotune with larger XBLOCK configs (256–2048) caused the compiler to allocate significantly more registers per thread (18 → 48), which reduced theoretical occupancy from 100% to 62.5% (limited by register file) and actually lowered achieved occupancy (61.24% → 55.85%). The larger tile sizes also caused a 6.6x increase in executed instructions (12M → 79M) and a 4.8x increase in kernel duration (24.45 µs → 118.24 µs). When a kernel already has low register pressure (≤20 registers/thread) and high theoretical occupancy (100%), adding autotune with large XBLOCK configs can trigger the compiler to unroll more aggressively, inflating register usage and destroying occupancy.

> **Applicable when**: Never applicable as described — this is a negative result. The pattern of adding autotune with large XBLOCK configs is harmful when the kernel already has low register count (≤20 regs/thread) and 100% theoretical occupancy, because the compiler may inflate register usage for larger tile sizes.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs (>128) when: (1) registers per thread are already low (≤20), (2) theoretical occupancy is already at or near 100%, (3) the kernel is a simple pointwise op where larger tiles cause the compiler to unroll loops and inflate register pressure. The autotune overhead and register inflation will dominate any tile-size benefit.

---

## Iteration 2

**Best so far**: 1.0012× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_and_fuse_reciprocal_for_latency_hiding | 0.995× | 0.601× | ✓ |  |
| B | increase_num_warps_and_replace_division_with_rsqrt_for_fma_fusion | 1.012× | 0.834× | ✓ |  |
| C | fix_broken_grid_and_reduce_num_warps_for_occupancy | 1.014× | 0.999× | ✓ | **✓ best** |
| D | fix_broken_grid_and_remove_register_inflating_autotune | 1.011× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | fix_broken_grid_a... | Δ | fix_broken_grid_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24450 ns | 40670 ns | +16220 ns (+66.3%) | 29310 ns | +4860 ns (+19.9%) | 24480 ns | +30 ns (+0.1%) | 24130 ns | -320 ns (-1.3%) |
| SM Throughput (%) | 63.46 % | 77.02 % | +13.56 pp | 63.61 % | +0.15 pp | 31.44 % | -32.02 pp | 38.35 % | -25.11 pp |
| Memory Throughput (%) | 25.43 % | 27.48 % | +2.05 pp | 38.80 % | +13.37 pp | 11.79 % | -13.64 pp | 24.83 % | -0.60 pp |
| DRAM Throughput (%) | 15.12 % | 9.07 % | -6.05 pp | 12.58 % | -2.54 pp | 7.53 % | -7.59 pp | 15.28 % | +0.16 pp |
| Achieved Occupancy (%) | 61.24 % | 81.23 % | +19.99 pp | 75.24 % | +14.00 pp | 30.90 % | -30.34 pp | 42.20 % | -19.04 pp |
| Registers/Thread | 18.00 register/thread | 21.00 register/thread | +3 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 33.41 | +16.71 | 33.41 | +16.71 | 8.35 | -8.35 | 16.70 | +0.00 |
| No Eligible (%) | 29.81 % | 18.81 % | -11.00 pp | 31.28 % | +1.47 pp | 62.95 % | +33.14 pp | 57.07 % | +27.26 pp |
| IPC Active | 2.80 inst/cycle | 3.24 inst/cycle | +0.44 | 2.74 inst/cycle | -0.06 | 1.47 inst/cycle | -1.33 | 1.71 inst/cycle | -1.09 |

---

## Iteration 3

**Best so far**: 1.0012× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_eviction_for_main_tensor_to_reduce_l1tex_stall | 1.005× | 1.012× | ✓ |  |
| B | replace_integer_division_with_float_reciprocal_for_fma_fusion | 1.008× | 1.010× | ✓ |  |
| C | algebraic_fma_fusion_via_precomputed_bn_scale_offset | 1.011× | 0.819× | ✓ |  |
| D | replace_sqrt_plus_reciprocal_with_rsqrt_for_fma_fusion | 1.003× | 1.010× | ✓ |  |
| E | add_triton_autotune | 0.996× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_evictio... | Δ | replace_integer_d... | Δ | algebraic_fma_fus... | Δ | replace_sqrt_plus... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24450 ns | 24160 ns | -290 ns (-1.2%) | 24220 ns | -230 ns (-0.9%) | 29860 ns | +5410 ns (+22.1%) | 24220 ns | -230 ns (-0.9%) | 24130 ns | -320 ns (-1.3%) |
| SM Throughput (%) | 63.46 % | 41.21 % | -22.25 pp | 19.26 % | -44.20 pp | 63.52 % | +0.06 pp | 38.52 % | -24.94 pp | 38.49 % | -24.97 pp |
| Memory Throughput (%) | 25.43 % | 24.88 % | -0.55 pp | 12.02 % | -13.41 pp | 36.83 % | +11.40 pp | 24.95 % | -0.48 pp | 24.95 % | -0.48 pp |
| DRAM Throughput (%) | 15.12 % | 15.29 % | +0.17 pp | 7.59 % | -7.53 pp | 12.37 % | -2.75 pp | 15.25 % | +0.13 pp | 15.31 % | +0.19 pp |
| Achieved Occupancy (%) | 61.24 % | 42.43 % | -18.81 pp | 24.93 % | -36.31 pp | 75.75 % | +14.51 pp | 42.11 % | -19.13 pp | 41.97 % | -19.27 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 19.00 register/thread | +1 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 8.35 | -8.35 | 33.41 | +16.71 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 29.81 % | 53.77 % | +23.96 pp | 76.82 % | +47.01 pp | 30.94 % | +1.13 pp | 56.96 % | +27.15 pp | 56.91 % | +27.10 pp |
| IPC Active | 2.80 inst/cycle | 1.83 inst/cycle | -0.97 | 0.90 inst/cycle | -1.90 | 2.75 inst/cycle | -0.05 | 1.70 inst/cycle | -1.10 | 1.72 inst/cycle | -1.08 |

---

## Iteration 4

**Best so far**: 1.0012× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_isolated_without_autotune_block_size_change | 1.006× | 1.012× | ✓ |  |
| B | isolated_fma_fusion_with_corrected_fixed_grid_no_autotune | 1.012× | 1.010× | ✓ |  |
| C | fma_fusion_isolated_fixed_block_size | 1.005× | 1.008× | ✓ |  |
| D | fix_grid_restore_block128_and_enable_l2_compression_hint | 1.005× | 1.011× | ✓ |  |
| E | add_triton_autotune | 1.009× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_isolat... | Δ | isolated_fma_fusi... | Δ | fma_fusion_isolat... | Δ | fix_grid_restore_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24450 ns | 24160 ns | -290 ns (-1.2%) | 24220 ns | -230 ns (-0.9%) | 24260 ns | -190 ns (-0.8%) | 24190 ns | -260 ns (-1.1%) | 24220 ns | -230 ns (-0.9%) |
| SM Throughput (%) | 63.46 % | 38.15 % | -25.31 pp | 38.49 % | -24.97 pp | 38.35 % | -25.11 pp | 42.90 % | -20.56 pp | 38.46 % | -25.00 pp |
| Memory Throughput (%) | 25.43 % | 24.62 % | -0.81 pp | 24.85 % | -0.58 pp | 24.75 % | -0.68 pp | 24.76 % | -0.67 pp | 24.99 % | -0.44 pp |
| DRAM Throughput (%) | 15.12 % | 15.29 % | +0.17 pp | 15.26 % | +0.14 pp | 15.22 % | +0.10 pp | 15.26 % | +0.14 pp | 15.24 % | +0.12 pp |
| Achieved Occupancy (%) | 61.24 % | 42.44 % | -18.80 pp | 42.47 % | -18.77 pp | 43.12 % | -18.12 pp | 43.48 % | -17.76 pp | 41.97 % | -19.27 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 16.70 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 | 16.70 | +0.00 |
| No Eligible (%) | 29.81 % | 57.23 % | +27.42 pp | 57.00 % | +27.19 pp | 56.98 % | +27.17 pp | 52.08 % | +22.27 pp | 57.14 % | +27.33 pp |
| IPC Active | 2.80 inst/cycle | 1.71 inst/cycle | -1.09 | 1.71 inst/cycle | -1.09 | 1.71 inst/cycle | -1.09 | 1.90 inst/cycle | -0.90 | 1.72 inst/cycle | -1.08 |

---

## Cumulative Progress

```
Baseline:  24450 ns  (1.000×)
Iter 1     24420 ns  (1.001×)  — add_autotune_with_larger_xblock_for_vectorized_loads

Final best: 24480 ns  (0.9988×)
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
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 256}, num_warps=4),
+        triton.Config({'XBLOCK': 512}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 4515840
@@ -21,12 +32,7 @@

     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
-    tmp11 = tmp2 * tmp10
+    tmp11 = tmp2 * tl.math.rsqrt(tmp5)
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
     tmp16 = tl.full([1], 0.0, tl.float32)
@@ -34,8 +40,6 @@

     tmp18 = tl.full([1], 6.0, tl.float32)
     tmp19 = tl.minimum(tmp17, tmp18)
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +49,7 @@

     arg_4 = inputs[4]
     xnumel = 4515840
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4515840, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_11[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4515840)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
