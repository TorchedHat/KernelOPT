# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17180 ns — **Best latency**: 16960 ns (**1.0130×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`replace_sqrt_reciprocal_with_rsqrt_to_reduce_nonfused_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_008/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 17
- **Duration**: 17180 ns
- **SM Throughput**: 50.98 % | **Memory**: 22.65 % | **DRAM**: 14.32 %
- **Achieved Occupancy**: 56.08 %
- **IPC Active**: 2.36 inst/cycle | **No Eligible**: 40.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.730× | 0.704× | ✓ |  |
| B | increase_xblock_with_autotune_to_hide_l1tex_latency | 0.921× | 0.913× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_latency_hiding | 0.924× | 0.702× | ✓ |  |
| D | autotune_xblock_tile_size | 0.726× | 0.699× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | increase_xblock_w... | Δ | increase_xblock_f... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17180 ns | 24420 ns | +7240 ns (+42.1%) | 18820 ns | +1640 ns (+9.5%) | 24480 ns | +7300 ns (+42.5%) | 24580 ns | +7400 ns (+43.1%) |
| SM Throughput (%) | 50.98 % | 78.98 % | +28.00 pp | 70.53 % | +19.55 pp | 79.00 % | +28.02 pp | 78.69 % | +27.71 pp |
| Memory Throughput (%) | 22.65 % | 23.28 % | +0.63 pp | 27.36 % | +4.71 pp | 23.22 % | +0.57 pp | 23.05 % | +0.40 pp |
| DRAM Throughput (%) | 14.32 % | 10.12 % | -4.20 pp | 13.05 % | -1.27 pp | 10.04 % | -4.28 pp | 9.98 % | -4.34 pp |
| Achieved Occupancy (%) | 56.08 % | 86.52 % | +30.44 pp | 69.23 % | +13.15 pp | 86.64 % | +30.56 pp | 87.05 % | +30.97 pp |
| Registers/Thread | 17.00 register/thread | 32.00 register/thread | +15 | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 32.00 register/thread | +15 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.67 % | 13.72 % | -26.95 pp | 16.29 % | -24.38 pp | 14.44 % | -26.23 pp | 14.01 % | -26.66 pp |
| IPC Active | 2.36 inst/cycle | 3.44 inst/cycle | +1.08 | 3.32 inst/cycle | +0.96 | 3.44 inst/cycle | +1.08 | 3.41 inst/cycle | +1.05 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.704×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) nearly doubled registers per thread (17→32) and more than doubled executed instructions (6.77M→14.98M), causing a 42% latency regression despite occupancy improving from 56% to 87%. The instruction explosion indicates the autotune selected a config that forced the compiler to spill or unroll more aggressively, overwhelming the compute pipeline. When the original kernel is already register-efficient (≤17 registers/thread) and the bottleneck is L1TEX latency rather than occupancy, adding autotune with large tile configs can backfire by increasing register pressure and instruction count faster than the latency hiding benefit accrues.

> **Applicable when**: Never apply this specific pattern (autotune with XBLOCK 512–2048 on a kernel that was originally register-efficient at XBLOCK=128). The anti-pattern is: adding autotune with large tile configs to a low-register kernel where xnumel is divisible by the original tile size.

> **Avoid when**: Do not add @triton.autotune with large XBLOCK configs when: (1) the original kernel uses ≤20 registers/thread, (2) xnumel is already evenly divisible by the current XBLOCK (no masking needed), and (3) the bottleneck is L1TEX latency stalls rather than insufficient occupancy. The autotune overhead and compiler register pressure increase can cause 2x+ instruction bloat and net regression even when occupancy improves.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_non_fused_fp32_instructions_via_fma_rewrite_and_fix_autotune_grid | 1.000× | 1.009× | ✓ |  |
| B | replace_integer_reciprocal_with_fp32_rsqrt_to_enable_fma_fusion | 1.006× | 1.019× | ✓ | **✓ best** |
| C | replace_non_fused_fp32_arithmetic_with_explicit_fma | 0.917× | 0.706× | ✓ |  |
| D | remove_broken_autotune_and_fix_grid_to_restore_register_efficiency | 0.996× | 0.994× | ✓ |  |
| E | add_triton_autotune | 0.994× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_non_fused_fp... | Δ | replace_integer_r... | Δ | replace_non_fused... | Δ | remove_broken_aut... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17180 ns | 17020 ns | -160 ns (-0.9%) | 16860 ns | -320 ns (-1.9%) | 24350 ns | +7170 ns (+41.7%) | 17280 ns | +100 ns (+0.6%) | 17090 ns | -90 ns (-0.5%) |
| SM Throughput (%) | 50.98 % | 40.90 % | -10.08 pp | 36.26 % | -14.72 pp | 79.04 % | +28.06 pp | 50.55 % | -0.43 pp | 60.30 % | +9.32 pp |
| Memory Throughput (%) | 22.65 % | 30.85 % | +8.20 pp | 22.63 % | -0.02 pp | 23.30 % | +0.65 pp | 22.49 % | -0.16 pp | 22.75 % | +0.10 pp |
| DRAM Throughput (%) | 14.32 % | 14.44 % | +0.12 pp | 14.56 % | +0.24 pp | 10.09 % | -4.23 pp | 14.23 % | -0.09 pp | 14.36 % | +0.04 pp |
| Achieved Occupancy (%) | 56.08 % | 33.44 % | -22.64 pp | 42.51 % | -13.57 pp | 86.68 % | +30.60 pp | 56.75 % | +0.67 pp | 60.89 % | +4.81 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 18.00 register/thread | +1 | 32.00 register/thread | +15 | 17.00 register/thread | +0 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.67 % | 49.44 % | +8.77 pp | 57.42 % | +16.75 pp | 13.69 % | -26.98 pp | 41.46 % | +0.79 pp | 29.87 % | -10.80 pp |
| IPC Active | 2.36 inst/cycle | 1.98 inst/cycle | -0.38 | 1.71 inst/cycle | -0.65 | 3.44 inst/cycle | +1.08 | 2.32 inst/cycle | -0.04 | 2.80 inst/cycle | +0.44 |


### Key Learnings

#### ❌ Plan C: `replace_non_fused_fp32_arithmetic_with_explicit_fma` (0.706×)

Rewriting the batch-norm arithmetic chain to use tl.math.fma and eliminating int32 intermediates was paired with a change to the autotune config list (adding larger XBLOCK configs: 512, 1024, 2048), which caused the compiler to select a larger tile and increase registers per thread from 24 to 32. This reduced Block Limit Registers from 21 to 16, and more than doubled executed instructions (10.35M → 14.98M), increasing duration by 29% (18.82 → 24.35 µs) despite occupancy improving from 69% to 87%. The FMA rewrite itself did not reduce instruction count as expected — the NCU FPInstructions rule still flagged the kernel after the change (fused: 1.13M, non-fused: 3.39M vs before: fused 0.56M, non-fused 1.69M), indicating the autotune config change drove instruction count up, not down. When the autotune config list is modified alongside arithmetic rewrites, the autotuner may select a larger tile that overwhelms any arithmetic savings.

> **Applicable when**: Never apply this pattern as described. This is a negative result. The FPInstructions NCU rule flagged non-fused FP32 arithmetic, but the attempted fix (tl.math.fma + autotune config changes) caused regression.

> **Avoid when**: Do not simultaneously change autotune config lists and arithmetic rewrite strategies. Do not add larger XBLOCK configs (1024, 2048) when the kernel already uses 24+ registers/thread — this pushes Block Limit Registers down (21→16) and causes instruction count to grow faster than any FMA savings. Do not use tl.math.fma as a standalone optimization when the kernel's bottleneck is instruction count driven by tile size selection, not arithmetic fusion.

---

## Iteration 3

**Best so far**: 1.019× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 0.996× | 1.019× | ✓ |  |
| B | fix_autotune_grid_lambda_with_register_safe_configs | 0.995× | 1.015× | ✓ |  |
| C | replace_sqrt_reciprocal_with_rsqrt_to_reduce_nonfused_fp32_instructions | 1.006× | 1.013× | ✓ | **✓ best** |
| D | revert_to_register_efficient_fixed_xblock_to_eliminate_instruction_bloat | 0.986× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_autotune_grid... | Δ | replace_sqrt_reci... | Δ | revert_to_registe... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17180 ns | 16860 ns | -320 ns (-1.9%) | 16930 ns | -250 ns (-1.5%) | 16960 ns | -220 ns (-1.3%) | 17120 ns | -60 ns (-0.3%) |
| SM Throughput (%) | 50.98 % | 35.65 % | -15.33 pp | 36.17 % | -14.81 pp | 34.22 % | -16.76 pp | 50.92 % | -0.06 pp |
| Memory Throughput (%) | 22.65 % | 22.26 % | -0.39 pp | 22.56 % | -0.09 pp | 22.37 % | -0.28 pp | 22.64 % | -0.01 pp |
| DRAM Throughput (%) | 14.32 % | 14.55 % | +0.23 pp | 14.53 % | +0.21 pp | 14.47 % | +0.15 pp | 14.36 % | +0.04 pp |
| Achieved Occupancy (%) | 56.08 % | 42.07 % | -14.01 pp | 42.25 % | -13.83 pp | 41.01 % | -15.07 pp | 56.42 % | +0.34 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 17.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.67 % | 57.50 % | +16.83 pp | 57.78 % | +17.11 pp | 66.58 % | +25.91 pp | 40.85 % | +0.18 pp |
| IPC Active | 2.36 inst/cycle | 1.68 inst/cycle | -0.68 | 1.70 inst/cycle | -0.66 | 1.33 inst/cycle | -1.03 | 2.38 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.019× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | preload_channel_scalars_into_registers_to_eliminate_repeated_l1tex_loads | 0.992× | 1.002× | ✓ |  |
| B | restructure_to_2d_grid_to_eliminate_modulo_and_improve_coalescing | 0.030× | 0.028× | ✓ |  |
| C | set_evict_first_on_large_streaming_tensor_to_improve_l1_cache_for_channel_vectors | 1.000× | 1.009× | ✓ |  |
| D | precompute_bn_scale_bias_scalars_to_reduce_per_element_loads_and_fp_ops | 1.002× | 1.019× | ✓ |  |


### Metric Comparison

| Metric | Baseline | preload_channel_s... | Δ | restructure_to_2d... | Δ | set_evict_first_o... | Δ | precompute_bn_sca... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17180 ns | 17150 ns | -30 ns (-0.2%) | 617380 ns | +600200 ns (+3493.6%) | 17020 ns | -160 ns (-0.9%) | 16860 ns | -320 ns (-1.9%) |
| SM Throughput (%) | 50.98 % | 79.22 % | +28.24 pp | 1.06 % | -49.92 pp | 34.34 % | -16.64 pp | 34.13 % | -16.85 pp |
| Memory Throughput (%) | 22.65 % | 54.03 % | +31.38 pp | 7.90 % | -14.75 pp | 22.37 % | -0.28 pp | 22.20 % | -0.45 pp |
| DRAM Throughput (%) | 14.32 % | 14.31 % | -0.01 pp | 0.01 % | -14.31 pp | 14.45 % | +0.13 pp | 14.56 % | +0.24 pp |
| Achieved Occupancy (%) | 56.08 % | 57.35 % | +1.27 pp | 85.80 % | +29.72 pp | 41.68 % | -14.40 pp | 41.26 % | -14.82 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 16.00 register/thread | -1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.67 % | 50.53 % | +9.86 pp | 99.29 % | +58.62 pp | 62.93 % | +22.26 pp | 65.83 % | +25.16 pp |
| IPC Active | 2.36 inst/cycle | 2.01 inst/cycle | -0.35 | 0.03 inst/cycle | -2.33 | 1.50 inst/cycle | -0.86 | 1.37 inst/cycle | -0.99 |


### Key Learnings

#### ❌ Plan B: `restructure_to_2d_grid_to_eliminate_modulo_and_improve_coalescing` (0.028×)

Restructuring a 1D pointwise kernel into a 2D grid (channel × spatial) to eliminate a non-power-of-2 modulo operation caused a 36x latency regression (16.86 µs → 617.38 µs). The new index pattern `flat_idx = c0 + sindex * 24` (channel-last layout with stride-24 access) introduced severe uncoalesced global memory accesses: 82% of all sectors were excessive (5.27M of 6.40M total), and memory throughput collapsed from 714 GB/s to 352 MB/s. Warp cycles per issued instruction exploded from 15.65 to 1949.12, and eligible warps per scheduler dropped from 0.76 to 0.01, indicating the GPU was almost entirely stalled on memory. The original 1D flat index with modulo, despite the modulo cost, produced coalesced access patterns that were far superior to the strided channel-last access introduced by the 2D restructure.

> **Applicable when**: Never apply this pattern. This is a negative result. Do not restructure a 1D flat-index kernel to a 2D channel×spatial grid when the tensor layout is channel-last (NHWC), as threads within a warp will access memory with stride equal to the channel count, causing severe uncoalescing.

> **Avoid when**: Do not use a 2D grid with channel as program_id(0) and spatial as program_id(1) when the tensor is stored in channel-last (NHWC/interleaved) layout. The resulting access pattern `flat_idx = channel + spatial * C` gives each thread in a warp a different spatial index but the same channel, producing stride-C accesses that are maximally uncoalesced. This anti-pattern applies regardless of whether the goal is to eliminate a modulo operation — the coalescing cost dominates any modulo savings by orders of magnitude.

---

## Iteration 5

**Best so far**: 1.019× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_channel_scalars_to_shared_memory_to_eliminate_l1tex_stalls | 0.991× | 1.004× | ✓ |  |
| B | increase_num_warps_per_block_to_hide_l1tex_latency_without_register_pressure | 0.995× | 1.007× | ✓ |  |
| C | replace_tl_gather_with_direct_vector_indexing_to_reduce_l1tex_stalls | 1.002× | 1.017× | ✓ |  |
| D | revert_2d_grid_to_coalesced_1d_flat_index_with_modulo_channel_lookup | 0.989× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_channel_sca... | Δ | increase_num_warp... | Δ | replace_tl_gather... | Δ | revert_2d_grid_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17180 ns | 17120 ns | -60 ns (-0.3%) | 17060 ns | -120 ns (-0.7%) | 16900 ns | -280 ns (-1.6%) | 16930 ns | -250 ns (-1.5%) |
| SM Throughput (%) | 50.98 % | 79.44 % | +28.46 pp | 34.25 % | -16.73 pp | 36.23 % | -14.75 pp | 36.90 % | -14.08 pp |
| Memory Throughput (%) | 22.65 % | 54.23 % | +31.58 pp | 22.41 % | -0.24 pp | 22.60 % | -0.05 pp | 22.74 % | +0.09 pp |
| DRAM Throughput (%) | 14.32 % | 14.35 % | +0.03 pp | 14.39 % | +0.07 pp | 14.53 % | +0.21 pp | 14.51 % | +0.19 pp |
| Achieved Occupancy (%) | 56.08 % | 48.15 % | -7.93 pp | 42.44 % | -13.64 pp | 42.20 % | -13.88 pp | 44.54 % | -11.54 pp |
| Registers/Thread | 17.00 register/thread | 18.00 register/thread | +1 | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 18.00 register/thread | +1 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 40.67 % | 50.16 % | +9.49 pp | 62.27 % | +21.60 pp | 57.73 % | +17.06 pp | 56.45 % | +15.78 pp |
| IPC Active | 2.36 inst/cycle | 1.99 inst/cycle | -0.37 | 1.50 inst/cycle | -0.86 | 1.70 inst/cycle | -0.66 | 1.70 inst/cycle | -0.66 |

---

## Cumulative Progress

```
Baseline:  17180 ns  (1.000×)
Iter 2     16860 ns  (1.019×)  — replace_integer_reciprocal_with_fp32_rsqrt_to_enable_fma_fusion

Final best: 16960 ns  (1.0130×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,13 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = tl.full([XBLOCK], True, tl.int1)
     x2 = xindex
     x0 = (xindex % 24)
     tmp0 = tl.load(in_out_ptr0 + (x2), None)
@@ -18,22 +19,13 @@

     tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
+    tmp_eps = tl.full([1], 1e-05, tl.float32)
+    tmp_var_e = tmp3 + tmp_eps
+    tmp_scale = tmp12 * libdevice.rsqrt(tmp_var_e)
+    tmp_sb = tl.math.fma(-tmp1, tmp_scale, tmp14)
+    tmp_out = tl.math.fma(tmp0, tmp_scale, tmp_sb)
+    tmp17 = tl.maximum(tl.full([1], 0.0, tl.float32), tmp_out)
     tl.store(in_out_ptr0 + (x2), tmp17, None)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
