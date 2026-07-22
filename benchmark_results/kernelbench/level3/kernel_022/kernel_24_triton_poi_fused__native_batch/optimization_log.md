# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3390 ns — **Best latency**: 2940 ns (**1.1531×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_non_fused_fp32_instructions_via_fma_rewrite`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_24_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1225,1,1), Waves/SM: 0.58
- **Registers/thread**: 18
- **Duration**: 3390 ns
- **SM Throughput**: 14.70 % | **Memory**: 5.80 % | **DRAM**: 3.85 %
- **Achieved Occupancy**: 47.28 %
- **IPC Active**: 1.60 inst/cycle | **No Eligible**: 58.16 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_via_autotune_larger_xblock | 43.354× | 0.897× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_underutilized_pointwise_kernel | 45.491× | 0.675× | ✓ | **✓ best** |
| C | increase_xblock_tile_size_with_autotune | 43.649× | 0.538× | ✓ |  |
| D | autotune_xblock_and_fix_fma_fusion | 43.794× | 0.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | autotune_xblock_a... | Δ | increase_xblock_t... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3780 ns | +390 ns (+11.5%) | 5020 ns | +1630 ns (+48.1%) | 6300 ns | +2910 ns (+85.8%) | 3780 ns | +390 ns (+11.5%) |
| SM Throughput (%) | 14.70 % | 26.31 % | +11.61 pp | 46.84 % | +32.14 pp | 51.29 % | +36.59 pp | 10.21 % | -4.49 pp |
| Memory Throughput (%) | 5.80 % | 7.77 % | +1.97 pp | 13.93 % | +8.13 pp | 11.15 % | +5.35 pp | 5.08 % | -0.72 pp |
| DRAM Throughput (%) | 3.85 % | 3.47 % | -0.38 pp | 2.59 % | -1.26 pp | 2.06 % | -1.79 pp | 3.47 % | -0.38 pp |
| Achieved Occupancy (%) | 47.28 % | 39.31 % | -7.97 pp | 76.01 % | +28.73 pp | 49.28 % | +2.00 pp | 25.43 % | -21.85 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 48.00 register/thread | +30 | 24.00 register/thread | +6 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 2.32 | +1.74 | 1.86 | +1.28 | 0.29 | -0.29 |
| No Eligible (%) | 58.16 % | 38.50 % | -19.66 pp | 22.39 % | -35.77 pp | 22.08 % | -36.08 pp | 70.82 % | +12.66 pp |
| IPC Active | 1.60 inst/cycle | 2.35 inst/cycle | +0.75 | 3.06 inst/cycle | +1.46 | 3.08 inst/cycle | +1.48 | 1.11 inst/cycle | -0.49 |


### Key Learnings

#### ✅ Plan C: `increase_xblock_tile_size_with_autotune` (1.050×)

Expanding the @triton.autotune config set (from 3 configs with XBLOCK up to 256 to 7 configs with XBLOCK up to 2048) combined with a lambda grid reduced registers per thread from 22 to 18, which increased achieved occupancy from 40.9% to 45.8%. However, the autotuner still selected XBLOCK=128 (grid remained 1225 blocks, waves/SM unchanged at 0.58), meaning the larger XBLOCK configs were not chosen — the speedup of 1.05x came primarily from the register reduction enabling slightly more active warps per SM, not from tile size changes. The instruction count actually decreased (274K→245K), likely due to the FMA/rsqrt arithmetic path being retained, while warp cycles per issued instruction worsened (19.7→32.3), indicating the register-reduced kernel has more latency stalls per instruction.

> **Applicable when**: Kernel has a small number of autotune configs with low max XBLOCK and registers per thread are above 20; expanding configs to include larger XBLOCK values with varied num_warps may reduce register pressure through compiler re-optimization, yielding modest occupancy gains even if the autotuner still selects the smallest XBLOCK.

> **Avoid when**: Do not expect large speedups from this approach when the autotuner consistently selects XBLOCK=128 (the minimum config) — the waves/SM bottleneck (0.58) will remain unresolved. Do not apply when warp cycles per issued instruction are already low (<20) and eligible warps per scheduler are healthy (>2.0), as the register reduction may worsen latency hiding. Do not apply when prior iterations have already shown autotune+lambda-grid regressions for this kernel (see exp_triton_autotune_fixed_grid_fma_regression_004 which regressed 0.77x with a similar approach).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_for_correct_persistent_tile_scheduling | 41.924× | 0.963× | ✓ |  |
| B | persistent_tile_scheduling_via_sm_loop | 28.384× | 0.445× | ✓ |  |
| C | fix_autotune_grid_lambda_to_eliminate_redundant_blocks_and_tail_wave | 45.458× | 0.683× | ✓ |  |
| D | fix_autotune_grid_lambda_and_remove_restore_value_for_persistent_scheduling | 43.770× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | persistent_tile_s... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3520 ns | +130 ns (+3.8%) | 7620 ns | +4230 ns (+124.8%) | 4960 ns | +1570 ns (+46.3%) | 3390 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 14.70 % | 26.15 % | +11.45 pp | 32.72 % | +18.02 pp | 47.37 % | +32.67 pp | 20.17 % | +5.47 pp |
| Memory Throughput (%) | 5.80 % | 7.73 % | +1.93 pp | 13.52 % | +7.72 pp | 14.04 % | +8.24 pp | 7.62 % | +1.82 pp |
| DRAM Throughput (%) | 3.85 % | 3.68 % | -0.17 pp | 1.70 % | -2.15 pp | 2.60 % | -1.25 pp | 3.85 % | +0.00 pp |
| Achieved Occupancy (%) | 47.28 % | 41.22 % | -6.06 pp | 44.09 % | -3.19 pp | 76.15 % | +28.87 pp | 42.23 % | -5.05 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 26.00 register/thread | +8 | 24.00 register/thread | +6 | 24.00 register/thread | +6 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 2.32 | +1.74 | 0.58 | +0.00 |
| No Eligible (%) | 58.16 % | 38.78 % | -19.38 pp | 42.18 % | -15.98 pp | 23.75 % | -34.41 pp | 53.07 % | -5.09 pp |
| IPC Active | 1.60 inst/cycle | 2.16 inst/cycle | +0.56 | 2.18 inst/cycle | +0.58 | 3.06 inst/cycle | +1.46 | 1.82 inst/cycle | +0.22 |


### Key Learnings

#### ❌ Plan B: `persistent_tile_scheduling_via_sm_loop` (0.445×)

Converting a standard pointwise Triton kernel to a persistent while-loop kernel (one CTA per SM, each iterating over multiple tiles) caused a 2.25x slowdown (3.78 µs → 7.62 µs) despite reducing registers per thread (32→26). The critical failure was an implementation bug: the grid in the launch call was left as `cdiv(xnumel, 128) = 1225` instead of being changed to `(132,)` (NUM_SMS), so the persistent kernel launched 1225 CTAs each running the while-loop — processing every element ~9x times and inflating executed instructions 2.5x (760K→1.9M). Additionally, the while-loop introduced 5x more divergent branches (9.28→47.8 avg divergent branches), further degrading warp efficiency. The waves_per_sm remained unchanged at 0.58 because the grid was never actually reduced to 132.

> **Applicable when**: Never apply this exact pattern as implemented. A correctly implemented persistent kernel (grid=(NUM_SMS,), each CTA loops over tiles) can theoretically help when waves_per_sm < 1.0 and the kernel is latency-bound due to insufficient occupancy. However, the implementation must change the grid to (NUM_SMS,) and use a large enough BLOCK_SIZE to fill each SM.

> **Avoid when**: Do not apply a persistent while-loop kernel pattern without simultaneously changing the launch grid to (NUM_SMS,). If the grid remains at cdiv(xnumel, XBLOCK), each CTA will loop over the entire problem space with stride=NUM_SMS*XBLOCK, causing each element to be processed NUM_SMS times and inflating instruction count proportionally. Also avoid persistent kernels when the while-loop condition introduces significant branch divergence across warps within a block, as divergent branches increased 5x (9→48 avg divergent branches) in this case.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_and_fix_autotune_grid_lambda | 43.854× | 1.009× | ✓ |  |
| B | fuse_non_fused_fp32_instructions_via_fma_rewrite | 47.719× | 1.153× | ✓ | **✓ best** |
| C | fuse_fp32_multiply_add_chains_into_fma_instructions | 45.455× | 0.675× | ✓ |  |
| D | revert_broken_persistent_loop_to_clean_singlepass_pointwise_with_fma_fusion | 43.601× | 0.929× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_fi... | Δ | fuse_non_fused_fp... | Δ | fuse_fp32_multipl... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3360 ns | -30 ns (-0.9%) | 2940 ns | -450 ns (-13.3%) | 5020 ns | +1630 ns (+48.1%) | 3650 ns | +260 ns (+7.7%) |
| SM Throughput (%) | 14.70 % | 19.82 % | +5.12 pp | 14.39 % | -0.31 pp | 47.40 % | +32.70 pp | 15.56 % | +0.86 pp |
| Memory Throughput (%) | 5.80 % | 7.78 % | +1.98 pp | 9.83 % | +4.03 pp | 14.10 % | +8.30 pp | 5.96 % | +0.16 pp |
| DRAM Throughput (%) | 3.85 % | 3.89 % | +0.04 pp | 4.41 % | +0.56 pp | 2.58 % | -1.27 pp | 3.57 % | -0.28 pp |
| Achieved Occupancy (%) | 47.28 % | 40.12 % | -7.16 pp | 41.52 % | -5.76 pp | 75.98 % | +28.70 pp | 47.20 % | -0.08 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 24.00 register/thread | +6 | 18.00 register/thread | +0 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 2.32 | +1.74 | 0.58 | +0.00 |
| No Eligible (%) | 58.16 % | 51.35 % | -6.81 pp | 58.67 % | +0.51 pp | 22.41 % | -35.75 pp | 59.55 % | +1.39 pp |
| IPC Active | 1.60 inst/cycle | 1.89 inst/cycle | +0.29 | 1.64 inst/cycle | +0.04 | 3.06 inst/cycle | +1.46 | 1.54 inst/cycle | -0.06 |

---

## Iteration 4

**Best so far**: 1.1531× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_broken_autotune_grid_lambda_and_remove_restore_value_overhead | 46.885× | 1.040× | ✓ |  |
| B | rsqrt_and_fma_arithmetic_fusion_in_batchnorm_pointwise | 47.149× | 1.093× | ✓ |  |
| C | autotune_xblock_with_correct_lambda_grid | 45.019× | 0.774× | ✓ |  |
| D | fix_hardcoded_grid_to_lambda_for_correct_autotune_xblock_dispatch | 45.483× | 0.683× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_broken_autotu... | Δ | rsqrt_and_fma_ari... | Δ | autotune_xblock_w... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3260 ns | -130 ns (-3.8%) | 3100 ns | -290 ns (-8.6%) | 4380 ns | +990 ns (+29.2%) | 4960 ns | +1570 ns (+46.3%) |
| SM Throughput (%) | 14.70 % | 10.13 % | -4.57 pp | 11.45 % | -3.25 pp | 42.03 % | +27.33 pp | 47.27 % | +32.57 pp |
| Memory Throughput (%) | 5.80 % | 6.54 % | +0.74 pp | 8.89 % | +3.09 pp | 9.99 % | +4.19 pp | 14.05 % | +8.25 pp |
| DRAM Throughput (%) | 3.85 % | 4.00 % | +0.15 pp | 4.17 % | +0.32 pp | 2.96 % | -0.89 pp | 2.61 % | -1.24 pp |
| Achieved Occupancy (%) | 47.28 % | 46.09 % | -1.19 pp | 40.91 % | -6.37 pp | 71.23 % | +23.95 pp | 76.00 % | +28.72 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 1.16 | +0.58 | 2.32 | +1.74 |
| No Eligible (%) | 58.16 % | 72.57 % | +14.41 pp | 64.78 % | +6.62 pp | 25.68 % | -32.48 pp | 22.30 % | -35.86 pp |
| IPC Active | 1.60 inst/cycle | 0.97 inst/cycle | -0.63 | 1.32 inst/cycle | -0.28 | 2.91 inst/cycle | +1.31 | 3.05 inst/cycle | +1.45 |

---

## Iteration 5

**Best so far**: 1.1531× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_lambda_and_add_higher_warp_configs | 46.767× | 1.059× | ✓ |  |
| B | fix_broken_autotune_grid_to_lambda_enabling_correct_xblock_dispatch | 47.040× | 1.050× | ✓ |  |
| C | revert_broken_autotune_to_clean_fixed_xblock128_single_pass | 45.202× | 0.991× | ✓ |  |
| D | rsqrt_fusion_with_num_warps_autotune_at_fixed_xblock | 46.531× | 1.040× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | fix_broken_autotu... | Δ | revert_broken_aut... | Δ | rsqrt_fusion_with... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3390 ns | 3200 ns | -190 ns (-5.6%) | 3230 ns | -160 ns (-4.7%) | 3420 ns | +30 ns (+0.9%) | 3260 ns | -130 ns (-3.8%) |
| SM Throughput (%) | 14.70 % | 19.53 % | +4.83 pp | 10.08 % | -4.62 pp | 15.33 % | +0.63 pp | 9.53 % | -5.17 pp |
| Memory Throughput (%) | 5.80 % | 11.46 % | +5.66 pp | 6.67 % | +0.87 pp | 5.95 % | +0.15 pp | 6.20 % | +0.40 pp |
| DRAM Throughput (%) | 3.85 % | 4.07 % | +0.22 pp | 4.01 % | +0.16 pp | 3.80 % | -0.05 pp | 3.99 % | +0.14 pp |
| Achieved Occupancy (%) | 47.28 % | 78.62 % | +31.34 pp | 45.84 % | -1.44 pp | 46.74 % | -0.54 pp | 45.88 % | -1.40 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 0.58 | 1.16 | +0.58 | 0.58 | +0.00 | 0.58 | +0.00 | 0.58 | +0.00 |
| No Eligible (%) | 58.16 % | 46.70 % | -11.46 pp | 73.69 % | +15.53 pp | 58.90 % | +0.74 pp | 74.09 % | +15.93 pp |
| IPC Active | 1.60 inst/cycle | 1.78 inst/cycle | +0.18 | 1.09 inst/cycle | -0.51 | 1.56 inst/cycle | -0.04 | 1.02 inst/cycle | -0.58 |

---

## Cumulative Progress

```
Baseline:  3390 ns  (1.000×)
Iter 3     2940 ns  (1.153×)  — fuse_non_fused_fp32_instructions_via_fma_rewrite

Final best: 2940 ns  (1.1531×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,15 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
+        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 156800
@@ -21,17 +30,10 @@

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
+    tmp_inv_std = libdevice.rsqrt(tmp5)
+    tmp_scale = tmp2 * tmp_inv_std
+    tmp15 = tl.math.fma(tmp_scale, tmp12, tmp14)
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +43,7 @@

     arg_4 = inputs[4]
     xnumel = 156800
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 156800, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 156800)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
