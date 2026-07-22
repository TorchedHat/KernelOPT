# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_47` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8290 ns — **Best latency**: 6210 ns (**1.3349×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fuse_batchnorm_rsqrt_chain_to_mufu_rsq_instruction`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_47`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_47_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (5390,1,1), Waves/SM: 2.55
- **Registers/thread**: 32
- **Duration**: 8290 ns
- **SM Throughput**: 52.25 % | **Memory**: 21.67 % | **DRAM**: 21.67 %
- **Achieved Occupancy**: 75.13 %
- **IPC Active**: 2.84 inst/cycle | **No Eligible**: 27.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_to_eliminate_partial_wave_and_improve_cache_reuse | 0.991× | 1.020× | ✓ |  |
| B | fix_launch_configuration_eliminate_partial_wave | 1.028× | 0.429× | ✓ |  |
| C | increase_xblock_to_eliminate_partial_wave_and_enable_vectorized_loads | 1.030× | 0.963× | ✓ |  |
| D | autotune_xblock_to_eliminate_partial_wave | 1.123× | 1.301× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | fix_launch_config... | Δ | increase_xblock_t... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 8130 ns | -160 ns (-1.9%) | 19300 ns | +11010 ns (+132.8%) | 8610 ns | +320 ns (+3.9%) | 6370 ns | -1920 ns (-23.2%) |
| SM Throughput (%) | 52.25 % | 47.49 % | -4.76 pp | 71.85 % | +19.60 pp | 62.61 % | +10.36 pp | 33.53 % | -18.72 pp |
| Memory Throughput (%) | 21.67 % | 22.12 % | +0.45 pp | 17.23 % | -4.44 pp | 22.68 % | +1.01 pp | 14.17 % | -7.50 pp |
| DRAM Throughput (%) | 21.67 % | 22.12 % | +0.45 pp | 9.30 % | -12.37 pp | 20.87 % | -0.80 pp | 14.17 % | -7.50 pp |
| Achieved Occupancy (%) | 75.13 % | 56.94 % | -18.19 pp | 65.60 % | -9.53 pp | 79.14 % | +4.01 pp | 60.63 % | -14.50 pp |
| Registers/Thread | 32.00 register/thread | 34.00 register/thread | +2 | 34.00 register/thread | +2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 2.55 | 1.70 | -0.85 | 6.81 | +4.26 | 2.55 | +0.00 | 1.28 | -1.27 |
| No Eligible (%) | 27.22 % | 30.25 % | +3.03 pp | 19.56 % | -7.66 pp | 17.69 % | -9.53 pp | 35.67 % | +8.45 pp |
| IPC Active | 2.84 inst/cycle | 2.75 inst/cycle | -0.09 | 3.18 inst/cycle | +0.34 | 3.27 inst/cycle | +0.43 | 2.44 inst/cycle | -0.40 |


### Key Learnings

#### ✅ Plan D: `autotune_xblock_to_eliminate_partial_wave` (1.386×)

The plan targeted fusing the BN epilogue scale+bias into tl.math.fma and switching streaming loads from evict_first back to evict_last, but the NCU profiling metrics before and after are nearly identical: duration moved from 5.95 µs to 5.98 µs (essentially flat), executed instructions unchanged at 1,315,160, warp cycles per instruction 15.2→14.96, and all cache hit rates flat. The 1.39x end-to-end speedup reported is inconsistent with the per-kernel NCU data and is likely attributable to measurement noise, cross-kernel effects, or the autotuner selecting a marginally better config (XBLOCK=128/num_warps=2 replacing XBLOCK=128/num_warps=8 in the config list). The actual code diff shows the dominant changes were: (1) swapping evict_first back to evict_last on streaming loads, (2) replacing tl.math.rsqrt with libdevice.rsqrt, and (3) a minor autotune config swap — none of which moved NCU metrics measurably at this kernel's operating point.

> **Applicable when**: Kernel is already near its NCU-measurable performance floor (all major bottlenecks addressed); end-to-end speedup is reported but per-kernel NCU metrics show no change; autotune config set includes redundant or suboptimal num_warps values.

> **Avoid when**: Per-kernel NCU metrics show clear improvement in instruction count, cache hit rates, or stall cycles — in that case the speedup is genuine and attributable to the code change. Do not use end-to-end speedup alone to validate micro-optimizations on short kernels (<10 µs) where measurement noise dominates.

---

## Iteration 2

**Best so far**: 1.3014× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_eviction_policy_streaming_loads_to_evict_first | 1.088× | 1.245× | ✓ |  |
| B | fix_broken_autotune_grid_to_enable_wave_alignment | 1.122× | 1.301× | ✓ |  |
| C | replace_sqrt_reciprocal_chain_with_rsqrt_to_fuse_fp32_instructions | 1.070× | 1.168× | ✓ |  |
| D | reduce_register_pressure_via_smaller_num_warps_and_xblock_configs | 1.148× | 1.245× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | fix_broken_autotu... | Δ | replace_sqrt_reci... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 6660 ns | -1630 ns (-19.7%) | 6370 ns | -1920 ns (-23.2%) | 7100 ns | -1190 ns (-14.4%) | 6660 ns | -1630 ns (-19.7%) |
| SM Throughput (%) | 52.25 % | 33.75 % | -18.50 pp | 33.61 % | -18.64 pp | 52.14 % | -0.11 pp | 33.60 % | -18.65 pp |
| Memory Throughput (%) | 21.67 % | 13.53 % | -8.14 pp | 14.13 % | -7.54 pp | 27.26 % | +5.59 pp | 13.55 % | -8.12 pp |
| DRAM Throughput (%) | 21.67 % | 13.53 % | -8.14 pp | 14.13 % | -7.54 pp | 25.26 % | +3.59 pp | 13.55 % | -8.12 pp |
| Achieved Occupancy (%) | 75.13 % | 61.98 % | -13.15 pp | 61.02 % | -14.11 pp | 72.16 % | -2.97 pp | 60.91 % | -14.22 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 2.55 | 1.28 | -1.27 | 1.28 | -1.27 | 2.55 | +0.00 | 1.28 | -1.27 |
| No Eligible (%) | 27.22 % | 38.28 % | +11.06 pp | 38.77 % | +11.55 pp | 22.15 % | -5.07 pp | 36.79 % | +9.57 pp |
| IPC Active | 2.84 inst/cycle | 2.29 inst/cycle | -0.55 | 2.44 inst/cycle | -0.40 | 3.08 inst/cycle | +0.24 | 2.44 inst/cycle | -0.40 |

---

## Iteration 3

**Best so far**: 1.3014× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_batchnorm_rsqrt_chain_to_mufu_rsq_instruction | 1.180× | 1.335× | ✓ | **✓ best** |
| B | fuse_batchnorm_arithmetic_chain_into_fma_via_tl_math_rsqrt | 1.167× | 1.393× | ✓ |  |
| C | fuse_batchnorm_sqrt_reciprocal_chain_to_rsqrt_mufu | 1.151× | 1.329× | ✓ |  |
| D | fix_broken_autotune_grid_and_add_small_xblock_configs_with_low_num_warps | 1.144× | 1.357× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_batchnorm_rs... | Δ | fuse_batchnorm_ar... | Δ | fuse_batchnorm_sq... | Δ | fix_broken_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 6210 ns | -2080 ns (-25.1%) | 5950 ns | -2340 ns (-28.2%) | 6240 ns | -2050 ns (-24.7%) | 6110 ns | -2180 ns (-26.3%) |
| SM Throughput (%) | 52.25 % | 28.02 % | -24.23 pp | 27.46 % | -24.79 pp | 27.14 % | -25.11 pp | 28.14 % | -24.11 pp |
| Memory Throughput (%) | 21.67 % | 14.53 % | -7.14 pp | 15.11 % | -6.56 pp | 14.47 % | -7.20 pp | 14.77 % | -6.90 pp |
| DRAM Throughput (%) | 21.67 % | 14.53 % | -7.14 pp | 15.11 % | -6.56 pp | 14.47 % | -7.20 pp | 14.77 % | -6.90 pp |
| Achieved Occupancy (%) | 75.13 % | 58.41 % | -16.72 pp | 58.60 % | -16.53 pp | 58.26 % | -16.87 pp | 57.41 % | -17.72 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 2.55 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 |
| No Eligible (%) | 27.22 % | 37.88 % | +10.66 pp | 39.19 % | +11.97 pp | 40.11 % | +12.89 pp | 36.88 % | +9.66 pp |
| IPC Active | 2.84 inst/cycle | 2.45 inst/cycle | -0.39 | 2.43 inst/cycle | -0.41 | 2.40 inst/cycle | -0.44 | 2.34 inst/cycle | -0.50 |


### Key Learnings

#### ✅ Plan C: `fuse_batchnorm_sqrt_reciprocal_chain_to_rsqrt_mufu` (1.329×)

Torch Inductor codegen emits a 4-instruction batch-norm normalization chain (tl.sqrt_rn → int32 literal 1 → integer/float division → float multiply) where a single MUFU.RSQ instruction suffices. This int32/float32 type mixing prevents FMA fusion and introduces warp divergence (Avg. Divergent Branches: 20.42 → 0.0, Avg. Active Threads Per Warp: 29.97 → 32.0). Replacing the chain with libdevice.rsqrt() reduces executed instructions by ~22% (1,681,680 → 1,315,160), eliminates divergent branches entirely, frees 2 registers per thread (32 → 30), and cuts kernel duration from 6.37 µs to 6.24 µs (1.33x speedup). The NCU FPInstructions rule firing with a high non-fused to fused ratio (>4:1) is the key signal for this pattern.

> **Applicable when**: NCU FPInstructions rule fires with non-fused to fused FP32 instruction ratio >3:1; kernel contains a batch-norm or layer-norm normalization path; Avg. Active Threads Per Warp is below 31 (indicating warp divergence from type-mixing); Avg. Divergent Branches > 0; libdevice is already imported or can be imported.

> **Avoid when**: The variance input to sqrt may be zero or negative (rsqrt is undefined for non-positive inputs); the kernel does not contain a sqrt+reciprocal chain (the FPInstructions rule fires for other reasons such as many exp/log ops); libdevice is unavailable in the target Triton version.

---

## Iteration 4

**Best so far**: 1.3349× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_to_autotune_configs_to_hide_l1tex_latency | 1.059× | 1.363× | ✓ |  |
| B | fuse_batchnorm_scale_bias_epilogue_into_fma_instruction | 1.150× | 1.393× | ✓ |  |
| C | fuse_batchnorm_epilogue_scale_bias_into_fma | 1.152× | 1.386× | ✓ |  |
| D | fuse_batchnorm_scale_bias_epilogue_into_fma | 1.169× | 1.357× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_to... | Δ | fuse_batchnorm_sc... | Δ | fuse_batchnorm_ep... | Δ | fuse_batchnorm_sc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 6080 ns | -2210 ns (-26.7%) | 5950 ns | -2340 ns (-28.2%) | 5980 ns | -2310 ns (-27.9%) | 6110 ns | -2180 ns (-26.3%) |
| SM Throughput (%) | 52.25 % | 26.70 % | -25.55 pp | 28.14 % | -24.11 pp | 27.60 % | -24.65 pp | 27.54 % | -24.71 pp |
| Memory Throughput (%) | 21.67 % | 14.81 % | -6.86 pp | 15.16 % | -6.51 pp | 15.07 % | -6.60 pp | 14.75 % | -6.92 pp |
| DRAM Throughput (%) | 21.67 % | 14.81 % | -6.86 pp | 15.16 % | -6.51 pp | 15.07 % | -6.60 pp | 14.75 % | -6.92 pp |
| Achieved Occupancy (%) | 75.13 % | 58.50 % | -16.63 pp | 56.60 % | -18.53 pp | 58.41 % | -16.72 pp | 57.92 % | -17.21 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 | 30.00 register/thread | -2 |
| Waves/SM | 2.55 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 |
| No Eligible (%) | 27.22 % | 36.07 % | +8.85 pp | 36.32 % | +9.10 pp | 40.02 % | +12.80 pp | 37.08 % | +9.86 pp |
| IPC Active | 2.84 inst/cycle | 2.41 inst/cycle | -0.43 | 2.38 inst/cycle | -0.46 | 2.42 inst/cycle | -0.42 | 2.40 inst/cycle | -0.44 |


### Key Learnings

#### ✅ Plan A: `add_num_stages_to_autotune_configs_to_hide_l1tex_latency` (1.363×)

Adding num_stages=2 variants to Triton autotune configs was intended to enable async double-buffering (cp.async) to hide L1TEX scoreboard stalls (37.8% of CPI stall budget). However, the per-kernel NCU metrics after the change show no meaningful improvement: warp cycles per issued instruction actually increased slightly (15.2 → 15.92), L1/TEX hit rate was flat (42.31% → 42.32%), L2 hit rate barely moved (33.95% → 34.35%), and kernel duration increased from 5.95 µs to 6.08 µs. The 1.36x reported speedup is an end-to-end measurement that does not align with the per-kernel NCU data, suggesting the speedup originates from a different source (likely the grid lambda fix enabling the autotuner to select a better XBLOCK) rather than the num_stages async prefetch mechanism. For short-duration pointwise kernels (~6 µs, ~9000 elapsed cycles) with few load iterations per block, num_stages>1 provides negligible benefit because there are insufficient loop iterations to amortize the double-buffering overhead.

> **Applicable when**: Triton kernel has a main loop with many iterations per block (e.g., reduction kernels or kernels with large XBLOCK relative to problem size); CPIStall L1TEX stalls are >30% of CPI; kernel duration is long enough (>50 µs) that double-buffering amortization overhead is negligible; the grid was previously hardcoded (not a lambda), preventing the autotuner from selecting different XBLOCK values.

> **Avoid when**: Kernel is a short-duration pointwise kernel (<10 µs, <10K elapsed cycles) with few load iterations per block — num_stages>1 provides no benefit and may slightly increase warp cycles per instruction; L1/TEX hit rate and warp stall cycles do not improve after adding num_stages (as seen here); the speedup is attributed to num_stages when the actual source is a concurrent grid lambda fix (confounded measurement).

---

## Iteration 5

**Best so far**: 1.3349× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_and_align_xblock_to_channel_boundary_to_eliminate_branch_divergence | 1.169× | 1.363× | ✓ |  |
| B | fix_broken_grid_lambda_and_add_low_warp_configs_to_reduce_not_selected_stall | 1.162× | 1.370× | ✓ |  |
| C | eliminate_predicated_fp32_zero_multiplies_from_tl_where_cat_chains | 1.167× | 1.341× | ✓ |  |
| D | replace_avgpool_division_with_multiply_to_enable_fma_fusion | 1.149× | 1.386× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_a... | Δ | fix_broken_grid_l... | Δ | eliminate_predica... | Δ | replace_avgpool_d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8290 ns | 6080 ns | -2210 ns (-26.7%) | 6050 ns | -2240 ns (-27.0%) | 6180 ns | -2110 ns (-25.5%) | 5980 ns | -2310 ns (-27.9%) |
| SM Throughput (%) | 52.25 % | 27.06 % | -25.19 pp | 14.02 % | -38.23 pp | 27.02 % | -25.23 pp | 28.08 % | -24.17 pp |
| Memory Throughput (%) | 21.67 % | 14.88 % | -6.79 pp | 7.44 % | -14.23 pp | 14.64 % | -7.03 pp | 15.08 % | -6.59 pp |
| DRAM Throughput (%) | 21.67 % | 14.88 % | -6.79 pp | 7.44 % | -14.23 pp | 14.64 % | -7.03 pp | 15.08 % | -6.59 pp |
| Achieved Occupancy (%) | 75.13 % | 57.60 % | -17.53 pp | 19.72 % | -55.41 pp | 58.99 % | -16.14 pp | 59.49 % | -15.64 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 30.00 register/thread | -2 |
| Waves/SM | 2.55 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 | 1.28 | -1.27 |
| No Eligible (%) | 27.22 % | 37.21 % | +9.99 pp | 65.66 % | +38.44 pp | 45.31 % | +18.09 pp | 40.12 % | +12.90 pp |
| IPC Active | 2.84 inst/cycle | 2.42 inst/cycle | -0.42 | 1.12 inst/cycle | -1.72 | 2.35 inst/cycle | -0.49 | 2.44 inst/cycle | -0.40 |


### Key Learnings

#### ✅ Plan D: `replace_avgpool_division_with_multiply_to_enable_fma_fusion` (1.386×)

Replacing a float division by a compile-time constant (e.g., `x / 4.0` via `tl.full([1], 4.0, tl.float32)`) with a reciprocal multiply (`x * 0.25`) was intended to enable FFMA fusion with adjacent operations in the normalization chain. However, the per-kernel NCU metrics show no meaningful improvement: executed instructions are identical (1,315,160), warp cycles per issued instruction actually increased (15.28 → 17.2), and kernel duration was flat (5.95 → 5.98 µs). The 1.39x reported end-to-end speedup matches the prior iteration's speedup exactly, strongly indicating the result is measurement noise or cross-kernel scheduling effects rather than a genuine improvement from this change. At this kernel's operating point (post-rsqrt fusion), the FPInstructions rule's remaining non-fused instructions are not in the avg_pool division path, or the LLVM backend already optimizes the division to a multiply internally, leaving no room for further FMA fusion from this change.

> **Applicable when**: FPInstructions NCU rule fires with high non-fused FP32 ratio; kernel contains explicit float division by a compile-time constant via tl.full(); the division is in a hot path adjacent to multiply-add chains; prior instruction-count reductions (e.g., rsqrt fusion) have NOT already been applied.

> **Avoid when**: The kernel has already undergone major instruction-count reductions (e.g., rsqrt chain fusion) that brought executed instructions to a floor; per-kernel NCU metrics show no change in executed instructions or warp stall cycles after the change; the LLVM backend already optimizes float division by power-of-2 constants to reciprocal multiply internally; kernel duration is <10 µs where measurement noise dominates end-to-end speedup reports.

---

## Cumulative Progress

```
Baseline:  8290 ns  (1.000×)
Iter 1     6370 ns  (1.301×)  — autotune_xblock_to_eliminate_partial_wave
Iter 3     6210 ns  (1.335×)  — fuse_batchnorm_rsqrt_chain_to_mufu_rsq_instruction

Final best: 6210 ns  (1.3349×)
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
+        triton.Config({'XBLOCK': 64},   num_warps=2),
+        triton.Config({'XBLOCK': 64},   num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=2),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_47(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
     xnumel = 689920
@@ -16,21 +32,21 @@

     x2 = xindex // 4928
     x3 = xindex // 352
     x4 = xindex
-    tmp33 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
-    tmp35 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
-    tmp44 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
-    tmp46 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
+    tmp33 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp35 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp44 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
+    tmp46 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
     tmp0 = x0
     tmp1 = tl.full([1], 0, tl.int64)
     tmp2 = tmp0 >= tmp1
     tmp3 = tl.full([1], 256, tl.int64)
     tmp4 = tmp0 < tmp3
-    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
-    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp7 = tmp5 + tmp6
-    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp9 = tmp7 + tmp8
-    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
     tmp11 = tmp9 + tmp10
     tmp12 = tl.full([1], 4.0, tl.float32)
     tmp13 = (tmp11 / tmp12)
@@ -40,35 +56,28 @@

     tmp17 = tl.full([1], 288, tl.int64)
     tmp18 = tmp0 < tmp17
     tmp19 = tmp16 & tmp18
-    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
     tmp21 = tmp0 >= tmp17
     tmp22 = tl.full([1], 320, tl.int64)
     tmp23 = tmp0 < tmp22
     tmp24 = tmp21 & tmp23
-    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
     tmp26 = tmp0 >= tmp22
     tmp27 = tl.full([1], 352, tl.int64)
     tmp28 = tmp0 < tmp27
-    tmp29 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
+    tmp29 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp26 & xmask, eviction_policy='evict_last', other=0.0)
     tmp30 = tl.where(tmp24, tmp25, tmp29)
     tmp31 = tl.where(tmp19, tmp20, tmp30)
     tmp32 = tl.where(tmp4, tmp15, tmp31)
     tmp34 = tmp32 - tmp33
     tmp36 = tl.full([1], 1e-05, tl.float32)
     tmp37 = tmp35 + tmp36
-    tmp38 = tl.sqrt_rn(tmp37)
-    tmp39 = tl.full([1], 1, tl.int32)
-    tmp40 = (tmp39 / tmp38)
-    tmp41 = tl.full([1], 1.0, tl.float32)
-    tmp42 = tmp40 * tmp41
+    tmp42 = libdevice.rsqrt(tmp37)
     tmp43 = tmp34 * tmp42
-    tmp45 = tmp43 * tmp44
-    tmp47 = tmp45 + tmp46
+    tmp47 = tl.math.fma(tmp43, tmp44, tmp46)
     tmp48 = tl.full([1], 0, tl.int32)
     tmp49 = tl.maximum(tmp48, tmp47)
-    tl.store(in_out_ptr0 + (x4), tmp49, xmask)
-
-
+    tl.store(in_out_ptr0 + (x4), tmp49, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -82,7 +91,7 @@

     arg_8 = inputs[8]
     xnumel = 689920
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_47[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 689920, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_47[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 689920)
     return arg_8
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
