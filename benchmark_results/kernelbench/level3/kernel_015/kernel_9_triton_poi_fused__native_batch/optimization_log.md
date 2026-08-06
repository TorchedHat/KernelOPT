# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 26850 ns — **Best latency**: 26590 ns (**1.0098×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`rewrite_batchnorm_arithmetic_as_explicit_fma_chain_to_reduce_non_fused_fp32_instructions`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_9_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (39200,1,1), Waves/SM: 18.56
- **Registers/thread**: 28
- **Duration**: 26850 ns
- **SM Throughput**: 65.53 % | **Memory**: 32.74 % | **DRAM**: 17.13 %
- **Achieved Occupancy**: 53.46 %
- **IPC Active**: 2.86 inst/cycle | **No Eligible**: 28.21 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse | 0.998× | 0.221× | ✓ |  |
| B | increase_xblock_for_occupancy_and_latency_hiding | 0.999× | 1.005× | ✓ |  |
| C | increase_xblock_for_vectorized_loads_and_better_occupancy | 1.004× | 0.431× | ✓ | **✓ best** |
| D | increase_xblock_with_autotune_for_occupancy_and_l1tex_latency_hiding | 0.905× | 0.832× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_f... | Δ | increase_xblock_f... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 26850 ns | 121600 ns | +94750 ns (+352.9%) | 26720 ns | -130 ns (-0.5%) | 62270 ns | +35420 ns (+131.9%) | 32290 ns | +5440 ns (+20.3%) |
| SM Throughput (%) | 65.53 % | 78.01 % | +12.48 pp | 65.65 % | +0.12 pp | 76.61 % | +11.08 pp | 73.87 % | +8.34 pp |
| Memory Throughput (%) | 32.74 % | 27.06 % | -5.68 pp | 32.78 % | +0.04 pp | 30.01 % | -2.73 pp | 35.58 % | +2.84 pp |
| DRAM Throughput (%) | 17.13 % | 3.76 % | -13.37 pp | 17.30 % | +0.17 pp | 7.35 % | -9.78 pp | 14.11 % | -3.02 pp |
| Achieved Occupancy (%) | 53.46 % | 85.47 % | +32.01 pp | 54.55 % | +1.09 pp | 82.33 % | +28.87 pp | 84.84 % | +31.38 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 18.56 | 74.24 | +55.68 | 18.56 | +0.00 | 37.12 | +18.56 | 18.56 | +0.00 |
| No Eligible (%) | 28.21 % | 20.73 % | -7.48 pp | 27.77 % | -0.44 pp | 20.74 % | -7.47 pp | 21.17 % | -7.04 pp |
| IPC Active | 2.86 inst/cycle | 3.18 inst/cycle | +0.32 | 2.88 inst/cycle | +0.02 | 3.16 inst/cycle | +0.30 | 3.15 inst/cycle | +0.29 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_with_autotune_for_better_occupancy_and_cache_reuse` (0.221×)

Adding @triton.autotune to select a larger XBLOCK (128→512) increased achieved occupancy from 53% to 85% and reduced the AchievedOccupancy gap, but caused a 4.5x explosion in executed instructions (13.6M→74.8M) and a 4.5x increase in kernel duration (26.85→121.6 µs). The autotune selected a larger block size that caused each thread to process more elements redundantly or with worse predication efficiency (Avg. Not Predicated Off Threads dropped from 30.46 to 29.7), and the grid was not updated to use meta['XBLOCK'] — the grid remained fixed at cdiv(xnumel, 128) while the kernel ran with XBLOCK=512, causing each element to be processed ~4x times. When the grid is not updated to match the autotuned XBLOCK, each CTA processes a larger tile but the total number of CTAs stays the same, resulting in massive redundant work. Always update the grid lambda to use meta['XBLOCK'] when applying autotune to XBLOCK.

> **Applicable when**: Never applicable as implemented — this is a negative result. The autotune approach is theoretically valid but the implementation was flawed: the grid was not updated to use meta['XBLOCK'].

> **Avoid when**: Do not add @triton.autotune over XBLOCK without simultaneously changing the grid from a fixed tuple (cdiv(xnumel, 128),) to a lambda (lambda meta: (cdiv(xnumel, meta['XBLOCK']),)). Mismatched grid and XBLOCK causes each element to be processed multiple times, multiplying instruction count and duration by the XBLOCK ratio.

#### ❌ Plan D: `increase_xblock_with_autotune_for_occupancy_and_l1tex_latency_hiding` (0.832×)

Adding triton.autotune with larger XBLOCK configs (512–2048) to a pointwise fused kernel that showed low achieved occupancy (53%) and high L1TEX stall (46%) caused a regression (0.83x). Although occupancy improved from 53% to 85%, the instruction count increased by 38% (13.6M → 18.8M), registers per thread rose from 28 to 32, and Warp Cycles Per Issued Instruction worsened from 11.8 to 17.1 cycles — indicating the autotune selected a config that introduced more register pressure and instruction overhead than the latency hiding benefit could compensate for. The dominant stall shifted from L1TEX dependency (before) to 'not selected' warp scheduling (after), meaning the kernel became compute-bound with excess warps competing for issue slots rather than hiding memory latency.

> **Applicable when**: Never: this specific direction (autotune larger XBLOCK for occupancy/L1TEX stall) caused regression on this kernel class.

> **Avoid when**: Kernel is a fused pointwise op (concat+BN+ReLU) with many conditional loads and BN parameter lookups; low achieved occupancy (53%) combined with high L1TEX stall (46%) does NOT guarantee that larger XBLOCK will help — if the kernel has high instruction count per element and register pressure is near the block limit, larger XBLOCK increases register spill risk and instruction overhead, worsening Warp Cycles Per Issued Instruction. Do not apply when the kernel already has 28+ registers/thread and the block limit on registers is already constraining occupancy.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | change_eviction_policy_streaming_loads_to_evict_first | 1.001× | 1.004× | ✓ |  |
| B | fix_autotune_grid_lambda_to_eliminate_redundant_work_and_enable_smaller_xblock | 1.012× | 0.998× | ✓ | **✓ best** |
| C | fix_autotune_grid_lambda_and_reduce_num_warps_for_scheduler_contention | 1.009× | 0.999× | ✓ |  |
| D | fix_grid_xblock_mismatch_and_align_tile_to_channel_width | 1.006× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | change_eviction_p... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fix_grid_xblock_m... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 26850 ns | 26750 ns | -100 ns (-0.4%) | 26910 ns | +60 ns (+0.2%) | 26880 ns | +30 ns (+0.1%) | 26850 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 65.53 % | 65.59 % | +0.06 pp | 36.75 % | -28.78 pp | 36.69 % | -28.84 pp | 65.56 % | +0.03 pp |
| Memory Throughput (%) | 32.74 % | 32.74 % | +0.00 pp | 24.75 % | -7.99 pp | 24.79 % | -7.95 pp | 32.74 % | +0.00 pp |
| DRAM Throughput (%) | 17.13 % | 16.75 % | -0.38 pp | 17.12 % | -0.01 pp | 17.14 % | +0.01 pp | 17.15 % | +0.02 pp |
| Achieved Occupancy (%) | 53.46 % | 53.22 % | -0.24 pp | 29.66 % | -23.80 pp | 29.68 % | -23.78 pp | 54.25 % | +0.79 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 18.56 | 18.56 | +0.00 | 9.28 | -9.28 | 9.28 | -9.28 | 18.56 | +0.00 |
| No Eligible (%) | 28.21 % | 27.74 % | -0.47 pp | 57.42 % | +29.21 pp | 57.63 % | +29.42 pp | 27.98 % | -0.23 pp |
| IPC Active | 2.86 inst/cycle | 2.85 inst/cycle | -0.01 | 1.69 inst/cycle | -1.17 | 1.68 inst/cycle | -1.18 | 2.87 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_for_software_pipelining_to_hide_l1tex_latency | 1.000× | 1.001× | ✓ |  |
| B | reduce_num_warps_to_increase_blocks_per_sm_and_close_occupancy_gap | 1.005× | 1.001× | ✓ |  |
| C | fix_grid_lambda_mismatch_and_align_xblock_to_channel_width | 1.011× | 1.007× | ✓ |  |
| D | fix_grid_lambda_and_tune_num_warps_for_l1tex_latency_hiding | 1.015× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | reduce_num_warps_... | Δ | fix_grid_lambda_m... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 26850 ns | 26820 ns | -30 ns (-0.1%) | 26820 ns | -30 ns (-0.1%) | 26660 ns | -190 ns (-0.7%) | 26590 ns | -260 ns (-1.0%) |
| SM Throughput (%) | 65.53 % | 65.67 % | +0.14 pp | 65.50 % | -0.03 pp | 36.41 % | -29.12 pp | 32.51 % | -33.02 pp |
| Memory Throughput (%) | 32.74 % | 32.80 % | +0.06 pp | 32.74 % | +0.00 pp | 24.61 % | -8.13 pp | 15.84 % | -16.90 pp |
| DRAM Throughput (%) | 17.13 % | 16.51 % | -0.62 pp | 17.21 % | +0.08 pp | 17.26 % | +0.13 pp | 7.70 % | -9.43 pp |
| Achieved Occupancy (%) | 53.46 % | 54.30 % | +0.84 pp | 54.56 % | +1.10 pp | 29.61 % | -23.85 pp | 27.38 % | -26.08 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 18.56 | 18.56 | +0.00 | 18.56 | +0.00 | 9.28 | -9.28 | 9.28 | -9.28 |
| No Eligible (%) | 28.21 % | 27.78 % | -0.43 pp | 27.86 % | -0.35 pp | 58.19 % | +29.98 pp | 62.12 % | +33.91 pp |
| IPC Active | 2.86 inst/cycle | 2.87 inst/cycle | +0.01 | 2.87 inst/cycle | +0.01 | 1.66 inst/cycle | -1.20 | 1.50 inst/cycle | -1.36 |

---

## Iteration 4

**Best so far**: 1.0098× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_lambda_mismatch_and_autotune_small_xblock_with_more_warps_for_l1tex_latency_hiding | 1.011× | 1.011× | ✓ |  |
| B | reduce_num_warps_and_fix_grid_lambda_for_l1tex_latency_hiding | 1.016× | 1.009× | ✓ | **✓ best** |
| C | fuse_batchnorm_arithmetic_into_fma_to_reduce_non_fused_fp32_instructions | 0.999× | 1.001× | ✓ |  |
| D | fuse_fp32_instructions_via_explicit_fma_in_batchnorm_arithmetic | 1.006× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_m... | Δ | reduce_num_warps_... | Δ | fuse_batchnorm_ar... | Δ | fuse_fp32_instruc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 26850 ns | 26560 ns | -290 ns (-1.1%) | 26620 ns | -230 ns (-0.9%) | 26820 ns | -30 ns (-0.1%) | 26750 ns | -100 ns (-0.4%) |
| SM Throughput (%) | 65.53 % | 32.63 % | -32.90 pp | 18.04 % | -47.49 pp | 65.38 % | -0.15 pp | 65.43 % | -0.10 pp |
| Memory Throughput (%) | 32.74 % | 14.74 % | -18.00 pp | 11.92 % | -20.82 pp | 32.66 % | -0.08 pp | 32.69 % | -0.05 pp |
| DRAM Throughput (%) | 17.13 % | 3.86 % | -13.27 pp | 7.69 % | -9.44 pp | 16.55 % | -0.58 pp | 17.27 % | +0.14 pp |
| Achieved Occupancy (%) | 53.46 % | 26.58 % | -26.88 pp | 13.62 % | -39.84 pp | 54.51 % | +1.05 pp | 53.47 % | +0.01 pp |
| Registers/Thread | 28.00 register/thread | 26.00 register/thread | -2 | 34.00 register/thread | +6 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 18.56 | 9.28 | -9.28 | 9.28 | -9.28 | 18.56 | +0.00 | 18.56 | +0.00 |
| No Eligible (%) | 28.21 % | 61.86 % | +33.65 pp | 73.28 % | +45.07 pp | 28.12 % | -0.09 pp | 28.03 % | -0.18 pp |
| IPC Active | 2.86 inst/cycle | 1.49 inst/cycle | -1.37 | 0.91 inst/cycle | -1.95 | 2.87 inst/cycle | +0.01 | 2.86 inst/cycle | +0.00 |

---

## Iteration 5

**Best so far**: 1.0098× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_batchnorm_arithmetic_as_explicit_fma_chain_to_reduce_non_fused_fp32_instructions | 1.017× | 1.010× | ✓ | **✓ best** |
| B | fix_grid_lambda_and_expand_autotune_with_larger_warps_for_l1tex_latency_hiding | 1.016× | 0.998× | ✓ |  |
| C | restructure_to_2d_channel_spatial_tiling_to_eliminate_conditional_loads | 0.392× | 0.301× | ✓ |  |
| D | fuse_bn_normalization_subtraction_multiply_into_fma_instruction | 0.999× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | rewrite_batchnorm... | Δ | fix_grid_lambda_a... | Δ | restructure_to_2d... | Δ | fuse_bn_normaliza... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 26850 ns | 26590 ns | -260 ns (-1.0%) | 26910 ns | +60 ns (+0.2%) | 89250 ns | +62400 ns (+232.4%) | 27010 ns | +160 ns (+0.6%) |
| SM Throughput (%) | 65.53 % | 32.65 % | -32.88 pp | 18.20 % | -47.33 pp | 85.03 % | +19.50 pp | 66.04 % | +0.51 pp |
| Memory Throughput (%) | 32.74 % | 14.76 % | -17.98 pp | 12.03 % | -20.71 pp | 50.19 % | +17.45 pp | 32.40 % | -0.34 pp |
| DRAM Throughput (%) | 17.13 % | 3.85 % | -13.28 pp | 7.61 % | -9.52 pp | 5.13 % | -12.00 pp | 16.44 % | -0.69 pp |
| Achieved Occupancy (%) | 53.46 % | 26.58 % | -26.88 pp | 13.29 % | -40.17 pp | 46.33 % | -7.13 pp | 55.46 % | +2.00 pp |
| Registers/Thread | 28.00 register/thread | 26.00 register/thread | -2 | 34.00 register/thread | +6 | 62.00 register/thread | +34 | 27.00 register/thread | -1 |
| Waves/SM | 18.56 | 9.28 | -9.28 | 9.28 | -9.28 | 37.12 | +18.56 | 18.56 | +0.00 |
| No Eligible (%) | 28.21 % | 62.04 % | +33.83 pp | 74.88 % | +46.67 pp | 12.97 % | -15.24 pp | 27.59 % | -0.62 pp |
| IPC Active | 2.86 inst/cycle | 1.47 inst/cycle | -1.39 | 0.92 inst/cycle | -1.94 | 3.48 inst/cycle | +0.62 | 2.87 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan C: `restructure_to_2d_channel_spatial_tiling_to_eliminate_conditional_loads` (0.301×)

Restructuring a flat 1D pointwise kernel into a 2D tile (CDIM=256 × YBLOCK) to eliminate a 4-way tl.where conditional load chain caused a 3.3x slowdown. The 2D layout caused registers per thread to explode from 26 to 62, dropping Block Limit Registers from 32 to 8 and halving theoretical occupancy from 100% to 50%. Executed instructions ballooned from 6.86M to 59.9M (~8.7x), overwhelming any benefit from eliminating the conditional chain. The CDIM=256 constexpr forces Triton to allocate 2D register tiles ([256, XBLOCK]) for all intermediate values, creating massive register pressure that is far worse than the original tl.where stalls.

> **Applicable when**: Never: this 2D restructuring approach is not applicable for this kernel class.

> **Avoid when**: Do not restructure a flat 1D pointwise kernel into a 2D tile with a large fixed constexpr channel dimension (e.g. CDIM=256) when the channel count is not a power of 2 and requires masking. The fixed CDIM forces all intermediate tensors to be [CDIM, YBLOCK] shaped, causing register count to explode (26→62 regs/thread) and Block Limit Registers to collapse (32→8), which halves theoretical occupancy and causes an ~8.7x instruction count blowup. The tl.where conditional chain in the original kernel is not the dominant bottleneck when registers per thread are already low (26) and theoretical occupancy is 100%.

---

## Cumulative Progress

```
Baseline:  26850 ns  (1.000×)
Iter 3     26590 ns  (1.010×)  — fix_grid_lambda_and_tune_num_warps_for_l1tex_latency_hiding

Final best: 26590 ns  (1.0098×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 32},  num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 64},  num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
     xnumel = 5017600
@@ -46,12 +58,10 @@

     tmp27 = tmp25 + tmp26
     tmp28 = tl.math.rsqrt(tmp27)
     tmp33 = tmp24 * tmp28
-    tmp35 = tmp33 * tmp34
-    tmp37 = tmp35 + tmp36
+    tmp37 = libdevice.fma(tmp33, tmp34, tmp36)
     tmp38 = tl.full([1], 0, tl.int32)
     tmp39 = tl.maximum(tmp38, tmp37)
     tl.store(in_out_ptr0 + (x2), tmp39, mask=xmask)
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -65,7 +75,7 @@

     arg_8 = inputs[8]
     xnumel = 5017600
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600)
     return arg_8
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
