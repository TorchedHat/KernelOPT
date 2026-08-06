# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5410 ns — **Best latency**: 5250 ns (**1.0305×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_grid_lambda_and_sweep_small_xblock_high_num_warps_for_l1tex_latency_hiding`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_39_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4410,1,1), Waves/SM: 2.09
- **Registers/thread**: 18
- **Duration**: 5410 ns
- **SM Throughput**: 35.15 % | **Memory**: 12.96 % | **DRAM**: 8.62 %
- **Achieved Occupancy**: 60.76 %
- **IPC Active**: 2.39 inst/cycle | **No Eligible**: 38.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_with_autotune_to_eliminate_partial_wave_and_improve_cache_reuse | 0.890× | 0.863× | ✓ |  |
| B | increase_xblock_eliminate_partial_wave | 0.888× | 0.838× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.908× | 0.853× | ✓ |  |
| D | increase_xblock_to_eliminate_partial_wave | 0.973× | 0.554× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_w... | Δ | increase_xblock_e... | Δ | add_autotune_with... | Δ | increase_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5410 ns | 6270 ns | +860 ns (+15.9%) | 6460 ns | +1050 ns (+19.4%) | 6340 ns | +930 ns (+17.2%) | 9760 ns | +4350 ns (+80.4%) |
| SM Throughput (%) | 35.15 % | 55.79 % | +20.64 pp | 55.73 % | +20.58 pp | 55.96 % | +20.81 pp | 68.07 % | +32.92 pp |
| Memory Throughput (%) | 12.96 % | 15.34 % | +2.38 pp | 15.30 % | +2.34 pp | 15.36 % | +2.40 pp | 15.38 % | +2.42 pp |
| DRAM Throughput (%) | 8.62 % | 7.42 % | -1.20 pp | 7.21 % | -1.41 pp | 7.35 % | -1.27 pp | 4.77 % | -3.85 pp |
| Achieved Occupancy (%) | 60.76 % | 75.47 % | +14.71 pp | 76.07 % | +15.31 pp | 75.58 % | +14.82 pp | 82.25 % | +21.49 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 2.09 | +0.00 | 2.09 | +0.00 | 4.18 | +2.09 |
| No Eligible (%) | 38.77 % | 18.86 % | -19.91 pp | 18.98 % | -19.79 pp | 18.85 % | -19.92 pp | 14.83 % | -23.94 pp |
| IPC Active | 2.39 inst/cycle | 3.21 inst/cycle | +0.82 | 3.18 inst/cycle | +0.79 | 3.21 inst/cycle | +0.82 | 3.38 inst/cycle | +0.99 |


### Key Learnings

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.853×)

Expanding the @triton.autotune config set by adding num_stages=1 and additional num_warps variants (e.g., XBLOCK=1024/num_warps=4, XBLOCK=2048/num_warps=8) produced zero change in executed instructions (2.77M both before and after), registers per thread (32 both), grid size (4410 both), and duration (6.27 vs 6.34 µs — within noise). The root cause of the regression persists: the grid is still hardcoded as `triton.cdiv(xnumel, 128)` rather than a lambda using `meta['XBLOCK']`, so the autotuner cannot actually reduce the number of blocks launched regardless of which config it selects. Adding more configs or num_stages to a broken grid lambda is a no-op that wastes autotuning budget.

> **Applicable when**: Never: expanding autotune configs without fixing the grid lambda is a no-op when the grid is hardcoded to a fixed XBLOCK value.

> **Avoid when**: Do not expand autotune config sets (adding num_stages variants, additional num_warps options) when the fundamental grid lambda bug is still present. All config variations will produce identical grid sizes and identical instruction counts, making the expansion useless. Fix the grid lambda first: `grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)`.

#### ❌ Plan D: `increase_xblock_to_eliminate_partial_wave` (0.554×)

Adding @triton.autotune with larger XBLOCK configs (256, 512, 1024) caused the autotuner to select XBLOCK=256 with num_warps=4, which nearly doubled registers per thread from 18 to 32. This reduced Block Limit Registers from 21 to 8 blocks/SM, cutting theoretical parallelism and increasing total elapsed cycles from 5.41 µs to 9.76 µs (1.8x slower). Although compute SM throughput improved (35% → 68%) and occupancy rose (60.8% → 82.3%), the 3.5x increase in executed instructions (1.5M → 5.2M) dominated, indicating the autotuner selected a configuration that performs redundant work or unrolls excessively, overwhelming any wave-elimination benefit.

> **Applicable when**: Never: this specific approach caused regression. Avoid adding autotune with larger XBLOCK configs when the kernel already has low register usage (≤20 regs/thread) and the autotune may select configs that increase register pressure.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs to kernels where: (1) the grid size is not updated to match the autotuned XBLOCK (grid remains fixed at old cdiv), causing the autotuner to run more work per block than expected; (2) larger XBLOCK configs cause register count to jump significantly (e.g., 18→32), reducing Block Limit Registers and increasing total instruction count by 3x+; (3) the kernel is already near its compute throughput ceiling for the given problem size.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | rewrite_arithmetic_for_fma_fusion_and_fix_autotune_grid_bug | 0.995× | 1.036× | ✓ |  |
| B | fix_autotune_grid_lambda_to_use_meta_xblock | 0.890× | 0.853× | ✓ |  |
| C | fix_autotune_grid_lambda_and_enable_fma_via_rsqrt | 1.025× | 1.030× | ✓ |  |
| D | fuse_bn_arithmetic_to_eliminate_non_fused_fp32_instructions | 1.043× | 1.030× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | rewrite_arithmeti... | Δ | fix_autotune_grid... | Δ | fix_autotune_grid... | Δ | fuse_bn_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5410 ns | 5220 ns | -190 ns (-3.5%) | 6340 ns | +930 ns (+17.2%) | 5250 ns | -160 ns (-3.0%) | 5250 ns | -160 ns (-3.0%) |
| SM Throughput (%) | 35.15 % | 22.07 % | -13.08 pp | 55.88 % | +20.73 pp | 22.06 % | -13.09 pp | 28.20 % | -6.95 pp |
| Memory Throughput (%) | 12.96 % | 13.78 % | +0.82 pp | 15.34 % | +2.38 pp | 13.87 % | +0.91 pp | 16.76 % | +3.80 pp |
| DRAM Throughput (%) | 8.62 % | 8.96 % | +0.34 pp | 7.38 % | -1.24 pp | 8.94 % | +0.32 pp | 8.86 % | +0.24 pp |
| Achieved Occupancy (%) | 60.76 % | 53.25 % | -7.51 pp | 75.56 % | +14.80 pp | 51.83 % | -8.93 pp | 59.72 % | -1.04 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 2.09 | +0.00 | 2.09 | +0.00 | 4.18 | +2.09 |
| No Eligible (%) | 38.77 % | 58.61 % | +19.84 pp | 18.97 % | -19.80 pp | 57.83 % | +19.06 pp | 50.97 % | +12.20 pp |
| IPC Active | 2.39 inst/cycle | 1.66 inst/cycle | -0.73 | 3.19 inst/cycle | +0.80 | 1.58 inst/cycle | -0.81 | 1.80 inst/cycle | -0.59 |

---

## Iteration 3

**Best so far**: 1.0305× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_at_fixed_xblock_to_hide_l1tex_latency | 0.995× | 1.044× | ✓ |  |
| B | increase_num_warps_at_fixed_xblock_to_hide_l1tex_stall_latency | 1.020× | 0.995× | ✓ |  |
| C | add_evict_first_on_streaming_load_to_improve_l1_hit_rate_for_scale_bias | 1.043× | 1.036× | ✓ |  |
| D | fix_grid_lambda_and_enable_fma_via_float_division | 1.023× | 1.019× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | increase_num_warp... | Δ | add_evict_first_o... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5410 ns | 5180 ns | -230 ns (-4.3%) | 5440 ns | +30 ns (+0.6%) | 5220 ns | -190 ns (-3.5%) | 5310 ns | -100 ns (-1.8%) |
| SM Throughput (%) | 35.15 % | 21.99 % | -13.16 pp | 13.07 % | -22.08 pp | 30.41 % | -4.74 pp | 29.47 % | -5.68 pp |
| Memory Throughput (%) | 12.96 % | 13.71 % | +0.75 pp | 13.28 % | +0.32 pp | 16.18 % | +3.22 pp | 19.47 % | +6.51 pp |
| DRAM Throughput (%) | 8.62 % | 9.01 % | +0.39 pp | 8.62 % | +0.00 pp | 8.88 % | +0.26 pp | 8.83 % | +0.21 pp |
| Achieved Occupancy (%) | 60.76 % | 52.89 % | -7.87 pp | 53.67 % | -7.09 pp | 58.64 % | -2.12 pp | 52.11 % | -8.65 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 22.00 register/thread | +4 | 16.00 register/thread | -2 | 32.00 register/thread | +14 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 1.04 | -1.05 | 4.18 | +2.09 | 2.09 | +0.00 |
| No Eligible (%) | 38.77 % | 59.99 % | +21.22 pp | 58.47 % | +19.70 pp | 45.16 % | +6.39 pp | 35.24 % | -3.53 pp |
| IPC Active | 2.39 inst/cycle | 1.67 inst/cycle | -0.72 | 1.49 inst/cycle | -0.90 | 2.14 inst/cycle | -0.25 | 2.58 inst/cycle | +0.19 |

---

## Iteration 4

**Best so far**: 1.0305× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | collapse_bn_normalization_and_scale_into_single_fma_to_reduce_non_fused_fp32_instructions | 1.023× | 1.025× | ✓ |  |
| B | fix_grid_lambda_and_sweep_small_xblock_high_num_warps_for_l1tex_latency_hiding | 1.000× | 1.030× | ✓ |  |
| C | fix_autotune_grid_lambda_to_use_meta_xblock_and_add_small_xblock_fallback | 1.024× | 1.050× | ✓ |  |
| D | rewrite_bn_arithmetic_as_fma_chain_to_eliminate_non_fused_fp32_instructions | 1.023× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | collapse_bn_norma... | Δ | fix_grid_lambda_a... | Δ | fix_autotune_grid... | Δ | rewrite_bn_arithm... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5410 ns | 5280 ns | -130 ns (-2.4%) | 5250 ns | -160 ns (-3.0%) | 5150 ns | -260 ns (-4.8%) | 5280 ns | -130 ns (-2.4%) |
| SM Throughput (%) | 35.15 % | 22.29 % | -12.86 pp | 17.20 % | -17.95 pp | 22.34 % | -12.81 pp | 12.50 % | -22.65 pp |
| Memory Throughput (%) | 12.96 % | 13.81 % | +0.85 pp | 8.48 % | -4.48 pp | 14.04 % | +1.08 pp | 12.67 % | -0.29 pp |
| DRAM Throughput (%) | 8.62 % | 8.84 % | +0.22 pp | 4.44 % | -4.18 pp | 9.02 % | +0.40 pp | 8.85 % | +0.23 pp |
| Achieved Occupancy (%) | 60.76 % | 53.38 % | -7.38 pp | 49.91 % | -10.85 pp | 52.43 % | -8.33 pp | 56.73 % | -4.03 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 22.00 register/thread | +4 |
| Waves/SM | 2.09 | 2.09 | +0.00 | 2.09 | +0.00 | 2.09 | +0.00 | 1.04 | -1.05 |
| No Eligible (%) | 38.77 % | 58.67 % | +19.90 pp | 66.58 % | +27.81 pp | 58.54 % | +19.77 pp | 58.23 % | +19.46 pp |
| IPC Active | 2.39 inst/cycle | 1.59 inst/cycle | -0.80 | 1.20 inst/cycle | -1.19 | 1.63 inst/cycle | -0.76 | 1.36 inst/cycle | -1.03 |

---

## Cumulative Progress

```
Baseline:  5410 ns  (1.000×)
Iter 2     5250 ns  (1.030×)  — fuse_bn_arithmetic_to_eliminate_non_fused_fp32_instructions

Final best: 5250 ns  (1.0305×)
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
+        triton.Config({'XBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 128}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 564480
@@ -21,21 +30,12 @@

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
-    tmp16 = tl.full([1], 0.0, tl.float32)
-    tmp17 = tl.maximum(tmp15, tmp16)
-    tmp18 = tl.full([1], 6.0, tl.float32)
-    tmp19 = tl.minimum(tmp17, tmp18)
+    tmp6 = tl.math.rsqrt(tmp5)
+    tmp11 = tmp2 * tmp6
+    tmp15 = libdevice.fma(tmp11, tmp12, tmp14)
+    tmp17 = tl.maximum(tmp15, tl.full([1], 0.0, tl.float32))
+    tmp19 = tl.minimum(tmp17, tl.full([1], 6.0, tl.float32))
     tl.store(in_out_ptr0 + (x2), tmp19, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -45,7 +45,7 @@

     arg_4 = inputs[4]
     xnumel = 564480
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 564480, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_39[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 564480)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
