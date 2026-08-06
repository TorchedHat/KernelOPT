# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12320 ns — **Best latency**: 12220 ns (**1.0082×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_l1tex_stall_via_rsqrt_and_evict_first_on_main_tensor`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_014/kernel_5_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 17
- **Duration**: 12320 ns
- **SM Throughput**: 47.62 % | **Memory**: 21.01 % | **DRAM**: 13.32 %
- **Achieved Occupancy**: 58.04 %
- **IPC Active**: 2.31 inst/cycle | **No Eligible**: 40.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_to_hide_memory_latency | 0.923× | 0.897× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_latency_hiding | 0.987× | 0.223× | ✓ |  |
| C | increase_xblock_vectorized_loads | 0.770× | 0.240× | ✓ |  |
| D | increase_xblock_with_autotune | 0.925× | 0.223× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | increase_xblock_v... | Δ | increase_xblock_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 13730 ns | +1410 ns (+11.4%) | 55170 ns | +42850 ns (+347.8%) | 51300 ns | +38980 ns (+316.4%) | 55170 ns | +42850 ns (+347.8%) |
| SM Throughput (%) | 47.62 % | 67.29 % | +19.67 pp | 85.77 % | +38.15 pp | 84.87 % | +37.25 pp | 85.79 % | +38.17 pp |
| Memory Throughput (%) | 21.01 % | 25.68 % | +4.67 pp | 16.86 % | -4.15 pp | 18.07 % | -2.94 pp | 16.86 % | -4.15 pp |
| DRAM Throughput (%) | 13.32 % | 11.94 % | -1.38 pp | 2.96 % | -10.36 pp | 3.19 % | -10.13 pp | 2.97 % | -10.35 pp |
| Achieved Occupancy (%) | 58.04 % | 69.53 % | +11.49 pp | 84.78 % | +26.74 pp | 55.15 % | -2.89 pp | 84.46 % | +26.42 pp |
| Registers/Thread | 17.00 register/thread | 24.00 register/thread | +7 | 32.00 register/thread | +15 | 48.00 register/thread | +31 | 32.00 register/thread | +15 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 29.70 | +22.28 | 23.76 | +16.34 | 29.70 | +22.28 |
| No Eligible (%) | 40.74 % | 16.97 % | -23.77 pp | 10.96 % | -29.78 pp | 11.39 % | -29.35 pp | 10.70 % | -30.04 pp |
| IPC Active | 2.31 inst/cycle | 3.32 inst/cycle | +1.01 | 3.56 inst/cycle | +1.25 | 3.53 inst/cycle | +1.22 | 3.56 inst/cycle | +1.25 |


### Key Learnings

#### ❌ Plan D: `increase_xblock_with_autotune` (0.223×)

Increasing XBLOCK from 128 to 512 via autotune caused registers per thread to jump from 17 to 32, and executed instructions to explode from ~4.6M to ~37.3M (8x increase) despite the same total work. This instruction explosion — likely from unrolled loops, extra masking logic, and register spills — increased kernel duration from 12.3 µs to 55.2 µs (4.5x slower). Although occupancy improved (58% → 84%) and scheduler eligibility improved (1.28 → 8.03 eligible warps), the massive instruction count increase dominated, causing a 4.5x regression. For short-duration pointwise kernels where the baseline is already memory-throughput-limited at high effective bandwidth (652 GB/s), increasing block size via autotune can backfire by inflating register pressure and instruction count far beyond the latency-hiding benefit.

> **Applicable when**: Kernel has very low register count (≤20 regs/thread), low occupancy due to L1TEX scoreboard stalls, and few warps per block. Theoretically, increasing block size should help hide latency.

> **Avoid when**: The kernel is already achieving high effective memory throughput (>500 GB/s) relative to device peak, duration is short (<20 µs), and the baseline instruction count is low. In these cases, increasing XBLOCK causes register pressure to double and instruction count to explode 8x, overwhelming any latency-hiding benefit. Also avoid when the autotune overhead itself may dominate short-running kernels.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_non_fused_fp32_instructions_via_rsqrt_and_fma_rewrite | 1.003× | 0.990× | ✓ |  |
| B | remove_autotune_to_restore_low_register_pressure_and_occupancy | 0.989× | 0.992× | ✓ |  |
| C | reduce_non_fused_fp32_instructions_via_fma_rewrite | 1.014× | 0.393× | ✓ | **✓ best** |
| D | reduce_instruction_count_via_small_xblock_and_fma_promotion | 1.002× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_non_fused_... | Δ | remove_autotune_t... | Δ | reduce_non_fused_... | Δ | reduce_instructio... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 12450 ns | +130 ns (+1.1%) | 12420 ns | +100 ns (+0.8%) | 31330 ns | +19010 ns (+154.3%) | 12160 ns | -160 ns (-1.3%) |
| SM Throughput (%) | 47.62 % | 38.44 % | -9.18 pp | 47.59 % | -0.03 pp | 76.02 % | +28.40 pp | 31.89 % | -15.73 pp |
| Memory Throughput (%) | 21.01 % | 28.64 % | +7.63 pp | 20.87 % | -0.14 pp | 30.60 % | +9.59 pp | 20.94 % | -0.07 pp |
| DRAM Throughput (%) | 13.32 % | 13.18 % | -0.14 pp | 13.22 % | -0.10 pp | 5.22 % | -8.10 pp | 13.50 % | +0.18 pp |
| Achieved Occupancy (%) | 58.04 % | 35.04 % | -23.00 pp | 58.18 % | +0.14 pp | 75.40 % | +17.36 pp | 41.69 % | -16.35 pp |
| Registers/Thread | 17.00 register/thread | 22.00 register/thread | +5 | 17.00 register/thread | +0 | 32.00 register/thread | +15 | 16.00 register/thread | -1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 29.70 | +22.28 | 7.42 | +0.00 |
| No Eligible (%) | 40.74 % | 48.57 % | +7.83 pp | 41.24 % | +0.50 pp | 18.51 % | -22.23 pp | 65.95 % | +25.21 pp |
| IPC Active | 2.31 inst/cycle | 2.03 inst/cycle | -0.28 | 2.34 inst/cycle | +0.03 | 3.25 inst/cycle | +0.94 | 1.35 inst/cycle | -0.96 |


### Key Learnings

#### ❌ Plan C: `reduce_non_fused_fp32_instructions_via_fma_rewrite` (0.393×)

Rewriting the BN+ReLU arithmetic chain to use tl.math.rsqrt and tl.math.fma (to reduce non-fused FP32 instructions) was bundled with an autotune config change that increased block size from 128 to 512, causing executed instructions to balloon from 6.96M to 18.85M (2.7x increase) and kernel duration to more than double (13.73 µs → 31.33 µs). The FMA arithmetic rewrite itself may have been neutral or mildly beneficial, but the confounding autotune block-size increase (same anti-pattern as iteration 1) dominated the result, inflating register count from 24 to 32 and waves-per-SM from 7.42 to 29.7. When the NCU FPInstructions rule fires on a kernel that is already in a regressed state from a prior block-size increase, the root cause is the block-size regression, not the FP instruction mix — fixing FP fusion while leaving the inflated block size in place cannot recover performance.

> **Applicable when**: Never apply this combined pattern. The FMA arithmetic rewrite (tl.math.rsqrt + tl.math.fma) may be valid in isolation, but only when the block size and grid configuration remain unchanged from a known-good baseline.

> **Avoid when**: Do not apply FMA rewrites simultaneously with autotune block-size changes. If the kernel is already in a regressed state from a prior block-size increase (high waves-per-SM, high instruction count), fix the block-size regression first before attempting arithmetic-level optimizations. The FPInstructions NCU rule firing on a regressed kernel does not mean FP fusion is the root cause — it may simply reflect the inflated instruction count from the block-size regression.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_for_l1tex_latency_hiding | 1.001× | 1.016× | ✓ |  |
| B | increase_num_warps_to_hide_l1tex_stall_latency | 0.988× | 1.002× | ✓ |  |
| C | fix_autotune_grid_to_use_meta_xblock_lambda | 0.988× | 1.005× | ✓ |  |
| D | revert_to_small_fixed_xblock_to_eliminate_instruction_explosion | 1.015× | 1.008× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_num_stages_fo... | Δ | increase_num_warp... | Δ | fix_autotune_grid... | Δ | revert_to_small_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 12130 ns | -190 ns (-1.5%) | 12290 ns | -30 ns (-0.2%) | 12260 ns | -60 ns (-0.5%) | 12220 ns | -100 ns (-0.8%) |
| SM Throughput (%) | 47.62 % | 31.87 % | -15.75 pp | 47.65 % | +0.03 pp | 38.28 % | -9.34 pp | 31.97 % | -15.65 pp |
| Memory Throughput (%) | 21.01 % | 20.96 % | -0.05 pp | 21.07 % | +0.06 pp | 28.52 % | +7.51 pp | 21.04 % | +0.03 pp |
| DRAM Throughput (%) | 13.32 % | 13.51 % | +0.19 pp | 13.34 % | +0.02 pp | 13.37 % | +0.05 pp | 13.43 % | +0.11 pp |
| Achieved Occupancy (%) | 58.04 % | 41.86 % | -16.18 pp | 56.64 % | -1.40 pp | 34.97 % | -23.07 pp | 41.77 % | -16.27 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 17.00 register/thread | +0 | 22.00 register/thread | +5 | 16.00 register/thread | -1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.74 % | 66.51 % | +25.77 pp | 41.02 % | +0.28 pp | 48.53 % | +7.79 pp | 66.31 % | +25.57 pp |
| IPC Active | 2.31 inst/cycle | 1.34 inst/cycle | -0.97 | 2.31 inst/cycle | +0.00 | 2.04 inst/cycle | -0.27 | 1.36 inst/cycle | -0.95 |

---

## Iteration 4

**Best so far**: 1.0082× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_first_to_streaming_main_tensor_load_to_improve_bn_param_l1_reuse | 1.013× | 1.002× | ✓ |  |
| B | add_evict_first_to_streaming_main_tensor_load | 1.000× | 1.016× | ✓ |  |
| C | add_evict_first_to_output_store_for_l2_streaming | 0.892× | 1.005× | ✓ |  |
| D | reduce_l1tex_stall_via_rsqrt_and_evict_first_on_main_tensor | 1.014× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_first_t... | Δ | add_evict_first_t... | Δ | add_evict_first_t... | Δ | reduce_l1tex_stal... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 12290 ns | -30 ns (-0.2%) | 12130 ns | -190 ns (-1.5%) | 12260 ns | -60 ns (-0.5%) | 12220 ns | -100 ns (-0.8%) |
| SM Throughput (%) | 47.62 % | 31.92 % | -15.70 pp | 31.89 % | -15.73 pp | 38.67 % | -8.95 pp | 31.73 % | -15.89 pp |
| Memory Throughput (%) | 21.01 % | 21.14 % | +0.13 pp | 21.16 % | +0.15 pp | 28.46 % | +7.45 pp | 20.91 % | -0.10 pp |
| DRAM Throughput (%) | 13.32 % | 13.35 % | +0.03 pp | 13.51 % | +0.19 pp | 13.36 % | +0.04 pp | 13.40 % | +0.08 pp |
| Achieved Occupancy (%) | 58.04 % | 43.58 % | -14.46 pp | 42.49 % | -15.55 pp | 35.79 % | -22.25 pp | 43.70 % | -14.34 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 16.00 register/thread | -1 | 22.00 register/thread | +5 | 16.00 register/thread | -1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 | 7.42 | +0.00 |
| No Eligible (%) | 40.74 % | 61.90 % | +21.16 pp | 62.78 % | +22.04 pp | 47.48 % | +6.74 pp | 61.99 % | +21.25 pp |
| IPC Active | 2.31 inst/cycle | 1.52 inst/cycle | -0.79 | 1.48 inst/cycle | -0.83 | 2.05 inst/cycle | -0.26 | 1.52 inst/cycle | -0.79 |

---

## Iteration 5

**Best so far**: 1.0082× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_via_larger_block_size_for_l1tex_latency_hiding | 0.999× | 1.013× | ✓ |  |
| B | restructure_to_2d_grid_to_eliminate_modulo_and_reduce_instruction_count | 1.013× | 1.011× | ✓ |  |
| C | increase_num_warps_at_fixed_xblock_to_hide_l1tex_stall_latency | 1.003× | 0.863× | ✓ |  |
| D | increase_num_warps_to_hide_l1tex_stall_without_changing_xblock | 1.000× | 1.013× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | restructure_to_2d... | Δ | increase_num_warp... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12320 ns | 12160 ns | -160 ns (-1.3%) | 12190 ns | -130 ns (-1.1%) | 14270 ns | +1950 ns (+15.8%) | 12160 ns | -160 ns (-1.3%) |
| SM Throughput (%) | 47.62 % | 31.90 % | -15.72 pp | 31.34 % | -16.28 pp | 61.20 % | +13.58 pp | 32.02 % | -15.60 pp |
| Memory Throughput (%) | 21.01 % | 21.08 % | +0.07 pp | 20.65 % | -0.36 pp | 33.74 % | +12.73 pp | 21.11 % | +0.10 pp |
| DRAM Throughput (%) | 13.32 % | 13.49 % | +0.17 pp | 13.44 % | +0.12 pp | 11.48 % | -1.84 pp | 13.45 % | +0.13 pp |
| Achieved Occupancy (%) | 58.04 % | 43.85 % | -14.19 pp | 42.83 % | -15.21 pp | 73.64 % | +15.60 pp | 42.92 % | -15.12 pp |
| Registers/Thread | 17.00 register/thread | 16.00 register/thread | -1 | 18.00 register/thread | +1 | 19.00 register/thread | +2 | 16.00 register/thread | -1 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 7.42 | +0.00 | 14.85 | +7.43 | 7.42 | +0.00 |
| No Eligible (%) | 40.74 % | 62.68 % | +21.94 pp | 61.95 % | +21.21 pp | 28.04 % | -12.70 pp | 61.80 % | +21.06 pp |
| IPC Active | 2.31 inst/cycle | 1.45 inst/cycle | -0.86 | 1.52 inst/cycle | -0.79 | 2.88 inst/cycle | +0.57 | 1.51 inst/cycle | -0.80 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_at_fixed_xblock_to_hide_l1tex_stall_latency` (0.863×)

Increasing num_warps from 4 to 8 (block size 128→256) via autotune with a fixed XBLOCK=128 config improved occupancy (42.5%→73.6%) and scheduler eligibility (0.69→2.49 eligible warps), but caused executed instructions to balloon from 2.82M to 6.77M (2.4x increase) and kernel duration to increase from 12.13 µs to 14.27 µs (1.16x slower). The autotune selected XBLOCK=256 (not XBLOCK=128 as intended), doubling the tile size and thus doubling the work per CTA while the grid remained fixed, causing each element to be processed with more instructions. The grid lambda `(cdiv(xnumel, 128),)` was not updated to use `meta['XBLOCK']`, so the autotune's selection of XBLOCK=256 caused the same instruction-explosion anti-pattern seen in prior iterations — the fix for the grid was applied in the kernel call but the hardcoded grid in `kernel_function` was not changed to a lambda.

> **Applicable when**: Never: increasing num_warps via autotune at a nominally fixed XBLOCK is unsafe if the autotune config set also includes larger XBLOCK values and the grid is not a proper lambda using meta['XBLOCK'].

> **Avoid when**: Do not add autotune configs with multiple XBLOCK values when the grid in kernel_function is hardcoded to a fixed tile size. Even if the intent is to only vary num_warps at a fixed XBLOCK, the autotune will select the best-performing config across all XBLOCK values, and a mismatched fixed grid will cause instruction explosion proportional to new_XBLOCK/old_XBLOCK. Also avoid when the kernel uses restore_value on an in-place buffer, as this multiplies autotuning overhead.

---

## Cumulative Progress

```
Baseline:  12320 ns  (1.000×)
Iter 3     12220 ns  (1.008×)  — revert_to_small_fixed_xblock_to_eliminate_instruction_explosion

Final best: 12220 ns  (1.0082×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,27 @@

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
+    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
     tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
     tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
     tmp12 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
+    tmp5 = tmp3 + tl.full([1], 1e-05, tl.float32)
+    inv_std = tl.math.rsqrt(tmp5)
+    normed = tmp2 * inv_std
+    scaled = tl.math.fma(normed, tmp12, tmp14)
+    zero = tl.zeros([1], tl.float32)
+    result = tl.maximum(zero, scaled)
+    tl.store(in_out_ptr0 + (x2), result, None, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
