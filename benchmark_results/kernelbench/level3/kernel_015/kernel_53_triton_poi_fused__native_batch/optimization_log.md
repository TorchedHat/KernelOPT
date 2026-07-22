# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_53` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7490 ns — **Best latency**: 7390 ns (**1.0135×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`streaming_eviction_on_output_store_in_isolation`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_53`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_53_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (7840,1,1), Waves/SM: 3.71
- **Registers/thread**: 18
- **Duration**: 7490 ns
- **SM Throughput**: 39.05 % | **Memory**: 16.66 % | **DRAM**: 11.01 %
- **Achieved Occupancy**: 57.49 %
- **IPC Active**: 2.26 inst/cycle | **No Eligible**: 44.86 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_for_latency_hiding | 0.987× | 0.992× | ✓ |  |
| B | autotune_xblock_and_num_warps_for_pointwise_bn_relu | 0.999× | 0.457× | ✓ |  |
| C | add_autotune_with_larger_xblock_for_vectorized_loads | 0.915× | 0.760× | ✓ |  |
| D | autotune_xblock_and_num_warps_to_hide_l1tex_latency | 1.003× | 0.253× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_a... | Δ | add_autotune_with... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7550 ns | +60 ns (+0.8%) | 16380 ns | +8890 ns (+118.7%) | 9860 ns | +2370 ns (+31.6%) | 29630 ns | +22140 ns (+295.6%) |
| SM Throughput (%) | 39.05 % | 39.78 % | +0.73 pp | 74.95 % | +35.90 pp | 65.37 % | +26.32 pp | 80.67 % | +41.62 pp |
| Memory Throughput (%) | 16.66 % | 16.93 % | +0.27 pp | 16.48 % | -0.18 pp | 17.68 % | +1.02 pp | 15.55 % | -1.11 pp |
| DRAM Throughput (%) | 11.01 % | 10.90 % | -0.11 pp | 5.01 % | -6.00 pp | 8.36 % | -2.65 pp | 2.77 % | -8.24 pp |
| Achieved Occupancy (%) | 57.49 % | 58.15 % | +0.66 pp | 84.81 % | +27.32 pp | 81.90 % | +24.41 pp | 85.02 % | +27.53 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 7.42 | +3.71 | 3.71 | +0.00 | 14.85 | +11.14 |
| No Eligible (%) | 44.86 % | 46.10 % | +1.24 pp | 13.98 % | -30.88 pp | 17.35 % | -27.51 pp | 12.77 % | -32.09 pp |
| IPC Active | 2.26 inst/cycle | 2.30 inst/cycle | +0.04 | 3.42 inst/cycle | +1.16 | 3.30 inst/cycle | +1.04 | 3.48 inst/cycle | +1.22 |


### Key Learnings

#### ❌ Plan C: `add_autotune_with_larger_xblock_for_vectorized_loads` (0.760×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) improved occupancy (57% → 82%) and scheduler eligibility (1.37 → 6.52 warps), but caused registers per thread to jump from 18 to 32 and total executed instructions to more than double (2.35M → 5.06M). The increased register pressure and instruction count outweighed the scheduling gains, resulting in a 32% slowdown (7.49 µs → 9.86 µs). When the kernel body is already register-efficient at small XBLOCK, forcing larger tiles through autotune can inflate register usage and instruction count enough to negate any memory-access or occupancy benefit.

> **Applicable when**: Kernel has very low register usage (<20/thread) AND the bottleneck is purely memory-bandwidth limited (not scoreboard/L1TEX stall driven by narrow transactions) AND larger tiles do not increase register pressure.

> **Avoid when**: Kernel already has moderate register usage (18–24/thread) and the IssueSlotUtilization stall is driven by L1TEX scoreboard waits on channel/scale vectors with high L1 hit rate (>60%); in this case, larger XBLOCK inflates register count and instruction volume without proportionally reducing memory latency, causing net regression. Also avoid when the grid is already producing multiple full waves (waves_per_sm > 3) and the partial-wave penalty is the only stated concern.

#### ❌ Plan D: `autotune_xblock_and_num_warps_to_hide_l1tex_latency` (0.253×)

Increasing block size via @triton.autotune to improve warp occupancy and hide L1TEX latency caused an 8x increase in executed instructions (2.35M → 18.79M) and a 4x increase in kernel duration (7.49us → 29.63us). The autotune selected XBLOCK=512 with 32 registers/thread, but the grid was not updated to use meta['XBLOCK'] — it remained hardcoded to cdiv(xnumel, 128), causing each element to be processed ~4x redundantly (4,014,080 threads for 1,003,520 elements). The implementation flaw was that the grid lambda was not applied: the launch still used the fixed grid=(cdiv(xnumel,128),) while the kernel ran with XBLOCK=512, causing massive over-computation. Additionally, adding xmask guards where the original kernel used tl.full([XBLOCK], True) (because xnumel was exactly divisible by the original XBLOCK=128) introduced unnecessary predication overhead when larger XBLOCK values don't evenly divide xnumel.

> **Applicable when**: Kernel is latency-bound with low eligible warps per scheduler (<2.0), high L1TEX stall fraction (>40% of CPI), low achieved occupancy (<65%), and few warps per CTA due to small block size. Register pressure is low (<20 registers/thread).

> **Avoid when**: The grid is not updated to use meta['XBLOCK'] alongside the autotune — this causes redundant thread launches proportional to the XBLOCK ratio increase. Do not apply when xnumel is not divisible by all candidate XBLOCK values without adding xmask guards that introduce predication overhead. Do not apply when the kernel is already compute-bound (SM throughput >80%).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_scoreboard_stalls | 1.002× | 1.000× | ✓ |  |
| B | rewrite_bn_arithmetic_to_exploit_fma_instructions | 0.954× | 0.963× | ✓ |  |
| C | restructure_arithmetic_to_enable_fma_fusion | 0.973× | 0.456× | ✓ |  |
| D | rewrite_arithmetic_as_explicit_fma_to_reduce_non_fused_fp32_instructions | 1.010× | 0.325× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | rewrite_bn_arithm... | Δ | restructure_arith... | Δ | rewrite_arithmeti... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7490 ns | +0 ns (+0.0%) | 7780 ns | +290 ns (+3.9%) | 16420 ns | +8930 ns (+119.2%) | 23070 ns | +15580 ns (+208.0%) |
| SM Throughput (%) | 39.05 % | 39.16 % | +0.11 pp | 55.35 % | +16.30 pp | 75.36 % | +36.31 pp | 74.25 % | +35.20 pp |
| Memory Throughput (%) | 16.66 % | 16.54 % | -0.12 pp | 24.74 % | +8.08 pp | 16.59 % | -0.07 pp | 20.33 % | +3.67 pp |
| DRAM Throughput (%) | 11.01 % | 10.99 % | -0.02 pp | 10.58 % | -0.43 pp | 5.00 % | -6.01 pp | 3.56 % | -7.45 pp |
| Achieved Occupancy (%) | 57.49 % | 58.02 % | +0.53 pp | 60.14 % | +2.65 pp | 85.31 % | +27.82 pp | 83.25 % | +25.76 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 32.00 register/thread | +14 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 7.42 | +3.71 | 14.85 | +11.14 |
| No Eligible (%) | 44.86 % | 41.72 % | -3.14 pp | 19.87 % | -24.99 pp | 14.12 % | -30.74 pp | 18.04 % | -26.82 pp |
| IPC Active | 2.26 inst/cycle | 2.28 inst/cycle | +0.02 | 3.19 inst/cycle | +0.93 | 3.37 inst/cycle | +1.11 | 3.28 inst/cycle | +1.02 |


### Key Learnings

#### ❌ Plan C: `restructure_arithmetic_to_enable_fma_fusion` (0.456×)

Restructuring BN arithmetic to use float literals and libdevice.fma (eliminating integer-to-float casts and no-op multiplies) was bundled with adding @triton.autotune and changing the grid from a fixed tuple to an adaptive lambda. The autotune selected XBLOCK=256 with 32 registers/thread (up from 18), causing executed instructions to balloon from 2.35M to 9.64M — a 4x increase — and duration to more than double (7.55µs → 16.42µs). The FMA arithmetic restructuring itself is sound, but the simultaneous introduction of autotune with a larger block size drove register pressure up and instruction count up, completely masking any FMA benefit. The root cause is identical to prior iterations: autotune selecting larger XBLOCK inflates register usage and instruction volume on this kernel.

> **Applicable when**: NCU FPInstructions rule flags a large fraction of non-fused FP32 instructions AND the kernel uses integer constants (tl.full([1], 1, tl.int32)) in FP arithmetic chains AND the kernel is NOT simultaneously being autotuned for block size.

> **Avoid when**: Do not combine FMA arithmetic restructuring with @triton.autotune block-size sweeps in the same iteration. On this kernel class (18 regs/thread at XBLOCK=128), any autotune config selecting XBLOCK>128 drives registers to 32/thread and multiplies instruction count by 4x, completely overwhelming any FMA savings. Do not apply when the kernel is already memory-latency bound (low compute SOL <50%) — FMA fusion only helps compute-bound kernels.

#### ❌ Plan D: `rewrite_arithmetic_as_explicit_fma_to_reduce_non_fused_fp32_instructions` (0.325×)

Rewriting BN+ReLU arithmetic to use tl.math.fma() was bundled with adding @triton.autotune and changing block size from 128 to 512, which caused executed instructions to explode from 2.35M to 13.69M (~6x) and duration to triple (7.55µs → 23.07µs). The FMA arithmetic change itself is a minor optimization, but the simultaneous addition of @triton.autotune with a fixed grid (cdiv(xnumel, 128)) caused the same redundant-computation anti-pattern seen in prior iterations: the grid was not updated to use meta['XBLOCK'], so the autotuned XBLOCK=512 processed each element ~4x redundantly. Additionally, registers per thread jumped from 18 to 32, further limiting occupancy headroom. The FPInstructions NCU rule does not justify a block-size change — arithmetic rewrites must be applied in isolation without touching launch configuration.

> **Applicable when**: Kernel has high non-fused FP32 instruction ratio (>3:1 non-fused to fused) and is compute-bound, AND the arithmetic rewrite is applied in complete isolation without any launch configuration changes.

> **Avoid when**: Do not bundle FMA arithmetic rewrites with @triton.autotune or block size changes — the confounding launch change dominates and causes regression. Do not apply when the kernel already has low register count (18/thread) and adding autotune would inflate registers to 32/thread. Do not apply when the grid is not updated to use meta['XBLOCK'] alongside any autotune addition.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_sm_count_to_eliminate_partial_wave | 0.986× | 0.996× | ✓ |  |
| B | fix_autotune_grid_to_use_meta_xblock_lambda | 0.973× | 0.963× | ✓ |  |
| C | apply_streaming_cache_eviction_on_output_store_to_free_l2_for_bn_parameters | 0.981× | 0.455× | ✓ |  |
| D | remove_broken_autotune_and_fix_xblock_grid_mismatch | 0.993× | 1.026× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_grid_to_sm_... | Δ | fix_autotune_grid... | Δ | apply_streaming_c... | Δ | remove_broken_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7520 ns | +30 ns (+0.4%) | 7780 ns | +290 ns (+3.9%) | 16480 ns | +8990 ns (+120.0%) | 7300 ns | -190 ns (-2.5%) |
| SM Throughput (%) | 39.05 % | 45.68 % | +6.63 pp | 55.55 % | +16.50 pp | 74.97 % | +35.92 pp | 35.10 % | -3.95 pp |
| Memory Throughput (%) | 16.66 % | 17.20 % | +0.54 pp | 24.76 % | +8.10 pp | 16.49 % | -0.17 pp | 17.59 % | +0.93 pp |
| DRAM Throughput (%) | 11.01 % | 10.97 % | -0.04 pp | 10.56 % | -0.45 pp | 4.98 % | -6.03 pp | 11.27 % | +0.26 pp |
| Achieved Occupancy (%) | 57.49 % | 60.43 % | +2.94 pp | 60.80 % | +3.31 pp | 84.90 % | +27.41 pp | 50.54 % | -6.95 pp |
| Registers/Thread | 18.00 register/thread | 20.00 register/thread | +2 | 32.00 register/thread | +14 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 7.42 | +3.71 | 3.71 | +0.00 |
| No Eligible (%) | 44.86 % | 33.37 % | -11.49 pp | 19.63 % | -25.23 pp | 14.28 % | -30.58 pp | 51.21 % | +6.35 pp |
| IPC Active | 2.26 inst/cycle | 2.61 inst/cycle | +0.35 | 3.19 inst/cycle | +0.93 | 3.43 inst/cycle | +1.17 | 2.03 inst/cycle | -0.23 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | align_grid_to_sm_multiple_to_eliminate_partial_wave | 1.011× | 1.026× | ✓ |  |
| B | reduce_num_warps_to_double_blocks_per_sm_and_hide_l1tex_latency | 1.001× | 0.959× | ✓ |  |
| C | remove_broken_autotune_restore_correct_fixed_xblock128_kernel | 1.015× | 1.026× | ✓ | **✓ best** |
| D | fix_grid_lambda_to_match_autotuned_xblock | 0.997× | 0.456× | ✓ |  |


### Metric Comparison

| Metric | Baseline | align_grid_to_sm_... | Δ | reduce_num_warps_... | Δ | remove_broken_aut... | Δ | fix_grid_lambda_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7300 ns | -190 ns (-2.5%) | 7810 ns | +320 ns (+4.3%) | 7300 ns | -190 ns (-2.5%) | 16420 ns | +8930 ns (+119.2%) |
| SM Throughput (%) | 39.05 % | 34.19 % | -4.86 pp | 45.09 % | +6.04 pp | 35.47 % | -3.58 pp | 74.98 % | +35.93 pp |
| Memory Throughput (%) | 16.66 % | 17.14 % | +0.48 pp | 16.91 % | +0.25 pp | 17.47 % | +0.81 pp | 16.48 % | -0.18 pp |
| DRAM Throughput (%) | 11.01 % | 11.27 % | +0.26 pp | 10.52 % | -0.49 pp | 11.29 % | +0.28 pp | 5.00 % | -6.01 pp |
| Achieved Occupancy (%) | 57.49 % | 51.11 % | -6.38 pp | 62.24 % | +4.75 pp | 48.44 % | -9.05 pp | 84.82 % | +27.33 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 20.00 register/thread | +2 | 16.00 register/thread | -2 | 32.00 register/thread | +14 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 7.42 | +3.71 |
| No Eligible (%) | 44.86 % | 48.94 % | +4.08 pp | 35.72 % | -9.14 pp | 58.21 % | +13.35 pp | 14.13 % | -30.73 pp |
| IPC Active | 2.26 inst/cycle | 1.97 inst/cycle | -0.29 | 2.55 inst/cycle | +0.29 | 1.66 inst/cycle | -0.60 | 3.42 inst/cycle | +1.16 |

---

## Iteration 5

**Best so far**: 1.026× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | apply_streaming_eviction_on_output_store_in_isolation | 1.013× | 1.018× | ✓ |  |
| B | streaming_eviction_on_output_store_in_isolation | 1.016× | 1.014× | ✓ | **✓ best** |
| C | pad_grid_to_next_sm_multiple_to_eliminate_partial_wave | 0.988× | 0.996× | ✓ |  |
| D | remove_broken_autotune_restore_xblock128_with_correct_fixed_grid | 1.004× | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | apply_streaming_e... | Δ | streaming_evictio... | Δ | pad_grid_to_next_... | Δ | remove_broken_aut... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7490 ns | 7360 ns | -130 ns (-1.7%) | 7390 ns | -100 ns (-1.3%) | 7520 ns | +30 ns (+0.4%) | 7550 ns | +60 ns (+0.8%) |
| SM Throughput (%) | 39.05 % | 35.12 % | -3.93 pp | 34.92 % | -4.13 pp | 45.94 % | +6.89 pp | 39.84 % | +0.79 pp |
| Memory Throughput (%) | 16.66 % | 17.56 % | +0.90 pp | 17.32 % | +0.66 pp | 17.31 % | +0.65 pp | 17.04 % | +0.38 pp |
| DRAM Throughput (%) | 11.01 % | 11.18 % | +0.17 pp | 11.13 % | +0.12 pp | 10.95 % | -0.06 pp | 10.88 % | -0.13 pp |
| Achieved Occupancy (%) | 57.49 % | 50.61 % | -6.88 pp | 48.49 % | -9.00 pp | 61.35 % | +3.86 pp | 59.18 % | +1.69 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 20.00 register/thread | +2 | 18.00 register/thread | +0 |
| Waves/SM | 3.71 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 | 3.71 | +0.00 |
| No Eligible (%) | 44.86 % | 51.31 % | +6.45 pp | 59.53 % | +14.67 pp | 36.12 % | -8.74 pp | 41.51 % | -3.35 pp |
| IPC Active | 2.26 inst/cycle | 2.03 inst/cycle | -0.23 | 1.67 inst/cycle | -0.59 | 2.64 inst/cycle | +0.38 | 2.32 inst/cycle | +0.06 |

---

## Cumulative Progress

```
Baseline:  7490 ns  (1.000×)
Iter 4     7300 ns  (1.026×)  — remove_broken_autotune_restore_correct_fixed_xblock128_kernel

Final best: 7390 ns  (1.0135×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,35 +5,29 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_53(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 1003520
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = tl.full([XBLOCK], True, tl.int1)
     x2 = xindex
     x0 = (xindex % 512)
-    tmp0 = tl.load(in_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
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
-    tl.store(out_ptr0 + (x2), tmp17, None)
-
-
+    tmp0  = tl.load(in_ptr0 + (x2), None)
+    tmp1  = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')  # mean
+    tmp3  = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')  # var
+    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')  # weight
+    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')  # bias
+    # Precompute per-channel scale: weight / sqrt(var + eps)
+    scale    = tmp12 / tl.sqrt(tmp3 + 1e-5)
+    # FMA #1: bias_adj = bias - mean*scale  (per channel, broadcast)
+    bias_adj = tl.math.fma(-tmp1, scale, tmp14)
+    # FMA #2: per-element normalized+scaled+shifted value
+    tmp_norm = tl.math.fma(tmp0, scale, bias_adj)
+    # ReLU with float zero to avoid int32->float32 conversion overhead
+    tmp17    = tl.maximum(tl.zeros([XBLOCK], tl.float32), tmp_norm)
+    tl.store(out_ptr0 + (x2), tmp17, None, cache_modifier='.cs')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
