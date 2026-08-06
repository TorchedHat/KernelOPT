# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 17120 ns — **Best latency**: 16930 ns (**1.0112×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_lambda_to_match_autotune_xblock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_relu_30`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_30_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (23520,1,1), Waves/SM: 11.14
- **Registers/thread**: 18
- **Duration**: 17120 ns
- **SM Throughput**: 51.67 % | **Memory**: 22.29 % | **DRAM**: 14.38 %
- **Achieved Occupancy**: 58.92 %
- **IPC Active**: 2.43 inst/cycle | **No Eligible**: 38.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_autotune | 0.483× | 0.371× | ✓ |  |
| B | autotune_xblock_increase_occupancy | 0.997× | 1.000× | ✓ |  |
| C | increase_xblock_and_add_autotune_for_vectorized_loads | 0.879× | 0.698× | ✓ |  |
| D | autotune_xblock_num_warps_num_stages | 1.006× | 0.989× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | autotune_xblock_i... | Δ | increase_xblock_a... | Δ | autotune_xblock_n... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 46180 ns | +29060 ns (+169.7%) | 17120 ns | +0 ns (+0.0%) | 24540 ns | +7420 ns (+43.3%) | 17310 ns | +190 ns (+1.1%) |
| SM Throughput (%) | 51.67 % | 74.92 % | +23.25 pp | 60.39 % | +8.72 pp | 78.68 % | +27.01 pp | 44.06 % | -7.61 pp |
| Memory Throughput (%) | 22.29 % | 26.93 % | +4.64 pp | 22.77 % | +0.48 pp | 22.84 % | +0.55 pp | 20.11 % | -2.18 pp |
| DRAM Throughput (%) | 14.38 % | 5.32 % | -9.06 pp | 14.37 % | -0.01 pp | 10.02 % | -4.36 pp | 14.21 % | -0.17 pp |
| Achieved Occupancy (%) | 58.92 % | 77.19 % | +18.27 pp | 62.01 % | +3.09 pp | 87.13 % | +28.21 pp | 39.34 % | -19.58 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 22.00 register/thread | +4 | 32.00 register/thread | +14 | 24.00 register/thread | +6 |
| Waves/SM | 11.14 | 44.55 | +33.41 | 11.14 | +0.00 | 11.14 | +0.00 | 5.57 | -5.57 |
| No Eligible (%) | 38.96 % | 21.89 % | -17.07 pp | 29.70 % | -9.26 pp | 13.99 % | -24.97 pp | 44.64 % | +5.68 pp |
| IPC Active | 2.43 inst/cycle | 3.14 inst/cycle | +0.71 | 2.76 inst/cycle | +0.33 | 3.42 inst/cycle | +0.99 | 2.18 inst/cycle | -0.25 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_autotune` (0.371×)

Increasing the XBLOCK tile size from 128 to 512 (via autotune) in a pointwise kernel where xnumel is NOT exactly divisible by the new tile size forces the addition of a bounds-check mask (xmask = xindex < xnumel). This mask causes predicated-off threads to still execute instructions, inflating the total executed instruction count by ~4x (6.96M → 27.19M) even though the same amount of work is done. The kernel duration increased from 17.12 µs to 46.18 µs (2.7x slower) despite occupancy improving from 58.9% to 77.2%, because the instruction overhead dominated. Additionally, the launch grid was not updated to use the autotune lambda (grid was hardcoded to cdiv(xnumel, 128)), so the larger XBLOCK configs ran with far too many blocks, each doing redundant masked-off work.

> **Applicable when**: Never apply this pattern as implemented. The autotune + larger XBLOCK strategy is theoretically valid for latency-hiding, but only when: (1) the launch grid is correctly updated to a lambda using meta['XBLOCK'], and (2) either xnumel is exactly divisible by all candidate XBLOCK values (avoiding mask overhead) or the mask overhead is acceptable.

> **Avoid when**: Do not add @triton.autotune with larger XBLOCK configs when the launch grid is not simultaneously updated to use a lambda grid (lambda meta: (cdiv(xnumel, meta['XBLOCK']),)). Do not add bounds-check masks (xmask = xindex < xnumel) when the original kernel used tl.full([XBLOCK], True, tl.int1) — this inflates instruction count by the ratio of new_XBLOCK/old_XBLOCK. Do not apply when the instruction count inflation from masking will dominate any latency-hiding benefit.

#### ❌ Plan C: `increase_xblock_and_add_autotune_for_vectorized_loads` (0.698×)

Adding @triton.autotune with larger XBLOCK configs (512–2048) improved occupancy (58.9%→87.1%) and eligible warps per scheduler (1.37→7.23), but the kernel regressed by 43% (17.12 us→24.54 us). The root cause is that the autotuned kernel ran with 32 registers/thread (up from 18), which halved the block limit from registers (21→16 blocks/SM) and more than doubled total executed instructions (6.96M→15.08M), overwhelming the gains from better occupancy. The grid was also not updated to use the lambda form, so the kernel still launched with XBLOCK=128 grid size (23520 blocks) while the autotuner selected larger XBLOCK configs internally, causing a mismatch between grid and tile size that inflated instruction count.

> **Applicable when**: Never apply this specific pattern (autotune with larger XBLOCK but hardcoded grid) — it is always wrong. The general autotune approach is valid only when the launch grid is also updated to a lambda: `grid = lambda meta: (cdiv(xnumel, meta['XBLOCK']),)`.

> **Avoid when**: Do not add @triton.autotune with variable XBLOCK configs if the launch grid is not simultaneously updated to a lambda that reads meta['XBLOCK']. Also avoid when increasing XBLOCK causes registers/thread to jump significantly (e.g., 18→32), as this reduces block-level occupancy and can negate latency-hiding benefits. For kernels already at 11+ waves/SM, occupancy improvements from larger blocks may not offset the instruction count increase.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_autotune_grid_lambda_to_enable_correct_xblock_tuning | 0.998× | 0.998× | ✓ |  |
| B | fix_lambda_grid_and_add_evict_first_for_streaming_load | 0.998× | 0.989× | ✓ |  |
| C | eliminate_non_fused_fp32_instructions_via_fma_arithmetic_rewrite | 0.870× | 0.699× | ✓ |  |
| D | fix_grid_remove_mask_and_fuse_fp32_instructions | 1.014× | 1.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_autotune_grid... | Δ | fix_lambda_grid_a... | Δ | eliminate_non_fus... | Δ | fix_grid_remove_m... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 17150 ns | +30 ns (+0.2%) | 17310 ns | +190 ns (+1.1%) | 24510 ns | +7390 ns (+43.2%) | 16860 ns | -260 ns (-1.5%) |
| SM Throughput (%) | 51.67 % | 60.37 % | +8.70 pp | 45.75 % | -5.92 pp | 78.82 % | +27.15 pp | 45.01 % | -6.66 pp |
| Memory Throughput (%) | 22.29 % | 22.78 % | +0.49 pp | 20.08 % | -2.21 pp | 22.81 % | +0.52 pp | 22.14 % | -0.15 pp |
| DRAM Throughput (%) | 14.38 % | 14.35 % | -0.03 pp | 14.22 % | -0.16 pp | 10.04 % | -4.34 pp | 14.58 % | +0.20 pp |
| Achieved Occupancy (%) | 58.92 % | 62.18 % | +3.26 pp | 40.04 % | -18.88 pp | 87.07 % | +28.15 pp | 42.07 % | -16.85 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 16.00 register/thread | -2 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 5.57 | -5.57 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 38.96 % | 30.25 % | -8.71 pp | 42.59 % | +3.63 pp | 14.44 % | -24.52 pp | 65.63 % | +26.67 pp |
| IPC Active | 2.43 inst/cycle | 2.78 inst/cycle | +0.35 | 2.28 inst/cycle | -0.15 | 3.43 inst/cycle | +1.00 | 1.35 inst/cycle | -1.08 |

---

## Iteration 3

**Best so far**: 1.0154× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency | 1.014× | 1.017× | ✓ |  |
| B | replace_int32_temporaries_with_fp32_literals_for_fma_fusion | 1.006× | 1.004× | ✓ |  |
| C | replace_int_div_with_fp32_rsqrt_isolated_arithmetic_rewrite | 1.018× | 1.011× | ✓ | **✓ best** |
| D | remove_broken_autotune_restore_fixed_xblock128_maskfree | 1.001× | 0.997× | ✓ |  |
| E | add_triton_autotune | 1.003× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | replace_int32_tem... | Δ | replace_int_div_w... | Δ | remove_broken_aut... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 16830 ns | -290 ns (-1.7%) | 17060 ns | -60 ns (-0.4%) | 16930 ns | -190 ns (-1.1%) | 17180 ns | +60 ns (+0.4%) | 17120 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 51.67 % | 45.61 % | -6.06 pp | 59.00 % | +7.33 pp | 22.75 % | -28.92 pp | 52.41 % | +0.74 pp | 44.96 % | -6.71 pp |
| Memory Throughput (%) | 22.29 % | 22.45 % | +0.16 pp | 22.14 % | -0.15 pp | 20.40 % | -1.89 pp | 22.63 % | +0.34 pp | 22.27 % | -0.02 pp |
| DRAM Throughput (%) | 14.38 % | 14.60 % | +0.22 pp | 14.42 % | +0.04 pp | 14.53 % | +0.15 pp | 14.33 % | -0.05 pp | 14.36 % | -0.02 pp |
| Achieved Occupancy (%) | 58.92 % | 41.61 % | -17.31 pp | 60.44 % | +1.52 pp | 27.78 % | -31.14 pp | 59.58 % | +0.66 pp | 42.76 % | -16.16 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 22.00 register/thread | +4 | 23.00 register/thread | +5 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 11.14 | +0.00 | 5.57 | -5.57 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 38.96 % | 65.91 % | +26.95 pp | 31.16 % | -7.80 pp | 70.49 % | +31.53 pp | 39.13 % | +0.17 pp | 58.49 % | +19.53 pp |
| IPC Active | 2.43 inst/cycle | 1.37 inst/cycle | -1.06 | 2.75 inst/cycle | +0.32 | 1.14 inst/cycle | -1.29 | 2.41 inst/cycle | -0.02 | 1.64 inst/cycle | -0.79 |

---

## Iteration 4

**Best so far**: 1.0154× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_to_match_autotune_xblock | 1.009× | 1.011× | ✓ |  |
| C | replace_sqrt_div_with_rsqrt_to_reduce_nonfused_fp32_instructions | 1.010× | 0.998× | ✓ |  |
| D | fuse_sqrt_reciprocal_to_rsqrt_for_fp_instruction_reduction | 1.016× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | replace_sqrt_div_... | Δ | fuse_sqrt_recipro... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 16930 ns | -190 ns (-1.1%) | 17150 ns | +30 ns (+0.2%) | 16860 ns | -260 ns (-1.5%) |
| SM Throughput (%) | 51.67 % | 22.90 % | -28.77 pp | 44.55 % | -7.12 pp | 45.31 % | -6.36 pp |
| Memory Throughput (%) | 22.29 % | 20.58 % | -1.71 pp | 22.04 % | -0.25 pp | 22.29 % | +0.00 pp |
| DRAM Throughput (%) | 14.38 % | 14.55 % | +0.17 pp | 14.36 % | -0.02 pp | 14.60 % | +0.22 pp |
| Achieved Occupancy (%) | 58.92 % | 27.97 % | -30.95 pp | 41.71 % | -17.21 pp | 41.60 % | -17.32 pp |
| Registers/Thread | 18.00 register/thread | 23.00 register/thread | +5 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 11.14 | 5.57 | -5.57 | 11.14 | +0.00 | 11.14 | +0.00 |
| No Eligible (%) | 38.96 % | 70.76 % | +31.80 pp | 58.33 % | +19.37 pp | 65.81 % | +26.85 pp |
| IPC Active | 2.43 inst/cycle | 1.14 inst/cycle | -1.29 | 1.65 inst/cycle | -0.78 | 1.36 inst/cycle | -1.07 |

---

## Iteration 5

**Best so far**: 1.0154× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_warps_to_hide_l1tex_latency_fixed_xblock | 1.004× | 1.017× | ✓ |  |
| B | fix_hardcoded_grid_to_lambda_enabling_autotune_xblock_selection | 1.018× | 1.002× | ✓ |  |
| C | increase_num_stages_to_pipeline_l1tex_loads | 1.011× | 0.985× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_warp... | Δ | fix_hardcoded_gri... | Δ | increase_num_stag... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 17120 ns | 16830 ns | -290 ns (-1.7%) | 17090 ns | -30 ns (-0.2%) | 17380 ns | +260 ns (+1.5%) |
| SM Throughput (%) | 51.67 % | 45.17 % | -6.50 pp | 22.82 % | -28.85 pp | 44.41 % | -7.26 pp |
| Memory Throughput (%) | 22.29 % | 22.22 % | -0.07 pp | 20.52 % | -1.77 pp | 21.96 % | -0.33 pp |
| DRAM Throughput (%) | 14.38 % | 14.60 % | +0.22 pp | 14.41 % | +0.03 pp | 14.14 % | -0.24 pp |
| Achieved Occupancy (%) | 58.92 % | 41.24 % | -17.68 pp | 27.00 % | -31.92 pp | 41.05 % | -17.87 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 23.00 register/thread | +5 | 18.00 register/thread | +0 |
| Waves/SM | 11.14 | 11.14 | +0.00 | 5.57 | -5.57 | 11.14 | +0.00 |
| No Eligible (%) | 38.96 % | 66.66 % | +27.70 pp | 71.25 % | +32.29 pp | 59.32 % | +20.36 pp |
| IPC Active | 2.43 inst/cycle | 1.35 inst/cycle | -1.08 | 1.15 inst/cycle | -1.28 | 1.65 inst/cycle | -0.78 |

---

## Cumulative Progress

```
Baseline:  17120 ns  (1.000×)
Iter 2     16860 ns  (1.015×)  — fix_grid_remove_mask_and_fuse_fp32_instructions

Final best: 16930 ns  (1.0112×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_relu_30(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 3010560
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    # No xmask: 3010560 % 128 == 0, all threads valid
     x2 = xindex
     x0 = (xindex % 384)
-    tmp0 = tl.load(in_ptr0 + (x2), None)
-    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
-    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
-    tmp12 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
-    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
+    tmp0 = tl.load(in_ptr0 + (x2))
+    tmp1 = tl.load(in_ptr1 + (x0), eviction_policy='evict_last')
+    tmp3 = tl.load(in_ptr2 + (x0), eviction_policy='evict_last')
+    tmp12 = tl.load(in_ptr3 + (x0), eviction_policy='evict_last')
+    tmp14 = tl.load(in_ptr4 + (x0), eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
     tmp4 = tl.full([1], 1e-05, tl.float32)
     tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
     tmp13 = tmp11 * tmp12
     tmp15 = tmp13 + tmp14
-    tmp16 = tl.full([1], 0, tl.int32)
-    tmp17 = tl.maximum(tmp16, tmp15)
-    tl.store(out_ptr0 + (x2), tmp17, None)
-
-
+    tmp17 = tl.maximum(0.0, tmp15)
+    tl.store(out_ptr0 + (x2), tmp17)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
