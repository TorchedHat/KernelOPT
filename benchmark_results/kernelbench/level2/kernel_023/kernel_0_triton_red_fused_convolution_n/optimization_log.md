# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused_convolution_native_group_norm_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 160350 ns — **Best latency**: 100800 ns (**1.5908×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_register_spill_with_xblock1_large_r0block_and_hoist_bias`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused_convolution_native_group_norm_0`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_023/kernel_0_triton_red_fused_convolution_n/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 1.11
- **Registers/thread**: 68
- **Duration**: 160350 ns
- **SM Throughput**: 64.56 % | **Memory**: 31.15 % | **DRAM**: 31.15 %
- **Achieved Occupancy**: 31.11 %
- **IPC Active**: 2.71 inst/cycle | **No Eligible**: 31.80 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_and_autotune_tile_sizes | 0.253× | 0.251× | ✓ |  |
| B | autotune_xblock_r0block_to_reduce_register_pressure_and_eliminate_partial_wave | 1.076× | 1.084× | ✓ | **✓ best** |
| C | add_autotune_increase_xblock_for_coalescing_and_occupancy | 0.999× | 1.003× | ✓ |  |
| D | autotune_xblock_r0block_to_reduce_partial_wave_and_improve_coalescing | 0.999× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_a... | Δ | autotune_xblock_r... | Δ | add_autotune_incr... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160350 ns | 638300 ns | +477950 ns (+298.1%) | 147970 ns | -12380 ns (-7.7%) | 159840 ns | -510 ns (-0.3%) | 160000 ns | -350 ns (-0.2%) |
| SM Throughput (%) | 64.56 % | 68.50 % | +3.94 pp | 58.02 % | -6.54 pp | 63.99 % | -0.57 pp | 64.30 % | -0.26 pp |
| Memory Throughput (%) | 31.15 % | 22.09 % | -9.06 pp | 33.76 % | +2.61 pp | 31.23 % | +0.08 pp | 31.25 % | +0.10 pp |
| DRAM Throughput (%) | 31.15 % | 7.83 % | -23.32 pp | 33.76 % | +2.61 pp | 31.23 % | +0.08 pp | 31.25 % | +0.10 pp |
| Achieved Occupancy (%) | 31.11 % | 11.43 % | -19.68 pp | 37.61 % | +6.50 pp | 31.21 % | +0.10 pp | 31.06 % | -0.05 pp |
| Registers/Thread | 68.00 register/thread | 255.00 register/thread | +187 | 32.00 register/thread | -36 | 68.00 register/thread | +0 | 68.00 register/thread | +0 |
| Waves/SM | 1.11 | 1.94 | +0.83 | 0.48 | -0.63 | 1.11 | +0.00 | 1.11 | +0.00 |
| No Eligible (%) | 31.80 % | 28.90 % | -2.90 pp | 37.65 % | +5.85 pp | 31.98 % | +0.18 pp | 32.10 % | +0.30 pp |
| IPC Active | 2.71 inst/cycle | 2.84 inst/cycle | +0.13 | 2.50 inst/cycle | -0.21 | 2.73 inst/cycle | +0.02 | 2.72 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0837× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_and_evict_last_stores_to_hide_l1tex_latency | 1.073× | 1.075× | ✓ |  |
| B | add_num_warps_2_configs_to_reduce_register_pressure_and_fix_store_eviction | 1.001× | 0.387× | ✓ |  |
| C | fix_hardcoded_grid_lambda_and_expand_low_register_autotune_configs | 0.998× | 0.957× | ✓ |  |
| D | replace_spilling_autotune_configs_with_small_r0block_xblock1_configs | 1.080× | 0.387× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | add_num_warps_2_c... | Δ | fix_hardcoded_gri... | Δ | replace_spilling_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160350 ns | 149220 ns | -11130 ns (-6.9%) | 414530 ns | +254180 ns (+158.5%) | 167490 ns | +7140 ns (+4.5%) | 414430 ns | +254080 ns (+158.5%) |
| SM Throughput (%) | 64.56 % | 58.24 % | -6.32 pp | 26.77 % | -37.79 pp | 59.65 % | -4.91 pp | 26.83 % | -37.73 pp |
| Memory Throughput (%) | 31.15 % | 33.50 % | +2.35 pp | 12.05 % | -19.10 pp | 29.88 % | -1.27 pp | 12.06 % | -19.09 pp |
| DRAM Throughput (%) | 31.15 % | 33.50 % | +2.35 pp | 12.05 % | -19.10 pp | 29.88 % | -1.27 pp | 12.06 % | -19.09 pp |
| Achieved Occupancy (%) | 31.11 % | 37.41 % | +6.30 pp | 24.05 % | -7.06 pp | 19.63 % | -11.48 pp | 24.06 % | -7.05 pp |
| Registers/Thread | 68.00 register/thread | 32.00 register/thread | -36 | 26.00 register/thread | -42 | 68.00 register/thread | +0 | 26.00 register/thread | -42 |
| Waves/SM | 1.11 | 0.48 | -0.63 | 0.24 | -0.87 | 0.55 | -0.56 | 0.24 | -0.87 |
| No Eligible (%) | 31.80 % | 37.66 % | +5.86 pp | 73.44 % | +41.64 pp | 36.11 % | +4.31 pp | 73.45 % | +41.65 pp |
| IPC Active | 2.71 inst/cycle | 2.51 inst/cycle | -0.20 | 1.06 inst/cycle | -1.65 | 2.48 inst/cycle | -0.23 | 1.06 inst/cycle | -1.65 |

---

## Iteration 3

**Best so far**: 1.0837× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_warps_1_configs_to_reduce_partial_wave_and_improve_store_coalescing | 1.075× | 1.084× | ✓ |  |
| B | fix_hardcoded_grid_lambda_and_prune_to_register_safe_configs_only | 1.075× | 1.087× | ✓ |  |
| C | increase_num_warps_and_r0block_to_hide_l1tex_latency | 1.205× | 0.388× | ✓ | **✓ best** |
| D | fix_grid_lambda_and_restore_xblock_gt1_configs_with_larger_r0block | 1.080× | 1.094× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_warps_1_c... | Δ | fix_hardcoded_gri... | Δ | increase_num_warp... | Δ | fix_grid_lambda_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160350 ns | 147870 ns | -12480 ns (-7.8%) | 147490 ns | -12860 ns (-8.0%) | 413500 ns | +253150 ns (+157.9%) | 146590 ns | -13760 ns (-8.6%) |
| SM Throughput (%) | 64.56 % | 58.34 % | -6.22 pp | 58.31 % | -6.25 pp | 26.82 % | -37.74 pp | 57.98 % | -6.58 pp |
| Memory Throughput (%) | 31.15 % | 33.82 % | +2.67 pp | 33.87 % | +2.72 pp | 12.09 % | -19.06 pp | 34.11 % | +2.96 pp |
| DRAM Throughput (%) | 31.15 % | 33.82 % | +2.67 pp | 33.87 % | +2.72 pp | 12.09 % | -19.06 pp | 34.11 % | +2.96 pp |
| Achieved Occupancy (%) | 31.11 % | 37.55 % | +6.44 pp | 37.26 % | +6.15 pp | 24.06 % | -7.05 pp | 37.28 % | +6.17 pp |
| Registers/Thread | 68.00 register/thread | 32.00 register/thread | -36 | 32.00 register/thread | -36 | 26.00 register/thread | -42 | 32.00 register/thread | -36 |
| Waves/SM | 1.11 | 0.48 | -0.63 | 0.48 | -0.63 | 0.24 | -0.87 | 0.48 | -0.63 |
| No Eligible (%) | 31.80 % | 37.67 % | +5.87 pp | 37.92 % | +6.12 pp | 73.44 % | +41.64 pp | 37.83 % | +6.03 pp |
| IPC Active | 2.71 inst/cycle | 2.49 inst/cycle | -0.22 | 2.49 inst/cycle | -0.22 | 1.06 inst/cycle | -1.65 | 2.49 inst/cycle | -0.22 |

---

## Iteration 4

**Best so far**: 1.0837× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | hoist_loop_invariant_bias_load_outside_reduction_loop | 1.643× | 1.624× | ✓ | **✓ best** |
| B | increase_num_warps_to_16_for_full_warp_occupancy_and_latency_hiding | 1.077× | 1.083× | ✓ |  |
| C | replace_welford_with_fma_online_variance_to_reduce_nonfused_fp32_instructions | 1.110× | 0.979× | ✓ |  |
| D | reorder_output_store_index_to_improve_coalescing | 1.164× | 0.146× | ✓ |  |


### Metric Comparison

| Metric | Baseline | hoist_loop_invari... | Δ | increase_num_warp... | Δ | replace_welford_w... | Δ | reorder_output_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160350 ns | 98750 ns | -61600 ns (-38.4%) | 148100 ns | -12250 ns (-7.6%) | 163740 ns | +3390 ns (+2.1%) | 1100000 ns | +939650 ns (+586.0%) |
| SM Throughput (%) | 64.56 % | 57.04 % | -7.52 pp | 58.48 % | -6.08 pp | 60.79 % | -3.77 pp | 70.00 % | +5.44 pp |
| Memory Throughput (%) | 31.15 % | 50.61 % | +19.46 pp | 33.77 % | +2.62 pp | 30.53 % | -0.62 pp | 25.35 % | -5.80 pp |
| DRAM Throughput (%) | 31.15 % | 50.61 % | +19.46 pp | 33.77 % | +2.62 pp | 30.53 % | -0.62 pp | 4.54 % | -26.61 pp |
| Achieved Occupancy (%) | 31.11 % | 44.97 % | +13.86 pp | 37.48 % | +6.37 pp | 36.27 % | +5.16 pp | 19.88 % | -11.23 pp |
| Registers/Thread | 68.00 register/thread | 32.00 register/thread | -36 | 32.00 register/thread | -36 | 39.00 register/thread | -29 | 87.00 register/thread | +19 |
| Waves/SM | 1.11 | 0.48 | -0.63 | 0.48 | -0.63 | 0.65 | -0.46 | 0.78 | -0.33 |
| No Eligible (%) | 31.80 % | 38.97 % | +7.17 pp | 37.35 % | +5.55 pp | 39.19 % | +7.39 pp | 27.73 % | -4.07 pp |
| IPC Active | 2.71 inst/cycle | 2.41 inst/cycle | -0.30 | 2.49 inst/cycle | -0.22 | 2.43 inst/cycle | -0.28 | 2.81 inst/cycle | +0.10 |


### Key Learnings

#### ✅ Plan A: `hoist_loop_invariant_bias_load_outside_reduction_loop` (1.689×)

This iteration applied the same bias-hoisting transformation as iteration 4, but additionally replaced the `triton_helpers.welford_reduce` helper call with an inlined Welford update using `tl.math.fma`. The combined change reduced warp cycles per issued instruction from 11.87 to 10.22 (CPIStall from L1TEX dropped from 62.7% to 55.3% of stall time) and increased eligible warps per scheduler from 1.17 to 1.73, improving scheduler efficiency. Waves per SM improved from 0.48 to 0.65, indicating the autotuner selected a config with better SM utilization from the expanded config space. The instruction count was essentially unchanged (43.5M vs 44.3M), confirming the speedup came from reduced memory latency stalls and better scheduling rather than instruction reduction.

> **Applicable when**: Triton Welford reduction kernel where: (1) bias/scalar loads inside the loop depend on a slowly-varying group index (e.g., group_id = r0_index // group_size) with few distinct values; (2) CPIStall from L1TEX is >50% of stall time; (3) FPInstructions rule fires with high non-fused/fused ratio; (4) the welford_reduce helper is used rather than inlined fma operations. Inlining Welford with tl.math.fma further reduces scheduler stalls by enabling fused multiply-add instructions.

> **Avoid when**: Do not inline Welford with tl.math.fma when registers_per_thread is already near the spill threshold (>50), as the additional live values (delta, delta2, weight_next) may push register usage over the limit. Do not apply when the number of distinct group values is large (>32), as preloading all variants consumes too many registers.

---

## Iteration 5

**Best so far**: 1.6238× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_inloop_integer_division_with_range_comparisons_for_group_selection | 1.702× | 1.617× | ✓ |  |
| B | hoist_loop_invariant_bias_load_outside_reduction_loop_v2 | 1.647× | 1.614× | ✓ |  |
| C | hoist_loop_invariant_bias_load_outside_reduction_loop | 1.693× | 1.689× | ✓ |  |
| D | fix_register_spill_with_xblock1_large_r0block_and_hoist_bias | 1.726× | 1.591× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_inloop_in... | Δ | hoist_loop_invari... | Δ | hoist_loop_invari... | Δ | fix_register_spil... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 160350 ns | 99140 ns | -61210 ns (-38.2%) | 99330 ns | -61020 ns (-38.1%) | 94940 ns | -65410 ns (-40.8%) | 100800 ns | -59550 ns (-37.1%) |
| SM Throughput (%) | 64.56 % | 57.33 % | -7.23 pp | 56.48 % | -8.08 pp | 59.52 % | -5.04 pp | 56.75 % | -7.81 pp |
| Memory Throughput (%) | 31.15 % | 50.46 % | +19.31 pp | 50.36 % | +19.21 pp | 52.66 % | +21.51 pp | 49.62 % | +18.47 pp |
| DRAM Throughput (%) | 31.15 % | 50.46 % | +19.31 pp | 50.36 % | +19.21 pp | 52.66 % | +21.51 pp | 49.62 % | +18.47 pp |
| Achieved Occupancy (%) | 31.11 % | 44.71 % | +13.60 pp | 45.08 % | +13.97 pp | 41.70 % | +10.59 pp | 44.93 % | +13.82 pp |
| Registers/Thread | 68.00 register/thread | 32.00 register/thread | -36 | 32.00 register/thread | -36 | 38.00 register/thread | -30 | 32.00 register/thread | -36 |
| Waves/SM | 1.11 | 0.48 | -0.63 | 0.48 | -0.63 | 0.65 | -0.46 | 0.48 | -0.63 |
| No Eligible (%) | 31.80 % | 39.82 % | +8.02 pp | 39.53 % | +7.73 pp | 34.77 % | +2.97 pp | 39.21 % | +7.41 pp |
| IPC Active | 2.71 inst/cycle | 2.44 inst/cycle | -0.27 | 2.40 inst/cycle | -0.31 | 2.61 inst/cycle | -0.10 | 2.43 inst/cycle | -0.28 |


### Key Learnings

#### ✅ Plan A: `replace_inloop_integer_division_with_range_comparisons_for_group_selection` (1.617×)

Replacing `r0_index // group_size` (integer division inside a hot reduction loop) with equivalent range comparisons (`r0_index < boundary_0`, `r0_index < boundary_1`) was predicted to eliminate expensive multi-instruction PTX division sequences. However, the NCU profiling metrics before and after are essentially identical: same instruction count (43.5M), same non-fused FP32 instruction count (15.6M), same duration (~99us), same warp cycles per instruction (~11.87), and same achieved occupancy (~45%). The 1.62x speedup reported is therefore NOT attributable to the kernel-level NCU metrics — the NCU kernel itself did not change measurably, suggesting the speedup came from reduced Python/framework overhead, autotuner cache effects, or measurement noise rather than the arithmetic substitution inside the kernel body.

> **Applicable when**: Triton reduction kernel with integer division by a compile-time-constant group size inside the hot loop body, where the FPInstructions NCU rule fires with a high non-fused:fused ratio (>3:1) and the division result is only used for group membership tests (== 0, == 1, etc.). The group boundaries must be exact integer multiples of the group size.

> **Avoid when**: Do not expect measurable NCU-level improvement when the Triton compiler already optimizes integer division by constants to multiply-shift sequences, or when CPIStall from L1TEX memory latency (~62% of stall time) dominates over arithmetic throughput. If NCU metrics (instruction count, duration, warp cycles) are unchanged before and after, the strategy did not affect the kernel hot path. Also do not apply when group boundaries are not compile-time constants or when the division result is used beyond equality comparisons.

#### ✅ Plan C: `hoist_loop_invariant_bias_load_outside_reduction_loop` (1.689×)

This iteration applied the same bias-hoisting transformation as iteration 4, but additionally replaced the `triton_helpers.welford_reduce` helper call with an inlined Welford update using `tl.math.fma`. The combined change reduced warp cycles per issued instruction from 11.87 to 10.22 (CPIStall from L1TEX dropped from 62.7% to 55.3% of stall time) and increased eligible warps per scheduler from 1.17 to 1.73, improving scheduler efficiency. Waves per SM improved from 0.48 to 0.65, indicating the autotuner selected a config with better SM utilization from the expanded config space. The instruction count was essentially unchanged (43.5M vs 44.3M), confirming the speedup came from reduced memory latency stalls and better scheduling rather than instruction reduction.

> **Applicable when**: Triton Welford reduction kernel where: (1) bias/scalar loads inside the loop depend on a slowly-varying group index (e.g., group_id = r0_index // group_size) with few distinct values; (2) CPIStall from L1TEX is >50% of stall time; (3) FPInstructions rule fires with high non-fused/fused ratio; (4) the welford_reduce helper is used rather than inlined fma operations. Inlining Welford with tl.math.fma further reduces scheduler stalls by enabling fused multiply-add instructions.

> **Avoid when**: Do not inline Welford with tl.math.fma when registers_per_thread is already near the spill threshold (>50), as the additional live values (delta, delta2, weight_next) may push register usage over the limit. Do not apply when the number of distinct group values is large (>32), as preloading all variants consumes too many registers.

---

## Cumulative Progress

```
Baseline:  160350 ns  (1.000×)
Iter 1     147970 ns  (1.084×)  — autotune_xblock_r0block_to_reduce_register_pressure_and_eliminate_partial_wave
Iter 4     98750 ns  (1.624×)  — hoist_loop_invariant_bias_load_outside_reduction_loop

Final best: 100800 ns  (1.5908×)
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
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK':  512}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused_convolution_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 1024
@@ -21,15 +33,22 @@

     tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
     tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
+    # Preload all 3 bias scalars outside the reduction loop
+    # in_ptr1 shape (24,) indexed as (group_id + 3*x0), group_id in {0,1,2}
+    # x0 = xindex % 8, shape [XBLOCK, 1]; bias_k shape [XBLOCK, 1]
+    bias0 = tl.load(in_ptr1 + (0 + 3*x0), mask=xmask, other=0.0)
+    bias1 = tl.load(in_ptr1 + (1 + 3*x0), mask=xmask, other=0.0)
+    bias2 = tl.load(in_ptr1 + (2 + 3*x0), mask=xmask, other=0.0)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_5 = r0_index
-        r0_3 = r0_index // 19800
-        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        r0_3 = r0_index // 19800  # shape [1, R0_BLOCK], needed for tl.where selection only
+        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        # tl.where broadcasts [XBLOCK,1] vs [1,R0_BLOCK] -> [XBLOCK,R0_BLOCK]
+        tmp1 = tl.where(r0_3 == 0, bias0, tl.where(r0_3 == 1, bias1, bias2))
         tmp2 = tmp0 + tmp1
         tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
         tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
@@ -42,10 +61,8 @@

     tmp4 = tmp5[:, None]
     tmp8 = tmp6[:, None]
     tmp9 = tmp7[:, None]
-    tl.store(out_ptr0 + (x4), tmp4, xmask)
-    tl.store(out_ptr1 + (x4), tmp8, xmask)
-
-
+    tl.store(out_ptr0 + (x4), tmp4, mask=xmask)
+    tl.store(out_ptr1 + (x4), tmp8, mask=xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -55,7 +72,7 @@

     xnumel = 1024
     r0_numel = 59400
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused_convolution_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 1024, 59400, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused_convolution_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 1024, 59400)
     return arg_3
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
