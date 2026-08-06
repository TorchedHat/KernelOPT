# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3710 ns — **Best latency**: 3230 ns (**1.1486×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`hide_l1tex_latency_via_wider_xblock_and_load_pipelining`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_30_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1715,1,1), Waves/SM: 0.81
- **Registers/thread**: 18
- **Duration**: 3710 ns
- **SM Throughput**: 19.16 % | **Memory**: 7.33 % | **DRAM**: 4.88 %
- **Achieved Occupancy**: 61.70 %
- **IPC Active**: 2.12 inst/cycle | **No Eligible**: 44.88 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_increase_tile_size | 0.964× | 0.898× | ✓ |  |
| B | autotune_xblock_num_warps_for_better_occupancy_and_cache_reuse | 0.977× | 0.959× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_pointwise_kernel | 0.940× | 1.000× | ✓ |  |
| D | autotune_xblock_to_increase_sm_utilization | 0.820× | 1.025× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_xblock_i... | Δ | autotune_xblock_n... | Δ | autotune_xblock_a... | Δ | autotune_xblock_t... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 4130 ns | +420 ns (+11.3%) | 3870 ns | +160 ns (+4.3%) | 3710 ns | +0 ns (+0.0%) | 3620 ns | -90 ns (-2.4%) |
| SM Throughput (%) | 19.16 % | 13.86 % | -5.30 pp | 34.62 % | +15.46 pp | 19.76 % | +0.60 pp | 26.25 % | +7.09 pp |
| Memory Throughput (%) | 7.33 % | 6.39 % | -0.94 pp | 9.99 % | +2.66 pp | 7.47 % | +0.14 pp | 9.71 % | +2.38 pp |
| DRAM Throughput (%) | 4.88 % | 4.42 % | -0.46 pp | 4.69 % | -0.19 pp | 4.89 % | +0.01 pp | 5.06 % | +0.18 pp |
| Achieved Occupancy (%) | 61.70 % | 34.33 % | -27.37 pp | 58.98 % | -2.72 pp | 60.56 % | -1.14 pp | 59.03 % | -2.67 pp |
| Registers/Thread | 18.00 register/thread | 24.00 register/thread | +6 | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 24.00 register/thread | +6 |
| Waves/SM | 0.81 | 0.41 | -0.40 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 44.88 % | 62.23 % | +17.35 pp | 26.96 % | -17.92 pp | 49.06 % | +4.18 pp | 34.79 % | -10.09 pp |
| IPC Active | 2.12 inst/cycle | 1.39 inst/cycle | -0.73 | 2.52 inst/cycle | +0.40 | 1.98 inst/cycle | -0.14 | 2.49 inst/cycle | +0.37 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.982× | 0.942× | ✓ |  |
| B | persistent_tile_scheduling_with_software_pipelining | 0.999× | 0.982× | ✓ |  |
| C | persistent_tile_scheduling | 0.942× | 0.839× | ✓ |  |
| D | persistent_tile_scheduling | 1.024× | 1.000× | ✓ | **✓ best** |
| E | add_triton_autotune | 0.999× | 0.982× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | persistent_tile_s... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3940 ns | +230 ns (+6.2%) | 3780 ns | +70 ns (+1.9%) | 4420 ns | +710 ns (+19.1%) | 3710 ns | +0 ns (+0.0%) | 3780 ns | +70 ns (+1.9%) |
| SM Throughput (%) | 19.16 % | 15.87 % | -3.29 pp | 18.13 % | -1.03 pp | 18.12 % | -1.04 pp | 13.19 % | -5.97 pp | 19.59 % | +0.43 pp |
| Memory Throughput (%) | 7.33 % | 6.79 % | -0.54 pp | 6.99 % | -0.34 pp | 6.41 % | -0.92 pp | 6.55 % | -0.78 pp | 7.33 % | +0.00 pp |
| DRAM Throughput (%) | 4.88 % | 4.59 % | -0.29 pp | 4.79 % | -0.09 pp | 4.12 % | -0.76 pp | 4.88 % | +0.00 pp | 4.82 % | -0.06 pp |
| Achieved Occupancy (%) | 61.70 % | 42.06 % | -19.64 pp | 58.36 % | -3.34 pp | 66.75 % | +5.05 pp | 29.12 % | -32.58 pp | 62.06 % | +0.36 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 44.88 % | 61.70 % | +16.82 pp | 48.82 % | +3.94 pp | 50.44 % | +5.56 pp | 70.60 % | +25.72 pp | 44.52 % | -0.36 pp |
| IPC Active | 2.12 inst/cycle | 1.49 inst/cycle | -0.63 | 2.02 inst/cycle | -0.10 | 1.95 inst/cycle | -0.17 | 0.99 inst/cycle | -1.13 | 2.09 inst/cycle | -0.03 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.839×)

Converting a standard one-CTA-per-tile Triton kernel to a persistent while-loop pattern increased registers per thread from 24 to 30, which reduced Block Limit Registers from 21 to 16 blocks/SM. Despite slightly higher achieved occupancy (59% → 67%), the while-loop body introduced additional stall cycles: Warp Cycles Per Issued Instruction rose from 14.78 to 21.53 (a 46% increase), and CPIStall on L1TEX operations became the dominant bottleneck (53.2% of stall cycles). The net effect was a 22% increase in kernel duration (3.62us → 4.42us), because the persistent loop's register overhead and L1TEX stalls outweighed any tail-wave elimination benefit for this small, low-occupancy kernel.

> **Applicable when**: Never beneficial in this form; persistent kernel pattern is only effective when grid is reduced to NUM_SMS (e.g., 132) so each SM processes multiple tiles. If grid remains at the original tile count, the while-loop adds overhead with no benefit.

> **Avoid when**: Do not apply persistent while-loop pattern when: (1) the grid size is not reduced to match SM count — the loop body adds register pressure without eliminating tail-wave waste; (2) the kernel is already register-pressure-limited (Block Limit Registers < Block Limit Warps); (3) the kernel has L1TEX stalls as a significant bottleneck, since the loop structure can amplify these stalls; (4) waves_per_sm is close to an integer (e.g., 0.81 ≈ 1 wave), meaning tail-wave waste is minimal.

#### ❌ Plan C: `persistent_tile_scheduling` (0.839×)

Converting a standard one-CTA-per-tile Triton kernel to a persistent while-loop pattern increased registers per thread from 24 to 30, which reduced Block Limit Registers from 21 to 16 blocks/SM. Despite slightly higher achieved occupancy (59% → 67%), the while-loop body introduced additional stall cycles: Warp Cycles Per Issued Instruction rose from 14.78 to 21.53 (a 46% increase), and CPIStall on L1TEX operations became the dominant bottleneck (53.2% of stall cycles). The net effect was a 22% increase in kernel duration (3.62us → 4.42us), because the persistent loop's register overhead and L1TEX stalls outweighed any tail-wave elimination benefit for this small, low-occupancy kernel.

> **Applicable when**: Never beneficial in this form; persistent kernel pattern is only effective when grid is reduced to NUM_SMS (e.g., 132) so each SM processes multiple tiles. If grid remains at the original tile count, the while-loop adds overhead with no benefit.

> **Avoid when**: Do not apply persistent while-loop pattern when: (1) the grid size is not reduced to match SM count — the loop body adds register pressure without eliminating tail-wave waste; (2) the kernel is already register-pressure-limited (Block Limit Registers < Block Limit Warps); (3) the kernel has L1TEX stalls as a significant bottleneck, since the loop structure can amplify these stalls; (4) waves_per_sm is close to an integer (e.g., 0.81 ≈ 1 wave), meaning tail-wave waste is minimal.

#### ❌ Plan D: `persistent_tile_scheduling` (0.839×)

Converting a standard one-CTA-per-tile Triton kernel to a persistent while-loop pattern increased registers per thread from 24 to 30, which reduced Block Limit Registers from 21 to 16 blocks/SM. Despite slightly higher achieved occupancy (59% → 67%), the while-loop body introduced additional stall cycles: Warp Cycles Per Issued Instruction rose from 14.78 to 21.53 (a 46% increase), and CPIStall on L1TEX operations became the dominant bottleneck (53.2% of stall cycles). The net effect was a 22% increase in kernel duration (3.62us → 4.42us), because the persistent loop's register overhead and L1TEX stalls outweighed any tail-wave elimination benefit for this small, low-occupancy kernel.

> **Applicable when**: Never beneficial in this form; persistent kernel pattern is only effective when grid is reduced to NUM_SMS (e.g., 132) so each SM processes multiple tiles. If grid remains at the original tile count, the while-loop adds overhead with no benefit.

> **Avoid when**: Do not apply persistent while-loop pattern when: (1) the grid size is not reduced to match SM count — the loop body adds register pressure without eliminating tail-wave waste; (2) the kernel is already register-pressure-limited (Block Limit Registers < Block Limit Warps); (3) the kernel has L1TEX stalls as a significant bottleneck, since the loop structure can amplify these stalls; (4) waves_per_sm is close to an integer (e.g., 0.81 ≈ 1 wave), meaning tail-wave waste is minimal.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_instruction_fusion_and_remove_broken_persistent_loop | 1.012× | 1.124× | ✓ |  |
| B | fma_fusion_via_fp32_arithmetic_rewrite | 0.989× | 1.000× | ✓ |  |
| C | replace_integer_reciprocal_sqrt_with_rsqrt_for_fma_fusion | 0.958× | 1.025× | ✓ |  |
| D | fuse_non_fused_fp32_instructions_with_fma | 1.001× | 0.992× | ✓ |  |
| E | add_triton_autotune | 1.024× | 0.914× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_instruction_f... | Δ | fma_fusion_via_fp... | Δ | replace_integer_r... | Δ | fuse_non_fused_fp... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3300 ns | -410 ns (-11.1%) | 3710 ns | +0 ns (+0.0%) | 3620 ns | -90 ns (-2.4%) | 3740 ns | +30 ns (+0.8%) | 4060 ns | +350 ns (+9.4%) |
| SM Throughput (%) | 19.16 % | 17.90 % | -1.26 pp | 11.99 % | -7.17 pp | 12.57 % | -6.59 pp | 19.24 % | +0.08 pp | 18.78 % | -0.38 pp |
| Memory Throughput (%) | 7.33 % | 11.97 % | +4.64 pp | 7.71 % | +0.38 pp | 8.39 % | +1.06 pp | 7.38 % | +0.05 pp | 6.62 % | -0.71 pp |
| DRAM Throughput (%) | 4.88 % | 5.49 % | +0.61 pp | 4.90 % | +0.02 pp | 5.03 % | +0.15 pp | 4.87 % | -0.01 pp | 4.48 % | -0.40 pp |
| Achieved Occupancy (%) | 61.70 % | 52.56 % | -9.14 pp | 58.36 % | -3.34 pp | 55.02 % | -6.68 pp | 63.03 % | +1.33 pp | 65.40 % | +3.70 pp |
| Registers/Thread | 18.00 register/thread | 32.00 register/thread | +14 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 |
| Waves/SM | 0.81 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 | 0.81 | +0.00 |
| No Eligible (%) | 44.88 % | 41.58 % | -3.30 pp | 62.61 % | +17.73 pp | 65.32 % | +20.44 pp | 45.04 % | +0.16 pp | 49.61 % | +4.73 pp |
| IPC Active | 2.12 inst/cycle | 1.99 inst/cycle | -0.13 | 1.40 inst/cycle | -0.72 | 1.36 inst/cycle | -0.76 | 2.06 inst/cycle | -0.06 | 1.88 inst/cycle | -0.24 |


### Key Learnings

#### ✅ Plan A: `fma_instruction_fusion_and_remove_broken_persistent_loop` (1.124×)

A non-functional persistent while-loop (grid not reduced to SM count) added branch divergence (Avg. Divergent Branches: 12.99 → 0.0) and suppressed warp eligibility (Eligible Warps Per Scheduler: 0.54 → 2.88), halving achieved occupancy (29.1% → 52.6%) and doubling IPC active (0.99 → 1.99). Replacing the sqrt+int-divide+mul-by-1.0 chain with tl.math.rsqrt and the separate MUL+ADD with tl.fma eliminated the divergent branch overhead and reduced instruction-level stalls, cutting elapsed cycles from 5545 to 4915 (11.3% reduction). The dominant gain came from removing the dead loop's branch divergence rather than arithmetic fusion alone, as the non-fused FP32 instruction count actually increased slightly due to more warps executing more work per unit time.

> **Applicable when**: A Triton kernel contains a while-loop persistent pattern where the launch grid was NOT reduced to SM count (grid >> num_sms), making the loop execute exactly once per CTA. Profiling shows high branch divergence (Avg. Divergent Branches > 5), low eligible warps per scheduler (< 1.0), low achieved occupancy (< 35%), and the FPInstructions NCU rule fires indicating non-fused sqrt+divide+multiply chains in BN-style normalization arithmetic.

> **Avoid when**: The persistent loop is correctly implemented (grid == num_sms) and each CTA genuinely processes multiple tiles — removing it would break the work distribution. Also do not apply if the kernel is already branch-divergence-free and the FPInstructions rule is not triggered.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_size_to_improve_wave_coverage_and_latency_hiding | 1.041× | 0.966× | ✓ |  |
| B | eviction_policy_streaming_vs_reuse_differentiation | 1.002× | 1.045× | ✓ |  |
| C | hide_l1tex_latency_via_wider_xblock_and_load_pipelining | 1.067× | 1.149× | ✓ | **✓ best** |
| D | increase_xblock_for_latency_hiding_and_rsqrt_fusion | 0.988× | 0.951× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_size... | Δ | eviction_policy_s... | Δ | hide_l1tex_latenc... | Δ | increase_xblock_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3840 ns | +130 ns (+3.5%) | 3550 ns | -160 ns (-4.3%) | 3230 ns | -480 ns (-12.9%) | 3900 ns | +190 ns (+5.1%) |
| SM Throughput (%) | 19.16 % | 6.98 % | -12.18 pp | 12.21 % | -6.95 pp | 9.27 % | -9.89 pp | 12.50 % | -6.66 pp |
| Memory Throughput (%) | 7.33 % | 3.93 % | -3.40 pp | 7.68 % | +0.35 pp | 9.31 % | +1.98 pp | 8.34 % | +1.01 pp |
| DRAM Throughput (%) | 4.88 % | 2.37 % | -2.51 pp | 5.09 % | +0.21 pp | 5.62 % | +0.74 pp | 4.66 % | -0.22 pp |
| Achieved Occupancy (%) | 61.70 % | 29.00 % | -32.70 pp | 54.58 % | -7.12 pp | 26.10 % | -35.60 pp | 58.67 % | -3.03 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 32.00 register/thread | +14 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.41 | -0.40 | 0.81 | +0.00 | 0.41 | -0.40 | 0.81 | +0.00 |
| No Eligible (%) | 44.88 % | 76.96 % | +32.08 pp | 62.10 % | +17.22 pp | 68.19 % | +23.31 pp | 63.80 % | +18.92 pp |
| IPC Active | 2.12 inst/cycle | 0.79 inst/cycle | -1.33 | 1.42 inst/cycle | -0.70 | 1.13 inst/cycle | -0.99 | 1.42 inst/cycle | -0.70 |

---

## Iteration 5

**Best so far**: 1.1486× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_small_xblock_configs_to_autotune_search_space | 1.025× | 1.025× | ✓ |  |
| B | autotune_num_stages_for_l1tex_latency_hiding | 0.986× | 1.054× | ✓ |  |
| C | 2d_grid_decomposition_for_l2_slice_balance | 0.713× | 0.644× | ✓ |  |
| D | evict_first_on_streaming_inout_ptr_isolated_no_autotune_change | 0.991× | 1.036× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_small_xblock_... | Δ | autotune_num_stag... | Δ | 2d_grid_decomposi... | Δ | evict_first_on_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3710 ns | 3620 ns | -90 ns (-2.4%) | 3520 ns | -190 ns (-5.1%) | 5760 ns | +2050 ns (+55.3%) | 3580 ns | -130 ns (-3.5%) |
| SM Throughput (%) | 19.16 % | 6.97 % | -12.19 pp | 6.73 % | -12.43 pp | 5.30 % | -13.86 pp | 13.31 % | -5.85 pp |
| Memory Throughput (%) | 7.33 % | 4.07 % | -3.26 pp | 4.09 % | -3.24 pp | 8.34 % | +1.01 pp | 8.30 % | +0.97 pp |
| DRAM Throughput (%) | 4.88 % | 2.52 % | -2.36 pp | 2.59 % | -2.29 pp | 0.09 % | -4.79 pp | 5.08 % | +0.20 pp |
| Achieved Occupancy (%) | 61.70 % | 31.71 % | -29.99 pp | 28.87 % | -32.83 pp | 29.81 % | -31.89 pp | 57.21 % | -4.49 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 0.81 | 0.41 | -0.40 | 0.41 | -0.40 | 0.41 | -0.40 | 0.81 | +0.00 |
| No Eligible (%) | 44.88 % | 76.85 % | +31.97 pp | 75.54 % | +30.66 pp | 87.92 % | +43.04 pp | 62.41 % | +17.53 pp |
| IPC Active | 2.12 inst/cycle | 0.70 inst/cycle | -1.42 | 0.84 inst/cycle | -1.28 | 0.46 inst/cycle | -1.66 | 1.50 inst/cycle | -0.62 |


### Key Learnings

#### ❌ Plan C: `2d_grid_decomposition_for_l2_slice_balance` (0.644×)

Switching from a 1D flat grid with modulo-based channel extraction to a 2D grid (channel × spatial) introduced severe uncoalesced global memory accesses: the NCU UncoalescedGlobalAccess rule fired with 68% excessive sectors (65170 of 96040 total), and Warp Cycles Per Issued Instruction nearly tripled from 13.57 to 36.72. The root cause is that the tensor's channel dimension is stride-1 (innermost), so assigning one channel per program_id(0) and iterating over spatial positions with stride 112 means consecutive threads in a warp access addresses 112 floats apart — a stride-112 pattern that is maximally uncoalesced. Despite eliminating the modulo instruction and achieving near-perfect L2 hit rate (55% → 98%), the uncoalesced access pattern increased CPIStall L1TEX stalls from 4.1 to 25.8 cycles per instruction, causing elapsed cycles to increase from 4806 to 8562 (78% regression).

> **Applicable when**: Never beneficial when the tensor's channel dimension is the innermost (stride-1) dimension. The 2D channel-decomposition pattern only helps when channels are the outermost or a high-stride dimension, so that assigning one channel per CTA results in contiguous spatial accesses.

> **Avoid when**: Do not apply 2D grid decomposition with channel-per-CTA when the tensor layout has channel as the innermost (stride-1) dimension (e.g., NHWC or any layout where channel stride = 1). In such layouts, iterating over spatial positions with a fixed channel index creates stride-C uncoalesced accesses where C is the channel count. The resulting UncoalescedGlobalAccess penalty (68% excessive sectors in this case) far outweighs any benefit from eliminating the modulo instruction or improving L2 slice balance. Also do not apply when L2 WorkloadImbalance is the only signal — verify that the main tensor access pattern will remain coalesced after the decomposition.

---

## Cumulative Progress

```
Baseline:  3710 ns  (1.000×)
Iter 4     3230 ns  (1.149×)  — hide_l1tex_latency_via_wider_xblock_and_load_pipelining

Final best: 3230 ns  (1.1486×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 256},  num_warps=2,  num_stages=2),
+        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 512},  num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
+        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=4),
+        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=3),
+        triton.Config({'XBLOCK': 2048}, num_warps=32, num_stages=4),
+    ],
+    key=['xnumel'],
+    restore_value=['in_out_ptr0'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
     xnumel = 219520
@@ -19,19 +36,11 @@

     tmp12 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
     tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
     tmp2 = tmp0 - tmp1
-    tmp4 = tl.full([1], 1e-05, tl.float32)
-    tmp5 = tmp3 + tmp4
-    tmp6 = tl.sqrt_rn(tmp5)
-    tmp7 = tl.full([1], 1, tl.int32)
-    tmp8 = (tmp7 / tmp6)
-    tmp9 = tl.full([1], 1.0, tl.float32)
-    tmp10 = tmp8 * tmp9
+    tmp5 = tmp3 + 1e-05
+    tmp10 = libdevice.rsqrt(tmp5)
     tmp11 = tmp2 * tmp10
-    tmp13 = tmp11 * tmp12
-    tmp15 = tmp13 + tmp14
+    tmp15 = tmp11 * tmp12 + tmp14
     tl.store(in_out_ptr0 + (x2), tmp15, xmask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -41,7 +50,7 @@

     arg_4 = inputs[4]
     xnumel = 219520
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_hardtanh_30[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 219520)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
