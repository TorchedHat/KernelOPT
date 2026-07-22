# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_max_pool2d_with_indices_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 13090 ns — **Best latency**: 12220 ns (**1.0712×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`add_evict_last_cache_hint_to_pooling_window_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_max_pool2d_with_indices_20`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_20_triton_poi_fused_max_pool2d_wi/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (15680,1,1), Waves/SM: 7.42
- **Registers/thread**: 24
- **Duration**: 13090 ns
- **SM Throughput**: 62.49 % | **Memory**: 34.44 % | **DRAM**: 12.52 %
- **Achieved Occupancy**: 74.81 %
- **IPC Active**: 2.90 inst/cycle | **No Eligible**: 27.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_xblock_tile_size_for_occupancy_and_ilp | 0.938× | 0.395× | ✓ |  |
| B | autotune_xblock_for_occupancy_and_l1_reuse | 0.933× | 0.919× | ✓ |  |
| C | autotune_xblock_and_num_warps_for_vectorized_loads | 0.927× | 0.923× | ✓ |  |
| D | autotune_xblock_and_num_warps | 1.047× | 1.065× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_xblock_t... | Δ | autotune_xblock_f... | Δ | autotune_xblock_a... | Δ | autotune_xblock_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13090 ns | 33150 ns | +20060 ns (+153.2%) | 14240 ns | +1150 ns (+8.8%) | 14180 ns | +1090 ns (+8.3%) | 12290 ns | -800 ns (-6.1%) |
| SM Throughput (%) | 62.49 % | 79.54 % | +17.05 pp | 67.85 % | +5.36 pp | 66.85 % | +4.36 pp | 37.85 % | -24.64 pp |
| Memory Throughput (%) | 34.44 % | 28.36 % | -6.08 pp | 30.22 % | -4.22 pp | 30.83 % | -3.61 pp | 17.93 % | -16.51 pp |
| DRAM Throughput (%) | 12.52 % | 4.93 % | -7.59 pp | 11.50 % | -1.02 pp | 11.56 % | -0.96 pp | 6.68 % | -5.84 pp |
| Achieved Occupancy (%) | 74.81 % | 87.96 % | +13.15 pp | 82.01 % | +7.20 pp | 82.73 % | +7.92 pp | 35.89 % | -38.92 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 22.00 register/thread | -2 | 22.00 register/thread | -2 | 22.00 register/thread | -2 |
| Waves/SM | 7.42 | 14.85 | +7.43 | 7.42 | +0.00 | 7.42 | +0.00 | 3.71 | -3.71 |
| No Eligible (%) | 27.60 % | 18.59 % | -9.01 pp | 26.95 % | -0.65 pp | 26.67 % | -0.93 pp | 48.87 % | +21.27 pp |
| IPC Active | 2.90 inst/cycle | 3.25 inst/cycle | +0.35 | 2.89 inst/cycle | -0.01 | 2.92 inst/cycle | +0.02 | 2.03 inst/cycle | -0.87 |

---

## Iteration 2

**Best so far**: 1.0651× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_to_autotune_configs_for_latency_hiding | 1.042× | 1.060× | ✓ |  |
| B | add_small_xblock_configs_to_reduce_cpi_stall | 1.058× | 1.068× | ✓ |  |
| C | add_small_xblock_configs_to_existing_autotune | 1.063× | 1.077× | ✓ | **✓ best** |
| D | reduce_alu_instruction_count_via_small_xblock_configs | 1.048× | 1.079× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_num_stages_to... | Δ | add_small_xblock_... | Δ | add_small_xblock_... | Δ | reduce_alu_instru... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13090 ns | 12350 ns | -740 ns (-5.7%) | 12260 ns | -830 ns (-6.3%) | 12160 ns | -930 ns (-7.1%) | 12130 ns | -960 ns (-7.3%) |
| SM Throughput (%) | 62.49 % | 38.43 % | -24.06 pp | 38.46 % | -24.03 pp | 39.78 % | -22.71 pp | 18.92 % | -43.57 pp |
| Memory Throughput (%) | 34.44 % | 18.44 % | -16.00 pp | 18.15 % | -16.29 pp | 14.75 % | -19.69 pp | 10.91 % | -23.53 pp |
| DRAM Throughput (%) | 12.52 % | 6.64 % | -5.88 pp | 6.69 % | -5.83 pp | 3.42 % | -9.10 pp | 3.43 % | -9.09 pp |
| Achieved Occupancy (%) | 74.81 % | 36.85 % | -37.96 pp | 34.89 % | -39.92 pp | 35.09 % | -39.72 pp | 14.12 % | -60.69 pp |
| Registers/Thread | 24.00 register/thread | 22.00 register/thread | -2 | 22.00 register/thread | -2 | 23.00 register/thread | -1 | 21.00 register/thread | -3 |
| Waves/SM | 7.42 | 3.71 | -3.71 | 3.71 | -3.71 | 3.71 | -3.71 | 3.71 | -3.71 |
| No Eligible (%) | 27.60 % | 47.92 % | +20.32 pp | 48.18 % | +20.58 pp | 47.62 % | +20.02 pp | 72.82 % | +45.22 pp |
| IPC Active | 2.90 inst/cycle | 2.01 inst/cycle | -0.89 | 2.01 inst/cycle | -0.89 | 2.04 inst/cycle | -0.86 | 0.98 inst/cycle | -1.92 |


### Key Learnings

#### ✅ Plan A: `add_num_stages_to_autotune_configs_for_latency_hiding` (1.060×)

When a Triton pointwise kernel is latency-bound due to L1TEX scoreboard stalls (CPIStall rule triggered, ~42% of warp cycles stalled on memory), adding num_stages=2 variants alongside num_stages=1 in the autotune config list allows the autotuner to select software-pipelined configurations that may overlap memory loads with ALU work. The change produced a modest 6% speedup; however, the per-kernel profiling metrics (warp cycles per issued instruction, eligible warps per scheduler, achieved occupancy) showed only marginal improvement, suggesting that for flat pointwise kernels with no explicit inner loop, Triton's software pipelining benefit is limited because the compiler cannot easily reorder independent scattered loads across pipeline stages. This strategy is low-risk since the autotuner will fall back to num_stages=1 if it performs better, but the expected large latency reduction (~15%) was not realized.

> **Applicable when**: Triton pointwise kernel is latency-bound (CPIStall rule triggered, L1TEX stall fraction >35% of warp cycles, eligible warps per scheduler <2.0, achieved occupancy <50%); autotune configs exist but only cover XBLOCK and num_warps without explicit num_stages; kernel has multiple independent memory loads per output element.

> **Avoid when**: Kernel is already bandwidth-bound (DRAM throughput >60%) or compute-bound (SM throughput >60%); kernel has no autotune decorator; num_stages>1 would push register usage above SM limits and reduce occupancy below current levels; kernel has an explicit inner loop where num_stages pipelining is already active.

---

## Iteration 3

**Best so far**: 1.0765× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_hardcoded_grid_and_add_large_xblock_configs_for_full_sm_utilization | 1.060× | 1.079× | ✓ |  |
| B | add_num_warps_1_single_warp_block_configs_to_eliminate_intra_block_scheduling_overhead | 1.044× | 1.065× | ✓ |  |
| C | increase_xblock_and_num_warps_for_ilp_and_latency_hiding | 0.584× | 0.498× | ✓ |  |
| D | fix_hardcoded_grid_to_eliminate_partial_wave_tail | 1.047× | 1.068× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_hardcoded_gri... | Δ | add_num_warps_1_s... | Δ | increase_xblock_a... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13090 ns | 12130 ns | -960 ns (-7.3%) | 12290 ns | -800 ns (-6.1%) | 26270 ns | +13180 ns (+100.7%) | 12260 ns | -830 ns (-6.3%) |
| SM Throughput (%) | 62.49 % | 18.61 % | -43.88 pp | 18.97 % | -43.52 pp | 74.40 % | +11.91 pp | 38.48 % | -24.01 pp |
| Memory Throughput (%) | 34.44 % | 10.65 % | -23.79 pp | 10.66 % | -23.78 pp | 27.73 % | -6.71 pp | 18.17 % | -16.27 pp |
| DRAM Throughput (%) | 12.52 % | 3.43 % | -9.09 pp | 3.39 % | -9.13 pp | 6.23 % | -6.29 pp | 6.69 % | -5.83 pp |
| Achieved Occupancy (%) | 74.81 % | 12.97 % | -61.84 pp | 13.29 % | -61.52 pp | 86.34 % | +11.53 pp | 35.54 % | -39.27 pp |
| Registers/Thread | 24.00 register/thread | 21.00 register/thread | -3 | 21.00 register/thread | -3 | 23.00 register/thread | -1 | 22.00 register/thread | -2 |
| Waves/SM | 7.42 | 3.71 | -3.71 | 3.71 | -3.71 | 14.85 | +7.43 | 3.71 | -3.71 |
| No Eligible (%) | 27.60 % | 73.77 % | +46.17 pp | 73.22 % | +45.62 pp | 26.68 % | -0.92 pp | 48.19 % | +20.59 pp |
| IPC Active | 2.90 inst/cycle | 1.01 inst/cycle | -1.89 | 1.00 inst/cycle | -1.90 | 2.91 inst/cycle | +0.01 | 2.03 inst/cycle | -0.87 |

---

## Iteration 4

**Best so far**: 1.0765× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_evict_last_cache_hints_to_pooling_loads_for_l1_reuse | 1.054× | 1.074× | ✓ |  |
| B | add_cache_eviction_hints_to_reduce_l1tex_stall | 1.064× | 1.046× | ✓ |  |
| C | add_evict_last_cache_hint_to_pooling_window_loads | 1.071× | 1.071× | ✓ | **✓ best** |
| D | replace_large_xblock_configs_with_small_xblock_and_fix_hardcoded_grid | 1.065× | 1.060× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_evict_last_ca... | Δ | add_cache_evictio... | Δ | add_evict_last_ca... | Δ | replace_large_xbl... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13090 ns | 12190 ns | -900 ns (-6.9%) | 12510 ns | -580 ns (-4.4%) | 12220 ns | -870 ns (-6.6%) | 12350 ns | -740 ns (-5.7%) |
| SM Throughput (%) | 62.49 % | 21.68 % | -40.81 pp | 44.92 % | -17.57 pp | 21.62 % | -40.87 pp | 18.91 % | -43.58 pp |
| Memory Throughput (%) | 34.44 % | 10.63 % | -23.81 pp | 18.53 % | -15.91 pp | 10.53 % | -23.91 pp | 10.68 % | -23.76 pp |
| DRAM Throughput (%) | 12.52 % | 3.42 % | -9.10 pp | 6.55 % | -5.97 pp | 3.41 % | -9.11 pp | 3.37 % | -9.15 pp |
| Achieved Occupancy (%) | 74.81 % | 13.84 % | -60.97 pp | 39.42 % | -35.39 pp | 13.94 % | -60.87 pp | 12.47 % | -62.34 pp |
| Registers/Thread | 24.00 register/thread | 21.00 register/thread | -3 | 21.00 register/thread | -3 | 21.00 register/thread | -3 | 21.00 register/thread | -3 |
| Waves/SM | 7.42 | 3.71 | -3.71 | 3.71 | -3.71 | 3.71 | -3.71 | 3.71 | -3.71 |
| No Eligible (%) | 27.60 % | 67.31 % | +39.71 pp | 33.77 % | +6.17 pp | 66.39 % | +38.79 pp | 73.25 % | +45.65 pp |
| IPC Active | 2.90 inst/cycle | 1.24 inst/cycle | -1.66 | 2.57 inst/cycle | -0.33 | 1.24 inst/cycle | -1.66 | 0.97 inst/cycle | -1.93 |


### Key Learnings

#### ✅ Plan A: `add_evict_last_cache_hints_to_pooling_loads_for_l1_reuse` (1.074×)

Adding eviction_policy='evict_last' to all spatially-clustered tl.load calls in a pooling kernel (9 loads covering a 3x3 window) reduced warp cycles per issued instruction from 8.65 to 7.09 (18% reduction) and cut the No Eligible warp fraction from 73.77% to 67.31%, yielding a 7.38% wall-time speedup. The L1/TEX hit rate did not measurably improve (1.98% → 1.80%), suggesting the speedup came from the hardware deprioritizing eviction of recently-loaded lines during the sequential 9-load sequence rather than from cross-block reuse. Notably, executed instructions increased (1.68M → 2.09M), indicating the grid lambda fix (hardcoded cdiv(xnumel,128) → lambda meta: cdiv(xnumel,meta['XBLOCK'])) also took effect, but the net result was still positive because the CPIStall reduction outweighed the instruction increase.

> **Applicable when**: Triton kernel with multiple spatially-clustered tl.load calls per output element (e.g., pooling windows, stencils, convolutions) where CPIStall is triggered with L1TEX scoreboard stalls >40% of warp cycles; L1/TEX hit rate is very low (<5%) but L2 hit rate is moderate-to-high (>70%), indicating data is warm at L2 but evicted from L1 before sequential loads within the same thread can reuse it; kernel is latency-bound rather than bandwidth-bound (DRAM throughput <10%).

> **Avoid when**: Kernel is memory-bandwidth-bound (DRAM throughput >50%) where eviction policy changes could reduce effective cache capacity for other warps; kernel has only 1-2 loads per output element where intra-thread L1 reuse is not possible; L1/TEX hit rate is already high (>30%) indicating L1 retention is not the bottleneck; kernel uses streaming access patterns where evict_first would be more appropriate to free cache lines for subsequent warps.

#### ✅ Plan C: `add_evict_last_cache_hint_to_pooling_window_loads` (1.071×)

Adding eviction_policy='evict_last' to all tl.load calls in a Triton stencil or pooling kernel reduced warp cycles per issued instruction from 8.65 to 6.83 (a 21% reduction) and increased eligible warps per scheduler from 0.39 to 0.53, yielding a 7.1% wall-time speedup. The mechanism is that evict_last instructs L1 to retain loaded cache lines longer, reducing L1TEX scoreboard stall cycles (CPIStall) even when the raw L1 hit rate metric does not visibly increase — the benefit manifests as reduced stall latency rather than a higher hit rate percentage. This strategy is most effective when L2 hit rate is high (>70%) confirming inter-block data reuse exists, but L1 hit rate is very low (<5%), indicating L1 is evicting lines before neighboring blocks can reuse them. Note that the autotuner also shifted configs (adding num_stages=1 explicitly and reordering), which may have contributed to the instruction count increase (1.68M → 2.09M); the net speedup is positive despite more instructions because stall cycles decreased more than instruction count increased.

> **Applicable when**: Triton stencil or pooling kernel where: (1) CPIStall rule is triggered with L1TEX scoreboard stalls >40% of warp cycles; (2) L1/TEX hit rate is very low (<5%); (3) L2 hit rate is high (>70%), confirming inter-block data reuse exists at L2 but not L1; (4) multiple loads per output element access overlapping address ranges across neighboring thread blocks (e.g., 3x3 pooling windows, convolution stencils); (5) kernel is latency-bound rather than bandwidth-bound (DRAM throughput <10%).

> **Avoid when**: Kernel is memory-bandwidth-bound (DRAM throughput >50%) where evict_last may thrash L1 and reduce bandwidth; L2 hit rate is low (<50%), indicating data is not being reused between blocks and evict_last will not help; kernel accesses memory in a purely streaming pattern with no inter-block reuse; kernel already uses eviction_policy hints.

---

## Iteration 5

**Best so far**: 1.0765× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_xblock_tile_size_with_dynamic_grid_for_ilp_and_wave_efficiency | 0.798× | 0.752× | ✓ |  |
| B | add_evict_first_to_output_store_to_prevent_l1_pollution | 1.060× | 1.062× | ✓ |  |
| D | combine_sm_aligned_grid_lambda_with_small_xblock_num_warps_1_configs | 1.057× | 1.065× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_xblock_t... | Δ | add_evict_first_t... | Δ | combine_sm_aligne... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 13090 ns | 17410 ns | +4320 ns (+33.0%) | 12320 ns | -770 ns (-5.9%) | 12290 ns | -800 ns (-6.1%) |
| SM Throughput (%) | 62.49 % | 70.30 % | +7.81 pp | 22.56 % | -39.93 pp | 22.61 % | -39.88 pp |
| Memory Throughput (%) | 34.44 % | 31.95 % | -2.49 pp | 10.70 % | -23.74 pp | 10.42 % | -24.02 pp |
| DRAM Throughput (%) | 12.52 % | 9.41 % | -3.11 pp | 3.38 % | -9.14 pp | 3.39 % | -9.13 pp |
| Achieved Occupancy (%) | 74.81 % | 86.04 % | +11.23 pp | 13.85 % | -60.96 pp | 14.39 % | -60.42 pp |
| Registers/Thread | 24.00 register/thread | 32.00 register/thread | +8 | 22.00 register/thread | -2 | 22.00 register/thread | -2 |
| Waves/SM | 7.42 | 7.42 | +0.00 | 3.71 | -3.71 | 3.71 | -3.71 |
| No Eligible (%) | 27.60 % | 20.25 % | -7.35 pp | 65.52 % | +37.92 pp | 66.16 % | +38.56 pp |
| IPC Active | 2.90 inst/cycle | 3.17 inst/cycle | +0.27 | 1.28 inst/cycle | -1.62 | 1.28 inst/cycle | -1.62 |


### Key Learnings

#### ❌ Plan A: `increase_xblock_tile_size_with_dynamic_grid_for_ilp_and_wave_efficiency` (0.752×)

Replacing the full autotune config list (XBLOCK 32–1024) with only large-tile configs (XBLOCK 256–4096, num_warps 4–16) caused a 25% wall-time regression despite dramatically improving occupancy (13.8% → 86%) and reducing No Eligible % (67.3% → 20.3%). The root cause mirrors exp_004: executed instructions exploded 4.5x (2.09M → 9.38M) because the autotuner selected XBLOCK=128 (block_size=128), processing 4x more elements per block, and the per-thread ALU work for this 9-load conditional max-pool kernel scales linearly with XBLOCK. The ALU pipeline became the bottleneck at 70.3% utilization (up from 21.7%), and warp cycles per issued instruction more than doubled (7.09 → 17.18 cycles). Occupancy improvements and scheduler efficiency gains are completely negated when total instruction count grows proportionally with XBLOCK for kernels where per-thread work scales linearly with tile size.

> **Applicable when**: Never applicable as a standalone strategy for kernels where per-thread instruction count scales linearly with XBLOCK and the kernel is not memory-bandwidth-bound. This is a confirmed anti-pattern for max-pool and stencil kernels with ≥9 conditional loads per output element.

> **Avoid when**: Kernel has many conditional loads per output element (≥5) where per-thread ALU work scales with XBLOCK; kernel is already instruction-count-bound or ALU-bound (compute throughput >50%); DRAM throughput is low (<20%) indicating the kernel is not memory-bandwidth-limited; the current best config already uses small XBLOCK (≤64) with low instruction count. Do NOT remove small XBLOCK configs (32, 64) from the autotune list for this class of kernel — the autotuner needs them to find the low-instruction-count optimum.

#### ✅ Plan B: `add_evict_first_to_output_store_to_prevent_l1_pollution` (1.062×)

Adding eviction_policy='evict_first' to the output tl.store in a Triton pooling/stencil kernel (where input loads already carry evict_last) reduced the No Eligible warp fraction from 67.31% to 65.52% and increased eligible warps per scheduler from 0.52 to 0.59, yielding a 6.25% wall-time speedup. The mechanism is that the streaming store hint (PTX st.global.cs) prevents the written output line from displacing the input cache lines that are pinned by evict_last, reducing L1TEX scoreboard stall cycles (CPIStall). The L2 hit rate also improved slightly (76.31% → 79.4%), consistent with better L1 retention of input data. Notably, instruction count increased slightly (2.09M → 2.16M) due to autotune config reordering, but the stall reduction outweighed this overhead.

> **Applicable when**: Triton stencil or pooling kernel where: (1) input loads already carry eviction_policy='evict_last'; (2) CPIStall is triggered with L1TEX scoreboard stalls >40% of warp cycles; (3) L1/TEX hit rate is very low (<5%) but L2 hit rate is moderate-to-high (>70%); (4) the output store uses the default eviction policy; (5) kernel is latency-bound rather than bandwidth-bound (DRAM throughput <10%). The evict_first store is the complementary hint to evict_last loads.

> **Avoid when**: Input loads do NOT already use evict_last (apply evict_last to loads first, as that is the higher-value change); kernel is bandwidth-bound (DRAM throughput >50%) where streaming stores may bypass L2 write-combining; kernel has multiple output stores where evict_first on all of them may cause excessive cache pressure; L1/TEX hit rate is already high (>20%) indicating L1 retention is not the bottleneck.

---

## Cumulative Progress

```
Baseline:  13090 ns  (1.000×)
Iter 1     12290 ns  (1.065×)  — autotune_xblock_and_num_warps
Iter 2     12160 ns  (1.077×)  — add_small_xblock_configs_to_existing_autotune

Final best: 12220 ns  (1.0712×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,31 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32},   num_warps=1),
+        triton.Config({'XBLOCK': 32},   num_warps=2),
+        triton.Config({'XBLOCK': 64},   num_warps=1),
+        triton.Config({'XBLOCK': 64},   num_warps=2),
+        triton.Config({'XBLOCK': 64},   num_warps=4),
+        triton.Config({'XBLOCK': 128},  num_warps=2),
+        triton.Config({'XBLOCK': 128},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=4),
+        triton.Config({'XBLOCK': 256},  num_warps=8),
+        triton.Config({'XBLOCK': 512},  num_warps=4),
+        triton.Config({'XBLOCK': 512},  num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=8),
+        triton.Config({'XBLOCK': 1024}, num_warps=16),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_max_pool2d_with_indices_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
     xnumel = 2007040
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[:]
-    xmask = tl.full([XBLOCK], True, tl.int1)[:]
+    xmask = xindex < xnumel
     x2 = ((xindex // 7168) % 28)
     x1 = ((xindex // 256) % 28)
     x6 = xindex
@@ -25,57 +44,55 @@

     tmp8 = tmp6 < tmp3
     tmp9 = tmp7 & tmp8
     tmp10 = tmp5 & tmp9
-    tmp11 = tl.load(in_ptr0 + ((-7424) + x6), tmp10, other=float("-inf"))
+    tmp11 = tl.load(in_ptr0 + ((-7424) + x6), xmask & tmp10, other=float("-inf"), eviction_policy='evict_last')
     tmp12 = x1
     tmp13 = tmp12 >= tmp1
     tmp14 = tmp12 < tmp3
     tmp15 = tmp13 & tmp14
     tmp16 = tmp5 & tmp15
-    tmp17 = tl.load(in_ptr0 + ((-7168) + x6), tmp16, other=float("-inf"))
+    tmp17 = tl.load(in_ptr0 + ((-7168) + x6), xmask & tmp16, other=float("-inf"), eviction_policy='evict_last')
     tmp18 = tl.maximum(tmp11, tmp17)
     tmp19 = 1 + x1
     tmp20 = tmp19 >= tmp1
     tmp21 = tmp19 < tmp3
     tmp22 = tmp20 & tmp21
     tmp23 = tmp5 & tmp22
-    tmp24 = tl.load(in_ptr0 + ((-6912) + x6), tmp23, other=float("-inf"))
+    tmp24 = tl.load(in_ptr0 + ((-6912) + x6), xmask & tmp23, other=float("-inf"), eviction_policy='evict_last')
     tmp25 = tl.maximum(tmp18, tmp24)
     tmp26 = x2
     tmp27 = tmp26 >= tmp1
     tmp28 = tmp26 < tmp3
     tmp29 = tmp27 & tmp28
     tmp30 = tmp29 & tmp9
-    tmp31 = tl.load(in_ptr0 + ((-256) + x6), tmp30, other=float("-inf"))
+    tmp31 = tl.load(in_ptr0 + ((-256) + x6), xmask & tmp30, other=float("-inf"), eviction_policy='evict_last')
     tmp32 = tl.maximum(tmp25, tmp31)
     tmp33 = tmp29 & tmp15
-    tmp34 = tl.load(in_ptr0 + (x6), tmp33, other=float("-inf"))
+    tmp34 = tl.load(in_ptr0 + (x6), xmask & tmp33, other=float("-inf"), eviction_policy='evict_last')
     tmp35 = tl.maximum(tmp32, tmp34)
     tmp36 = tmp29 & tmp22
-    tmp37 = tl.load(in_ptr0 + (256 + x6), tmp36, other=float("-inf"))
+    tmp37 = tl.load(in_ptr0 + (256 + x6), xmask & tmp36, other=float("-inf"), eviction_policy='evict_last')
     tmp38 = tl.maximum(tmp35, tmp37)
     tmp39 = 1 + x2
     tmp40 = tmp39 >= tmp1
     tmp41 = tmp39 < tmp3
     tmp42 = tmp40 & tmp41
     tmp43 = tmp42 & tmp9
-    tmp44 = tl.load(in_ptr0 + (6912 + x6), tmp43, other=float("-inf"))
+    tmp44 = tl.load(in_ptr0 + (6912 + x6), xmask & tmp43, other=float("-inf"), eviction_policy='evict_last')
     tmp45 = tl.maximum(tmp38, tmp44)
     tmp46 = tmp42 & tmp15
-    tmp47 = tl.load(in_ptr0 + (7168 + x6), tmp46, other=float("-inf"))
+    tmp47 = tl.load(in_ptr0 + (7168 + x6), xmask & tmp46, other=float("-inf"), eviction_policy='evict_last')
     tmp48 = tl.maximum(tmp45, tmp47)
     tmp49 = tmp42 & tmp22
-    tmp50 = tl.load(in_ptr0 + (7424 + x6), tmp49, other=float("-inf"))
+    tmp50 = tl.load(in_ptr0 + (7424 + x6), xmask & tmp49, other=float("-inf"), eviction_policy='evict_last')
     tmp51 = tl.maximum(tmp48, tmp50)
-    tl.store(out_ptr0 + (x6), tmp51, None)
-
-
+    tl.store(out_ptr0 + (x6), tmp51, xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
     xnumel = 2007040
     grid = (triton.cdiv(xnumel, 128),)
-    triton_poi_fused_max_pool2d_with_indices_20[grid](arg_0, arg_1, 2007040, XBLOCK=128)
+    triton_poi_fused_max_pool2d_with_indices_20[grid](arg_0, arg_1, 2007040)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
