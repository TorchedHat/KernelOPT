# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7980000 ns — **Best latency**: 616160 ns (**12.9512×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`use_make_block_ptr_smem_swizzle_to_reduce_shared_pipeline_pressure`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu3/kernel_022/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7980000 ns
- **SM Throughput**: 41.84 % | **Memory**: 57.51 % | **DRAM**: 0.94 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.78 inst/cycle | **No Eligible**: 55.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | enable_tf32_and_expand_autotune_configs | 11.518× | 12.766× | ✓ |  |
| D | enable_tensor_cores_and_expand_autotune_configs | 13.746× | 12.755× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.000× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | enable_tf32_and_e... | Δ | enable_tensor_cor... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 625090 ns | -7354910 ns (-92.2%) | 625630 ns | -7354370 ns (-92.2%) | 7960000 ns | -20000 ns (-0.3%) |
| SM Throughput (%) | 41.84 % | 60.18 % | +18.34 pp | 60.43 % | +18.59 pp | 41.76 % | -0.08 pp |
| Memory Throughput (%) | 57.51 % | 55.88 % | -1.63 pp | 55.15 % | -2.36 pp | 57.40 % | -0.11 pp |
| DRAM Throughput (%) | 0.94 % | 11.87 % | +10.93 pp | 11.95 % | +11.01 pp | 0.95 % | +0.01 pp |
| Achieved Occupancy (%) | 12.50 % | 12.38 % | -0.12 pp | 12.44 % | -0.06 pp | 12.46 % | -0.04 pp |
| Registers/Thread | 255.00 register/thread | 198.00 register/thread | -57 | 198.00 register/thread | -57 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.40 % | 72.42 % | +17.02 pp | 72.44 % | +17.04 pp | 55.39 % | -0.01 pp |
| IPC Active | 1.78 inst/cycle | 1.10 inst/cycle | -0.68 | 1.11 inst/cycle | -0.67 | 1.78 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan A: `enable_tf32_and_expand_autotune_configs` (12.909×)

The actual code diff between the slow and fast kernels in this iteration consists entirely of removing inline comments from the @triton.autotune config list — no functional code was changed. All launch parameters (block_size=128, grid_size=512, registers_per_thread=198, dynamic_smem=98.3 KB, waves_per_sm=1.94) are identical before and after, and all profiling metrics changed by less than 1% (Duration: 618.75 µs → 618.18 µs, Compute SM Throughput: 59.87% → 60.47%). The reported 12.9x speedup reflects the cumulative gain from prior iterations (TF32 enablement, autotune expansion) that were already baked into the 'before' kernel of this iteration, not from the comment removal itself. Removing comments from autotune config lists has zero performance impact; the speedup number here is an artifact of the measurement baseline.

> **Applicable when**: Never — comment removal has no performance impact under any conditions.

> **Avoid when**: Always: removing comments from autotune config lists is a no-op. Do not attribute speedup to this change. The 12.9x speedup in this iteration is inherited from prior TF32 + autotune expansion changes, not from comment removal.

---

## Iteration 2

**Best so far**: 12.7551× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_partial_wave_via_grid_aligned_tile_sizes | 7.246× | 7.748× | ✓ |  |
| B | eliminate_partial_wave_via_group_m_and_register_pressure_reduction | 13.863× | 12.922× | ✓ | **✓ best** |
| C | fix_shared_memory_bank_conflicts_via_smem_padding | 0.987× | 1.047× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_partial... | Δ | eliminate_partial... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 1030000 ns | -6950000 ns (-87.1%) | 617570 ns | -7362430 ns (-92.3%) | 7620000 ns | -360000 ns (-4.5%) |
| SM Throughput (%) | 41.84 % | 35.69 % | -6.15 pp | 60.03 % | +18.19 pp | 43.61 % | +1.77 pp |
| Memory Throughput (%) | 57.51 % | 30.42 % | -27.09 pp | 53.35 % | -4.16 pp | 61.13 % | +3.62 pp |
| DRAM Throughput (%) | 0.94 % | 14.59 % | +13.65 pp | 12.01 % | +11.07 pp | 1.00 % | +0.06 pp |
| Achieved Occupancy (%) | 12.50 % | 6.25 % | -6.25 pp | 12.32 % | -0.18 pp | 12.46 % | -0.04 pp |
| Registers/Thread | 255.00 register/thread | 116.00 register/thread | -139 | 198.00 register/thread | -57 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 7.76 | +5.82 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.40 % | 73.94 % | +18.54 pp | 72.34 % | +16.94 pp | 54.00 % | -1.40 pp |
| IPC Active | 1.78 inst/cycle | 1.04 inst/cycle | -0.74 | 1.10 inst/cycle | -0.68 | 1.84 inst/cycle | +0.06 |

---

## Iteration 3

**Best so far**: 12.9216× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_kernel_scheduling_to_eliminate_partial_wave | — | fail | ✓ |  |
| B | reduce_smem_per_block_to_restore_two_blocks_per_sm | 11.412× | 9.513× | ✓ |  |
| C | reduce_register_pressure_via_smaller_tiles_and_fewer_warps | — | fail | ✓ |  |
| D | add_triton_autotune | 13.883× | 12.897× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_smem_per_b... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 838850 ns | -7141150 ns (-89.5%) | 618750 ns | -7361250 ns (-92.2%) |
| SM Throughput (%) | 41.84 % | 44.02 % | +2.18 pp | 59.87 % | +18.03 pp |
| Memory Throughput (%) | 57.51 % | 85.11 % | +27.60 pp | 55.80 % | -1.71 pp |
| DRAM Throughput (%) | 0.94 % | 17.54 % | +16.60 pp | 12.26 % | +11.32 pp |
| Achieved Occupancy (%) | 12.50 % | 12.37 % | -0.13 pp | 12.38 % | -0.12 pp |
| Registers/Thread | 255.00 register/thread | 95.00 register/thread | -160 | 198.00 register/thread | -57 |
| Waves/SM | 1.94 | 7.76 | +5.82 | 1.94 | +0.00 |
| No Eligible (%) | 55.40 % | 68.84 % | +13.44 pp | 72.33 % | +16.93 pp |
| IPC Active | 1.78 inst/cycle | 1.27 inst/cycle | -0.51 | 1.10 inst/cycle | -0.68 |

---

## Iteration 4

**Best so far**: 12.9216× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fuse_matmul_epilogue_logsumexp_mish_into_single_kernel | 13.885× | 12.909× | ✓ | **✓ best** |
| B | single_pass_online_logsumexp_to_eliminate_double_read | 13.862× | 12.741× | ✓ |  |
| C | fuse_matmul_epilogue_into_single_pass_online_logsumexp_reduction | 11.383× | 9.755× | ✓ |  |
| D | single_pass_online_logsumexp_in_fused_reduction_kernel | 11.367× | 9.396× | ✓ |  |
| E | add_triton_autotune | 13.828× | 12.942× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_matmul_epilo... | Δ | single_pass_onlin... | Δ | fuse_matmul_epilo... | Δ | single_pass_onlin... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 618180 ns | -7361820 ns (-92.3%) | 626340 ns | -7353660 ns (-92.2%) | 818080 ns | -7161920 ns (-89.7%) | 849280 ns | -7130720 ns (-89.4%) | 616580 ns | -7363420 ns (-92.3%) |
| SM Throughput (%) | 41.84 % | 60.47 % | +18.63 pp | 59.78 % | +17.94 pp | 45.22 % | +3.38 pp | 44.64 % | +2.80 pp | 59.85 % | +18.01 pp |
| Memory Throughput (%) | 57.51 % | 55.61 % | -1.90 pp | 56.14 % | -1.37 pp | 86.52 % | +29.01 pp | 84.19 % | +26.68 pp | 56.26 % | -1.25 pp |
| DRAM Throughput (%) | 0.94 % | 11.87 % | +10.93 pp | 11.92 % | +10.98 pp | 17.99 % | +17.05 pp | 17.33 % | +16.39 pp | 12.36 % | +11.42 pp |
| Achieved Occupancy (%) | 12.50 % | 12.41 % | -0.09 pp | 12.35 % | -0.15 pp | 12.30 % | -0.20 pp | 12.37 % | -0.13 pp | 12.44 % | -0.06 pp |
| Registers/Thread | 255.00 register/thread | 198.00 register/thread | -57 | 198.00 register/thread | -57 | 95.00 register/thread | -160 | 95.00 register/thread | -160 | 198.00 register/thread | -57 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 7.76 | +5.82 | 7.76 | +5.82 | 1.94 | +0.00 |
| No Eligible (%) | 55.40 % | 72.40 % | +17.00 pp | 72.33 % | +16.93 pp | 69.40 % | +14.00 pp | 69.12 % | +13.72 pp | 72.47 % | +17.07 pp |
| IPC Active | 1.78 inst/cycle | 1.10 inst/cycle | -0.68 | 1.11 inst/cycle | -0.67 | 1.23 inst/cycle | -0.55 | 1.23 inst/cycle | -0.55 | 1.10 inst/cycle | -0.68 |


### Key Learnings

#### ❌ Plan B: `single_pass_online_logsumexp_to_eliminate_double_read` (12.741×)

Replacing a two-pass logsumexp (separate max and sum-exp loops) with a single-pass online algorithm was expected to halve memory traffic for the reduction kernel, but the profiling data shows essentially no change across all metrics: Duration increased slightly (618 µs → 626 µs), Executed Instructions are identical (115,308,544), DRAM Throughput is unchanged (12.26% → 11.92%), and L2 Hit Rate is nearly identical (81.26% → 82.06%). The diff shows an empty code change, indicating the optimization was either not applied or the reduction kernel's memory traffic is negligible relative to the dominant matmul kernel, making the two-pass vs. one-pass distinction irrelevant to end-to-end latency. This strategy is ineffective when the profiled kernel is the matmul (not the reduction), and the reduction kernel's contribution to total runtime is too small to measure.

> **Applicable when**: The reduction kernel (logsumexp/softmax) is the dominant contributor to end-to-end latency; the buffer being read twice is large enough to cause measurable DRAM traffic (e.g., exceeds L2 cache capacity); profiling shows the reduction kernel specifically has high DRAM throughput and two distinct load phases for the same buffer

> **Avoid when**: The profiled kernel is the matmul (not the reduction); the reduction kernel's runtime is negligible compared to the matmul kernel; the buffer being read twice fits entirely in L2 cache (L2 Hit Rate >80% means the second pass is served from cache, not DRAM); the code diff is empty (change was not actually applied); all key metrics (Duration, Executed Instructions, DRAM Throughput) are identical before and after

---

## Iteration 5

**Best so far**: 12.9216× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | single_pass_online_logsumexp_to_halve_l2_traffic_in_fused_reduction_kernel | 11.412× | 9.347× | ✓ |  |
| C | use_make_block_ptr_smem_swizzle_to_reduce_shared_pipeline_pressure | 15.290× | 12.951× | ✓ | **✓ best** |
| D | replace_autotune_configs_with_smem_under_76kb_to_enable_3_blocks_per_sm | 9.435× | 9.597× | ✓ |  |
| E | add_triton_autotune | 13.881× | 12.612× | ✓ |  |


### Metric Comparison

| Metric | Baseline | single_pass_onlin... | Δ | use_make_block_pt... | Δ | replace_autotune_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 853790 ns | -7126210 ns (-89.3%) | 616160 ns | -7363840 ns (-92.3%) | 831520 ns | -7148480 ns (-89.6%) | 632740 ns | -7347260 ns (-92.1%) |
| SM Throughput (%) | 41.84 % | 44.87 % | +3.03 pp | 60.69 % | +18.85 pp | 45.98 % | +4.14 pp | 59.72 % | +17.88 pp |
| Memory Throughput (%) | 57.51 % | 83.01 % | +25.50 pp | 56.56 % | -0.95 pp | 82.06 % | +24.55 pp | 54.66 % | -2.85 pp |
| DRAM Throughput (%) | 0.94 % | 17.20 % | +16.26 pp | 12.61 % | +11.67 pp | 17.35 % | +16.41 pp | 12.12 % | +11.18 pp |
| Achieved Occupancy (%) | 12.50 % | 12.40 % | -0.10 pp | 12.23 % | -0.27 pp | 23.51 % | +11.01 pp | 12.41 % | -0.09 pp |
| Registers/Thread | 255.00 register/thread | 95.00 register/thread | -160 | 199.00 register/thread | -56 | 64.00 register/thread | -191 | 198.00 register/thread | -57 |
| Waves/SM | 1.94 | 7.76 | +5.82 | 1.94 | +0.00 | 3.88 | +1.94 | 1.94 | +0.00 |
| No Eligible (%) | 55.40 % | 69.60 % | +14.20 pp | 76.15 % | +20.75 pp | 52.44 % | -2.96 pp | 72.42 % | +17.02 pp |
| IPC Active | 1.78 inst/cycle | 1.24 inst/cycle | -0.54 | 0.95 inst/cycle | -0.83 | 1.90 inst/cycle | +0.12 | 1.10 inst/cycle | -0.68 |


### Key Learnings

#### ✅ Plan D: `replace_autotune_configs_with_smem_under_76kb_to_enable_3_blocks_per_sm` (9.597×)

When a Triton matmul kernel is doubly constrained by Block Limit Registers=2 AND Block Limit Shared Mem=2 (Theoretical Occupancy=12.5%), replacing all autotune configs with entries satisfying num_stages×(BLOCK_M×BLOCK_K + BLOCK_K×BLOCK_N)×4 < 76,800 bytes (the H200 threshold for 3 blocks/SM with 228 KB SMEM/SM) breaks the SMEM ceiling: Block Limit Shared Mem rose from 2 to 4, Theoretical Occupancy doubled from 12.5% to 25%, Active Warps Per Scheduler nearly doubled (1.98→3.76), and Eligible Warps Per Scheduler improved from 0.34 to 0.85. However, the smaller tiles reduced arithmetic intensity per CTA, shifting the kernel from compute-bound (Compute SM Throughput 60%→46%) to more memory-bound (Memory Throughput 56%→82%), and the winning config (SMEM=49 KB, registers=64) increased kernel duration from 617 µs to 832 µs despite better occupancy — the net effect was a regression relative to the prior best (12.94x), yielding only 9.60x speedup. The occupancy gain was real but insufficient to compensate for the arithmetic intensity loss from smaller tiles.

> **Applicable when**: Triton matmul where Block Limit Shared Mem is the sole binding occupancy limiter (Block Limit Registers > Block Limit Shared Mem); problem size is large enough that smaller tiles still maintain arithmetic intensity above the memory ridge point; the target occupancy tier (e.g., 3 blocks/SM) can be reached without reducing tile sizes so much that the kernel becomes memory-bandwidth-bound

> **Avoid when**: Block Limit Registers is co-equal or lower than Block Limit Shared Mem (both=2 as in this case) — reducing SMEM alone won't help if registers remain the binding limiter; the problem size or matrix dimensions are such that smaller tiles (e.g., BLOCK_M=64, BLOCK_N=64) reduce arithmetic intensity below the ridge point, making the kernel memory-bound and increasing duration despite better occupancy; the prior best config already achieves high Compute SM Throughput (>55%) with 2 blocks/SM — the occupancy gain from 3 blocks/SM may not compensate for the arithmetic intensity loss

---

## Cumulative Progress

```
Baseline:  7980000 ns  (1.000×)
Iter 1     625630 ns  (12.755×)  — enable_tensor_cores_and_expand_autotune_configs
Iter 2     617570 ns  (12.922×)  — eliminate_partial_wave_via_group_m_and_register_pressure_reduction
Iter 5     616160 ns  (12.951×)  — use_make_block_ptr_smem_swizzle_to_reduce_shared_pipeline_pressure

Final best: 616160 ns  (12.9512×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,36 @@

 import triton.language as tl
 from triton.language.extra import libdevice
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # NEW: larger BLOCK_K, more stages, higher num_warps for H200 Tensor Core saturation
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=16),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=16),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        # num_stages=2: halve SMEM per block (~98KB->~49KB), enabling 3 blocks/SM on H200 (228KB SMEM/SM)
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=2, num_warps=4),
+        # GROUP_M=4/16: align tile grouping with 132-SM wave boundary, reduce workload imbalance
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4},  num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 16}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 4},  num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 16}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 4},  num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 16}, num_stages=3, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -35,19 +57,31 @@

 
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
-
-    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
-    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
+
+    A_block_ptr = tl.make_block_ptr(
+        base=A_ptr,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B_ptr,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        k_remaining = K - k * BLOCK_K
-        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
-        b = tl.load(B_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
-        acc = tl.dot(a, b, acc, allow_tf32=False)
-        A_ptrs += BLOCK_K * stride_ak
-        B_ptrs += BLOCK_K * stride_bk
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(a, b, acc, allow_tf32=True)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
@@ -62,10 +96,16 @@

         triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
         triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
         triton.Config({'BLOCK_N': 256}, num_stages=3, num_warps=4),
+        # NEW: explore BLOCK_N=128 and 512 with varied num_warps per plan
+        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
+        triton.Config({'BLOCK_N': 512}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_N': 1024}, num_stages=3, num_warps=8),
     ],
     key=['N'],
 )
 @triton.jit
+
 def fused_reduction_kernel(
     mm_ptr, bias_ptr, out_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
