# Kernel Optimization Log

## Summary

- **Kernel**: `triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 579200 ns — **Best latency**: 530530 ns (**1.0917×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`increase_warps_per_block_for_l1tex_latency_hiding_and_grid_wave_alignment`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_046/kernel_5_triton_red_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 2.22
- **Registers/thread**: 72
- **Duration**: 579200 ns
- **SM Throughput**: 24.53 % | **Memory**: 70.51 % | **DRAM**: 13.25 %
- **Achieved Occupancy**: 40.59 %
- **IPC Active**: 1.20 inst/cycle | **No Eligible**: 69.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_autotune | 1.078× | 0.914× | ✓ |  |
| B | memory_coalescing_via_xblock_autotune | 1.005× | 1.039× | ✓ |  |
| C | software_pipelining_with_autotune_tile_sizes | 1.109× | 1.000× | ✓ | **✓ best** |
| D | autotune_xblock_r0block_tile_sizes | 0.997× | 1.048× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | memory_coalescing... | Δ | software_pipelini... | Δ | autotune_xblock_r... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 579200 ns | 633860 ns | +54660 ns (+9.4%) | 557440 ns | -21760 ns (-3.8%) | 579200 ns | +0 ns (+0.0%) | 552640 ns | -26560 ns (-4.6%) |
| SM Throughput (%) | 24.53 % | 21.98 % | -2.55 pp | 24.41 % | -0.12 pp | 23.63 % | -0.90 pp | 24.64 % | +0.11 pp |
| Memory Throughput (%) | 70.51 % | 63.37 % | -7.14 pp | 70.18 % | -0.33 pp | 67.28 % | -3.23 pp | 70.14 % | -0.37 pp |
| DRAM Throughput (%) | 13.25 % | 11.74 % | -1.51 pp | 13.77 % | +0.52 pp | 13.26 % | +0.01 pp | 13.89 % | +0.64 pp |
| Achieved Occupancy (%) | 40.59 % | 30.73 % | -9.86 pp | 40.48 % | -0.11 pp | 40.72 % | +0.13 pp | 40.75 % | +0.16 pp |
| Registers/Thread | 72.00 register/thread | 96.00 register/thread | +24 | 72.00 register/thread | +0 | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 3.10 | +0.88 | 2.22 | +0.00 | 2.22 | +0.00 | 2.22 | +0.00 |
| No Eligible (%) | 69.96 % | 72.35 % | +2.39 pp | 70.10 % | +0.14 pp | 69.74 % | -0.22 pp | 69.59 % | -0.37 pp |
| IPC Active | 1.20 inst/cycle | 1.10 inst/cycle | -0.10 | 1.20 inst/cycle | +0.00 | 1.21 inst/cycle | +0.01 | 1.21 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan D: `autotune_xblock_r0block_tile_sizes` (1.048×)

When a Triton reduction kernel is hardcoded with XBLOCK=1 and the NCU UncoalescedGlobalAccess rule fires (70%+ excessive sectors), adding @triton.autotune over XBLOCK, R0_BLOCK, num_warps, and num_stages allows the runtime to discover tile configurations that improve L1 reuse and reduce partial-wave waste. In this case, the autotune yielded a modest ~4.8% speedup; the core bottleneck (uncoalesced strided access pattern and L1TEX stalls at 62% of warp cycles) persisted because the memory access indexing itself was not restructured. The strategy is most effective when increasing XBLOCK genuinely improves coalescing or cache reuse, not just when it reduces grid size.

> **Applicable when**: Triton reduction kernel has XBLOCK=1 hardcoded; UncoalescedGlobalAccess rule fires with 50%+ excessive sectors; L1TEX stall accounts for 50%+ of warp cycles; kernel has a partial wave (LaunchConfiguration rule fires with >20% waste); grid size equals xnumel (one block per output row).

> **Avoid when**: The access pattern is inherently strided and XBLOCK increase does not change the stride (speedup will be marginal as seen here); when register pressure is already at the block limit and increasing XBLOCK would spill; when the kernel correctness depends on XBLOCK=1 (e.g., warp-level reductions without proper axis handling).

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_via_max_contiguous_multiple_of_hints | 0.992× | 1.038× | ✓ |  |
| B | vectorized_loads_with_multiple_of_hints_and_pipelining | 1.001× | 1.011× | ✓ |  |
| C | vectorized_loads_via_max_contiguous_multiple_of_hints | 1.107× | 1.005× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 1.123× | 1.123× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 579200 ns | 557760 ns | -21440 ns (-3.7%) | 572700 ns | -6500 ns (-1.1%) | 576220 ns | -2980 ns (-0.5%) | 515680 ns | -63520 ns (-11.0%) |
| SM Throughput (%) | 24.53 % | 23.80 % | -0.73 pp | 24.46 % | -0.07 pp | 23.65 % | -0.88 pp | 29.55 % | +5.02 pp |
| Memory Throughput (%) | 70.51 % | 67.77 % | -2.74 pp | 70.34 % | -0.17 pp | 67.34 % | -3.17 pp | 82.51 % | +12.00 pp |
| DRAM Throughput (%) | 13.25 % | 13.76 % | +0.51 pp | 13.40 % | +0.15 pp | 13.32 % | +0.07 pp | 15.66 % | +2.41 pp |
| Achieved Occupancy (%) | 40.59 % | 40.59 % | +0.00 pp | 40.50 % | -0.09 pp | 40.85 % | +0.26 pp | 47.12 % | +6.53 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 72.00 register/thread | +0 | 72.00 register/thread | +0 | 64.00 register/thread | -8 |
| Waves/SM | 2.22 | 2.22 | +0.00 | 2.22 | +0.00 | 2.22 | +0.00 | 1.94 | -0.28 |
| No Eligible (%) | 69.96 % | 69.64 % | -0.32 pp | 69.96 % | +0.00 pp | 69.77 % | -0.19 pp | 68.76 % | -1.20 pp |
| IPC Active | 1.20 inst/cycle | 1.21 inst/cycle | +0.01 | 1.20 inst/cycle | +0.00 | 1.22 inst/cycle | +0.02 | 1.25 inst/cycle | +0.05 |


### Key Learnings

#### ✅ Plan D: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (1.218×)

When a Triton kernel's register count per thread limits Block Limit Registers below Block Limit Warps, adding maxnreg=N to all triton.Config entries forces the compiler to cap register allocation, allowing more blocks to co-reside on each SM and improving latency hiding. In this iteration, restoring maxnreg=64 across all 19 autotune configs (combined with tl.max_contiguous/tl.multiple_of hints on a contiguous pointer) maintained 64 registers/thread and 50% theoretical occupancy while reducing total executed instructions from 119.6M to 112.9M and duration from 513.8µs to 475.4µs — a 21.8% speedup. The L1/TEX hit rate improved from 78.5% to 81.8% and memory throughput rose from 781 to 798 GB/s, confirming that the pointer alignment hints helped the compiler generate more efficient memory access code alongside the register cap.

> **Applicable when**: TheoreticalOccupancy rule fires and Block Limit Registers < Block Limit Warps; registers_per_thread is high (>48) relative to SM register file capacity; kernel is memory-latency bound (L1TEX stall accounts for 40%+ of warp cycles, No Eligible >60%); a contiguous pointer exists that lacks tl.max_contiguous/tl.multiple_of alignment hints, causing suboptimal vectorization.

> **Avoid when**: Block Limit Registers is already >= Block Limit Warps (register pressure is not the occupancy bottleneck); the kernel is compute-bound rather than latency-bound (high SM Busy, low No Eligible); heavy register spilling to local memory would increase L1TEX stalls more than the occupancy gain reduces them — watch for new local load/store MemoryCacheAccessPattern warnings in NCU after applying.

---

## Iteration 3

**Best so far**: 1.1232× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_for_large_nonreused_tensor | 1.126× | 1.127× | ✓ |  |
| C | eviction_policy_streaming_for_non_reused_l1tex_loads | 1.043× | 1.041× | ✓ |  |
| D | cache_intermediate_in_sram_to_eliminate_second_pass_global_loads | 1.178× | 1.060× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | eviction_policy_s... | Δ | cache_intermediat... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 579200 ns | 513790 ns | -65410 ns (-11.3%) | 556190 ns | -23010 ns (-4.0%) | 546590 ns | -32610 ns (-5.6%) |
| SM Throughput (%) | 24.53 % | 29.47 % | +4.94 pp | 25.22 % | +0.69 pp | 27.20 % | +2.67 pp |
| Memory Throughput (%) | 70.51 % | 82.29 % | +11.78 pp | 72.46 % | +1.95 pp | 77.62 % | +7.11 pp |
| DRAM Throughput (%) | 13.25 % | 15.90 % | +2.65 pp | 13.86 % | +0.61 pp | 13.82 % | +0.57 pp |
| Achieved Occupancy (%) | 40.59 % | 47.57 % | +6.98 pp | 39.05 % | -1.54 pp | 39.71 % | -0.88 pp |
| Registers/Thread | 72.00 register/thread | 64.00 register/thread | -8 | 72.00 register/thread | +0 | 72.00 register/thread | +0 |
| Waves/SM | 2.22 | 1.94 | -0.28 | 2.22 | +0.00 | 2.22 | +0.00 |
| No Eligible (%) | 69.96 % | 68.93 % | -1.03 pp | 71.18 % | +1.22 pp | 68.12 % | -1.84 pp |
| IPC Active | 1.20 inst/cycle | 1.24 inst/cycle | +0.04 | 1.16 inst/cycle | -0.04 | 1.28 inst/cycle | +0.08 |


### Key Learnings

#### ✅ Plan A: `eviction_policy_streaming_for_large_nonreused_tensor` (1.127×)

When a kernel loads a large tensor (far exceeding L1 capacity) with evict_last alongside smaller tensors that genuinely benefit from L1 caching, the large tensor thrashes L1 and evicts the smaller reusable tensors. Switching the large tensor's tl.load calls to eviction_policy='evict_first' (PTX .cs cache streaming operator) prevents L1 pollution, freeing capacity for smaller tensors. In this case, changing a 128MB tensor from evict_last to evict_first reduced L1/TEX Hit Rate from 82.1% to 78.49% (confirming the large tensor was no longer cached) while L2 Hit Rate improved from 64.26% to 67.79% and overall memory throughput rose from 769 to 781 GB/s, yielding a modest ~1.3% duration reduction; the gain was limited because the fundamental strided uncoalesced access pattern (70% excessive sectors) remained the dominant bottleneck.

> **Applicable when**: L1/TEX Cache Throughput >80% and MemoryCacheAccessPattern rule fires; kernel mixes a large tensor (>>L1 capacity, e.g. >1MB) with small tensors (<L2 capacity) in the same reduction loop; the large tensor is accessed with evict_last; CPIStall on L1TEX accounts for 30%+ of warp cycles; the small tensors are genuinely reused across loop iterations.

> **Avoid when**: All tensors are similarly sized and none fit in L1 (eviction policy change has no differential benefit); the kernel is compute-bound rather than memory-latency bound; the large tensor has a coalesced access pattern and L1 prefetching is already effective; the dominant bottleneck is uncoalesced access (70%+ excessive sectors) rather than eviction policy — gains will be marginal as seen here.

---

## Iteration 4

**Best so far**: 1.1232× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_register_spill_to_local_memory_via_maxnreg_increase | 1.127× | 1.120× | ✓ |  |
| B | increase_warps_per_block_for_l1tex_latency_hiding_and_grid_wave_alignment | 1.335× | 1.092× | ✓ | **✓ best** |
| C | restore_maxnreg64_to_raise_occupancy_and_latency_hiding | 1.214× | 1.218× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_registe... | Δ | increase_warps_pe... | Δ | restore_maxnreg64... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 579200 ns | 517310 ns | -61890 ns (-10.7%) | 530530 ns | -48670 ns (-8.4%) | 475360 ns | -103840 ns (-17.9%) |
| SM Throughput (%) | 24.53 % | 29.46 % | +4.93 pp | 25.86 % | +1.33 pp | 30.07 % | +5.54 pp |
| Memory Throughput (%) | 70.51 % | 82.27 % | +11.76 pp | 73.07 % | +2.56 pp | 85.91 % | +15.40 pp |
| DRAM Throughput (%) | 13.25 % | 15.83 % | +2.58 pp | 9.87 % | -3.38 pp | 16.24 % | +2.99 pp |
| Achieved Occupancy (%) | 40.59 % | 47.72 % | +7.13 pp | 36.87 % | -3.72 pp | 47.90 % | +7.31 pp |
| Registers/Thread | 72.00 register/thread | 64.00 register/thread | -8 | 80.00 register/thread | +8 | 64.00 register/thread | -8 |
| Waves/SM | 2.22 | 1.94 | -0.28 | 5.17 | +2.95 | 1.94 | -0.28 |
| No Eligible (%) | 69.96 % | 68.95 % | -1.01 pp | 70.54 % | +0.58 pp | 68.51 % | -1.45 pp |
| IPC Active | 1.20 inst/cycle | 1.24 inst/cycle | +0.04 | 1.18 inst/cycle | -0.02 | 1.26 inst/cycle | +0.06 |

---

## Iteration 5

**Best so far**: 1.1232× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_two_pass_global_reload_via_shared_memory_staging | 1.126× | 1.124× | ✓ |  |
| C | transpose_reduction_index_for_coalesced_in_ptr0_access | 0.780× | 0.792× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_two_pas... | Δ | transpose_reducti... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 579200 ns | 515260 ns | -63940 ns (-11.0%) | 730850 ns | +151650 ns (+26.2%) |
| SM Throughput (%) | 24.53 % | 29.52 % | +4.99 pp | 17.03 % | -7.50 pp |
| Memory Throughput (%) | 70.51 % | 82.43 % | +11.92 pp | 74.94 % | +4.43 pp |
| DRAM Throughput (%) | 13.25 % | 15.89 % | +2.64 pp | 12.35 % | -0.90 pp |
| Achieved Occupancy (%) | 40.59 % | 47.68 % | +7.09 pp | 64.62 % | +24.03 pp |
| Registers/Thread | 72.00 register/thread | 64.00 register/thread | -8 | 40.00 register/thread | -32 |
| Waves/SM | 2.22 | 1.94 | -0.28 | 2.59 | +0.37 |
| No Eligible (%) | 69.96 % | 68.95 % | -1.01 pp | 82.28 % | +12.32 pp |
| IPC Active | 1.20 inst/cycle | 1.24 inst/cycle | +0.04 | 0.71 inst/cycle | -0.49 |


### Key Learnings

#### ❌ Plan C: `transpose_reduction_index_for_coalesced_in_ptr0_access` (0.792×)

Swapping the decomposition of a flat reduction index from (r % fast_dim, r // fast_dim) to (r // slow_dim, r % slow_dim) to make in_ptr0 accesses contiguous caused a severe regression (0.79x). Despite L1/TEX Hit Rate improving from 81.8% to 91.3% and L2 Hit Rate from 60.3% to 80.8%, Warp Cycles Per Issued Instruction more than doubled (24.3 → 58.4 cycles) and No Eligible warps worsened from 68.5% to 82.3%, indicating the new index arithmetic (integer division and modulo by 512 per element) introduced heavy LG queue stall pressure. The reindexing also forced the output store to use a non-contiguous scatter pattern (_out_orig_r0 = dim1 + 32*dim2), creating new uncoalesced global stores (65.6% excessive sectors on stores, per MemoryCacheAccessPattern) that did not exist before, and the autotune config set was simultaneously narrowed (removing maxnreg=64), which changed register count from 64 to 40 and altered occupancy dynamics in ways that confounded the measurement.

> **Applicable when**: Never: this specific pattern caused regression. The general idea of reordering reduction index decomposition to coalesce one tensor's access is only safe when ALL tensor accesses (including output stores) remain coalesced or neutral under the new ordering, and when the index arithmetic cost (div/mod by large non-power-of-2 values) is negligible relative to memory latency savings.

> **Avoid when**: The output store index depends on the reduction index decomposition — reordering will scatter writes and introduce new uncoalesced stores that negate the load coalescing benefit. When the new decomposition requires div/mod by large values (e.g., 512) that generate expensive integer arithmetic per element, causing LG queue stalls (Warp Cycles Per Issued Instruction >2x baseline). When other tensors (in_ptr2 here) that were previously accessed contiguously become strided under the new decomposition. When the autotune config set is simultaneously changed (confounds attribution). When the dominant bottleneck is uncoalesced loads but the output is already contiguous — fixing loads while breaking stores is a net loss.

---

## Cumulative Progress

```
Baseline:  579200 ns  (1.000×)
Iter 2     515680 ns  (1.123×)  — reduce_register_pressure_via_maxnreg_to_improve_occupancy

Final best: 530530 ns  (1.0917×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original num_stages=2 configs (keep all 10):
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=8, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=8, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4, num_stages=2, maxnreg=64),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128},  num_warps=8, num_stages=2, maxnreg=64),
+        # NEW num_stages=3 variants for software pipelining:
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=8, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4, num_stages=3, maxnreg=64),
+        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128},  num_warps=8, num_stages=3, maxnreg=64),
+    ],
+    key=['xnumel', 'r0_numel'],
+)
 @triton.jit
 def triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
     xnumel = 2048
@@ -20,14 +47,16 @@

     _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
+        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
+        _r0_ptr2 = tl.max_contiguous(tl.multiple_of(r0_1, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tmp2 = tl.load(in_ptr2 + _r0_ptr2, mask=r0_mask, eviction_policy='evict_last', other=0.0)
+        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
         tmp3 = tmp1 * tmp2
         tmp4 = tmp0 - tmp3
         tmp6 = tl.sqrt_rn(tmp5)
@@ -41,14 +70,16 @@

     tmp12 = tl.sum(_tmp12, 1)[:, None]
     for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
         r0_index = r0_offset + r0_base
+        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
         r0_mask = r0_index < r0_numel
         roffset = r0_offset
         rindex = r0_index
         r0_1 = r0_index
-        tmp14 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
-        tmp15 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
-        tmp16 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
-        tmp19 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp14 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        tmp15 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
+        _r0_ptr2_2 = tl.max_contiguous(tl.multiple_of(r0_1, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tmp16 = tl.load(in_ptr2 + _r0_ptr2_2, mask=r0_mask, eviction_policy='evict_first', other=0.0)
+        tmp19 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
         tmp17 = tmp15 * tmp16
         tmp18 = tmp14 - tmp17
         tmp20 = tl.sqrt_rn(tmp19)
@@ -58,9 +89,8 @@

         tmp24 = tl.sqrt_rn(tmp12)
         tmp25 = tl.maximum(tmp24, tmp21)
         tmp26 = (tmp23 / tmp25)
-        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp26, r0_mask & xmask)
-
-
+        _out_offset = tl.max_contiguous(tl.multiple_of(r0_1 + 16384*x0, [1, R0_BLOCK]), [1, R0_BLOCK])
+        tl.store(out_ptr1 + _out_offset, tmp26, mask=r0_mask & xmask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -71,7 +101,7 @@

     xnumel = 2048
     r0_numel = 16384
     grid = (triton.cdiv(xnumel, 1),)
-    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384, XBLOCK=1, R0_BLOCK=1024)
+    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384)
     return arg_4
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
