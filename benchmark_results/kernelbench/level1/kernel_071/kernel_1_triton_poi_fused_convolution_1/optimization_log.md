# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 285920 ns — **Best latency**: 316900 ns (**0.9022×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_071/kernel_1_triton_poi_fused_convolution_1/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (126464,1,1), Waves/SM: 59.88
- **Registers/thread**: 25
- **Duration**: 285920 ns
- **SM Throughput**: 19.83 % | **Memory**: 78.80 % | **DRAM**: 72.17 %
- **Achieved Occupancy**: 94.05 %
- **IPC Active**: 0.79 inst/cycle | **No Eligible**: 79.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_autotune_larger_tiles | 1.000× | 0.785× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_autotune_larg... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 285920 ns | 364420 ns | +78500 ns (+27.5%) |
| SM Throughput (%) | 19.83 % | 46.33 % | +26.50 pp |
| Memory Throughput (%) | 78.80 % | 81.00 % | +2.20 pp |
| DRAM Throughput (%) | 72.17 % | 56.43 % | -15.74 pp |
| Achieved Occupancy (%) | 94.05 % | 91.15 % | -2.90 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 |
| Waves/SM | 59.88 | 119.76 | +59.88 |
| No Eligible (%) | 79.74 % | 53.57 % | -26.17 pp |
| IPC Active | 0.79 inst/cycle | 1.82 inst/cycle | +1.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | shared_memory_tiled_transpose_coalescing | 1.356× | 1.236× | ✓ | **✓ best** |
| D | change_eviction_policy_for_transpose_cache_efficiency | 0.938× | 0.711× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_til... | Δ | change_eviction_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 285920 ns | 231360 ns | -54560 ns (-19.1%) | 402180 ns | +116260 ns (+40.7%) |
| SM Throughput (%) | 19.83 % | 38.14 % | +18.31 pp | 42.09 % | +22.26 pp |
| Memory Throughput (%) | 78.80 % | 54.06 % | -24.74 pp | 71.04 % | -7.76 pp |
| DRAM Throughput (%) | 72.17 % | 21.77 % | -50.40 pp | 50.43 % | -21.74 pp |
| Achieved Occupancy (%) | 94.05 % | 94.25 % | +0.20 pp | 92.19 % | -1.86 pp |
| Registers/Thread | 25.00 register/thread | 27.00 register/thread | +2 | 32.00 register/thread | +7 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 119.76 | +59.88 |
| No Eligible (%) | 79.74 % | 64.34 % | -15.40 pp | 56.95 % | -22.79 pp |
| IPC Active | 0.79 inst/cycle | 1.44 inst/cycle | +0.65 | 1.71 inst/cycle | +0.92 |


### Key Learnings

#### ✅ Plan C: `shared_memory_tiled_transpose_coalescing` (1.236×)

For a tensor transpose kernel where the input has stride-1 along one logical dimension (e.g., channel) and the output has stride-1 along another (e.g., spatial), mapping the stride-1 input dimension to the fast-varying tile axis (innermost arange) coalesces loads into single 128-byte transactions per warp. Simultaneously mapping the stride-1 output dimension to the fast-varying store axis coalesces stores. This restructuring raised L1/TEX hit rate from 2.35% to 51.75% and L2 hit rate from 51.65% to 77.21%, reducing total memory throughput from 2.77 TB/s to 1.07 TB/s and cutting kernel duration by ~37% (364 µs → 231 µs). The key insight is that the tile axis assignment — not shared memory — determines coalescing; YBLOCK should index the dimension contiguous in the source, XBLOCK the dimension contiguous in the destination.

> **Applicable when**: Kernel performs a layout transpose where L1/TEX hit rate is near 0%, Mem Busy > 70%, and NCU reports MemoryCacheAccessPattern or UncoalescedGlobalAccess. The input tensor has stride-1 along one logical dimension that is mapped to the slow tile axis (YBLOCK/column), while the fast tile axis (XBLOCK/row) indexes a high-stride dimension. Swapping the axis assignment so the stride-1 input dimension is the fast-varying axis will coalesce loads.

> **Avoid when**: When the access pattern is already coalesced (L1 hit rate > 30%, no UncoalescedGlobalAccess rule). Also avoid when the transpose involves dimensions that cannot be cleanly separated into independent fast/slow axes (e.g., non-power-of-2 tile sizes without proper masking). Note: after this fix, UncoalescedGlobalAccess for stores worsened (2% → 64% excessive sectors), suggesting stores became less coalesced — further tuning of the store axis may be needed.

---

## Iteration 3

**Best so far**: 1.2358× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_contiguity_hints_on_channel_dim | 0.828× | 0.507× | ✓ |  |
| B | increase_num_stages_to_pipeline_memory_latency | 1.359× | 1.241× | ✓ | **✓ best** |
| D | coalesced_square_tile_register_transpose | 0.818× | 0.529× | ✓ |  |
| E | add_triton_autotune | 0.528× | 0.522× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | increase_num_stag... | Δ | coalesced_square_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285920 ns | 563680 ns | +277760 ns (+97.1%) | 230460 ns | -55460 ns (-19.4%) | 540190 ns | +254270 ns (+88.9%) | 547650 ns | +261730 ns (+91.5%) |
| SM Throughput (%) | 19.83 % | 11.59 % | -8.24 pp | 38.44 % | +18.61 pp | 51.60 % | +31.77 pp | 26.95 % | +7.12 pp |
| Memory Throughput (%) | 78.80 % | 73.79 % | -5.01 pp | 54.21 % | -24.59 pp | 79.47 % | +0.67 pp | 68.54 % | -10.26 pp |
| DRAM Throughput (%) | 72.17 % | 10.32 % | -61.85 pp | 21.87 % | -50.30 pp | 38.65 % | -33.52 pp | 37.51 % | -34.66 pp |
| Achieved Occupancy (%) | 94.05 % | 88.45 % | -5.60 pp | 94.23 % | +0.18 pp | 78.14 % | -15.91 pp | 72.64 % | -21.41 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 27.00 register/thread | +2 | 32.00 register/thread | +7 | 39.00 register/thread | +14 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 73.70 | +13.82 | 79.84 | +19.96 |
| No Eligible (%) | 79.74 % | 88.30 % | +8.56 pp | 64.46 % | -15.28 pp | 49.82 % | -29.92 pp | 75.37 % | -4.37 pp |
| IPC Active | 0.79 inst/cycle | 0.47 inst/cycle | -0.32 | 1.42 inst/cycle | +0.63 | 2.00 inst/cycle | +1.21 | 0.98 inst/cycle | +0.19 |


### Key Learnings

#### ❌ Plan A: `vectorized_loads_contiguity_hints_on_channel_dim` (0.507×)

Adding tl.max_contiguous and tl.multiple_of hints (with N=1) to the channel and spatial index tensors in a Triton transpose kernel caused a 2x slowdown despite the theoretical motivation of enabling vectorized loads. The hints triggered MIO queue stalls (CPIStall: 42 cycles/warp on MIO, 39.8 cycles/warp on LG queue) that did not exist before, driving warp cycles per issued instruction from 41.46 to 122.42 and eligible warps per scheduler from 2.71 to 0.78. Although L1 and L2 hit rates improved (51.75%→79.95% and 77.21%→88.11%), the MIO pipeline saturation and LG queue pressure caused the kernel to run 2.4x longer. The conservative N=1 hints appear to have changed the compiler's instruction selection in a way that serialized memory operations rather than vectorizing them, adding overhead without the intended benefit.

> **Applicable when**: Never — tl.max_contiguous/tl.multiple_of with N=1 is a no-op semantically but causes compiler behavior changes that introduce MIO and LG queue stalls in transpose kernels.

> **Avoid when**: Do not apply tl.max_contiguous or tl.multiple_of hints with N=1 to index tensors in transpose kernels. These conservative hints do not enable vectorization (N=1 provides no alignment/contiguity guarantee beyond what the compiler already knows) but appear to alter the compiler's instruction scheduling in ways that saturate the MIO pipeline and LG memory queue. The diagnostic signal is: after adding hints, Warp Cycles Per Issued Instruction increases dramatically (>3x), No Eligible warp % rises above 85%, and CPIStall rules appear for MIO queue and LG queue stalls. Also do not apply when the kernel already has good L1 hit rate (>50%) and the bottleneck is scheduler stalls rather than cache misses — adding hints will not help and may worsen MIO pressure. For meaningful vectorization, hints require N≥4 (for float4/128-bit loads) and the index must genuinely be a multiple of N at tile boundaries.

#### ✅ Plan B: `increase_num_stages_to_pipeline_memory_latency` (1.241×)

Adding num_stages=3 and num_stages=4 configs to a Triton autotune decorator, alongside a corrected lambda-based grid, yielded a modest 1.24x speedup on a memory-latency-bound kernel (64% no-eligible-warp cycles, IssueSlotUtilization flagged). However, the profiling metrics before and after are nearly identical — duration dropped only 0.9 µs (231.36 → 230.46 µs), and no scheduler or warp-state metrics meaningfully improved. The speedup likely reflects autotuning selecting a marginally better tile config rather than deeper software pipelining actually hiding HBM latency, suggesting that when uncoalesced stores (64% excessive sectors) dominate the bottleneck, increasing num_stages provides minimal benefit.

> **Applicable when**: Triton kernel with IssueSlotUtilization flagged, >60% no-eligible-warp cycles, all autotune configs at num_stages=2, and the bottleneck is memory latency (not uncoalesced access). Most effective when DRAM throughput is high (>50%) and L2 hit rate is low, indicating the software pipeline is not deep enough to hide HBM round-trip latency.

> **Avoid when**: When the dominant bottleneck is uncoalesced global memory access (UncoalescedGlobalAccess rule with >40% excessive sectors) rather than pure latency hiding. In that case, increasing num_stages cannot compensate for the excess sector traffic — coalescing must be fixed first. Also avoid when the kernel already has high L1/L2 hit rates and the bottleneck is compute-bound.

#### ❌ Plan D: `coalesced_square_tile_register_transpose` (0.529×)

Replacing a working axis-swapped 2D tile transpose (1.24x speedup) with a square-tile variant that decomposes both nc and hw via integer division (n=nc//128, c=nc%128, h=hw//250, w=hw%250) caused a 2x regression. The instruction count exploded from 64M to 211M (3.3x), and dynamic shared memory per block quadrupled (4.1 KB → 16.4 KB), reducing theoretical occupancy from 100% to 81.25% and achieved occupancy from 94.25% to 78.14%. Despite L1 hit rate remaining similar (~52%), the massive instruction overhead from the extra integer divisions and the occupancy drop caused duration to increase from 231 µs to 540 µs. The autotune config selection also shifted to larger tiles (XBLOCK=128/YBLOCK=128), further amplifying the per-CTA work and shared memory pressure.

> **Applicable when**: Never — adding extra hw decomposition (h=hw//250, w=hw%250) to an already-working coalesced transpose tile is harmful. The extra integer divisions triple instruction count without improving coalescing.

> **Avoid when**: Do not add extra integer division decomposition (h=hw//250, w=hw%250) to a transpose kernel that already achieves coalesced loads and stores via axis swapping. When the existing kernel already has L1 hit rate ~52% and L2 hit rate ~77%, the access pattern is already well-coalesced; adding arithmetic to decompose the flat hw index into (h,w) components only inflates instruction count and shared memory usage, reducing occupancy and increasing duration. The diagnostic signal for this anti-pattern is: instruction count increases >2x while L1/L2 hit rates remain similar.

#### ❌ Plan E: `add_triton_autotune` (0.129×)

Replacing a balanced autotune config set (XBLOCK 16–128, YBLOCK 16–128, 27 regs/thread, 94% occupancy) with large-XBLOCK-only configs (XBLOCK=128–1024, YBLOCK=4–32) combined with tl.max_contiguous/tl.multiple_of hints (N=1) caused a catastrophic 7.8x regression. Registers per thread exploded from 27 to 80, collapsing theoretical occupancy from 100% to 37.5% and achieved occupancy from 94.23% to 35.88%. Eligible warps per scheduler dropped from 2.7 to 0.36, and the CPIStall rule appeared (LG queue full, 44.6% of stall cycles), indicating the MIO pipeline was saturated. Instruction count tripled (64M → 210M) and kernel duration increased from 230 µs to 2.22 ms — nearly 10x longer. The combination of large tile sizes (driving register pressure) and N=1 contiguity hints (altering compiler instruction selection) is doubly harmful: the hints do not enable vectorization at N=1 but do change scheduling, while the large tiles prevent the occupancy needed to hide the resulting stalls.

> **Applicable when**: Never — this combination of large-XBLOCK-only autotune configs (≥256) with reduced YBLOCK (≤16) and N=1 contiguity hints is harmful for 2D transpose kernels that already achieve near-100% occupancy.

> **Avoid when**: Do not replace a balanced 2D autotune config set with large-XBLOCK-only configs when the kernel already achieves near-100% theoretical occupancy at 27 registers/thread. Increasing XBLOCK to 512–1024 drives register usage to 80/thread, which limits Block Limit Registers to 6 blocks/SM and collapses theoretical occupancy to 37.5%. The resulting occupancy drop causes eligible warps/scheduler to fall below 0.5, triggering LG queue saturation (CPIStall). Additionally, do not combine large tile configs with N=1 tl.max_contiguous hints — N=1 provides no vectorization benefit but alters compiler instruction scheduling, compounding the register pressure problem. The autotune key should include both 'ynumel' and 'xnumel' for 2D-tiled kernels, not just 'xnumel'.

---

## Iteration 4

**Best so far**: 1.2406× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_store_hints_on_hw_1d_index | 1.361× | 1.241× | ✓ | **✓ best** |
| B | shared_memory_staged_transpose_with_tl_trans | 0.818× | 0.852× | ✓ |  |
| C | register_tile_transpose_with_tl_trans | 1.194× | 0.913× | ✓ |  |
| D | remove_harmful_n1_contiguity_hints_to_unblock_mio_pipeline | 1.359× | 1.242× | ✓ |  |
| E | add_triton_autotune | 0.695× | 0.129× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_store_... | Δ | shared_memory_sta... | Δ | register_tile_tra... | Δ | remove_harmful_n1... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285920 ns | 230340 ns | -55580 ns (-19.4%) | 335650 ns | +49730 ns (+17.4%) | 313120 ns | +27200 ns (+9.5%) | 230270 ns | -55650 ns (-19.5%) | 2220000 ns | +1934080 ns (+676.4%) |
| SM Throughput (%) | 19.83 % | 38.07 % | +18.24 pp | 32.40 % | +12.57 pp | 32.09 % | +12.26 pp | 38.31 % | +18.48 pp | 13.63 % | -6.20 pp |
| Memory Throughput (%) | 78.80 % | 54.29 % | -24.51 pp | 77.09 % | -1.71 pp | 78.53 % | -0.27 pp | 54.46 % | -24.34 pp | 74.51 % | -4.29 pp |
| DRAM Throughput (%) | 72.17 % | 21.88 % | -50.29 pp | 32.91 % | -39.26 pp | 34.76 % | -37.41 pp | 21.89 % | -50.28 pp | 9.68 % | -62.49 pp |
| Achieved Occupancy (%) | 94.05 % | 94.23 % | +0.18 pp | 93.64 % | -0.41 pp | 93.83 % | -0.22 pp | 94.22 % | +0.17 pp | 35.88 % | -58.17 pp |
| Registers/Thread | 25.00 register/thread | 27.00 register/thread | +2 | 27.00 register/thread | +2 | 29.00 register/thread | +4 | 27.00 register/thread | +2 | 80.00 register/thread | +55 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 159.68 | +99.80 |
| No Eligible (%) | 79.74 % | 64.18 % | -15.56 pp | 70.52 % | -9.22 pp | 70.66 % | -9.08 pp | 64.43 % | -15.31 pp | 88.04 % | +8.30 pp |
| IPC Active | 0.79 inst/cycle | 1.43 inst/cycle | +0.64 | 1.17 inst/cycle | +0.38 | 1.18 inst/cycle | +0.39 | 1.42 inst/cycle | +0.63 | 0.48 inst/cycle | -0.31 |


### Key Learnings

#### ✅ Plan A: `vectorized_store_hints_on_hw_1d_index` (1.241×)

Applying tl.max_contiguous(tl.multiple_of(hw_start + hw_idx, XBLOCK), XBLOCK) to the 1D hw index vector before it is broadcast into a 2D store pointer yielded a 1.24x speedup, but the NCU profiling metrics before and after are nearly identical — duration changed by only 0.12 µs (230.46 → 230.34 µs), and no memory or scheduler metrics meaningfully improved. The speedup is almost certainly attributable to the simultaneous autotune config set change (removing num_stages=3/4 configs and adding larger YBLOCK configs like 128/256) rather than the contiguity hints themselves, since the UncoalescedGlobalAccess rule (64% excessive sectors, 7.1 bytes/sector) remained completely unchanged after the optimization. The hints did not reduce excessive sectors, did not improve L1/L2 hit rates, and did not change instruction count, confirming the Triton compiler did not emit vectorized stores from these hints on a 2D broadcast pattern.

> **Applicable when**: Triton transpose kernel with UncoalescedGlobalAccess (>60% excessive sectors) on stores, where the store's fast axis (XBLOCK) indexes a stride-1 output dimension and hw_start = pid_hw * XBLOCK is always a multiple of XBLOCK. The 1D hw index is genuinely contiguous and XBLOCK-aligned before broadcast.

> **Avoid when**: When the dominant bottleneck is uncoalesced stores in a 2D broadcast pattern (hw[None,:] in a 2D tile) — the Triton compiler does not propagate contiguity hints through the [None,:] broadcast into the 2D composite pointer, so the hints do not reduce excessive sectors. The NCU UncoalescedGlobalAccess rule remaining unchanged (64% excessive sectors before and after) confirms this. Do not rely on this approach to fix store coalescing in 2D tiled kernels. Also do not conflate speedup from autotune config changes with speedup from the hint itself.

#### ❌ Plan E: `add_triton_autotune` (0.129×)

Replacing a balanced autotune config set (XBLOCK 16–128, YBLOCK 16–128, 27 regs/thread, 94% occupancy) with large-XBLOCK-only configs (XBLOCK=128–1024, YBLOCK=4–32) combined with tl.max_contiguous/tl.multiple_of hints (N=1) caused a catastrophic 7.8x regression. Registers per thread exploded from 27 to 80, collapsing theoretical occupancy from 100% to 37.5% and achieved occupancy from 94.23% to 35.88%. Eligible warps per scheduler dropped from 2.7 to 0.36, and the CPIStall rule appeared (LG queue full, 44.6% of stall cycles), indicating the MIO pipeline was saturated. Instruction count tripled (64M → 210M) and kernel duration increased from 230 µs to 2.22 ms — nearly 10x longer. The combination of large tile sizes (driving register pressure) and N=1 contiguity hints (altering compiler instruction selection) is doubly harmful: the hints do not enable vectorization at N=1 but do change scheduling, while the large tiles prevent the occupancy needed to hide the resulting stalls.

> **Applicable when**: Never — this combination of large-XBLOCK-only autotune configs (≥256) with reduced YBLOCK (≤16) and N=1 contiguity hints is harmful for 2D transpose kernels that already achieve near-100% occupancy.

> **Avoid when**: Do not replace a balanced 2D autotune config set with large-XBLOCK-only configs when the kernel already achieves near-100% theoretical occupancy at 27 registers/thread. Increasing XBLOCK to 512–1024 drives register usage to 80/thread, which limits Block Limit Registers to 6 blocks/SM and collapses theoretical occupancy to 37.5%. The resulting occupancy drop causes eligible warps/scheduler to fall below 0.5, triggering LG queue saturation (CPIStall). Additionally, do not combine large tile configs with N=1 tl.max_contiguous hints — N=1 provides no vectorization benefit but alters compiler instruction scheduling, compounding the register pressure problem. The autotune key should include both 'ynumel' and 'xnumel' for 2D-tiled kernels, not just 'xnumel'.

---

## Iteration 5

**Best so far**: 1.2413× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | flatten_1d_store_to_eliminate_broadcast_uncoalesced_sectors | 1.381× | 1.455× | ✓ |  |
| B | reduce_index_arithmetic_via_precomputed_strides | 1.810× | 0.902× | ✓ | **✓ best** |
| C | vectorized_loads_channel_hint_with_proper_alignment_N_equals_YBLOCK | 1.191× | 0.912× | ✓ |  |
| D | fix_hardcoded_grid_to_use_autotune_meta_tile_sizes | 1.191× | 0.910× | ✓ |  |


### Metric Comparison

| Metric | Baseline | flatten_1d_store_... | Δ | reduce_index_arit... | Δ | vectorized_loads_... | Δ | fix_hardcoded_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 285920 ns | 196540 ns | -89380 ns (-31.3%) | 316900 ns | +30980 ns (+10.8%) | 313440 ns | +27520 ns (+9.6%) | 314080 ns | +28160 ns (+9.8%) |
| SM Throughput (%) | 19.83 % | 57.10 % | +37.27 pp | 41.53 % | +21.70 pp | 32.37 % | +12.54 pp | 32.30 % | +12.47 pp |
| Memory Throughput (%) | 78.80 % | 54.26 % | -24.54 pp | 77.20 % | -1.60 pp | 78.56 % | -0.24 pp | 78.38 % | -0.42 pp |
| DRAM Throughput (%) | 72.17 % | 54.26 % | -17.91 pp | 34.43 % | -37.74 pp | 34.72 % | -37.45 pp | 34.67 % | -37.50 pp |
| Achieved Occupancy (%) | 94.05 % | 92.62 % | -1.43 pp | 93.73 % | -0.32 pp | 93.82 % | -0.23 pp | 93.77 % | -0.28 pp |
| Registers/Thread | 25.00 register/thread | 32.00 register/thread | +7 | 32.00 register/thread | +7 | 29.00 register/thread | +4 | 29.00 register/thread | +4 |
| Waves/SM | 59.88 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 | 59.88 | +0.00 |
| No Eligible (%) | 79.74 % | 45.65 % | -34.09 pp | 59.10 % | -20.64 pp | 70.63 % | -9.11 pp | 70.36 % | -9.38 pp |
| IPC Active | 0.79 inst/cycle | 2.16 inst/cycle | +1.37 | 1.65 inst/cycle | +0.86 | 1.19 inst/cycle | +0.40 | 1.20 inst/cycle | +0.41 |

---

## Cumulative Progress

```
Baseline:  285920 ns  (1.000×)
Iter 2     231360 ns  (1.236×)  — shared_memory_tiled_transpose_coalescing
Iter 3     230460 ns  (1.241×)  — increase_num_stages_to_pipeline_memory_latency
Iter 4     230340 ns  (1.241×)  — vectorized_store_hints_on_hw_1d_index

Final best: 316900 ns  (0.9022×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,24 +5,75 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 32,  'YBLOCK': 32},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 64},  num_warps=4, num_stages=3),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 128}, num_warps=8, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 16,  'YBLOCK': 64},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 16},  num_warps=4, num_stages=2),
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 64},  num_warps=8, num_stages=3),
+        triton.Config({'XBLOCK': 64,  'YBLOCK': 128}, num_warps=8, num_stages=3),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
-    ynumel = 4096
-    xnumel = 31500
-    yoffset = tl.program_id(1) * YBLOCK
-    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
-    xoffset = tl.program_id(0) * XBLOCK
-    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
-    xmask = xindex < xnumel
-    x2 = xindex
-    y0 = (yindex % 128)
-    y1 = yindex // 128
-    y3 = yindex
-    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 4032000*y1), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (x2 + 31500*y3), tmp0, xmask)
+    # ynumel=4096 (N*C=32*128), xnumel=31500 (H*W=126*250)
+    # Input  stride: (4032000, 1, 32000, 128) -- contiguous along C (stride=1)
+    # Output stride: (4032000, 31500, 250, 1) -- contiguous along W (stride=1)
+    pid_nc = tl.program_id(0)  # tiles N*C (ynumel=4096)
+    pid_hw = tl.program_id(1)  # tiles H*W (xnumel=31500)
 
+    nc_off = pid_nc * XBLOCK + tl.arange(0, XBLOCK)  # [XBLOCK]
+    hw_off = pid_hw * YBLOCK + tl.arange(0, YBLOCK)  # [YBLOCK]
+    nc_mask = nc_off < ynumel
+    hw_mask = hw_off < xnumel
 
+    # --- Scalar CTA-level base values (4 IDIVs total per CTA, not per element) ---
+    nc_base = pid_nc * XBLOCK          # scalar int
+    n_base  = nc_base // 128           # scalar IDIV (once per CTA)
+    c_base  = nc_base % 128            # scalar IMOD (once per CTA)
+    hw_base = pid_hw * YBLOCK          # scalar int
+    h_base  = hw_base // 250           # scalar IDIV (once per CTA)
+    w_base  = hw_base % 250            # scalar IMOD (once per CTA)
+
+    # --- Per-element: only additions + small-range carry correction ---
+    # nc decomposition: XBLOCK divides 128, so c_local is always in [0,128) — no wrap
+    nc_arange = tl.arange(0, XBLOCK)   # [XBLOCK]
+    c_l_1d = c_base + nc_arange        # [XBLOCK], no modulo needed (no wrap guaranteed)
+
+    # hw decomposition: carry trick handles at most one row boundary per tile (YBLOCK <= 250)
+    hw_arange = tl.arange(0, YBLOCK)   # [YBLOCK]
+    w_local   = w_base + hw_arange     # [YBLOCK], range [w_base, w_base+YBLOCK)
+    h_carry   = (w_local >= 250).to(tl.int32)  # [YBLOCK], 0 or 1
+    w_l_1d    = w_local - h_carry * 250        # [YBLOCK], equivalent to % 250
+    h_l_1d    = h_base + h_carry               # [YBLOCK]
+
+    # COALESCED LOAD: nc is the fast (column) axis, C stride=1 in input
+    # Broadcast to 2D tile [YBLOCK, XBLOCK]:
+    c_l_2d = c_l_1d[None, :]          # [1, XBLOCK]
+    h_l_2d = h_l_1d[:, None]          # [YBLOCK, 1]
+    w_l_2d = w_l_1d[:, None]          # [YBLOCK, 1]
+    mask_load = hw_mask[:, None] & nc_mask[None, :]  # [YBLOCK, XBLOCK]
+    load_idx = n_base * 4032000 + c_l_2d + h_l_2d * 32000 + w_l_2d * 128
+    tmp = tl.load(in_ptr0 + load_idx, mask=mask_load)  # [YBLOCK, XBLOCK]
+
+    # CRITICAL: tl.trans() swaps axes -> [XBLOCK, YBLOCK]
+    tmp_t = tl.trans(tmp)
+
+    # COALESCED STORE: hw is the fast (column) axis, stride=1 in output
+    # Store index using same scalar bases:
+    c_s_local = c_base + tl.arange(0, XBLOCK)  # [XBLOCK], same as c_l_1d
+    c_s_2d = c_s_local[:, None]       # [XBLOCK, 1]
+    hw_store_1d = hw_base + tl.arange(0, YBLOCK)  # [YBLOCK], flat hw index (contiguous in output)
+    mask_store = nc_mask[:, None] & hw_mask[None, :]  # [XBLOCK, YBLOCK]
+    store_idx = n_base * 4032000 + c_s_2d * 31500 + hw_store_1d[None, :]
+    tl.store(out_ptr0 + store_idx, tmp_t, mask=mask_store)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +81,7 @@

     ynumel = 4096
     xnumel = 31500
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
