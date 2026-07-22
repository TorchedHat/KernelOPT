# Kernel Optimization Log

## Summary

- **Kernel**: `mean_dim1_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 14040000 ns — **Best latency**: 2330000 ns (**6.0258×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`memory_coalescing_via_larger_block_col`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `mean_dim1_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_048/synthesis/kernel_0_triton_red_fused_mean_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (524160,1,1), Waves/SM: 248.18
- **Registers/thread**: 25
- **Duration**: 14040000 ns
- **SM Throughput**: 15.82 % | **Memory**: 96.35 % | **DRAM**: 12.50 %
- **Achieved Occupancy**: 99.28 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.19 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_loads | 1.061× | 0.946× | ✓ |  |
| B | vectorized_loads_via_contiguous_hints | 1.190× | 1.009× | ✓ |  |
| C | memory_coalescing_via_larger_block_col | 1.198× | 6.026× | ✓ | **✓ best** |
| D | fix_uncoalesced_access_via_larger_block_col | 1.191× | 6.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | fix_uncoalesced_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14040000 ns | 14840000 ns | +800000 ns (+5.7%) | 13920000 ns | -120000 ns (-0.9%) | 2330000 ns | -11710000 ns (-83.4%) | 2340000 ns | -11700000 ns (-83.3%) |
| SM Throughput (%) | 15.82 % | 14.93 % | -0.89 pp | 15.91 % | +0.09 pp | 32.58 % | +16.76 pp | 31.78 % | +15.96 pp |
| Memory Throughput (%) | 96.35 % | 95.81 % | -0.54 pp | 96.93 % | +0.58 pp | 81.58 % | -14.77 pp | 81.52 % | -14.83 pp |
| DRAM Throughput (%) | 12.50 % | 11.83 % | -0.67 pp | 12.61 % | +0.11 pp | 81.58 % | +69.08 pp | 81.52 % | +69.02 pp |
| Achieved Occupancy (%) | 99.28 % | 99.26 % | -0.02 pp | 99.28 % | +0.00 pp | 61.39 % | -37.89 pp | 61.38 % | -37.90 pp |
| Registers/Thread | 25.00 register/thread | 24.00 register/thread | -1 | 25.00 register/thread | +0 | 48.00 register/thread | +23 | 48.00 register/thread | +23 |
| Waves/SM | 248.18 | 248.18 | +0.00 | 248.18 | +0.00 | 12.41 | -235.77 | 12.41 | -235.77 |
| No Eligible (%) | 86.19 % | 85.47 % | -0.72 pp | 86.11 % | -0.08 pp | 68.01 % | -18.18 pp | 68.05 % | -18.14 pp |
| IPC Active | 0.55 inst/cycle | 0.58 inst/cycle | +0.03 | 0.55 inst/cycle | +0.00 | 1.28 inst/cycle | +0.73 | 1.29 inst/cycle | +0.74 |


### Key Learnings

#### ✅ Plan D: `fix_uncoalesced_access_via_larger_block_col` (6.000×)

When a Triton kernel tiles a 3D tensor [batch, reduction, column] with a small BLOCK_COL (< 32), warp threads access non-contiguous memory locations along the reduction dimension (stride >> 1), causing only 4/32 bytes per cache sector to be utilized and triggering the UncoalescedGlobalAccess rule with 87% excessive sectors. Increasing BLOCK_COL to ≥ 32 (matching or exceeding the warp width of 32 threads) ensures that threads within a warp access contiguous column elements (stride_c=1), raising sector utilization from 4/32 to ~26/32 bytes and boosting effective memory throughput from 614 GB/s to 4 TB/s. The fix requires only changing the autotune config list — no kernel body changes — and the grid shrinks automatically as each program handles more output columns.

> **Applicable when**: UncoalescedGlobalAccess rule fires with >50% excessive sectors; MemoryCacheAccessPattern reports <8 bytes/sector utilized for global loads; kernel tiles a multi-dimensional tensor where the innermost (contiguous) dimension is the column axis but BLOCK_COL in autotune configs is smaller than warp width (32); L2 cache throughput is high (>80%) while DRAM throughput is low (<20%), indicating L2 is saturated by wasted sector traffic rather than actual data

> **Avoid when**: The reduction dimension IS the contiguous dimension in memory (stride_reduction=1), in which case small BLOCK_RED causes uncoalesced access instead; tensor is stored column-major; BLOCK_COL is already ≥ 32 in all configs; the bottleneck is compute-bound rather than memory-bound

---

## Iteration 2

**Best so far**: 6.0258× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | transpose_tile_to_eliminate_shared_memory_bank_conflicts | 1.075× | 5.875× | ✓ |  |
| B | add_num_stages_software_pipelining_to_reduce_smem_conflicts_and_register_pressure | 1.191× | 6.000× | ✓ |  |
| C | eviction_policy_streaming_loads_with_pruned_autotune_configs | 1.143× | 6.000× | ✓ |  |
| D | shared_memory_staged_reduction_with_coalesced_col_tiling | 1.125× | 5.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | transpose_tile_to... | Δ | add_num_stages_so... | Δ | eviction_policy_s... | Δ | shared_memory_sta... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14040000 ns | 2390000 ns | -11650000 ns (-83.0%) | 2340000 ns | -11700000 ns (-83.3%) | 2340000 ns | -11700000 ns (-83.3%) | 2360000 ns | -11680000 ns (-83.2%) |
| SM Throughput (%) | 15.82 % | 28.66 % | +12.84 pp | 32.33 % | +16.51 pp | 44.91 % | +29.09 pp | 40.86 % | +25.04 pp |
| Memory Throughput (%) | 96.35 % | 81.61 % | -14.74 pp | 81.45 % | -14.90 pp | 80.61 % | -15.74 pp | 81.57 % | -14.78 pp |
| DRAM Throughput (%) | 12.50 % | 81.61 % | +69.11 pp | 81.45 % | +68.95 pp | 80.61 % | +68.11 pp | 81.57 % | +69.07 pp |
| Achieved Occupancy (%) | 99.28 % | 73.60 % | -25.68 pp | 61.38 % | -37.90 pp | 96.78 % | -2.50 pp | 61.39 % | -37.89 pp |
| Registers/Thread | 25.00 register/thread | 40.00 register/thread | +15 | 48.00 register/thread | +23 | 32.00 register/thread | +7 | 48.00 register/thread | +23 |
| Waves/SM | 248.18 | 10.34 | -237.84 | 12.41 | -235.77 | 7.76 | -240.42 | 12.41 | -235.77 |
| No Eligible (%) | 86.19 % | 71.65 % | -14.54 pp | 68.10 % | -18.09 pp | 53.78 % | -32.41 pp | 58.68 % | -27.51 pp |
| IPC Active | 0.55 inst/cycle | 1.14 inst/cycle | +0.59 | 1.28 inst/cycle | +0.73 | 1.81 inst/cycle | +1.26 | 1.64 inst/cycle | +1.09 |


### Key Learnings

#### ✅ Plan A: `transpose_tile_to_eliminate_shared_memory_bank_conflicts` (5.875×)

Transposing a 2D tile from [BLOCK_RED, BLOCK_COL] to [BLOCK_COL, BLOCK_RED] and switching tl.sum from axis=0 to axis=1 was predicted to replace shared-memory tree reductions with warp-shuffle reductions, eliminating bank conflicts. However, the profiling data shows this did NOT work as intended: dynamic shared memory per block remained at 512 bytes, bank conflict percentages were essentially unchanged (load: 60.01%→60.03%, store: 66.33%→65.21%), and kernel duration actually increased from 2.33ms to 2.39ms. The absolute count of excessive shared wavefronts dropped dramatically (25M→393K) only because the winning autotuned config changed to one with far fewer shared memory operations, not because bank conflicts were structurally eliminated — the conflict rate remained identical.

> **Applicable when**: N/A — this strategy did not produce improvement

> **Avoid when**: Triton kernel with tl.sum over a 2D tile where bank conflicts are reported by NCU. Transposing the tile layout and switching from axis=0 to axis=1 reduction does NOT eliminate shared memory usage or bank conflicts in Triton on Hopper (H200/H100) — the dynamic SMEM allocation and bank conflict rates remain unchanged. The strategy is ineffective regardless of tile orientation because Triton's codegen for 2D reductions continues to use shared memory for inter-warp communication regardless of which axis is reduced.

---

## Iteration 3

**Best so far**: 6.0258× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_scalar_row_accumulation | 1.002× | 5.850× | ✓ |  |
| B | replace_tl_sum_with_tl_dot_ones_vector_to_fuse_fp32_and_eliminate_smem_bank_conflicts | 0.864× | 4.618× | ✓ |  |
| C | reduce_register_pressure_via_smaller_block_red_and_higher_num_warps | 1.129× | 1.238× | ✓ |  |
| D | eviction_policy_streaming_loads_and_stores | 1.005× | 5.802× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | replace_tl_sum_wi... | Δ | reduce_register_p... | Δ | eviction_policy_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 14040000 ns | 2400000 ns | -11640000 ns (-82.9%) | 3040000 ns | -11000000 ns (-78.3%) | 11340000 ns | -2700000 ns (-19.2%) | 2420000 ns | -11620000 ns (-82.8%) |
| SM Throughput (%) | 15.82 % | 74.71 % | +58.89 pp | 36.92 % | +21.10 pp | 41.84 % | +26.02 pp | 37.61 % | +21.79 pp |
| Memory Throughput (%) | 96.35 % | 73.96 % | -22.39 pp | 84.84 % | -11.51 pp | 71.85 % | -24.50 pp | 81.30 % | -15.05 pp |
| DRAM Throughput (%) | 12.50 % | 73.96 % | +61.46 pp | 57.88 % | +45.38 pp | 15.43 % | +2.93 pp | 81.30 % | +68.80 pp |
| Achieved Occupancy (%) | 99.28 % | 93.78 % | -5.50 pp | 61.39 % | -37.89 pp | 98.12 % | -1.16 pp | 73.56 % | -25.72 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 48.00 register/thread | +23 | 22.00 register/thread | -3 | 40.00 register/thread | +15 |
| Waves/SM | 248.18 | 3.88 | -244.30 | 12.41 | -235.77 | 15.52 | -232.66 | 10.34 | -237.84 |
| No Eligible (%) | 86.19 % | 25.12 % | -61.07 pp | 62.85 % | -23.34 pp | 66.53 % | -19.66 pp | 62.40 % | -23.79 pp |
| IPC Active | 0.55 inst/cycle | 2.99 inst/cycle | +2.44 | 1.49 inst/cycle | +0.94 | 1.34 inst/cycle | +0.79 | 1.49 inst/cycle | +0.94 |


### Key Learnings

#### ✅ Plan A: `reduce_register_pressure_via_scalar_row_accumulation` (5.850×)

When a Triton reduction kernel loads a 2D tile [BLOCK_RED, BLOCK_COL] into registers and reduces with tl.sum(tile, axis=0), register count is proportional to BLOCK_RED×BLOCK_COL, which limits Block Limit Registers and caps theoretical occupancy (e.g., 48 regs/thread → Block Limit Registers=10 → 62.5% occupancy). Replacing the 2D tile load with a scalar row-by-row loop using tl.static_range(BLOCK_RED) — loading one [BLOCK_COL] row at a time and accumulating element-wise — eliminates the BLOCK_RED register dimension entirely, dropping registers from 48 to 25/thread, raising Block Limit Registers from 10 to 16, and pushing theoretical occupancy to 100% (achieved 93.78%). This also eliminates all shared memory usage (dynamic SMEM drops from 512 to 0 bytes/block), completely removing the 60–66% shared memory bank conflicts that plagued the 2D tl.sum(axis=0) tree reduction, and raises eligible warps per scheduler from 0.52 to 2.71 and issue slot utilization from 32% to 75%.

> **Applicable when**: Triton reduction kernel uses a 2D tile [BLOCK_RED, BLOCK_COL] with tl.sum(tile, axis=0); NCU reports TheoreticalOccupancy limited by registers (Block Limit Registers < Block Limit Warps); registers per thread > 32; SharedMemoryConflicts and/or UncoalescedSharedAccess rules fire with >50% bank conflict rates; eligible warps per scheduler < 1.0 and No Eligible > 60%.

> **Avoid when**: BLOCK_RED is very large (>64) and tl.static_range unrolling would cause excessive instruction count bloat; the kernel is already register-efficient (<32 regs/thread) and occupancy is not register-limited; the 2D tile is needed for a tl.dot (GEMM) inner loop where the 2D layout is structurally required; the kernel is already compute-bound rather than occupancy/latency-bound.

#### ✅ Plan C: `reduce_register_pressure_via_smaller_block_red_and_higher_num_warps` (1.238×)

When a Triton reduction kernel is register-limited (e.g., 48 registers/thread capping theoretical occupancy at 62.5%), shrinking BLOCK_RED (the tile size along the reduction axis) and raising num_warps dramatically reduces register usage per thread (48→22) and pushes theoretical occupancy to 100%. However, the achieved speedup was only 1.24x and the kernel duration actually increased from 2.34ms to 11.34ms in absolute terms — the winning autotuned config (block_size=512, grid_size=8192) shifted the bottleneck from DRAM-bandwidth-bound (81% DRAM throughput) to L1-cache-bound (72.8% L1 throughput, only 15% DRAM throughput), indicating the smaller BLOCK_RED tiles cause far more loop iterations and instruction overhead that overwhelms the occupancy gain. The eligible warps per scheduler remained unchanged at 0.52 despite active warps rising from 9.86 to 15.71, confirming that the stall is not occupancy-limited but rather intrinsic to the memory access pattern.

> **Applicable when**: Kernel is register-limited (registers_per_thread > 40, Block Limit Registers < 12, theoretical occupancy < 70%) AND the kernel is NOT yet DRAM-bandwidth-bound (DRAM throughput < 60%). In that case, reducing tile size to lower register pressure may unlock occupancy gains that outweigh the extra loop iterations.

> **Avoid when**: Kernel is already DRAM-bandwidth-bound (DRAM throughput > 70%): shrinking BLOCK_RED increases loop iteration count and instruction overhead, shifting the bottleneck to L1/compute without proportionally reducing memory traffic. Also avoid when eligible warps per scheduler remains low (~0.5) even after occupancy increases — this signals the stall is intrinsic to the access pattern, not occupancy-limited. Do not apply when the kernel has already been coalesced and is near peak DRAM bandwidth.

#### ❌ Plan D: `eviction_policy_streaming_loads_and_stores` (5.802×)

Adding eviction_policy='evict_first' to tl.load and tl.store calls in a DRAM-bandwidth-bound streaming reduction kernel produced a regression: duration increased from 2.34ms to 2.42ms (~3.4% slower). Despite the MemoryL2Compression rule firing with 81% potential speedup and near-zero L2 compression success, the evict_first hint did not reduce L2 Cache Throughput (97.66% → 97.19%, essentially unchanged) nor improve DRAM throughput (81.45% → 81.3%). The kernel was already at DRAM bandwidth saturation, and the L2 is acting as a pass-through for DRAM traffic rather than a cache for reused data; evict_first provides no benefit when the bottleneck is raw DRAM bandwidth rather than L2 thrashing from reused data. The concurrent change to tile transposition (axis=1 reduction) also reduced register pressure (48→40 registers/thread) and improved occupancy (62.5%→75%), but this did not translate to throughput gains because the kernel remained DRAM-bound.

> **Applicable when**: Never for kernels already at DRAM bandwidth saturation (DRAM throughput >80%). The MemoryL2Compression rule firing with high speedup_pct is misleading when the kernel is already DRAM-bound — the L2 compression opportunity does not translate to actual speedup because the bottleneck is DRAM bandwidth, not L2 capacity.

> **Avoid when**: Kernel is already at DRAM bandwidth saturation (DRAM throughput >75%) and L2 hit rate is low (<20%), indicating data is streaming through L2 without reuse. In this regime, evict_first cannot improve throughput because the L2 is already effectively acting as a pass-through; the bottleneck is the DRAM interface itself, not L2 eviction policy. Do not trust MemoryL2Compression rule speedup_pct estimates when DRAM throughput is already near peak.

---

## Cumulative Progress

```
Baseline:  14040000 ns  (1.000×)
Iter 1     2330000 ns  (6.026×)  — memory_coalescing_via_larger_block_col

Final best: 2330000 ns  (6.0258×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -2,16 +2,36 @@

 import triton
 import triton.language as tl
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_COL': 1, 'BLOCK_RED': 256}, num_warps=4),
-        triton.Config({'BLOCK_COL': 1, 'BLOCK_RED': 512}, num_warps=8),
-        triton.Config({'BLOCK_COL': 1, 'BLOCK_RED': 1024}, num_warps=16),
-        triton.Config({'BLOCK_COL': 2, 'BLOCK_RED': 256}, num_warps=4),
-        triton.Config({'BLOCK_COL': 4, 'BLOCK_RED': 256}, num_warps=4),
-        triton.Config({'BLOCK_COL': 8, 'BLOCK_RED': 128}, num_warps=4),
-        triton.Config({'BLOCK_COL': 16, 'BLOCK_RED': 64}, num_warps=4),
-        triton.Config({'BLOCK_COL': 32, 'BLOCK_RED': 32}, num_warps=4),
+        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 64},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 64},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 64},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 128}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 128}, num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 32},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 64},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 64},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 128}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 128}, num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 16},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 16},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 32},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 32},  num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 64},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 64},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 8},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 8},   num_warps=4,  num_stages=3),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 16},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 16},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 32},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 32},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_COL': 512, 'BLOCK_RED': 8},   num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_COL': 512, 'BLOCK_RED': 8},   num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_COL': 512, 'BLOCK_RED': 16},  num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_COL': 512, 'BLOCK_RED': 16},  num_warps=16, num_stages=3),
     ],
     key=['N_BATCH', 'N_COLS', 'N_RED'],
 )
@@ -56,7 +76,6 @@

     
     out_offsets = batch_idx * N_COLS + col_offsets
     tl.store(out_ptr + out_offsets, acc, mask=col_mask)
-
 
 def kernel_function(x):
     N_BATCH, N_RED, N_COLS = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
