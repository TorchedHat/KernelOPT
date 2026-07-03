# Kernel Optimization Log

## Summary

- **Kernel**: `conv1d_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7790000 ns — **Best latency**: 5330000 ns (**1.4615×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_tl_trans_register_spill_via_direct_block_m_block_k_load`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `conv1d_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_067/synthesis/kernel_0_unknown.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (131070,1,1), Waves/SM: 330.98
- **Registers/thread**: 143
- **Duration**: 7790000 ns
- **SM Throughput**: 55.96 % | **Memory**: 34.52 % | **DRAM**: 11.17 %
- **Achieved Occupancy**: 18.61 %
- **IPC Active**: 2.19 inst/cycle | **No Eligible**: 45.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.003× | ✓ |  |
| D | reduce_register_pressure_via_loop_restructuring_and_smaller_tiles | 1.548× | 1.462× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 7790000 ns | 7770000 ns | -20000 ns (-0.3%) | 5330000 ns | -2460000 ns (-31.6%) |
| SM Throughput (%) | 55.96 % | 55.96 % | +0.00 pp | 29.54 % | -26.42 pp |
| Memory Throughput (%) | 34.52 % | 34.52 % | +0.00 pp | 60.59 % | +26.07 pp |
| DRAM Throughput (%) | 11.17 % | 11.20 % | +0.03 pp | 16.33 % | +5.16 pp |
| Achieved Occupancy (%) | 18.61 % | 18.61 % | +0.00 pp | 12.32 % | -6.29 pp |
| Registers/Thread | 143.00 register/thread | 143.00 register/thread | +0 | 242.00 register/thread | +99 |
| Waves/SM | 330.98 | 330.98 | +0.00 | 496.48 | +165.50 |
| No Eligible (%) | 45.27 % | 45.26 % | -0.01 pp | 70.38 % | +25.11 pp |
| IPC Active | 2.19 inst/cycle | 2.19 inst/cycle | +0.00 | 1.18 inst/cycle | -1.01 |

---

## Iteration 2

**Best so far**: 1.4615× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_x_load_via_transposed_tile | 0.740× | 0.545× | ✓ |  |
| B | coalesce_global_loads_via_input_layout_transposition | 1.550× | 1.462× | ✓ | **✓ best** |
| C | split_flat_k_loop_with_register_constrained_autotune_configs | 0.926× | 0.869× | ✓ |  |
| D | split_flat_k_loop_to_eliminate_divmod_register_pressure | 1.544× | 1.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_x_load_... | Δ | coalesce_global_l... | Δ | split_flat_k_loop... | Δ | split_flat_k_loop... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7790000 ns | 14280000 ns | +6490000 ns (+83.3%) | 5330000 ns | -2460000 ns (-31.6%) | 8970000 ns | +1180000 ns (+15.1%) | 5330000 ns | -2460000 ns (-31.6%) |
| SM Throughput (%) | 55.96 % | 17.22 % | -38.74 pp | 29.59 % | -26.37 pp | 39.72 % | -16.24 pp | 29.59 % | -26.37 pp |
| Memory Throughput (%) | 34.52 % | 76.75 % | +42.23 pp | 60.73 % | +26.21 pp | 47.40 % | +12.88 pp | 60.76 % | +26.24 pp |
| DRAM Throughput (%) | 11.17 % | 6.10 % | -5.07 pp | 16.33 % | +5.16 pp | 9.70 % | -1.47 pp | 16.34 % | +5.17 pp |
| Achieved Occupancy (%) | 18.61 % | 12.43 % | -6.18 pp | 12.32 % | -6.29 pp | 12.28 % | -6.33 pp | 12.32 % | -6.29 pp |
| Registers/Thread | 143.00 register/thread | 255.00 register/thread | +112 | 242.00 register/thread | +99 | 255.00 register/thread | +112 | 242.00 register/thread | +99 |
| Waves/SM | 330.98 | 496.48 | +165.50 | 496.48 | +165.50 | 992.95 | +661.97 | 496.48 | +165.50 |
| No Eligible (%) | 45.27 % | 83.68 % | +38.41 pp | 70.40 % | +25.13 pp | 62.52 % | +17.25 pp | 70.42 % | +25.15 pp |
| IPC Active | 2.19 inst/cycle | 0.65 inst/cycle | -1.54 | 1.18 inst/cycle | -1.01 | 1.50 inst/cycle | -0.69 | 1.18 inst/cycle | -1.01 |

---

## Iteration 3

**Best so far**: 1.4615× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_register_pressure_relief_via_tiny_tiles | 0.740× | 0.546× | ✓ |  |
| B | expand_autotune_with_register_budget_small_tile_configs | 1.102× | 0.868× | ✓ |  |
| C | coalesce_x_load_via_launcher_side_permute_contiguous | 0.925× | 0.868× | ✓ |  |
| D | revert_tl_trans_and_constrain_autotune_to_small_low_register_tiles | 1.162× | 0.647× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_register... | Δ | expand_autotune_w... | Δ | coalesce_x_load_v... | Δ | revert_tl_trans_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7790000 ns | 14260000 ns | +6470000 ns (+83.1%) | 8980000 ns | +1190000 ns (+15.3%) | 8980000 ns | +1190000 ns (+15.3%) | 12040000 ns | +4250000 ns (+54.6%) |
| SM Throughput (%) | 55.96 % | 17.21 % | -38.75 pp | 39.71 % | -16.25 pp | 39.66 % | -16.30 pp | 44.86 % | -11.10 pp |
| Memory Throughput (%) | 34.52 % | 76.71 % | +42.19 pp | 47.39 % | +12.87 pp | 47.33 % | +12.81 pp | 71.68 % | +37.16 pp |
| DRAM Throughput (%) | 11.17 % | 6.11 % | -5.06 pp | 9.70 % | -1.47 pp | 9.70 % | -1.47 pp | 10.87 % | -0.30 pp |
| Achieved Occupancy (%) | 18.61 % | 12.43 % | -6.18 pp | 12.28 % | -6.33 pp | 12.28 % | -6.33 pp | 18.41 % | -0.20 pp |
| Registers/Thread | 143.00 register/thread | 255.00 register/thread | +112 | 255.00 register/thread | +112 | 255.00 register/thread | +112 | 155.00 register/thread | +12 |
| Waves/SM | 330.98 | 496.48 | +165.50 | 992.95 | +661.97 | 992.95 | +661.97 | 1323.94 | +992.96 |
| No Eligible (%) | 45.27 % | 83.68 % | +38.41 pp | 62.52 % | +17.25 pp | 62.50 % | +17.23 pp | 55.12 % | +9.85 pp |
| IPC Active | 2.19 inst/cycle | 0.65 inst/cycle | -1.54 | 1.50 inst/cycle | -0.69 | 1.50 inst/cycle | -0.69 | 1.80 inst/cycle | -0.39 |


### Key Learnings

#### ❌ Plan B: `expand_autotune_with_register_budget_small_tile_configs` (0.868×)

When a Triton kernel already uses 242+ registers/thread (Block Limit Registers = 2), adding small-tile autotune configs (BLOCK_M=16/32, num_warps=2) does not reduce register pressure — the autotuner selected a config that increased registers to 255/thread, doubled the grid size (131K→262K blocks), and doubled executed instructions (1.1B→2.4B), causing a 1.69x slowdown. Occupancy remained pinned at 12.5% (Block Limit Registers still = 2), confirming the compiler did not emit fewer registers for the smaller tile configs. The strategy fails because Triton's register allocator is driven by the kernel body's live variable count, not the tile size alone; when the kernel body is already register-heavy (nested loops with large accumulator + multiple pointer arithmetic chains), smaller BLOCK_M/BLOCK_K tiles do not proportionally reduce register usage but do increase grid overhead and total work.

> **Applicable when**: Never — this strategy is counterproductive when the kernel body already has high register pressure (>200 regs/thread). Adding small-tile configs to the autotune list does not reduce register count; the compiler determines register usage from the kernel body's live variables, not the tile dimensions.

> **Avoid when**: Any kernel where registers_per_thread > 200 and Block Limit Registers is the binding occupancy constraint. In this regime, the autotuner may select a smaller-tile config that increases grid size and total instructions without reducing register count, causing net regression. The correct fix requires restructuring the kernel body to reduce live variable count (e.g., eliminating vector temporaries, reducing accumulator size, or using a fundamentally different algorithm).

---

## Iteration 4

**Best so far**: 1.4615× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | restore_large_tile_autotune_configs_to_recover_from_transpose_regression | 0.925× | 0.713× | ✓ |  |
| B | coalesce_weight_loads_via_launcher_side_permute_to_cin_k_cout | 0.739× | 0.547× | ✓ |  |
| C | coalesce_x_load_via_transpose_with_full_large_tile_autotune | 0.925× | 0.868× | ✓ |  |
| D | eliminate_tl_trans_register_spill_via_direct_block_m_block_k_load | 1.551× | 1.462× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | restore_large_til... | Δ | coalesce_weight_l... | Δ | coalesce_x_load_v... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7790000 ns | 10930000 ns | +3140000 ns (+40.3%) | 14250000 ns | +6460000 ns (+82.9%) | 8980000 ns | +1190000 ns (+15.3%) | 5330000 ns | -2460000 ns (-31.6%) |
| SM Throughput (%) | 55.96 % | 16.45 % | -39.51 pp | 17.21 % | -38.75 pp | 39.71 % | -16.25 pp | 29.58 % | -26.38 pp |
| Memory Throughput (%) | 34.52 % | 40.64 % | +6.12 pp | 76.72 % | +42.20 pp | 47.38 % | +12.86 pp | 60.70 % | +26.18 pp |
| DRAM Throughput (%) | 11.17 % | 26.95 % | +15.78 pp | 6.11 % | -5.06 pp | 9.70 % | -1.47 pp | 16.33 % | +5.16 pp |
| Achieved Occupancy (%) | 18.61 % | 12.48 % | -6.13 pp | 12.43 % | -6.18 pp | 12.28 % | -6.33 pp | 12.32 % | -6.29 pp |
| Registers/Thread | 143.00 register/thread | 255.00 register/thread | +112 | 255.00 register/thread | +112 | 255.00 register/thread | +112 | 242.00 register/thread | +99 |
| Waves/SM | 330.98 | 248.24 | -82.74 | 496.48 | +165.50 | 992.95 | +661.97 | 496.48 | +165.50 |
| No Eligible (%) | 45.27 % | 86.17 % | +40.90 pp | 83.69 % | +38.42 pp | 62.49 % | +17.22 pp | 70.35 % | +25.08 pp |
| IPC Active | 2.19 inst/cycle | 0.55 inst/cycle | -1.64 | 0.65 inst/cycle | -1.54 | 1.50 inst/cycle | -0.69 | 1.18 inst/cycle | -1.01 |


### Key Learnings

#### ❌ Plan B: `coalesce_weight_loads_via_launcher_side_permute_to_cin_k_cout` (0.547×)

Permuting the weight tensor in the launcher (C_OUT,C_IN,K) → (C_IN,K,C_OUT) to make stride_wco=1 was combined with a tl.trans() call on the X tile inside the kernel body. Despite the weight permute being a launcher-only change, the bundled tl.trans(x_tile_T) on a (BLOCK_K, BLOCK_M) tile triggered catastrophic register spill: NCU reported local load/store utilization of only 1.0 of 32 bytes per sector, uncoalesced shared access exploded from 11K to 1.56B excessive wavefronts, and warp cycles per issued instruction more than doubled (5.23→12.18). The L1 hit rate improved slightly (87%→90%) and L2 hit rate improved (65%→79%), confirming the weight coalescing worked, but the spill-induced stalls (CPIStall: 7.9 cycles/warp on L1TEX, 64.6% of total stall) overwhelmed any memory access benefit, causing a 1.83x wall-time regression (8.98ms→14.25ms).

> **Applicable when**: Never apply tl.trans() on large tiles (BLOCK_K × BLOCK_M) when the kernel is already at 255 registers/thread and Block Limit Registers = 2. The transposed load pattern forces the compiler to keep both the transposed tile and the accumulator live simultaneously, causing spill to local memory that is far more expensive than the uncoalesced access it was meant to fix.

> **Avoid when**: Any Triton kernel where registers_per_thread ≥ 242 and Block Limit Registers = 2 — adding tl.trans() on any tile of size BLOCK_K × BLOCK_M will trigger register spill (NCU signature: local load/store bytes_per_sector = 1.0). The definitive spill indicator is MemoryCacheAccessPattern showing 1.0 bytes/sector for local loads/stores combined with CPIStall on L1TEX dominating warp stall cycles.

#### ❌ Plan C: `coalesce_x_load_via_transpose_with_full_large_tile_autotune` (0.868×)

After a prior iteration applied a launcher-side transpose (X from (N,C_IN,L_IN) to (N,L_IN,C_IN)) while simultaneously pruning large autotune configs, this iteration attempted to recover by restoring large-tile configs (BLOCK_M=128, BLOCK_K=64) alongside the transpose. However, all profiling metrics are statistically identical before and after (duration 8.98ms, grid 262K, instructions 2.39B, registers 255, occupancy 12.28%), indicating the autotuner did not select any of the newly added large-tile configs. The kernel body operating on the transposed layout appears to prevent the compiler from emitting a lower-register-count kernel even with large tiles, keeping Block Limit Registers=2 and grid size pinned at 262K. The transpose + large-tile autotune combination is not sufficient to recover from the regressed state introduced in iteration 3.

> **Applicable when**: Never — restoring large-tile autotune configs after a transpose-induced regression does not help when the kernel body operating on the transposed layout forces 255 registers/thread regardless of tile size. The autotuner will not select large-tile configs if they do not outperform the currently selected config.

> **Avoid when**: Any scenario where the kernel is already in a regressed state (registers=255, Block Limit Registers=2, grid doubled from baseline) due to a prior transpose + config-pruning change. Adding large-tile configs back to the autotune list is insufficient to recover; the kernel body itself must be restructured to reduce register pressure before the autotuner can benefit from larger tiles.

#### ✅ Plan D: `eliminate_tl_trans_register_spill_via_direct_block_m_block_k_load` (1.462×)

When a Triton kernel loads an input tile as (BLOCK_K, BLOCK_M) and then calls tl.dot(tl.trans(x_tile), w_tile), the compiler must materialize the transposed tile in registers or spill it to local memory, consuming all available registers (255/thread) and capping occupancy at the 2-block-per-SM limit. The definitive NCU signature is MemoryCacheAccessPattern showing exactly 1.0 of 32 bytes utilized per sector for both local loads and local stores. The fix is to swap the broadcasting axes on the pointer construction so the tile is loaded directly as (BLOCK_M, BLOCK_K), enabling tl.dot(x_tile, w_tile) with no transpose — this eliminates the spill, reduces registers/thread (255→242), halves total executed instructions (2.39B→1.12B), and cuts wall time by 40% (8.98ms→5.33ms). The autotuner simultaneously selected a 2x smaller grid (262K→131K blocks), confirming the restructured kernel body enabled a more efficient tile configuration.

> **Applicable when**: A Triton kernel loads a 2D tile with transposed broadcasting axes (e.g., batch[None,:] and in_pos_base[None,:] for BLOCK_M dimension, offs_c[:,None] for BLOCK_K dimension) producing shape (BLOCK_K, BLOCK_M), then calls tl.dot(tl.trans(x_tile), w_tile). NCU MemoryCacheAccessPattern shows ~1.0 bytes/sector for local loads and local stores (register spill signature). registers_per_thread is at or near 255 and Block Limit Registers = 2.

> **Avoid when**: The tile is already loaded in (BLOCK_M, BLOCK_K) layout and tl.trans() is not present. Also do not apply if the weight tile (BLOCK_K, BLOCK_N) layout would need to change to maintain correct dot product inner dimensions — verify that swapping x axes preserves the (BLOCK_M, BLOCK_K) × (BLOCK_K, BLOCK_N) contraction.

---

## Iteration 5

**Best so far**: 1.4615× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesce_output_store_via_transposed_y_layout | 0.739× | 0.547× | ✓ |  |
| B | eliminate_divmod_register_pressure_via_3d_grid_launch | 0.944× | 0.724× | ✓ |  |
| C | strip_large_tile_configs_to_restore_shared_memory_budget_and_l1_hit_rate | 1.282× | 1.040× | ✓ |  |
| D | eliminate_tl_trans_register_spill_via_direct_block_m_block_k_load | 1.534× | 1.462× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesce_output_s... | Δ | eliminate_divmod_... | Δ | strip_large_tile_... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7790000 ns | 14250000 ns | +6460000 ns (+82.9%) | 10760000 ns | +2970000 ns (+38.1%) | 7490000 ns | -300000 ns (-3.9%) | 5330000 ns | -2460000 ns (-31.6%) |
| SM Throughput (%) | 55.96 % | 17.22 % | -38.74 pp | 17.76 % | -38.20 pp | 40.06 % | -15.90 pp | 29.58 % | -26.38 pp |
| Memory Throughput (%) | 34.52 % | 76.74 % | +42.22 pp | 43.84 % | +9.32 pp | 49.17 % | +14.65 pp | 60.68 % | +26.16 pp |
| DRAM Throughput (%) | 11.17 % | 6.11 % | -5.06 pp | 26.73 % | +15.56 pp | 11.62 % | +0.45 pp | 16.34 % | +5.17 pp |
| Achieved Occupancy (%) | 18.61 % | 12.43 % | -6.18 pp | 12.47 % | -6.14 pp | 12.23 % | -6.38 pp | 12.32 % | -6.29 pp |
| Registers/Thread | 143.00 register/thread | 255.00 register/thread | +112 | 255.00 register/thread | +112 | 252.00 register/thread | +109 | 242.00 register/thread | +99 |
| Waves/SM | 330.98 | 496.48 | +165.50 | 248.24 | -82.74 | 992.95 | +661.97 | 496.48 | +165.50 |
| No Eligible (%) | 45.27 % | 83.69 % | +38.42 pp | 84.98 % | +39.71 pp | 65.68 % | +20.41 pp | 70.40 % | +25.13 pp |
| IPC Active | 2.19 inst/cycle | 0.65 inst/cycle | -1.54 | 0.60 inst/cycle | -1.59 | 1.37 inst/cycle | -0.82 | 1.18 inst/cycle | -1.01 |


### Key Learnings

#### ❌ Plan B: `eliminate_divmod_register_pressure_via_3d_grid_launch` (0.724×)

Replacing BLOCK_M-wide batch/l_out vectors (computed via integer div/mod on offs_m) with scalar batch decoded from a 2D grid (pid_m // num_lout_tiles, pid_m % num_lout_tiles) failed to reduce register pressure: registers/thread increased from 242 to 255, and the autotuner selected a config with block_size=256 and 196KB dynamic shared memory, making both Block Limit Registers and Block Limit Shared Mem equal to 1. This collapsed L1/TEX hit rate from 77% to 2.2%, doubled warp cycles per issued instruction (6.65→13.28), and introduced register spill to local memory (1.0 of 32 bytes per sector for local loads/stores), causing a 1.38x wall-time regression (5.33ms→10.76ms). The scalar batch decode did not free enough registers to allow more blocks per SM; instead, the simultaneous autotune config change (adding large-tile configs with num_warps=8, num_stages=4) caused the autotuner to select a shared-memory-saturating config that hit both limits simultaneously.

> **Applicable when**: Never — decoding batch as a scalar from pid_m via integer div/mod (rather than from a dedicated grid axis) does not eliminate the BLOCK_M-wide l_out_offs vector and does not meaningfully reduce register pressure. The autotuner may simultaneously select a large-tile config that saturates shared memory, causing double occupancy collapse.

> **Avoid when**: Any Triton conv1d kernel where registers/thread is already at 242-255 and Block Limit Registers=2. Replacing vector batch/l_out with scalar batch decoded from pid_m does not free enough registers to increase blocks/SM, and bundling this change with new large-tile autotune configs (num_warps=8, num_stages=4) risks the autotuner selecting a config that simultaneously hits Block Limit Registers=1 and Block Limit Shared Mem=1, collapsing L1 hit rate to ~2% and introducing register spill.

#### ✅ Plan D: `eliminate_tl_trans_register_spill_via_direct_block_m_block_k_load` (1.462×)

When a Triton kernel loads an input tile as (BLOCK_K, BLOCK_M) and then calls tl.dot(tl.trans(x_tile), w_tile), the compiler must materialize the transposed tile in registers or spill it to local memory, consuming all available registers (255/thread) and capping occupancy at the 2-block-per-SM limit. The definitive NCU signature is MemoryCacheAccessPattern showing exactly 1.0 of 32 bytes utilized per sector for both local loads and local stores. The fix is to swap the broadcasting axes on the pointer construction so the tile is loaded directly as (BLOCK_M, BLOCK_K), enabling tl.dot(x_tile, w_tile) with no transpose — this eliminates the spill, reduces registers/thread (255→242), halves total executed instructions (2.39B→1.12B), and cuts wall time by 40% (8.98ms→5.33ms). The autotuner simultaneously selected a 2x smaller grid (262K→131K blocks), confirming the restructured kernel body enabled a more efficient tile configuration.

> **Applicable when**: A Triton kernel loads a 2D tile with transposed broadcasting axes (e.g., batch[None,:] and in_pos_base[None,:] for BLOCK_M dimension, offs_c[:,None] for BLOCK_K dimension) producing shape (BLOCK_K, BLOCK_M), then calls tl.dot(tl.trans(x_tile), w_tile). NCU MemoryCacheAccessPattern shows ~1.0 bytes/sector for local loads and local stores (register spill signature). registers_per_thread is at or near 255 and Block Limit Registers = 2.

> **Avoid when**: The tile is already loaded in (BLOCK_M, BLOCK_K) layout and tl.trans() is not present. Also do not apply if the weight tile (BLOCK_K, BLOCK_N) layout would need to change to maintain correct dot product inner dimensions — verify that swapping x axes preserves the (BLOCK_M, BLOCK_K) × (BLOCK_K, BLOCK_N) contraction.

---

## Cumulative Progress

```
Baseline:  7790000 ns  (1.000×)
Iter 1     5330000 ns  (1.462×)  — reduce_register_pressure_via_loop_restructuring_and_smaller_tiles

Final best: 5330000 ns  (1.4615×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 @triton.autotune(
     configs=[
@@ -12,6 +13,26 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=3),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=5),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=5),
+        # New small-tile configs to reduce register pressure and increase occupancy
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=1, num_stages=1),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=1),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
+        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=1),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=1),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
+        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
     ],
     key=['M_TOTAL', 'N_OUT', 'K_TOTAL'],
 )
@@ -38,36 +59,35 @@

 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-    for k_start in range(0, K_TOTAL, BLOCK_K):
-        offs_k = k_start + tl.arange(0, BLOCK_K)
+    # Two nested loops: outer over kernel positions, inner over input channels
+    for kk in range(KERNEL_SIZE):
+        # in_pos_base is (BLOCK_M,) — kk is a scalar, no vector div/mod needed
+        in_pos_base = l_out * STRIDE_CONV + kk * DILATION - PAD
+        mask_pos = (in_pos_base >= 0) & (in_pos_base < L_IN)
 
-        # Decode K -> (in_c, kk)
-        in_c = offs_k // KERNEL_SIZE
-        kk = offs_k % KERNEL_SIZE
+        for c_start in range(0, C_IN, BLOCK_K):
+            offs_c = c_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
+            mask_c = offs_c < C_IN
 
-        # Input position
-        in_pos = l_out[:, None] * STRIDE_CONV + kk[None, :] * DILATION - PAD
+            # X tile loaded directly as (BLOCK_M, BLOCK_K) — no tl.trans needed
+            x_ptrs = (X_ptr
+                      + batch[:, None] * stride_xn
+                      + offs_c[None, :] * stride_xc
+                      + in_pos_base[:, None] * stride_xl)
+            mask_x = ((offs_m[:, None] < M_TOTAL) & (batch[:, None] < N)
+                      & mask_pos[:, None] & mask_c[None, :])
+            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)  # (BLOCK_M, BLOCK_K)
 
-        mask_m = (offs_m[:, None] < M_TOTAL) & (batch[:, None] < N)
-        mask_k = offs_k[None, :] < K_TOTAL
-        mask_ic = in_c[None, :] < C_IN
-        mask_pos = (in_pos >= 0) & (in_pos < L_IN)
+            # W tile: (BLOCK_K, BLOCK_N) — w[co, in_c, kk], kk is scalar
+            w_ptrs = (W_ptr
+                      + offs_n[None, :] * stride_wco
+                      + offs_c[:, None] * stride_wci
+                      + kk * stride_wk)
+            mask_w = mask_c[:, None] & (offs_n[None, :] < N_OUT)
+            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)  # (BLOCK_K, BLOCK_N)
 
-        mask_x = mask_m & mask_k & mask_ic & mask_pos
-
-        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + in_pos * stride_xl
-        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
-
-        # Weight: w[co, in_c, kk] -> shape (BLOCK_K, BLOCK_N)
-        # offs_k indexes (in_c, kk), offs_n indexes co
-        mask_w_k = (offs_k[:, None] < K_TOTAL) & (in_c[:, None] < C_IN) & (kk[:, None] < KERNEL_SIZE)
-        mask_w_n = offs_n[None, :] < N_OUT
-        mask_w = mask_w_k & mask_w_n
-
-        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kk[:, None] * stride_wk
-        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
-
-        acc = tl.dot(x_tile, w_tile, acc=acc)
+            # Direct dot product: (BLOCK_M, BLOCK_K) x (BLOCK_K, BLOCK_N) — no transpose, no spill
+            acc = tl.dot(x_tile, w_tile, acc=acc)
 
     # Store output
     mask_out_m = (offs_m[:, None] < M_TOTAL) & (batch[:, None] < N) & (l_out[:, None] < L_OUT)
@@ -76,7 +96,6 @@

 
     y_ptrs = Y_ptr + batch[:, None] * stride_yn + offs_n[None, :] * stride_yco + l_out[:, None] * stride_yl
     tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)
-
 
 def kernel_function(weight, x):
     # weight: (C_out, C_in, K), x: (N, C_in, L_in)
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
