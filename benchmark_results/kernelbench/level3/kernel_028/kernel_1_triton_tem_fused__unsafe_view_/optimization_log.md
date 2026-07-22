# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 81020 ns — **Best latency**: 75680 ns (**1.0706×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`precompute_a_matrix_index_to_eliminate_local_memory_spill`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_1_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (112,1,1), Waves/SM: 0.42
- **Registers/thread**: 128
- **Duration**: 81020 ns
- **SM Throughput**: 11.30 % | **Memory**: 77.91 % | **DRAM**: 0.71 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.821× | 0.822× | ✓ |  |
| B | increase_block_n_to_reduce_shared_memory_bank_conflicts_and_improve_occupancy | 0.859× | 0.808× | ✓ |  |
| C | memory_coalescing_via_tile_reshape_and_autotune | 0.928× | 0.928× | ✓ |  |
| D | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.930× | 0.919× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | increase_block_n_... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 81020 ns | 98560 ns | +17540 ns (+21.6%) | 100260 ns | +19240 ns (+23.7%) | 87330 ns | +6310 ns (+7.8%) | 88190 ns | +7170 ns (+8.8%) |
| SM Throughput (%) | 11.30 % | 16.94 % | +5.64 pp | 16.62 % | +5.32 pp | 19.11 % | +7.81 pp | 19.07 % | +7.77 pp |
| Memory Throughput (%) | 77.91 % | 64.49 % | -13.42 pp | 63.34 % | -14.57 pp | 72.45 % | -5.46 pp | 72.31 % | -5.60 pp |
| DRAM Throughput (%) | 0.71 % | 0.58 % | -0.13 pp | 0.58 % | -0.13 pp | 0.65 % | -0.06 pp | 0.65 % | -0.06 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 226.00 register/thread | +98 | 154.00 register/thread | +26 | 154.00 register/thread | +26 |
| Waves/SM | 0.42 | 0.28 | -0.14 | 0.85 | +0.43 | 0.28 | -0.14 | 0.28 | -0.14 |
| No Eligible (%) | 86.27 % | 78.89 % | -7.38 pp | 78.96 % | -7.31 pp | 76.31 % | -9.96 pp | 76.31 % | -9.96 pp |
| IPC Active | 0.55 inst/cycle | 0.84 inst/cycle | +0.29 | 0.83 inst/cycle | +0.28 | 0.95 inst/cycle | +0.40 | 0.95 inst/cycle | +0.40 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.936× | 0.456× | ✓ |  |
| B | reduce_register_pressure_via_num_warps_and_num_stages_pipeline | 0.929× | 0.715× | ✓ |  |
| C | reduce_block_m_to_increase_occupancy_and_sm_coverage | 1.060× | 1.070× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.836× | 0.789× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | reduce_block_m_to... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 81020 ns | 177700 ns | +96680 ns (+119.3%) | 113310 ns | +32290 ns (+39.9%) | 75710 ns | -5310 ns (-6.6%) | 102620 ns | +21600 ns (+26.7%) |
| SM Throughput (%) | 11.30 % | 16.81 % | +5.51 pp | 14.17 % | +2.87 pp | 11.60 % | +0.30 pp | 16.62 % | +5.32 pp |
| Memory Throughput (%) | 77.91 % | 52.55 % | -25.36 pp | 30.99 % | -46.92 pp | 78.70 % | +0.79 pp | 63.67 % | -14.24 pp |
| DRAM Throughput (%) | 0.71 % | 0.32 % | -0.39 pp | 0.51 % | -0.20 pp | 0.76 % | +0.05 pp | 0.57 % | -0.14 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 | 254.00 register/thread | +126 | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.14 | -0.28 | 0.21 | -0.21 | 0.21 | -0.21 | 0.28 | -0.14 |
| No Eligible (%) | 86.27 % | 83.91 % | -2.36 pp | 63.85 % | -22.42 pp | 85.94 % | -0.33 pp | 79.66 % | -6.61 pp |
| IPC Active | 0.55 inst/cycle | 0.64 inst/cycle | +0.09 | 0.72 inst/cycle | +0.17 | 0.56 inst/cycle | +0.01 | 0.82 inst/cycle | +0.27 |


### Key Learnings

#### ❌ Plan A: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (0.789×)

Capping registers via maxnreg=128 in a Triton Config successfully reduced registers/thread (154→128) but triggered heavy register spilling to local memory, as evidenced by a new MemoryCacheAccessPattern rule firing (only 1 of 32 bytes per sector utilized for local loads/stores) and L2 Compression Input Sectors exploding from 25,971 to 462,068 (18x increase). Despite the register reduction, achieved occupancy remained flat at 6.25%, scheduler eligibility slightly worsened (0.24→0.20 eligible warps/scheduler), warp cycles per instruction increased (4.23→4.85), and duration grew 17.5% (87→103 µs). The anti-pattern is applying maxnreg when the kernel is already near the register spill threshold — forcing fewer registers causes the compiler to spill to local memory, which is accessed with poor coalescing (stride-between-threads pattern) and overwhelms L2 with uncoalesced traffic, negating any occupancy benefit.

> **Applicable when**: Never beneficial in this configuration. In general, maxnreg is only safe when the kernel has significant register headroom above the spill threshold AND the register reduction would meaningfully increase occupancy (i.e., Block Limit Registers is the sole binding constraint, not shared memory).

> **Avoid when**: Block Limit Shared Mem is co-binding or more restrictive than Block Limit Registers (occupancy won't improve even with fewer registers); registers/thread are already near the compiler's natural minimum for the algorithm (forcing lower will cause spilling); the kernel shows no local memory traffic before the change (any new local traffic is pure overhead). When L2 Compression Input Sectors increases dramatically after applying maxnreg, the spill overhead is dominating.

#### ❌ Plan D: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (0.789×)

Capping registers via maxnreg=128 in a Triton Config successfully reduced registers/thread (154→128) but triggered heavy register spilling to local memory, as evidenced by a new MemoryCacheAccessPattern rule firing (only 1 of 32 bytes per sector utilized for local loads/stores) and L2 Compression Input Sectors exploding from 25,971 to 462,068 (18x increase). Despite the register reduction, achieved occupancy remained flat at 6.25%, scheduler eligibility slightly worsened (0.24→0.20 eligible warps/scheduler), warp cycles per instruction increased (4.23→4.85), and duration grew 17.5% (87→103 µs). The anti-pattern is applying maxnreg when the kernel is already near the register spill threshold — forcing fewer registers causes the compiler to spill to local memory, which is accessed with poor coalescing (stride-between-threads pattern) and overwhelms L2 with uncoalesced traffic, negating any occupancy benefit.

> **Applicable when**: Never beneficial in this configuration. In general, maxnreg is only safe when the kernel has significant register headroom above the spill threshold AND the register reduction would meaningfully increase occupancy (i.e., Block Limit Registers is the sole binding constraint, not shared memory).

> **Avoid when**: Block Limit Shared Mem is co-binding or more restrictive than Block Limit Registers (occupancy won't improve even with fewer registers); registers/thread are already near the compiler's natural minimum for the algorithm (forcing lower will cause spilling); the kernel shows no local memory traffic before the change (any new local traffic is pure overhead). When L2 Compression Input Sectors increases dramatically after applying maxnreg, the spill overhead is dominating.

---

## Iteration 3

**Best so far**: 1.0701× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_halve_shared_memory_and_double_sm_occupancy | 1.056× | 1.056× | ✓ |  |
| B | remove_maxnreg_and_reduce_block_m_to_eliminate_register_spill | 1.061× | 0.994× | ✓ | **✓ best** |
| C | persistent_kernel_scheduling_to_eliminate_wave_boundary_idle | 0.892× | 0.795× | ✓ |  |
| D | increase_block_k_to_reduce_k_loop_iterations_and_shared_memory_conflict_overhead | 0.824× | 0.823× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | remove_maxnreg_an... | Δ | persistent_kernel... | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 81020 ns | 76700 ns | -4320 ns (-5.3%) | 81500 ns | +480 ns (+0.6%) | 101950 ns | +20930 ns (+25.8%) | 98400 ns | +17380 ns (+21.5%) |
| SM Throughput (%) | 11.30 % | 11.53 % | +0.23 pp | 10.61 % | -0.69 pp | 7.96 % | -3.34 pp | 16.88 % | +5.58 pp |
| Memory Throughput (%) | 77.91 % | 77.82 % | -0.09 pp | 73.15 % | -4.76 pp | 31.01 % | -46.90 pp | 64.23 % | -13.68 pp |
| DRAM Throughput (%) | 0.71 % | 0.74 % | +0.03 pp | 0.70 % | -0.01 pp | 0.56 % | -0.15 pp | 0.59 % | -0.12 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 80.00 register/thread | -48 | 96.00 register/thread | -32 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.14 | -0.28 | 0.17 | -0.25 | 0.21 | -0.21 | 0.28 | -0.14 |
| No Eligible (%) | 86.27 % | 85.97 % | -0.30 pp | 87.14 % | +0.87 pp | 80.79 % | -5.48 pp | 78.86 % | -7.41 pp |
| IPC Active | 0.55 inst/cycle | 0.56 inst/cycle | +0.01 | 0.51 inst/cycle | -0.04 | 0.77 inst/cycle | +0.22 | 0.84 inst/cycle | +0.29 |


### Key Learnings

#### ✅ Plan A: `reduce_block_k_to_halve_shared_memory_and_double_sm_occupancy` (1.056×)

Reducing BLOCK_K from 64 to 32 in a Triton GEMM kernel halved dynamic shared memory per block (49.15 KB → 24.58 KB), raising Block Limit Shared Mem from 4 to 6 and theoretical occupancy from 25% to 37.5%, while also reducing registers per thread from 96 to 80. However, achieved occupancy remained stubbornly flat at 6.25% because the grid (112 blocks on 132 SMs, waves_per_sm=0.21) is too small to fill the GPU regardless of per-SM capacity. The modest 1.056x speedup appears to come from the reduced register pressure (96→80 regs/thread) slightly improving instruction scheduling, not from the intended occupancy improvement. Bank conflicts and scheduler eligibility were completely unchanged (17-way conflicts, 82.3% conflict rate, 0.14 eligible warps/scheduler), confirming that when waves_per_sm < 0.5, theoretical occupancy improvements do not translate to achieved occupancy gains.

> **Applicable when**: Kernel is shared-memory-bound (Block Limit Shared Mem is the minimum block limit) AND waves_per_sm >= 0.5 so that more blocks per SM can actually be scheduled. Dynamic SMEM per block is large relative to per-SM shared memory capacity. Reducing BLOCK_K is safe when K is exactly divisible by the new BLOCK_K (preserving EVEN_K=True) and the resulting tl.dot tile dimensions remain valid for Tensor Cores (>= 16 in each dimension).

> **Avoid when**: Grid is too small to fill the GPU (waves_per_sm < 0.5): achieved occupancy cannot improve regardless of theoretical occupancy because there are not enough blocks. When bank conflicts are the primary bottleneck (SharedMemoryConflicts rule with >80% conflict rate), halving BLOCK_K does not reduce the per-access conflict degree — it only halves the number of conflicting accesses per K-loop iteration while doubling the number of iterations, leaving total conflict overhead unchanged. Do not expect speedup from occupancy improvement alone when achieved occupancy is pinned by grid size.

#### ❌ Plan D: `increase_block_k_to_reduce_k_loop_iterations_and_shared_memory_conflict_overhead` (0.823×)

Increasing BLOCK_K from 32 to 64 (while simultaneously restoring BLOCK_M from 32 back to 64) nearly doubled executed instructions (6.96M → 13.18M) and increased registers per thread (96 → 128), which reduced Block Limit Shared Mem from 4 to 3 and theoretical occupancy from 25% to 18.75%. Although bank conflicts per shared load improved (17-way → 12-way) and warp cycles per instruction improved (7.12 → 4.73), the absolute instruction count increase dominated, extending duration from 75.71 µs to 98.4 µs (a 30% regression). The critical confound is that BLOCK_M was simultaneously changed from 32 back to 64, which is the primary driver of the instruction count doubling — the BLOCK_K increase alone cannot be cleanly attributed as the cause. When BLOCK_M is the binding register-pressure lever (as established by exp_002_reduce_block_m), reverting it while also increasing BLOCK_K compounds register pressure and instruction count, negating any benefit from fewer K-loop iterations.

> **Applicable when**: Never apply BLOCK_K increase simultaneously with BLOCK_M increase in this kernel. In general, increasing BLOCK_K to reduce K-loop iterations is only beneficial when: (1) BLOCK_M is held constant at its optimal value, (2) the resulting shared memory increase does not reduce Block Limit Shared Mem below the current binding limit, and (3) register pressure does not increase (registers/thread stays flat or decreases).

> **Avoid when**: When BLOCK_M has been reduced to relieve register pressure (as the primary optimization), do not simultaneously revert BLOCK_M while increasing BLOCK_K — the instruction count will double. When Block Limit Shared Mem is already at 3-4 (near the binding constraint), increasing BLOCK_K will push shared memory per block higher and reduce Block Limit Shared Mem further, cutting theoretical occupancy. When registers/thread are already at 96 and near the Block Limit Registers boundary, a BLOCK_K increase that raises registers to 128 will tighten the register constraint.

---

## Iteration 4

**Best so far**: 1.0701× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_b_matrix_bank_conflicts_via_explicit_smem_transpose_staging | 1.000× | 0.999× | ✓ |  |
| B | eliminate_b_matrix_bank_conflicts_via_transposed_load_and_smem_padding | 1.062× | 0.994× | ✓ | **✓ best** |
| C | eliminate_shared_memory_bank_conflicts_via_explicit_smem_padding_and_swizzle | 1.021× | 1.029× | ✓ |  |
| D | eliminate_shared_memory_bank_conflicts_via_explicit_smem_swizzle_padding | 0.892× | 0.891× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_b_matri... | Δ | eliminate_b_matri... | Δ | eliminate_shared_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 81020 ns | 81090 ns | +70 ns (+0.1%) | 81540 ns | +520 ns (+0.6%) | 78750 ns | -2270 ns (-2.8%) | 90880 ns | +9860 ns (+12.2%) |
| SM Throughput (%) | 11.30 % | 10.89 % | -0.41 pp | 10.60 % | -0.70 pp | 11.25 % | -0.05 pp | 9.00 % | -2.30 pp |
| Memory Throughput (%) | 77.91 % | 73.48 % | -4.43 pp | 73.12 % | -4.79 pp | 75.75 % | -2.16 pp | 34.57 % | -43.34 pp |
| DRAM Throughput (%) | 0.71 % | 0.70 % | -0.01 pp | 0.70 % | -0.01 pp | 0.73 % | +0.02 pp | 0.63 % | -0.08 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 94.00 register/thread | -34 | 96.00 register/thread | -32 | 96.00 register/thread | -32 | 156.00 register/thread | +28 |
| Waves/SM | 0.42 | 0.17 | -0.25 | 0.17 | -0.25 | 0.21 | -0.21 | 0.28 | -0.14 |
| No Eligible (%) | 86.27 % | 86.58 % | +0.31 pp | 87.13 % | +0.86 pp | 86.33 % | +0.06 pp | 77.50 % | -8.77 pp |
| IPC Active | 0.55 inst/cycle | 0.53 inst/cycle | -0.02 | 0.51 inst/cycle | -0.04 | 0.55 inst/cycle | +0.00 | 0.90 inst/cycle | +0.35 |

---

## Iteration 5

**Best so far**: 1.0701× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | precompute_a_matrix_index_to_eliminate_local_memory_spill | 1.063× | 1.071× | ✓ | **✓ best** |
| B | replace_a_matrix_gather_with_block_ptr_to_eliminate_smem_bank_conflicts | 1.053× | 1.058× | ✓ |  |
| C | hoist_a_matrix_index_computation_out_of_k_loop_to_eliminate_mio_stalls | 1.060× | 0.993× | ✓ |  |
| D | hoist_a_matrix_gather_index_outside_k_loop_to_reduce_register_pressure | 0.958× | 0.952× | ✓ |  |


### Metric Comparison

| Metric | Baseline | precompute_a_matr... | Δ | replace_a_matrix_... | Δ | hoist_a_matrix_in... | Δ | hoist_a_matrix_ga... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 81020 ns | 75680 ns | -5340 ns (-6.6%) | 76610 ns | -4410 ns (-5.4%) | 81630 ns | +610 ns (+0.8%) | 85090 ns | +4070 ns (+5.0%) |
| SM Throughput (%) | 11.30 % | 11.37 % | +0.07 pp | 11.28 % | -0.02 pp | 10.55 % | -0.75 pp | 9.53 % | -1.77 pp |
| Memory Throughput (%) | 77.91 % | 78.83 % | +0.92 pp | 78.24 % | +0.33 pp | 72.72 % | -5.19 pp | 37.30 % | -40.61 pp |
| DRAM Throughput (%) | 0.71 % | 0.76 % | +0.05 pp | 0.76 % | +0.05 pp | 0.70 % | -0.01 pp | 0.67 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 96.00 register/thread | -32 | 80.00 register/thread | -48 | 96.00 register/thread | -32 | 128.00 register/thread | +0 |
| Waves/SM | 0.42 | 0.21 | -0.21 | 0.14 | -0.28 | 0.17 | -0.25 | 0.21 | -0.21 |
| No Eligible (%) | 86.27 % | 86.19 % | -0.08 pp | 86.34 % | +0.07 pp | 87.13 % | +0.86 pp | 76.98 % | -9.29 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.55 inst/cycle | +0.00 | 0.51 inst/cycle | -0.04 | 0.92 inst/cycle | +0.37 |


### Key Learnings

#### ✅ Plan A: `precompute_a_matrix_index_to_eliminate_local_memory_spill` (1.071×)

When a memory load inside a hot K-loop uses a non-linear index expression involving integer division and modulo (e.g., `base + 150528*((k + 768*(m%196)) // 150528)`), the compiler cannot simplify it at compile time and may spill intermediate values to local memory, triggering the MemoryCacheAccessPattern rule (1/32 bytes utilized per sector for local loads/stores). Hoisting the invariant portion of the index (the row base pointer, computed once per pid_m before the loop) and replacing the in-loop expression with a simple linear offset eliminates the per-iteration division/modulo, reducing total executed instructions (~3%) and L2 traffic (L2 Compression Input Sectors: 33,636 → 30,055). The speedup is modest (~7%) because the dominant bottleneck remains shared memory bank conflicts (17-way, 82% conflict rate) and low occupancy (6.25%), which are unaffected by this change.

> **Applicable when**: MemoryCacheAccessPattern rule fires for local loads/stores (1/32 bytes per sector utilization), indicating the compiler is spilling intermediate index values to local memory. The hot loop contains a non-linear index expression with integer division or modulo that depends on loop-invariant outer indices (e.g., tile row/column offsets). The invariant portion can be factored out and precomputed before the loop as a 1D vector.

> **Avoid when**: The index expression is already linear (no division/modulo), or the non-linear terms depend on the inner loop variable (k_idx) and cannot be hoisted. When the dominant bottleneck is shared memory bank conflicts or low occupancy (not local memory spill), this change will yield only marginal improvement. Do not apply if the mathematical equivalence of the simplified index cannot be verified for all valid tile coordinates (risk of correctness regression).

---

## Cumulative Progress

```
Baseline:  81020 ns  (1.000×)
Iter 2     75710 ns  (1.070×)  — reduce_block_m_to_increase_occupancy_and_sm_coverage
Iter 5     75680 ns  (1.071×)  — precompute_a_matrix_index_to_eliminate_local_memory_spill

Final best: 75680 ns  (1.0706×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,28 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_addmm_clone_t_unfold_view_1(in_ptr0, arg_B, in_ptr2, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_M : tl.constexpr = 32
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -56,25 +70,41 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # HOISTED: precompute A-matrix row base pointers outside K-loop to eliminate
+    # per-iteration modulo/division that causes local memory spill.
+    # a_row_base[m] = (offs_a_m[m] // 196) * 150528 + (offs_a_m[m] % 196) * 768
+    # This is mathematically equivalent to the original complex index expression
+    # since the wrap term 150528*((k + 768*(m%196)) // 150528) == 150528*(m//196)
+    # for all valid k in [0, 767] and m in [0, 391].
+    a_batch = offs_a_m // 196          # which batch element (0 or 1 for M=392)
+    a_spatial = offs_a_m % 196         # spatial position within batch
+    a_row_base = a_batch * 150528 + a_spatial * 768  # shape [BLOCK_M]
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 768*idx_m
-        tmp2 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n + 768*idx_m + 150528*((idx_n + 768*((idx_m % 196))) // 150528), [BLOCK_M, BLOCK_K])), None, eviction_policy='evict_last')
-        a = tmp2.broadcast_to(xindex.shape)
+        # NEW: simple linear A-matrix index — no division/modulo in hot loop
+        k_offs = offs_k[None, :] + (k_idx * BLOCK_K)   # shape [1, BLOCK_K]
+        a_idx = a_row_base[:, None] + k_offs             # shape [BLOCK_M, BLOCK_K]
+        a_mask_new = (offs_a_m[:, None] < M) & (k_offs < K)
+        a = tl.load(in_ptr2 + a_idx, a_mask_new, eviction_policy='evict_last')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 512*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # Use tl.make_block_ptr with order=(1,0) to stage B in row-major shared memory
+        # layout, eliminating column-stride 12-way bank conflicts.
+        # B is indexed as B[k, n] with stride_k=stride_bk=1, stride_n=stride_bn=768
+        b_ptr = tl.make_block_ptr(
+            base=B,
+            shape=(K, N),
+            strides=(stride_bk, stride_bn),
+            offsets=(k_idx * BLOCK_K, pid_n * BLOCK_N),
+            block_shape=(BLOCK_K, BLOCK_N),
+            order=(1, 0),
+        )
+        b = tl.load(b_ptr, boundary_check=(0, 1))
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +118,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
