# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_37` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 29980 ns — **Best latency**: 32990 ns (**0.9088×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_37`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_37_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.31
- **Registers/thread**: 89
- **Duration**: 29980 ns
- **SM Throughput**: 11.99 % | **Memory**: 78.86 % | **DRAM**: 2.66 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.32 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_for_occupancy_and_bank_conflict_reduction | 1.030× | 0.909× | ✓ | **✓ best** |
| C | autotune_tile_sizes_for_occupancy_and_coalescing | 0.602× | 0.574× | ✓ |  |
| D | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 0.527× | 0.496× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29980 ns | 32990 ns | +3010 ns (+10.0%) | 52260 ns | +22280 ns (+74.3%) | 60480 ns | +30500 ns (+101.7%) |
| SM Throughput (%) | 11.99 % | 9.31 % | -2.68 pp | 11.74 % | -0.25 pp | 18.88 % | +6.89 pp |
| Memory Throughput (%) | 78.86 % | 66.35 % | -12.51 pp | 84.17 % | +5.31 pp | 73.28 % | -5.58 pp |
| DRAM Throughput (%) | 2.66 % | 2.41 % | -0.25 pp | 1.52 % | -1.14 pp | 1.32 % | -1.34 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 89.00 register/thread | 96.00 register/thread | +7 | 96.00 register/thread | +7 | 128.00 register/thread | +39 |
| Waves/SM | 0.31 | 0.09 | -0.22 | 0.19 | -0.12 | 0.23 | -0.08 |
| No Eligible (%) | 86.32 % | 78.78 % | -7.54 pp | 86.87 % | +0.55 pp | 78.90 % | -7.42 pp |
| IPC Active | 0.55 inst/cycle | 0.42 inst/cycle | -0.13 | 0.52 inst/cycle | -0.03 | 0.84 inst/cycle | +0.29 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tf32_tensor_cores | 0.606× | 0.575× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 29980 ns | 52130 ns | +22150 ns (+73.9%) |
| SM Throughput (%) | 11.99 % | 11.71 % | -0.28 pp |
| Memory Throughput (%) | 78.86 % | 83.94 % | +5.08 pp |
| DRAM Throughput (%) | 2.66 % | 1.53 % | -1.13 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 89.00 register/thread | 96.00 register/thread | +7 |
| Waves/SM | 0.31 | 0.19 | -0.12 |
| No Eligible (%) | 86.32 % | 86.90 % | +0.58 pp |
| IPC Active | 0.55 inst/cycle | 0.52 inst/cycle | -0.03 |


### Key Learnings

#### ⚠️ Plan B: `tf32_tensor_cores` (0.000×)

Plan 'tf32_tensor_cores' was applied but failed correctness. Error: Correctness FAILED:
seed=0 output[0]: max_diff=7.737e-02 > atol=0.001
seed=42 output[0]: max_diff=7.777e-02 > atol=0.001
seed=1337 output[0]: max_diff=7.552e-02 > atol=0.001

> **Applicable when**: The kernel uses ALLOW_TF32=False, forcing tl.dot to use scalar FP32 FFMA instructions that completely bypass Tensor Cores. On H200 SXM5, TF32 Tensor Cores deliver 989 TFLOPS vs 67 TFLOPS for FP32 FFMA — a 14.8x peak throughput ratio. The kernel achieves only 9.31% Compute SOL and 3.12% occupancy (2 active warps/scheduler vs 16 max), with 78.78% of scheduler cycles having no eligible warp. The primary bottleneck is the MIO/shared-memory stall (CPIStall: 66% of 4.7 cycles between instructions) caused by 17.9-way shared memory bank conflicts (83% of wavefronts). Enabling TF32 routes tl.dot through Tensor Core MMA units, which have fundamentally different execution characteristics and can dramatically increase arithmetic throughput. CORRECTNESS NOTE: A prior attempt combining TF32 + tile reshape failed with max_diff ~0.077. This plan isolates TF32 only (no tile changes) to determine if the correctness failure was caused by the tile reshape or TF32 itself.

> **Avoid when**: Previous implementation failed: Correctness FAILED:
seed=0 output[0]: max_diff=7.737e-02 > atol=0.001
seed=42 output[0]: max_diff=7.777e-02 > atol=0.001
seed=1337 output[0]: max_diff=7.552e-02 > atol=0.001

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 0.602× | 0.519× | ✓ |  |
| B | tma_loads_via_block_ptr | 0.605× | 0.577× | ✓ |  |
| C | memory_coalescing_fix_b_matrix_load_order | 0.603× | 0.577× | ✓ |  |
| D | shared_memory_b_tile_padding_to_eliminate_bank_conflicts | 0.605× | 0.575× | ✓ |  |
| E | add_triton_autotune | 0.822× | 0.786× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | tma_loads_via_blo... | Δ | memory_coalescing... | Δ | shared_memory_b_t... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29980 ns | 57760 ns | +27780 ns (+92.7%) | 51970 ns | +21990 ns (+73.3%) | 52000 ns | +22020 ns (+73.4%) | 52130 ns | +22150 ns (+73.9%) | 38140 ns | +8160 ns (+27.2%) |
| SM Throughput (%) | 11.99 % | 13.06 % | +1.07 pp | 11.92 % | -0.07 pp | 11.75 % | -0.24 pp | 11.75 % | -0.24 pp | 15.21 % | +3.22 pp |
| Memory Throughput (%) | 78.86 % | 79.43 % | +0.57 pp | 85.17 % | +6.31 pp | 84.25 % | +5.39 pp | 84.24 % | +5.38 pp | 64.47 % | -14.39 pp |
| DRAM Throughput (%) | 2.66 % | 1.38 % | -1.28 pp | 1.53 % | -1.13 pp | 1.53 % | -1.13 pp | 1.52 % | -1.14 pp | 2.09 % | -0.57 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 89.00 register/thread | 48.00 register/thread | -41 | 76.00 register/thread | -13 | 96.00 register/thread | +7 | 96.00 register/thread | +7 | 158.00 register/thread | +69 |
| Waves/SM | 0.31 | 0.10 | -0.21 | 0.16 | -0.15 | 0.19 | -0.12 | 0.19 | -0.12 | 0.16 | -0.15 |
| No Eligible (%) | 86.32 % | 85.45 % | -0.87 pp | 86.73 % | +0.41 pp | 86.87 % | +0.55 pp | 86.91 % | +0.59 pp | 65.49 % | -20.83 pp |
| IPC Active | 0.55 inst/cycle | 0.58 inst/cycle | +0.03 | 0.53 inst/cycle | -0.02 | 0.52 inst/cycle | -0.03 | 0.52 inst/cycle | -0.03 | 0.69 inst/cycle | +0.14 |


### Key Learnings

#### ❌ Plan A: `reduce_register_pressure_via_maxnreg_to_improve_occupancy` (0.519×)

Adding maxnreg=48 to Triton autotune configs halved registers per thread (96→48) and doubled Block Limit Registers (5→10), raising theoretical occupancy from 31.25% to 56.25%. However, achieved occupancy remained stuck at 6.25% (4 active warps/SM) because the grid was too small (124 blocks, 0.19→0.10 waves/SM) to fill the SM slots. Worse, the register cap forced heavy register spilling to local memory: executed instructions jumped from 4.83M to 5.96M (+23%), a new MemoryCacheAccessPattern rule fired reporting only 1/32 bytes utilized per sector for local loads/stores (76.95% speedup potential), and L2 traffic ballooned (L2 Compression Input Sectors: 16,794→483,405, a 28x increase). The spill overhead more than cancelled the marginal improvement in eligible warps (0.13→0.15), resulting in a 10.7% longer kernel duration.

> **Applicable when**: Never apply maxnreg capping when the kernel's achieved occupancy is already limited by grid size (waves_per_sm < 0.5) rather than by register pressure alone. If the grid cannot fill the SM slots even at higher theoretical occupancy, maxnreg only adds spill overhead without enabling more concurrent warps.

> **Avoid when**: Do not apply maxnreg when: (1) waves_per_sm < 1.0 and grid_size < sm_count, meaning the grid is too small to benefit from higher occupancy; (2) the kernel already has severe shared memory bank conflicts (17.9-way) that dominate stall cycles — spilling to local memory adds a second bottleneck on top of the existing one; (3) the register count is high because the kernel has a large live-variable working set (not redundant temporaries), as forced spilling will generate strided local memory traffic that is far more expensive than the register stalls it replaces.

#### ❌ Plan B: `tma_loads_via_block_ptr` (0.577×)

Replacing scalar pointer arithmetic with tl.make_block_ptr + tl.advance in a Triton GEMM kernel did not eliminate the 17.9-way shared memory bank conflicts: the SharedMemoryConflicts rule reported the same bank conflict count (7,142,400) and wavefront percentage (83.26%) before and after. The L2 hit rate actually degraded from 62.68% to 38.89%, and instruction count increased slightly (4.83M to 4.89M), resulting in no meaningful latency improvement (52.13us vs 51.97us). The root cause is that tl.make_block_ptr controls the global memory access pattern but does NOT change how Triton lays out data in shared memory — the bank conflict pattern is determined by the shared memory layout chosen by the Triton compiler, which was unchanged by this transformation. When the bottleneck is shared memory bank conflicts (not global memory coalescing), block pointer / TMA descriptor changes are ineffective.

> **Applicable when**: Never — this is a negative example. tl.make_block_ptr is appropriate when the bottleneck is global memory access coalescing (high DRAM throughput, low L2 hit rate from uncoalesced global loads). It does NOT fix shared memory bank conflicts.

> **Avoid when**: Do not apply tl.make_block_ptr / TMA descriptor changes when the primary bottleneck is SharedMemoryConflicts (17.9-way bank conflicts, 83%+ conflict wavefront percentage, 74%+ MIO CPIStall). The bank conflict pattern is determined by the shared memory layout chosen by the Triton compiler, not by the global memory access descriptor. tl.make_block_ptr changes how data is fetched from global memory into shared memory but does not change the intra-shared-memory access pattern during tl.dot computation. Additionally, the L2 hit rate degraded (62.68% → 38.89%) after this change, suggesting the block pointer approach altered prefetch/caching behavior negatively.

#### ❌ Plan C: `memory_coalescing_fix_b_matrix_load_order` (0.577×)

Correcting the B matrix global memory load index expression (from a malformed broadcast_to pattern to a direct stride-correct index) produced zero measurable improvement: duration changed by only 0.13µs (52.13→52.00µs), bank conflict way remained exactly 17.9-way, and all shared memory conflict metrics were completely unchanged. This reveals that the 17.9-way shared memory bank conflicts are caused by the shared memory store/load access pattern (how data is laid out in SRAM after the global load), not by the global memory load index itself. Fixing the HBM address arithmetic does not change how threads access the shared memory tile, so the conflict pattern persists entirely.

> **Applicable when**: Never — this is a negative example. Documents that fixing the global memory (HBM) load address does not affect shared memory bank conflicts.

> **Avoid when**: Do not assume that shared memory bank conflicts are caused by incorrect global memory load indices. Bank conflicts arise from how threads within a warp index into the shared memory array (the store pattern and load pattern within SRAM), which is independent of the HBM address used to fetch data. When SharedMemoryConflicts reports 17.9-way conflicts and UncoalescedSharedAccess reports 76% excessive wavefronts, the fix must target the shared memory indexing (e.g., padding the shared memory array, transposing the tile layout, or swapping row/column access order in the smem store/load), not the global load address.

---

## Cumulative Progress

```
Baseline:  29980 ns  (1.000×)

Final best: 32990 ns  (0.9088×)
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
+        triton.Config({}, num_warps=4,  num_stages=3),
+        triton.Config({}, num_warps=4,  num_stages=4),
+        triton.Config({}, num_warps=8,  num_stages=3),
+        triton.Config({}, num_warps=8,  num_stages=4),
+        triton.Config({}, num_warps=4,  num_stages=2),
+        triton.Config({}, num_warps=8,  num_stages=2),
+        triton.Config({}, num_warps=4,  num_stages=5),
+        triton.Config({}, num_warps=8,  num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_37(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -92,8 +106,6 @@

     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
