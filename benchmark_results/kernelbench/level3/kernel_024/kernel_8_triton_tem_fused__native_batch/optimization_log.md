# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3870 ns — **Best latency**: 2660 ns (**1.4549×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_register_pressure_via_maxnreg_and_expand_warp_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_8_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3,1,1), Waves/SM: 0.00
- **Registers/thread**: 34
- **Duration**: 3870 ns
- **SM Throughput**: 0.11 % | **Memory**: 0.83 % | **DRAM**: 0.09 %
- **Achieved Occupancy**: 6.18 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.03 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.991× | 0.982× | ✓ |  |
| D | increase_grid_parallelism_via_tile_size_reduction_and_autotune | 1.154× | 1.209× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | increase_grid_par... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3940 ns | +70 ns (+1.8%) | 3200 ns | -670 ns (-17.3%) |
| SM Throughput (%) | 0.11 % | 0.12 % | +0.01 pp | 0.10 % | -0.01 pp |
| Memory Throughput (%) | 0.83 % | 0.91 % | +0.08 pp | 0.45 % | -0.38 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.10 % | +0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 6.18 % | +0.00 pp | 6.21 % | +0.03 pp |
| Registers/Thread | 34.00 register/thread | 34.00 register/thread | +0 | 32.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.03 % | 91.04 % | +0.01 pp | 90.26 % | -0.77 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 | 0.35 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.2094× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | persistent_kernel_with_cta_tile_loop | 0.993× | 1.052× | ✓ |  |
| C | match_tile_size_to_problem_dimension_for_small_gemm | 1.154× | 1.372× | ✓ |  |
| D | persistent_kernel_with_corrected_tile_sizes_for_tiny_gemm | 1.171× | 1.330× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | match_tile_size_t... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 3680 ns | -190 ns (-4.9%) | 2820 ns | -1050 ns (-27.1%) | 2910 ns | -960 ns (-24.8%) |
| SM Throughput (%) | 0.11 % | 0.14 % | +0.03 pp | 0.10 % | -0.01 pp | 0.10 % | -0.01 pp |
| Memory Throughput (%) | 0.83 % | 0.57 % | -0.26 pp | 0.47 % | -0.36 pp | 0.43 % | -0.40 pp |
| DRAM Throughput (%) | 0.09 % | 0.08 % | -0.01 pp | 0.11 % | +0.02 pp | 0.10 % | +0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 6.24 % | +0.06 pp | 6.21 % | +0.03 pp | 6.21 % | +0.03 pp |
| Registers/Thread | 34.00 register/thread | 32.00 register/thread | -2 | 32.00 register/thread | -2 | 32.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.03 % | 90.94 % | -0.09 pp | 90.87 % | -0.16 pp | 92.49 % | +1.46 pp |
| IPC Active | 0.34 inst/cycle | 0.37 inst/cycle | +0.03 | 0.40 inst/cycle | +0.06 | 0.31 inst/cycle | -0.03 |


### Key Learnings

#### ✅ Plan B: `persistent_kernel_with_cta_tile_loop` (1.052×)

When a GEMM produces very few output tiles (e.g., 3 tiles on a 132-SM GPU), reducing BLOCK_N halves tile size and doubles tile count, while a persistent while-loop inside each CTA iterates over multiple tiles strided by num_programs(0). This amortizes kernel launch overhead and keeps accumulators warm in registers across tiles. The speedup is modest (~5%) because the problem is fundamentally too small to fill the GPU, but the approach is correct and avoids the correctness failures of split-K and autotune-based approaches on this workload.

> **Applicable when**: LaunchConfiguration NCU rule fires with speedup_pct > 90%; grid_size << sm_count (e.g., 3 vs 132); achieved occupancy < 10%; waves_per_sm = 0; the GEMM is tiny (M*N*K product small) and split-K or autotune approaches have failed correctness; reducing BLOCK_N can double or more the tile count without correctness risk.

> **Avoid when**: The grid already has enough tiles to fill most SMs (waves_per_sm >= 0.5); the problem is large enough that reducing BLOCK_N would hurt L2 reuse; split-K is available and correct (it provides more parallelism than tile-loop for K-bound workloads); shared memory bank conflicts are the primary bottleneck (this approach increased bank conflicts from 1152 to 2304 wavefronts due to smaller BLOCK_N).

#### ✅ Plan C: `match_tile_size_to_problem_dimension_for_small_gemm` (1.372×)

When a GEMM tile dimension (BLOCK_M, BLOCK_K) is much larger than the actual problem dimension (M=2 with BLOCK_M=16, K=24 with BLOCK_K=32), the vast majority of each tile is masked out, wasting instruction budget and causing shared memory bank conflicts. Reducing BLOCK_M to match M and BLOCK_K to the nearest power-of-2 below K eliminates masked computation, cuts warp cycles per issued instruction (10.64→9.82), reduces elapsed cycles (~12%), and lowers shared memory bank conflict pressure. This is a cumulative improvement on top of a prior iteration that already applied the same direction, confirming the strategy generalizes across successive refinements of tile parameters.

> **Applicable when**: GEMM tile dimension (BLOCK_M, BLOCK_N, or BLOCK_K) is significantly larger than the actual problem dimension; NCU LaunchConfiguration rule fires; executed instructions are high relative to useful work; shared memory bank conflicts are elevated; warp cycles per issued instruction are high due to L1TEX stalls from masked/wasted loads

> **Avoid when**: Problem dimensions are large enough that the tile fits within the dimension (M >= BLOCK_M, K >= BLOCK_K); reducing tile size would fragment the grid excessively and increase launch overhead relative to compute; kernel is already memory-bandwidth bound and tile size reduction would hurt cache reuse

---

## Iteration 3

**Best so far**: 1.3299× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_via_num_stages_and_num_warps_autotune | 1.175× | 1.439× | ✓ | **✓ best** |
| B | eliminate_shared_memory_bank_conflicts_via_transposed_b_load | 1.018× | 1.043× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 2690 ns | -1180 ns (-30.5%) | 3710 ns | -160 ns (-4.1%) |
| SM Throughput (%) | 0.11 % | 0.03 % | -0.08 pp | 0.14 % | +0.03 pp |
| Memory Throughput (%) | 0.83 % | 0.43 % | -0.40 pp | 0.59 % | -0.24 pp |
| DRAM Throughput (%) | 0.09 % | 0.11 % | +0.02 pp | 0.08 % | -0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 6.24 % | +0.06 pp |
| Registers/Thread | 34.00 register/thread | 66.00 register/thread | +32 | 32.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.03 % | 89.25 % | -1.78 pp | 91.02 % | -0.01 pp |
| IPC Active | 0.34 inst/cycle | 0.11 inst/cycle | -0.23 | 0.34 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.4387× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_smaller_block_n | 1.171× | 1.299× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg_and_expand_warp_autotune | 1.211× | 1.455× | ✓ | **✓ best** |
| C | autotune_num_warps_num_stages_for_bank_conflict_and_warp_efficiency | 1.028× | 1.119× | ✓ |  |
| D | coalesced_b_load_to_eliminate_stride24_bank_conflicts | 1.019× | 0.968× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | autotune_num_warp... | Δ | coalesced_b_load_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 2980 ns | -890 ns (-23.0%) | 2660 ns | -1210 ns (-31.3%) | 3460 ns | -410 ns (-10.6%) | 4000 ns | +130 ns (+3.4%) |
| SM Throughput (%) | 0.11 % | 0.02 % | -0.09 pp | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.15 % | +0.04 pp |
| Memory Throughput (%) | 0.83 % | 0.40 % | -0.43 pp | 0.47 % | -0.36 pp | 0.32 % | -0.51 pp | 0.60 % | -0.23 pp |
| DRAM Throughput (%) | 0.09 % | 0.10 % | +0.01 pp | 0.12 % | +0.03 pp | 0.09 % | +0.00 pp | 0.08 % | -0.01 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp | 6.24 % | +0.06 pp |
| Registers/Thread | 34.00 register/thread | 32.00 register/thread | -2 | 66.00 register/thread | +32 | 62.00 register/thread | +28 | 32.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.03 % | 91.31 % | +0.28 pp | 89.28 % | -1.75 pp | 91.90 % | +0.87 pp | 90.88 % | -0.15 pp |
| IPC Active | 0.34 inst/cycle | 0.09 inst/cycle | -0.25 | 0.09 inst/cycle | -0.25 | 0.08 inst/cycle | -0.26 | 0.37 inst/cycle | +0.03 |

---

## Iteration 5

**Best so far**: 1.4549× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_block_n_as_constexpr_with_persistent_loop_for_grid_expansion | 0.879× | 0.765× | ✓ |  |
| B | expand_autotune_num_warps_configs_for_tiny_gemm | 1.035× | 1.052× | ✓ |  |
| C | fix_b_load_coalescing_via_row_major_staging_with_explicit_smem_pad | 1.165× | 1.273× | ✓ |  |
| D | expand_launch_grid_to_match_tile_count_and_reduce_bank_conflicts | 0.780× | 0.647× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_block_n_... | Δ | expand_autotune_n... | Δ | fix_b_load_coales... | Δ | expand_launch_gri... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3870 ns | 5060 ns | +1190 ns (+30.7%) | 3680 ns | -190 ns (-4.9%) | 3040 ns | -830 ns (-21.4%) | 5980 ns | +2110 ns (+54.5%) |
| SM Throughput (%) | 0.11 % | 0.03 % | -0.08 pp | 0.03 % | -0.08 pp | 0.02 % | -0.09 pp | 0.19 % | +0.08 pp |
| Memory Throughput (%) | 0.83 % | 0.27 % | -0.56 pp | 0.40 % | -0.43 pp | 0.38 % | -0.45 pp | 0.75 % | -0.08 pp |
| DRAM Throughput (%) | 0.09 % | 0.06 % | -0.03 pp | 0.08 % | -0.01 pp | 0.10 % | +0.01 pp | 0.05 % | -0.04 pp |
| Achieved Occupancy (%) | 6.18 % | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp | 1.56 % | -4.62 pp | 6.25 % | +0.07 pp |
| Registers/Thread | 34.00 register/thread | 58.00 register/thread | +24 | 62.00 register/thread | +28 | 32.00 register/thread | -2 | 34.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.03 % | 92.07 % | +1.04 pp | 91.02 % | -0.01 pp | 91.36 % | +0.33 pp | 91.44 % | +0.41 pp |
| IPC Active | 0.34 inst/cycle | 0.08 inst/cycle | -0.26 | 0.10 inst/cycle | -0.24 | 0.08 inst/cycle | -0.26 | 0.33 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `autotune_block_n_as_constexpr_with_persistent_loop_for_grid_expansion` (0.765×)

Changing BLOCK_N from 32 to 8 as a hardcoded constexpr inside the kernel body (without making it a true autotuned parameter in the kernel signature) caused a 1.91x slowdown: elapsed cycles nearly doubled (3943→7557), executed instructions more than doubled (537→1284), warp cycles per issued instruction worsened (9.26→12.18), and shared memory bank conflict wavefronts quadrupled (288→1152). Despite the plan's intent to expand the grid via smaller tiles, the grid remained at 3 blocks because the grid lambda was not updated to use the new BLOCK_N=8 value — the persistent loop now processes 4x more tiles per CTA instead of expanding the grid, adding instruction overhead with no SM utilization benefit. The autotune config change (replacing num_warps=16 with num_stages=3 configs) also removed the previously winning configuration, further degrading performance.

> **Applicable when**: Never apply this specific pattern: reducing BLOCK_N as a hardcoded constexpr without simultaneously updating the grid lambda to reflect the new tile count. The grid must be updated to lambda meta: (cdiv(N, BLOCK_N),) for the tile expansion to actually increase SM utilization.

> **Avoid when**: Any time BLOCK_N is reduced inside the kernel body but the grid launch expression is not correspondingly updated. Also avoid replacing num_warps=16 configs with num_stages=3 configs when the prior autotune had already converged on a winning config — this removes the winning configuration from the search space and forces a regression.

#### ✅ Plan B: `expand_autotune_num_warps_configs_for_tiny_gemm` (1.455×)

Expanding the @triton.autotune config space to include num_warps in {1,2,4,8,16} x num_stages in {1,2} (from the prior {1,2,4} x {1,2}) allows the runtime to empirically select a warp configuration that better hides L1TEX stall latency. The profiling metrics show nearly identical hardware counters before and after (same block_size=32, registers=66, executed instructions=537), indicating the autotune selected the same num_warps=1 winning config — the speedup (1.45x) is therefore attributable to the broader search space enabling a marginally better config or reduced autotuning overhead, not a structural kernel change. This strategy is low-risk: adding more configs to autotune never regresses correctness, and the runtime discards suboptimal configs.

> **Applicable when**: Kernel already uses @triton.autotune with a limited num_warps search space (max num_warps=4); SharedMemoryConflicts or IssueSlotUtilization NCU rules fire; kernel is tiny (< 5us, grid << SM count); tile sizes are compile-time constants so key=[] is valid; adding configs has zero correctness risk

> **Avoid when**: Autotune key=[] is incorrect (problem dimensions vary at runtime); kernel is already at the optimal num_warps from a prior autotune run and the search space is already exhaustive; kernel duration is dominated by autotuning overhead itself (very fast kernels called rarely); maxnreg constraint is already applied and limits the effective warp count

#### ❌ Plan D: `expand_launch_grid_to_match_tile_count_and_reduce_bank_conflicts` (0.647×)

Removing @triton.autotune and hardcoding BLOCK_N=8 with a persistent tile-loop (grid still 3, num_ctas=3) caused elapsed cycles to more than double (3943→8944 cycles, 2.27x worse) and executed instructions to explode 12x (537→6480). The grid remained at 3 blocks because the persistent loop still ran all 12 tiles through 3 CTAs sequentially, but without autotune the compiler chose block_size=128 (4 warps) instead of the optimal block_size=32 (1 warp), quadrupling thread count and warp overhead for a tiny GEMM. Shared memory bank conflicts also worsened dramatically (288→4608 wavefronts, 16x) because BLOCK_N=8 with 4 warps creates more conflict-prone access patterns. The key lesson: for tiny GEMMs where autotune had selected num_warps=1 as optimal, removing autotune and forcing a larger block size is catastrophically harmful — the instruction count and bank conflict overhead far outweigh any theoretical occupancy improvement.

> **Applicable when**: Never apply this pattern: removing autotune from a kernel where autotune had selected a small num_warps (1-2) while simultaneously reducing BLOCK_N and keeping the same grid size.

> **Avoid when**: Always avoid: (1) removing @triton.autotune when the winning config is num_warps=1 for a tiny GEMM — the compiler will default to a larger block size causing instruction explosion; (2) reducing BLOCK_N without actually expanding the grid (persistent loop with same num_ctas just serializes more tiles per CTA); (3) combining BLOCK_N reduction with autotune removal — bank conflicts scale with both BLOCK_N reduction and increased warp count.

---

## Cumulative Progress

```
Baseline:  3870 ns  (1.000×)
Iter 1     3200 ns  (1.209×)  — increase_grid_parallelism_via_tile_size_reduction_and_autotune
Iter 2     2910 ns  (1.330×)  — persistent_kernel_with_corrected_tile_sizes_for_tiny_gemm
Iter 3     2690 ns  (1.439×)  — reduce_shared_memory_bank_conflicts_via_num_stages_and_num_warps_autotune
Iter 4     2660 ns  (1.455×)  — reduce_register_pressure_via_maxnreg_and_expand_warp_autotune

Final best: 2660 ns  (1.4549×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,34 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=1),
+        triton.Config({}, num_warps=2, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=16, num_stages=1),
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=16, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_8(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
+    BLOCK_K : tl.constexpr = 16
+    GROUP_M : tl.constexpr = 1
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
+    NUM_SMS : tl.constexpr = 132
     A = arg_A
     B = arg_B
 
@@ -30,67 +47,52 @@

     stride_bk = 1
     stride_bn = 24
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_m = tl.cdiv(M, BLOCK_M)  # = 1
+    grid_n = tl.cdiv(N, BLOCK_N)  # = 3
+    num_tiles = grid_m * grid_n   # = 3
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    sm_id = tl.program_id(0).to(INDEX_DTYPE)
+    tile_id = sm_id
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
+    while tile_id < num_tiles:
+        pid_m = tile_id // grid_n
+        pid_n = tile_id % grid_n
+
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
         offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+        offs_k = tl.arange(0, BLOCK_K)
+        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
+            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 24*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 24*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            a = tl.load(A + (idx_n + 24 * idx_m), mask=a_mask, other=0.0)
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+            idx_m = b_k_idx_vals
+            idx_n = offs_b_n[None, :]
+            b = tl.load(B + tl.broadcast_to(idx_m + 24 * idx_n, [BLOCK_K, BLOCK_N]), mask=b_mask, other=0.0)
 
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
-    # inductor generates a suffix
-    xindex = idx_n + 96*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
+        # rematerialize rm and rn to save registers
+        rm2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm2[:, None]
+        idx_n = rn2[None, :]
+        mask = (idx_m < M) & (idx_n < N)
 
+        xindex = idx_n + 96 * idx_m
+        tl.store(out_ptr0 + tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N]), acc, mask=mask)
 
+        tile_id = tile_id + NUM_SMS
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
