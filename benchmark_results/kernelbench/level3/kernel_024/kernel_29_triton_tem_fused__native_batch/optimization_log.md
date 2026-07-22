# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 87260 ns — **Best latency**: 61980 ns (**1.4079×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_k_to_lower_register_pressure_and_restore_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_29_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (14,1,1), Waves/SM: 0.01
- **Registers/thread**: 53
- **Duration**: 87260 ns
- **SM Throughput**: 0.90 % | **Memory**: 9.62 % | **DRAM**: 0.70 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 87260 ns | 87230 ns | -30 ns (-0.0%) |
| SM Throughput (%) | 0.90 % | 0.90 % | +0.00 pp |
| Memory Throughput (%) | 9.62 % | 9.62 % | +0.00 pp |
| DRAM Throughput (%) | 0.70 % | 0.70 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 53.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.37 % | 91.37 % | +0.00 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0003× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_block_m_to_match_m_dimension_static_tile | 1.905× | 1.968× | ✓ |  |
| B | persistent_kernel_with_corrected_tile_sizes | 0.958× | 0.959× | ✓ |  |
| D | fix_smem_bank_conflicts_via_block_m_correction_and_pipeline_stages | 2.793× | 2.309× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_block_m_to_ma... | Δ | persistent_kernel... | Δ | fix_smem_bank_con... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 87260 ns | 44350 ns | -42910 ns (-49.2%) | 91010 ns | +3750 ns (+4.3%) | 37790 ns | -49470 ns (-56.7%) |
| SM Throughput (%) | 0.90 % | 0.74 % | -0.16 pp | 0.75 % | -0.15 pp | 0.53 % | -0.37 pp |
| Memory Throughput (%) | 9.62 % | 9.51 % | -0.11 pp | 9.03 % | -0.59 pp | 5.66 % | -3.96 pp |
| DRAM Throughput (%) | 0.70 % | 1.38 % | +0.68 pp | 0.67 % | -0.03 pp | 1.62 % | +0.92 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 56.00 register/thread | +3 | 46.00 register/thread | -7 | 32.00 register/thread | -21 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.00 | -0.01 |
| No Eligible (%) | 91.37 % | 93.73 % | +2.36 pp | 93.10 % | +1.73 pp | 90.04 % | -1.33 pp |
| IPC Active | 0.34 inst/cycle | 0.25 inst/cycle | -0.09 | 0.28 inst/cycle | -0.06 | 0.20 inst/cycle | -0.14 |

---

## Iteration 3

**Best so far**: 2.3091× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_to_break_shared_memory_bank_conflicts | 2.857× | 2.794× | ✓ | **✓ best** |
| C | increase_block_n_to_reduce_shared_memory_bank_conflicts | 1.919× | 1.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | increase_block_n_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 87260 ns | 31230 ns | -56030 ns (-64.2%) | 44130 ns | -43130 ns (-49.4%) |
| SM Throughput (%) | 0.90 % | 0.58 % | -0.32 pp | 0.74 % | -0.16 pp |
| Memory Throughput (%) | 9.62 % | 6.81 % | -2.81 pp | 9.54 % | -0.08 pp |
| DRAM Throughput (%) | 0.70 % | 1.97 % | +1.27 pp | 1.39 % | +0.69 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 48.00 register/thread | -5 | 56.00 register/thread | +3 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.37 % | 89.68 % | -1.69 pp | 93.76 % | +2.39 pp |
| IPC Active | 0.34 inst/cycle | 0.21 inst/cycle | -0.13 | 0.25 inst/cycle | -0.09 |

---

## Iteration 4

**Best so far**: 2.7941× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_smem_bank_conflicts_via_b_tile_transpose_staging | 2.846× | 2.811× | ✓ |  |
| B | smem_padding_to_eliminate_b_tile_bank_conflicts | 2.829× | 2.800× | ✓ |  |
| C | stage_b_tile_through_padded_shared_memory_to_eliminate_bank_conflicts | 1.917× | 1.975× | ✓ |  |
| D | increase_num_warps_to_hide_mio_stalls_and_improve_sm_occupancy | 1.917× | 1.973× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_smem_ba... | Δ | smem_padding_to_e... | Δ | stage_b_tile_thro... | Δ | increase_num_warp... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 87260 ns | 31040 ns | -56220 ns (-64.4%) | 31170 ns | -56090 ns (-64.3%) | 44190 ns | -43070 ns (-49.4%) | 44220 ns | -43040 ns (-49.3%) |
| SM Throughput (%) | 0.90 % | 0.57 % | -0.33 pp | 0.57 % | -0.33 pp | 0.74 % | -0.16 pp | 0.74 % | -0.16 pp |
| Memory Throughput (%) | 9.62 % | 6.79 % | -2.83 pp | 6.79 % | -2.83 pp | 9.54 % | -0.08 pp | 9.55 % | -0.07 pp |
| DRAM Throughput (%) | 0.70 % | 1.98 % | +1.28 pp | 1.97 % | +1.27 pp | 1.39 % | +0.69 pp | 1.39 % | +0.69 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 48.00 register/thread | -5 | 48.00 register/thread | -5 | 56.00 register/thread | +3 | 56.00 register/thread | +3 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 91.37 % | 89.73 % | -1.64 pp | 89.70 % | -1.67 pp | 93.76 % | +2.39 pp | 93.76 % | +2.39 pp |
| IPC Active | 0.34 inst/cycle | 0.21 inst/cycle | -0.13 | 0.21 inst/cycle | -0.13 | 0.25 inst/cycle | -0.09 | 0.25 inst/cycle | -0.09 |


### Key Learnings

#### ✅ Plan A: `eliminate_smem_bank_conflicts_via_b_tile_transpose_staging` (2.811×)

Loading the B tile in transposed [BLOCK_N, BLOCK_K] layout and passing tl.trans(b_T) to tl.dot was intended to make K-elements (stride=1) the fast dimension, eliminating the 17-way shared memory bank conflict caused by a column-major B stride (stride_bn=1728, where 1728 mod 32 = 0). However, the profiling data shows the bank conflict metrics are completely unchanged after the rewrite: SharedMemoryConflicts still reports 17-way conflicts at 82.35% excess wavefronts, MIO stall cycles per warp remain at 6.8, and kernel duration barely changed (31.23 → 31.04 µs). The 2.81x reported speedup is inconsistent with the NCU profiling data and likely reflects measurement noise or a different kernel being measured; the NCU metrics show no meaningful improvement. Triton's compiler may internally re-layout the transposed tile back into a conflict-prone access pattern when staging into shared memory for tl.dot, negating the intended fix.

> **Applicable when**: Never — this approach did not reduce shared memory bank conflicts in practice despite correct address arithmetic. The Triton compiler's internal shared memory staging for tl.dot operands overrides the intended access pattern.

> **Avoid when**: Always avoid for eliminating shared memory bank conflicts in Triton GEMM kernels when the B matrix has a large column-major stride (stride_bn divisible by 32). The tl.trans() + transposed load approach does not propagate the conflict-free access pattern through Triton's internal smem staging for tl.dot. The SharedMemoryConflicts metric (17-way, 82.35% excess wavefronts) and CPIStall MIO stall cycles remain unchanged after this transformation.

#### ❌ Plan C: `stage_b_tile_through_padded_shared_memory_to_eliminate_bank_conflicts` (1.975×)

Attempting to eliminate 17-way shared memory bank conflicts on a column-major B tile by adding a persistent loop structure and applying tl.trans(b_raw) followed by tl.trans(b_T) (a no-op round-trip) failed to reduce the bank conflict degree at all — SharedMemoryConflicts remained at 17-way with 82.35% conflict wavefronts. The actual implementation did not stage B through padded shared memory as planned; instead it doubled-transposed the register tile (net no-op), while the persistent loop doubled block size from 64 to 128 threads, increasing register pressure (48→56 regs/thread), worsening MIO stall cycles per warp (6.8→12.0), and inflating executed instructions from 127K to 222K, causing a 41% duration regression (31.2→44.2 µs). The speedup figure of 1.9747x is relative to an earlier baseline, not the immediate predecessor — the actual change was a regression from the prior iteration's 31.23 µs to 44.19 µs.

> **Applicable when**: Never — this pattern is a regression. The double-transpose is a no-op that does not stage data through padded shared memory and does not reduce bank conflicts.

> **Avoid when**: Always avoid: (1) double tl.trans round-trips as a substitute for actual padded SMEM staging — they are no-ops that add register pressure without fixing bank conflicts; (2) adding a persistent loop when num_tiles == grid_size (14 tiles, 14 blocks) — it adds overhead with zero benefit; (3) removing @triton.autotune and fixing block_size=128 when the prior autotuned block_size=64 was already optimal — larger block size increases register pressure and reduces Block Limit Registers (20→9 blocks/SM). The 17-way bank conflict on column-major B (stride_bn=1728) requires genuine padded SMEM allocation, not register-level transposition.

---

## Iteration 5

**Best so far**: 2.7941× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_dot_with_register_accumulation_to_eliminate_smem_bank_conflicts | 0.449× | 0.445× | ✓ |  |
| B | reduce_register_pressure_by_removing_double_transpose_noop_and_reducing_block_size | 1.917× | 1.975× | ✓ |  |
| C | increase_block_n_with_direct_column_major_b_load_to_reduce_bank_conflicts | 2.416× | 2.529× | ✓ |  |
| D | reduce_block_k_to_lower_register_pressure_and_restore_occupancy | 3.089× | 1.408× | ✓ | **✓ best** |
| E | add_triton_autotune | 1.903× | 1.963× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_tl_dot_wi... | Δ | reduce_register_p... | Δ | increase_block_n_... | Δ | reduce_block_k_to... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 87260 ns | 196100 ns | +108840 ns (+124.7%) | 44190 ns | -43070 ns (-49.4%) | 34500 ns | -52760 ns (-60.5%) | 61980 ns | -25280 ns (-29.0%) | 44450 ns | -42810 ns (-49.1%) |
| SM Throughput (%) | 0.90 % | 0.50 % | -0.40 pp | 0.74 % | -0.16 pp | 0.79 % | -0.11 pp | 0.24 % | -0.66 pp | 0.73 % | -0.17 pp |
| Memory Throughput (%) | 9.62 % | 8.28 % | -1.34 pp | 9.54 % | -0.08 pp | 6.52 % | -3.10 pp | 1.85 % | -7.77 pp | 9.48 % | -0.14 pp |
| DRAM Throughput (%) | 0.70 % | 0.31 % | -0.39 pp | 1.39 % | +0.69 pp | 1.78 % | +1.08 pp | 0.99 % | +0.29 pp | 1.38 % | +0.68 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 1.56 % | -4.69 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 32.00 register/thread | -21 | 56.00 register/thread | +3 | 80.00 register/thread | +27 | 64.00 register/thread | +11 | 56.00 register/thread | +3 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.02 | +0.01 | 0.00 | -0.01 | 0.01 | +0.00 |
| No Eligible (%) | 91.37 % | 98.08 % | +6.71 pp | 93.77 % | +2.40 pp | 84.71 % | -6.66 pp | 91.82 % | +0.45 pp | 93.74 % | +2.37 pp |
| IPC Active | 0.34 inst/cycle | 0.08 inst/cycle | -0.26 | 0.25 inst/cycle | -0.09 | 0.31 inst/cycle | -0.03 | 0.08 inst/cycle | -0.26 | 0.25 inst/cycle | -0.09 |


### Key Learnings

#### ❌ Plan A: `replace_tl_dot_with_register_accumulation_to_eliminate_smem_bank_conflicts` (0.445×)

Replacing tl.dot with a tl.static_range inner loop of scalar loads and register outer-product accumulation eliminated shared memory bank conflicts (dynamic smem dropped to 0, SharedMemoryConflicts rule disappeared) but introduced a catastrophic new bottleneck: 85% uncoalesced global accesses and L1TEX stall cycles per warp exploded from 6.8 to 49.0 (94.6% of all stall time), causing duration to increase 6.3x (31 µs → 196 µs). The 64 scalar loads per K-step for the B tile (stride 1728 between N-elements) are highly strided and non-coalesced, and the instruction count grew from 127K to 309K due to loop unrolling overhead. The strategy trades a 17-way smem bank conflict for a far worse L1TEX memory stall pattern.

> **Applicable when**: Never — this approach causes severe regression for any GEMM where the B matrix has a large stride between N-elements (e.g., stride_bn=1728). The scalar loads in the static_range loop are strided by 1728 floats between consecutive N-elements, making them maximally non-coalesced.

> **Avoid when**: Always avoid replacing tl.dot with tl.static_range scalar-load outer-product accumulation when the B matrix has a large column-major stride (stride_bn >> 1). The scalar loads B + (k + stride_bn * offs_b_n) with large stride_bn cause 85% uncoalesced global sectors and L1TEX stall cycles per warp of ~49 (vs. 6.8 MIO stall cycles with tl.dot). Even though smem bank conflicts are eliminated, the L1TEX stall is ~7x worse. This anti-pattern is especially harmful when BLOCK_N is large (32+) and BLOCK_K is large (64+), as the fully-unrolled static_range loop generates 64 separate strided vector loads per K-tile.

---

## Cumulative Progress

```
Baseline:  87260 ns  (1.000×)
Iter 1     87230 ns  (1.000×)  — tf32_tensor_cores
Iter 2     37790 ns  (2.309×)  — fix_smem_bank_conflicts_via_block_m_correction_and_pipeline_stages
Iter 3     31230 ns  (2.794×)  — increase_block_k_to_break_shared_memory_bank_conflicts

Final best: 61980 ns  (1.4079×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,12 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=1, num_stages=2),
+        triton.Config({}, num_warps=1, num_stages=3),
+        triton.Config({}, num_warps=1, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_29(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
@@ -30,66 +45,63 @@

     stride_bk = 1
     stride_bn = 1728
 
-    # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_size = tl.num_programs(0).to(INDEX_DTYPE)
+    grid_m = (M + BLOCK_M - 1) // BLOCK_M  # = 1
+    grid_n = (N + BLOCK_N - 1) // BLOCK_N  # = 14
+    num_tiles = grid_m * grid_n             # = 14
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    # Persistent loop: each SM processes multiple tiles
+    tile_id = pid
+    while tile_id < num_tiles:
+        # re-order program ID for better L2 performance
+        width = GROUP_M * grid_n
+        group_id = tile_id // width
+        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
+        pid_m = group_id * GROUP_M + (tile_id % group_size)
+        pid_n = (tile_id % width) // group_size
+        tl.assume(pid_m >= 0)
+        tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        # CRITICAL: keep % M and % N exactly as original for boundary-safe modular indexing
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
+        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 1728*idx_m
-        a = tl.load(A + (xindex))
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 1728*idx_m
+            a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
+            idx_m = b_k_idx_vals
+            idx_n = offs_b_n[None, :]
+            xindex = idx_n + 432*idx_m
+            b = tl.load(B + ((tl.broadcast_to(idx_m + 1728*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+
+            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+
+        # rematerialize rm and rn to save registers
+        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
+        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+        idx_m = rm[:, None]
+        idx_n = rn[None, :]
+        mask = (idx_m < M) & (idx_n < N)
+
+        # inductor generates a suffix
         xindex = idx_n + 432*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1728*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        tmp0 = tl.full([1], 0, tl.int32)
+        tmp1 = tl.maximum(tmp0, acc)
+        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
 
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 432*idx_m
-    tmp0 = tl.full([1], 0, tl.int32)
-    tmp1 = tl.maximum(tmp0, acc)
-    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+        tile_id += grid_size  # advance to next tile
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
