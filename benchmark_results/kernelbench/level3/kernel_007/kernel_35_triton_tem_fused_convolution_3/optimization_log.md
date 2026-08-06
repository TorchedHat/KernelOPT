# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27870 ns — **Best latency**: 27550 ns (**1.0116×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_shared_memory_bank_conflicts_via_b_order_row_major`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_35`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_35_triton_tem_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62,1,1), Waves/SM: 0.05
- **Registers/thread**: 53
- **Duration**: 27870 ns
- **SM Throughput**: 3.59 % | **Memory**: 39.31 % | **DRAM**: 2.78 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.33 inst/cycle | **No Eligible**: 91.81 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_occupancy_and_bank_conflicts | 1.306× | 0.999× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_sizes | 1.291× | 0.999× | ✓ |  |
| D | autotune_tile_sizes_to_increase_grid_and_eliminate_bank_conflicts | 1.306× | 1.037× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27870 ns | 27900 ns | +30 ns (+0.1%) | 27900 ns | +30 ns (+0.1%) | 26880 ns | -990 ns (-3.6%) |
| SM Throughput (%) | 3.59 % | 3.58 % | -0.01 pp | 3.58 % | -0.01 pp | 3.01 % | -0.58 pp |
| Memory Throughput (%) | 39.31 % | 39.14 % | -0.17 pp | 39.14 % | -0.17 pp | 22.22 % | -17.09 pp |
| DRAM Throughput (%) | 2.78 % | 2.77 % | -0.01 pp | 2.78 % | +0.00 pp | 2.89 % | +0.11 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 53.00 register/thread | 53.00 register/thread | +0 | 53.00 register/thread | +0 | 64.00 register/thread | +11 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 | 0.03 | -0.02 |
| No Eligible (%) | 91.81 % | 91.80 % | -0.01 pp | 91.73 % | -0.08 pp | 85.56 % | -6.25 pp |
| IPC Active | 0.33 inst/cycle | 0.33 inst/cycle | +0.00 | 0.33 inst/cycle | +0.00 | 0.29 inst/cycle | -0.04 |

---

## Iteration 2

**Best so far**: 1.0368× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_shared_memory_bank_conflicts_via_b_load_transpose | 1.293× | 1.036× | ✓ |  |
| C | eliminate_shared_memory_bank_conflicts_via_block_k_padding | 1.304× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_shared_... | Δ | eliminate_shared_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27870 ns | 26910 ns | -960 ns (-3.4%) | 27900 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 3.59 % | 3.07 % | -0.52 pp | 3.59 % | +0.00 pp |
| Memory Throughput (%) | 39.31 % | 22.64 % | -16.67 pp | 39.20 % | -0.11 pp |
| DRAM Throughput (%) | 2.78 % | 2.88 % | +0.10 pp | 2.78 % | +0.00 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 64.00 register/thread | +11 | 53.00 register/thread | +0 |
| Waves/SM | 0.05 | 0.03 | -0.02 | 0.05 | +0.00 |
| No Eligible (%) | 91.81 % | 85.66 % | -6.15 pp | 91.81 % | +0.00 pp |
| IPC Active | 0.33 inst/cycle | 0.29 inst/cycle | -0.04 | 0.33 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0368× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_m_to_improve_grid_coverage_and_occupancy | 0.904× | 0.731× | ✓ |  |
| B | tma_loads_replace_scalar_pointer_arithmetic | 1.345× | 1.398× | ✓ |  |
| C | eliminate_tl_trans_and_increase_block_k_to_reduce_bank_conflicts | 1.292× | 0.999× | ✓ |  |
| D | tma_loads_via_block_ptr_for_transposed_b | 1.359× | 1.002× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_block_m_... | Δ | tma_loads_replace... | Δ | eliminate_tl_tran... | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27870 ns | 38110 ns | +10240 ns (+36.7%) | 19940 ns | -7930 ns (-28.5%) | 27900 ns | +30 ns (+0.1%) | 27810 ns | -60 ns (-0.2%) |
| SM Throughput (%) | 3.59 % | 4.30 % | +0.71 pp | 4.53 % | +0.94 pp | 3.59 % | +0.00 pp | 3.78 % | +0.19 pp |
| Memory Throughput (%) | 39.31 % | 31.90 % | -7.41 pp | 30.72 % | -8.59 pp | 39.26 % | -0.05 pp | 39.77 % | +0.46 pp |
| DRAM Throughput (%) | 2.78 % | 2.03 % | -0.75 pp | 3.89 % | +1.11 pp | 2.78 % | +0.00 pp | 2.79 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 64.00 register/thread | +11 | 66.00 register/thread | +13 | 53.00 register/thread | +0 | 52.00 register/thread | -1 |
| Waves/SM | 0.05 | 0.06 | +0.01 | 0.03 | -0.02 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 91.81 % | 90.08 % | -1.73 pp | 78.73 % | -13.08 pp | 91.81 % | +0.00 pp | 91.38 % | -0.43 pp |
| IPC Active | 0.33 inst/cycle | 0.39 inst/cycle | +0.06 | 0.43 inst/cycle | +0.10 | 0.33 inst/cycle | +0.00 | 0.34 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan A: `increase_block_m_to_improve_grid_coverage_and_occupancy` (0.849×)

This iteration attempted to increase BLOCK_M to 128 while simultaneously reverting from TMA block pointers back to scalar pointer arithmetic loads. The register count exploded from 66 to 164 registers/thread, collapsing theoretical occupancy from 43.75% to 18.75% and increasing elapsed cycles from 29855 to 49233 (64% more). Although Warp Cycles Per Issued Instruction improved from 4.68 to 2.45 (fewer stalls per instruction), the instruction count tripled from 715K to 2.32M due to the larger tile and the overhead of scalar address computation, resulting in a net 18% slowdown. The critical error was reverting TMA block pointers to scalar loads: the register pressure from computing per-thread addresses for a 128-row tile caused register spilling and destroyed the occupancy gains that motivated the change.

> **Applicable when**: Never beneficial — documented as anti-pattern. Do not revert TMA block pointers to scalar loads when increasing BLOCK_M.

> **Avoid when**: When TMA block pointers are already in use (from a prior successful optimization), do not revert to scalar pointer arithmetic even when increasing BLOCK_M. Scalar loads for large tiles (BLOCK_M=128) require per-thread address computation for 128 rows, causing register pressure to spike (66→164 registers/thread) and collapsing theoretical occupancy (43.75%→18.75%). The achieved occupancy remains unchanged (3.12%) because the grid is too small, but the instruction count triples. Always keep TMA block pointers when changing tile sizes on Hopper.

---

## Iteration 4

**Best so far**: 1.0368× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_m_and_warps_to_improve_occupancy_and_hide_mio_stalls | 0.875× | 0.849× | ✓ |  |
| B | eliminate_shared_memory_bank_conflicts_via_b_order_row_major | 1.362× | 1.012× | ✓ | **✓ best** |
| C | tma_block_ptr_loads_with_software_pipelining | 1.360× | 1.009× | ✓ |  |
| D | tma_block_ptr_loads_eliminate_trans_and_bank_conflicts | 0.926× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_m_... | Δ | eliminate_shared_... | Δ | tma_block_ptr_loa... | Δ | tma_block_ptr_loa... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27870 ns | 32830 ns | +4960 ns (+17.8%) | 27550 ns | -320 ns (-1.1%) | 27620 ns | -250 ns (-0.9%) | 30780 ns | +2910 ns (+10.4%) |
| SM Throughput (%) | 3.59 % | 8.93 % | +5.34 pp | 3.72 % | +0.13 pp | 3.78 % | +0.19 pp | 5.81 % | +2.22 pp |
| Memory Throughput (%) | 39.31 % | 29.98 % | -9.33 pp | 39.88 % | +0.57 pp | 39.81 % | +0.50 pp | 40.13 % | +0.82 pp |
| DRAM Throughput (%) | 2.78 % | 2.37 % | -0.41 pp | 2.81 % | +0.03 pp | 2.81 % | +0.03 pp | 2.52 % | -0.26 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 164.00 register/thread | +111 | 58.00 register/thread | +5 | 52.00 register/thread | -1 | 66.00 register/thread | +13 |
| Waves/SM | 0.05 | 0.08 | +0.03 | 0.06 | +0.01 | 0.05 | +0.00 | 0.07 | +0.02 |
| No Eligible (%) | 91.81 % | 59.27 % | -32.54 pp | 91.51 % | -0.30 pp | 91.38 % | -0.43 pp | 86.64 % | -5.17 pp |
| IPC Active | 0.33 inst/cycle | 0.81 inst/cycle | +0.48 | 0.34 inst/cycle | +0.01 | 0.34 inst/cycle | +0.01 | 0.53 inst/cycle | +0.20 |

---

## Iteration 5

**Best so far**: 1.0368× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tma_block_ptr_for_a_matrix_to_eliminate_scalar_load_stalls | 1.360× | 1.010× | ✓ |  |
| B | increase_block_k_to_reduce_bank_conflict_amortization | 0.731× | 0.702× | ✓ |  |
| C | increase_block_k_to_reduce_shared_memory_bank_conflicts | 1.342× | 0.970× | ✓ |  |
| D | restore_tma_block_ptr_loads_with_reduced_block_m | 1.016× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_block_ptr_for... | Δ | increase_block_k_... | Δ | increase_block_k_... | Δ | restore_tma_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27870 ns | 27580 ns | -290 ns (-1.0%) | 39710 ns | +11840 ns (+42.5%) | 28740 ns | +870 ns (+3.1%) | 27780 ns | -90 ns (-0.3%) |
| SM Throughput (%) | 3.59 % | 3.76 % | +0.17 pp | 8.20 % | +4.61 pp | 3.63 % | +0.04 pp | 3.78 % | +0.19 pp |
| Memory Throughput (%) | 39.31 % | 39.62 % | +0.31 pp | 39.37 % | +0.06 pp | 40.64 % | +1.33 pp | 39.78 % | +0.47 pp |
| DRAM Throughput (%) | 2.78 % | 2.81 % | +0.03 pp | 1.96 % | -0.82 pp | 2.70 % | -0.08 pp | 2.79 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 53.00 register/thread | 52.00 register/thread | -1 | 112.00 register/thread | +59 | 52.00 register/thread | -1 | 52.00 register/thread | -1 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.16 | +0.11 | 0.05 | +0.00 | 0.05 | +0.00 |
| No Eligible (%) | 91.81 % | 91.34 % | -0.47 pp | 81.38 % | -10.43 pp | 91.71 % | -0.10 pp | 91.38 % | -0.43 pp |
| IPC Active | 0.33 inst/cycle | 0.34 inst/cycle | +0.01 | 0.74 inst/cycle | +0.41 | 0.33 inst/cycle | +0.00 | 0.35 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan B: `increase_block_k_to_reduce_bank_conflict_amortization` (0.702×)

This iteration attempted to reduce shared memory bank conflict overhead by increasing BLOCK_K from 32 to 64 and simultaneously reverting from TMA block pointers to scalar pointer arithmetic loads with tl.trans(). The register count jumped from 58 to 112 registers/thread, and dynamic shared memory per block ballooned from 12 KB to 74 KB, collapsing theoretical occupancy from 50% to 18.75%. Although Warp Cycles Per Issued Instruction improved from 11.73 to 5.36 (fewer stalls per instruction), the instruction count tripled from 811K to 2.6M and elapsed cycles grew from 41K to 59K, yielding a 30% slowdown. The root cause is that reverting TMA block pointers to scalar loads (with tl.trans) for a larger BLOCK_K tile caused register pressure to spike and shared memory to balloon, destroying occupancy and negating any bank-conflict reduction benefit.

> **Applicable when**: Never beneficial as described — documented as anti-pattern. Do not combine BLOCK_K increase with reversion from TMA block pointers to scalar loads.

> **Avoid when**: When TMA block pointers are already in use, do not revert to scalar pointer arithmetic (with tl.trans for B) even when increasing BLOCK_K. Scalar loads for large tiles require per-thread address computation, causing register pressure to spike (58→112 registers/thread) and shared memory to balloon (12KB→74KB), collapsing theoretical occupancy (50%→18.75%). The achieved occupancy remains unchanged (6.25%) because the grid is too small, but instruction count triples. If BLOCK_K must be increased, keep TMA block pointers and only update block_shape.

---

## Cumulative Progress

```
Baseline:  27870 ns  (1.000×)
Iter 1     26880 ns  (1.037×)  — autotune_tile_sizes_to_increase_grid_and_eliminate_bank_conflicts

Final best: 27550 ns  (1.0116×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,21 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_35(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -57,24 +72,28 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),  # row-major: N contiguous in shared mem, eliminates 18-way bank conflict
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 480*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 16*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')  # [BLOCK_K, BLOCK_N]
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +107,6 @@

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
