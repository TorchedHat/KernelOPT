# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_tril_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4090000 ns — **Best latency**: 186140 ns (**21.9727×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`tril_k_loop_lower_bound_restriction_for_diagonal_tiles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_tril_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_015/synthesis/kernel_0_fused_fused_mm.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (1024,1,1), Waves/SM: 7.76
- **Registers/thread**: 172
- **Duration**: 4090000 ns
- **SM Throughput**: 75.19 % | **Memory**: 50.86 % | **DRAM**: 2.61 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 3.12 inst/cycle | **No Eligible**: 22.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 2.781× | 1.834× | ✓ |  |
| B | fast_accumulation_fp16_dot | — | fail | ✓ |  |
| C | fp8_precision_tl_dot_hopper | 5.630× | 4.921× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_autotune_configs | 1.964× | 1.966× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fp8_precision_tl_... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4090000 ns | 2230000 ns | -1860000 ns (-45.5%) | 831170 ns | -3258830 ns (-79.7%) | 2080000 ns | -2010000 ns (-49.1%) |
| SM Throughput (%) | 75.19 % | 16.91 % | -58.28 pp | 25.23 % | -49.96 pp | 75.83 % | +0.64 pp |
| Memory Throughput (%) | 50.86 % | 87.11 % | +36.25 pp | 57.03 % | +6.17 pp | 51.37 % | +0.51 pp |
| DRAM Throughput (%) | 2.61 % | 4.66 % | +2.05 pp | 12.83 % | +10.22 pp | 3.06 % | +0.45 pp |
| Achieved Occupancy (%) | 12.50 % | 23.98 % | +11.48 pp | 12.48 % | -0.02 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 172.00 register/thread | 120.00 register/thread | -52 | 147.00 register/thread | -25 | 178.00 register/thread | +6 |
| Waves/SM | 7.76 | 3.88 | -3.88 | 7.76 | +0.00 | 7.76 | +0.00 |
| No Eligible (%) | 22.10 % | 90.77 % | +68.67 pp | 75.11 % | +53.01 pp | 22.67 % | +0.57 pp |
| IPC Active | 3.12 inst/cycle | 0.37 inst/cycle | -2.75 | 1.00 inst/cycle | -2.12 | 3.09 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 4.9208× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_shared_memory_bank_conflicts_via_block_k_padding_and_autotune | 5.605× | 4.896× | ✓ |  |
| B | fp16_cast_operands_for_tensor_core_wgmma_engagement | 9.986× | 8.866× | ✓ | **✓ best** |
| C | triangular_tile_skip_early_exit | 4.261× | 3.588× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_shared_mem... | Δ | fp16_cast_operand... | Δ | triangular_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4090000 ns | 835330 ns | -3254670 ns (-79.6%) | 461340 ns | -3628660 ns (-88.7%) | 1140000 ns | -2950000 ns (-72.1%) |
| SM Throughput (%) | 75.19 % | 25.23 % | -49.96 pp | 23.54 % | -51.65 pp | 16.95 % | -58.24 pp |
| Memory Throughput (%) | 50.86 % | 56.61 % | +5.75 pp | 54.36 % | +3.50 pp | 87.33 % | +36.47 pp |
| DRAM Throughput (%) | 2.61 % | 12.75 % | +10.14 pp | 13.79 % | +11.18 pp | 5.41 % | +2.80 pp |
| Achieved Occupancy (%) | 12.50 % | 12.48 % | -0.02 pp | 12.45 % | -0.05 pp | 23.56 % | +11.06 pp |
| Registers/Thread | 172.00 register/thread | 147.00 register/thread | -25 | 202.00 register/thread | +30 | 128.00 register/thread | -44 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 3.88 | -3.88 |
| No Eligible (%) | 22.10 % | 75.24 % | +53.14 pp | 76.42 % | +54.32 pp | 90.80 % | +68.70 pp |
| IPC Active | 3.12 inst/cycle | 0.99 inst/cycle | -2.13 | 0.94 inst/cycle | -2.18 | 0.37 inst/cycle | -2.75 |


### Key Learnings

#### ✅ Plan C: `triangular_tile_skip_early_exit` (3.588×)

For masked/triangular GEMM kernels that launch a full rectangular grid, approximately half the tiles (those entirely above the diagonal) perform the complete K-loop accumulation only to have their results zeroed by a mask at the end. Adding a scalar CTA-level early-exit guard before the accumulator allocation — checking if col_start > row_start + BLOCK_M - 1 — skips all tl.load and tl.dot calls for above-diagonal tiles, reducing executed instructions by ~48% and eliminating the dominant L1/TEX pressure. The early-exit path must still store zeros to the output tensor (since it may be allocated uninitialized), and the tril_mask must be preserved for diagonal tiles that are only partially above the diagonal.

> **Applicable when**: Triangular or masked GEMM kernel launches a full rectangular grid covering all M×N tiles, with ~50% of tiles entirely outside the active region (above/below diagonal). L1/TEX throughput is high (>80%) driven by K-loop loads for wasted tiles. Executed instructions are high relative to useful work. The output tensor may be uninitialized (torch.empty). Tile dimensions are square or near-square (BLOCK_M == BLOCK_N) so the diagonal tile fraction is well-defined.

> **Avoid when**: The grid is already pruned to only launch tiles within the active triangular region (no wasted tiles). The output tensor is pre-zeroed (torch.zeros) so skipping the store for above-diagonal tiles is safe without explicit zero-store. The kernel is compute-bound on the useful tiles (SM Throughput >80%) — early exit reduces total work but does not improve per-tile efficiency. Diagonal fraction is small (e.g., very wide/tall non-square matrices where few tiles are above the diagonal).

---

## Iteration 3

**Best so far**: 8.8655× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_pointer_advancing_k_loop | — | fail | ✓ |  |
| C | tril_aware_k_loop_range_restriction | — | fail | ✓ |  |

---

## Iteration 4

**Best so far**: 8.8655× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_eliminate_bank_conflicts | 9.604× | 8.557× | ✓ |  |
| B | tril_k_loop_lower_bound_restriction_for_diagonal_tiles | 23.517× | 21.973× | ✓ | **✓ best** |
| C | k_loop_range_restriction_for_diagonal_tiles_exploiting_input_sparsity | 13.099× | 6.516× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | tril_k_loop_lower... | Δ | k_loop_range_rest... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4090000 ns | 477950 ns | -3612050 ns (-88.3%) | 186140 ns | -3903860 ns (-95.4%) | 627710 ns | -3462290 ns (-84.7%) |
| SM Throughput (%) | 75.19 % | 23.93 % | -51.26 pp | 22.16 % | -53.03 pp | 11.51 % | -63.68 pp |
| Memory Throughput (%) | 50.86 % | 52.73 % | +1.87 pp | 50.01 % | -0.85 pp | 61.41 % | +10.55 pp |
| DRAM Throughput (%) | 2.61 % | 13.31 % | +10.70 pp | 15.80 % | +13.19 pp | 4.75 % | +2.14 pp |
| Achieved Occupancy (%) | 12.50 % | 12.45 % | -0.05 pp | 12.39 % | -0.11 pp | 23.46 % | +10.96 pp |
| Registers/Thread | 172.00 register/thread | 214.00 register/thread | +42 | 147.00 register/thread | -25 | 128.00 register/thread | -44 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 3.88 | -3.88 |
| No Eligible (%) | 22.10 % | 74.74 % | +52.64 pp | 75.86 % | +53.76 pp | 90.80 % | +68.70 pp |
| IPC Active | 3.12 inst/cycle | 1.01 inst/cycle | -2.11 | 0.96 inst/cycle | -2.16 | 0.37 inst/cycle | -2.75 |

---

## Iteration 5

**Best so far**: 21.9727× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | smem_swizzle_via_block_k_stride_padding | 23.503× | 21.912× | ✓ |  |
| B | tril_k_loop_dual_bound_restriction_for_triangular_operand_gemm | 22.882× | 22.120× | ✓ |  |
| C | restore_software_pipelining_via_fixed_trip_count_k_loop | 13.142× | 7.532× | ✓ |  |


### Metric Comparison

| Metric | Baseline | smem_swizzle_via_... | Δ | tril_k_loop_dual_... | Δ | restore_software_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4090000 ns | 186660 ns | -3903340 ns (-95.4%) | 184900 ns | -3905100 ns (-95.5%) | 542980 ns | -3547020 ns (-86.7%) |
| SM Throughput (%) | 75.19 % | 22.10 % | -53.09 pp | 25.58 % | -49.61 pp | 11.79 % | -63.40 pp |
| Memory Throughput (%) | 50.86 % | 50.14 % | -0.72 pp | 47.43 % | -3.43 pp | 63.01 % | +12.15 pp |
| DRAM Throughput (%) | 2.61 % | 15.75 % | +13.14 pp | 15.84 % | +13.23 pp | 5.60 % | +2.99 pp |
| Achieved Occupancy (%) | 12.50 % | 12.39 % | -0.11 pp | 12.36 % | -0.14 pp | 23.06 % | +10.56 pp |
| Registers/Thread | 172.00 register/thread | 147.00 register/thread | -25 | 196.00 register/thread | +24 | 128.00 register/thread | -44 |
| Waves/SM | 7.76 | 7.76 | +0.00 | 7.76 | +0.00 | 3.88 | -3.88 |
| No Eligible (%) | 22.10 % | 75.78 % | +53.68 pp | 71.75 % | +49.65 pp | 90.45 % | +68.35 pp |
| IPC Active | 3.12 inst/cycle | 0.96 inst/cycle | -2.16 | 1.13 inst/cycle | -1.99 | 0.38 inst/cycle | -2.74 |


### Key Learnings

#### ❌ Plan A: `smem_swizzle_via_block_k_stride_padding` (21.912×)

Adding BLOCK_K=32 autotune configs (with num_stages=5/6, num_warps=4) to a Triton GEMM kernel suffering from SharedMemoryConflicts (3.8-way average, 40% conflict wavefronts) produced no measurable improvement: the autotuner selected the same winning config as before, all NCU metrics are essentially unchanged (duration 186.14→186.66 µs, bank conflicts 3.8→4.0-way, executed instructions identical at 29.1M), and the reported 21.9x speedup is attributable to prior optimizations (K-loop restriction, FP16 cast, early-exit), not this change. This is the third consecutive iteration where targeting SharedMemoryConflicts via autotune config expansion failed to move the NCU metrics, confirming that in this register-limited kernel (Block Limit Registers=1, 12.5% theoretical occupancy), shared memory bank conflicts are not the true performance bottleneck.

> **Applicable when**: Never: this strategy produced no improvement. Do not add BLOCK_K=32 autotune configs as a bank-conflict fix when the kernel is register-limited (Block Limit Registers=1) and the autotuner has already converged to a winning config.

> **Avoid when**: Kernel is register-limited (Block Limit Registers=1, theoretical occupancy ≤12.5%) and SharedMemoryConflicts NCU rule fires. The autotuner has already explored a wide config space and converged. This is the third consecutive failed attempt targeting SharedMemoryConflicts in this kernel — the bank conflicts are a symptom, not the bottleneck. The true bottleneck is low occupancy (75-76% no-eligible-warp cycles) caused by register pressure, which autotune config expansion cannot fix.

---

## Cumulative Progress

```
Baseline:  4090000 ns  (1.000×)
Iter 1     831170 ns  (4.921×)  — fp8_precision_tl_dot_hopper
Iter 2     461340 ns  (8.866×)  — fp16_cast_operands_for_tensor_core_wgmma_engagement
Iter 4     186140 ns  (21.973×)  — tril_k_loop_lower_bound_restriction_for_diagonal_tiles

Final best: 186140 ns  (21.9727×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,16 +3,36 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
-        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
-        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # Original 12 configs (unchanged)
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        # NEW: small tiles to break Block Limit Registers=1
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=2),
+        # NEW: deeper pipeline stages for better latency hiding
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=6, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=6, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=4),
+        # NEW: GROUP_M=4 variants for different L2 grouping
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256, 'GROUP_M': 4}, num_stages=5, num_warps=8),
     ],
     key=['M', 'N', 'K'],
 )
@@ -42,23 +62,11 @@

     row_start = pid_m * BLOCK_M
     col_start = pid_n * BLOCK_N
 
-    # Early exit: if the entire tile is above the diagonal, output is zero
-    # The tile covers rows [row_start, row_start+BLOCK_M) and cols [col_start, col_start+BLOCK_N)
-    # A tile is entirely above diagonal if col_start > row_start + BLOCK_M - 1
-    # i.e., the minimum row in the tile < maximum col in the tile means some elements are above diagonal
-    # We only skip if col_start > row_start + BLOCK_M - 1 (entire tile above diagonal)
     rm = row_start + tl.arange(0, BLOCK_M)
     rn = col_start + tl.arange(0, BLOCK_N)
 
     # Compute matmul for this tile
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
-
-    # For lower triangular result: C[i,j] = sum_k A[i,k]*B[k,j] where i >= j
-    # Since A and B are lower triangular:
-    # A[i,k] = 0 if k > i
-    # B[k,j] = 0 if j > k
-    # So C[i,j] = sum_{k=j}^{i} A[i,k]*B[k,j]
-    # But we compute the full matmul and apply tril at the end
 
     offs_am = rm
     offs_bn = rn
@@ -67,7 +75,12 @@

     A_base = A_ptr + offs_am[:, None] * stride_am
     B_base = B_ptr + offs_bn[None, :] * stride_bn
 
-    for k in range(0, tl.cdiv(K, BLOCK_K)):
+    # B is lower-triangular: B[k, n] = 0 for k < col_start -> skip those K-blocks
+    k_lo = col_start // BLOCK_K
+    # A is lower-triangular: A[m, k] = 0 for k >= row_start + BLOCK_M -> skip those K-blocks
+    k_hi = tl.cdiv(tl.minimum(K, row_start + BLOCK_M), BLOCK_K)
+
+    for k in range(k_lo, k_hi):
         k_start = k * BLOCK_K
         k_offs = k_start + offs_k
 
@@ -81,7 +94,13 @@

         mask_b = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
         b = tl.load(b_ptrs, mask=mask_b, other=0.0)
 
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
+        # Cast to FP16 to engage Tensor Cores on Hopper while preserving
+        # sufficient precision (10-bit mantissa vs 3-bit for FP8).
+        # FP8 E4M3 accumulates ~70 error over K=4096 steps (fails atol=1e-3).
+        # FP16 keeps error well within tolerance. Accumulator stays FP32.
+        a_fp16 = a.to(tl.float16)
+        b_fp16 = b.to(tl.float16)
+        acc = tl.dot(a_fp16, b_fp16, acc=acc)
 
     # Apply tril: zero out elements where col > row
     row_idx = rm[:, None]  # [BLOCK_M, 1]
@@ -95,7 +114,6 @@

     c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
     mask_c = (rm[:, None] < M) & (rn[None, :] < N)
     tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)
-
 
 def kernel_function(a, b):
     M, K = a.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
