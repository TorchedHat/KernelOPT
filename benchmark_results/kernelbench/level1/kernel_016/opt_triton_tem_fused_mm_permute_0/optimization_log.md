# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_transpose_a_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3880000 ns — **Best latency**: 6180000 ns (**0.6278×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_transpose_a_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_016/synthesis/kernel_0_triton_tem_fused_mm_permute_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2048,1,1), Waves/SM: 2.59
- **Registers/thread**: 72
- **Duration**: 3880000 ns
- **SM Throughput**: 81.58 % | **Memory**: 71.85 % | **DRAM**: 4.59 %
- **Achieved Occupancy**: 33.88 %
- **IPC Active**: 3.37 inst/cycle | **No Eligible**: 15.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_larger_tiles_and_more_warps | 1.017× | 1.000× | ✓ |  |
| B | expand_autotune_configs_reduce_smem_pressure | 1.006× | 0.628× | ✓ |  |
| C | reduce_shared_memory_pressure_and_bank_conflicts_via_autotune_expansion | 1.019× | 0.990× | ✓ | **✓ best** |
| D | eliminate_smem_bank_conflicts_via_padding_and_expand_autotune | 1.017× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | expand_autotune_c... | Δ | reduce_shared_mem... | Δ | eliminate_smem_ba... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3880000 ns | 3880000 ns | +0 ns (+0.0%) | 6180000 ns | +2300000 ns (+59.3%) | 3920000 ns | +40000 ns (+1.0%) | 3920000 ns | +40000 ns (+1.0%) |
| SM Throughput (%) | 81.58 % | 81.58 % | +0.00 pp | 72.16 % | -9.42 pp | 80.60 % | -0.98 pp | 81.55 % | -0.03 pp |
| Memory Throughput (%) | 71.85 % | 71.85 % | +0.00 pp | 92.21 % | +20.36 pp | 71.13 % | -0.72 pp | 70.98 % | -0.87 pp |
| DRAM Throughput (%) | 4.59 % | 4.60 % | +0.01 pp | 4.39 % | -0.20 pp | 4.29 % | -0.30 pp | 4.61 % | +0.02 pp |
| Achieved Occupancy (%) | 33.88 % | 33.88 % | +0.00 pp | 53.98 % | +20.10 pp | 23.64 % | -10.24 pp | 33.79 % | -0.09 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 40.00 register/thread | -32 | 72.00 register/thread | +0 | 80.00 register/thread | +8 |
| Waves/SM | 2.59 | 2.59 | +0.00 | 6.90 | +4.31 | 3.88 | +1.29 | 2.59 | +0.00 |
| No Eligible (%) | 15.82 % | 15.81 % | -0.01 pp | 28.19 % | +12.37 pp | 16.90 % | +1.08 pp | 15.86 % | +0.04 pp |
| IPC Active | 3.37 inst/cycle | 3.37 inst/cycle | +0.00 | 2.87 inst/cycle | -0.50 | 3.32 inst/cycle | -0.05 | 3.37 inst/cycle | +0.00 |


### Key Learnings

#### ❌ Plan B: `expand_autotune_configs_reduce_smem_pressure` (0.628×)

Adding smaller tile configs (BLOCK_M/N=32) and larger BLOCK_K=128 with more pipeline stages (num_stages=5) to the autotune search space increased theoretical occupancy from 37.5% to 56.25% and grid size from 2048 to 8192, but caused a 37% regression. The autotuner selected a config that shifted the bottleneck from compute-bound (SM throughput 81.6%, Warp Cycles/Inst 6.44) to memory-bound (Mem Busy 92.2%, Max Bandwidth 87.9%, Warp Cycles/Inst 12.03), indicating the smaller tiles reduced arithmetic intensity below the roofline threshold. When a kernel is already compute-bound with healthy L2 hit rates (82.6%), adding smaller tile configs risks the autotuner selecting a tile size with insufficient reuse, converting a compute-bound kernel into a memory-bandwidth-bound one.

> **Applicable when**: Never: this strategy caused regression. Avoid when kernel is already compute-bound (SM throughput >80%, FMA pipeline >70%) with healthy L2 hit rates (>80%) and low warp stall cycles (<7 cycles/inst).

> **Avoid when**: Kernel is compute-bound (SM SOL >80%, FMA utilization >70%) with good L2 hit rate (>80%) and low warp cycles per instruction (<8). Adding smaller tile configs to autotune search space in this regime risks the autotuner selecting a memory-bandwidth-bound configuration, increasing warp stall cycles from ~6 to ~12 and doubling kernel duration.

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_num_stages_and_num_warps_tuning | 1.005× | 1.000× | ✓ |  |
| B | eliminate_a_tile_bank_conflicts_via_smem_padding_constexpr | 0.958× | 0.968× | ✓ |  |
| C | convert_b_to_block_ptr_and_reduce_register_pressure | 0.993× | 0.987× | ✓ |  |
| D | pretranspose_a_for_coalesced_l1_access | 1.019× | 0.628× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | eliminate_a_tile_... | Δ | convert_b_to_bloc... | Δ | pretranspose_a_fo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3880000 ns | 3880000 ns | +0 ns (+0.0%) | 4010000 ns | +130000 ns (+3.4%) | 3930000 ns | +50000 ns (+1.3%) | 6180000 ns | +2300000 ns (+59.3%) |
| SM Throughput (%) | 81.58 % | 81.59 % | +0.01 pp | 78.84 % | -2.74 pp | 81.45 % | -0.13 pp | 72.15 % | -9.43 pp |
| Memory Throughput (%) | 71.85 % | 71.86 % | +0.01 pp | 68.71 % | -3.14 pp | 70.82 % | -1.03 pp | 92.21 % | +20.36 pp |
| DRAM Throughput (%) | 4.59 % | 4.61 % | +0.02 pp | 3.83 % | -0.76 pp | 4.63 % | +0.04 pp | 4.39 % | -0.20 pp |
| Achieved Occupancy (%) | 33.88 % | 33.88 % | +0.00 pp | 12.45 % | -21.43 pp | 28.74 % | -5.14 pp | 53.98 % | +20.10 pp |
| Registers/Thread | 72.00 register/thread | 72.00 register/thread | +0 | 86.00 register/thread | +14 | 92.00 register/thread | +20 | 40.00 register/thread | -32 |
| Waves/SM | 2.59 | 2.59 | +0.00 | 7.76 | +5.17 | 3.10 | +0.51 | 6.90 | +4.31 |
| No Eligible (%) | 15.82 % | 15.81 % | -0.01 pp | 18.66 % | +2.84 pp | 15.94 % | +0.12 pp | 28.19 % | +12.37 pp |
| IPC Active | 3.37 inst/cycle | 3.37 inst/cycle | +0.00 | 3.25 inst/cycle | -0.12 | 3.36 inst/cycle | -0.01 | 2.87 inst/cycle | -0.50 |


### Key Learnings

#### ❌ Plan D: `pretranspose_a_for_coalesced_l1_access` (0.628×)

Pre-transposing A from (K, M) to (M, K) contiguous layout before the kernel launch was intended to convert strided L1 misses into coalesced accesses, but instead caused a 37% regression. The profiling after shows Mem Busy rose from 71.9% to 92.2%, Max Bandwidth from 70% to 87.9%, and Warp Cycles per Issued Instruction doubled from 6.44 to 12.03, indicating the kernel shifted from compute-bound to severely memory-bound. The autotuner selected a smaller-tile configuration (grid 2048→8192, waves 2.59→6.9) with the new config set, which reduced arithmetic intensity and eliminated the compute-bound regime that the baseline relied on, while the pre-transpose overhead added latency without delivering the expected L1 hit rate improvement.

> **Applicable when**: Never: this strategy caused regression in this context. Avoid when the baseline kernel is already compute-bound (SM throughput >80%, FMA >70%) with good L2 hit rates (>80%), because the autotuner may select a smaller-tile config from the new config set that destroys arithmetic intensity.

> **Avoid when**: Kernel is compute-bound (SM SOL >80%) with L2 hit rate >80% and low warp stall cycles (<7 cycles/inst). In this regime, pre-transposing A changes the stride layout but the autotuner simultaneously selects a smaller-tile config, shifting the bottleneck from compute to memory bandwidth (Mem Busy 92%, Warp Cycles/Inst 12). The pre-transpose overhead and config set change together cause a 37% regression. Also avoid when the config set is changed simultaneously, as the autotuner interaction confounds the effect of the access pattern change.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_tail_wave_via_grid_aligned_tile_configs | 1.017× | 0.987× | ✓ |  |
| C | convert_b_pointer_arithmetic_to_block_ptr_reduce_register_pressure | 0.961× | 0.975× | ✓ |  |
| D | restrict_autotune_to_large_tile_compute_bound_configs | 1.017× | 0.834× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tail_wa... | Δ | convert_b_pointer... | Δ | restrict_autotune... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3880000 ns | 3930000 ns | +50000 ns (+1.3%) | 3980000 ns | +100000 ns (+2.6%) | 4650000 ns | +770000 ns (+19.8%) |
| SM Throughput (%) | 81.58 % | 81.46 % | -0.12 pp | 79.21 % | -2.37 pp | 67.41 % | -14.17 pp |
| Memory Throughput (%) | 71.85 % | 70.83 % | -1.02 pp | 69.25 % | -2.60 pp | 60.17 % | -11.68 pp |
| DRAM Throughput (%) | 4.59 % | 4.63 % | +0.04 pp | 3.85 % | -0.74 pp | 3.32 % | -1.27 pp |
| Achieved Occupancy (%) | 33.88 % | 28.74 % | -5.14 pp | 12.47 % | -21.41 pp | 12.46 % | -21.42 pp |
| Registers/Thread | 72.00 register/thread | 92.00 register/thread | +20 | 78.00 register/thread | +6 | 72.00 register/thread | +0 |
| Waves/SM | 2.59 | 3.10 | +0.51 | 7.76 | +5.17 | 7.76 | +5.17 |
| No Eligible (%) | 15.82 % | 15.94 % | +0.12 pp | 18.29 % | +2.47 pp | 30.62 % | +14.80 pp |
| IPC Active | 3.37 inst/cycle | 3.36 inst/cycle | -0.01 | 3.27 inst/cycle | -0.10 | 2.78 inst/cycle | -0.59 |


### Key Learnings

#### ❌ Plan D: `restrict_autotune_to_large_tile_compute_bound_configs` (0.834×)

Replacing a broad autotune config list with only large-tile configs (BLOCK_M/N >= 64, BLOCK_K <= 64) caused a 17% regression. The autotuner selected a config with very large shared memory per block (98.3 KB vs 32.8 KB baseline), which collapsed theoretical occupancy from 37.5% to 12.5% (Block Limit Shared Mem dropped from 6 to 2 blocks per SM). Although Warp Cycles per Issued Instruction fell from 6.44 to 2.87 (suggesting less stall per instruction), the severe occupancy reduction meant fewer active warps to hide latency, and SM throughput dropped from 81.6% to 67.4%, increasing duration from 3.88ms to 4.65ms. When restricting autotune configs to large tiles, the autotuner may select a tile size whose shared memory footprint exceeds the SM's capacity to sustain multiple concurrent blocks, causing an occupancy cliff that outweighs any arithmetic intensity benefit.

> **Applicable when**: Never: this strategy caused regression. Avoid when the baseline kernel already achieves compute-bound operation (SM throughput >80%) with moderate shared memory usage (Block Limit Shared Mem >= 6) and theoretical occupancy ~37%.

> **Avoid when**: Kernel is compute-bound (SM SOL >80%) with Block Limit Shared Mem >= 6 and theoretical occupancy ~37%. Restricting autotune to large-tile-only configs risks the autotuner selecting a config with excessive shared memory per block (>64 KB), collapsing Block Limit Shared Mem to 2 and theoretical occupancy to 12.5%, which reduces SM throughput from 81% to 67% despite lower warp stall cycles. Also avoid when the kernel has already been modified (e.g., pre-transposed A) such that the config set interacts with a changed memory access pattern.

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | prune_register_heavy_512_tile_configs_to_restore_occupancy | 1.017× | 0.987× | ✓ |  |
| B | convert_c_store_to_block_ptr_reduce_register_pressure | 0.968× | 0.789× | ✓ |  |
| C | smem_budget_constrained_autotune_to_restore_occupancy | 0.953× | 0.975× | ✓ |  |
| D | add_triton_autotune | 1.006× | 0.987× | ✓ |  |


### Metric Comparison

| Metric | Baseline | prune_register_he... | Δ | convert_c_store_t... | Δ | smem_budget_const... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3880000 ns | 3930000 ns | +50000 ns (+1.3%) | 4920000 ns | +1040000 ns (+26.8%) | 3980000 ns | +100000 ns (+2.6%) | 3930000 ns | +50000 ns (+1.3%) |
| SM Throughput (%) | 81.58 % | 81.45 % | -0.13 pp | 64.13 % | -17.45 pp | 78.42 % | -3.16 pp | 81.62 % | +0.04 pp |
| Memory Throughput (%) | 71.85 % | 70.82 % | -1.03 pp | 56.02 % | -15.83 pp | 69.85 % | -2.00 pp | 70.97 % | -0.88 pp |
| DRAM Throughput (%) | 4.59 % | 4.64 % | +0.05 pp | 3.12 % | -1.47 pp | 4.41 % | -0.18 pp | 4.63 % | +0.04 pp |
| Achieved Occupancy (%) | 33.88 % | 28.74 % | -5.14 pp | 12.43 % | -21.45 pp | 33.55 % | -0.33 pp | 28.74 % | -5.14 pp |
| Registers/Thread | 72.00 register/thread | 92.00 register/thread | +20 | 80.00 register/thread | +8 | 77.00 register/thread | +5 | 92.00 register/thread | +20 |
| Waves/SM | 2.59 | 3.10 | +0.51 | 7.76 | +5.17 | 2.59 | +0.00 | 3.10 | +0.51 |
| No Eligible (%) | 15.82 % | 15.94 % | +0.12 pp | 33.93 % | +18.11 pp | 19.07 % | +3.25 pp | 15.94 % | +0.12 pp |
| IPC Active | 3.37 inst/cycle | 3.36 inst/cycle | -0.01 | 2.64 inst/cycle | -0.73 | 3.24 inst/cycle | -0.13 | 3.36 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan B: `convert_c_store_to_block_ptr_reduce_register_pressure` (0.789×)

Converting the C-store epilogue from manual pointer arithmetic (offs_cm, offs_cn, c_ptrs, c_mask) to tl.make_block_ptr + tl.store with boundary_check reduced registers per thread from 92 to 80, but simultaneously the autotune config set was changed and the autotuner selected a config with 98.3 KB SMEM/block (vs 32.8 KB baseline), collapsing Block Limit Shared Mem from 5 to 2 and theoretical occupancy from 31.25% to 12.5%. Despite the register reduction, SM throughput dropped from 81.5% to 64.1% and duration increased from 3.93ms to 4.92ms. The root cause is that the autotune config list was modified alongside the epilogue change, allowing the autotuner to select a high-SMEM config that created a severe occupancy cliff — the register savings were irrelevant because SMEM became the binding limiter at 2 blocks/SM.

> **Applicable when**: Never: this strategy caused regression. Avoid when the autotune config set is changed simultaneously with the epilogue refactor, as the autotuner may select a high-SMEM config that collapses occupancy regardless of register savings.

> **Avoid when**: The autotune config list is modified at the same time as the epilogue change. Even if register pressure is reduced (92→80 regs/thread), if the new config set allows the autotuner to select a tile with SMEM > 64 KB/block, Block Limit Shared Mem collapses to 2 and theoretical occupancy drops to 12.5%, negating any register benefit. Also avoid when the kernel is already compute-bound (SM SOL >80%) with Block Limit Shared Mem >= 5 — the SMEM constraint is the binding limiter and register changes alone cannot improve occupancy.

---

## Cumulative Progress

```
Baseline:  3880000 ns  (1.000×)

Final best: 6180000 ns  (0.6278×)
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
@@ -17,6 +18,19 @@

         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        # New configs: BLOCK_K=128 and larger num_warps=8 variants
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
+        # New large-tile configs: single-wave grid sizes to eliminate partial-wave tail
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=16),
+        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 512, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 512, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=16),
+        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=16),
     ],
     key=['M', 'N', 'K'],
 )
@@ -65,40 +79,51 @@

     # Row/col offsets for this tile
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    offs_k = tl.arange(0, BLOCK_K)
-    
-    # Pointers into A: A has shape (K, M), stride (stride_ak, stride_am)
-    # A^T[m, k] = A[k, m]
-    # For tile: A^T[offs_m, offs_k] = A[offs_k, offs_m]
-    # a_ptrs[bm, bk] = a_ptr + offs_k[bk] * stride_ak + offs_m[bm] * stride_am
-    a_ptrs = a_ptr + (offs_k[None, :] * stride_ak + offs_m[:, None] * stride_am)
-    
-    # Pointers into B: B has shape (K, N), stride (stride_bk, stride_bn)
-    # b_ptrs[bk, bn] = b_ptr + offs_k[bk] * stride_bk + offs_n[bn] * stride_bn
-    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
+    
+    # Use tl.make_block_ptr for A tile with order=(1,0) to eliminate bank conflicts.
+    # A has shape (K, M) with strides (stride_ak, stride_am).
+    # We view A^T as shape (M, K) with strides (stride_am, stride_ak).
+    # order=(1, 0) means the fast-varying dimension is axis=1 (K), which is row-major
+    # in the (M, K) view — this tells Triton to use a bank-conflict-free smem layout.
+    a_block_ptr = tl.make_block_ptr(
+        base=a_ptr,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    
+    # Use tl.make_block_ptr for B tile to reduce register pressure.
+    # B has shape (K, N) with strides (stride_bk, stride_bn).
+    # order=(1, 0) means fast-varying dimension is axis=1 (N), row-major.
+    b_block_ptr = tl.make_block_ptr(
+        base=b_ptr,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
     
     # Accumulator
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     
     for k in range(0, tl.cdiv(K, BLOCK_K)):
-        # Boundary masks
-        k_offs = k * BLOCK_K + offs_k
-        mask_k = k_offs < K
-        
-        # Load A tile: shape (BLOCK_M, BLOCK_K)
-        a_mask = (offs_m[:, None] < M) & (mask_k[None, :])
-        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
-        
-        # Load B tile: shape (BLOCK_K, BLOCK_N)
-        b_mask = (mask_k[:, None]) & (offs_n[None, :] < N)
-        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
+        # Load A tile: shape (BLOCK_M, BLOCK_K) using block pointer
+        # boundary_check=(0, 1) handles M and K not divisible by block sizes
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        
+        # Load B tile: shape (BLOCK_K, BLOCK_N) using block pointer
+        # boundary_check=(0, 1) handles K and N not divisible by block sizes
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
         
         # Accumulate
         acc = tl.dot(a, b, acc=acc, allow_tf32=False)
         
         # Advance pointers
-        a_ptrs += BLOCK_K * stride_ak
-        b_ptrs += BLOCK_K * stride_bk
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
     
     # Write output
     offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -106,7 +131,6 @@

     c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
     tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)
-
 
 def kernel_function(a, b):
     """
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
