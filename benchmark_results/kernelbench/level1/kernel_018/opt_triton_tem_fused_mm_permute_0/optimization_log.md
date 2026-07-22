# Kernel Optimization Log

## Summary

- **Kernel**: `matmul_transpose_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 6310000 ns — **Best latency**: 2160000 ns (**2.9213×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`small_tile_low_register_pressure_autotune_to_break_occupancy_deadlock`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `matmul_transpose_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_018/synthesis/kernel_0_triton_tem_fused_mm_permute_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 6310000 ns
- **SM Throughput**: 47.81 % | **Memory**: 65.32 % | **DRAM**: 1.25 %
- **Achieved Occupancy**: 12.49 %
- **IPC Active**: 2.01 inst/cycle | **No Eligible**: 49.73 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 5.260× | 2.921× | ✓ | **✓ best** |
| B | fix_transposed_access_pattern_via_tl_block_ptr_and_swizzled_smem | 0.990× | 0.725× | ✓ |  |
| C | eviction_policies_streaming_loads_stores | 1.000× | 1.002× | ✓ |  |
| D | reduce_register_pressure_via_smaller_tile_autotune_configs | 1.004× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_transposed_ac... | Δ | eviction_policies... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6310000 ns | 2160000 ns | -4150000 ns (-65.8%) | 8700000 ns | +2390000 ns (+37.9%) | 6300000 ns | -10000 ns (-0.2%) | 6310000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 47.81 % | 17.53 % | -30.28 pp | 35.24 % | -12.57 pp | 47.87 % | +0.06 pp | 47.87 % | +0.06 pp |
| Memory Throughput (%) | 65.32 % | 88.14 % | +22.82 pp | 78.63 % | +13.31 pp | 65.41 % | +0.09 pp | 65.41 % | +0.09 pp |
| DRAM Throughput (%) | 1.25 % | 4.06 % | +2.81 pp | 1.11 % | -0.14 pp | 1.26 % | +0.01 pp | 1.25 % | +0.00 pp |
| Achieved Occupancy (%) | 12.49 % | 12.27 % | -0.22 pp | 12.50 % | +0.01 pp | 12.49 % | +0.00 pp | 12.49 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 207.00 register/thread | -48 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 3.88 | +1.94 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 49.73 % | 92.53 % | +42.80 pp | 63.58 % | +13.85 pp | 49.97 % | +0.24 pp | 49.75 % | +0.02 pp |
| IPC Active | 2.01 inst/cycle | 0.30 inst/cycle | -1.71 | 1.46 inst/cycle | -0.55 | 2.01 inst/cycle | +0.00 | 2.01 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 2.9213× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | coalesced_loads_with_register_transpose_for_transposed_gemm | 5.257× | 2.935× | ✓ |  |
| B | algebraic_transpose_elimination_via_input_reorder | 1.517× | 1.331× | ✓ |  |
| D | revert_block_ptr_to_manual_pointer_arithmetic_for_transposed_gemm | 1.004× | 0.749× | ✓ |  |


### Metric Comparison

| Metric | Baseline | coalesced_loads_w... | Δ | algebraic_transpo... | Δ | revert_block_ptr_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6310000 ns | 2150000 ns | -4160000 ns (-65.9%) | 4740000 ns | -1570000 ns (-24.9%) | 8430000 ns | +2120000 ns (+33.6%) |
| SM Throughput (%) | 47.81 % | 17.54 % | -30.27 pp | 63.97 % | +16.16 pp | 36.25 % | -11.56 pp |
| Memory Throughput (%) | 65.32 % | 88.19 % | +22.87 pp | 37.89 % | -27.43 pp | 81.01 % | +15.69 pp |
| DRAM Throughput (%) | 1.25 % | 4.08 % | +2.83 pp | 1.41 % | +0.16 pp | 1.15 % | -0.10 pp |
| Achieved Occupancy (%) | 12.49 % | 12.27 % | -0.22 pp | 12.50 % | +0.01 pp | 12.50 % | +0.01 pp |
| Registers/Thread | 255.00 register/thread | 207.00 register/thread | -48 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 3.88 | +1.94 |
| No Eligible (%) | 49.73 % | 92.53 % | +42.80 pp | 33.78 % | -15.95 pp | 62.49 % | +12.76 pp |
| IPC Active | 2.01 inst/cycle | 0.30 inst/cycle | -1.71 | 2.65 inst/cycle | +0.64 | 1.50 inst/cycle | -0.51 |


### Key Learnings

#### ✅ Plan A: `coalesced_loads_with_register_transpose_for_transposed_gemm` (2.935×)

Swapping the index layout in pointer arithmetic so that the warp-parallel dimension aligns with the unit-stride axis (coalesced), then applying tl.trans() to reinterpret the register tile layout, is a zero-cost transformation in Triton on Hopper/Ampere. However, in this case the profiling metrics are essentially unchanged before and after: L1/TEX throughput remains at 91.1%, excessive global sectors remain at 201M, excessive shared wavefronts remain at 253M, and duration only drops from 2.16ms to 2.15ms. The reported 2.93x speedup appears to reflect a comparison against a much earlier baseline (pre-TF32), not against the immediate prior kernel, meaning the tl.trans() index swap itself contributed negligible improvement when the kernel was already in a memory-bound state with the same access pattern characteristics.

> **Applicable when**: Triton kernel computing C = A.T @ B.T (or any transposed GEMM) where NCU reports UncoalescedGlobalAccess with >50% excessive sectors AND the kernel is not already L1/TEX saturated (L1/TEX throughput < 80%). The tl.trans() approach is zero-cost on Hopper/Ampere and avoids the shared memory blowup of tl.make_block_ptr.

> **Avoid when**: When the kernel is already L1/TEX saturated (L1/TEX Cache Throughput >= 88%) and Mem Busy >= 88%: in this regime, the access pattern change does not reduce the excessive sectors or wavefronts as reported by NCU, and the speedup is negligible (<1%). The bottleneck in this state is structural (low occupancy, register pressure, partial waves) rather than the load index layout.

#### ✅ Plan B: `algebraic_transpose_elimination_via_input_reorder` (1.331×)

Exploiting the identity A.T @ B.T = (B @ A).T to reorder operands and achieve row-major tile access eliminated shared memory bank conflicts and uncoalesced global accesses (L1/TEX throughput dropped from 91% to 39%, warp cycles per instruction dropped from 26.3 to 3.0), but the kernel became severely compute-bound (SM throughput rose from 17.5% to 64%) and executed ~21x more instructions (110M → 2.4B). The net result was a 1.33x speedup in wall-clock time, but the kernel was already in a memory-bound state from a prior TF32 optimization that reduced instruction count dramatically — the algebraic reorder undid that instruction reduction by forcing scalar FP32 paths (allow_tf32=False) while the prior state used TF32 tensor cores. The strategy is only beneficial when the kernel is genuinely bottlenecked by uncoalesced shared/global memory access AND is not already using tensor cores to reduce instruction count.

> **Applicable when**: Kernel computing C = A.T @ B.T is bottlenecked by uncoalesced shared/global memory access (UncoalescedSharedAccess >50% excess wavefronts, UncoalescedGlobalAccess >50% excess sectors) AND is NOT already using tensor cores (allow_tf32=True or equivalent). The kernel must be memory-bound with low compute utilization (<30% SM throughput) due to access pattern inefficiency rather than instruction throughput.

> **Avoid when**: The kernel is already using TF32 tensor cores (allow_tf32=True) which dramatically reduces instruction count (~21x). Applying algebraic reorder while keeping allow_tf32=False will shift the bottleneck from memory to compute and increase instruction count by ~21x, causing net regression in wall-clock time even though memory metrics improve. Do not apply when the prior optimization already achieved low instruction count via tensor cores.

---

## Iteration 3

**Best so far**: 2.9213× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | small_tile_low_register_pressure_autotune_to_break_occupancy_deadlock | 5.208× | 2.921× | ✓ |  |
| B | reduce_register_pressure_via_restore_allow_tf32_and_eliminate_partial_wave | 4.903× | 2.868× | ✓ |  |
| C | restore_tensor_cores_with_small_tile_autotune_configs | 3.259× | 2.921× | ✓ |  |


### Metric Comparison

| Metric | Baseline | small_tile_low_re... | Δ | reduce_register_p... | Δ | restore_tensor_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 6310000 ns | 2160000 ns | -4150000 ns (-65.8%) | 2200000 ns | -4110000 ns (-65.1%) | 2160000 ns | -4150000 ns (-65.8%) |
| SM Throughput (%) | 47.81 % | 17.53 % | -30.28 pp | 17.11 % | -30.70 pp | 17.48 % | -30.33 pp |
| Memory Throughput (%) | 65.32 % | 88.15 % | +22.83 pp | 87.02 % | +21.70 pp | 87.86 % | +22.54 pp |
| DRAM Throughput (%) | 1.25 % | 4.11 % | +2.86 pp | 4.82 % | +3.57 pp | 4.20 % | +2.95 pp |
| Achieved Occupancy (%) | 12.49 % | 12.27 % | -0.22 pp | 11.81 % | -0.68 pp | 12.27 % | -0.22 pp |
| Registers/Thread | 255.00 register/thread | 207.00 register/thread | -48 | 211.00 register/thread | -44 | 207.00 register/thread | -48 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 49.73 % | 92.53 % | +42.80 pp | 92.72 % | +42.99 pp | 92.53 % | +42.80 pp |
| IPC Active | 2.01 inst/cycle | 0.30 inst/cycle | -1.71 | 0.29 inst/cycle | -1.72 | 0.30 inst/cycle | -1.71 |


### Key Learnings

#### ❌ Plan A: `small_tile_low_register_pressure_autotune_to_break_occupancy_deadlock` (2.921×)

Appending small-tile autotune configs (BLOCK_M in {16,32}, BLOCK_N in {32,64}, BLOCK_K in {32,64}, num_warps in {2,4}) to a Triton GEMM kernel that is register-bound (207 regs/thread, Block Limit Registers=2, 12.5% theoretical occupancy) produced zero measurable improvement: all metrics are statistically identical before and after (duration 2.15ms vs 2.16ms, registers/thread still 207, Block Limit Registers still 2, eligible warps/scheduler still 0.09). The autotune framework selected the same large-tile config as before, indicating the small-tile configs either compiled to the same register count or were slower in wall-clock time due to more K-loop iterations. When the Triton compiler does not reduce register pressure for small tiles (possibly due to loop overhead, pipelining, or accumulator register allocation), adding small-tile configs to the autotune set has no effect on occupancy or latency.

> **Applicable when**: Never effective in this form — when the Triton compiler allocates the same number of registers regardless of tile size (due to pipeline overhead, loop unrolling, or accumulator sizing), adding small-tile configs to the autotune set does not reduce register pressure or improve occupancy.

> **Avoid when**: When the kernel is already register-bound (Block Limit Registers = 2, Theoretical Occupancy = 12.5%, registers/thread >= 200) and all profiling metrics are saturated (L1/TEX >= 88%, Mem Busy >= 88%, eligible warps/scheduler <= 0.09): adding small-tile autotune configs will not be selected by autotune if they are slower in wall-clock time, and will not reduce register pressure if the Triton compiler allocates registers based on pipeline depth rather than accumulator size. The autotune framework empirically confirms the large-tile config remains optimal.

---

## Cumulative Progress

```
Baseline:  6310000 ns  (1.000×)
Iter 1     2160000 ns  (2.921×)  — tf32_tensor_cores

Final best: 2160000 ns  (2.9213×)
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
@@ -13,6 +14,15 @@

         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
         triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -60,30 +70,20 @@

     # Initialize accumulator
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
     
-    # A has shape [K, M]: A.T[m, k] = A[k, m]
-    # For a tile: A_tile[BLOCK_M, BLOCK_K] where A_tile[i, j] = A[offs_k[j], offs_m[i]]
-    # Pointer: A_ptr + offs_k[j] * stride_a_row + offs_m[i] * stride_a_col
-    
-    # B has shape [N, K]: B.T[k, n] = B[n, k]
-    # For a tile: B_tile[BLOCK_K, BLOCK_N] where B_tile[j, l] = B[offs_n[l], offs_k[j]]
-    # Pointer: B_ptr + offs_n[l] * stride_b_row + offs_k[j] * stride_b_col
-    
     for k in range(0, tl.cdiv(K, BLOCK_K)):
         k_offs = k * BLOCK_K + offs_k
         
-        # Load A tile: shape [BLOCK_M, BLOCK_K]
-        # A.T[offs_m, k_offs] = A[k_offs, offs_m]
-        a_ptrs = A_ptr + k_offs[None, :] * stride_a_row + offs_m[:, None] * stride_a_col
-        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
-        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
-        
-        # Load B tile: shape [BLOCK_K, BLOCK_N]
-        # B.T[k_offs, offs_n] = B[offs_n, k_offs]
-        b_ptrs = B_ptr + offs_n[None, :] * stride_b_row + k_offs[:, None] * stride_b_col
-        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)
-        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
-        
-        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
+        # Load A as [BLOCK_K, BLOCK_M]: consecutive threads read consecutive M-elements (stride_a_col=1)
+        a_ptrs = A_ptr + k_offs[:, None] * stride_a_row + offs_m[None, :] * stride_a_col
+        mask_a = (k_offs[:, None] < K) & (offs_m[None, :] < M)
+        a = tl.trans(tl.load(a_ptrs, mask=mask_a, other=0.0))  # load [BLOCK_K,BLOCK_M], trans -> [BLOCK_M,BLOCK_K]
+
+        # Load B as [BLOCK_N, BLOCK_K]: consecutive threads read consecutive K-elements (stride_b_col=1)
+        b_ptrs = B_ptr + offs_n[:, None] * stride_b_row + k_offs[None, :] * stride_b_col
+        mask_b = (offs_n[:, None] < N) & (k_offs[None, :] < K)
+        b = tl.trans(tl.load(b_ptrs, mask=mask_b, other=0.0))  # load [BLOCK_N,BLOCK_K], trans -> [BLOCK_K,BLOCK_N]
+
+        acc = tl.dot(a, b, acc=acc, allow_tf32=True)
     
     # Store result
     offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -91,7 +91,6 @@

     c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
     mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
     tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)
-
 
 def kernel_function(a, b):
     """
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
