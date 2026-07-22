# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 24770 ns — **Best latency**: 22820 ns (**1.0855×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`split_k_partial_reduction_to_expand_grid_beyond_27_tiles`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_30`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_30_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (54,1,1), Waves/SM: 0.05
- **Registers/thread**: 60
- **Duration**: 24770 ns
- **SM Throughput**: 3.45 % | **Memory**: 34.01 % | **DRAM**: 2.47 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.37 inst/cycle | **No Eligible**: 90.75 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.993× | 1.009× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.398× | 1.042× | ✓ |  |
| C | autotune_tile_sizes_for_tiny_m_gemm_with_eviction_policies | 1.391× | 1.335× | ✓ |  |
| D | autotune_tile_sizes_for_tiny_gemm | 1.411× | 1.281× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24770 ns | 24540 ns | -230 ns (-0.9%) | 23780 ns | -990 ns (-4.0%) | 18560 ns | -6210 ns (-25.1%) | 19330 ns | -5440 ns (-22.0%) |
| SM Throughput (%) | 3.45 % | 3.49 % | +0.04 pp | 2.67 % | -0.78 pp | 3.14 % | -0.31 pp | 2.86 % | -0.59 pp |
| Memory Throughput (%) | 34.01 % | 34.37 % | +0.36 pp | 18.94 % | -15.07 pp | 23.24 % | -10.77 pp | 21.90 % | -12.11 pp |
| DRAM Throughput (%) | 2.47 % | 2.49 % | +0.02 pp | 2.57 % | +0.10 pp | 3.30 % | +0.83 pp | 3.18 % | +0.71 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 60.00 register/thread | 60.00 register/thread | +0 | 54.00 register/thread | -6 | 80.00 register/thread | +20 | 128.00 register/thread | +68 |
| Waves/SM | 0.05 | 0.05 | +0.00 | 0.05 | +0.00 | 0.07 | +0.02 | 0.07 | +0.02 |
| No Eligible (%) | 90.75 % | 90.72 % | -0.03 pp | 92.62 % | +1.87 pp | 82.91 % | -7.84 pp | 84.42 % | -6.33 pp |
| IPC Active | 0.37 inst/cycle | 0.37 inst/cycle | +0.00 | 0.29 inst/cycle | -0.08 | 0.34 inst/cycle | -0.03 | 0.31 inst/cycle | -0.06 |

---

## Iteration 2

**Best so far**: 1.2814× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_and_autotune_block_n_to_reduce_shared_memory_bank_conflicts | 1.419× | 1.273× | ✓ |  |
| B | reduce_block_k_to_cut_register_pressure_and_add_evict_first | 1.400× | 1.024× | ✓ |  |
| C | increase_block_k_to_reduce_shared_memory_bank_conflicts | 1.419× | 1.148× | ✓ | **✓ best** |
| D | shrink_block_m_to_match_tiny_m_and_autotune_warps_stages | 1.400× | 1.028× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | reduce_block_k_to... | Δ | increase_block_k_... | Δ | shrink_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24770 ns | 19460 ns | -5310 ns (-21.4%) | 24190 ns | -580 ns (-2.3%) | 21570 ns | -3200 ns (-12.9%) | 24100 ns | -670 ns (-2.7%) |
| SM Throughput (%) | 3.45 % | 2.88 % | -0.57 pp | 2.23 % | -1.22 pp | 2.99 % | -0.46 pp | 2.24 % | -1.21 pp |
| Memory Throughput (%) | 34.01 % | 22.07 % | -11.94 pp | 17.76 % | -16.25 pp | 21.16 % | -12.85 pp | 17.81 % | -16.20 pp |
| DRAM Throughput (%) | 2.47 % | 3.16 % | +0.69 pp | 2.53 % | +0.06 pp | 2.84 % | +0.37 pp | 2.54 % | +0.07 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 6.25 % | +0.01 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 60.00 register/thread | 128.00 register/thread | +68 | 80.00 register/thread | +20 | 64.00 register/thread | +4 | 80.00 register/thread | +20 |
| Waves/SM | 0.05 | 0.07 | +0.02 | 0.03 | -0.02 | 0.07 | +0.02 | 0.03 | -0.02 |
| No Eligible (%) | 90.75 % | 84.25 % | -6.50 pp | 87.95 % | -2.80 pp | 91.91 % | +1.16 pp | 87.95 % | -2.80 pp |
| IPC Active | 0.37 inst/cycle | 0.32 inst/cycle | -0.05 | 0.24 inst/cycle | -0.13 | 0.32 inst/cycle | -0.05 | 0.24 inst/cycle | -0.13 |

---

## Iteration 3

**Best so far**: 1.2814× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | pad_shared_memory_b_tile_to_break_bank_conflicts | 1.475× | 0.798× | ✓ |  |
| B | break_bank_conflicts_via_non_power_of_two_block_n_with_evict_first | 1.409× | 1.148× | ✓ |  |
| C | autotune_block_n_256_and_block_k_16_to_break_17way_bank_conflicts | 1.154× | 0.257× | ✓ |  |
| D | change_block_n_to_break_shared_memory_bank_conflict_alignment | 2.068× | 1.047× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | pad_shared_memory... | Δ | break_bank_confli... | Δ | autotune_block_n_... | Δ | change_block_n_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24770 ns | 31040 ns | +6270 ns (+25.3%) | 21570 ns | -3200 ns (-12.9%) | 96220 ns | +71450 ns (+288.5%) | 23650 ns | -1120 ns (-4.5%) |
| SM Throughput (%) | 3.45 % | 1.45 % | -2.00 pp | 3.01 % | -0.44 pp | 0.99 % | -2.46 pp | 1.36 % | -2.09 pp |
| Memory Throughput (%) | 34.01 % | 14.57 % | -19.44 pp | 21.23 % | -12.78 pp | 10.59 % | -23.42 pp | 5.40 % | -28.61 pp |
| DRAM Throughput (%) | 2.47 % | 1.98 % | -0.49 pp | 2.84 % | +0.37 pp | 0.64 % | -1.83 pp | 2.58 % | +0.11 pp |
| Achieved Occupancy (%) | 6.24 % | 1.56 % | -4.68 pp | 6.25 % | +0.01 pp | 1.56 % | -4.68 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 60.00 register/thread | 254.00 register/thread | +194 | 64.00 register/thread | +4 | 158.00 register/thread | +98 | 44.00 register/thread | -16 |
| Waves/SM | 0.05 | 0.07 | +0.02 | 0.07 | +0.02 | 0.03 | -0.02 | 0.02 | -0.03 |
| No Eligible (%) | 90.75 % | 83.58 % | -7.17 pp | 91.93 % | +1.18 pp | 90.94 % | +0.19 pp | 92.37 % | +1.62 pp |
| IPC Active | 0.37 inst/cycle | 0.16 inst/cycle | -0.21 | 0.32 inst/cycle | -0.05 | 0.09 inst/cycle | -0.28 | 0.15 inst/cycle | -0.22 |

---

## Iteration 4

**Best so far**: 1.2814× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | stream_k_split_k_reduction_to_expand_grid_coverage | 1.418× | 1.150× | ✓ |  |
| B | persistent_kernel_scheduling_to_fill_sm_grid | 1.951× | 1.092× | ✓ |  |
| C | revert_block_ptr_to_manual_pointer_arithmetic_to_restore_occupancy | 1.506× | 0.999× | ✓ |  |
| D | restore_block_k64_block_n128_and_exclude_num_warps1_to_fix_autotuner_regression | 1.418× | 1.286× | ✓ |  |
| E | add_triton_autotune | 1.411× | 1.144× | ✓ |  |


### Metric Comparison

| Metric | Baseline | stream_k_split_k_... | Δ | persistent_kernel... | Δ | revert_block_ptr_... | Δ | restore_block_k64... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24770 ns | 21540 ns | -3230 ns (-13.0%) | 22690 ns | -2080 ns (-8.4%) | 24800 ns | +30 ns (+0.1%) | 19260 ns | -5510 ns (-22.2%) | 21660 ns | -3110 ns (-12.6%) |
| SM Throughput (%) | 3.45 % | 3.00 % | -0.45 pp | 0.73 % | -2.72 pp | 2.05 % | -1.40 pp | 2.89 % | -0.56 pp | 2.95 % | -0.50 pp |
| Memory Throughput (%) | 34.01 % | 21.21 % | -12.80 pp | 2.86 % | -31.15 pp | 28.27 % | -5.74 pp | 22.19 % | -11.82 pp | 20.86 % | -13.15 pp |
| DRAM Throughput (%) | 2.47 % | 2.84 % | +0.37 pp | 2.69 % | +0.22 pp | 2.47 % | +0.00 pp | 3.19 % | +0.72 pp | 2.82 % | +0.35 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 60.00 register/thread | 64.00 register/thread | +4 | 43.00 register/thread | -17 | 128.00 register/thread | +68 | 128.00 register/thread | +68 | 64.00 register/thread | +4 |
| Waves/SM | 0.05 | 0.07 | +0.02 | 0.02 | -0.03 | 0.07 | +0.02 | 0.07 | +0.02 | 0.07 | +0.02 |
| No Eligible (%) | 90.75 % | 91.90 % | +1.15 pp | 92.58 % | +1.83 pp | 87.87 % | -2.88 pp | 84.21 % | -6.54 pp | 91.98 % | +1.23 pp |
| IPC Active | 0.37 inst/cycle | 0.32 inst/cycle | -0.05 | 0.15 inst/cycle | -0.22 | 0.24 inst/cycle | -0.13 | 0.31 inst/cycle | -0.06 | 0.32 inst/cycle | -0.05 |

---

## Iteration 5

**Best so far**: 1.2814× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | xor_swizzle_k_row_index_to_eliminate_17way_bank_conflicts | 1.105× | 0.576× | ✓ |  |
| C | split_k_partial_reduction_to_expand_grid_beyond_27_tiles | 2.205× | 1.085× | ✓ | **✓ best** |
| D | transpose_b_tile_load_for_conflict_free_shared_memory | 1.488× | 1.005× | ✓ |  |


### Metric Comparison

| Metric | Baseline | xor_swizzle_k_row... | Δ | split_k_partial_r... | Δ | transpose_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 24770 ns | 43010 ns | +18240 ns (+73.6%) | 22820 ns | -1950 ns (-7.9%) | 24640 ns | -130 ns (-0.5%) |
| SM Throughput (%) | 3.45 % | 2.49 % | -0.96 pp | 0.74 % | -2.71 pp | 2.06 % | -1.39 pp |
| Memory Throughput (%) | 34.01 % | 11.26 % | -22.75 pp | 2.90 % | -31.11 pp | 28.35 % | -5.66 pp |
| DRAM Throughput (%) | 2.47 % | 1.44 % | -1.03 pp | 2.68 % | +0.21 pp | 2.49 % | +0.02 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 60.00 register/thread | 255.00 register/thread | +195 | 43.00 register/thread | -17 | 128.00 register/thread | +68 |
| Waves/SM | 0.05 | 0.10 | +0.05 | 0.02 | -0.03 | 0.07 | +0.02 |
| No Eligible (%) | 90.75 % | 86.63 % | -4.12 pp | 92.55 % | +1.80 pp | 87.83 % | -2.92 pp |
| IPC Active | 0.37 inst/cycle | 0.27 inst/cycle | -0.10 | 0.15 inst/cycle | -0.22 | 0.24 inst/cycle | -0.13 |


### Key Learnings

#### ❌ Plan A: `xor_swizzle_k_row_index_to_eliminate_17way_bank_conflicts` (0.576×)

Applying an XOR bijection (offs_k = tl.arange(0, BLOCK_K) ^ SWIZZLE_K) to the K-dimension index inside the K-loop of a Triton GEMM kernel caused registers per thread to explode from 128 to 255, doubling kernel duration from 19.26 µs to 43.01 µs (0.58x speedup). The bank conflicts were completely unaffected — the 17-way conflict way and 82.34% conflict wavefront percentage remained identical — because Triton's compiler operates on virtual register indices and does not propagate the XOR permutation into the physical shared memory layout; the actual PTX load addresses are determined by the compiler's register allocation, not the index arithmetic in the Triton IR. The register explosion (128→255) triggered severe register spill to local memory, evidenced by new MemoryCacheAccessPattern warnings (1 of 32 bytes utilized per sector for local loads and stores), which nearly doubled executed instructions (445K→870K) and cut theoretical occupancy from 18.75% to 12.5% (Block_Limit_Registers dropped from 8 to 4 blocks per SM).

> **Applicable when**: Never: this is a negative example documenting a regression pattern to avoid.

> **Avoid when**: Do not apply XOR index permutation to K-dimension indices in a Triton GEMM kernel as a technique to eliminate shared memory bank conflicts. Triton's compiler does not translate index-space XOR permutations into physical shared memory layout changes — the bank conflict pattern is determined by the compiler's register allocation and PTX generation, not by arithmetic on virtual index tensors. Additionally, moving tl.arange(0, BLOCK_K) inside the K-loop (from outside) causes the compiler to re-allocate the index tensor on every iteration, dramatically increasing register pressure (128→255 registers/thread) and triggering register spill to local memory. This anti-pattern is especially harmful when the kernel is already register-limited (Block_Limit_Registers is the binding occupancy constraint) — any increase in register count per thread will further reduce theoretical occupancy and cause spill.

---

## Cumulative Progress

```
Baseline:  24770 ns  (1.000×)
Iter 1     19330 ns  (1.281×)  — autotune_tile_sizes_for_tiny_gemm

Final best: 22820 ns  (1.0855×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,36 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=5),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_30(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 2
+    BLOCK_N : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
+    NUM_SMS : tl.constexpr = 132
     A = arg_A
     B = arg_B
 
@@ -30,68 +49,59 @@

     stride_bk = 1
     stride_bn = 432
 
-    # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    grid_m = tl.cdiv(M, BLOCK_M)   # = 1
+    grid_n = tl.cdiv(N, BLOCK_N)   # = 27
+    num_tiles = grid_m * grid_n     # = 27
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    for tile_id in range(pid, num_tiles, NUM_SMS):
+        # re-order program ID for better L2 performance
+        width = GROUP_M * grid_n
+        group_id = tile_id // width
+        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
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
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
+            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
+            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
+            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 432*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+            idx_m = offs_a_m[:, None]
+            idx_n = a_k_idx_vals
+            xindex = idx_n + 432*idx_m
+            a = tl.load(A + (xindex), mask=a_mask, other=0.0, eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
+            idx_m = b_k_idx_vals
+            idx_n = offs_b_n[None, :]
+            xindex = idx_n + 1728*idx_m
+            b = tl.load(B + ((tl.broadcast_to(idx_m + 432*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0, eviction_policy='evict_first')
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
         xindex = idx_n + 1728*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 432*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
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
-    xindex = idx_n + 1728*idx_m
-    tmp0 = tl.sigmoid(acc)
-    tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
-
-
+        tmp0 = tl.sigmoid(acc)
+        tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
