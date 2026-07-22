# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 73790 ns — **Best latency**: 60260 ns (**1.2245×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`reduce_block_m_to_lower_register_pressure_and_improve_sm_coverage`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_33_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.19
- **Registers/thread**: 92
- **Duration**: 73790 ns
- **SM Throughput**: 11.36 % | **Memory**: 83.00 % | **DRAM**: 1.54 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.56 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.002× | 0.999× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.172× | 1.197× | ✓ | **✓ best** |
| C | add_eviction_policies_and_autotune_tile_sizes | 1.103× | 1.106× | ✓ |  |
| D | autotune_tile_sizes_and_warps_to_fix_bank_conflicts_and_occupancy | 0.982× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | add_eviction_poli... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 73790 ns | 73860 ns | +70 ns (+0.1%) | 61660 ns | -12130 ns (-16.4%) | 66720 ns | -7070 ns (-9.6%) | 75260 ns | +1470 ns (+2.0%) |
| SM Throughput (%) | 11.36 % | 11.35 % | -0.01 pp | 12.68 % | +1.32 pp | 12.60 % | +1.24 pp | 20.90 % | +9.54 pp |
| Memory Throughput (%) | 83.00 % | 82.99 % | -0.01 pp | 55.11 % | -27.89 pp | 86.76 % | +3.76 pp | 81.59 % | -1.41 pp |
| DRAM Throughput (%) | 1.54 % | 1.54 % | +0.00 pp | 1.85 % | +0.31 pp | 1.71 % | +0.17 pp | 1.51 % | -0.03 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 92.00 register/thread | 92.00 register/thread | +0 | 128.00 register/thread | +36 | 94.00 register/thread | +2 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.19 | +0.00 | 0.12 | -0.07 | 0.19 | +0.00 | 0.23 | +0.04 |
| No Eligible (%) | 87.56 % | 87.55 % | -0.01 pp | 71.61 % | -15.95 pp | 86.14 % | -1.42 pp | 76.93 % | -10.63 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.57 inst/cycle | +0.07 | 0.55 inst/cycle | +0.05 | 0.92 inst/cycle | +0.42 |

---

## Iteration 2

**Best so far**: 1.1967× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | swap_block_m_n_to_eliminate_shared_memory_bank_conflicts | 1.194× | 1.219× | ✓ | **✓ best** |
| C | swap_tile_shape_and_increase_block_k_to_reduce_bank_conflicts | 1.103× | 0.956× | ✓ |  |
| D | increase_block_k_to_resolve_shared_memory_bank_conflicts | 0.856× | 0.841× | ✓ |  |


### Metric Comparison

| Metric | Baseline | swap_block_m_n_to... | Δ | swap_tile_shape_a... | Δ | increase_block_k_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 73790 ns | 60540 ns | -13250 ns (-18.0%) | 77180 ns | +3390 ns (+4.6%) | 87710 ns | +13920 ns (+18.9%) |
| SM Throughput (%) | 11.36 % | 13.05 % | +1.69 pp | 10.62 % | -0.74 pp | 18.56 % | +7.20 pp |
| Memory Throughput (%) | 83.00 % | 50.94 % | -32.06 pp | 74.76 % | -8.24 pp | 72.79 % | -10.21 pp |
| DRAM Throughput (%) | 1.54 % | 1.88 % | +0.34 pp | 1.47 % | -0.07 pp | 1.30 % | -0.24 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 92.00 register/thread | 128.00 register/thread | +36 | 80.00 register/thread | -12 | 168.00 register/thread | +76 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.16 | -0.03 | 0.31 | +0.12 |
| No Eligible (%) | 87.56 % | 71.14 % | -16.42 pp | 88.32 % | +0.76 pp | 79.00 % | -8.56 pp |
| IPC Active | 0.50 inst/cycle | 0.57 inst/cycle | +0.07 | 0.46 inst/cycle | -0.04 | 0.84 inst/cycle | +0.34 |


### Key Learnings

#### ✅ Plan A: `swap_block_m_n_to_eliminate_shared_memory_bank_conflicts` (1.219×)

Swapping BLOCK_M and BLOCK_N (e.g., 64→32 and 32→64) in a Triton GEMM kernel changes the B-matrix tile shape and its shared memory access stride, which can partially reduce shared memory bank conflicts when the original stride is a multiple of 32 (the number of shared memory banks). In this case the change reduced bank conflict ways from 13.3 to 12.0 and Mem Busy from 55.1% to 50.9%, yielding a modest 1.22x speedup; however, the dominant bank conflict signal (SharedMemoryConflicts, ~77-80% wavefront ratio) and excessive uncoalesced wavefronts remained largely intact, indicating the tile swap alone is insufficient to fully resolve stride-aligned bank conflicts. The grid size must be verified to remain consistent after the swap (ceil(M/BLOCK_M) × ceil(N/BLOCK_N) must equal the original grid size).

> **Applicable when**: Triton GEMM kernel where SharedMemoryConflicts NCU rule is triggered with >70% bank-conflict wavefront ratio, the B-matrix stride is a multiple of 32 (causing all-same-bank conflicts), and swapping BLOCK_M/BLOCK_N preserves the grid size (ceil(M/BLOCK_M)*ceil(N/BLOCK_N) is invariant under the swap).

> **Avoid when**: The bank conflict root cause is not stride-alignment (e.g., conflicts arise from irregular access patterns not fixed by tile shape); when swapping BLOCK_M/BLOCK_N changes the grid size and requires additional correctness fixes; when the kernel is already at the minimum tile size for the given problem dimensions; when the dominant bottleneck is compute or DRAM bandwidth rather than shared memory conflicts.

#### ❌ Plan D: `increase_block_k_to_resolve_shared_memory_bank_conflicts` (0.841×)

Doubling BLOCK_K from 32 to 64 (combined with doubling BLOCK_N from 32 to 64 and block size from 64 to 128) caused a 42% regression (87.71 µs vs 61.66 µs). Despite the theoretical argument that BLOCK_K=64 changes the shared memory access stride, the bank conflict ratio barely improved (12.0-way vs 13.3-way, 77.69% vs 79.91% of wavefronts). The dominant negative effects were: (1) executed instructions more than doubled (12.98M vs 6.20M), indicating the larger tile configuration did significantly more work per block; (2) registers per thread jumped from 128 to 168, reducing theoretical occupancy from 25% to 18.75% and limiting Block Limit Registers to 3 blocks/SM; (3) dynamic shared memory per block ballooned from 12 KB to 65 KB, further constraining occupancy via Block Limit Shared Mem=3; (4) Warp Cycles Per Issued Instruction worsened from 3.52 to 4.77. The bank conflict pattern was not broken because the access stride in shared memory is determined by the tile layout and warp access pattern, not BLOCK_K alone.

> **Applicable when**: Never — this specific combination of changes caused regression. Do not apply.

> **Avoid when**: When the kernel already has high register pressure (>=128 registers/thread): doubling BLOCK_K and BLOCK_N simultaneously will push registers to 168+/thread, collapsing theoretical occupancy from 25% to <19% and Block Limit Registers to 3. When K is not evenly divisible by the new BLOCK_K (e.g., K=672, BLOCK_K=64: 672%64=32≠0), the required boundary masking adds instruction overhead and doubles executed instruction count. When bank conflicts are 12-13 way (not 32-way), the conflict pattern is not purely stride-based and BLOCK_K changes alone will not resolve it. When the existing kernel duration is already short (<70 µs) and the grid is small (124 blocks < 132 SMs), the overhead of larger tiles outweighs any bank conflict reduction.

---

## Iteration 3

**Best so far**: 1.2189× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_block_k_to_break_shared_memory_bank_conflict_stride_alignment | 1.072× | 1.034× | ✓ |  |
| B | reduce_register_pressure_with_maxnreg_to_improve_occupancy_and_hide_bank_conflict_stalls | 1.105× | 0.956× | ✓ |  |
| C | reduce_block_m_to_lower_register_pressure_and_improve_sm_coverage | 1.605× | 1.224× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_block_k_to... | Δ | reduce_register_p... | Δ | reduce_block_m_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 73790 ns | 71330 ns | -2460 ns (-3.3%) | 77150 ns | +3360 ns (+4.6%) | 60260 ns | -13530 ns (-18.3%) |
| SM Throughput (%) | 11.36 % | 11.08 % | -0.28 pp | 10.63 % | -0.73 pp | 12.91 % | +1.55 pp |
| Memory Throughput (%) | 83.00 % | 43.20 % | -39.80 pp | 74.79 % | -8.21 pp | 50.41 % | -32.59 pp |
| DRAM Throughput (%) | 1.54 % | 1.59 % | +0.05 pp | 1.47 % | -0.07 pp | 1.89 % | +0.35 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 92.00 register/thread | 128.00 register/thread | +36 | 80.00 register/thread | -12 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.16 | -0.03 | 0.12 | -0.07 |
| No Eligible (%) | 87.56 % | 75.76 % | -11.80 pp | 88.39 % | +0.83 pp | 71.24 % | -16.32 pp |
| IPC Active | 0.50 inst/cycle | 0.49 inst/cycle | -0.01 | 0.47 inst/cycle | -0.03 | 0.58 inst/cycle | +0.08 |

---

## Iteration 4

**Best so far**: 1.2245× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | shared_memory_padding_to_break_stride_aligned_bank_conflicts | 1.596× | 1.214× | ✓ |  |
| B | coalesce_b_matrix_global_load_with_max_contiguous_hints | 1.073× | 1.033× | ✓ |  |


### Metric Comparison

| Metric | Baseline | shared_memory_pad... | Δ | coalesce_b_matrix... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 73790 ns | 60770 ns | -13020 ns (-17.6%) | 71420 ns | -2370 ns (-3.2%) |
| SM Throughput (%) | 11.36 % | 13.04 % | +1.68 pp | 11.03 % | -0.33 pp |
| Memory Throughput (%) | 83.00 % | 50.91 % | -32.09 pp | 43.02 % | -39.98 pp |
| DRAM Throughput (%) | 1.54 % | 1.88 % | +0.34 pp | 1.59 % | +0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 92.00 register/thread | 128.00 register/thread | +36 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 |
| No Eligible (%) | 87.56 % | 71.15 % | -16.41 pp | 75.66 % | -11.90 pp |
| IPC Active | 0.50 inst/cycle | 0.58 inst/cycle | +0.08 | 0.49 inst/cycle | -0.01 |

---

## Iteration 5

**Best so far**: 1.2245× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads | 1.562× | 1.588× | ✓ |  |
| C | use_block_ptr_api_for_coalesced_ab_loads_to_reduce_uncoalesced_shared_access | 1.073× | 0.890× | ✓ |  |
| D | tma_loads | 1.525× | 1.090× | ✓ |  |
| E | add_triton_autotune | 1.599× | 1.212× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ | use_block_ptr_api... | Δ | tma_loads | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 73790 ns | 46460 ns | -27330 ns (-37.0%) | 82880 ns | +9090 ns (+12.3%) | 67710 ns | -6080 ns (-8.2%) | 60900 ns | -12890 ns (-17.5%) |
| SM Throughput (%) | 11.36 % | 17.00 % | +5.64 pp | 9.59 % | -1.77 pp | 12.77 % | +1.41 pp | 12.97 % | +1.61 pp |
| Memory Throughput (%) | 83.00 % | 66.24 % | -16.76 pp | 37.25 % | -45.75 pp | 85.72 % | +2.72 pp | 50.62 % | -32.38 pp |
| DRAM Throughput (%) | 1.54 % | 2.50 % | +0.96 pp | 1.40 % | -0.14 pp | 1.68 % | +0.14 pp | 1.87 % | +0.33 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 92.00 register/thread | 128.00 register/thread | +36 | 128.00 register/thread | +36 | 92.00 register/thread | +0 | 128.00 register/thread | +36 |
| Waves/SM | 0.19 | 0.12 | -0.07 | 0.12 | -0.07 | 0.19 | +0.00 | 0.12 | -0.07 |
| No Eligible (%) | 87.56 % | 61.89 % | -25.67 pp | 78.83 % | -8.73 pp | 86.02 % | -1.54 pp | 71.26 % | -16.30 pp |
| IPC Active | 0.50 inst/cycle | 0.76 inst/cycle | +0.26 | 0.42 inst/cycle | -0.08 | 0.56 inst/cycle | +0.06 | 0.57 inst/cycle | +0.07 |

---

## Cumulative Progress

```
Baseline:  73790 ns  (1.000×)
Iter 1     61660 ns  (1.197×)  — autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy
Iter 2     60540 ns  (1.219×)  — swap_block_m_n_to_eliminate_shared_memory_bank_conflicts
Iter 3     60260 ns  (1.224×)  — reduce_block_m_to_lower_register_pressure_and_improve_sm_coverage

Final best: 60260 ns  (1.2245×)
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
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_33(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
+    EVEN_K : tl.constexpr = True   # 672 % 32 == 0, no boundary masking required
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_M : tl.constexpr = 32   # reduced from 64: acc[] shrinks 4096->2048 elements
+    BLOCK_N : tl.constexpr = 64   # kept at 64: grid = ceil(1960/32)*ceil(112/64) = 62*2 = 124 (unchanged)
+    BLOCK_K : tl.constexpr = 32   # reduced from 64: 672%32==0 -> EVEN_K=True
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -65,13 +79,29 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 672*idx_m
-        a = tl.load(A + (xindex))
+        if EVEN_K:
+            a = tl.load(A + (xindex))
+        else:
+            mask_k_a = (offs_k[None, :] + k_idx * BLOCK_K) < K
+            a = tl.load(A + (xindex), mask=mask_k_a, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 112*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 672*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # Load B transposed: shape [BLOCK_N, BLOCK_K] to break stride-aligned bank conflicts.
+        # Original load: B tile [BLOCK_K=32, BLOCK_N=64], smem stride=64 floats, 64%32==0 -> 32-way conflict.
+        # Transposed load: B tile [BLOCK_N=64, BLOCK_K=32], smem stride=32 floats.
+        # When tl.dot reads columns of b_t (rows of B), threads access consecutive elements -> no conflict.
+        # b_t[n, k] = B[k, n] in logical space; we load b_t then tl.trans to get [BLOCK_K, BLOCK_N] for dot.
+        idx_n_t = offs_b_n[:, None]   # [BLOCK_N, 1]
+        idx_k_t = b_k_idx_vals.T      # [1, BLOCK_K] - transpose of b_k_idx_vals which was [BLOCK_K, 1]
+        # B layout: B[k, n] accessed as B + k*1 + n*672 (stride_bk=1, stride_bn=672)
+        # b_t[n, k] = B[n_global, k_global] = B + k_global + 672*n_global
+        if EVEN_K:
+            b_t = tl.load(B + (tl.broadcast_to(idx_k_t + 672 * idx_n_t, [BLOCK_N, BLOCK_K])))
+        else:
+            mask_k_b = (offs_k[None, :] + k_idx * BLOCK_K) < K
+            b_t = tl.load(B + (tl.broadcast_to(idx_k_t + 672 * idx_n_t, [BLOCK_N, BLOCK_K])), mask=mask_k_b, other=0.0)
 
+        # Transpose b_t from [BLOCK_N, BLOCK_K] back to [BLOCK_K, BLOCK_N] for tl.dot
+        b = tl.trans(b_t)
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
 
@@ -86,8 +116,6 @@

     # inductor generates a suffix
     xindex = idx_n + 112*idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
