# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 70430 ns — **Best latency**: 41440 ns (**1.6996×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tma_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_33_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (32,1,1), Waves/SM: 0.03
- **Registers/thread**: 56
- **Duration**: 70430 ns
- **SM Throughput**: 2.02 % | **Memory**: 22.23 % | **DRAM**: 1.64 %
- **Achieved Occupancy**: 6.22 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.40 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 70430 ns | 70270 ns | -160 ns (-0.2%) |
| SM Throughput (%) | 2.02 % | 2.02 % | +0.00 pp |
| Memory Throughput (%) | 22.23 % | 22.23 % | +0.00 pp |
| DRAM Throughput (%) | 1.64 % | 1.64 % | +0.00 pp |
| Achieved Occupancy (%) | 6.22 % | 6.23 % | +0.01 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 |
| No Eligible (%) | 91.40 % | 91.39 % | -0.01 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads | 1.653× | 1.700× | ✓ | **✓ best** |
| C | fix_b_matrix_load_coalescing_and_reduce_block_m | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 70430 ns | 41440 ns | -28990 ns (-41.2%) | 70530 ns | +100 ns (+0.1%) |
| SM Throughput (%) | 2.02 % | 2.32 % | +0.30 pp | 2.02 % | +0.00 pp |
| Memory Throughput (%) | 22.23 % | 20.61 % | -1.62 pp | 22.22 % | -0.01 pp |
| DRAM Throughput (%) | 1.64 % | 2.79 % | +1.15 pp | 1.63 % | -0.01 pp |
| Achieved Occupancy (%) | 6.22 % | 3.12 % | -3.10 pp | 6.23 % | +0.01 pp |
| Registers/Thread | 56.00 register/thread | 156.00 register/thread | +100 | 56.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.08 | +0.05 | 0.03 | +0.00 |
| No Eligible (%) | 91.40 % | 79.80 % | -11.60 pp | 91.38 % | -0.02 pp |
| IPC Active | 0.34 inst/cycle | 0.40 inst/cycle | +0.06 | 0.34 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.6996× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_smem_layout_order_to_eliminate_bank_conflicts | 1.506× | 1.493× | ✓ |  |
| B | fix_b_matrix_transpose_access_pattern_for_contiguous_loads | 1.583× | 1.616× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_smem_layout... | Δ | fix_b_matrix_tran... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 70430 ns | 47170 ns | -23260 ns (-33.0%) | 43580 ns | -26850 ns (-38.1%) |
| SM Throughput (%) | 2.02 % | 2.13 % | +0.11 pp | 2.21 % | +0.19 pp |
| Memory Throughput (%) | 22.23 % | 18.19 % | -4.04 pp | 19.69 % | -2.54 pp |
| DRAM Throughput (%) | 1.64 % | 2.45 % | +0.81 pp | 2.65 % | +1.01 pp |
| Achieved Occupancy (%) | 6.22 % | 3.12 % | -3.10 pp | 3.12 % | -3.10 pp |
| Registers/Thread | 56.00 register/thread | 128.00 register/thread | +72 | 190.00 register/thread | +134 |
| Waves/SM | 0.03 | 0.08 | +0.05 | 0.08 | +0.05 |
| No Eligible (%) | 91.40 % | 81.56 % | -9.84 pp | 80.79 % | -10.61 pp |
| IPC Active | 0.34 inst/cycle | 0.37 inst/cycle | +0.03 | 0.38 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.6996× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_b_transpose_smem_conflicts_and_cap_registers | 1.506× | 1.497× | ✓ |  |
| B | eliminate_tl_trans_via_correct_b_block_ptr_kn_layout | 1.506× | 1.495× | ✓ |  |
| C | fix_b_block_ptr_order_for_column_major_layout | 1.502× | 1.493× | ✓ |  |
| D | reduce_block_k_and_remove_maxnreg_cap_to_fix_register_spilling_and_smem_conflicts | 1.484× | 1.405× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_b_trans... | Δ | eliminate_tl_tran... | Δ | fix_b_block_ptr_o... | Δ | reduce_block_k_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 70430 ns | 47040 ns | -23390 ns (-33.2%) | 47100 ns | -23330 ns (-33.1%) | 47170 ns | -23260 ns (-33.0%) | 50110 ns | -20320 ns (-28.9%) |
| SM Throughput (%) | 2.02 % | 2.13 % | +0.11 pp | 2.14 % | +0.12 pp | 2.13 % | +0.11 pp | 2.36 % | +0.34 pp |
| Memory Throughput (%) | 22.23 % | 18.21 % | -4.02 pp | 18.26 % | -3.97 pp | 18.22 % | -4.01 pp | 17.09 % | -5.14 pp |
| DRAM Throughput (%) | 1.64 % | 2.45 % | +0.81 pp | 2.45 % | +0.81 pp | 2.45 % | +0.81 pp | 2.30 % | +0.66 pp |
| Achieved Occupancy (%) | 6.22 % | 3.12 % | -3.10 pp | 3.12 % | -3.10 pp | 3.12 % | -3.10 pp | 3.12 % | -3.10 pp |
| Registers/Thread | 56.00 register/thread | 128.00 register/thread | +72 | 128.00 register/thread | +72 | 128.00 register/thread | +72 | 94.00 register/thread | +38 |
| Waves/SM | 0.03 | 0.08 | +0.05 | 0.08 | +0.05 | 0.08 | +0.05 | 0.04 | +0.01 |
| No Eligible (%) | 91.40 % | 81.55 % | -9.85 pp | 81.55 % | -9.85 pp | 81.54 % | -9.86 pp | 79.60 % | -11.80 pp |
| IPC Active | 0.34 inst/cycle | 0.37 inst/cycle | +0.03 | 0.37 inst/cycle | +0.03 | 0.37 inst/cycle | +0.03 | 0.41 inst/cycle | +0.07 |


### Key Learnings

#### ❌ Plan A: `eliminate_b_transpose_smem_conflicts_and_cap_registers` (1.497×)

Restructuring the B block pointer from shape (N,K) to (K,N) to eliminate tl.trans() and adding maxnreg=128 to autotune configs did not resolve the 17-way shared memory bank conflicts (82.34% excessive wavefronts remained identical before and after). Kernel duration increased from 43.58 µs to 47.04 µs, and memory throughput dropped from 130.04 to 120.53 GB/s. The maxnreg=128 cap (down from 190) triggered new MemoryCacheAccessPattern warnings for uncoalesced local loads/stores (1.0 of 32 bytes utilized per sector), indicating register spilling to local memory that offset any potential benefit from removing the transpose. The bank conflicts persisted because the root cause lies in the TMA swizzling interaction with the tile shape, not the tl.trans() call itself.

> **Applicable when**: Never — this combination caused regression in duration and memory throughput.

> **Avoid when**: When the kernel uses TMA block pointers on Hopper and SharedMemoryConflicts persist at 17-way with 82.34% excessive wavefronts: restructuring B from (N,K)+tl.trans() to (K,N) direct load does NOT resolve the bank conflicts — they remain identical. Additionally, adding maxnreg=128 when the kernel naturally uses 190 registers/thread causes register spilling to local memory, introducing MemoryCacheAccessPattern warnings (1.0/32 bytes utilized per sector) and reducing memory throughput. Do not apply this combination when the baseline already achieves 130+ GB/s memory throughput with TMA block pointers.

#### ❌ Plan C: `fix_b_block_ptr_order_for_column_major_layout` (1.493×)

Changing the B matrix TMA block pointer from a transposed load pattern (shape=(N,K), order=(1,0), with tl.trans()) to a direct load pattern (shape=(K,N), order=(0,1)) did not eliminate the 17-way shared memory bank conflicts — they remained at exactly 17-way with 82.34% excessive wavefronts. The kernel duration increased from 43.58 µs to 47.17 µs (~8% regression), and memory throughput dropped from 130.04 to 120.2 GB/s. The simultaneous addition of maxnreg=128 (down from 190 registers/thread) introduced new MemoryCacheAccessPattern warnings for uncoalesced local loads/stores, indicating register spilling to local memory that partially caused the regression. The bank conflict root cause on Hopper TMA kernels with this tile shape (BLOCK_K=64, BLOCK_N=128) is not resolved by order parameter changes alone.

> **Applicable when**: Never — this combination of order parameter change + maxnreg reduction caused regression on a kernel that already had TMA block pointers active.

> **Avoid when**: When a Triton GEMM on Hopper already uses TMA block pointers and SharedMemoryConflicts persist at 17-way with 82%+ excessive wavefronts: changing the B block pointer from a transposed-load pattern (shape=(N,K), tl.trans()) to a direct-load pattern (shape=(K,N), order=(0,1)) does NOT resolve the bank conflicts for tile shape BLOCK_K=64, BLOCK_N=128. Additionally, adding maxnreg=128 when the kernel naturally allocates 156-190 registers/thread causes register spilling to local memory (MemoryCacheAccessPattern warnings, 1.0/32 bytes utilized per sector), reducing memory throughput. Do not apply this when the baseline already achieves 130+ GB/s memory throughput with TMA block pointers.

---

## Cumulative Progress

```
Baseline:  70430 ns  (1.000×)
Iter 2     41440 ns  (1.700×)  — tma_loads

Final best: 41440 ns  (1.6996×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,30 +5,41 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_addmm_convolution_mean_relu_sigmoid_t_view_33(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
-    BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
+    ALLOW_TF32: tl.constexpr = False
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 2
+    BLOCK_N: tl.constexpr = 128
+    BLOCK_K: tl.constexpr = 64
+    GROUP_M: tl.constexpr = 8
+    INDEX_DTYPE: tl.constexpr = tl.int32
 
-    M = 2
-    N = 1000
-    K = 1408
+    M: tl.constexpr = 2
+    N: tl.constexpr = 1000
+    K: tl.constexpr = 1408
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 1408
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 1408
+
+    stride_am: tl.constexpr = 1408
+    stride_ak: tl.constexpr = 1
+    stride_bk: tl.constexpr = 1
+    stride_bn: tl.constexpr = 1408  # B is (N, K) row-major with stride_bn=1408
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -44,37 +55,34 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
-    if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
-        offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
-    else:
-        offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
+    # Create block pointers for TMA loads
+    A_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # Load B as (N, K) — contiguous along K (stride=1), eliminating bank conflicts
+    B_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(N, K),
+        strides=(stride_bn, stride_bk),   # (1408, 1) — K is now the fast dim
+        offsets=(pid_n * BLOCK_N, 0),
+        block_shape=(BLOCK_N, BLOCK_K),
+        order=(1, 0),                      # K is contiguous (innermost)
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 1408*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1000*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 1408*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+    for _ in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))  # shape (BLOCK_N, BLOCK_K)
+        b = tl.trans(b)                                   # -> (BLOCK_K, BLOCK_N) for dot
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (0, BLOCK_K))  # advance along K dim
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,12 +92,10 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 1000*idx_m
+    xindex = idx_n + 1000 * idx_m
     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
