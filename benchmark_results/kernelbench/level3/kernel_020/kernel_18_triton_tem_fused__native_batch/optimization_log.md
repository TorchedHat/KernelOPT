# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 23230 ns — **Best latency**: 17020 ns (**1.3649×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_load_stride_to_eliminate_bank_conflicts_and_reduce_registers`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_18_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 96
- **Duration**: 23230 ns
- **SM Throughput**: 10.64 % | **Memory**: 75.61 % | **DRAM**: 5.32 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.50 inst/cycle | **No Eligible**: 87.36 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_to_fix_smem_bank_conflicts_and_occupancy | 1.315× | 1.352× | ✓ | **✓ best** |
| D | add_autotune_larger_tiles_more_warps | 1.133× | 0.947× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | add_autotune_larg... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 23230 ns | 17180 ns | -6050 ns (-26.0%) | 24540 ns | +1310 ns (+5.6%) |
| SM Throughput (%) | 10.64 % | 13.67 % | +3.03 pp | 9.88 % | -0.76 pp |
| Memory Throughput (%) | 75.61 % | 57.96 % | -17.65 pp | 71.48 % | -4.13 pp |
| DRAM Throughput (%) | 5.32 % | 7.21 % | +1.89 pp | 5.04 % | -0.28 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 156.00 register/thread | +60 | 96.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.19 | -0.04 |
| No Eligible (%) | 87.36 % | 66.33 % | -21.03 pp | 88.23 % | +0.87 pp |
| IPC Active | 0.50 inst/cycle | 0.66 inst/cycle | +0.16 | 0.47 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.3522× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_stride_to_eliminate_bank_conflicts_and_reduce_registers | 1.320× | 1.365× | ✓ | **✓ best** |
| B | tma_loads_replace_scalar_pointer_arithmetic | 1.277× | 0.992× | ✓ |  |
| D | tma_loads_with_reduced_block_k | 1.271× | 0.905× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_replace... | Δ | tma_loads_with_re... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 23230 ns | 17020 ns | -6210 ns (-26.7%) | 23420 ns | +190 ns (+0.8%) | 25660 ns | +2430 ns (+10.5%) |
| SM Throughput (%) | 10.64 % | 13.48 % | +2.84 pp | 10.54 % | -0.10 pp | 9.40 % | -1.24 pp |
| Memory Throughput (%) | 75.61 % | 57.41 % | -18.20 pp | 74.28 % | -1.33 pp | 67.60 % | -8.01 pp |
| DRAM Throughput (%) | 5.32 % | 7.26 % | +1.94 pp | 5.28 % | -0.04 pp | 4.81 % | -0.51 pp |
| Achieved Occupancy (%) | 6.24 % | 3.12 % | -3.12 pp | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 156.00 register/thread | +60 | 80.00 register/thread | -16 | 80.00 register/thread | -16 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.16 | -0.07 | 0.16 | -0.07 |
| No Eligible (%) | 87.36 % | 66.93 % | -20.43 pp | 87.51 % | +0.15 pp | 88.82 % | +1.46 pp |
| IPC Active | 0.50 inst/cycle | 0.66 inst/cycle | +0.16 | 0.50 inst/cycle | +0.00 | 0.45 inst/cycle | -0.05 |

---

## Iteration 3

**Best so far**: 1.3649× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | eliminate_b_matrix_transpose_to_fix_smem_bank_conflicts | 1.241× | 0.904× | ✓ |  |
| C | transpose_b_tile_load_to_eliminate_smem_bank_conflicts | 1.255× | 0.992× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_b_matri... | Δ | transpose_b_tile_... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 23230 ns | 25700 ns | +2470 ns (+10.6%) | 23420 ns | +190 ns (+0.8%) |
| SM Throughput (%) | 10.64 % | 9.39 % | -1.25 pp | 10.57 % | -0.07 pp |
| Memory Throughput (%) | 75.61 % | 67.52 % | -8.09 pp | 74.56 % | -1.05 pp |
| DRAM Throughput (%) | 5.32 % | 4.80 % | -0.52 pp | 5.27 % | -0.05 pp |
| Achieved Occupancy (%) | 6.24 % | 6.25 % | +0.01 pp | 6.24 % | +0.00 pp |
| Registers/Thread | 96.00 register/thread | 80.00 register/thread | -16 | 80.00 register/thread | -16 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.16 | -0.07 |
| No Eligible (%) | 87.36 % | 88.90 % | +1.54 pp | 87.46 % | +0.10 pp |
| IPC Active | 0.50 inst/cycle | 0.44 inst/cycle | -0.06 | 0.50 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.3649× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | load_b_as_k_major_to_eliminate_smem_bank_conflicts_via_natural_dot_layout | 1.253× | 0.990× | ✓ |  |
| C | fix_b_block_ptr_order_to_eliminate_smem_bank_conflicts | 1.254× | 0.900× | ✓ |  |


### Metric Comparison

| Metric | Baseline | load_b_as_k_major... | Δ | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 23230 ns | 23460 ns | +230 ns (+1.0%) | 25820 ns | +2590 ns (+11.1%) |
| SM Throughput (%) | 10.64 % | 10.62 % | -0.02 pp | 9.41 % | -1.23 pp |
| Memory Throughput (%) | 75.61 % | 74.79 % | -0.82 pp | 67.73 % | -7.88 pp |
| DRAM Throughput (%) | 5.32 % | 5.27 % | -0.05 pp | 4.78 % | -0.54 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 6.25 % | +0.01 pp |
| Registers/Thread | 96.00 register/thread | 80.00 register/thread | -16 | 80.00 register/thread | -16 |
| Waves/SM | 0.23 | 0.16 | -0.07 | 0.16 | -0.07 |
| No Eligible (%) | 87.36 % | 87.58 % | +0.22 pp | 88.85 % | +1.49 pp |
| IPC Active | 0.50 inst/cycle | 0.50 inst/cycle | +0.00 | 0.44 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  23230 ns  (1.000×)
Iter 1     17180 ns  (1.352×)  — autotune_tile_sizes_to_fix_smem_bank_conflicts_and_occupancy
Iter 2     17020 ns  (1.365×)  — fix_b_matrix_load_stride_to_eliminate_bank_conflicts_and_reduce_registers

Final best: 17020 ns  (1.3649×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=2, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_18(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -12,7 +26,7 @@

     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 64
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 64
+    BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -44,37 +58,40 @@

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
+    # Build block pointers ONCE before the k-loop
+    # A: physical shape [M, K] = [7840, 192], row-major strides [192, 1]
+    a_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B: loaded transposed as [N, K] to eliminate shared memory bank conflicts.
+    # arg_1 has shape (32,192,1,1) strides (192,1,1,1); as [N,K] matrix: strides=[192,1]
+    # stride_bn=192 is a multiple of 32 banks; loading N-major makes N the fast dim (stride=1),
+    # so consecutive warp lanes access consecutive banks — zero bank conflicts.
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(N, K),
+        strides=(stride_bn, stride_bk),  # (192, 1)
+        offsets=(pid_n * BLOCK_N, 0),
+        block_shape=(BLOCK_N, BLOCK_K),
+        order=(1, 0),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 192*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # Block pointer loads: address generation offloaded to hardware
+        a = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        # b is [BLOCK_N, BLOCK_K]; tl.trans gives [BLOCK_K, BLOCK_N] for correct A[M,K] @ B[K,N]
+        acc += tl.dot(a, tl.trans(b), allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        # Advance block pointers along K dimension
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))  # advance K dim (index 1 in [N,K] layout)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +103,6 @@

     # inductor generates a suffix
     xindex = idx_n + 32*idx_m
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
