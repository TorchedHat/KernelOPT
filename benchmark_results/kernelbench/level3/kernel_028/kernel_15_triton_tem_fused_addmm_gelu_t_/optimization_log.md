# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_gelu_t_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 54080 ns — **Best latency**: 10660 ns (**5.0732×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`replace_tl_dot_with_scalar_reduction_for_tiny_mn`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_gelu_t_15`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_028/kernel_15_triton_tem_fused_addmm_gelu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1,1,1), Waves/SM: 0.00
- **Registers/thread**: 44
- **Duration**: 54080 ns
- **SM Throughput**: 0.07 % | **Memory**: 0.69 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 0.34 inst/cycle | **No Eligible**: 91.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| D | reduce_tile_sizes_and_add_autotune_for_tiny_matmul | 1.566× | 0.911× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_tile_sizes... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 54080 ns | 54080 ns | +0 ns (+0.0%) | 59360 ns | +5280 ns (+9.8%) |
| SM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.01 % | -0.06 pp |
| Memory Throughput (%) | 0.69 % | 0.69 % | +0.00 pp | 0.16 % | -0.53 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.22 % | -0.01 pp | 1.56 % | -4.67 pp |
| Registers/Thread | 44.00 register/thread | 44.00 register/thread | +0 | 80.00 register/thread | +36 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.46 % | 91.48 % | +0.02 pp | 94.25 % | +2.79 pp |
| IPC Active | 0.34 inst/cycle | 0.34 inst/cycle | +0.00 | 0.06 inst/cycle | -0.28 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_tl_dot_with_scalar_reduction_for_tiny_mn | 4.185× | 5.073× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | replace_tl_dot_wi... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 54080 ns | 10660 ns | -43420 ns (-80.3%) |
| SM Throughput (%) | 0.07 % | 0.15 % | +0.08 pp |
| Memory Throughput (%) | 0.69 % | 0.27 % | -0.42 pp |
| DRAM Throughput (%) | 0.04 % | 0.20 % | +0.16 pp |
| Achieved Occupancy (%) | 6.23 % | 6.24 % | +0.01 pp |
| Registers/Thread | 44.00 register/thread | 127.00 register/thread | +83 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 91.46 % | 87.27 % | -4.19 pp |
| IPC Active | 0.34 inst/cycle | 0.50 inst/cycle | +0.16 |

---

## Iteration 3

**Best so far**: 5.0732× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 5.0732× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Cumulative Progress

```
Baseline:  54080 ns  (1.000×)
Iter 2     10660 ns  (5.073×)  — replace_tl_dot_with_scalar_reduction_for_tiny_mn

Final best: 10660 ns  (5.0732×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,91 +5,66 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_addmm_gelu_t_15(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 16
+    BLOCK_M : tl.constexpr = 2
     BLOCK_N : tl.constexpr = 16
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 256
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
 
     M = 2
     N = 10
     K = 2048
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 2048
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 2048
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    pid = tl.program_id(0)
 
-    # re-order program ID for better L2 performance
-    width = GROUP_M * grid_n
-    group_id = pid // width
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
-    pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    rm = tl.arange(0, BLOCK_M)   # [0, 1]
+    rn = tl.arange(0, BLOCK_N)   # [0..15]
+    mask_m = rm < M               # [True, True]
+    mask_n = rn < N               # [True]*10 + [False]*6
 
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
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+    for k_start in range(0, K, BLOCK_K):
+        rk = k_start + tl.arange(0, BLOCK_K)  # [k_start..k_start+BLOCK_K-1]
+        mask_k = rk < K
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 2048*idx_m
-        a = tl.load(A + (xindex))
+        # A: shape (2, 2048), row-major stride (2048, 1)
+        # A[m, k] = arg_A + m*2048 + k
+        a_ptrs = arg_A + rm[:, None] * 2048 + rk[None, :]  # [BLOCK_M, BLOCK_K]
+        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 10*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 2048*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B: shape (10, 2048), row-major stride (2048, 1)
+        # B[n, k] = arg_B + n*2048 + k  =>  B^T[k, n] = arg_B + n*2048 + k
+        b_ptrs = arg_B + rn[None, :] * 2048 + rk[:, None]  # [BLOCK_K, BLOCK_N]
+        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # Register-only outer product accumulation: [BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]
+        # Use tl.expand_dims to create 3D tensors and sum over K axis
+        # a: [BLOCK_M, BLOCK_K] -> [BLOCK_M, BLOCK_K, 1]
+        # b: [BLOCK_K, BLOCK_N] -> [1, BLOCK_K, BLOCK_N]
+        a3 = tl.expand_dims(a, 2)   # [BLOCK_M, BLOCK_K, 1]
+        b3 = tl.expand_dims(b, 0)   # [1, BLOCK_K, BLOCK_N]
+        acc += tl.sum(a3 * b3, axis=1)  # [BLOCK_M, BLOCK_N]
 
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
+    # epilogue: add bias and store
     idx_m = rm[:, None]
     idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
+    mask = mask_m[:, None] & mask_n[None, :]
 
-    # inductor generates a suffix
-    xindex = idx_n + 10*idx_m
+    xindex = idx_n + 10 * idx_m
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
