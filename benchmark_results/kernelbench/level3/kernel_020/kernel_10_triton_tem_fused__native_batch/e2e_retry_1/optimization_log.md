# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38620 ns — **Best latency**: 38370 ns (**1.0065×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_with_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_10_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (282,1,1), Waves/SM: 0.53
- **Registers/thread**: 128
- **Duration**: 38620 ns
- **SM Throughput**: 14.68 % | **Memory**: 60.19 % | **DRAM**: 7.31 %
- **Achieved Occupancy**: 13.41 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.65 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.008× | 1.006× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 1.018× | 1.006× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 38620 ns | 38400 ns | -220 ns (-0.6%) | 38370 ns | -250 ns (-0.6%) |
| SM Throughput (%) | 14.68 % | 14.72 % | +0.04 pp | 14.50 % | -0.18 pp |
| Memory Throughput (%) | 60.19 % | 60.35 % | +0.16 pp | 61.59 % | +1.40 pp |
| DRAM Throughput (%) | 7.31 % | 7.36 % | +0.05 pp | 7.36 % | +0.05 pp |
| Achieved Occupancy (%) | 13.41 % | 13.41 % | +0.00 pp | 13.35 % | -0.06 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.53 | 0.53 | +0.00 | 0.71 | +0.18 |
| No Eligible (%) | 78.65 % | 78.62 % | -0.03 pp | 79.25 % | +0.60 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 0.83 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  38620 ns  (1.000×)
Iter 1     38370 ns  (1.006×)  — tma_loads_with_autotune_tile_sizes

Final best: 38370 ns  (1.0065×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,30 +5,33 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    ALLOW_TF32 : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 32
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
 
     M = 36000
     N = 24
     K = 96
+
     if M * N == 0:
         # early exit due to zero-size input(s)
         return
-    stride_am = 96
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 96
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -44,37 +47,33 @@

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
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+    # Create block pointers for A (shape M x K, row-major: strides (96, 1))
+    A_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(96, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # Create block pointers for B (shape K x N, column-major: strides (1, 96))
+    B_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 96),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 96*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
+        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,10 +83,8 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 24*idx_m
+    xindex = idx_n + 24 * idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
