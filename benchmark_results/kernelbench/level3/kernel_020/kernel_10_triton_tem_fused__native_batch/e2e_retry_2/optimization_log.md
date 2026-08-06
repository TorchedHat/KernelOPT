# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38850 ns — **Best latency**: 38210 ns (**1.0167×** vs baseline)
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
- **Duration**: 38850 ns
- **SM Throughput**: 14.75 % | **Memory**: 60.48 % | **DRAM**: 7.27 %
- **Achieved Occupancy**: 13.41 %
- **IPC Active**: 0.85 inst/cycle | **No Eligible**: 78.71 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.009× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 1.005× | 1.017× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 38850 ns | 38500 ns | -350 ns (-0.9%) | 38210 ns | -640 ns (-1.6%) |
| SM Throughput (%) | 14.75 % | 14.81 % | +0.06 pp | 14.76 % | +0.01 pp |
| Memory Throughput (%) | 60.48 % | 60.73 % | +0.25 pp | 61.33 % | +0.85 pp |
| DRAM Throughput (%) | 7.27 % | 7.33 % | +0.06 pp | 7.39 % | +0.12 pp |
| Achieved Occupancy (%) | 13.41 % | 13.39 % | -0.02 pp | 13.30 % | -0.11 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.53 | 0.53 | +0.00 | 0.53 | +0.00 |
| No Eligible (%) | 78.71 % | 78.63 % | -0.08 pp | 78.72 % | +0.01 pp |
| IPC Active | 0.85 inst/cycle | 0.85 inst/cycle | +0.00 | 0.85 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  38850 ns  (1.000×)
Iter 1     38210 ns  (1.017×)  — tma_loads_with_autotune_tile_sizes

Final best: 38210 ns  (1.0167×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 32
@@ -16,19 +26,14 @@

     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
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
@@ -44,50 +49,45 @@

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
+    # Create block pointers for TMA loads (eliminates strided scalar pointer arithmetic)
+    # arg_A: shape (36000, 96), strides (96, 1) — row-major
+    a_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(96, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # arg_B: shape (24, 96, 1, 1) viewed as (K=96, N=24), strides stride_k=1, stride_n=96
+    b_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 96),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(a_block_ptr, boundary_check=(0, 1))
+        b = tl.load(b_block_ptr, boundary_check=(0, 1))
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
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
-
-    # rematerialize rm and rn to save registers
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
-    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    idx_m = rm[:, None]
-    idx_n = rn[None, :]
-    mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 24*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    # TMA store via block pointer
+    out_block_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(24, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
+    tl.store(out_block_ptr, acc, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
