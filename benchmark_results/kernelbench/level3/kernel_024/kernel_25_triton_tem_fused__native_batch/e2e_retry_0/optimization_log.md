# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 16740 ns — **Best latency**: 10400 ns (**1.6096×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_with_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_024/kernel_25_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (36,1,1), Waves/SM: 0.03
- **Registers/thread**: 56
- **Duration**: 16740 ns
- **SM Throughput**: 2.20 % | **Memory**: 21.68 % | **DRAM**: 1.63 %
- **Achieved Occupancy**: 6.24 %
- **IPC Active**: 0.36 inst/cycle | **No Eligible**: 90.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.002× | 1.008× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 1.456× | 1.610× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 16740 ns | 16610 ns | -130 ns (-0.8%) | 10400 ns | -6340 ns (-37.9%) |
| SM Throughput (%) | 2.20 % | 2.20 % | +0.00 pp | 1.89 % | -0.31 pp |
| Memory Throughput (%) | 21.68 % | 21.64 % | -0.04 pp | 17.57 % | -4.11 pp |
| DRAM Throughput (%) | 1.63 % | 1.65 % | +0.02 pp | 2.63 % | +1.00 pp |
| Achieved Occupancy (%) | 6.24 % | 6.24 % | +0.00 pp | 3.12 % | -3.12 pp |
| Registers/Thread | 56.00 register/thread | 56.00 register/thread | +0 | 56.00 register/thread | +0 |
| Waves/SM | 0.03 | 0.03 | +0.00 | 0.02 | -0.01 |
| No Eligible (%) | 90.91 % | 90.91 % | +0.00 pp | 83.09 % | -7.82 pp |
| IPC Active | 0.36 inst/cycle | 0.36 inst/cycle | +0.00 | 0.33 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  16740 ns  (1.000×)
Iter 1     10400 ns  (1.610×)  — tma_loads_with_autotune_tile_sizes

Final best: 10400 ns  (1.6096×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,17 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(configs=[
+    triton.Config({}, num_warps=2, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=3),
+    triton.Config({}, num_warps=4, num_stages=4),
+    triton.Config({}, num_warps=4, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=4),
+    triton.Config({}, num_warps=8, num_stages=3),
+    triton.Config({}, num_warps=2, num_stages=2),
+    triton.Config({}, num_warps=8, num_stages=2),
+], key=[])
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_mean_relu_sigmoid_25(arg_A, arg_B, out_ptr1):
     EVEN_K : tl.constexpr = True
@@ -44,37 +55,32 @@

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
+    # Use tl.make_block_ptr for TMA-style loads
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1)
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
-        xindex = idx_n + 288*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1152*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 288*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(a, b, acc, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -87,8 +93,6 @@

     xindex = idx_n + 1152*idx_m
     tmp0 = tl.sigmoid(acc)
     tl.store(out_ptr1 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp0, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
