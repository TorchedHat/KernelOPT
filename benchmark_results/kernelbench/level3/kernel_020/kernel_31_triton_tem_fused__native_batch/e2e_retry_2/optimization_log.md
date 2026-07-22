# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 36640 ns — **Best latency**: 36860 ns (**0.9940×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_31_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (90,1,1), Waves/SM: 0.23
- **Registers/thread**: 168
- **Duration**: 36640 ns
- **SM Throughput**: 17.69 % | **Memory**: 39.71 % | **DRAM**: 0.81 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.16 inst/cycle | **No Eligible**: 70.63 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotuned_tile_sizes | 1.042× | 0.994× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 36640 ns | 36860 ns | +220 ns (+0.6%) |
| SM Throughput (%) | 17.69 % | 17.42 % | -0.27 pp |
| Memory Throughput (%) | 39.71 % | 39.68 % | -0.03 pp |
| DRAM Throughput (%) | 0.81 % | 0.79 % | -0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 70.63 % | 70.85 % | +0.22 pp |
| IPC Active | 1.16 inst/cycle | 1.17 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  36640 ns  (1.000×)

Final best: 36860 ns  (0.9940×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,18 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_31(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -44,37 +56,32 @@

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
 
+    # Create TMA block pointers before the k-loop for async pipelining
+    a_desc = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    b_desc = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
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
-        xindex = idx_n + 1152*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-        
+        a = tl.load(a_desc, boundary_check=(0, 1))
+        b = tl.load(b_desc, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        a_desc = tl.advance(a_desc, (0, BLOCK_K))
+        b_desc = tl.advance(b_desc, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +93,6 @@

     # inductor generates a suffix
     xindex = idx_n + 1152*idx_m
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
