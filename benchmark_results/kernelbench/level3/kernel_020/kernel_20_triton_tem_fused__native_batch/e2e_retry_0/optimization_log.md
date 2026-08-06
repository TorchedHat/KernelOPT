# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 31620 ns — **Best latency**: 31490 ns (**1.0041×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_via_block_ptr_with_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_20_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (123,1,1), Waves/SM: 0.23
- **Registers/thread**: 128
- **Duration**: 31620 ns
- **SM Throughput**: 19.52 % | **Memory**: 74.65 % | **DRAM**: 4.89 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.91 inst/cycle | **No Eligible**: 77.29 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_with_autotune | 1.008× | 1.004× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 31620 ns | 31490 ns | -130 ns (-0.4%) |
| SM Throughput (%) | 19.52 % | 19.62 % | +0.10 pp |
| Memory Throughput (%) | 74.65 % | 75.70 % | +1.05 pp |
| DRAM Throughput (%) | 4.89 % | 4.91 % | +0.02 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.23 | 0.23 | +0.00 |
| No Eligible (%) | 77.29 % | 77.28 % | -0.01 pp |
| IPC Active | 0.91 inst/cycle | 0.91 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  31620 ns  (1.000×)
Iter 1     31490 ns  (1.004×)  — tma_loads_via_block_ptr_with_autotune

Final best: 31490 ns  (1.0041×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_20(arg_A, arg_B, out_ptr0):
@@ -29,6 +30,8 @@

     stride_ak = 1
     stride_bk = 1
     stride_bn = 240
+    stride_cm = 40
+    stride_cn = 1
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -44,53 +47,42 @@

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
+    # Create block pointers for TMA-style loads
+    A_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0)
+    )
+    B_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0)
+    )
+    C_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(stride_cm, stride_cn),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0)
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
+        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 240*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 40*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
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
-    xindex = idx_n + 40*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(C_ptr, acc, boundary_check=(0, 1))
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
