# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_relu_t_0` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1380000 ns — **Best latency**: 1370000 ns (**1.0073×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_with_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_relu_t_0`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_003/kernel_0_triton_tem_fused_addmm_relu_t_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.32
- **Registers/thread**: 168
- **Duration**: 1380000 ns
- **SM Throughput**: 27.51 % | **Memory**: 62.00 % | **DRAM**: 1.03 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.23 inst/cycle | **No Eligible**: 69.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_autotune_tile_sizes | 1.002× | 1.007× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 1380000 ns | 1370000 ns | -10000 ns (-0.7%) |
| SM Throughput (%) | 27.51 % | 27.64 % | +0.13 pp |
| Memory Throughput (%) | 62.00 % | 62.09 % | +0.09 pp |
| DRAM Throughput (%) | 1.03 % | 1.04 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.32 | 0.32 | +0.00 |
| No Eligible (%) | 69.28 % | 69.21 % | -0.07 pp |
| IPC Active | 1.23 inst/cycle | 1.23 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  1380000 ns  (1.000×)
Iter 1     1370000 ns  (1.007×)  — tma_loads_with_autotune_tile_sizes

Final best: 1370000 ns  (1.0073×)
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
 def triton_tem_fused_addmm_relu_t_0(arg_A, arg_B, in_ptr2, out_ptr1):
@@ -44,37 +45,24 @@

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
+    # Use block_ptr (TMA-style) loads to offload address generation
+    A_block_ptr = tl.make_block_ptr(
+        base=A, shape=(M, K), strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
+    B_block_ptr = tl.make_block_ptr(
+        base=B, shape=(K, N), strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 8192*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 1024*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 8192*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -90,8 +78,6 @@

     tmp2 = tl.full([1], 0, tl.int32)
     tmp3 = tl.maximum(tmp2, tmp1)
     tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 1024*idx_m, [BLOCK_M, BLOCK_N])), tmp3, mask)
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
