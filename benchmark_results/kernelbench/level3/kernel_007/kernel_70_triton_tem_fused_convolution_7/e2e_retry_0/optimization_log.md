# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_70` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 55550 ns — **Best latency**: 55390 ns (**1.0029×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_with_block_ptr_and_autotune`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_70`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_70_triton_tem_fused_convolution_7/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (124,1,1), Waves/SM: 0.16
- **Registers/thread**: 80
- **Duration**: 55550 ns
- **SM Throughput**: 12.42 % | **Memory**: 84.58 % | **DRAM**: 1.62 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.55 inst/cycle | **No Eligible**: 86.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.003× | 1.000× | ✓ |  |
| B | tma_loads_with_block_ptr_and_autotune | 1.006× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 55550 ns | 55580 ns | +30 ns (+0.1%) | 55390 ns | -160 ns (-0.3%) |
| SM Throughput (%) | 12.42 % | 12.42 % | +0.00 pp | 12.53 % | +0.11 pp |
| Memory Throughput (%) | 84.58 % | 84.58 % | +0.00 pp | 85.13 % | +0.55 pp |
| DRAM Throughput (%) | 1.62 % | 1.62 % | +0.00 pp | 1.63 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 79.00 register/thread | -1 |
| Waves/SM | 0.16 | 0.16 | +0.00 | 0.16 | +0.00 |
| No Eligible (%) | 86.18 % | 86.19 % | +0.01 pp | 86.07 % | -0.11 pp |
| IPC Active | 0.55 inst/cycle | 0.55 inst/cycle | +0.00 | 0.56 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  55550 ns  (1.000×)
Iter 1     55390 ns  (1.003×)  — tma_loads_with_block_ptr_and_autotune

Final best: 55390 ns  (1.0029×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,94 +5,74 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_convolution_70(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
+    M : tl.constexpr = 1960
+    N : tl.constexpr = 128
+    K : tl.constexpr = 528
 
-    M = 1960
-    N = 128
-    K = 528
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 528
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 528
 
-    # based on triton.ops.matmul
-    pid = tl.program_id(0).to(INDEX_DTYPE)
-    grid_m = (M + BLOCK_M - 1) // BLOCK_M
-    grid_n = (N + BLOCK_N - 1) // BLOCK_N
+    pid = tl.program_id(0)
+    grid_m = tl.cdiv(M, BLOCK_M)
+    grid_n = tl.cdiv(N, BLOCK_N)
 
     # re-order program ID for better L2 performance
     width = GROUP_M * grid_n
     group_id = pid // width
     group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
     pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
+    pid_n = (pid % width) // group_size
+
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
-    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+    # TMA block pointers — contiguous tile access, eliminates bank conflicts
+    # A: shape (M=1960, K=528), strides (528, 1) — row-major
+    A_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(528, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B: original tensor shape (128, 528) with strides (528, 1),
+    # viewed as (K=528, N=128) transposed: stride_bk=1, stride_bn=528
+    B_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, 528),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(1, 0),
+    )
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_ptr, boundary_check=(0, 1), padding_option='zero')
+        b = tl.load(B_ptr, boundary_check=(0, 1), padding_option='zero')
+        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
+        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
+        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 528*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 128*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 528*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
+    # epilogue: bias add and store
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
     idx_m = rm[:, None]
     idx_n = rn[None, :]
     mask = (idx_m < M) & (idx_n < N)
 
-    # inductor generates a suffix
-    xindex = idx_n + 128*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc + tmp0
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+    bias = tl.load(in_ptr0 + idx_n, mask=(idx_n < N), other=0.0, eviction_policy='evict_last')
+    tmp1 = acc + bias
+    tl.store(out_ptr0 + idx_n + N * idx_m, tmp1, mask=mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
