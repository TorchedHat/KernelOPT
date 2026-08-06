# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_t_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4010000 ns — **Best latency**: 3990000 ns (**1.0050×** vs baseline)
- **Iterations completed**: 1
- **Outcome**: Best configuration **`tma_loads_with_block_ptr_and_autotuned_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_t_4`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_033/kernel_3_triton_tem_fused_addmm_t_4/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 255
- **Duration**: 4010000 ns
- **SM Throughput**: 37.75 % | **Memory**: 86.16 % | **DRAM**: 3.14 %
- **Achieved Occupancy**: 12.25 %
- **IPC Active**: 1.55 inst/cycle | **No Eligible**: 61.24 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.999× | 1.000× | ✓ |  |
| B | tma_loads_with_block_ptr_and_autotuned_tile_sizes | 1.009× | 1.005× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_bl... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 4010000 ns | 4010000 ns | +0 ns (+0.0%) | 3990000 ns | -20000 ns (-0.5%) |
| SM Throughput (%) | 37.75 % | 37.79 % | +0.04 pp | 37.87 % | +0.12 pp |
| Memory Throughput (%) | 86.16 % | 86.25 % | +0.09 pp | 85.86 % | -0.30 pp |
| DRAM Throughput (%) | 3.14 % | 3.14 % | +0.00 pp | 3.12 % | -0.02 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.22 % | -0.03 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 254.00 register/thread | -1 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 61.24 % | 61.25 % | +0.01 pp | 61.18 % | -0.06 pp |
| IPC Active | 1.55 inst/cycle | 1.55 inst/cycle | +0.00 | 1.55 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  4010000 ns  (1.000×)
Iter 1     3990000 ns  (1.005×)  — tma_loads_with_block_ptr_and_autotuned_tile_sizes

Final best: 3990000 ns  (1.0050×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,91 +5,67 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused_addmm_t_4(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 64
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
+    ALLOW_TF32: tl.constexpr = False
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 64
+    BLOCK_N: tl.constexpr = 128
+    BLOCK_K: tl.constexpr = 64
+    GROUP_M: tl.constexpr = 8
+    M: tl.constexpr = 256
+    N: tl.constexpr = 8192
+    K: tl.constexpr = 16384
 
-    M = 256
-    N = 8192
-    K = 16384
-    if M * N == 0:
-        # early exit due to zero-size input(s)
-        return
-    stride_am = 16384
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 16384
-
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
-    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
+    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
     pid_m = group_id * GROUP_M + (pid % group_size)
-    pid_n = (pid % width) // (group_size)
-    tl.assume(pid_m >= 0)
-    tl.assume(pid_n >= 0)
+    pid_n = (pid % width) // group_size
 
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
+    # TMA block_ptr for A: shape (M, K), row-major strides (K, 1)
+    A_block_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(K, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # TMA block_ptr for B: stored as (N, K) with strides (N, 1) per element
+    # B[k,n] = arg_B[n,k] => virtual shape (K, N), strides (1, N)
+    B_block_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, K),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
+    for _ in range(0, tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
+        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 16384*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 8192*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16384*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
-
-    # rematerialize rm and rn to save registers
+    # epilogue: bias broadcast + store
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
     idx_m = rm[:, None]
     idx_n = rn[None, :]
     mask = (idx_m < M) & (idx_n < N)
-
-    # inductor generates a suffix
-    xindex = idx_n + 8192*idx_m
-    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
-    tmp1 = acc + tmp0
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
+    bias = tl.load(in_ptr0 + rn, mask=rn < N, eviction_policy='evict_last')
+    tmp1 = acc + bias[None, :]
+    tl.store(out_ptr0 + (idx_n + N * idx_m), tmp1, mask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=1, N=2, K=2, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
