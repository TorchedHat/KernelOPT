# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3400000 ns — **Best latency**: 3390000 ns (**1.0029×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`memory_coalescing_via_block_k_revert_and_b_order_fix`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_006/kernel_5_triton_tem_fused_convolution_5/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (3920,1,1), Waves/SM: 14.85
- **Registers/thread**: 255
- **Duration**: 3400000 ns
- **SM Throughput**: 40.78 % | **Memory**: 59.57 % | **DRAM**: 7.11 %
- **Achieved Occupancy**: 12.43 %
- **IPC Active**: 1.66 inst/cycle | **No Eligible**: 58.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ |  |
| B | tma_loads_with_reduced_register_pressure | 1.002× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_re... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3400000 ns | 3400000 ns | +0 ns (+0.0%) | 3390000 ns | -10000 ns (-0.3%) |
| SM Throughput (%) | 40.78 % | 40.66 % | -0.12 pp | 40.61 % | -0.17 pp |
| Memory Throughput (%) | 59.57 % | 59.40 % | -0.17 pp | 59.06 % | -0.51 pp |
| DRAM Throughput (%) | 7.11 % | 7.12 % | +0.01 pp | 7.01 % | -0.10 pp |
| Achieved Occupancy (%) | 12.43 % | 12.44 % | +0.01 pp | 12.48 % | +0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 58.51 % | 58.62 % | +0.11 pp | 58.76 % | +0.25 pp |
| IPC Active | 1.66 inst/cycle | 1.66 inst/cycle | +0.00 | 1.65 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_with_software_pipelining_num_stages | 1.018× | 1.003× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | tma_loads_with_so... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3400000 ns | 3390000 ns | -10000 ns (-0.3%) |
| SM Throughput (%) | 40.78 % | 40.63 % | -0.15 pp |
| Memory Throughput (%) | 59.57 % | 59.08 % | -0.49 pp |
| DRAM Throughput (%) | 7.11 % | 7.01 % | -0.10 pp |
| Achieved Occupancy (%) | 12.43 % | 12.48 % | +0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 |
| No Eligible (%) | 58.51 % | 58.79 % | +0.28 pp |
| IPC Active | 1.66 inst/cycle | 1.65 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | reduce_register_pressure_via_safe_block_n_reduction_to_96 | 0.870× | 0.163× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3400000 ns | 20870000 ns | +17470000 ns (+513.8%) |
| SM Throughput (%) | 40.78 % | 8.58 % | -32.20 pp |
| Memory Throughput (%) | 59.57 % | 17.37 % | -42.20 pp |
| DRAM Throughput (%) | 7.11 % | 13.17 % | +6.06 pp |
| Achieved Occupancy (%) | 12.43 % | 6.25 % | -6.18 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 14.85 | 29.70 | +14.85 |
| No Eligible (%) | 58.51 % | 91.32 % | +32.81 pp |
| IPC Active | 1.66 inst/cycle | 0.35 inst/cycle | -1.31 |


### Key Learnings

#### ❌ Plan D: `reduce_register_pressure_via_safe_block_n_reduction_to_96` (0.163×)

Changing BLOCK_K from 32 to 64 while keeping BLOCK_N=128 (or changing to 96) doubled the dynamic shared memory per block from 65 KB to 131 KB, causing Block Limit Shared Mem to drop from 3 to 1. This halved theoretical occupancy from 12.5% to 6.25%, reduced active warps/scheduler from 2.0 to 1.0, and collapsed eligible warps/scheduler from 0.75 to 0.09 — causing 91.3% of cycles to have no eligible warp (up from 58.8%). The kernel duration increased 6x (3.39ms → 20.87ms). When shared memory per block is already near the per-SM limit, increasing BLOCK_K to improve arithmetic intensity will instead destroy occupancy by making Block Limit Shared Mem the binding constraint.

> **Applicable when**: Never apply this specific change (BLOCK_K 32->64 in this kernel configuration). As a general signal: when dynamic shared memory per block is already ≥50% of the per-SM shared memory budget, increasing tile dimensions that grow shared memory will reduce Block Limit Shared Mem and hurt occupancy.

> **Avoid when**: Do not increase BLOCK_K when the kernel is already near the shared memory per-SM limit. The before state had 65KB/block with Block Limit Shared Mem=3; doubling BLOCK_K pushed it to 131KB/block with Block Limit Shared Mem=1, causing a 6x slowdown. This pattern is harmful whenever shared memory is the binding occupancy constraint.

---

## Iteration 4

**Best so far**: 1.0029× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_via_block_k_revert_and_b_order_fix | 1.016× | 1.003× | ✓ |  |
| D | revert_block_k_to_restore_shared_memory_occupancy | 1.017× | 1.003× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ | revert_block_k_to... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3400000 ns | 3390000 ns | -10000 ns (-0.3%) | 3390000 ns | -10000 ns (-0.3%) |
| SM Throughput (%) | 40.78 % | 40.62 % | -0.16 pp | 40.59 % | -0.19 pp |
| Memory Throughput (%) | 59.57 % | 59.07 % | -0.50 pp | 59.02 % | -0.55 pp |
| DRAM Throughput (%) | 7.11 % | 7.01 % | -0.10 pp | 6.99 % | -0.12 pp |
| Achieved Occupancy (%) | 12.43 % | 12.47 % | +0.04 pp | 12.48 % | +0.05 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 58.51 % | 58.72 % | +0.21 pp | 58.75 % | +0.24 pp |
| IPC Active | 1.66 inst/cycle | 1.65 inst/cycle | -0.01 | 1.65 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  3400000 ns  (1.000×)
Iter 1     3390000 ns  (1.003×)  — tma_loads_with_reduced_register_pressure

Final best: 3390000 ns  (1.0029×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,14 +5,28 @@

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
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=5),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_5(in_ptr0, arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
+    EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -44,37 +58,27 @@

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
+    # TMA block_ptr setup before k loop
+    # A: shape (M, K), strides (K, 1) — row-major, A[m,k] = A_ptr + m*480 + k
+    A_block_ptr = tl.make_block_ptr(
+        base=arg_A, shape=(M, K), strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
+    # B: shape (K, N), strides (1, K) — column-major, B[k,n] = B_ptr + k + n*480
+    B_block_ptr = tl.make_block_ptr(
+        base=arg_B, shape=(K, N), strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N), order=(0, 1))
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
-
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 480*idx_m
-        a = tl.load(A + (xindex))
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+    # TMA loads inside k loop (replaces both scalar loads)
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
+        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -88,8 +92,6 @@

     tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, [BLOCK_M, BLOCK_N])), mask, eviction_policy='evict_last')
     tmp1 = acc + tmp0
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), tmp1, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
