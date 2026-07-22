# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_addmm_native_layer_norm_t_view_1` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12390000 ns — **Best latency**: 12390000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_addmm_native_layer_norm_t_view_1`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_044/kernel_1_triton_tem_fused_addmm_native_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (9216,1,1), Waves/SM: 34.91
- **Registers/thread**: 255
- **Duration**: 12390000 ns
- **SM Throughput**: 41.49 % | **Memory**: 60.57 % | **DRAM**: 1.48 %
- **Achieved Occupancy**: 12.46 %
- **IPC Active**: 1.67 inst/cycle | **No Eligible**: 58.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.160× | 0.159× | ✓ |  |
| B | tma_loads_with_autotune_tile_sizes | 0.959× | 0.186× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_au... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12390000 ns | 78130000 ns | +65740000 ns (+530.6%) | 66550000 ns | +54160000 ns (+437.1%) |
| SM Throughput (%) | 41.49 % | 8.03 % | -33.46 pp | 9.41 % | -32.08 pp |
| Memory Throughput (%) | 60.57 % | 16.39 % | -44.18 pp | 18.82 % | -41.75 pp |
| DRAM Throughput (%) | 1.48 % | 12.73 % | +11.25 pp | 13.12 % | +11.64 pp |
| Achieved Occupancy (%) | 12.46 % | 6.25 % | -6.21 pp | 6.25 % | -6.21 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 69.82 | +34.91 | 69.82 | +34.91 |
| No Eligible (%) | 58.10 % | 91.92 % | +33.82 pp | 90.56 % | +32.46 pp |
| IPC Active | 1.67 inst/cycle | 0.32 inst/cycle | -1.35 | 0.38 inst/cycle | -1.29 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | revert_tma_block_ptr_to_scalar_pointer_arithmetic | 0.948× | 0.190× | ✓ |  |
| C | restore_block_k32_and_autotune_smaller_tiles_for_occupancy | 0.999× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | revert_tma_block_... | Δ | restore_block_k32... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 12390000 ns | 65170000 ns | +52780000 ns (+426.0%) | 12360000 ns | -30000 ns (-0.2%) |
| SM Throughput (%) | 41.49 % | 9.51 % | -31.98 pp | 41.49 % | +0.00 pp |
| Memory Throughput (%) | 60.57 % | 19.16 % | -41.41 pp | 60.57 % | +0.00 pp |
| DRAM Throughput (%) | 1.48 % | 12.84 % | +11.36 pp | 1.54 % | +0.06 pp |
| Achieved Occupancy (%) | 12.46 % | 6.25 % | -6.21 pp | 12.46 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 34.91 | 69.82 | +34.91 | 34.91 | +0.00 |
| No Eligible (%) | 58.10 % | 90.41 % | +32.31 pp | 58.02 % | -0.08 pp |
| IPC Active | 1.67 inst/cycle | 0.38 inst/cycle | -1.29 | 1.68 inst/cycle | +0.01 |


### Key Learnings

#### ❌ Plan A: `revert_tma_block_ptr_to_scalar_pointer_arithmetic` (0.190×)

Replacing tl.make_block_ptr with scalar offset index arithmetic was expected to reduce dynamic shared memory from 131 KB to ~65 KB per block, restoring Block Limit Shared Mem from 1 to 3 and doubling occupancy. However, after the revert, dynamic shared memory remained at 131 KB/block and Block Limit Shared Mem stayed at 1, yielding no occupancy improvement (6.25% before and after). The kernel duration barely changed (66.55 ms → 65.17 ms, speedup 0.19x relative to the original 12.39 ms baseline), confirming that the shared memory footprint is driven by something other than the tl.make_block_ptr descriptors alone — likely the large tile sizes (BLOCK_M=128, BLOCK_N=128, BLOCK_K=64) themselves. The CPIStall reason also shifted from instruction-cache misses (46.9% of stalls) to L1TEX scoreboard dependency stalls (61.7%), indicating scalar pointer arithmetic introduced more memory dependency stalls without resolving the occupancy bottleneck.

> **Applicable when**: Never — this revert did not achieve the expected shared memory reduction. The 131 KB/block footprint persists regardless of whether tl.make_block_ptr or scalar pointer arithmetic is used, because the bottleneck is the tile size (BLOCK_M=128, BLOCK_N=128, BLOCK_K=64), not the pointer style.

> **Avoid when**: When the kernel's shared memory footprint is dominated by large tile sizes (BLOCK_M × BLOCK_N × BLOCK_K × dtype_size) rather than TMA descriptor metadata. Reverting tl.make_block_ptr to scalar pointers will not reduce shared memory if the tile dimensions are unchanged. Additionally, scalar pointer arithmetic introduces more L1TEX scoreboard dependency stalls (CPIStall shifted from 46.9% instruction-cache stalls to 61.7% L1TEX stalls), making the kernel slightly worse in instruction-level latency hiding.

---

## Cumulative Progress

```
Baseline:  12390000 ns  (1.000×)

Final best: 12390000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,88 +5,88 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=16, num_stages=3),
+        triton.Config({}, num_warps=16, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_addmm_native_layer_norm_t_view_1(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 128
-    BLOCK_K : tl.constexpr = 32
+    BLOCK_K : tl.constexpr = 64
     GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
-    A = arg_A
-    B = arg_B
 
     M = 65536
     N = 2304
     K = 768
+
     if M * N == 0:
-        # early exit due to zero-size input(s)
         return
-    stride_am = 768
-    stride_ak = 1
-    stride_bk = 1
-    stride_bn = 768
 
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
+    # TMA block pointers: address generation offloaded to hardware unit
+    # A has shape (M=65536, K=768), strides (768, 1) — row-major
+    A_ptr = tl.make_block_ptr(
+        base=arg_A,
+        shape=(M, K),
+        strides=(K, 1),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B has shape (2304, 768), strides (768, 1)
+    # Viewed as (K=768, N=2304): element [k, n] = B[n, k] => strides=(1, K=768)
+    B_ptr = tl.make_block_ptr(
+        base=arg_B,
+        shape=(K, N),
+        strides=(1, K),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+    # Output block pointer
+    out_block_ptr = tl.make_block_ptr(
+        base=out_ptr0,
+        shape=(M, N),
+        strides=(N, 1),
+        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
+        block_shape=(BLOCK_M, BLOCK_N),
+        order=(1, 0),
+    )
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
+    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
 
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
+        b = tl.load(B_ptr, boundary_check=(0, 1), eviction_policy='evict_first')
+        acc = tl.dot(a, b, acc, allow_tf32=False, out_dtype=tl.float32)
+        A_ptr = tl.advance(A_ptr, (0, BLOCK_K))
+        B_ptr = tl.advance(B_ptr, (BLOCK_K, 0))
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex), eviction_policy='evict_first')
-
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 2304*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_first')
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
-    xindex = idx_n + 2304*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
-
+    tl.store(out_block_ptr, acc, boundary_check=(0, 1), eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
