# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 54620 ns — **Best latency**: 54620 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_24_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (72,1,1), Waves/SM: 0.14
- **Registers/thread**: 128
- **Duration**: 54620 ns
- **SM Throughput**: 12.01 % | **Memory**: 46.65 % | **DRAM**: 1.70 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.92 inst/cycle | **No Eligible**: 76.91 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_and_fix_smem_bank_conflicts | 0.988× | 1.008× | ✓ |  |
| C | eviction_policies_streaming_loads | 0.999× | 1.004× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 54620 ns | 54210 ns | -410 ns (-0.8%) | 54430 ns | -190 ns (-0.3%) |
| SM Throughput (%) | 12.01 % | 12.36 % | +0.35 pp | 12.00 % | -0.01 pp |
| Memory Throughput (%) | 46.65 % | 47.06 % | +0.41 pp | 46.61 % | -0.04 pp |
| DRAM Throughput (%) | 1.70 % | 1.71 % | +0.01 pp | 1.71 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.14 | 0.14 | +0.00 | 0.14 | +0.00 |
| No Eligible (%) | 76.91 % | 76.45 % | -0.46 pp | 76.89 % | -0.02 pp |
| IPC Active | 0.92 inst/cycle | 0.94 inst/cycle | +0.02 | 0.92 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_smem_bank_conflicts_via_block_ptr_order | 0.993× | 1.008× | ✓ |  |
| B | tma_loads_fix_smem_order_and_pipeline_stages | 0.994× | 1.006× | ✓ |  |
| C | increase_num_warps_to_hide_mio_stall_latency | 0.999× | 1.001× | ✓ |  |
| D | tma_loads_via_block_ptr | 0.997× | 1.008× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_smem_bank_con... | Δ | tma_loads_fix_sme... | Δ | increase_num_warp... | Δ | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 54620 ns | 54180 ns | -440 ns (-0.8%) | 54270 ns | -350 ns (-0.6%) | 54560 ns | -60 ns (-0.1%) | 54210 ns | -410 ns (-0.8%) |
| SM Throughput (%) | 12.01 % | 12.33 % | +0.32 pp | 12.35 % | +0.34 pp | 11.99 % | -0.02 pp | 12.36 % | +0.35 pp |
| Memory Throughput (%) | 46.65 % | 46.98 % | +0.33 pp | 47.04 % | +0.39 pp | 46.56 % | -0.09 pp | 47.07 % | +0.42 pp |
| DRAM Throughput (%) | 1.70 % | 1.71 % | +0.01 pp | 1.71 % | +0.01 pp | 1.70 % | +0.00 pp | 1.71 % | +0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.14 | 0.14 | +0.00 | 0.14 | +0.00 | 0.14 | +0.00 | 0.14 | +0.00 |
| No Eligible (%) | 76.91 % | 76.40 % | -0.51 pp | 76.43 % | -0.48 pp | 76.96 % | +0.05 pp | 76.42 % | -0.49 pp |
| IPC Active | 0.92 inst/cycle | 0.94 inst/cycle | +0.02 | 0.94 inst/cycle | +0.02 | 0.92 inst/cycle | +0.00 | 0.94 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  54620 ns  (1.000×)

Final best: 54620 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,17 +5,30 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=5),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=5),
+        triton.Config({}, num_warps=2, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_24(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
-    ACC_TYPE : tl.constexpr = tl.float32
-    BLOCK_M : tl.constexpr = 64
-    BLOCK_N : tl.constexpr = 64
-    BLOCK_K : tl.constexpr = 32
-    GROUP_M : tl.constexpr = 8
-    ALLOW_TF32 : tl.constexpr = False
-    INDEX_DTYPE : tl.constexpr = tl.int32
+    EVEN_K: tl.constexpr = True
+    USE_FAST_ACCUM: tl.constexpr = False
+    ACC_TYPE: tl.constexpr = tl.float32
+    BLOCK_M: tl.constexpr = 64
+    BLOCK_N: tl.constexpr = 64
+    BLOCK_K: tl.constexpr = 32
+    GROUP_M: tl.constexpr = 8
+    ALLOW_TF32: tl.constexpr = False
+    INDEX_DTYPE: tl.constexpr = tl.int32
     A = arg_A
     B = arg_B
 
@@ -44,37 +57,34 @@

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
+    # Use tl.make_block_ptr for TMA-style coalesced loads (eliminates bank conflicts)
+    # A has shape (M, K) with strides (480, 1) — row-major
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+    # B has shape (K, N) with strides (1, 480) — column-major K dimension
+    B_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
-    for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-
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
-        xindex = idx_n + 112*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 480*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
-
+    for _ in range(tl.cdiv(K, BLOCK_K)):
+        a = tl.load(A_block_ptr, boundary_check=(0, 1))
+        b = tl.load(B_block_ptr, boundary_check=(0, 1))
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
+        B_block_ptr = tl.advance(B_block_ptr, (BLOCK_K, 0))
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -84,10 +94,8 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 112*idx_m
+    xindex = idx_n + 112 * idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
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
