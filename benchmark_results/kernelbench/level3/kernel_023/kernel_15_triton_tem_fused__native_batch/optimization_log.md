# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 32290 ns — **Best latency**: 62240 ns (**0.5188×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_15_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (71,1,1), Waves/SM: 0.18
- **Registers/thread**: 168
- **Duration**: 32290 ns
- **SM Throughput**: 13.34 % | **Memory**: 30.02 % | **DRAM**: 3.32 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 1.13 inst/cycle | **No Eligible**: 71.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.995× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 32290 ns | 32350 ns | +60 ns (+0.2%) |
| SM Throughput (%) | 13.34 % | 13.23 % | -0.11 pp |
| Memory Throughput (%) | 30.02 % | 29.75 % | -0.27 pp |
| DRAM Throughput (%) | 3.32 % | 3.31 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 |
| No Eligible (%) | 71.72 % | 71.70 % | -0.02 pp |
| IPC Active | 1.13 inst/cycle | 1.13 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_load_indexing_and_autotune_tile_sizes | 1.019× | 1.015× | ✓ | **✓ best** |
| C | eviction_policies_for_streaming_gemm_loads | 1.000× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_indexi... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 32290 ns | 31810 ns | -480 ns (-1.5%) | 32640 ns | +350 ns (+1.1%) |
| SM Throughput (%) | 13.34 % | 13.43 % | +0.09 pp | 13.28 % | -0.06 pp |
| Memory Throughput (%) | 30.02 % | 30.52 % | +0.50 pp | 29.87 % | -0.15 pp |
| DRAM Throughput (%) | 3.32 % | 3.36 % | +0.04 pp | 3.28 % | -0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 168.00 register/thread | 238.00 register/thread | +70 | 168.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.27 | +0.09 | 0.18 | +0.00 |
| No Eligible (%) | 71.72 % | 71.59 % | -0.13 pp | 71.54 % | -0.18 pp |
| IPC Active | 1.13 inst/cycle | 1.14 inst/cycle | +0.01 | 1.14 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0151× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_with_maxnreg_to_improve_occupancy | 1.020× | 0.546× | ✓ | **✓ best** |
| D | autotune_with_preserved_b_index_and_smem_padding | 1.001× | 0.549× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ | autotune_with_pre... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 32290 ns | 59170 ns | +26880 ns (+83.2%) | 58850 ns | +26560 ns (+82.3%) |
| SM Throughput (%) | 13.34 % | 7.29 % | -6.05 pp | 7.35 % | -5.99 pp |
| Memory Throughput (%) | 30.02 % | 11.01 % | -19.01 pp | 11.16 % | -18.86 pp |
| DRAM Throughput (%) | 3.32 % | 1.82 % | -1.50 pp | 1.83 % | -1.49 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 255.00 register/thread | +87 |
| Waves/SM | 0.18 | 0.13 | -0.05 | 0.18 | +0.00 |
| No Eligible (%) | 71.72 % | 69.74 % | -1.98 pp | 69.17 % | -2.55 pp |
| IPC Active | 1.13 inst/cycle | 0.61 inst/cycle | -0.52 | 0.62 inst/cycle | -0.51 |

---

## Iteration 4

**Best so far**: 1.0151× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_k_to_reduce_k_loop_iterations_and_improve_arithmetic_intensity | 0.814× | 0.162× | ✓ |  |
| B | tma_loads | 1.007× | 0.312× | ✓ |  |
| D | smem_padding_to_eliminate_shared_memory_bank_conflicts | 1.033× | 0.519× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | increase_block_k_... | Δ | tma_loads | Δ | smem_padding_to_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 32290 ns | 199010 ns | +166720 ns (+516.3%) | 103420 ns | +71130 ns (+220.3%) | 62240 ns | +29950 ns (+92.8%) |
| SM Throughput (%) | 13.34 % | 3.18 % | -10.16 pp | 4.42 % | -8.92 pp | 7.26 % | -6.08 pp |
| Memory Throughput (%) | 30.02 % | 6.61 % | -23.41 pp | 7.15 % | -22.87 pp | 12.03 % | -17.99 pp |
| DRAM Throughput (%) | 3.32 % | 0.63 % | -2.69 pp | 1.04 % | -2.28 pp | 1.73 % | -1.59 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 | 255.00 register/thread | +87 | 255.00 register/thread | +87 |
| Waves/SM | 0.18 | 0.54 | +0.36 | 0.18 | +0.00 | 0.13 | -0.05 |
| No Eligible (%) | 71.72 % | 88.01 % | +16.29 pp | 82.16 % | +10.44 pp | 69.15 % | -2.57 pp |
| IPC Active | 1.13 inst/cycle | 0.24 inst/cycle | -0.89 | 0.35 inst/cycle | -0.78 | 0.61 inst/cycle | -0.52 |

---

## Iteration 5

**Best so far**: 1.0151× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | autotune_tile_sizes_as_constexpr_kernel_arguments | 0.994× | 0.546× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 32290 ns | 59170 ns | +26880 ns (+83.2%) |
| SM Throughput (%) | 13.34 % | 7.31 % | -6.03 pp |
| Memory Throughput (%) | 30.02 % | 11.00 % | -19.02 pp |
| DRAM Throughput (%) | 3.32 % | 1.82 % | -1.50 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 168.00 register/thread | 255.00 register/thread | +87 |
| Waves/SM | 0.18 | 0.13 | -0.05 |
| No Eligible (%) | 71.72 % | 68.85 % | -2.87 pp |
| IPC Active | 1.13 inst/cycle | 0.62 inst/cycle | -0.51 |


### Key Learnings

#### ❌ Plan B: `autotune_tile_sizes_as_constexpr_kernel_arguments` (0.546×)

The plan intended to promote BLOCK_M/BLOCK_N/BLOCK_K to tl.constexpr kernel arguments and add them to autotune configs to increase grid coverage, but the actual diff did NOT change the tile sizes or grid — it only reordered autotune configs and restructured the A and B matrix load patterns (replacing the original scalar pointer arithmetic with a different index formula). The grid remained at 71 blocks (0.13 waves), achieved occupancy stayed at 3.12%, and registers stayed at 255/thread — none of the intended occupancy improvements materialized. The restructured B-load changed the index formula from `B + k_offs*1 + n_offs*144` to `B + broadcast(idx_n + 40*idx_m)`, which altered the effective memory access pattern: L1/TEX hit rate dropped from 69.86% to 48.01% and L2 hit rate dropped from 76.0% to 57.17%, while dynamic shared memory per block increased from 24.6 KB to 49.2 KB (reducing Block Limit Shared Mem from 5 to 4). The net result was a 1.83x slowdown relative to the before-state. When the B-matrix has a non-standard stride layout (stride_bk=1, stride_bn=144), any restructuring of its load formula risks silently changing the effective access pattern and degrading cache performance.

> **Applicable when**: Never — this is a negative result. The plan to promote tile sizes to constexpr kernel args was never actually implemented; the diff only restructured load patterns.

> **Avoid when**: When the B-matrix has a non-standard stride layout (e.g., stride_bk=1, stride_bn=144 for a column-major-like K×N view), any modification to the B-load index formula — even one that appears mathematically equivalent — risks changing the effective memory access pattern and collapsing L1/TEX hit rate. Do not restructure B-matrix loads when L1 hit rate is already above 50% and the kernel is register-bound (registers_per_thread=255). Also: promoting tile sizes to constexpr kernel args requires the grid lambda to also change to use META['BLOCK_M'] etc.; if the grid is not updated, the tile size change has no effect on grid coverage.

---

## Cumulative Progress

```
Baseline:  32290 ns  (1.000×)
Iter 2     31810 ns  (1.015×)  — fix_b_load_indexing_and_autotune_tile_sizes

Final best: 62240 ns  (0.5188×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,20 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_15(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -44,12 +58,7 @@

     tl.assume(pid_m >= 0)
     tl.assume(pid_n >= 0)
 
-    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
     rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
-    if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
-        offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
-    else:
-        offs_a_m = rm % M
     if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
@@ -57,27 +66,29 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    # Create A block pointer before the k loop (enables TMA / software pipelining)
+    A_block_ptr = tl.make_block_ptr(
+        base=A,
+        shape=(M, K),
+        strides=(stride_am, stride_ak),
+        offsets=(pid_m * BLOCK_M, 0),
+        block_shape=(BLOCK_M, BLOCK_K),
+        order=(1, 0),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        # A load via block_ptr — triggers padded SMEM staging, eliminating bank conflicts
+        a = tl.load(A_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_K))
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 144*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        # B load: B[k, n] = k*stride_bk + n*stride_bn = k*1 + n*144
+        k_offs = offs_k[:, None] + (k_idx * BLOCK_K)  # [BLOCK_K, 1]
+        n_offs = offs_b_n[None, :]  # [1, BLOCK_N]
+        b_ptrs = B + k_offs * 1 + n_offs * 144  # [BLOCK_K, BLOCK_N]
+        b_mask = k_offs < K  # broadcasts over N dim
+        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 40*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
-
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -87,10 +98,8 @@

     mask = (idx_m < M) & (idx_n < N)
 
     # inductor generates a suffix
-    xindex = idx_n + 40*idx_m
+    xindex = idx_n + 40 * idx_m
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
