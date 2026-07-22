# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused_convolution_68` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 33410 ns — **Best latency**: 32380 ns (**1.0318×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_bank_conflicts_via_shared_memory_transpose`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused_convolution_68`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_68_triton_tem_fused_convolution_6/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (62,1,1), Waves/SM: 0.06
- **Registers/thread**: 64
- **Duration**: 33410 ns
- **SM Throughput**: 5.81 % | **Memory**: 37.40 % | **DRAM**: 2.57 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.53 inst/cycle | **No Eligible**: 86.85 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.998× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 0.997× | 0.991× | ✓ |  |
| C | memory_coalescing_via_autotune_tile_and_access_fix | 1.007× | 0.827× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_increase_parallelism_and_fix_bank_conflicts | 1.004× | 0.823× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | memory_coalescing... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33410 ns | 33470 ns | +60 ns (+0.2%) | 33700 ns | +290 ns (+0.9%) | 40420 ns | +7010 ns (+21.0%) | 40580 ns | +7170 ns (+21.5%) |
| SM Throughput (%) | 5.81 % | 5.80 % | -0.01 pp | 5.82 % | +0.01 pp | 4.53 % | -1.28 pp | 4.55 % | -1.26 pp |
| Memory Throughput (%) | 37.40 % | 37.38 % | -0.02 pp | 37.47 % | +0.07 pp | 30.67 % | -6.73 pp | 30.84 % | -6.56 pp |
| DRAM Throughput (%) | 2.57 % | 2.57 % | +0.00 pp | 2.55 % | -0.02 pp | 2.13 % | -0.44 pp | 2.12 % | -0.45 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 | 64.00 register/thread | +0 | 62.00 register/thread | -2 | 62.00 register/thread | -2 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.06 | +0.00 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 86.85 % | 86.86 % | +0.01 pp | 86.81 % | -0.04 pp | 89.63 % | +2.78 pp | 89.56 % | +2.71 pp |
| IPC Active | 0.53 inst/cycle | 0.53 inst/cycle | +0.00 | 0.53 inst/cycle | +0.00 | 0.42 inst/cycle | -0.11 | 0.42 inst/cycle | -0.11 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_shared_memory_transpose | 1.034× | 1.032× | ✓ | **✓ best** |
| B | fix_b_matrix_bank_conflicts_via_smem_transpose_and_padding | 1.014× | 1.019× | ✓ |  |
| C | fix_b_matrix_bank_conflicts_via_transposed_load_and_smaller_tiles | 1.028× | 0.762× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | fix_b_matrix_bank... | Δ | fix_b_matrix_bank... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33410 ns | 32380 ns | -1030 ns (-3.1%) | 32800 ns | -610 ns (-1.8%) | 43870 ns | +10460 ns (+31.3%) |
| SM Throughput (%) | 5.81 % | 6.04 % | +0.23 pp | 5.75 % | -0.06 pp | 4.30 % | -1.51 pp |
| Memory Throughput (%) | 37.40 % | 38.51 % | +1.11 pp | 37.96 % | +0.56 pp | 28.38 % | -9.02 pp |
| DRAM Throughput (%) | 2.57 % | 2.65 % | +0.08 pp | 2.62 % | +0.05 pp | 1.96 % | -0.61 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 62.00 register/thread | -2 | 64.00 register/thread | +0 | 62.00 register/thread | -2 |
| Waves/SM | 0.06 | 0.06 | +0.00 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 86.85 % | 86.28 % | -0.57 pp | 86.96 % | +0.11 pp | 90.23 % | +3.38 pp |
| IPC Active | 0.53 inst/cycle | 0.55 inst/cycle | +0.02 | 0.52 inst/cycle | -0.01 | 0.39 inst/cycle | -0.14 |

---

## Iteration 3

**Best so far**: 1.0318× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_bank_conflicts_via_correct_transposed_load_n_outer_k_inner | 1.012× | 1.018× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 33410 ns | 32830 ns | -580 ns (-1.7%) |
| SM Throughput (%) | 5.81 % | 5.75 % | -0.06 pp |
| Memory Throughput (%) | 37.40 % | 38.01 % | +0.61 pp |
| DRAM Throughput (%) | 2.57 % | 2.62 % | +0.05 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 64.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.06 | +0.00 |
| No Eligible (%) | 86.85 % | 86.93 % | +0.08 pp |
| IPC Active | 0.53 inst/cycle | 0.52 inst/cycle | -0.01 |

---

## Iteration 4

**Best so far**: 1.0318× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | promote_tile_sizes_dynamic_grid_with_register_cap | 0.905× | 0.856× | ✓ |  |
| B | tma_loads_via_block_ptr_to_bypass_bank_conflicts | 1.014× | 1.010× | ✓ |  |
| C | promote_tile_sizes_to_constexpr_args_with_correct_dynamic_lambda_grid | 1.014× | 1.015× | ✓ |  |


### Metric Comparison

| Metric | Baseline | promote_tile_size... | Δ | tma_loads_via_blo... | Δ | promote_tile_size... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 33410 ns | 39010 ns | +5600 ns (+16.8%) | 33090 ns | -320 ns (-1.0%) | 32930 ns | -480 ns (-1.4%) |
| SM Throughput (%) | 5.81 % | 5.77 % | -0.04 pp | 5.90 % | +0.09 pp | 5.74 % | -0.07 pp |
| Memory Throughput (%) | 37.40 % | 34.53 % | -2.87 pp | 37.74 % | +0.34 pp | 37.94 % | +0.54 pp |
| DRAM Throughput (%) | 2.57 % | 2.20 % | -0.37 pp | 2.60 % | +0.03 pp | 2.61 % | +0.04 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 64.00 register/thread | 32.00 register/thread | -32 | 64.00 register/thread | +0 | 64.00 register/thread | +0 |
| Waves/SM | 0.06 | 0.04 | -0.02 | 0.06 | +0.00 | 0.06 | +0.00 |
| No Eligible (%) | 86.85 % | 87.04 % | +0.19 pp | 86.56 % | -0.29 pp | 86.93 % | +0.08 pp |
| IPC Active | 0.53 inst/cycle | 0.52 inst/cycle | -0.01 | 0.54 inst/cycle | +0.01 | 0.52 inst/cycle | -0.01 |

---

## Cumulative Progress

```
Baseline:  33410 ns  (1.000×)
Iter 2     32380 ns  (1.032×)  — fix_b_matrix_bank_conflicts_via_shared_memory_transpose

Final best: 32380 ns  (1.0318×)
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
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused_convolution_68(in_ptr0, arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
@@ -54,30 +68,28 @@

         offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
     else:
         offs_b_n = rn % N
-    offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
-        a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
-        a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
+        k_start = k_idx * BLOCK_K
+        k_offsets = tl.arange(0, BLOCK_K)  # shape [BLOCK_K]
+        k_abs = k_offsets + k_start         # absolute K indices
 
-        idx_m = offs_a_m[:, None]
-        idx_n = a_k_idx_vals
-        xindex = idx_n + 528*idx_m
-        a = tl.load(A + (xindex), mask=a_mask, other=0.0)
+        # A tile: [BLOCK_M, BLOCK_K], stride_am=528, stride_ak=1
+        a_m = offs_a_m[:, None]             # [BLOCK_M, 1]
+        a_k = k_abs[None, :]                # [1, BLOCK_K]
+        a_mask = a_k < K
+        a = tl.load(A + a_m * 528 + a_k * 1, mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 32*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 528*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
+        # B tile: load as [BLOCK_N, BLOCK_K] with K as INNER (stride=1, contiguous)
+        # N-outer (stride=528, slow) and K-inner (stride=1, fast) -> zero bank conflicts
+        b_n_T = offs_b_n[:, None]           # [BLOCK_N, 1], N is OUTER (stride=528, slow)
+        b_k_T = k_abs[None, :]              # [1, BLOCK_K], K is INNER (stride=1, fast)
+        b_mask_T = b_k_T < K               # mask on K dimension, broadcasts to [BLOCK_N, BLOCK_K]
+        b_T = tl.load(B + b_n_T * 528 + b_k_T * 1, mask=b_mask_T, other=0.0)  # [BLOCK_N, BLOCK_K]
+        b = tl.trans(b_T)                   # [BLOCK_K, BLOCK_N] — correct shape for tl.dot(a, b)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -91,8 +103,6 @@

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
