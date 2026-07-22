# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 38300 ns — **Best latency**: 37890 ns (**1.0108×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`add_eviction_policies_and_autotune_tile_sizes`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_10_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (282,1,1), Waves/SM: 0.53
- **Registers/thread**: 128
- **Duration**: 38300 ns
- **SM Throughput**: 14.69 % | **Memory**: 60.25 % | **DRAM**: 7.37 %
- **Achieved Occupancy**: 13.41 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.58 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | add_eviction_policies_and_autotune_tile_sizes | 1.347× | 1.011× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_eviction_poli... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 38300 ns | 37890 ns | -410 ns (-1.1%) |
| SM Throughput (%) | 14.69 % | 14.60 % | -0.09 pp |
| Memory Throughput (%) | 60.25 % | 60.67 % | +0.42 pp |
| DRAM Throughput (%) | 7.37 % | 7.45 % | +0.08 pp |
| Achieved Occupancy (%) | 13.41 % | 13.54 % | +0.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.53 | 0.53 | +0.00 |
| No Eligible (%) | 78.58 % | 79.27 % | +0.69 pp |
| IPC Active | 0.86 inst/cycle | 0.83 inst/cycle | -0.03 |

---

## Iteration 2

**Best so far**: 1.0108× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_coalescing_to_eliminate_bank_conflicts | 1.332× | 1.022× | ✓ |  |
| B | tma_loads_correct_b_stride_descriptor | 1.018× | 1.006× | ✓ |  |
| C | shared_memory_padding_to_eliminate_bank_conflicts | 1.343× | 1.022× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_correct... | Δ | shared_memory_pad... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 38300 ns | 37470 ns | -830 ns (-2.2%) | 38080 ns | -220 ns (-0.6%) | 37470 ns | -830 ns (-2.2%) |
| SM Throughput (%) | 14.69 % | 14.44 % | -0.25 pp | 14.91 % | +0.22 pp | 14.54 % | -0.15 pp |
| Memory Throughput (%) | 60.25 % | 59.96 % | -0.29 pp | 61.28 % | +1.03 pp | 60.42 % | +0.17 pp |
| DRAM Throughput (%) | 7.37 % | 7.53 % | +0.16 pp | 7.41 % | +0.04 pp | 7.53 % | +0.16 pp |
| Achieved Occupancy (%) | 13.41 % | 13.54 % | +0.13 pp | 13.33 % | -0.08 pp | 13.54 % | +0.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 154.00 register/thread | +26 | 128.00 register/thread | +0 |
| Waves/SM | 0.53 | 0.53 | +0.00 | 0.71 | +0.18 | 0.53 | +0.00 |
| No Eligible (%) | 78.58 % | 79.10 % | +0.52 pp | 78.37 % | -0.21 pp | 79.10 % | +0.52 pp |
| IPC Active | 0.86 inst/cycle | 0.83 inst/cycle | -0.03 | 0.86 inst/cycle | +0.00 | 0.83 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0108× | **Memory items**: 4

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_layout_transpose_for_conflict_free_shared_memory | 1.327× | 1.012× | ✓ |  |
| B | eliminate_tl_trans_bank_conflicts_via_direct_b_load | 1.331× | 1.016× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_layout... | Δ | eliminate_tl_tran... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 38300 ns | 37860 ns | -440 ns (-1.1%) | 37700 ns | -600 ns (-1.6%) |
| SM Throughput (%) | 14.69 % | 14.68 % | -0.01 pp | 14.76 % | +0.07 pp |
| Memory Throughput (%) | 60.25 % | 60.95 % | +0.70 pp | 61.31 % | +1.06 pp |
| DRAM Throughput (%) | 7.37 % | 7.46 % | +0.09 pp | 7.48 % | +0.11 pp |
| Achieved Occupancy (%) | 13.41 % | 13.53 % | +0.12 pp | 13.56 % | +0.15 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.53 | 0.53 | +0.00 | 0.53 | +0.00 |
| No Eligible (%) | 78.58 % | 79.13 % | +0.55 pp | 79.15 % | +0.57 pp |
| IPC Active | 0.86 inst/cycle | 0.82 inst/cycle | -0.04 | 0.83 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  38300 ns  (1.000×)
Iter 1     37890 ns  (1.011×)  — add_eviction_policies_and_autotune_tile_sizes

Final best: 37890 ns  (1.0108×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=3),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_10(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -60,21 +72,18 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
 
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 96*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
-
+        # B[k, n] = B + k*stride_bk + n*stride_bn = B + k*1 + n*96
+        b_k = offs_k + k_idx * BLOCK_K          # [BLOCK_K], absolute K indices
+        b_ptrs = B + b_k[:, None] + 96 * offs_b_n[None, :]  # [BLOCK_K, BLOCK_N]
+        b = tl.load(b_ptrs, eviction_policy='evict_last')
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -85,9 +94,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 24*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
