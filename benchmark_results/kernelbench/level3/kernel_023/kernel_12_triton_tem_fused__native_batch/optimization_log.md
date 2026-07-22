# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 41730 ns — **Best latency**: 41730 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_12_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.62
- **Registers/thread**: 154
- **Duration**: 41730 ns
- **SM Throughput**: 18.96 % | **Memory**: 79.39 % | **DRAM**: 8.84 %
- **Achieved Occupancy**: 11.79 %
- **IPC Active**: 0.86 inst/cycle | **No Eligible**: 78.52 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 1.000× | ✓ | **✓ best** |
| D | autotune_tile_sizes_to_reduce_bank_conflicts_and_improve_occupancy | 0.902× | 0.903× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 41730 ns | 41730 ns | +0 ns (+0.0%) | 46210 ns | +4480 ns (+10.7%) |
| SM Throughput (%) | 18.96 % | 18.64 % | -0.32 pp | 32.06 % | +13.10 pp |
| Memory Throughput (%) | 79.39 % | 78.05 % | -1.34 pp | 71.94 % | -7.45 pp |
| DRAM Throughput (%) | 8.84 % | 8.84 % | +0.00 pp | 7.99 % | -0.85 pp |
| Achieved Occupancy (%) | 11.79 % | 11.78 % | -0.01 pp | 11.85 % | +0.06 pp |
| Registers/Thread | 154.00 register/thread | 154.00 register/thread | +0 | 168.00 register/thread | +14 |
| Waves/SM | 0.62 | 0.62 | +0.00 | 0.62 | +0.00 |
| No Eligible (%) | 78.52 % | 78.52 % | +0.00 pp | 63.92 % | -14.60 pp |
| IPC Active | 0.86 inst/cycle | 0.85 inst/cycle | -0.01 | 1.46 inst/cycle | +0.60 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_b_matrix_load_transposition_to_eliminate_bank_conflicts | 0.883× | 0.875× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 41730 ns | 47680 ns | +5950 ns (+14.3%) |
| SM Throughput (%) | 18.96 % | 31.35 % | +12.39 pp |
| Memory Throughput (%) | 79.39 % | 70.32 % | -9.07 pp |
| DRAM Throughput (%) | 8.84 % | 7.74 % | -1.10 pp |
| Achieved Occupancy (%) | 11.79 % | 11.88 % | +0.09 pp |
| Registers/Thread | 154.00 register/thread | 234.00 register/thread | +80 |
| Waves/SM | 0.62 | 0.93 | +0.31 |
| No Eligible (%) | 78.52 % | 64.37 % | -14.15 pp |
| IPC Active | 0.86 inst/cycle | 1.42 inst/cycle | +0.56 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_load_n_boundary_mask_and_smem_transpose_to_eliminate_bank_conflicts | 0.881× | 0.873× | ✓ |  |
| B | fix_b_load_coalescing_via_transposed_tile_shape | 0.878× | 0.876× | ✓ |  |
| C | fix_b_matrix_n_boundary_mask_and_add_streaming_eviction_policies | 0.882× | 0.867× | ✓ |  |
| D | fix_b_matrix_n_dimension_mask_to_restore_correctness_and_enable_tile_size_increase | 0.882× | 0.867× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_load_n_boun... | Δ | fix_b_load_coales... | Δ | fix_b_matrix_n_bo... | Δ | fix_b_matrix_n_di... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 41730 ns | 47780 ns | +6050 ns (+14.5%) | 47620 ns | +5890 ns (+14.1%) | 48130 ns | +6400 ns (+15.3%) | 48130 ns | +6400 ns (+15.3%) |
| SM Throughput (%) | 18.96 % | 31.56 % | +12.60 pp | 31.34 % | +12.38 pp | 31.65 % | +12.69 pp | 31.00 % | +12.04 pp |
| Memory Throughput (%) | 79.39 % | 70.80 % | -8.59 pp | 70.27 % | -9.12 pp | 70.99 % | -8.40 pp | 69.53 % | -9.86 pp |
| DRAM Throughput (%) | 8.84 % | 7.73 % | -1.11 pp | 7.76 % | -1.08 pp | 7.67 % | -1.17 pp | 7.67 % | -1.17 pp |
| Achieved Occupancy (%) | 11.79 % | 11.84 % | +0.05 pp | 11.87 % | +0.08 pp | 11.86 % | +0.07 pp | 11.86 % | +0.07 pp |
| Registers/Thread | 154.00 register/thread | 234.00 register/thread | +80 | 234.00 register/thread | +80 | 234.00 register/thread | +80 | 234.00 register/thread | +80 |
| Waves/SM | 0.62 | 0.93 | +0.31 | 0.93 | +0.31 | 0.93 | +0.31 | 0.93 | +0.31 |
| No Eligible (%) | 78.52 % | 64.45 % | -14.07 pp | 64.35 % | -14.17 pp | 64.36 % | -14.16 pp | 64.38 % | -14.14 pp |
| IPC Active | 0.86 inst/cycle | 1.42 inst/cycle | +0.56 | 1.42 inst/cycle | +0.56 | 1.42 inst/cycle | +0.56 | 1.42 inst/cycle | +0.56 |


### Key Learnings

#### ❌ Plan C: `fix_b_matrix_n_boundary_mask_and_add_streaming_eviction_policies` (0.867×)

Adding a full 2D boundary mask to the B-matrix load (K & N dimensions) and adding eviction_policy='evict_first' to both A and B loads produced a 0.867x regression. All profiling metrics are essentially unchanged (Duration: 47.68→48.13 µs, Mem Busy: 70.32→70.99%, L1 hit rate: 1.59%→1.59%, bank conflicts: 55.37→55.39% speedup potential), indicating the change had no meaningful effect on the bottlenecks (shared memory bank conflicts, low occupancy) and introduced a small overhead. The root bottleneck — 7.9-way shared memory bank conflicts causing 55%+ speedup potential — remains completely unaddressed by mask changes or eviction policies.

> **Applicable when**: Never — this specific combination did not help and caused a slight regression

> **Avoid when**: When the dominant bottleneck is shared memory bank conflicts (7.9-way, 55%+ speedup potential) and low occupancy (12.5% theoretical). Adding boundary masks and eviction_policy='evict_first' to streaming loads does not address bank conflicts or register pressure, and the wider 2D mask computation adds marginal overhead. L1 hit rate remained at 1.59% before and after, confirming eviction policies had no measurable effect on cache behavior.

---

## Cumulative Progress

```
Baseline:  41730 ns  (1.000×)

Final best: 41730 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,27 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=4),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_12(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = False
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
-    BLOCK_N : tl.constexpr = 32
+    BLOCK_N : tl.constexpr = 64
     BLOCK_K : tl.constexpr = 32
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
@@ -60,20 +74,20 @@

     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
-        b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
         
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 144*idx_m
         a = tl.load(A + (xindex), mask=a_mask, other=0.0)
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 144*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
+        # B is stored as (24, 144) with strides (144, 1): B[n, k] = B_ptr + n*144 + k*1
+        # For matmul we need B transposed: B[k, n] = B_ptr + k*stride_bk + n*stride_bn = B_ptr + k*1 + n*144
+        rk = offs_k + k_idx * BLOCK_K   # [BLOCK_K] absolute K indices
+        b_ptrs = B + rk[:, None] * 1 + offs_b_n[None, :] * 144  # [BLOCK_K, BLOCK_N]
+        b_mask_k = rk[:, None] < K       # [BLOCK_K, 1]
+        b = tl.load(b_ptrs, mask=b_mask_k, other=0.0)
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -89,8 +103,6 @@

     # inductor generates a suffix
     xindex = idx_n + 24*idx_m
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
