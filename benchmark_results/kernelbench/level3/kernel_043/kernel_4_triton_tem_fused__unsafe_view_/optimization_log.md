# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 8220 ns — **Best latency**: 8100 ns (**1.0148×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_grid_y_to_1024_for_correct_full_batch_head_parallelism`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_priority/kernel_043/kernel_4_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (64,1,1), Waves/SM: 0.12
- **Registers/thread**: 124
- **Duration**: 8220 ns
- **SM Throughput**: 15.32 % | **Memory**: 15.39 % | **DRAM**: 1.05 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.63 inst/cycle | **No Eligible**: 56.77 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.992× | 1.000× | ✓ |  |
| B | fix_batch_grid_dimension_for_bmm_parallelism | 0.993× | 1.031× | ✓ |  |
| D | add_autotune_increase_tile_sizes_and_stages | 0.988× | 1.028× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | fix_batch_grid_di... | Δ | add_autotune_incr... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8220 ns | 8220 ns | +0 ns (+0.0%) | 7970 ns | -250 ns (-3.0%) | 8000 ns | -220 ns (-2.7%) |
| SM Throughput (%) | 15.32 % | 14.95 % | -0.37 pp | 15.38 % | +0.06 pp | 15.34 % | +0.02 pp |
| Memory Throughput (%) | 15.39 % | 15.02 % | -0.37 pp | 15.45 % | +0.06 pp | 15.40 % | +0.01 pp |
| DRAM Throughput (%) | 1.05 % | 1.04 % | -0.01 pp | 1.08 % | +0.03 pp | 1.07 % | +0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 56.77 % | 58.42 % | +1.65 pp | 57.11 % | +0.34 pp | 56.70 % | -0.07 pp |
| IPC Active | 1.63 inst/cycle | 1.70 inst/cycle | +0.07 | 1.75 inst/cycle | +0.12 | 1.72 inst/cycle | +0.09 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | reduce_register_pressure_via_maxnreg_to_improve_occupancy | 1.000× | 0.988× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 8220 ns | 8320 ns | +100 ns (+1.2%) |
| SM Throughput (%) | 15.32 % | 14.72 % | -0.60 pp |
| Memory Throughput (%) | 15.39 % | 14.79 % | -0.60 pp |
| DRAM Throughput (%) | 1.05 % | 1.03 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 |
| No Eligible (%) | 56.77 % | 58.58 % | +1.81 pp |
| IPC Active | 1.63 inst/cycle | 1.74 inst/cycle | +0.11 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_output_coverage_by_correcting_batch_grid_launch | 0.998× | 1.007× | ✓ |  |
| D | fix_batch_head_grid_with_correct_strides_for_bmm_parallelism | 1.003× | 1.015× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_output_covera... | Δ | fix_batch_head_gr... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 8220 ns | 8160 ns | -60 ns (-0.7%) | 8100 ns | -120 ns (-1.5%) |
| SM Throughput (%) | 15.32 % | 14.98 % | -0.34 pp | 15.35 % | +0.03 pp |
| Memory Throughput (%) | 15.39 % | 15.05 % | -0.34 pp | 15.42 % | +0.03 pp |
| DRAM Throughput (%) | 1.05 % | 1.05 % | +0.00 pp | 1.06 % | +0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 56.77 % | 58.16 % | +1.39 pp | 56.74 % | -0.03 pp |
| IPC Active | 1.63 inst/cycle | 1.67 inst/cycle | +0.04 | 1.67 inst/cycle | +0.04 |

---

## Iteration 4

**Best so far**: 1.0148× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_y_to_cover_all_batch_head_slices | 1.000× | 1.031× | ✓ |  |
| C | fix_batch_grid_with_correct_two_level_batch_index_decomposition | 0.993× | 1.015× | ✓ |  |
| D | persistent_kernel_correct_4d_batch_stride_decomposition | 1.000× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_y_to_cov... | Δ | fix_batch_grid_wi... | Δ | persistent_kernel... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8220 ns | 7970 ns | -250 ns (-3.0%) | 8100 ns | -120 ns (-1.5%) | 8160 ns | -60 ns (-0.7%) |
| SM Throughput (%) | 15.32 % | 15.33 % | +0.01 pp | 15.24 % | -0.08 pp | 15.05 % | -0.27 pp |
| Memory Throughput (%) | 15.39 % | 15.39 % | +0.00 pp | 15.31 % | -0.08 pp | 15.11 % | -0.28 pp |
| DRAM Throughput (%) | 1.05 % | 1.08 % | +0.03 pp | 1.06 % | +0.01 pp | 1.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 56.77 % | 57.90 % | +1.13 pp | 56.63 % | -0.14 pp | 58.23 % | +1.46 pp |
| IPC Active | 1.63 inst/cycle | 1.72 inst/cycle | +0.09 | 1.72 inst/cycle | +0.09 | 1.73 inst/cycle | +0.10 |

---

## Iteration 5

**Best so far**: 1.0148× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_y_to_1024_for_correct_full_batch_head_parallelism | 0.997× | 1.015× | ✓ |  |
| B | fix_grid_y_to_1024_for_full_batch_parallelism_kernel_launch_only | 1.001× | 1.031× | ✓ |  |
| C | fix_grid_y_to_1024_for_full_batch_parallelism_minimal_change | 0.992× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_y_to_102... | Δ | fix_grid_y_to_102... | Δ | fix_grid_y_to_102... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 8220 ns | 8100 ns | -120 ns (-1.5%) | 7970 ns | -250 ns (-3.0%) | 8130 ns | -90 ns (-1.1%) |
| SM Throughput (%) | 15.32 % | 15.27 % | -0.05 pp | 14.99 % | -0.33 pp | 15.20 % | -0.12 pp |
| Memory Throughput (%) | 15.39 % | 15.33 % | -0.06 pp | 15.06 % | -0.33 pp | 15.27 % | -0.12 pp |
| DRAM Throughput (%) | 1.05 % | 1.06 % | +0.01 pp | 1.08 % | +0.03 pp | 1.05 % | +0.00 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 124.00 register/thread | 124.00 register/thread | +0 | 124.00 register/thread | +0 | 124.00 register/thread | +0 |
| Waves/SM | 0.12 | 0.12 | +0.00 | 0.12 | +0.00 | 0.12 | +0.00 |
| No Eligible (%) | 56.77 % | 58.08 % | +1.31 pp | 56.71 % | -0.06 pp | 57.15 % | +0.38 pp |
| IPC Active | 1.63 inst/cycle | 1.72 inst/cycle | +0.09 | 1.68 inst/cycle | +0.05 | 1.68 inst/cycle | +0.05 |

---

## Cumulative Progress

```
Baseline:  8220 ns  (1.000×)
Iter 3     8100 ns  (1.015×)  — fix_batch_head_grid_with_correct_strides_for_bmm_parallelism

Final best: 8100 ns  (1.0148×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,35 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        # Original configs (baseline)
+        triton.Config({}, num_stages=3, num_warps=4),
+        triton.Config({}, num_stages=4, num_warps=4),
+        triton.Config({}, num_stages=3, num_warps=8),
+        triton.Config({}, num_stages=4, num_warps=8),
+        triton.Config({}, num_stages=5, num_warps=4),
+        triton.Config({}, num_stages=5, num_warps=8),
+        triton.Config({}, num_stages=2, num_warps=4),
+        triton.Config({}, num_stages=2, num_warps=8),
+        # maxnreg=96: moderate register cap (~31% theoretical occupancy)
+        triton.Config({}, num_stages=3, num_warps=4, maxnreg=96),
+        triton.Config({}, num_stages=4, num_warps=4, maxnreg=96),
+        triton.Config({}, num_stages=3, num_warps=8, maxnreg=96),
+        triton.Config({}, num_stages=4, num_warps=8, maxnreg=96),
+        triton.Config({}, num_stages=2, num_warps=4, maxnreg=96),
+        triton.Config({}, num_stages=2, num_warps=8, maxnreg=96),
+        # maxnreg=64: aggressive register cap (~50% theoretical occupancy)
+        triton.Config({}, num_stages=3, num_warps=4, maxnreg=64),
+        triton.Config({}, num_stages=4, num_warps=4, maxnreg=64),
+        triton.Config({}, num_stages=3, num_warps=8, maxnreg=64),
+        triton.Config({}, num_stages=4, num_warps=8, maxnreg=64),
+        triton.Config({}, num_stages=2, num_warps=4, maxnreg=64),
+        triton.Config({}, num_stages=2, num_warps=8, maxnreg=64),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_addmm_bmm_clone_split_transpose_view_4(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -23,13 +52,15 @@

     N = 512
     K = 96
 
-    stride_aq = 49152
+    stride_aq = 49152   # per-head stride for A (96*512)
     stride_am = 96
     stride_ak = 1
+    stride_ab = 393216  # per-batch stride for A (8*49152)
 
-    stride_bq = 49152
+    stride_bq = 49152   # per-head stride for B (96*512)
     stride_bk = 512
     stride_bn = 1
+    stride_bb = 393216  # per-batch stride for B (8*49152)
 
     # based on triton.ops.matmul
     pid = tl.program_id(0).to(INDEX_DTYPE)
@@ -58,9 +89,11 @@

 
     rk = tl.arange(0, BLOCK_K)
 
-    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # batch dimension for BMM
-    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + idx_q*stride_aq)
-    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + idx_q*stride_bq)
+    idx_q = tl.program_id(1).to(INDEX_DTYPE)  # flat batch*head index 0..1023
+    batch_idx = idx_q // 8
+    head_idx  = idx_q % 8
+    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak + batch_idx * stride_ab + head_idx * stride_aq)
+    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn + batch_idx * stride_bb + head_idx * stride_bq)
 
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
     for k in range(K, 0, -BLOCK_K):
@@ -86,8 +119,6 @@

     xindex = idx_n + 512*idx_m + 262144*idx_q
     tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
