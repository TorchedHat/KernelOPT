# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 29760 ns — **Best latency**: 28670 ns (**1.0380×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`fix_b_matrix_load_stride_to_eliminate_shared_memory_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_23_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (186,1,1), Waves/SM: 0.47
- **Registers/thread**: 80
- **Duration**: 29760 ns
- **SM Throughput**: 9.62 % | **Memory**: 60.81 % | **DRAM**: 1.36 %
- **Achieved Occupancy**: 9.72 %
- **IPC Active**: 0.57 inst/cycle | **No Eligible**: 85.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.992× | 0.998× | ✓ |  |
| B | autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy | 1.010× | 1.020× | ✓ | **✓ best** |
| C | reduce_shared_memory_via_autotune_tile_sizes | 1.003× | 1.021× | ✓ |  |
| D | increase_tile_sizes_with_autotune_to_reduce_bank_conflicts_and_improve_occupancy | 0.543× | 0.523× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | autotune_tile_siz... | Δ | reduce_shared_mem... | Δ | increase_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29760 ns | 29820 ns | +60 ns (+0.2%) | 29180 ns | -580 ns (-1.9%) | 29150 ns | -610 ns (-2.0%) | 56860 ns | +27100 ns (+91.1%) |
| SM Throughput (%) | 9.62 % | 9.63 % | +0.01 pp | 9.88 % | +0.26 pp | 9.90 % | +0.28 pp | 16.45 % | +6.83 pp |
| Memory Throughput (%) | 60.81 % | 60.94 % | +0.13 pp | 60.73 % | -0.08 pp | 60.88 % | +0.07 pp | 62.69 % | +1.88 pp |
| DRAM Throughput (%) | 1.36 % | 1.36 % | +0.00 pp | 1.38 % | +0.02 pp | 1.38 % | +0.02 pp | 0.71 % | -0.65 pp |
| Achieved Occupancy (%) | 9.72 % | 9.72 % | +0.00 pp | 9.69 % | -0.03 pp | 9.69 % | -0.03 pp | 9.77 % | +0.05 pp |
| Registers/Thread | 80.00 register/thread | 80.00 register/thread | +0 | 64.00 register/thread | -16 | 64.00 register/thread | -16 | 128.00 register/thread | +48 |
| Waves/SM | 0.47 | 0.47 | +0.00 | 0.18 | -0.29 | 0.18 | -0.29 | 0.35 | -0.12 |
| No Eligible (%) | 85.74 % | 85.84 % | +0.10 pp | 85.54 % | -0.20 pp | 85.55 % | -0.19 pp | 76.25 % | -9.49 pp |
| IPC Active | 0.57 inst/cycle | 0.57 inst/cycle | +0.00 | 0.57 inst/cycle | +0.00 | 0.58 inst/cycle | +0.01 | 0.94 inst/cycle | +0.37 |

---

## Iteration 2

**Best so far**: 1.0199× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | proper_autotune_with_meta_tile_sizes_and_dynamic_grid | 1.059× | 1.046× | ✓ | **✓ best** |
| C | proper_autotune_tile_sizes_as_meta_with_dynamic_grid | 1.006× | 1.012× | ✓ |  |
| D | autotune_tile_sizes_as_meta_parameters_with_dynamic_grid | 0.809× | 0.547× | ✓ |  |


### Metric Comparison

| Metric | Baseline | proper_autotune_w... | Δ | proper_autotune_t... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29760 ns | 28450 ns | -1310 ns (-4.4%) | 29410 ns | -350 ns (-1.2%) | 54370 ns | +24610 ns (+82.7%) |
| SM Throughput (%) | 9.62 % | 10.31 % | +0.69 pp | 9.83 % | +0.21 pp | 16.29 % | +6.67 pp |
| Memory Throughput (%) | 60.81 % | 58.87 % | -1.94 pp | 60.46 % | -0.35 pp | 61.31 % | +0.50 pp |
| DRAM Throughput (%) | 1.36 % | 1.41 % | +0.05 pp | 1.37 % | +0.01 pp | 0.74 % | -0.62 pp |
| Achieved Occupancy (%) | 9.72 % | 9.67 % | -0.05 pp | 9.70 % | -0.02 pp | 9.68 % | -0.04 pp |
| Registers/Thread | 80.00 register/thread | 62.00 register/thread | -18 | 64.00 register/thread | -16 | 128.00 register/thread | +48 |
| Waves/SM | 0.47 | 0.18 | -0.29 | 0.18 | -0.29 | 0.35 | -0.12 |
| No Eligible (%) | 85.74 % | 85.10 % | -0.64 pp | 85.60 % | -0.14 pp | 76.86 % | -8.88 pp |
| IPC Active | 0.57 inst/cycle | 0.59 inst/cycle | +0.02 | 0.58 inst/cycle | +0.01 | 0.93 inst/cycle | +0.36 |

---

## Iteration 3

**Best so far**: 1.046× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_shared_memory_bank_conflicts_via_correct_meta_autotune | 1.009× | 1.021× | ✓ |  |
| B | fix_smem_bank_conflicts_via_proper_meta_autotune_with_dynamic_grid | 1.057× | 1.047× | ✓ |  |
| C | fix_shared_memory_bank_conflicts_via_proper_meta_parameter_autotune | 1.059× | 1.046× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_shared_memory... | Δ | fix_smem_bank_con... | Δ | fix_shared_memory... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29760 ns | 29150 ns | -610 ns (-2.0%) | 28420 ns | -1340 ns (-4.5%) | 28450 ns | -1310 ns (-4.4%) |
| SM Throughput (%) | 9.62 % | 9.92 % | +0.30 pp | 10.30 % | +0.68 pp | 10.31 % | +0.69 pp |
| Memory Throughput (%) | 60.81 % | 61.01 % | +0.20 pp | 58.81 % | -2.00 pp | 58.83 % | -1.98 pp |
| DRAM Throughput (%) | 1.36 % | 1.38 % | +0.02 pp | 1.41 % | +0.05 pp | 1.41 % | +0.05 pp |
| Achieved Occupancy (%) | 9.72 % | 9.69 % | -0.03 pp | 9.67 % | -0.05 pp | 9.67 % | -0.05 pp |
| Registers/Thread | 80.00 register/thread | 64.00 register/thread | -16 | 62.00 register/thread | -18 | 62.00 register/thread | -18 |
| Waves/SM | 0.47 | 0.18 | -0.29 | 0.18 | -0.29 | 0.18 | -0.29 |
| No Eligible (%) | 85.74 % | 85.49 % | -0.25 pp | 85.05 % | -0.69 pp | 85.03 % | -0.71 pp |
| IPC Active | 0.57 inst/cycle | 0.58 inst/cycle | +0.01 | 0.60 inst/cycle | +0.03 | 0.60 inst/cycle | +0.03 |

---

## Iteration 4

**Best so far**: 1.046× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_bank_conflicts_via_conflict_free_load_pattern | 1.026× | 1.049× | ✓ |  |
| B | fix_b_matrix_load_stride_to_eliminate_shared_memory_bank_conflicts | 1.066× | 1.038× | ✓ | **✓ best** |
| C | fix_smem_bank_conflicts_via_meta_tile_sizes_with_correct_b_indexing | 1.061× | 1.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_bank... | Δ | fix_b_matrix_load... | Δ | fix_smem_bank_con... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29760 ns | 28380 ns | -1380 ns (-4.6%) | 28670 ns | -1090 ns (-3.7%) | 28350 ns | -1410 ns (-4.7%) |
| SM Throughput (%) | 9.62 % | 9.85 % | +0.23 pp | 10.23 % | +0.61 pp | 10.31 % | +0.69 pp |
| Memory Throughput (%) | 60.81 % | 58.57 % | -2.24 pp | 58.39 % | -2.42 pp | 58.87 % | -1.94 pp |
| DRAM Throughput (%) | 1.36 % | 1.41 % | +0.05 pp | 1.40 % | +0.04 pp | 1.41 % | +0.05 pp |
| Achieved Occupancy (%) | 9.72 % | 9.70 % | -0.02 pp | 9.69 % | -0.03 pp | 9.67 % | -0.05 pp |
| Registers/Thread | 80.00 register/thread | 64.00 register/thread | -16 | 62.00 register/thread | -18 | 62.00 register/thread | -18 |
| Waves/SM | 0.47 | 0.18 | -0.29 | 0.18 | -0.29 | 0.18 | -0.29 |
| No Eligible (%) | 85.74 % | 85.60 % | -0.14 pp | 85.17 % | -0.57 pp | 85.06 % | -0.68 pp |
| IPC Active | 0.57 inst/cycle | 0.58 inst/cycle | +0.01 | 0.59 inst/cycle | +0.02 | 0.59 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.046× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | meta_autotune_tile_sizes_with_correct_b_strided_index_and_dynamic_grid | 1.024× | 1.037× | ✓ |  |
| B | eliminate_smem_bank_conflicts_via_b_tile_transpose_with_padding | 1.027× | 1.050× | ✓ |  |
| C | load_b_transposed_to_eliminate_stride240_bank_conflicts | 1.028× | 1.037× | ✓ |  |
| D | add_triton_autotune | 1.037× | 0.921× | ✓ |  |


### Metric Comparison

| Metric | Baseline | meta_autotune_til... | Δ | eliminate_smem_ba... | Δ | load_b_transposed... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 29760 ns | 28700 ns | -1060 ns (-3.6%) | 28350 ns | -1410 ns (-4.7%) | 28700 ns | -1060 ns (-3.6%) | 32320 ns | +2560 ns (+8.6%) |
| SM Throughput (%) | 9.62 % | 10.28 % | +0.66 pp | 9.82 % | +0.20 pp | 10.20 % | +0.58 pp | 8.10 % | -1.52 pp |
| Memory Throughput (%) | 60.81 % | 58.65 % | -2.16 pp | 58.35 % | -2.46 pp | 58.24 % | -2.57 pp | 50.59 % | -10.22 pp |
| DRAM Throughput (%) | 1.36 % | 1.40 % | +0.04 pp | 1.41 % | +0.05 pp | 1.40 % | +0.04 pp | 1.24 % | -0.12 pp |
| Achieved Occupancy (%) | 9.72 % | 9.67 % | -0.05 pp | 9.69 % | -0.03 pp | 9.69 % | -0.03 pp | 9.23 % | -0.49 pp |
| Registers/Thread | 80.00 register/thread | 62.00 register/thread | -18 | 64.00 register/thread | -16 | 62.00 register/thread | -18 | 55.00 register/thread | -25 |
| Waves/SM | 0.47 | 0.18 | -0.29 | 0.18 | -0.29 | 0.18 | -0.29 | 0.16 | -0.31 |
| No Eligible (%) | 85.74 % | 85.08 % | -0.66 pp | 85.59 % | -0.15 pp | 85.15 % | -0.59 pp | 89.17 % | +3.43 pp |
| IPC Active | 0.57 inst/cycle | 0.59 inst/cycle | +0.02 | 0.57 inst/cycle | +0.00 | 0.59 inst/cycle | +0.02 | 0.43 inst/cycle | -0.14 |

---

## Cumulative Progress

```
Baseline:  29760 ns  (1.000×)
Iter 1     29180 ns  (1.020×)  — autotune_tile_sizes_to_fix_bank_conflicts_and_occupancy
Iter 2     28450 ns  (1.046×)  — proper_autotune_with_meta_tile_sizes_and_dynamic_grid

Final best: 28670 ns  (1.0380×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=2, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_23(arg_A, arg_B, out_ptr0):
-    EVEN_K : tl.constexpr = False
+    EVEN_K : tl.constexpr = True
     USE_FAST_ACCUM : tl.constexpr = False
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 32
     BLOCK_N : tl.constexpr = 32
-    BLOCK_K : tl.constexpr = 128
+    BLOCK_K : tl.constexpr = 16
     GROUP_M : tl.constexpr = 8
     ALLOW_TF32 : tl.constexpr = False
     INDEX_DTYPE : tl.constexpr = tl.int32
@@ -58,10 +72,10 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
         b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
         b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
@@ -75,9 +89,7 @@

         xindex = idx_n + 80*idx_m
         b = tl.load(B + ((tl.broadcast_to(idx_m + 240*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)
 
-        
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -89,8 +101,6 @@

     # inductor generates a suffix
     xindex = idx_n + 80*idx_m
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
