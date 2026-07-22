# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 45570 ns — **Best latency**: 40480 ns (**1.1257×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`eliminate_tldot_replace_with_register_outer_product_to_bypass_smem_bank_conflicts`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_020/kernel_5_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1960,1,1), Waves/SM: 4.95
- **Registers/thread**: 168
- **Duration**: 45570 ns
- **SM Throughput**: 30.52 % | **Memory**: 77.03 % | **DRAM**: 13.09 %
- **Achieved Occupancy**: 17.93 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.60 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.000× | 0.996× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 45570 ns | 45760 ns | +190 ns (+0.4%) |
| SM Throughput (%) | 30.52 % | 30.53 % | +0.01 pp |
| Memory Throughput (%) | 77.03 % | 77.04 % | +0.01 pp |
| DRAM Throughput (%) | 13.09 % | 13.03 % | -0.06 pp |
| Achieved Occupancy (%) | 17.93 % | 17.96 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 |
| No Eligible (%) | 66.60 % | 66.64 % | +0.04 pp |
| IPC Active | 1.33 inst/cycle | 1.32 inst/cycle | -0.01 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_layout_to_eliminate_shared_memory_bank_conflicts | 0.999× | 1.002× | ✓ |  |
| B | tma_loads_via_block_ptr | 1.011× | 1.007× | ✓ | **✓ best** |
| C | fix_correctness_then_add_eviction_policies | 0.985× | 0.996× | ✓ |  |
| D | fix_b_load_indexing_and_increase_tile_sizes_for_parallelism | 0.976× | 0.983× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_via_blo... | Δ | fix_correctness_t... | Δ | fix_b_load_indexi... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45570 ns | 45470 ns | -100 ns (-0.2%) | 45250 ns | -320 ns (-0.7%) | 45760 ns | +190 ns (+0.4%) | 46340 ns | +770 ns (+1.7%) |
| SM Throughput (%) | 30.52 % | 30.51 % | -0.01 pp | 29.94 % | -0.58 pp | 30.34 % | -0.18 pp | 29.97 % | -0.55 pp |
| Memory Throughput (%) | 77.03 % | 76.99 % | -0.04 pp | 77.25 % | +0.22 pp | 76.26 % | -0.77 pp | 76.25 % | -0.78 pp |
| DRAM Throughput (%) | 13.09 % | 13.20 % | +0.11 pp | 13.22 % | +0.13 pp | 8.34 % | -4.75 pp | 12.89 % | -0.20 pp |
| Achieved Occupancy (%) | 17.93 % | 17.93 % | +0.00 pp | 17.90 % | -0.03 pp | 17.93 % | +0.00 pp | 17.91 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.60 % | 66.80 % | +0.20 pp | 67.09 % | +0.49 pp | 66.50 % | -0.10 pp | 67.40 % | +0.80 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.31 inst/cycle | -0.02 | 1.33 inst/cycle | +0.00 | 1.30 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0071× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts | 1.011× | 1.015× | ✓ | **✓ best** |
| B | eliminate_b_matrix_bank_conflicts_via_transposed_smem_staging | 0.994× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ | eliminate_b_matri... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45570 ns | 44900 ns | -670 ns (-1.5%) | 45250 ns | -320 ns (-0.7%) |
| SM Throughput (%) | 30.52 % | 29.77 % | -0.75 pp | 30.39 % | -0.13 pp |
| Memory Throughput (%) | 77.03 % | 76.82 % | -0.21 pp | 76.54 % | -0.49 pp |
| DRAM Throughput (%) | 13.09 % | 13.30 % | +0.21 pp | 13.26 % | +0.17 pp |
| Achieved Occupancy (%) | 17.93 % | 17.92 % | -0.01 pp | 17.91 % | -0.02 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.60 % | 67.28 % | +0.68 pp | 66.60 % | +0.00 pp |
| IPC Active | 1.33 inst/cycle | 1.30 inst/cycle | -0.03 | 1.33 inst/cycle | +0.00 |

---

## Iteration 4

**Best so far**: 1.0149× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_smem_bank_conflicts_via_explicit_padded_staging | 0.983× | 0.985× | ✓ |  |
| C | replace_transposed_b_load_with_standard_k_n_layout_to_eliminate_bank_conflicts | 0.993× | 0.997× | ✓ |  |
| D | tma_loads_via_make_block_ptr_for_a_and_b_matrices | 1.006× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_smem_bank_c... | Δ | replace_transpose... | Δ | tma_loads_via_mak... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 45570 ns | 46270 ns | +700 ns (+1.5%) | 45700 ns | +130 ns (+0.3%) | 45090 ns | -480 ns (-1.1%) |
| SM Throughput (%) | 30.52 % | 29.00 % | -1.52 pp | 30.59 % | +0.07 pp | 30.20 % | -0.32 pp |
| Memory Throughput (%) | 77.03 % | 74.54 % | -2.49 pp | 77.18 % | +0.15 pp | 77.92 % | +0.89 pp |
| DRAM Throughput (%) | 13.09 % | 12.91 % | -0.18 pp | 13.07 % | -0.02 pp | 13.23 % | +0.14 pp |
| Achieved Occupancy (%) | 17.93 % | 18.03 % | +0.10 pp | 17.93 % | +0.00 pp | 17.90 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.60 % | 68.64 % | +2.04 pp | 66.74 % | +0.14 pp | 67.15 % | +0.55 pp |
| IPC Active | 1.33 inst/cycle | 1.25 inst/cycle | -0.08 | 1.32 inst/cycle | -0.01 | 1.31 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0149× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | bypass_shared_memory_bank_conflicts_via_explicit_k_loop_accumulation | 0.678× | 0.630× | ✓ |  |
| C | eliminate_tldot_replace_with_register_outer_product_to_bypass_smem_bank_conflicts | 1.164× | 1.126× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | bypass_shared_mem... | Δ | eliminate_tldot_r... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 45570 ns | 72350 ns | +26780 ns (+58.8%) | 40480 ns | -5090 ns (-11.2%) |
| SM Throughput (%) | 30.52 % | 60.15 % | +29.63 pp | 53.30 % | +22.78 pp |
| Memory Throughput (%) | 77.03 % | 78.64 % | +1.61 pp | 72.00 % | -5.03 pp |
| DRAM Throughput (%) | 13.09 % | 8.25 % | -4.84 pp | 14.72 % | +1.63 pp |
| Achieved Occupancy (%) | 17.93 % | 17.69 % | -0.24 pp | 17.46 % | -0.47 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 4.95 | 4.95 | +0.00 | 4.95 | +0.00 |
| No Eligible (%) | 66.60 % | 54.14 % | -12.46 pp | 42.50 % | -24.10 pp |
| IPC Active | 1.33 inst/cycle | 1.85 inst/cycle | +0.52 | 2.31 inst/cycle | +0.98 |

---

## Cumulative Progress

```
Baseline:  45570 ns  (1.000×)
Iter 2     45250 ns  (1.007×)  — tma_loads_via_block_ptr
Iter 3     44900 ns  (1.015×)  — fix_b_block_ptr_order_to_eliminate_shared_memory_bank_conflicts
Iter 5     40480 ns  (1.126×)  — eliminate_tldot_replace_with_register_outer_product_to_bypass_smem_bank_conflicts

Final best: 40480 ns  (1.1257×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_5(arg_A, arg_B, out_ptr0):
@@ -58,23 +59,33 @@

     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
-        
+
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 16*idx_m
         a = tl.load(A + (xindex))
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        # B[k, n] at offset k*stride_bk + n*stride_bn = k*1 + n*16
+        # Load B in standard [BLOCK_K, BLOCK_N] layout to eliminate shared memory bank conflicts
+        b_k = offs_k[:, None] + (k_idx * BLOCK_K)   # shape [BLOCK_K, 1]
+        b_n = offs_b_n[None, :]                       # shape [1, BLOCK_N]
+        b_ptrs = B + b_k * 1 + b_n * 16              # shape [BLOCK_K, BLOCK_N], stride_bk=1, stride_bn=16
+        b = tl.load(b_ptrs, mask=(b_k < K) & (b_n < N), other=0.0)
 
-        
-        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
-        
+        # Replace tl.dot with explicit outer product loop to bypass SMEM bank conflicts.
+        # Extract column k of a (shape [BLOCK_M]) and row k of b (shape [BLOCK_N])
+        # using tl.sum with a one-hot mask over the K dimension.
+        for k in tl.static_range(BLOCK_K):
+            # Extract column k from a: shape [BLOCK_M, BLOCK_K] -> [BLOCK_M]
+            k_mask_k = (tl.arange(0, BLOCK_K) == k)  # [BLOCK_K] one-hot
+            a_col = tl.sum(a * k_mask_k[None, :], axis=1)  # [BLOCK_M]
+            # Extract row k from b: shape [BLOCK_K, BLOCK_N] -> [BLOCK_N]
+            k_mask_bk = (tl.arange(0, BLOCK_K) == k)  # [BLOCK_K] one-hot
+            b_row = tl.sum(b * k_mask_bk[:, None], axis=0)  # [BLOCK_N]
+            # Outer product accumulate
+            acc = acc + tl.expand_dims(a_col, 1).to(ACC_TYPE) * tl.expand_dims(b_row, 0).to(ACC_TYPE)
 
     # rematerialize rm and rn to save registers
     rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
@@ -86,8 +97,6 @@

     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
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
