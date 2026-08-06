# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 53730 ns — **Best latency**: 52220 ns (**1.0289×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`evict_first_on_streaming_a_load_and_output_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_023/kernel_7_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2250,1,1), Waves/SM: 5.68
- **Registers/thread**: 168
- **Duration**: 53730 ns
- **SM Throughput**: 30.24 % | **Memory**: 76.33 % | **DRAM**: 14.34 %
- **Achieved Occupancy**: 17.73 %
- **IPC Active**: 1.33 inst/cycle | **No Eligible**: 66.72 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.004× | 1.015× | ✓ | **✓ best** |
| C | reduce_block_n_and_add_evict_first_output_store | 0.996× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | reduce_block_n_an... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53730 ns | 52930 ns | -800 ns (-1.5%) | 53760 ns | +30 ns (+0.1%) |
| SM Throughput (%) | 30.24 % | 30.45 % | +0.21 pp | 30.13 % | -0.11 pp |
| Memory Throughput (%) | 76.33 % | 76.87 % | +0.54 pp | 75.73 % | -0.60 pp |
| DRAM Throughput (%) | 14.34 % | 14.49 % | +0.15 pp | 14.27 % | -0.07 pp |
| Achieved Occupancy (%) | 17.73 % | 17.74 % | +0.01 pp | 17.77 % | +0.04 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 5.68 | 5.68 | +0.00 | 5.68 | +0.00 |
| No Eligible (%) | 66.72 % | 66.81 % | +0.09 pp | 66.57 % | -0.15 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.34 inst/cycle | +0.01 |

---

## Iteration 2

**Best so far**: 1.0151× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_group_m_and_num_stages_for_l2_reuse_and_pipelining | 1.033× | 1.011× | ✓ | **✓ best** |
| D | fix_b_matrix_index_and_eliminate_bank_conflicts | 0.994× | 1.011× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_group_m_... | Δ | fix_b_matrix_inde... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53730 ns | 53150 ns | -580 ns (-1.1%) | 53150 ns | -580 ns (-1.1%) |
| SM Throughput (%) | 30.24 % | 30.08 % | -0.16 pp | 30.02 % | -0.22 pp |
| Memory Throughput (%) | 76.33 % | 75.94 % | -0.39 pp | 75.46 % | -0.87 pp |
| DRAM Throughput (%) | 14.34 % | 14.42 % | +0.08 pp | 14.43 % | +0.09 pp |
| Achieved Occupancy (%) | 17.73 % | 17.76 % | +0.03 pp | 17.76 % | +0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 5.68 | 5.68 | +0.00 | 5.68 | +0.00 |
| No Eligible (%) | 66.72 % | 66.55 % | -0.17 pp | 66.70 % | -0.02 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.33 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0151× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_load_remove_outer_broadcast_to_only | 1.036× | 1.026× | ✓ |  |
| B | tma_loads_with_correct_b_matrix_coalesced_access | 1.038× | 1.033× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_load... | Δ | tma_loads_with_co... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 53730 ns | 52380 ns | -1350 ns (-2.5%) | 52000 ns | -1730 ns (-3.2%) |
| SM Throughput (%) | 30.24 % | 30.33 % | +0.09 pp | 29.56 % | -0.68 pp |
| Memory Throughput (%) | 76.33 % | 76.55 % | +0.22 pp | 76.32 % | -0.01 pp |
| DRAM Throughput (%) | 14.34 % | 14.67 % | +0.33 pp | 14.71 % | +0.37 pp |
| Achieved Occupancy (%) | 17.73 % | 17.72 % | -0.01 pp | 17.70 % | -0.03 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 | 168.00 register/thread | +0 |
| Waves/SM | 5.68 | 5.68 | +0.00 | 5.68 | +0.00 |
| No Eligible (%) | 66.72 % | 66.61 % | -0.11 pp | 67.22 % | +0.50 pp |
| IPC Active | 1.33 inst/cycle | 1.33 inst/cycle | +0.00 | 1.31 inst/cycle | -0.02 |

---

## Iteration 4

**Best so far**: 1.0333× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_block_ptr_order_to_match_k_contiguous_layout | 1.038× | 1.027× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_b_block_ptr_o... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 53730 ns | 52320 ns | -1410 ns (-2.6%) |
| SM Throughput (%) | 30.24 % | 29.87 % | -0.37 pp |
| Memory Throughput (%) | 76.33 % | 77.07 % | +0.74 pp |
| DRAM Throughput (%) | 14.34 % | 14.64 % | +0.30 pp |
| Achieved Occupancy (%) | 17.73 % | 17.72 % | -0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 5.68 | 5.68 | +0.00 |
| No Eligible (%) | 66.72 % | 67.36 % | +0.64 pp |
| IPC Active | 1.33 inst/cycle | 1.31 inst/cycle | -0.02 |

---

## Iteration 5

**Best so far**: 1.0333× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | evict_first_on_streaming_a_load_and_output_store | 1.056× | 1.029× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_on_st... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 53730 ns | 52220 ns | -1510 ns (-2.8%) |
| SM Throughput (%) | 30.24 % | 30.14 % | -0.10 pp |
| Memory Throughput (%) | 76.33 % | 77.08 % | +0.75 pp |
| DRAM Throughput (%) | 14.34 % | 14.67 % | +0.33 pp |
| Achieved Occupancy (%) | 17.73 % | 17.74 % | +0.01 pp |
| Registers/Thread | 168.00 register/thread | 168.00 register/thread | +0 |
| Waves/SM | 5.68 | 5.68 | +0.00 |
| No Eligible (%) | 66.72 % | 67.07 % | +0.35 pp |
| IPC Active | 1.33 inst/cycle | 1.31 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  53730 ns  (1.000×)
Iter 1     52930 ns  (1.015×)  — tf32_tensor_cores
Iter 3     52000 ns  (1.033×)  — tma_loads_with_correct_b_matrix_coalesced_access

Final best: 52220 ns  (1.0289×)
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
+        triton.Config({}, num_warps=4, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=1),
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=8, num_stages=1),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=2, num_stages=2),
+        triton.Config({}, num_warps=2, num_stages=1),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_7(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -57,20 +71,26 @@

     offs_k = tl.arange(0, BLOCK_K)
     acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
 
+    b_block_ptr = tl.make_block_ptr(
+        base=B,
+        shape=(K, N),
+        strides=(stride_bk, stride_bn),
+        offsets=(0, pid_n * BLOCK_N),
+        block_shape=(BLOCK_K, BLOCK_N),
+        order=(0, 1),
+    )
+
     for k_idx in range(0, tl.cdiv(K, BLOCK_K)):
         
         a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
-        b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)
 
         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 16*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 96*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 16*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option='zero')
+        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +105,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 96*idx_m
-    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask)
-
-
+    tl.store(out_ptr0 + (tl.broadcast_to(xindex, [BLOCK_M, BLOCK_N])), acc, mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
