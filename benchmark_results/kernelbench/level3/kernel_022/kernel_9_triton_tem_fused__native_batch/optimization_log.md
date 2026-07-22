# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 27810 ns — **Best latency**: 26850 ns (**1.0358×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fix_b_matrix_load_index_bug_and_add_tma_loads`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_022/kernel_9_triton_tem_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (245,1,1), Waves/SM: 0.46
- **Registers/thread**: 128
- **Duration**: 27810 ns
- **SM Throughput**: 17.80 % | **Memory**: 73.08 % | **DRAM**: 8.86 %
- **Achieved Occupancy**: 11.83 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 79.10 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 0.994× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 27810 ns | 27780 ns | -30 ns (-0.1%) |
| SM Throughput (%) | 17.80 % | 17.76 % | -0.04 pp |
| Memory Throughput (%) | 73.08 % | 72.93 % | -0.15 pp |
| DRAM Throughput (%) | 8.86 % | 8.86 % | +0.00 pp |
| Achieved Occupancy (%) | 11.83 % | 11.81 % | -0.02 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 |
| No Eligible (%) | 79.10 % | 79.14 % | +0.04 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_b_matrix_coalesced_load_via_shared_memory_transpose | 0.997× | 1.004× | ✓ |  |
| B | fix_b_matrix_load_index_bug_and_add_tma_loads | 1.016× | 1.036× | ✓ | **✓ best** |
| C | eviction_policies_streaming_a_evict_first_reuse_b_evict_last | 0.997× | 1.006× | ✓ |  |
| D | fix_b_matrix_load_coalescing_and_increase_tile_sizes | 1.002× | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_coal... | Δ | fix_b_matrix_load... | Δ | eviction_policies... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27810 ns | 27710 ns | -100 ns (-0.4%) | 26850 ns | -960 ns (-3.5%) | 27650 ns | -160 ns (-0.6%) | 27970 ns | +160 ns (+0.6%) |
| SM Throughput (%) | 17.80 % | 17.78 % | -0.02 pp | 17.53 % | -0.27 pp | 17.82 % | +0.02 pp | 17.72 % | -0.08 pp |
| Memory Throughput (%) | 73.08 % | 72.95 % | -0.13 pp | 74.01 % | +0.93 pp | 73.05 % | -0.03 pp | 72.68 % | -0.40 pp |
| DRAM Throughput (%) | 8.86 % | 8.88 % | +0.02 pp | 9.17 % | +0.31 pp | 8.90 % | +0.04 pp | 8.79 % | -0.07 pp |
| Achieved Occupancy (%) | 11.83 % | 11.82 % | -0.01 pp | 11.80 % | -0.03 pp | 11.81 % | -0.02 pp | 11.84 % | +0.01 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.10 % | 79.07 % | -0.03 pp | 79.20 % | +0.10 pp | 79.07 % | -0.03 pp | 79.13 % | +0.03 pp |
| IPC Active | 0.83 inst/cycle | 0.84 inst/cycle | +0.01 | 0.83 inst/cycle | +0.00 | 0.84 inst/cycle | +0.01 | 0.83 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0358× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_b_matrix_index_correctness_and_eliminate_bank_conflicts_via_coalesced_load | 0.994× | 1.010× | ✓ |  |
| C | eliminate_b_matrix_transpose_to_fix_shared_memory_bank_conflicts | 1.000× | 1.006× | ✓ |  |
| D | software_pipeline_num_stages_to_hide_mio_stalls | 0.994× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_b_matrix_inde... | Δ | eliminate_b_matri... | Δ | software_pipeline... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 27810 ns | 27520 ns | -290 ns (-1.0%) | 27650 ns | -160 ns (-0.6%) | 27870 ns | +60 ns (+0.2%) |
| SM Throughput (%) | 17.80 % | 17.56 % | -0.24 pp | 17.75 % | -0.05 pp | 16.62 % | -1.18 pp |
| Memory Throughput (%) | 73.08 % | 71.99 % | -1.09 pp | 72.80 % | -0.28 pp | 73.85 % | +0.77 pp |
| DRAM Throughput (%) | 8.86 % | 8.94 % | +0.08 pp | 8.90 % | +0.04 pp | 8.83 % | -0.03 pp |
| Achieved Occupancy (%) | 11.83 % | 11.82 % | -0.01 pp | 11.82 % | -0.01 pp | 11.87 % | +0.04 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.10 % | 79.06 % | -0.04 pp | 79.10 % | +0.00 pp | 80.34 % | +1.24 pp |
| IPC Active | 0.83 inst/cycle | 0.84 inst/cycle | +0.01 | 0.84 inst/cycle | +0.01 | 0.79 inst/cycle | -0.04 |

---

## Iteration 4

**Best so far**: 1.0358× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_register_usage_with_maxnreg_to_increase_occupancy | 1.001× | 1.006× | ✓ |  |
| B | fix_b_matrix_load_to_contiguous_k_inner_layout_with_register_cap | 0.994× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_register_usag... | Δ | fix_b_matrix_load... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 27810 ns | 27650 ns | -160 ns (-0.6%) | 27620 ns | -190 ns (-0.7%) |
| SM Throughput (%) | 17.80 % | 17.15 % | -0.65 pp | 17.71 % | -0.09 pp |
| Memory Throughput (%) | 73.08 % | 71.37 % | -1.71 pp | 72.68 % | -0.40 pp |
| DRAM Throughput (%) | 8.86 % | 8.90 % | +0.04 pp | 8.91 % | +0.05 pp |
| Achieved Occupancy (%) | 11.83 % | 11.90 % | +0.07 pp | 11.80 % | -0.03 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.46 | 0.46 | +0.00 | 0.46 | +0.00 |
| No Eligible (%) | 79.10 % | 79.78 % | +0.68 pp | 79.10 % | +0.00 pp |
| IPC Active | 0.83 inst/cycle | 0.81 inst/cycle | -0.02 | 0.84 inst/cycle | +0.01 |

---

## Cumulative Progress

```
Baseline:  27810 ns  (1.000×)
Iter 2     26850 ns  (1.036×)  — fix_b_matrix_load_index_bug_and_add_tma_loads

Final best: 26850 ns  (1.0358×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,10 +5,11 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
 @triton.jit
 def triton_tem_fused__native_batch_norm_legit_no_training_convolution_hardtanh_9(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
-    USE_FAST_ACCUM : tl.constexpr = False
+    USE_FAST_ACCUM : tl.constexpr = True
     ACC_TYPE : tl.constexpr = tl.float32
     BLOCK_M : tl.constexpr = 128
     BLOCK_N : tl.constexpr = 32
@@ -65,12 +66,13 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 96*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
-        idx_m = b_k_idx_vals
-        idx_n = offs_b_n[None, :]
-        xindex = idx_n + 24*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 96*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        idx_m = b_k_idx_vals          # shape [BLOCK_K, 1]
+        idx_n = offs_b_n[None, :]     # shape [1, BLOCK_N]
+        # B has stride_bk=1, stride_bn=96: B[k, n] is at offset k*1 + n*96
+        # idx_m [BLOCK_K,1] + 96*idx_n [1,BLOCK_N] broadcasts naturally to [BLOCK_K, BLOCK_N]
+        b = tl.load(B + (idx_m + 96 * idx_n), eviction_policy='evict_last')
 
         
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +87,7 @@

 
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
