# Kernel Optimization Log

## Summary

- **Kernel**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 196000 ns — **Best latency**: 193660 ns (**1.0121×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`tf32_tensor_cores`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_029/kernel_54_triton_tem_fused__unsafe_view_/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (192,1,1), Waves/SM: 0.73
- **Registers/thread**: 255
- **Duration**: 196000 ns
- **SM Throughput**: 27.33 % | **Memory**: 62.13 % | **DRAM**: 1.15 %
- **Achieved Occupancy**: 9.61 %
- **IPC Active**: 1.35 inst/cycle | **No Eligible**: 66.11 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | tf32_tensor_cores | 1.002× | 1.012× | ✓ | **✓ best** |
| B | tma_loads_with_corrected_b_matrix_indexing | 0.977× | 0.975× | ✓ |  |
| C | annotate_loads_stores_with_eviction_policies | 1.002× | 1.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tf32_tensor_cores | Δ | tma_loads_with_co... | Δ | annotate_loads_st... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 196000 ns | 193660 ns | -2340 ns (-1.2%) | 200990 ns | +4990 ns (+2.5%) | 194080 ns | -1920 ns (-1.0%) |
| SM Throughput (%) | 27.33 % | 27.78 % | +0.45 pp | 26.76 % | -0.57 pp | 27.90 % | +0.57 pp |
| Memory Throughput (%) | 62.13 % | 63.16 % | +1.03 pp | 60.83 % | -1.30 pp | 63.41 % | +1.28 pp |
| DRAM Throughput (%) | 1.15 % | 1.16 % | +0.01 pp | 1.12 % | -0.03 pp | 1.16 % | +0.01 pp |
| Achieved Occupancy (%) | 9.61 % | 9.60 % | -0.01 pp | 9.64 % | +0.03 pp | 9.61 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.73 | +0.00 | 0.73 | +0.00 | 0.73 | +0.00 |
| No Eligible (%) | 66.11 % | 66.67 % | +0.56 pp | 66.83 % | +0.72 pp | 66.14 % | +0.03 pp |
| IPC Active | 1.35 inst/cycle | 1.33 inst/cycle | -0.02 | 1.33 inst/cycle | -0.02 | 1.34 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | cap_registers_with_maxnreg_to_eliminate_spill_and_raise_occupancy | 1.001× | 0.990× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_wit... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 196000 ns | 197950 ns | +1950 ns (+1.0%) |
| SM Throughput (%) | 27.33 % | 27.01 % | -0.32 pp |
| Memory Throughput (%) | 62.13 % | 61.01 % | -1.12 pp |
| DRAM Throughput (%) | 1.15 % | 1.13 % | -0.02 pp |
| Achieved Occupancy (%) | 9.61 % | 9.64 % | +0.03 pp |
| Registers/Thread | 255.00 register/thread | 254.00 register/thread | -1 |
| Waves/SM | 0.73 | 0.73 | +0.00 |
| No Eligible (%) | 66.11 % | 67.14 % | +1.03 pp |
| IPC Active | 1.35 inst/cycle | 1.31 inst/cycle | -0.04 |

---

## Iteration 4

**Best so far**: 1.0121× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | tma_loads_via_block_ptr_for_bank_conflict_elimination | 0.967× | 0.976× | ✓ |  |


### Metric Comparison

| Metric | Baseline | tma_loads_via_blo... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 196000 ns | 200930 ns | +4930 ns (+2.5%) |
| SM Throughput (%) | 27.33 % | 26.76 % | -0.57 pp |
| Memory Throughput (%) | 62.13 % | 60.43 % | -1.70 pp |
| DRAM Throughput (%) | 1.15 % | 1.12 % | -0.03 pp |
| Achieved Occupancy (%) | 9.61 % | 9.67 % | +0.06 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 |
| Waves/SM | 0.73 | 0.73 | +0.00 |
| No Eligible (%) | 66.11 % | 66.93 % | +0.82 pp |
| IPC Active | 1.35 inst/cycle | 1.32 inst/cycle | -0.03 |

---

## Cumulative Progress

```
Baseline:  196000 ns  (1.000×)
Iter 2     193660 ns  (1.012×)  — tf32_tensor_cores

Final best: 193660 ns  (1.0121×)
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
+        triton.Config({}, num_warps=8, num_stages=2),
+        triton.Config({}, num_warps=4, num_stages=3),
+        triton.Config({}, num_warps=8, num_stages=3),
+        triton.Config({}, num_warps=4, num_stages=4),
+        triton.Config({}, num_warps=8, num_stages=4),
+    ],
+    key=[],
+)
 @triton.jit
 def triton_tem_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_t_transpose_view_54(arg_A, arg_B, out_ptr0):
     EVEN_K : tl.constexpr = True
@@ -65,12 +77,12 @@

         idx_m = offs_a_m[:, None]
         idx_n = a_k_idx_vals
         xindex = idx_n + 768*idx_m
-        a = tl.load(A + (xindex))
+        a = tl.load(A + (xindex), eviction_policy='evict_first')
 
         idx_m = b_k_idx_vals
         idx_n = offs_b_n[None, :]
         xindex = idx_n + 3072*idx_m
-        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)))
+        b = tl.load(B + ((tl.broadcast_to(idx_m + 768*idx_n, [BLOCK_K, BLOCK_N])).broadcast_to(xindex.shape)), eviction_policy='evict_last')
 
 
         acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)
@@ -85,9 +97,7 @@

 
     # inductor generates a suffix
     xindex = idx_n + 3072*idx_m
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
