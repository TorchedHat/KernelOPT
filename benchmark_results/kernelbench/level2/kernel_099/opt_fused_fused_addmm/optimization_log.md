# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 7980000 ns — **Best latency**: 7990000 ns (**0.9987×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_099/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 1.94
- **Registers/thread**: 255
- **Duration**: 7980000 ns
- **SM Throughput**: 41.65 % | **Memory**: 57.24 % | **DRAM**: 0.93 %
- **Achieved Occupancy**: 12.50 %
- **IPC Active**: 1.78 inst/cycle | **No Eligible**: 55.36 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads | 1.001× | 0.999× | ✓ | **✓ best** |
| B | memory_coalescing_via_contiguous_stride_and_register_reduction | 1.001× | 0.999× | ✓ |  |
| C | online_softmax_single_pass_with_gelu_cache | 1.001× | 0.999× | ✓ |  |
| D | reduce_register_pressure_via_smaller_block_n_configs | 0.998× | 1.001× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | memory_coalescing... | Δ | online_softmax_si... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 7990000 ns | +10000 ns (+0.1%) | 7990000 ns | +10000 ns (+0.1%) | 7990000 ns | +10000 ns (+0.1%) | 7970000 ns | -10000 ns (-0.1%) |
| SM Throughput (%) | 41.65 % | 41.71 % | +0.06 pp | 41.48 % | -0.17 pp | 41.68 % | +0.03 pp | 41.68 % | +0.03 pp |
| Memory Throughput (%) | 57.24 % | 57.34 % | +0.10 pp | 57.02 % | -0.22 pp | 57.29 % | +0.05 pp | 57.29 % | +0.05 pp |
| DRAM Throughput (%) | 0.93 % | 0.94 % | +0.01 pp | 0.95 % | +0.02 pp | 0.96 % | +0.03 pp | 0.95 % | +0.02 pp |
| Achieved Occupancy (%) | 12.50 % | 12.50 % | +0.00 pp | 12.46 % | -0.04 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.36 % | 55.40 % | +0.04 pp | 55.54 % | +0.18 pp | 55.38 % | +0.02 pp | 55.55 % | +0.19 pp |
| IPC Active | 1.78 inst/cycle | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | cap_registers_via_maxnreg_to_improve_occupancy | 1.000× | 1.002× | ✓ |  |
| B | cap_registers_to_increase_occupancy | 1.000× | 1.000× | ✓ |  |
| C | shared_memory_gelu_cache_to_eliminate_recompute_and_register_spills | 1.000× | 1.000× | ✓ |  |
| D | cache_gelu_in_output_buffer_to_reduce_register_pressure | 1.000× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | cap_registers_via... | Δ | cap_registers_to_... | Δ | shared_memory_gel... | Δ | cache_gelu_in_out... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 7960000 ns | -20000 ns (-0.3%) | 7980000 ns | +0 ns (+0.0%) | 7980000 ns | +0 ns (+0.0%) | 7990000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 41.65 % | 41.72 % | +0.07 pp | 41.67 % | +0.02 pp | 41.79 % | +0.14 pp | 41.58 % | -0.07 pp |
| Memory Throughput (%) | 57.24 % | 57.35 % | +0.11 pp | 57.28 % | +0.04 pp | 57.44 % | +0.20 pp | 57.15 % | -0.09 pp |
| DRAM Throughput (%) | 0.93 % | 0.96 % | +0.03 pp | 0.94 % | +0.01 pp | 0.94 % | +0.01 pp | 0.94 % | +0.01 pp |
| Achieved Occupancy (%) | 12.50 % | 12.43 % | -0.07 pp | 12.46 % | -0.04 pp | 12.50 % | +0.00 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.36 % | 55.52 % | +0.16 pp | 55.48 % | +0.12 pp | 55.55 % | +0.19 pp | 55.60 % | +0.24 pp |
| IPC Active | 1.78 inst/cycle | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 | 1.79 inst/cycle | +0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | specialize_contiguous_strides_to_eliminate_local_memory_spills | 1.000× | 1.000× | ✓ |  |
| C | fuse_gemm_epilogue_to_eliminate_two_pass_register_spill | 0.999× | 1.000× | ✓ |  |
| D | split_two_pass_softmax_into_separate_row_reduction_and_store_passes_with_smaller_block_n | 0.995× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | specialize_contig... | Δ | fuse_gemm_epilogu... | Δ | split_two_pass_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 7980000 ns | 7980000 ns | +0 ns (+0.0%) | 7980000 ns | +0 ns (+0.0%) | 7990000 ns | +10000 ns (+0.1%) |
| SM Throughput (%) | 41.65 % | 41.71 % | +0.06 pp | 41.71 % | +0.06 pp | 41.65 % | +0.00 pp |
| Memory Throughput (%) | 57.24 % | 57.35 % | +0.11 pp | 57.34 % | +0.10 pp | 57.25 % | +0.01 pp |
| DRAM Throughput (%) | 0.93 % | 0.94 % | +0.01 pp | 0.95 % | +0.02 pp | 0.94 % | +0.01 pp |
| Achieved Occupancy (%) | 12.50 % | 12.46 % | -0.04 pp | 12.46 % | -0.04 pp | 12.50 % | +0.00 pp |
| Registers/Thread | 255.00 register/thread | 255.00 register/thread | +0 | 255.00 register/thread | +0 | 255.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 55.36 % | 55.58 % | +0.22 pp | 55.51 % | +0.15 pp | 55.38 % | +0.02 pp |
| IPC Active | 1.78 inst/cycle | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 | 1.78 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  7980000 ns  (1.000×)

Final best: 7990000 ns  (0.9987×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,6 +5,7 @@

 
 
 # GEMM kernel: computes X @ W^T (no bias) -> [M, N]
+
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
@@ -62,20 +63,28 @@

     tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc, mask=mask)
 
 
-# Fused kernel: bias_add + GELU + Softmax (two-pass over columns)
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
-        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
-        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
-        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
-        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
+        triton.Config({'BLOCK_N': 64},   num_stages=2, num_warps=4,  maxnreg=128),
+        triton.Config({'BLOCK_N': 64},   num_stages=2, num_warps=8,  maxnreg=128),
+        triton.Config({'BLOCK_N': 128},  num_stages=2, num_warps=4,  maxnreg=128),
+        triton.Config({'BLOCK_N': 128},  num_stages=2, num_warps=8,  maxnreg=128),
+        triton.Config({'BLOCK_N': 128},  num_stages=2, num_warps=16, maxnreg=128),
+        triton.Config({'BLOCK_N': 256},  num_stages=2, num_warps=4,  maxnreg=128),
+        triton.Config({'BLOCK_N': 256},  num_stages=2, num_warps=8,  maxnreg=128),
+        triton.Config({'BLOCK_N': 256},  num_stages=2, num_warps=16, maxnreg=128),
+        triton.Config({'BLOCK_N': 512},  num_stages=2, num_warps=8,  maxnreg=128),
+        triton.Config({'BLOCK_N': 512},  num_stages=2, num_warps=16, maxnreg=128),
+        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8,  maxnreg=128),
+        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=16, maxnreg=128),
+        triton.Config({'BLOCK_N': 128},  num_stages=2, num_warps=8,  maxnreg=64),
+        triton.Config({'BLOCK_N': 256},  num_stages=2, num_warps=8,  maxnreg=64),
+        triton.Config({'BLOCK_N': 512},  num_stages=2, num_warps=8,  maxnreg=64),
     ],
     key=['N'],
 )
 @triton.jit
+
 def bias_gelu_softmax_kernel(
     X_ptr, bias_ptr, out_ptr,
     M, N,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
