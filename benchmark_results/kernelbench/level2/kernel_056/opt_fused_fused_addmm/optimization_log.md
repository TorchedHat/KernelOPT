# Kernel Optimization Log

## Summary

- **Kernel**: `linear_sigmoid_sum_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 189410000 ns — **Best latency**: 189410000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `linear_sigmoid_sum_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_056/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (128,1,1), Waves/SM: 0.24
- **Registers/thread**: 128
- **Duration**: 189410000 ns
- **SM Throughput**: 14.51 % | **Memory**: 23.65 % | **DRAM**: 0.52 %
- **Achieved Occupancy**: 6.25 %
- **IPC Active**: 0.54 inst/cycle | **No Eligible**: 86.59 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads | 1.002× | 0.990× | ✓ | **✓ best** |
| B | vectorized_loads_with_contiguous_hints | 1.000× | 1.000× | ✓ |  |
| C | memory_coalescing_and_occupancy_via_BLOCK_M_tiling | 0.194× | 0.208× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 189410000 ns | 191320000 ns | +1910000 ns (+1.0%) | 189420000 ns | +10000 ns (+0.0%) | 912500000 ns | +723090000 ns (+381.8%) |
| SM Throughput (%) | 14.51 % | 14.69 % | +0.18 pp | 14.51 % | +0.00 pp | 11.87 % | -2.64 pp |
| Memory Throughput (%) | 23.65 % | 23.42 % | -0.23 pp | 23.65 % | +0.00 pp | 81.98 % | +58.33 pp |
| DRAM Throughput (%) | 0.52 % | 0.58 % | +0.06 pp | 0.53 % | +0.01 pp | 0.10 % | -0.42 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 128.00 register/thread | +0 | 128.00 register/thread | +0 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.12 | -0.12 |
| No Eligible (%) | 86.59 % | 84.86 % | -1.73 pp | 86.59 % | +0.00 pp | 75.52 % | -11.07 pp |
| IPC Active | 0.54 inst/cycle | 0.61 inst/cycle | +0.07 | 0.54 inst/cycle | +0.00 | 0.49 inst/cycle | -0.05 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_maxnreg_and_pipeline_stages | 0.632× | 0.546× | ✓ |  |
| B | reduce_register_pressure_via_maxnreg_to_boost_occupancy | 0.958× | 0.500× | ✓ |  |
| C | revert_to_scalar_1d_accumulator_eliminate_bank_conflicts | 1.000× | 0.159× | ✓ |  |
| D | add_triton_autotune | 1.059× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | reduce_register_p... | Δ | revert_to_scalar_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 189410000 ns | 346880000 ns | +157470000 ns (+83.1%) | 379060000 ns | +189650000 ns (+100.1%) | 1190000000 ns | +1000590000 ns (+528.3%) | 189410000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 14.51 % | 9.98 % | -4.53 pp | 9.22 % | -5.29 pp | 1.94 % | -12.57 pp | 14.52 % | +0.01 pp |
| Memory Throughput (%) | 23.65 % | 16.24 % | -7.41 pp | 14.98 % | -8.67 pp | 3.42 % | -20.23 pp | 23.66 % | +0.01 pp |
| DRAM Throughput (%) | 0.52 % | 0.33 % | -0.19 pp | 0.32 % | -0.20 pp | 0.08 % | -0.44 pp | 0.51 % | -0.01 pp |
| Achieved Occupancy (%) | 6.25 % | 6.25 % | +0.00 pp | 6.25 % | +0.00 pp | 3.12 % | -3.13 pp | 6.25 % | +0.00 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 | 64.00 register/thread | -64 | 64.00 register/thread | -64 | 128.00 register/thread | +0 |
| Waves/SM | 0.24 | 0.12 | -0.12 | 0.12 | -0.12 | 0.06 | -0.18 | 0.24 | +0.00 |
| No Eligible (%) | 86.59 % | 92.05 % | +5.46 pp | 91.75 % | +5.16 pp | 96.00 % | +9.41 pp | 86.59 % | +0.00 pp |
| IPC Active | 0.54 inst/cycle | 0.32 inst/cycle | -0.22 | 0.33 inst/cycle | -0.21 | 0.08 inst/cycle | -0.46 | 0.54 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_loop_reorder_k_innermost_with_reduced_register_pressure | 0.538× | 0.159× | ✓ |  |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 189410000 ns | 1190000000 ns | +1000590000 ns (+528.3%) |
| SM Throughput (%) | 14.51 % | 1.94 % | -12.57 pp |
| Memory Throughput (%) | 23.65 % | 3.42 % | -20.23 pp |
| DRAM Throughput (%) | 0.52 % | 0.08 % | -0.44 pp |
| Achieved Occupancy (%) | 6.25 % | 3.12 % | -3.13 pp |
| Registers/Thread | 128.00 register/thread | 64.00 register/thread | -64 |
| Waves/SM | 0.24 | 0.06 | -0.18 |
| No Eligible (%) | 86.59 % | 96.00 % | +9.41 pp |
| IPC Active | 0.54 inst/cycle | 0.08 inst/cycle | -0.46 |

---

## Cumulative Progress

```
Baseline:  189410000 ns  (1.000×)

Final best: 189410000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -1,6 +1,7 @@

 import torch
 import triton
 import triton.language as tl
+
 
 @triton.autotune(
     configs=[
@@ -10,6 +11,12 @@

         triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
         triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
         triton.Config({'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_N': 512, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
     ],
     key=['M', 'N', 'K'],
 )
@@ -35,6 +42,7 @@

     # We tile over the output dimension N
     for n_start in tl.range(0, N, BLOCK_N):
         n_offs = n_start + tl.arange(0, BLOCK_N)
+        n_offs = tl.max_contiguous(tl.multiple_of(n_offs, BLOCK_N), BLOCK_N)  # ADD: alignment+contiguity hint
         n_mask = n_offs < N
         
         # Accumulate dot product for this block of output neurons
@@ -42,15 +50,16 @@

         
         for k_start in tl.range(0, K, BLOCK_K):
             k_offs = k_start + tl.arange(0, BLOCK_K)
+            k_offs = tl.max_contiguous(tl.multiple_of(k_offs, BLOCK_K), BLOCK_K)  # ADD: alignment+contiguity hint
             k_mask = k_offs < K
             
             # Load X[row_m, k_offs]: shape [BLOCK_K]
             x_ptrs = X_ptr + row_m * stride_xm + k_offs * stride_xk
-            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K]
+            x_vals = tl.load(x_ptrs, mask=k_mask, other=0.0)  # [BLOCK_K] — now emits 128-bit vectorized loads
             
             # Load W[n_offs, k_offs]: shape [BLOCK_N, BLOCK_K]
             w_ptrs = W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
-            w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K]
+            w_vals = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # [BLOCK_N, BLOCK_K] — vectorized along K
             
             # acc[n] += sum_k(x[k] * w[n,k])
             acc += tl.sum(w_vals * x_vals[None, :], axis=1)
@@ -68,7 +77,6 @@

     
     # Store result
     tl.store(Out_ptr + row_m, total_sum)
-
 
 def kernel_function(weight, bias, x):
     M, K = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
