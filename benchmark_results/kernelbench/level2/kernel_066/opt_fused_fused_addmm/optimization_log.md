# Kernel Optimization Log

## Summary

- **Kernel**: `gemm_bias_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 4300000 ns — **Best latency**: 4300000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gemm_bias_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L2_gpu4/kernel_066/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (256,1,1), Waves/SM: 0.97
- **Registers/thread**: 242
- **Duration**: 4300000 ns
- **SM Throughput**: 35.76 % | **Memory**: 79.19 % | **DRAM**: 5.17 %
- **Achieved Occupancy**: 12.25 %
- **IPC Active**: 1.48 inst/cycle | **No Eligible**: 62.99 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_softmax_loads | 1.000× | 0.998× | ✓ |  |
| B | fix_weight_load_coalescing_to_eliminate_shared_memory_bank_conflicts | 1.000× | 1.000× | ✓ |  |
| C | online_softmax_attention | 1.000× | 1.000× | ✓ |  |
| D | fuse_softmax_passes_and_expand_autotune_configs | 1.000× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | fix_weight_load_c... | Δ | online_softmax_at... | Δ | fuse_softmax_pass... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4300000 ns | 4310000 ns | +10000 ns (+0.2%) | 4300000 ns | +0 ns (+0.0%) | 4300000 ns | +0 ns (+0.0%) | 4300000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 35.76 % | 35.82 % | +0.06 pp | 35.83 % | +0.07 pp | 35.84 % | +0.08 pp | 35.80 % | +0.04 pp |
| Memory Throughput (%) | 79.19 % | 79.31 % | +0.12 pp | 79.34 % | +0.15 pp | 79.35 % | +0.16 pp | 79.27 % | +0.08 pp |
| DRAM Throughput (%) | 5.17 % | 5.17 % | +0.00 pp | 5.17 % | +0.00 pp | 5.17 % | +0.00 pp | 5.17 % | +0.00 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp |
| Registers/Thread | 242.00 register/thread | 242.00 register/thread | +0 | 242.00 register/thread | +0 | 242.00 register/thread | +0 | 242.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 62.99 % | 63.02 % | +0.03 pp | 63.00 % | +0.01 pp | 63.00 % | +0.01 pp | 63.00 % | +0.01 pp |
| IPC Active | 1.48 inst/cycle | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_softmax_kernel_with_maxnreg | 0.998× | 1.002× | ✓ |  |
| B | fix_weight_load_layout_and_reduce_register_pressure | 0.999× | 1.000× | ✓ |  |
| C | reduce_softmax_block_n_to_cut_register_pressure_and_raise_occupancy | 1.000× | 0.998× | ✓ |  |
| D | reduce_register_pressure_via_smaller_block_n_and_online_softmax_fusion | 1.000× | 0.998× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | fix_weight_load_l... | Δ | reduce_softmax_bl... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4300000 ns | 4290000 ns | -10000 ns (-0.2%) | 4300000 ns | +0 ns (+0.0%) | 4310000 ns | +10000 ns (+0.2%) | 4310000 ns | +10000 ns (+0.2%) |
| SM Throughput (%) | 35.76 % | 35.85 % | +0.09 pp | 35.81 % | +0.05 pp | 35.83 % | +0.07 pp | 35.73 % | -0.03 pp |
| Memory Throughput (%) | 79.19 % | 79.39 % | +0.20 pp | 79.28 % | +0.09 pp | 79.34 % | +0.15 pp | 79.12 % | -0.07 pp |
| DRAM Throughput (%) | 5.17 % | 5.18 % | +0.01 pp | 5.17 % | +0.00 pp | 5.16 % | -0.01 pp | 5.16 % | -0.01 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp |
| Registers/Thread | 242.00 register/thread | 242.00 register/thread | +0 | 242.00 register/thread | +0 | 242.00 register/thread | +0 | 242.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 62.99 % | 63.01 % | +0.02 pp | 63.02 % | +0.03 pp | 63.01 % | +0.02 pp | 63.00 % | +0.01 pp |
| IPC Active | 1.48 inst/cycle | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_on_softmax_streaming_pass_loads | 0.999× | 1.002× | ✓ |  |
| B | evict_first_streaming_softmax_passes | 1.000× | 1.002× | ✓ |  |
| C | eviction_policies_softmax_streaming_passes | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | evict_first_on_so... | Δ | evict_first_strea... | Δ | eviction_policies... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 4300000 ns | 4290000 ns | -10000 ns (-0.2%) | 4290000 ns | -10000 ns (-0.2%) | 4300000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 35.76 % | 35.80 % | +0.04 pp | 35.79 % | +0.03 pp | 35.80 % | +0.04 pp |
| Memory Throughput (%) | 79.19 % | 79.27 % | +0.08 pp | 79.24 % | +0.05 pp | 79.28 % | +0.09 pp |
| DRAM Throughput (%) | 5.17 % | 5.18 % | +0.01 pp | 5.18 % | +0.01 pp | 5.18 % | +0.01 pp |
| Achieved Occupancy (%) | 12.25 % | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp | 12.25 % | +0.00 pp |
| Registers/Thread | 242.00 register/thread | 242.00 register/thread | +0 | 242.00 register/thread | +0 | 242.00 register/thread | +0 |
| Waves/SM | 0.97 | 0.97 | +0.00 | 0.97 | +0.00 | 0.97 | +0.00 |
| No Eligible (%) | 62.99 % | 62.99 % | +0.00 pp | 63.01 % | +0.02 pp | 63.02 % | +0.03 pp |
| IPC Active | 1.48 inst/cycle | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 | 1.48 inst/cycle | +0.00 |

---

## Cumulative Progress

```
Baseline:  4300000 ns  (1.000×)

Final best: 4300000 ns  (1.0000×)
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
@@ -22,51 +23,16 @@

     stride_om, stride_on,
     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
 ):
-    # Each program handles one row of output (for softmax)
-    # But we need to compute the full row for softmax normalization
-    # Strategy: each program_id(0) handles one row, iterates over N in BLOCK_N chunks
-    # This is a row-wise softmax after GEMM
-    
     row_id = tl.program_id(0)
-    
+
     if row_id >= M:
         return
-    
-    # We need to compute the full row first, then do softmax
-    # For large N (16384), we do online softmax
-    
-    # Phase 1: Compute GEMM row and find max (online)
+
     row_max = tl.full((), float('-inf'), dtype=tl.float32)
     row_sum = tl.zeros((), dtype=tl.float32)
-    
-    # We'll store intermediate results... but 16384 floats per row is too much for SRAM
-    # Need a different approach: 2-pass over N
-    
-    # Actually for softmax we need 2 passes over the data
-    # Pass 1: compute all GEMM values and find max + sum
-    # Pass 2: normalize
-    # But storing 16384 values per row in SRAM is not feasible
-    
-    # Better: store to global memory, then do softmax
-    # But that defeats the purpose of fusion
-    
-    # Let's do: each row is handled by one program
-    # Pass 1: iterate over N in BLOCK_N chunks, compute GEMM tile, track max
-    # Pass 2: iterate over N again, compute exp and sum
-    # Pass 3: normalize
-    # This reads X*K twice but avoids storing intermediate GEMM output
-    
-    # Actually we need to store the GEMM output temporarily or recompute
-    # For 16384 cols, storing in SRAM is not feasible (64KB limit)
-    # We'll use global memory as temp buffer
-    
-    # This approach won't work well as a single kernel without temp storage
-    # Let's use a different decomposition
+
     pass
 
-
-# Better approach: separate GEMM kernel + fused softmax kernel
-# The GEMM produces output[128, 16384], then softmax reads it once
 
 @triton.autotune(
     configs=[
@@ -76,10 +42,14 @@

         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
         triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
         triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
     ],
     key=['M', 'N', 'K'],
 )
 @triton.jit
+
 def gemm_bias_kernel(
     X_ptr, W_ptr, B_ptr, Out_ptr,
     M, N, K,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
