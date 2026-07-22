# Kernel Optimization Log

## Summary

- **Kernel**: `l2_norm_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 9980000 ns — **Best latency**: 9910000 ns (**1.0071×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`increase_block_rows_and_pipeline_depth_for_l2_reuse`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `l2_norm_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_039/synthesis/kernel_0_triton_red_fused_div_linalg_vector_norm_.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (32768,1,1), Waves/SM: 31.03
- **Registers/thread**: 30
- **Duration**: 9980000 ns
- **SM Throughput**: 23.31 % | **Memory**: 58.25 % | **DRAM**: 58.25 %
- **Achieved Occupancy**: 95.48 %
- **IPC Active**: 0.93 inst/cycle | **No Eligible**: 76.82 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policy_streaming_pass1_loads | 0.925× | 1.088× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints | 1.000× | 1.001× | ✓ |  |
| C | memory_coalescing_multi_row_tiles | 0.675× | 1.004× | ✓ |  |
| D | increase_block_rows_and_pipeline_depth_for_l2_reuse | 1.024× | 1.007× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policy_s... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_block_ro... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9980000 ns | 9170000 ns | -810000 ns (-8.1%) | 9970000 ns | -10000 ns (-0.1%) | 9940000 ns | -40000 ns (-0.4%) | 9910000 ns | -70000 ns (-0.7%) |
| SM Throughput (%) | 23.31 % | 28.98 % | +5.67 pp | 23.17 % | -0.14 pp | 23.18 % | -0.13 pp | 23.46 % | +0.15 pp |
| Memory Throughput (%) | 58.25 % | 60.28 % | +2.03 pp | 58.30 % | +0.05 pp | 58.33 % | +0.08 pp | 58.46 % | +0.21 pp |
| DRAM Throughput (%) | 58.25 % | 60.28 % | +2.03 pp | 58.30 % | +0.05 pp | 58.33 % | +0.08 pp | 58.46 % | +0.21 pp |
| Achieved Occupancy (%) | 95.48 % | 96.32 % | +0.84 pp | 95.52 % | +0.04 pp | 95.48 % | +0.00 pp | 95.52 % | +0.04 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 76.82 % | 70.91 % | -5.91 pp | 76.82 % | +0.00 pp | 76.70 % | -0.12 pp | 76.72 % | -0.10 pp |
| IPC Active | 0.93 inst/cycle | 1.17 inst/cycle | +0.24 | 0.93 inst/cycle | +0.00 | 0.93 inst/cycle | +0.00 | 0.93 inst/cycle | +0.00 |


### Key Learnings

#### ✅ Plan A: `eviction_policy_streaming_pass1_loads` (1.088×)

In multi-pass kernels where Pass 1 reads data purely for reduction (e.g., sum-of-squares) and Pass 2 re-reads the same data for normalization, Pass 1 loads pollute the L2 cache and evict lines needed by Pass 2. Adding eviction_policy='evict_first' to Pass 1 tl.load calls (and write-once tl.store calls) tells the L2 to deprioritize those lines, preserving L2 capacity for Pass 2 reads. This reduces L1TEX scoreboard stall cycles (65.7→52.9 cycles/instruction) and improves IPC (0.93→1.17), yielding ~8.8% speedup on a memory-bound kernel where CPIStall accounts for >80% of stalls.

> **Applicable when**: Kernel performs 2+ passes over the same large input tensor; Pass 1 is a pure reduction (streaming, no reuse); L2 hit rate is low (<50%) despite data being read multiple times; CPIStall rule fires with L1TEX scoreboard stalls >70% of total stall cycles; working set vastly exceeds L2 capacity.

> **Avoid when**: Single-pass kernels where data is read only once (evict_first provides no benefit over default); kernels where Pass 1 data IS reused within the same pass (e.g., attention with KV reuse); when L2 hit rate is already high (>70%) indicating the cache is working well.

---

## Iteration 2

**Best so far**: 1.0071× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_rsqrt_and_pipelined_autotune_configs | 0.933× | 1.084× | ✓ |  |
| B | fix_grid_launch_to_exploit_block_rows_tiling | 1.023× | 1.003× | ✓ |  |
| C | fma_fused_pass1_accumulation | 0.577× | 0.856× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_rsqrt_and_p... | Δ | fix_grid_launch_t... | Δ | fma_fused_pass1_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 9980000 ns | 9210000 ns | -770000 ns (-7.7%) | 9950000 ns | -30000 ns (-0.3%) | 11660000 ns | +1680000 ns (+16.8%) |
| SM Throughput (%) | 23.31 % | 26.09 % | +2.78 pp | 23.25 % | -0.06 pp | 15.19 % | -8.12 pp |
| Memory Throughput (%) | 58.25 % | 60.12 % | +1.87 pp | 58.32 % | +0.07 pp | 54.49 % | -3.76 pp |
| DRAM Throughput (%) | 58.25 % | 60.12 % | +1.87 pp | 58.32 % | +0.07 pp | 54.49 % | -3.76 pp |
| Achieved Occupancy (%) | 95.48 % | 96.29 % | +0.81 pp | 95.47 % | -0.01 pp | 94.23 % | -1.25 pp |
| Registers/Thread | 30.00 register/thread | 32.00 register/thread | +2 | 30.00 register/thread | +0 | 32.00 register/thread | +2 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 76.82 % | 73.82 % | -3.00 pp | 76.74 % | -0.08 pp | 84.73 % | +7.91 pp |
| IPC Active | 0.93 inst/cycle | 1.05 inst/cycle | +0.12 | 0.93 inst/cycle | +0.00 | 0.61 inst/cycle | -0.32 |

---

## Iteration 3

**Best so far**: 1.0071× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | pad_n_power_of_two_at_wrapper_level_eliminate_partial_sectors | 0.968× | 1.082× | ✓ |  |
| C | revert_2d_accumulator_and_fix_grid_launch | 0.732× | 1.082× | ✓ |  |


### Metric Comparison

| Metric | Baseline | pad_n_power_of_tw... | Δ | revert_2d_accumul... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 9980000 ns | 9220000 ns | -760000 ns (-7.6%) | 9220000 ns | -760000 ns (-7.6%) |
| SM Throughput (%) | 23.31 % | 22.25 % | -1.06 pp | 28.93 % | +5.62 pp |
| Memory Throughput (%) | 58.25 % | 60.10 % | +1.85 pp | 60.00 % | +1.75 pp |
| DRAM Throughput (%) | 58.25 % | 60.10 % | +1.85 pp | 60.00 % | +1.75 pp |
| Achieved Occupancy (%) | 95.48 % | 96.30 % | +0.82 pp | 96.28 % | +0.80 pp |
| Registers/Thread | 30.00 register/thread | 28.00 register/thread | -2 | 30.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 |
| No Eligible (%) | 76.82 % | 77.67 % | +0.85 pp | 70.98 % | -5.84 pp |
| IPC Active | 0.93 inst/cycle | 0.90 inst/cycle | -0.03 | 1.17 inst/cycle | +0.24 |

---

## Cumulative Progress

```
Baseline:  9980000 ns  (1.000×)
Iter 1     9910000 ns  (1.007×)  — increase_block_rows_and_pipeline_depth_for_l2_reuse

Final best: 9910000 ns  (1.0071×)
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
@@ -11,6 +12,12 @@

         triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=2),
         triton.Config({'BLOCK_N': 256, 'BLOCK_ROWS': 1}, num_warps=4, num_stages=4),
         triton.Config({'BLOCK_N': 8192, 'BLOCK_ROWS': 1}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=5),
+        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=5),
+        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=6),
+        triton.Config({'BLOCK_N': 512, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=6),
+        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=4),
     ],
     key=['N'],
 )
@@ -37,20 +44,19 @@

     for col_off in tl.range(0, N, BLOCK_N):
         cols = col_off + tl.arange(0, BLOCK_N)
         mask = cols < N
-        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
+        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
         acc += tl.sum(x * x, axis=0)
     
-    # Compute L2 norm
-    norm = tl.sqrt(acc)
+    # Compute inverse L2 norm (rsqrt is single MUFU.RSQ instruction, avoids FDIV)
+    inv_norm = tl.where(acc > 0.0, tl.math.rsqrt(acc), 0.0)
     
-    # Pass 2: divide by norm and store
+    # Pass 2: multiply by inv_norm and store (FMUL instead of FDIV)
     for col_off in tl.range(0, N, BLOCK_N):
         cols = col_off + tl.arange(0, BLOCK_N)
         mask = cols < N
         x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
-        out = x / norm
-        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask)
-
+        out = x * inv_norm
+        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask, eviction_policy='evict_first')
 
 def kernel_function(x):
     M, N = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
