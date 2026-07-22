# Kernel Optimization Log

## Summary

- **Kernel**: `mse_partial_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1870000 ns — **Best latency**: 1870000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `mse_partial_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_094/synthesis/kernel_0_fused_fused_mean.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (32768,1,1), Waves/SM: 41.37
- **Registers/thread**: 33
- **Duration**: 1870000 ns
- **SM Throughput**: 6.72 % | **Memory**: 93.65 % | **DRAM**: 93.65 %
- **Achieved Occupancy**: 74.45 %
- **IPC Active**: 0.27 inst/cycle | **No Eligible**: 93.22 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_evict_first_expanded_autotune_configs | 0.986× | 1.005× | ✓ |  |
| C | expand_autotune_configs_larger_block_row_and_evict_first_streaming_loads | 0.987× | 1.000× | ✓ |  |
| D | fix_uncoalesced_stores_and_vectorize_streaming_loads | 0.986× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | expand_autotune_c... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1870000 ns | 1860000 ns | -10000 ns (-0.5%) | 1870000 ns | +0 ns (+0.0%) | 1890000 ns | +20000 ns (+1.1%) |
| SM Throughput (%) | 6.72 % | 19.99 % | +13.27 pp | 22.47 % | +15.75 pp | 19.79 % | +13.07 pp |
| Memory Throughput (%) | 93.65 % | 93.79 % | +0.14 pp | 93.27 % | -0.38 pp | 92.30 % | -1.35 pp |
| DRAM Throughput (%) | 93.65 % | 93.79 % | +0.14 pp | 93.27 % | -0.38 pp | 92.30 % | -1.35 pp |
| Achieved Occupancy (%) | 74.45 % | 62.03 % | -12.42 pp | 99.23 % | +24.78 pp | 98.76 % | +24.31 pp |
| Registers/Thread | 33.00 register/thread | 48.00 register/thread | +15 | 25.00 register/thread | -8 | 32.00 register/thread | -1 |
| Waves/SM | 41.37 | 49.65 | +8.28 | 31.03 | -10.34 | 15.52 | -25.85 |
| No Eligible (%) | 93.22 % | 79.88 % | -13.34 pp | 77.65 % | -15.57 pp | 79.99 % | -13.23 pp |
| IPC Active | 0.27 inst/cycle | 0.80 inst/cycle | +0.53 | 0.89 inst/cycle | +0.62 | 0.80 inst/cycle | +0.53 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 0

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_partial_buffer_allocation_to_enable_larger_block_row_coalesced_stores | 0.986× | 1.000× | ✓ |  |
| B | fix_partial_buffer_allocation_for_coalesced_stores_with_larger_block_row | 0.987× | 1.000× | ✓ |  |
| C | fix_uncoalesced_stores_correct_partial_buffer_sizing_with_larger_block_row | 0.986× | 0.989× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_partial_buffe... | Δ | fix_partial_buffe... | Δ | fix_uncoalesced_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1870000 ns | 1870000 ns | +0 ns (+0.0%) | 1870000 ns | +0 ns (+0.0%) | 1890000 ns | +20000 ns (+1.1%) |
| SM Throughput (%) | 6.72 % | 20.08 % | +13.36 pp | 22.27 % | +15.55 pp | 19.83 % | +13.11 pp |
| Memory Throughput (%) | 93.65 % | 93.68 % | +0.03 pp | 93.29 % | -0.36 pp | 92.28 % | -1.37 pp |
| DRAM Throughput (%) | 93.65 % | 93.68 % | +0.03 pp | 93.29 % | -0.36 pp | 92.28 % | -1.37 pp |
| Achieved Occupancy (%) | 74.45 % | 62.04 % | -12.41 pp | 99.21 % | +24.76 pp | 98.74 % | +24.29 pp |
| Registers/Thread | 33.00 register/thread | 48.00 register/thread | +15 | 25.00 register/thread | -8 | 32.00 register/thread | -1 |
| Waves/SM | 41.37 | 49.65 | +8.28 | 31.03 | -10.34 | 15.52 | -25.85 |
| No Eligible (%) | 93.22 % | 79.89 % | -13.33 pp | 77.48 % | -15.74 pp | 79.97 % | -13.25 pp |
| IPC Active | 0.27 inst/cycle | 0.80 inst/cycle | +0.53 | 0.90 inst/cycle | +0.63 | 0.79 inst/cycle | +0.52 |

---

## Cumulative Progress

```
Baseline:  1870000 ns  (1.000×)

Final best: 1870000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,26 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 2048}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 512}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 4096}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_ROW': 2, 'BLOCK_COL': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_ROW': 4, 'BLOCK_COL': 512}, num_warps=8, num_stages=2),
+        # BLOCK_ROW=1: grid=32768, good for small inputs
+        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 4096},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_ROW': 1, 'BLOCK_COL': 8192},  num_warps=8,  num_stages=3),
+        # BLOCK_ROW=2: grid=16384, reduces partial store scatter
+        triton.Config({'BLOCK_ROW': 2, 'BLOCK_COL': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_ROW': 2, 'BLOCK_COL': 4096},  num_warps=8,  num_stages=3),
+        # BLOCK_ROW=4: grid=8192
+        triton.Config({'BLOCK_ROW': 4, 'BLOCK_COL': 1024},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_ROW': 4, 'BLOCK_COL': 2048},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_ROW': 4, 'BLOCK_COL': 2048},  num_warps=16, num_stages=4),
+        # BLOCK_ROW=8: grid=4096, good coalescing of partial stores
+        triton.Config({'BLOCK_ROW': 8, 'BLOCK_COL': 512},   num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_ROW': 8, 'BLOCK_COL': 1024},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_ROW': 8, 'BLOCK_COL': 1024},  num_warps=16, num_stages=4),
+        # BLOCK_ROW=16: grid=2048, fewest CTAs, best partial store coalescing
+        triton.Config({'BLOCK_ROW': 16, 'BLOCK_COL': 256},  num_warps=8,  num_stages=3),
+        triton.Config({'BLOCK_ROW': 16, 'BLOCK_COL': 512},  num_warps=16, num_stages=4),
     ],
     key=['N_ROWS', 'N_COLS'],
 )
@@ -34,15 +46,16 @@

     
     for col_start in range(0, N_COLS, BLOCK_COL):
         col_offsets = col_start + tl.arange(0, BLOCK_COL)
+        col_offsets = tl.max_contiguous(tl.multiple_of(col_offsets, BLOCK_COL), BLOCK_COL)
         col_mask = col_offsets < N_COLS
         
         mask = row_mask[:, None] & col_mask[None, :]
         
         pred_ptrs = pred_ptr + row_offsets[:, None] * N_COLS + col_offsets[None, :]
-        tgt_ptrs = tgt_ptr + row_offsets[:, None] * N_COLS + col_offsets[None, :]
+        tgt_ptrs  = tgt_ptr  + row_offsets[:, None] * N_COLS + col_offsets[None, :]
         
-        pred = tl.load(pred_ptrs, mask=mask, other=0.0)
-        tgt = tl.load(tgt_ptrs, mask=mask, other=0.0)
+        pred = tl.load(pred_ptrs, mask=mask, other=0.0, eviction_policy='evict_first')
+        tgt  = tl.load(tgt_ptrs,  mask=mask, other=0.0, eviction_policy='evict_first')
         
         diff = pred - tgt
         sq = diff * diff
@@ -67,6 +80,7 @@

     key=['N_PARTIAL'],
 )
 @triton.jit
+
 def mse_final_kernel(
     partial_ptr,
     out_ptr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
