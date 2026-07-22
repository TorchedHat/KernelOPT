# Kernel Optimization Log

## Summary

- **Kernel**: `hinge_loss_row_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 1160000 ns — **Best latency**: 1160000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `hinge_loss_row_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_100/synthesis/kernel_0_fused_fused_clamp.py`
- **Framework**: triton
- **Launch**: Block (256,1,1), Grid (32768,1,1), Waves/SM: 31.03
- **Registers/thread**: 25
- **Duration**: 1160000 ns
- **SM Throughput**: 51.98 % | **Memory**: 77.26 % | **DRAM**: 75.54 %
- **Achieved Occupancy**: 98.54 %
- **IPC Active**: 2.05 inst/cycle | **No Eligible**: 48.51 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_pred_cache_target | 1.058× | 1.000× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 1.000× | 1.000× | ✓ |  |
| C | memory_coalescing_2d_grid_coalesced_partial_stores | 0.095× | 0.082× | ✓ |  |
| D | fuse_row_and_reduce_kernels_with_2d_tiling | 1.080× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | fuse_row_and_redu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1160000 ns | 1160000 ns | +0 ns (+0.0%) | 1160000 ns | +0 ns (+0.0%) | 14240000 ns | +13080000 ns (+1127.6%) | 1160000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 51.98 % | 56.71 % | +4.73 pp | 51.91 % | -0.07 pp | 59.28 % | +7.30 pp | 52.61 % | +0.63 pp |
| Memory Throughput (%) | 77.26 % | 77.00 % | -0.26 pp | 77.16 % | -0.10 pp | 62.64 % | -14.62 pp | 78.20 % | +0.94 pp |
| DRAM Throughput (%) | 75.54 % | 75.54 % | +0.00 pp | 75.70 % | +0.16 pp | 6.49 % | -69.05 pp | 75.52 % | -0.02 pp |
| Achieved Occupancy (%) | 98.54 % | 98.53 % | -0.01 pp | 98.54 % | +0.00 pp | 12.47 % | -86.07 pp | 98.58 % | +0.04 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 255.00 register/thread | +230 | 25.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 124.12 | +93.09 | 31.03 | +0.00 |
| No Eligible (%) | 48.51 % | 42.84 % | -5.67 pp | 48.47 % | -0.04 pp | 40.49 % | -8.02 pp | 48.68 % | +0.17 pp |
| IPC Active | 2.05 inst/cycle | 2.29 inst/cycle | +0.24 | 2.06 inst/cycle | +0.01 | 2.38 inst/cycle | +0.33 | 2.05 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | eviction_policy_pred_evict_first_target_evict_last | 1.039× | 1.000× | ✓ |  |
| D | reduce_register_pressure_scalar_accumulator_1d_grid | 1.079× | 1.204× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policy_p... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 1160000 ns | 1160000 ns | +0 ns (+0.0%) | 963040 ns | -196960 ns (-17.0%) |
| SM Throughput (%) | 51.98 % | 56.71 % | +4.73 pp | 45.45 % | -6.53 pp |
| Memory Throughput (%) | 77.26 % | 77.01 % | -0.25 pp | 90.87 % | +13.61 pp |
| DRAM Throughput (%) | 75.54 % | 75.55 % | +0.01 pp | 90.87 % | +15.33 pp |
| Achieved Occupancy (%) | 98.54 % | 98.48 % | -0.06 pp | 98.11 % | -0.43 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 30.00 register/thread | +5 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 15.52 | -15.51 |
| No Eligible (%) | 48.51 % | 42.47 % | -6.04 pp | 55.34 % | +6.83 pp |
| IPC Active | 2.05 inst/cycle | 2.30 inst/cycle | +0.25 | 1.78 inst/cycle | -0.27 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | small_block_rows_amortize_target_loads_low_register_pressure | 1.014× | 1.115× | ✓ |  |
| B | vectorized_loads_max_contiguous_multiple_of_hints | 1.080× | 1.206× | ✓ |  |
| C | fused_fma_max_reduce_non_fused_fp32_instructions | 1.039× | 1.227× | ✓ |  |
| D | vectorized_loads_contiguous_pred_access_hints | 1.039× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | small_block_rows_... | Δ | vectorized_loads_... | Δ | fused_fma_max_red... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 1160000 ns | 1040000 ns | -120000 ns (-10.3%) | 961470 ns | -198530 ns (-17.1%) | 945440 ns | -214560 ns (-18.5%) | 1160000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 51.98 % | 40.80 % | -11.18 pp | 45.37 % | -6.61 pp | 45.14 % | -6.84 pp | 56.78 % | +4.80 pp |
| Memory Throughput (%) | 77.26 % | 84.17 % | +6.91 pp | 91.02 % | +13.76 pp | 92.50 % | +15.24 pp | 77.09 % | -0.17 pp |
| DRAM Throughput (%) | 75.54 % | 84.17 % | +8.63 pp | 91.02 % | +15.48 pp | 92.50 % | +16.96 pp | 75.67 % | +0.13 pp |
| Achieved Occupancy (%) | 98.54 % | 99.25 % | +0.71 pp | 98.05 % | -0.49 pp | 74.01 % | -24.53 pp | 98.49 % | -0.05 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 30.00 register/thread | +5 | 40.00 register/thread | +15 | 25.00 register/thread | +0 |
| Waves/SM | 31.03 | 15.52 | -15.51 | 15.52 | -15.51 | 41.37 | +10.34 | 31.03 | +0.00 |
| No Eligible (%) | 48.51 % | 58.46 % | +9.95 pp | 55.48 % | +6.97 pp | 55.80 % | +7.29 pp | 43.02 % | -5.49 pp |
| IPC Active | 2.05 inst/cycle | 1.65 inst/cycle | -0.40 | 1.79 inst/cycle | -0.26 | 1.77 inst/cycle | -0.28 | 2.28 inst/cycle | +0.23 |


### Key Learnings

#### ✅ Plan B: `vectorized_loads_max_contiguous_multiple_of_hints` (1.206×)

Adding tl.max_contiguous and tl.multiple_of alignment hints to the inner-loop column index in a Triton kernel produced a marginal 1.2065x speedup, but the profiling metrics before and after are nearly identical: DRAM throughput moved from 90.87% to 91.02%, L1TEX hit rate from 46.31% to 46.07%, bytes-per-sector utilization remained at 4.0/32, and warp cycles per issued instruction barely changed (35.28 → 35.21). The kernel was already fully DRAM-saturated (>90% DRAM throughput) before the change, meaning the vectorization hints did not materially reduce transaction count or stall cycles — the speedup is within measurement noise and the MemoryCacheAccessPattern bottleneck remains unresolved.

> **Applicable when**: Kernel is NOT yet DRAM-saturated (<85% DRAM throughput); MemoryCacheAccessPattern rule fires with low bytes-per-sector utilization on global loads; the inner-loop index is provably aligned (loop step equals BLOCK_N, starts at 0); stride_n == 1 (row-major contiguous tensor). In that regime, the hints can enable 128-bit vectorized loads and reduce transaction count.

> **Avoid when**: Kernel is already DRAM-saturated (>90% DRAM throughput) — the hints will not reduce bandwidth consumption further and any speedup will be within noise. Also ineffective when the MemoryCacheAccessPattern bottleneck is on global stores (not loads), or when stride_n != 1 (non-contiguous memory layout prevents vectorization regardless of hints).

---

## Cumulative Progress

```
Baseline:  1160000 ns  (1.000×)

Final best: 1160000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,16 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
+        triton.Config({'BLOCK_N': 1024}, num_warps=4, num_stages=2),
         triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=2),
         triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_N': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_N': 512},  num_warps=8, num_stages=2),
         triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
-        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
     ],
     key=['M', 'N'],
 )
@@ -29,9 +31,10 @@

 
     for col_off in range(0, N, BLOCK_N):
         cols = col_off + tl.arange(0, BLOCK_N)
-        mask = cols < N
-        pred = tl.load(pred_ptr + row_id * pred_stride_m + cols * pred_stride_n, mask=mask, other=0.0)
-        tgt = tl.load(target_ptr + cols, mask=mask, other=0.0)
+        col_mask = cols < N
+        pred_ptrs = pred_ptr + row_id * pred_stride_m + cols * pred_stride_n
+        pred = tl.load(pred_ptrs, mask=col_mask, other=0.0)
+        tgt = tl.load(target_ptr + cols, mask=col_mask, other=0.0)
         val = 1.0 - pred * tgt
         val = tl.where(val > 0.0, val, 0.0)
         acc += tl.sum(val, axis=0)
@@ -49,6 +52,7 @@

     key=['M'],
 )
 @triton.jit
+
 def hinge_loss_reduce_kernel(
     partial_ptr, out_ptr,
     M, total_elements,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
