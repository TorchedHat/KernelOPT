# Kernel Optimization Log

## Summary

- **Kernel**: `fused_linear_sum_logsumexp_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 12450 ns — **Best latency**: 12450 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `fused_linear_sum_logsumexp_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_018/synthesis/kernel_0_fused_fused_addmm.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1024,1,1), Waves/SM: 0.48
- **Registers/thread**: 29
- **Duration**: 12450 ns
- **SM Throughput**: 21.77 % | **Memory**: 56.10 % | **DRAM**: 56.10 %
- **Achieved Occupancy**: 45.91 %
- **IPC Active**: 1.07 inst/cycle | **No Eligible**: 72.46 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads | 0.987× | 0.980× | ✓ |  |
| B | vectorized_loads_and_coalesced_stores_via_block_m_expansion | 0.985× | 0.942× | ✓ |  |
| C | memory_coalescing_increase_block_m | 0.763× | 0.184× | ✓ |  |
| D | increase_BLOCK_M_for_better_parallelism_and_cache_reuse | 0.965× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | increase_BLOCK_M_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12450 ns | 12700 ns | +250 ns (+2.0%) | 13220 ns | +770 ns (+6.2%) | 67740 ns | +55290 ns (+444.1%) | 12450 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 21.77 % | 26.64 % | +4.87 pp | 21.23 % | -0.54 pp | 4.20 % | -17.57 pp | 21.41 % | -0.36 pp |
| Memory Throughput (%) | 56.10 % | 54.86 % | -1.24 pp | 52.75 % | -3.35 pp | 10.28 % | -45.82 pp | 56.06 % | -0.04 pp |
| DRAM Throughput (%) | 56.10 % | 54.86 % | -1.24 pp | 52.75 % | -3.35 pp | 10.28 % | -45.82 pp | 56.06 % | -0.04 pp |
| Achieved Occupancy (%) | 45.91 % | 45.99 % | +0.08 pp | 46.27 % | +0.36 pp | 6.25 % | -39.66 pp | 45.96 % | +0.05 pp |
| Registers/Thread | 29.00 register/thread | 29.00 register/thread | +0 | 29.00 register/thread | +0 | 255.00 register/thread | +226 | 29.00 register/thread | +0 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.06 | -0.42 | 0.48 | +0.00 |
| No Eligible (%) | 72.46 % | 66.41 % | -6.05 pp | 72.08 % | -0.38 pp | 72.58 % | +0.12 pp | 72.02 % | -0.44 pp |
| IPC Active | 1.07 inst/cycle | 1.35 inst/cycle | +0.28 | 1.11 inst/cycle | +0.04 | 1.10 inst/cycle | +0.03 | 1.10 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_num_stages_and_evict_first_streaming_loads | 0.987× | 0.968× | ✓ |  |
| B | increase_num_stages_for_memory_latency_hiding | 0.976× | 0.990× | ✓ |  |
| C | increase_num_warps_to_hide_l1tex_latency | 0.966× | 0.995× | ✓ |  |
| D | reduce_register_pressure_via_maxnreg_cap | 0.832× | 0.055× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_num_stag... | Δ | increase_num_stag... | Δ | increase_num_warp... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 12450 ns | 12860 ns | +410 ns (+3.3%) | 12580 ns | +130 ns (+1.0%) | 12510 ns | +60 ns (+0.5%) | 225600 ns | +213150 ns (+1712.0%) |
| SM Throughput (%) | 21.77 % | 23.90 % | +2.13 pp | 26.30 % | +4.53 pp | 21.78 % | +0.01 pp | 2.85 % | -18.92 pp |
| Memory Throughput (%) | 56.10 % | 54.14 % | -1.96 pp | 55.35 % | -0.75 pp | 55.70 % | -0.40 pp | 4.24 % | -51.86 pp |
| DRAM Throughput (%) | 56.10 % | 54.14 % | -1.96 pp | 55.35 % | -0.75 pp | 55.70 % | -0.40 pp | 3.14 % | -52.96 pp |
| Achieved Occupancy (%) | 45.91 % | 46.54 % | +0.63 pp | 45.64 % | -0.27 pp | 45.89 % | -0.02 pp | 6.25 % | -39.66 pp |
| Registers/Thread | 29.00 register/thread | 29.00 register/thread | +0 | 29.00 register/thread | +0 | 29.00 register/thread | +0 | 64.00 register/thread | +35 |
| Waves/SM | 0.48 | 0.48 | +0.00 | 0.48 | +0.00 | 0.48 | +0.00 | 0.02 | -0.46 |
| No Eligible (%) | 72.46 % | 69.75 % | -2.71 pp | 66.13 % | -6.33 pp | 72.13 % | -0.33 pp | 89.28 % | +16.82 pp |
| IPC Active | 1.07 inst/cycle | 1.16 inst/cycle | +0.09 | 1.33 inst/cycle | +0.26 | 1.11 inst/cycle | +0.04 | 0.43 inst/cycle | -0.64 |

---

## Cumulative Progress

```
Baseline:  12450 ns  (1.000×)

Final best: 12450 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,20 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 1024}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048}, num_warps=8, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 512}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 512}, num_warps=4, num_stages=2),
-        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 512}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 1,   'BLOCK_K': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 4,   'BLOCK_K': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 16,  'BLOCK_K': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 16,  'BLOCK_K': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 512},  num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 1024}, num_warps=8, num_stages=2),
     ],
     key=['M', 'N', 'K'],
 )
@@ -70,7 +76,6 @@

     # So output = acc
 
     tl.store(OUT_ptr + rows, acc, mask=row_mask)
-
 
 def kernel_function(p_linear_weight, p_linear_bias, x):
     M, K = x.shape
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
