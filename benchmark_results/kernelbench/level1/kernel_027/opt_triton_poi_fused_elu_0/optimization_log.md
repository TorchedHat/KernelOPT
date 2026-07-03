# Kernel Optimization Log

## Summary

- **Kernel**: `selu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3010000 ns — **Best latency**: 3420000 ns (**0.8801×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `selu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_027/synthesis/kernel_0_triton_poi_fused_elu_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 29
- **Duration**: 3010000 ns
- **SM Throughput**: 67.63 % | **Memory**: 86.94 % | **DRAM**: 86.94 %
- **Achieved Occupancy**: 84.58 %
- **IPC Active**: 2.68 inst/cycle | **No Eligible**: 33.30 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_and_add_num_stages_for_memory_latency_hiding | 1.001× | 0.898× | ✓ |  |
| B | increase_block_size_and_vectorized_loads_for_hbm_bandwidth | 1.018× | 0.880× | ✓ | **✓ best** |
| C | vectorized_loads_with_software_pipelining | 0.976× | 0.915× | ✓ |  |
| D | add_num_stages_software_pipelining_to_hide_l1tex_latency | 1.002× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | increase_block_si... | Δ | vectorized_loads_... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3350000 ns | +340000 ns (+11.3%) | 3420000 ns | +410000 ns (+13.6%) | 3290000 ns | +280000 ns (+9.3%) | 3010000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 67.63 % | 58.50 % | -9.13 pp | 69.20 % | +1.57 pp | 67.19 % | -0.44 pp | 66.76 % | -0.87 pp |
| Memory Throughput (%) | 86.94 % | 78.08 % | -8.86 pp | 76.47 % | -10.47 pp | 79.64 % | -7.30 pp | 86.82 % | -0.12 pp |
| DRAM Throughput (%) | 86.94 % | 78.08 % | -8.86 pp | 76.47 % | -10.47 pp | 79.64 % | -7.30 pp | 86.82 % | -0.12 pp |
| Achieved Occupancy (%) | 84.58 % | 42.22 % | -42.36 pp | 83.91 % | -0.67 pp | 82.09 % | -2.49 pp | 84.49 % | -0.09 pp |
| Registers/Thread | 29.00 register/thread | 63.00 register/thread | +34 | 25.00 register/thread | -4 | 23.00 register/thread | -6 | 29.00 register/thread | +0 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 1489.45 | +744.72 | 1489.45 | +744.72 | 744.73 | +0.00 |
| No Eligible (%) | 33.30 % | 41.60 % | +8.30 pp | 31.92 % | -1.38 pp | 34.14 % | +0.84 pp | 33.19 % | -0.11 pp |
| IPC Active | 2.68 inst/cycle | 2.34 inst/cycle | -0.34 | 2.67 inst/cycle | -0.01 | 2.64 inst/cycle | -0.04 | 2.68 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | streaming_cache_hints_and_larger_block_autotune | 0.976× | 0.990× | ✓ |  |
| B | fma_fusion_and_store_eviction_policy_for_memory_bound_selu | 0.980× | 0.918× | ✓ |  |
| C | reduce_register_pressure_via_num_warps_and_exp_simplification | 1.006× | 0.974× | ✓ |  |
| D | fix_load_eviction_policy_and_autotune_configs_for_streaming_kernel | 0.952× | 0.984× | ✓ |  |


### Metric Comparison

| Metric | Baseline | streaming_cache_h... | Δ | fma_fusion_and_st... | Δ | reduce_register_p... | Δ | fix_load_eviction... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3040000 ns | +30000 ns (+1.0%) | 3280000 ns | +270000 ns (+9.0%) | 3090000 ns | +80000 ns (+2.7%) | 3060000 ns | +50000 ns (+1.7%) |
| SM Throughput (%) | 67.63 % | 68.34 % | +0.71 pp | 69.70 % | +2.07 pp | 23.50 % | -44.13 pp | 66.90 % | -0.73 pp |
| Memory Throughput (%) | 86.94 % | 86.02 % | -0.92 pp | 79.63 % | -7.31 pp | 84.77 % | -2.17 pp | 85.54 % | -1.40 pp |
| DRAM Throughput (%) | 86.94 % | 86.02 % | -0.92 pp | 79.63 % | -7.31 pp | 84.77 % | -2.17 pp | 85.54 % | -1.40 pp |
| Achieved Occupancy (%) | 84.58 % | 84.17 % | -0.41 pp | 82.55 % | -2.03 pp | 39.34 % | -45.24 pp | 89.74 % | +5.16 pp |
| Registers/Thread | 29.00 register/thread | 29.00 register/thread | +0 | 25.00 register/thread | -4 | 52.00 register/thread | +23 | 32.00 register/thread | +3 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 1489.45 | +744.72 | 372.36 | -372.37 | 372.36 | -372.37 |
| No Eligible (%) | 33.30 % | 31.28 % | -2.02 pp | 30.52 % | -2.78 pp | 76.19 % | +42.89 pp | 33.76 % | +0.46 pp |
| IPC Active | 2.68 inst/cycle | 2.75 inst/cycle | +0.07 | 2.78 inst/cycle | +0.10 | 0.93 inst/cycle | -1.75 | 2.65 inst/cycle | -0.03 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_constexpr_constant_folding | 0.976× | 0.993× | ✓ |  |
| B | constexpr_constant_folding_to_eliminate_non_fused_fp32_multiplications | 0.953× | 0.984× | ✓ |  |
| C | fma_fusion_via_expm1_intrinsic_and_constant_folding | 0.993× | 0.635× | ✓ |  |
| D | restore_l1_caching_and_fma_fusion_for_memory_latency_reduction | 0.992× | 0.389× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_co... | Δ | constexpr_constan... | Δ | fma_fusion_via_ex... | Δ | restore_l1_cachin... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3010000 ns | 3030000 ns | +20000 ns (+0.7%) | 3060000 ns | +50000 ns (+1.7%) | 4740000 ns | +1730000 ns (+57.5%) | 7730000 ns | +4720000 ns (+156.8%) |
| SM Throughput (%) | 67.63 % | 68.17 % | +0.54 pp | 66.80 % | -0.83 pp | 42.50 % | -25.13 pp | 42.82 % | -24.81 pp |
| Memory Throughput (%) | 86.94 % | 86.30 % | -0.64 pp | 85.60 % | -1.34 pp | 55.25 % | -31.69 pp | 33.85 % | -53.09 pp |
| DRAM Throughput (%) | 86.94 % | 86.30 % | -0.64 pp | 85.60 % | -1.34 pp | 55.25 % | -31.69 pp | 33.85 % | -53.09 pp |
| Achieved Occupancy (%) | 84.58 % | 84.12 % | -0.46 pp | 89.98 % | +5.40 pp | 22.52 % | -62.06 pp | 48.29 % | -36.29 pp |
| Registers/Thread | 29.00 register/thread | 29.00 register/thread | +0 | 32.00 register/thread | +3 | 74.00 register/thread | +45 | 16.00 register/thread | -13 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 372.36 | -372.37 | 744.73 | +0.00 | 5957.82 | +5213.09 |
| No Eligible (%) | 33.30 % | 31.10 % | -2.20 pp | 33.82 % | +0.52 pp | 55.59 % | +22.29 pp | 57.15 % | +23.85 pp |
| IPC Active | 2.68 inst/cycle | 2.75 inst/cycle | +0.07 | 2.66 inst/cycle | -0.02 | 1.72 inst/cycle | -0.96 | 1.71 inst/cycle | -0.97 |

---

## Cumulative Progress

```
Baseline:  3010000 ns  (1.000×)

Final best: 3420000 ns  (0.8801×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,35 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
+        # num_stages=1 (baseline, no pipelining)
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=1),
+        # num_stages=2 (1 prefetch buffer ahead)
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
+        # num_stages=4 (3 prefetch buffers ahead)
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
+        # large block configs for streaming workloads
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=1),
     ],
     key=['numel'],
 )
@@ -30,7 +51,7 @@

     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < numel
 
-    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
+    x = tl.load(in_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
 
     # SELU: scale * (x if x > 0 else alpha * (exp(x) - 1))
     is_positive = x > 0.0
@@ -39,8 +60,7 @@

     neg_val = SCALE * ALPHA * libdevice.expm1(x)
     result = tl.where(is_positive, pos_val, neg_val)
 
-    tl.store(out_ptr + offsets, result, mask=mask)
-
+    tl.store(out_ptr + offsets, result, mask=mask, cache_modifier='.cs')
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
