# Kernel Optimization Log

## Summary

- **Kernel**: `gelu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3150000 ns — **Best latency**: 3090000 ns (**1.0194×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`evict_last_load_policy_with_reduced_warp_count_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gelu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_026/synthesis/kernel_0_triton_poi_fused_gelu_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 32
- **Duration**: 3150000 ns
- **SM Throughput**: 73.01 % | **Memory**: 82.97 % | **DRAM**: 82.97 %
- **Achieved Occupancy**: 87.96 %
- **IPC Active**: 2.89 inst/cycle | **No Eligible**: 28.18 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_software_pipelining_and_expand_autotune_configs | 0.998× | 1.016× | ✓ |  |
| B | add_num_stages_pipelining_and_larger_blocks_to_autotune | 0.990× | 0.994× | ✓ |  |
| C | vectorized_loads_stores_with_larger_block_sizes | 0.956× | 0.981× | ✓ |  |
| D | add_num_stages_and_cache_eviction_hints_for_memory_bound_gelu | 1.029× | 1.010× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_num_stages_so... | Δ | add_num_stages_pi... | Δ | vectorized_loads_... | Δ | add_num_stages_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3150000 ns | 3100000 ns | -50000 ns (-1.6%) | 3170000 ns | +20000 ns (+0.6%) | 3210000 ns | +60000 ns (+1.9%) | 3120000 ns | -30000 ns (-1.0%) |
| SM Throughput (%) | 73.01 % | 71.73 % | -1.28 pp | 72.55 % | -0.46 pp | 73.68 % | +0.67 pp | 74.94 % | +1.93 pp |
| Memory Throughput (%) | 82.97 % | 84.37 % | +1.40 pp | 82.61 % | -0.36 pp | 81.63 % | -1.34 pp | 83.89 % | +0.92 pp |
| DRAM Throughput (%) | 82.97 % | 84.37 % | +1.40 pp | 82.61 % | -0.36 pp | 81.63 % | -1.34 pp | 83.89 % | +0.92 pp |
| Achieved Occupancy (%) | 87.96 % | 63.70 % | -24.26 pp | 87.69 % | -0.27 pp | 87.78 % | -0.18 pp | 87.72 % | -0.24 pp |
| Registers/Thread | 32.00 register/thread | 38.00 register/thread | +6 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 744.73 | 496.48 | -248.25 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 28.18 % | 27.94 % | -0.24 pp | 27.44 % | -0.74 pp | 26.00 % | -2.18 pp | 25.08 % | -3.10 pp |
| IPC Active | 2.89 inst/cycle | 2.87 inst/cycle | -0.02 | 2.86 inst/cycle | -0.03 | 2.98 inst/cycle | +0.09 | 3.04 inst/cycle | +0.15 |

---

## Iteration 2

**Best so far**: 1.0096× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_register_pressure_via_tanh_gelu_approximation | 1.016× | 1.029× | ✓ |  |
| B | fma_fusion_and_streaming_eviction_policy_fix | 0.985× | 0.997× | ✓ |  |
| C | bf16_output_store_with_fma_arithmetic_restructure | 1.004× | 1.003× | ✓ |  |
| D | larger_block_sizes_to_reduce_cta_count_and_l2_thrashing | 0.956× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_register_p... | Δ | fma_fusion_and_st... | Δ | bf16_output_store... | Δ | larger_block_size... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3150000 ns | 3060000 ns | -90000 ns (-2.9%) | 3160000 ns | +10000 ns (+0.3%) | 3140000 ns | -10000 ns (-0.3%) | 3210000 ns | +60000 ns (+1.9%) |
| SM Throughput (%) | 73.01 % | 71.83 % | -1.18 pp | 72.48 % | -0.53 pp | 70.48 % | -2.53 pp | 74.67 % | +1.66 pp |
| Memory Throughput (%) | 82.97 % | 85.42 % | +2.45 pp | 82.71 % | -0.26 pp | 83.23 % | +0.26 pp | 81.57 % | -1.40 pp |
| DRAM Throughput (%) | 82.97 % | 85.42 % | +2.45 pp | 82.71 % | -0.26 pp | 83.23 % | +0.26 pp | 81.57 % | -1.40 pp |
| Achieved Occupancy (%) | 87.96 % | 81.06 % | -6.90 pp | 87.12 % | -0.84 pp | 87.44 % | -0.52 pp | 87.41 % | -0.55 pp |
| Registers/Thread | 32.00 register/thread | 28.00 register/thread | -4 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 28.18 % | 28.90 % | +0.72 pp | 27.52 % | -0.66 pp | 28.85 % | +0.67 pp | 26.38 % | -1.80 pp |
| IPC Active | 2.89 inst/cycle | 2.83 inst/cycle | -0.06 | 2.94 inst/cycle | +0.05 | 2.85 inst/cycle | -0.04 | 2.96 inst/cycle | +0.07 |

---

## Iteration 3

**Best so far**: 1.0096× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_thread_divergence_via_mask_free_tiling_and_fma_restructure | 1.015× | 0.981× | ✓ |  |
| B | reduce_active_warps_for_l2_cache_coherence_and_data_locality | 1.014× | 1.006× | ✓ |  |
| C | evict_last_load_policy_with_reduced_warp_count_configs | 1.037× | 1.019× | ✓ | **✓ best** |
| D | fma_gelu_arithmetic_restructure_for_fused_fp32_instructions | 0.961× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_thread_... | Δ | reduce_active_war... | Δ | evict_last_load_p... | Δ | fma_gelu_arithmet... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3150000 ns | 3210000 ns | +60000 ns (+1.9%) | 3130000 ns | -20000 ns (-0.6%) | 3090000 ns | -60000 ns (-1.9%) | 3180000 ns | +30000 ns (+1.0%) |
| SM Throughput (%) | 73.01 % | 62.02 % | -10.99 pp | 70.30 % | -2.71 pp | 73.74 % | +0.73 pp | 72.50 % | -0.51 pp |
| Memory Throughput (%) | 82.97 % | 81.57 % | -1.40 pp | 83.58 % | +0.61 pp | 84.76 % | +1.79 pp | 82.35 % | -0.62 pp |
| DRAM Throughput (%) | 82.97 % | 81.57 % | -1.40 pp | 83.58 % | +0.61 pp | 84.76 % | +1.79 pp | 82.35 % | -0.62 pp |
| Achieved Occupancy (%) | 87.96 % | 42.73 % | -45.23 pp | 87.03 % | -0.93 pp | 86.92 % | -1.04 pp | 87.16 % | -0.80 pp |
| Registers/Thread | 32.00 register/thread | 51.00 register/thread | +19 | 32.00 register/thread | +0 | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 28.18 % | 38.41 % | +10.23 pp | 29.53 % | +1.35 pp | 25.92 % | -2.26 pp | 27.42 % | -0.76 pp |
| IPC Active | 2.89 inst/cycle | 2.46 inst/cycle | -0.43 | 2.83 inst/cycle | -0.06 | 2.96 inst/cycle | +0.07 | 2.94 inst/cycle | +0.05 |

---

## Iteration 4

**Best so far**: 1.0194× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | bypass_l1_with_cg_cache_modifier_for_streaming_gelu | 0.976× | 0.409× | ✓ |  |
| B | grouped_pid_swizzle_for_l2_cache_locality | 1.007× | 1.029× | ✓ |  |
| D | l2_cache_bypass_with_low_warp_count_for_register_pressure_relief | 1.019× | 1.043× | ✓ |  |


### Metric Comparison

| Metric | Baseline | bypass_l1_with_cg... | Δ | grouped_pid_swizz... | Δ | l2_cache_bypass_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3150000 ns | 7700000 ns | +4550000 ns (+144.4%) | 3060000 ns | -90000 ns (-2.9%) | 3020000 ns | -130000 ns (-4.1%) |
| SM Throughput (%) | 73.01 % | 30.22 % | -42.79 pp | 70.86 % | -2.15 pp | 67.41 % | -5.60 pp |
| Memory Throughput (%) | 82.97 % | 33.98 % | -48.99 pp | 85.40 % | +2.43 pp | 86.74 % | +3.77 pp |
| DRAM Throughput (%) | 82.97 % | 33.98 % | -48.99 pp | 85.40 % | +2.43 pp | 86.74 % | +3.77 pp |
| Achieved Occupancy (%) | 87.96 % | 14.33 % | -73.63 pp | 65.64 % | -22.32 pp | 68.69 % | -19.27 pp |
| Registers/Thread | 32.00 register/thread | 30.00 register/thread | -2 | 40.00 register/thread | +8 | 40.00 register/thread | +8 |
| Waves/SM | 744.73 | 2978.91 | +2234.18 | 496.48 | -248.25 | 496.48 | -248.25 |
| No Eligible (%) | 28.18 % | 69.64 % | +41.46 pp | 29.73 % | +1.55 pp | 32.63 % | +4.45 pp |
| IPC Active | 2.89 inst/cycle | 1.21 inst/cycle | -1.68 | 2.80 inst/cycle | -0.09 | 2.67 inst/cycle | -0.22 |

---

## Iteration 5

**Best so far**: 1.0194× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | mask_free_large_block_tiling_to_eliminate_thread_divergence | 0.994× | 0.677× | ✓ |  |
| B | eliminate_intermediate_register_via_inline_fma_gelu_restructure | 1.015× | 1.043× | ✓ |  |
| C | remove_cache_modifiers_and_add_large_block_autotune_configs | 1.007× | 1.016× | ✓ |  |
| D | add_triton_autotune | 1.018× | 1.043× | ✓ |  |


### Metric Comparison

| Metric | Baseline | mask_free_large_b... | Δ | eliminate_interme... | Δ | remove_cache_modi... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3150000 ns | 4650000 ns | +1500000 ns (+47.6%) | 3020000 ns | -130000 ns (-4.1%) | 3100000 ns | -50000 ns (-1.6%) | 3020000 ns | -130000 ns (-4.1%) |
| SM Throughput (%) | 73.01 % | 43.27 % | -29.74 pp | 70.94 % | -2.07 pp | 70.71 % | -2.30 pp | 67.39 % | -5.62 pp |
| Memory Throughput (%) | 82.97 % | 56.31 % | -26.66 pp | 86.64 % | +3.67 pp | 84.44 % | +1.47 pp | 86.71 % | +3.74 pp |
| DRAM Throughput (%) | 82.97 % | 56.31 % | -26.66 pp | 86.64 % | +3.67 pp | 84.44 % | +1.47 pp | 86.71 % | +3.74 pp |
| Achieved Occupancy (%) | 87.96 % | 22.87 % | -65.09 pp | 67.42 % | -20.54 pp | 62.10 % | -25.86 pp | 68.72 % | -19.24 pp |
| Registers/Thread | 32.00 register/thread | 84.00 register/thread | +52 | 40.00 register/thread | +8 | 40.00 register/thread | +8 | 40.00 register/thread | +8 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 496.48 | -248.25 | 496.48 | -248.25 | 496.48 | -248.25 |
| No Eligible (%) | 28.18 % | 54.88 % | +26.70 pp | 29.00 % | +0.82 pp | 29.38 % | +1.20 pp | 33.27 % | +5.09 pp |
| IPC Active | 2.89 inst/cycle | 1.78 inst/cycle | -1.11 | 2.87 inst/cycle | -0.02 | 2.82 inst/cycle | -0.07 | 2.66 inst/cycle | -0.23 |

---

## Cumulative Progress

```
Baseline:  3150000 ns  (1.000×)
Iter 1     3120000 ns  (1.010×)  — add_num_stages_and_cache_eviction_hints_for_memory_bound_gelu
Iter 3     3090000 ns  (1.019×)  — evict_last_load_policy_with_reduced_warp_count_configs

Final best: 3090000 ns  (1.0194×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,16 +4,19 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=2,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=2,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=1),
     ],
     key=['n_elements'],
 )
@@ -29,17 +32,21 @@

     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
 
-    x = tl.load(input_ptr + offsets, mask=mask)
+    x = tl.load(input_ptr + offsets, mask=mask, other=0.0, cache_modifier='.cg')
 
-    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
-    # = 0.5 * x * (1 + erf(x * 0.7071067811865476))
-    x_half = x * 0.5
-    x_scaled = x * 0.7071067811865476
-    erf_val = libdevice.erf(x_scaled)
-    result = x_half * (1.0 + erf_val)
+    # tanh-GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
+    # FMA-restructured for higher throughput
+    x2 = x * x
+    x_cubed = x2 * x
+    # inner = sqrt(2/pi)*0.044715 * x^3 + sqrt(2/pi) * x
+    # C1 = 0.7978845608028654 * 0.044715 = 0.035676274473817765
+    inner = libdevice.fma(0.035676274473817765, x_cubed, 0.7978845608028654 * x)
+    tanh_val = libdevice.tanh(inner)
+    half_x = 0.5 * x
+    # result = 0.5*x*(1+tanh) = 0.5*x*tanh + 0.5*x = fma(half_x, tanh_val, half_x)
+    result = libdevice.fma(half_x, tanh_val, half_x)
 
-    tl.store(output_ptr + offsets, result, mask=mask)
-
+    tl.store(output_ptr + offsets, result, mask=mask, cache_modifier='.cg')
 
 def kernel_function(*inputs):
     x = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
