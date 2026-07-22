# Kernel Optimization Log

## Summary

- **Kernel**: `gelu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 125380 ns — **Best latency**: 138820 ns (**0.9032×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `gelu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_088/synthesis/kernel_0_triton_poi_fused_add_mul_pow_tanh_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (65536,1,1), Waves/SM: 31.03
- **Registers/thread**: 25
- **Duration**: 125380 ns
- **SM Throughput**: 76.22 % | **Memory**: 83.23 % | **DRAM**: 83.23 %
- **Achieved Occupancy**: 88.33 %
- **IPC Active**: 3.11 inst/cycle | **No Eligible**: 20.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_block_size_reduce_divergence_and_memory_stalls | 0.999× | 1.007× | ✓ |  |
| B | add_software_pipelining_and_fma_arithmetic | 0.986× | 0.981× | ✓ |  |
| C | vectorized_loads_stores_with_fp16_precision | 0.983× | 0.893× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_block_si... | Δ | add_software_pipe... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 125380 ns | 124510 ns | -870 ns (-0.7%) | 127810 ns | +2430 ns (+1.9%) | 140420 ns | +15040 ns (+12.0%) |
| SM Throughput (%) | 76.22 % | 75.94 % | -0.28 pp | 76.26 % | +0.04 pp | 67.31 % | -8.91 pp |
| Memory Throughput (%) | 83.23 % | 83.79 % | +0.56 pp | 81.41 % | -1.82 pp | 74.33 % | -8.90 pp |
| DRAM Throughput (%) | 83.23 % | 83.79 % | +0.56 pp | 81.41 % | -1.82 pp | 74.33 % | -8.90 pp |
| Achieved Occupancy (%) | 88.33 % | 88.18 % | -0.15 pp | 88.01 % | -0.32 pp | 34.94 % | -53.39 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 26.00 register/thread | +1 | 76.00 register/thread | +51 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 20.69 | -10.34 |
| No Eligible (%) | 20.61 % | 21.28 % | +0.67 pp | 21.67 % | +1.06 pp | 31.72 % | +11.11 pp |
| IPC Active | 3.11 inst/cycle | 3.13 inst/cycle | +0.02 | 3.07 inst/cycle | -0.04 | 2.76 inst/cycle | -0.35 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | expand_autotune_configs_with_larger_tiles_and_more_stages | 1.001× | 1.000× | ✓ | **✓ best** |
| B | remove_evict_first_and_expand_autotune_configs_with_larger_tiles | 0.989× | 1.014× | ✓ |  |
| C | reduce_register_pressure_via_num_stages_and_num_warps_autotune | 0.983× | 0.949× | ✓ |  |


### Metric Comparison

| Metric | Baseline | expand_autotune_c... | Δ | remove_evict_firs... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 125380 ns | 125380 ns | +0 ns (+0.0%) | 123680 ns | -1700 ns (-1.4%) | 132160 ns | +6780 ns (+5.4%) |
| SM Throughput (%) | 76.22 % | 77.79 % | +1.57 pp | 79.92 % | +3.70 pp | 73.09 % | -3.13 pp |
| Memory Throughput (%) | 83.23 % | 83.25 % | +0.02 pp | 82.92 % | -0.31 pp | 78.98 % | -4.25 pp |
| DRAM Throughput (%) | 83.23 % | 83.25 % | +0.02 pp | 82.92 % | -0.31 pp | 78.98 % | -4.25 pp |
| Achieved Occupancy (%) | 88.33 % | 88.21 % | -0.12 pp | 87.96 % | -0.37 pp | 40.96 % | -47.37 pp |
| Registers/Thread | 25.00 register/thread | 25.00 register/thread | +0 | 25.00 register/thread | +0 | 72.00 register/thread | +47 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 17.73 | -13.30 |
| No Eligible (%) | 20.61 % | 20.75 % | +0.14 pp | 19.18 % | -1.43 pp | 25.60 % | +4.99 pp |
| IPC Active | 3.11 inst/cycle | 3.16 inst/cycle | +0.05 | 3.27 inst/cycle | +0.16 | 2.96 inst/cycle | -0.15 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_arithmetic_restructure_and_evict_first_store | 1.001× | 0.988× | ✓ |  |
| B | eliminate_boundary_divergence_with_aligned_grid_and_l1_cache_hints | 0.968× | 0.907× | ✓ |  |
| C | reduce_register_spill_via_smaller_block_configs_targeting_local_memory_coalescing | 1.001× | 0.764× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fma_arithmetic_re... | Δ | eliminate_boundar... | Δ | reduce_register_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 125380 ns | 126850 ns | +1470 ns (+1.2%) | 138300 ns | +12920 ns (+10.3%) | 164000 ns | +38620 ns (+30.8%) |
| SM Throughput (%) | 76.22 % | 77.59 % | +1.37 pp | 65.44 % | -10.78 pp | 59.81 % | -16.41 pp |
| Memory Throughput (%) | 83.23 % | 82.13 % | -1.10 pp | 75.24 % | -7.99 pp | 64.22 % | -19.01 pp |
| DRAM Throughput (%) | 83.23 % | 82.13 % | -1.10 pp | 75.24 % | -7.99 pp | 64.22 % | -19.01 pp |
| Achieved Occupancy (%) | 88.33 % | 88.21 % | -0.12 pp | 34.87 % | -53.46 pp | 48.87 % | -39.46 pp |
| Registers/Thread | 25.00 register/thread | 26.00 register/thread | +1 | 78.00 register/thread | +53 | 17.00 register/thread | -8 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 20.69 | -10.34 | 62.06 | +31.03 |
| No Eligible (%) | 20.61 % | 21.18 % | +0.57 pp | 32.78 % | +12.17 pp | 38.50 % | +17.89 pp |
| IPC Active | 3.11 inst/cycle | 3.21 inst/cycle | +0.10 | 2.71 inst/cycle | -0.40 | 2.41 inst/cycle | -0.70 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | explicit_fma_arithmetic_and_eliminate_boundary_divergence_via_aligned_grid | 1.003× | 1.008× | ✓ | **✓ best** |
| B | replace_tanh_gelu_with_sigmoid_approximation_to_reduce_register_pressure | 0.973× | 0.945× | ✓ |  |
| C | fix_fp32_input_pointer_type_and_prune_small_block_autotune_configs | 1.000× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | explicit_fma_arit... | Δ | replace_tanh_gelu... | Δ | fix_fp32_input_po... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 125380 ns | 124350 ns | -1030 ns (-0.8%) | 132670 ns | +7290 ns (+5.8%) | 125790 ns | +410 ns (+0.3%) |
| SM Throughput (%) | 76.22 % | 73.16 % | -3.06 pp | 57.08 % | -19.14 pp | 79.50 % | +3.28 pp |
| Memory Throughput (%) | 83.23 % | 83.96 % | +0.73 pp | 78.50 % | -4.73 pp | 83.05 % | -0.18 pp |
| DRAM Throughput (%) | 83.23 % | 83.96 % | +0.73 pp | 78.50 % | -4.73 pp | 83.05 % | -0.18 pp |
| Achieved Occupancy (%) | 88.33 % | 88.04 % | -0.29 pp | 23.20 % | -65.13 pp | 87.72 % | -0.61 pp |
| Registers/Thread | 25.00 register/thread | 28.00 register/thread | +3 | 108.00 register/thread | +83 | 25.00 register/thread | +0 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 15.52 | -15.51 | 31.03 | +0.00 |
| No Eligible (%) | 20.61 % | 25.72 % | +5.11 pp | 40.11 % | +19.50 pp | 18.37 % | -2.24 pp |
| IPC Active | 3.11 inst/cycle | 2.94 inst/cycle | -0.17 | 2.37 inst/cycle | -0.74 | 3.29 inst/cycle | +0.18 |

---

## Iteration 5

**Best so far**: 1.0083× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_libdevice_tanh_with_erf_gelu_to_eliminate_transcendental_divergence | 1.001× | 0.920× | ✓ |  |
| B | replace_tanh_gelu_with_erf_gelu_to_eliminate_thread_divergence_and_increase_fma_density | 1.004× | 0.903× | ✓ | **✓ best** |
| C | replace_manual_exp_tanh_with_libdevice_tanh_to_cut_register_pressure | 0.965× | 0.652× | ✓ |  |
| D | add_triton_autotune | 0.982× | 0.386× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ | replace_tanh_gelu... | Δ | replace_manual_ex... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 125380 ns | 136260 ns | +10880 ns (+8.7%) | 138820 ns | +13440 ns (+10.7%) | 192320 ns | +66940 ns (+53.4%) | 324700 ns | +199320 ns (+159.0%) |
| SM Throughput (%) | 76.22 % | 72.03 % | -4.19 pp | 76.83 % | +0.61 pp | 44.89 % | -31.33 pp | 37.64 % | -38.58 pp |
| Memory Throughput (%) | 83.23 % | 76.89 % | -6.34 pp | 75.54 % | -7.69 pp | 54.07 % | -29.16 pp | 32.50 % | -50.73 pp |
| DRAM Throughput (%) | 83.23 % | 76.89 % | -6.34 pp | 75.54 % | -7.69 pp | 54.07 % | -29.16 pp | 32.50 % | -50.73 pp |
| Achieved Occupancy (%) | 88.33 % | 87.50 % | -0.83 pp | 87.81 % | -0.52 pp | 17.99 % | -70.34 pp | 49.92 % | -38.41 pp |
| Registers/Thread | 25.00 register/thread | 31.00 register/thread | +6 | 31.00 register/thread | +6 | 149.00 register/thread | +124 | 16.00 register/thread | -9 |
| Waves/SM | 31.03 | 31.03 | +0.00 | 31.03 | +0.00 | 20.69 | -10.34 | 248.24 | +217.21 |
| No Eligible (%) | 20.61 % | 27.02 % | +6.41 pp | 22.75 % | +2.14 pp | 54.18 % | +33.57 pp | 62.18 % | +41.57 pp |
| IPC Active | 3.11 inst/cycle | 2.91 inst/cycle | -0.20 | 3.02 inst/cycle | -0.09 | 1.82 inst/cycle | -1.29 | 1.51 inst/cycle | -1.60 |

---

## Cumulative Progress

```
Baseline:  125380 ns  (1.000×)
Iter 4     124350 ns  (1.008×)  — explicit_fma_arithmetic_and_eliminate_boundary_divergence_via_aligned_grid

Final best: 138820 ns  (0.9032×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,29 @@

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
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
     ],
     key=['n_elements'],
 )
@@ -28,16 +43,10 @@

 
     x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
 
-    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
-    x_half = x * 0.5
-    x_cube = x * x * x
-    inner = x + 0.044715 * x_cube
-    inner = inner * 0.7978845608028654  # sqrt(2/pi)
-    tanh_val = libdevice.tanh(inner)
-    result = x_half * (1.0 + tanh_val)
+    # GELU: x * 0.5 * (1 + erf(x / sqrt(2))) — exact erf-based formulation
+    result = x * 0.5 * (1.0 + libdevice.erf(x * 0.7071067811865476))
 
-    tl.store(out_ptr + offsets, result, mask=mask)
-
+    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)  # downcast + 128-bit store
 
 def kernel_function(x):
     output = torch.empty_like(x)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
