# Kernel Optimization Log

## Summary

- **Kernel**: `tanh_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 3020000 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `tanh_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_022/synthesis/kernel_0_triton_poi_fused_tanh_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 17
- **Duration**: 3020000 ns
- **SM Throughput**: 56.21 % | **Memory**: 86.63 % | **DRAM**: 86.63 %
- **Achieved Occupancy**: 87.64 %
- **IPC Active**: 2.24 inst/cycle | **No Eligible**: 43.89 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | 0.972× | 0.993× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 1.000× | 1.000× | ✓ |  |
| C | vectorized_loads_reduce_thread_divergence | 0.976× | 0.947× | ✓ |  |
| D | increase_block_size_and_add_num_stages_for_latency_hiding | 0.986× | 0.959× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | vectorized_loads_... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3020000 ns | +0 ns (+0.0%) | 3190000 ns | +170000 ns (+5.6%) | 3150000 ns | +130000 ns (+4.3%) |
| SM Throughput (%) | 56.21 % | 58.88 % | +2.67 pp | 56.49 % | +0.28 pp | 52.07 % | -4.14 pp | 51.34 % | -4.87 pp |
| Memory Throughput (%) | 86.63 % | 86.03 % | -0.60 pp | 86.63 % | +0.00 pp | 82.09 % | -4.54 pp | 83.15 % | -3.48 pp |
| DRAM Throughput (%) | 86.63 % | 86.03 % | -0.60 pp | 86.63 % | +0.00 pp | 82.09 % | -4.54 pp | 83.15 % | -3.48 pp |
| Achieved Occupancy (%) | 87.64 % | 87.82 % | +0.18 pp | 87.66 % | +0.02 pp | 79.70 % | -7.94 pp | 53.12 % | -34.52 pp |
| Registers/Thread | 17.00 register/thread | 17.00 register/thread | +0 | 17.00 register/thread | +0 | 28.00 register/thread | +11 | 45.00 register/thread | +28 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 372.36 | -372.37 | 297.89 | -446.84 |
| No Eligible (%) | 43.89 % | 40.83 % | -3.06 pp | 43.39 % | -0.50 pp | 48.26 % | +4.37 pp | 48.15 % | +4.26 pp |
| IPC Active | 2.24 inst/cycle | 2.39 inst/cycle | +0.15 | 2.24 inst/cycle | +0.00 | 2.07 inst/cycle | -0.17 | 2.09 inst/cycle | -0.15 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | reduce_thread_divergence_via_larger_block_and_num_stages | 1.000× | 1.000× | ✓ |  |
| B | add_num_stages_software_pipelining_and_expand_autotune_configs | 0.972× | 0.774× | ✓ |  |
| C | reduce_register_pressure_via_autotune_num_warps_expansion | 1.003× | 1.000× | ✓ | **✓ best** |
| D | increase_warp_count_and_block_size_for_latency_hiding_and_divergence_reduction | 0.982× | 0.947× | ✓ |  |


### Metric Comparison

| Metric | Baseline | reduce_thread_div... | Δ | add_num_stages_so... | Δ | reduce_register_p... | Δ | increase_warp_cou... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3020000 ns | +0 ns (+0.0%) | 3900000 ns | +880000 ns (+29.1%) | 3020000 ns | +0 ns (+0.0%) | 3190000 ns | +170000 ns (+5.6%) |
| SM Throughput (%) | 56.21 % | 55.90 % | -0.31 pp | 61.19 % | +4.98 pp | 56.71 % | +0.50 pp | 51.63 % | -4.58 pp |
| Memory Throughput (%) | 86.63 % | 86.70 % | +0.07 pp | 67.09 % | -19.54 pp | 86.77 % | +0.14 pp | 82.05 % | -4.58 pp |
| DRAM Throughput (%) | 86.63 % | 86.70 % | +0.07 pp | 67.09 % | -19.54 pp | 86.77 % | +0.14 pp | 82.05 % | -4.58 pp |
| Achieved Occupancy (%) | 87.64 % | 87.59 % | -0.05 pp | 69.83 % | -17.81 pp | 89.45 % | +1.81 pp | 79.74 % | -7.90 pp |
| Registers/Thread | 17.00 register/thread | 17.00 register/thread | +0 | 16.00 register/thread | -1 | 17.00 register/thread | +0 | 28.00 register/thread | +11 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 2978.91 | +2234.18 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 43.89 % | 44.03 % | +0.14 pp | 38.81 % | -5.08 pp | 43.23 % | -0.66 pp | 48.31 % | +4.42 pp |
| IPC Active | 2.24 inst/cycle | 2.26 inst/cycle | +0.02 | 2.47 inst/cycle | +0.23 | 2.24 inst/cycle | +0.00 | 2.07 inst/cycle | -0.17 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | prune_autotune_configs_to_force_large_block_size_eliminating_mask_divergence | 0.986× | 0.962× | ✓ |  |
| B | alignment_hints_and_pruned_autotune_for_divergence_reduction | 0.986× | 0.959× | ✓ |  |
| C | alignment_hints_tl_multiple_of_max_contiguous_for_vectorized_coalesced_loads | 0.982× | 0.947× | ✓ |  |
| D | prune_autotune_configs_and_add_alignment_hints | 0.973× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | prune_autotune_co... | Δ | alignment_hints_a... | Δ | alignment_hints_t... | Δ | prune_autotune_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3140000 ns | +120000 ns (+4.0%) | 3150000 ns | +130000 ns (+4.3%) | 3190000 ns | +170000 ns (+5.6%) | 3040000 ns | +20000 ns (+0.7%) |
| SM Throughput (%) | 56.21 % | 52.41 % | -3.80 pp | 51.26 % | -4.95 pp | 51.57 % | -4.64 pp | 59.46 % | +3.25 pp |
| Memory Throughput (%) | 86.63 % | 83.37 % | -3.26 pp | 83.13 % | -3.50 pp | 82.09 % | -4.54 pp | 86.01 % | -0.62 pp |
| DRAM Throughput (%) | 86.63 % | 83.37 % | -3.26 pp | 83.13 % | -3.50 pp | 82.09 % | -4.54 pp | 86.01 % | -0.62 pp |
| Achieved Occupancy (%) | 87.64 % | 80.17 % | -7.47 pp | 53.01 % | -34.63 pp | 79.77 % | -7.87 pp | 87.84 % | +0.20 pp |
| Registers/Thread | 17.00 register/thread | 28.00 register/thread | +11 | 45.00 register/thread | +28 | 28.00 register/thread | +11 | 17.00 register/thread | +0 |
| Waves/SM | 744.73 | 372.36 | -372.37 | 297.89 | -446.84 | 372.36 | -372.37 | 744.73 | +0.00 |
| No Eligible (%) | 43.89 % | 46.47 % | +2.58 pp | 48.07 % | +4.18 pp | 48.14 % | +4.25 pp | 40.95 % | -2.94 pp |
| IPC Active | 2.24 inst/cycle | 2.11 inst/cycle | -0.13 | 2.06 inst/cycle | -0.18 | 2.08 inst/cycle | -0.16 | 2.36 inst/cycle | +0.12 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_libdevice_tanh_with_fast_polynomial_approximation | 0.973× | 0.993× | ✓ |  |
| B | replace_libdevice_tanh_with_hardware_approx_tanh_to_reduce_nonfused_fp32_instructions | 0.986× | 0.962× | ✓ |  |
| C | replace_libdevice_tanh_with_fused_fp32_approximation_to_reduce_register_pressure | 0.979× | 0.971× | ✓ |  |
| D | prune_autotune_to_large_blocks_and_switch_load_to_l1_cache_for_l2_hit_rate | 0.976× | 0.944× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_libdevice... | Δ | replace_libdevice... | Δ | replace_libdevice... | Δ | prune_autotune_to... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3140000 ns | +120000 ns (+4.0%) | 3110000 ns | +90000 ns (+3.0%) | 3200000 ns | +180000 ns (+6.0%) |
| SM Throughput (%) | 56.21 % | 33.90 % | -22.31 pp | 50.34 % | -5.87 pp | 32.80 % | -23.41 pp | 50.48 % | -5.73 pp |
| Memory Throughput (%) | 86.63 % | 86.06 % | -0.57 pp | 83.33 % | -3.30 pp | 84.19 % | -2.44 pp | 81.78 % | -4.85 pp |
| DRAM Throughput (%) | 86.63 % | 86.06 % | -0.57 pp | 83.33 % | -3.30 pp | 84.19 % | -2.44 pp | 81.78 % | -4.85 pp |
| Achieved Occupancy (%) | 87.64 % | 85.97 % | -1.67 pp | 79.80 % | -7.84 pp | 54.63 % | -33.01 pp | 52.79 % | -34.85 pp |
| Registers/Thread | 17.00 register/thread | 25.00 register/thread | +8 | 28.00 register/thread | +11 | 44.00 register/thread | +27 | 45.00 register/thread | +28 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 372.36 | -372.37 | 297.89 | -446.84 | 297.89 | -446.84 |
| No Eligible (%) | 43.89 % | 71.63 % | +27.74 pp | 49.08 % | +5.19 pp | 75.05 % | +31.16 pp | 49.19 % | +5.30 pp |
| IPC Active | 2.24 inst/cycle | 1.13 inst/cycle | -1.11 | 2.03 inst/cycle | -0.21 | 1.00 inst/cycle | -1.24 | 2.04 inst/cycle | -0.20 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)

Final best: 3020000 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,14 @@

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
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
     ],
     key=['n_elements'],
 )
@@ -19,12 +19,14 @@

 def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
     pid = tl.program_id(0)
     block_start = pid * BLOCK_SIZE
-    offsets = block_start + tl.arange(0, BLOCK_SIZE)
+    offsets = tl.max_contiguous(
+        tl.multiple_of(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
+        BLOCK_SIZE
+    )
     mask = offsets < n_elements
-    x = tl.load(input_ptr + offsets, mask=mask)
+    x = tl.load(input_ptr + offsets, mask=mask, eviction_policy='evict_first')
     y = libdevice.tanh(x)
-    tl.store(output_ptr + offsets, y, mask=mask)
-
+    tl.store(output_ptr + offsets, y, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(x):
     output = torch.empty_like(x)
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
