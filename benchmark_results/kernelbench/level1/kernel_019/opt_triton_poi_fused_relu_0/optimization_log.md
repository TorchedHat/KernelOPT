# Kernel Optimization Log

## Summary

- **Kernel**: `relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 2960000 ns (**1.0203×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: Best configuration **`cache_eviction_policy_streaming_hint_load_store`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `relu_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L1_gpu0_v2/kernel_019/synthesis/kernel_0_triton_poi_fused_relu_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 18
- **Duration**: 3020000 ns
- **SM Throughput**: 8.71 % | **Memory**: 86.52 % | **DRAM**: 86.52 %
- **Achieved Occupancy**: 85.23 %
- **IPC Active**: 0.35 inst/cycle | **No Eligible**: 91.27 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_load_store | 0.978× | 0.993× | ✓ |  |
| B | vectorized_loads | 1.001× | 1.000× | ✓ |  |
| C | memory_coalescing_vectorized_loads_with_software_pipelining | 1.010× | 1.003× | ✓ | **✓ best** |
| D | add_num_stages_and_expand_autotune_configs | 1.001× | 0.784× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads | Δ | memory_coalescing... | Δ | add_num_stages_an... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3020000 ns | +0 ns (+0.0%) | 3010000 ns | -10000 ns (-0.3%) | 3850000 ns | +830000 ns (+27.5%) |
| SM Throughput (%) | 8.71 % | 11.34 % | +2.63 pp | 8.79 % | +0.08 pp | 9.27 % | +0.56 pp | 13.20 % | +4.49 pp |
| Memory Throughput (%) | 86.52 % | 86.00 % | -0.52 pp | 86.60 % | +0.08 pp | 86.98 % | +0.46 pp | 67.95 % | -18.57 pp |
| DRAM Throughput (%) | 86.52 % | 86.00 % | -0.52 pp | 86.60 % | +0.08 pp | 86.98 % | +0.46 pp | 67.95 % | -18.57 pp |
| Achieved Occupancy (%) | 85.23 % | 85.04 % | -0.19 pp | 84.99 % | -0.24 pp | 76.70 % | -8.53 pp | 30.91 % | -54.32 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 30.00 register/thread | +12 | 16.00 register/thread | -2 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 372.36 | -372.37 | 1489.45 | +744.72 |
| No Eligible (%) | 91.27 % | 88.65 % | -2.62 pp | 91.26 % | -0.01 pp | 90.61 % | -0.66 pp | 90.47 % | -0.80 pp |
| IPC Active | 0.35 inst/cycle | 0.46 inst/cycle | +0.11 | 0.35 inst/cycle | +0.00 | 0.38 inst/cycle | +0.03 | 0.38 inst/cycle | +0.03 |

---

## Iteration 2

**Best so far**: 1.0033× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_load_eviction_policy_and_expand_large_block_autotune | 0.964× | 0.993× | ✓ |  |
| B | l2_compressible_output_allocation_with_store_evict_first | 1.016× | 1.020× | ✓ | **✓ best** |
| C | force_larger_block_sizes_to_increase_memory_level_parallelism | 0.927× | 0.953× | ✓ |  |
| D | prune_autotune_configs_enforce_minimum_block_size | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_load_eviction... | Δ | l2_compressible_o... | Δ | force_larger_bloc... | Δ | prune_autotune_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 2960000 ns | -60000 ns (-2.0%) | 3170000 ns | +150000 ns (+5.0%) | 3020000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 8.71 % | 11.33 % | +2.62 pp | 11.56 % | +2.85 pp | 11.00 % | +2.29 pp | 8.73 % | +0.02 pp |
| Memory Throughput (%) | 86.52 % | 86.16 % | -0.36 pp | 88.26 % | +1.74 pp | 82.59 % | -3.93 pp | 86.61 % | +0.09 pp |
| DRAM Throughput (%) | 86.52 % | 86.16 % | -0.36 pp | 88.26 % | +1.74 pp | 82.59 % | -3.93 pp | 86.61 % | +0.09 pp |
| Achieved Occupancy (%) | 85.23 % | 85.09 % | -0.14 pp | 85.20 % | -0.03 pp | 78.60 % | -6.63 pp | 85.07 % | -0.16 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 91.27 % | 88.59 % | -2.68 pp | 88.43 % | -2.84 pp | 88.95 % | -2.32 pp | 91.20 % | -0.07 pp |
| IPC Active | 0.35 inst/cycle | 0.46 inst/cycle | +0.11 | 0.47 inst/cycle | +0.12 | 0.44 inst/cycle | +0.09 | 0.35 inst/cycle | +0.00 |

---

## Iteration 3

**Best so far**: 1.0203× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_load_eviction_policy_streaming_input | 0.978× | 0.990× | ✓ |  |
| B | cache_eviction_policy_streaming_hint_load_store | 1.017× | 1.020× | ✓ | **✓ best** |
| C | persistent_grid_stride_loop_to_reduce_launch_overhead_and_increase_memory_level_parallelism | 0.939× | 0.956× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_load_eviction... | Δ | cache_eviction_po... | Δ | persistent_grid_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3050000 ns | +30000 ns (+1.0%) | 2960000 ns | -60000 ns (-2.0%) | 3160000 ns | +140000 ns (+4.6%) |
| SM Throughput (%) | 8.71 % | 11.47 % | +2.76 pp | 11.55 % | +2.84 pp | 15.72 % | +7.01 pp |
| Memory Throughput (%) | 86.52 % | 85.91 % | -0.61 pp | 88.27 % | +1.75 pp | 82.93 % | -3.59 pp |
| DRAM Throughput (%) | 86.52 % | 85.91 % | -0.61 pp | 88.27 % | +1.75 pp | 82.93 % | -3.59 pp |
| Achieved Occupancy (%) | 85.23 % | 85.13 % | -0.10 pp | 85.08 % | -0.15 pp | 58.20 % | -27.03 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 40.00 register/thread | +22 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 496.48 | -248.25 |
| No Eligible (%) | 91.27 % | 88.52 % | -2.75 pp | 88.44 % | -2.83 pp | 84.56 % | -6.71 pp |
| IPC Active | 0.35 inst/cycle | 0.45 inst/cycle | +0.10 | 0.46 inst/cycle | +0.11 | 0.62 inst/cycle | +0.27 |

---

## Iteration 4

**Best so far**: 1.0203× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eliminate_tl_zeros_allocation_and_fix_load_eviction_with_tl_where | 0.998× | 0.997× | ✓ |  |
| C | reduce_register_pressure_via_flat_grid_elimination_of_stride_loop | 0.934× | 0.947× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eliminate_tl_zero... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3030000 ns | +10000 ns (+0.3%) | 3190000 ns | +170000 ns (+5.6%) |
| SM Throughput (%) | 8.71 % | 10.36 % | +1.65 pp | 11.43 % | +2.72 pp |
| Memory Throughput (%) | 86.52 % | 86.30 % | -0.22 pp | 81.94 % | -4.58 pp |
| DRAM Throughput (%) | 86.52 % | 86.30 % | -0.22 pp | 81.94 % | -4.58 pp |
| Achieved Occupancy (%) | 85.23 % | 85.11 % | -0.12 pp | 52.63 % | -32.60 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 44.00 register/thread | +26 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 297.89 | -446.84 |
| No Eligible (%) | 91.27 % | 89.70 % | -1.57 pp | 90.70 % | -0.57 pp |
| IPC Active | 0.35 inst/cycle | 0.41 inst/cycle | +0.06 | 0.37 inst/cycle | +0.02 |

---

## Iteration 5

**Best so far**: 1.0203× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | split_block_dual_independent_loads_to_increase_memory_level_parallelism | 0.998× | 0.997× | ✓ |  |
| C | reduce_register_pressure_eliminate_tl_zeros_to_improve_occupancy | 0.964× | 0.987× | ✓ |  |
| D | replace_tl_zeros_maximum_with_tl_where_to_reduce_register_pressure | 0.934× | 0.947× | ✓ |  |
| E | add_triton_autotune | 0.997× | 0.391× | ✓ |  |


### Metric Comparison

| Metric | Baseline | split_block_dual_... | Δ | reduce_register_p... | Δ | replace_tl_zeros_... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3030000 ns | +10000 ns (+0.3%) | 3060000 ns | +40000 ns (+1.3%) | 3190000 ns | +170000 ns (+5.6%) | 7730000 ns | +4710000 ns (+156.0%) |
| SM Throughput (%) | 8.71 % | 10.24 % | +1.53 pp | 10.99 % | +2.28 pp | 11.39 % | +2.68 pp | 26.36 % | +17.65 pp |
| Memory Throughput (%) | 86.52 % | 86.36 % | -0.16 pp | 85.48 % | -1.04 pp | 82.02 % | -4.50 pp | 33.88 % | -52.64 pp |
| DRAM Throughput (%) | 86.52 % | 86.36 % | -0.16 pp | 85.48 % | -1.04 pp | 82.02 % | -4.50 pp | 33.88 % | -52.64 pp |
| Achieved Occupancy (%) | 85.23 % | 85.14 % | -0.09 pp | 79.89 % | -5.34 pp | 52.58 % | -32.65 pp | 43.31 % | -41.92 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 24.00 register/thread | +6 | 44.00 register/thread | +26 | 16.00 register/thread | -2 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 372.36 | -372.37 | 297.89 | -446.84 | 5957.82 | +5213.09 |
| No Eligible (%) | 91.27 % | 89.70 % | -1.57 pp | 89.77 % | -1.50 pp | 90.74 % | -0.53 pp | 77.77 % | -13.50 pp |
| IPC Active | 0.35 inst/cycle | 0.41 inst/cycle | +0.06 | 0.41 inst/cycle | +0.06 | 0.37 inst/cycle | +0.02 | 0.89 inst/cycle | +0.54 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)
Iter 1     3010000 ns  (1.003×)  — memory_coalescing_vectorized_loads_with_software_pipelining
Iter 2     2960000 ns  (1.020×)  — l2_compressible_output_allocation_with_store_evict_first

Final best: 2960000 ns  (1.0203×)
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
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=16),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
     ],
     key=['n_elements'],
 )
@@ -21,10 +23,8 @@

     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
     x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
-    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
-    result = tl.maximum(zero, x)
-    tl.store(output_ptr + offsets, result, mask=mask)
-
+    result = tl.where(x > 0.0, x, 0.0)
+    tl.store(output_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
