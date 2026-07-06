# Kernel Optimization Log

## Summary

- **Kernel**: `leaky_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 7730000 ns (**0.3907×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `leaky_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_020/synthesis/kernel_0_triton_poi_fused_leaky_relu_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 18
- **Duration**: 3020000 ns
- **SM Throughput**: 11.06 % | **Memory**: 86.73 % | **DRAM**: 86.73 %
- **Achieved Occupancy**: 85.24 %
- **IPC Active**: 0.44 inst/cycle | **No Eligible**: 88.87 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_data | 0.978× | 0.990× | ✓ |  |
| B | vectorized_loads_via_contiguous_hints | 0.991× | 1.000× | ✓ |  |
| C | increase_block_size_and_add_software_pipelining | 1.001× | 1.007× | ✓ | **✓ best** |
| D | increase_block_size_and_add_vectorized_loads_via_autotune | 0.991× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_block_si... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3050000 ns | +30000 ns (+1.0%) | 3020000 ns | +0 ns (+0.0%) | 3000000 ns | -20000 ns (-0.7%) | 3020000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 11.06 % | 13.71 % | +2.65 pp | 11.06 % | +0.00 pp | 17.19 % | +6.13 pp | 11.18 % | +0.12 pp |
| Memory Throughput (%) | 86.73 % | 85.92 % | -0.81 pp | 86.77 % | +0.04 pp | 87.20 % | +0.47 pp | 86.55 % | -0.18 pp |
| DRAM Throughput (%) | 86.73 % | 85.92 % | -0.81 pp | 86.77 % | +0.04 pp | 87.20 % | +0.47 pp | 86.55 % | -0.18 pp |
| Achieved Occupancy (%) | 85.24 % | 85.15 % | -0.09 pp | 85.13 % | -0.11 pp | 82.53 % | -2.71 pp | 85.09 % | -0.15 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 1489.45 | +744.72 | 744.73 | +0.00 |
| No Eligible (%) | 88.87 % | 86.27 % | -2.60 pp | 88.88 % | +0.01 pp | 85.11 % | -3.76 pp | 88.87 % | +0.00 pp |
| IPC Active | 0.44 inst/cycle | 0.55 inst/cycle | +0.11 | 0.45 inst/cycle | +0.01 | 0.60 inst/cycle | +0.16 | 0.44 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0067× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_multiply_add_leaky_relu_computation | 1.001× | 1.007× | ✓ |  |
| B | fused_fma_leaky_relu_with_larger_block_autotune | 0.973× | 0.962× | ✓ |  |
| C | streaming_store_evict_first_cache_policy | 0.989× | 0.853× | ✓ |  |
| D | add_num_stages_software_pipelining_to_autotune | 0.978× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_multiply_ad... | Δ | fused_fma_leaky_r... | Δ | streaming_store_e... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3000000 ns | -20000 ns (-0.7%) | 3140000 ns | +120000 ns (+4.0%) | 3540000 ns | +520000 ns (+17.2%) | 3040000 ns | +20000 ns (+0.7%) |
| SM Throughput (%) | 11.06 % | 21.25 % | +10.19 pp | 8.64 % | -2.42 pp | 8.94 % | -2.12 pp | 13.87 % | +2.81 pp |
| Memory Throughput (%) | 86.73 % | 87.29 % | +0.56 pp | 83.24 % | -3.49 pp | 81.25 % | -5.48 pp | 85.96 % | -0.77 pp |
| DRAM Throughput (%) | 86.73 % | 87.29 % | +0.56 pp | 83.24 % | -3.49 pp | 81.25 % | -5.48 pp | 85.96 % | -0.77 pp |
| Achieved Occupancy (%) | 85.24 % | 82.39 % | -2.85 pp | 51.49 % | -33.75 pp | 39.92 % | -45.32 pp | 85.16 % | -0.08 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 47.00 register/thread | +29 | 64.00 register/thread | +46 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 1489.45 | +744.72 | 297.89 | -446.84 | 186.18 | -558.55 | 744.73 | +0.00 |
| No Eligible (%) | 88.87 % | 82.98 % | -5.89 pp | 91.28 % | +2.41 pp | 90.77 % | +1.90 pp | 86.27 % | -2.60 pp |
| IPC Active | 0.44 inst/cycle | 0.69 inst/cycle | +0.25 | 0.35 inst/cycle | -0.09 | 0.36 inst/cycle | -0.08 | 0.55 inst/cycle | +0.11 |

---

## Iteration 3

**Best so far**: 1.0067× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | simplify_leaky_relu_to_tl_where_eliminate_abs_instruction | 1.001× | 1.007× | ✓ |  |
| B | fma_instruction_fusion_leaky_relu_arithmetic | 0.978× | 0.993× | ✓ |  |
| C | reduce_register_pressure_via_simpler_leaky_relu_computation | 1.001× | 0.391× | ✓ | **✓ best** |
| D | reduce_register_pressure_via_small_block_autotune | 0.898× | 0.907× | ✓ |  |
| E | add_triton_autotune | 1.001× | 0.391× | ✓ |  |


### Metric Comparison

| Metric | Baseline | simplify_leaky_re... | Δ | fma_instruction_f... | Δ | reduce_register_p... | Δ | reduce_register_p... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3000000 ns | -20000 ns (-0.7%) | 3040000 ns | +20000 ns (+0.7%) | 7730000 ns | +4710000 ns (+156.0%) | 3330000 ns | +310000 ns (+10.3%) | 7730000 ns | +4710000 ns (+156.0%) |
| SM Throughput (%) | 11.06 % | 17.00 % | +5.94 pp | 15.81 % | +4.75 pp | 26.32 % | +15.26 pp | 30.60 % | +19.54 pp | 32.90 % | +21.84 pp |
| Memory Throughput (%) | 86.73 % | 87.13 % | +0.40 pp | 86.01 % | -0.72 pp | 33.87 % | -52.86 pp | 78.67 % | -8.06 pp | 33.89 % | -52.84 pp |
| DRAM Throughput (%) | 86.73 % | 87.13 % | +0.40 pp | 86.01 % | -0.72 pp | 33.87 % | -52.86 pp | 78.67 % | -8.06 pp | 33.89 % | -52.84 pp |
| Achieved Occupancy (%) | 85.24 % | 82.54 % | -2.70 pp | 85.10 % | -0.14 pp | 44.43 % | -40.81 pp | 70.54 % | -14.70 pp | 41.60 % | -43.64 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 22.00 register/thread | +4 | 16.00 register/thread | -2 | 16.00 register/thread | -2 | 16.00 register/thread | -2 |
| Waves/SM | 744.73 | 1489.45 | +744.72 | 744.73 | +0.00 | 5957.82 | +5213.09 | 2978.91 | +2234.18 | 5957.82 | +5213.09 |
| No Eligible (%) | 88.87 % | 84.98 % | -3.89 pp | 84.18 % | -4.69 pp | 80.21 % | -8.66 pp | 77.92 % | -10.95 pp | 80.25 % | -8.62 pp |
| IPC Active | 0.44 inst/cycle | 0.60 inst/cycle | +0.16 | 0.64 inst/cycle | +0.20 | 0.79 inst/cycle | +0.35 | 0.89 inst/cycle | +0.45 | 0.79 inst/cycle | +0.35 |

---

## Iteration 4

**Best so far**: 1.0067× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | alignment_hints_max_contiguous_multiple_of_for_coalesced_loads | 1.001× | 1.007× | ✓ |  |
| B | memory_access_alignment_hints_for_wider_loads | 0.978× | 0.993× | ✓ |  |
| C | large_block_alignment_hints_reduce_l1tex_stalls | 1.000× | 0.977× | ✓ |  |
| D | prune_small_block_autotune_configs_to_fix_grid_explosion | 1.000× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | alignment_hints_m... | Δ | memory_access_ali... | Δ | large_block_align... | Δ | prune_small_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3000000 ns | -20000 ns (-0.7%) | 3040000 ns | +20000 ns (+0.7%) | 3090000 ns | +70000 ns (+2.3%) | 3020000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 11.06 % | 17.00 % | +5.94 pp | 15.84 % | +4.78 pp | 11.39 % | +0.33 pp | 11.08 % | +0.02 pp |
| Memory Throughput (%) | 86.73 % | 87.25 % | +0.52 pp | 86.00 % | -0.73 pp | 84.83 % | -1.90 pp | 86.59 % | -0.14 pp |
| DRAM Throughput (%) | 86.73 % | 87.25 % | +0.52 pp | 86.00 % | -0.73 pp | 84.83 % | -1.90 pp | 86.59 % | -0.14 pp |
| Achieved Occupancy (%) | 85.24 % | 82.50 % | -2.74 pp | 85.14 % | -0.10 pp | 71.63 % | -13.61 pp | 85.05 % | -0.19 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 22.00 register/thread | +4 | 22.00 register/thread | +4 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 1489.45 | +744.72 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 88.87 % | 85.12 % | -3.75 pp | 83.99 % | -4.88 pp | 88.51 % | -0.36 pp | 88.93 % | +0.06 pp |
| IPC Active | 0.44 inst/cycle | 0.59 inst/cycle | +0.15 | 0.64 inst/cycle | +0.20 | 0.45 inst/cycle | +0.01 | 0.45 inst/cycle | +0.01 |

---

## Iteration 5

**Best so far**: 1.0067× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_kernel_scheduling_eliminate_wave_tail_overhead | 0.983× | 0.990× | ✓ |  |
| B | l2_evict_first_output_store_with_constexpr_negative_slope_fma | 0.997× | 0.997× | ✓ |  |
| C | force_large_block_sizes_eliminate_small_configs_simplify_compute | 0.964× | 0.993× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_kernel... | Δ | l2_evict_first_ou... | Δ | force_large_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3050000 ns | +30000 ns (+1.0%) | 3030000 ns | +10000 ns (+0.3%) | 3040000 ns | +20000 ns (+0.7%) |
| SM Throughput (%) | 11.06 % | 43.54 % | +32.48 pp | 12.35 % | +1.29 pp | 13.70 % | +2.64 pp |
| Memory Throughput (%) | 86.73 % | 85.92 % | -0.81 pp | 86.41 % | -0.32 pp | 86.05 % | -0.68 pp |
| DRAM Throughput (%) | 86.73 % | 85.92 % | -0.81 pp | 86.41 % | -0.32 pp | 86.05 % | -0.68 pp |
| Achieved Occupancy (%) | 85.24 % | 85.60 % | +0.36 pp | 85.18 % | -0.06 pp | 85.15 % | -0.09 pp |
| Registers/Thread | 18.00 register/thread | 22.00 register/thread | +4 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 88.87 % | 59.19 % | -29.68 pp | 87.64 % | -1.23 pp | 86.13 % | -2.74 pp |
| IPC Active | 0.44 inst/cycle | 1.62 inst/cycle | +1.18 | 0.49 inst/cycle | +0.05 | 0.55 inst/cycle | +0.11 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)
Iter 1     3000000 ns  (1.007×)  — increase_block_size_and_add_software_pipelining

Final best: 7730000 ns  (0.3907×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,24 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
     ],
     key=['n_elements'],
 )
@@ -24,20 +34,18 @@

 ):
     pid = tl.program_id(0)
     block_start = pid * BLOCK_SIZE
-    offsets = block_start + tl.arange(0, BLOCK_SIZE)
+    offsets = tl.max_contiguous(
+        tl.multiple_of(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
+        BLOCK_SIZE
+    )
     mask = offsets < n_elements
 
     x = tl.load(input_ptr + offsets, mask=mask)
 
-    # LeakyReLU: x if x > 0 else negative_slope * x
-    zero = tl.full([BLOCK_SIZE], 0.0, dtype=tl.float32)
     x_f32 = x.to(tl.float32)
-    is_positive = x_f32 > zero
-    scaled = x_f32 * negative_slope
-    result = tl.where(is_positive, x_f32, scaled)
+    result = tl.where(x_f32 >= 0, x_f32, x_f32 * negative_slope)
 
     tl.store(output_ptr + offsets, result.to(output_ptr.dtype.element_ty), mask=mask)
-
 
 def kernel_function(*inputs):
     x = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
