# Kernel Optimization Log

## Summary

- **Kernel**: `hardsigmoid_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 2960000 ns (**1.0203×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: Best configuration **`fuse_fp32_instructions_and_expand_autotune_configs`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `hardsigmoid_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_028/synthesis/kernel_0_triton_poi_fused_hardsigmoid_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 18
- **Duration**: 3020000 ns
- **SM Throughput**: 17.12 % | **Memory**: 86.59 % | **DRAM**: 86.59 %
- **Achieved Occupancy**: 85.16 %
- **IPC Active**: 0.69 inst/cycle | **No Eligible**: 82.66 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_l2 | 0.977× | 0.990× | ✓ |  |
| B | vectorized_loads_with_contiguous_hints | 0.991× | 1.000× | ✓ |  |
| C | increase_block_size_and_add_num_stages_for_memory_latency_hiding | 0.990× | 1.000× | ✓ |  |
| D | add_num_stages_software_pipelining_and_prefer_large_blocks | 0.990× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | increase_block_si... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3050000 ns | +30000 ns (+1.0%) | 3020000 ns | +0 ns (+0.0%) | 3020000 ns | +0 ns (+0.0%) | 3020000 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 17.12 % | 19.86 % | +2.74 pp | 17.37 % | +0.25 pp | 17.19 % | +0.07 pp | 17.31 % | +0.19 pp |
| Memory Throughput (%) | 86.59 % | 85.94 % | -0.65 pp | 86.63 % | +0.04 pp | 86.58 % | -0.01 pp | 86.62 % | +0.03 pp |
| DRAM Throughput (%) | 86.59 % | 85.94 % | -0.65 pp | 86.63 % | +0.04 pp | 86.58 % | -0.01 pp | 86.62 % | +0.03 pp |
| Achieved Occupancy (%) | 85.16 % | 85.15 % | -0.01 pp | 85.11 % | -0.05 pp | 85.17 % | +0.01 pp | 85.03 % | -0.13 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 82.66 % | 80.11 % | -2.55 pp | 82.86 % | +0.20 pp | 82.86 % | +0.20 pp | 82.86 % | +0.20 pp |
| IPC Active | 0.69 inst/cycle | 0.79 inst/cycle | +0.10 | 0.69 inst/cycle | +0.00 | 0.69 inst/cycle | +0.00 | 0.69 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fused_multiply_add_arithmetic_rewrite | 1.001× | 1.007× | ✓ |  |
| B | fuse_fp32_instructions_and_expand_autotune_configs | 1.012× | 1.020× | ✓ | **✓ best** |
| C | evict_first_streaming_with_expanded_autotune_num_stages | 0.964× | 0.993× | ✓ |  |
| D | l2_compression_compressible_output_allocation | 0.980× | 0.977× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fused_multiply_ad... | Δ | fuse_fp32_instruc... | Δ | evict_first_strea... | Δ | l2_compression_co... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3000000 ns | -20000 ns (-0.7%) | 2960000 ns | -60000 ns (-2.0%) | 3040000 ns | +20000 ns (+0.7%) | 3090000 ns | +70000 ns (+2.3%) |
| SM Throughput (%) | 17.12 % | 19.18 % | +2.06 pp | 18.00 % | +0.88 pp | 20.00 % | +2.88 pp | 17.52 % | +0.40 pp |
| Memory Throughput (%) | 86.59 % | 87.27 % | +0.68 pp | 88.28 % | +1.69 pp | 86.13 % | -0.46 pp | 84.52 % | -2.07 pp |
| DRAM Throughput (%) | 86.59 % | 87.27 % | +0.68 pp | 88.28 % | +1.69 pp | 86.13 % | -0.46 pp | 84.52 % | -2.07 pp |
| Achieved Occupancy (%) | 85.16 % | 82.60 % | -2.56 pp | 85.27 % | +0.11 pp | 85.08 % | -0.08 pp | 85.20 % | +0.04 pp |
| Registers/Thread | 18.00 register/thread | 16.00 register/thread | -2 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 1489.45 | +744.72 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 |
| No Eligible (%) | 82.66 % | 80.87 % | -1.79 pp | 81.89 % | -0.77 pp | 80.23 % | -2.43 pp | 82.48 % | -0.18 pp |
| IPC Active | 0.69 inst/cycle | 0.77 inst/cycle | +0.08 | 0.73 inst/cycle | +0.04 | 0.79 inst/cycle | +0.10 | 0.71 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0203× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_input_eviction_policy_evict_last_to_evict_first | 0.964× | 0.993× | ✓ |  |
| B | expand_autotune_num_warps_for_memory_latency_hiding | 1.002× | 0.392× | ✓ |  |
| C | increase_num_warps_small_block_for_latency_hiding | 0.977× | 0.776× | ✓ |  |
| D | deeper_software_pipelining_num_stages_autotune_expansion | 0.980× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_input_evictio... | Δ | expand_autotune_n... | Δ | increase_num_warp... | Δ | deeper_software_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 7700000 ns | +4680000 ns (+155.0%) | 3890000 ns | +870000 ns (+28.8%) | 3100000 ns | +80000 ns (+2.6%) |
| SM Throughput (%) | 17.12 % | 17.85 % | +0.73 pp | 13.23 % | -3.89 pp | 27.37 % | +10.25 pp | 17.50 % | +0.38 pp |
| Memory Throughput (%) | 86.59 % | 85.99 % | -0.60 pp | 34.00 % | -52.59 pp | 67.29 % | -19.30 pp | 84.45 % | -2.14 pp |
| DRAM Throughput (%) | 86.59 % | 85.99 % | -0.60 pp | 34.00 % | -52.59 pp | 67.29 % | -19.30 pp | 84.45 % | -2.14 pp |
| Achieved Occupancy (%) | 85.16 % | 85.17 % | +0.01 pp | 25.88 % | -59.28 pp | 51.48 % | -33.68 pp | 85.18 % | +0.02 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 16.00 register/thread | -2 | 16.00 register/thread | -2 | 18.00 register/thread | +0 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 2978.91 | +2234.18 | 2978.91 | +2234.18 | 744.73 | +0.00 |
| No Eligible (%) | 82.66 % | 82.30 % | -0.36 pp | 88.85 % | +6.19 pp | 72.92 % | -9.74 pp | 82.36 % | -0.30 pp |
| IPC Active | 0.69 inst/cycle | 0.71 inst/cycle | +0.02 | 0.45 inst/cycle | -0.24 | 1.08 inst/cycle | +0.39 | 0.71 inst/cycle | +0.02 |


### Key Learnings

#### ❌ Plan C: `increase_num_warps_small_block_for_latency_hiding` (0.776×)

Adding smaller BLOCK_SIZE configs (256, 512) to the autotune search space caused the autotuner to select a much smaller block size, quadrupling the grid size (1.57M → 6.29M blocks) and increasing waves/SM from 745 to 2979. This dramatically reduced achieved occupancy from 85% to 51% and increased total instruction count by ~2x, raising kernel duration from 2.96 ms to 3.89 ms. The per-warp stall cycles improved (75 → 30 cycles) but the massive increase in kernel launch overhead and total work overwhelmed any latency-hiding benefit. When a memory-bound kernel already has high occupancy (>80%) and high DRAM utilization (>85%), adding smaller block sizes to the autotune config set risks the autotuner selecting an inefficient small-block configuration that increases total overhead without proportionally improving throughput.

> **Applicable when**: Never — this specific strategy (adding smaller block configs to autotune) caused regression in this context.

> **Avoid when**: Kernel is already memory-bound with high DRAM utilization (>80%) and high achieved occupancy (>80%). Adding smaller BLOCK_SIZE configs to the autotune search space risks the autotuner selecting a configuration with far more blocks/waves, increasing total overhead and reducing occupancy. Also avoid when the kernel already has many waves per SM (>500), as further increasing grid size provides no benefit and adds scheduling overhead.

---

## Iteration 4

**Best so far**: 1.0203× | **Memory items**: 2

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | replace_degenerate_fma_with_true_fma_arithmetic_rewrite | 0.964× | 0.993× | ✓ |  |
| B | prune_autotune_to_large_blocks_and_fix_store_eviction_policy | 0.963× | 0.993× | ✓ |  |
| C | prune_small_block_autotune_configs_and_use_fma_clamp | 0.973× | 0.993× | ✓ |  |
| D | prune_small_block_autotune_configs_force_large_blocks | 1.001× | 1.007× | ✓ |  |


### Metric Comparison

| Metric | Baseline | replace_degenerat... | Δ | prune_autotune_to... | Δ | prune_small_block... | Δ | prune_small_block... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3040000 ns | +20000 ns (+0.7%) | 3040000 ns | +20000 ns (+0.7%) | 3000000 ns | -20000 ns (-0.7%) |
| SM Throughput (%) | 17.12 % | 15.81 % | -1.31 pp | 17.96 % | +0.84 pp | 17.75 % | +0.63 pp | 17.00 % | -0.12 pp |
| Memory Throughput (%) | 86.59 % | 85.99 % | -0.60 pp | 86.12 % | -0.47 pp | 86.00 % | -0.59 pp | 87.25 % | +0.66 pp |
| DRAM Throughput (%) | 86.59 % | 85.99 % | -0.60 pp | 86.12 % | -0.47 pp | 86.00 % | -0.59 pp | 87.25 % | +0.66 pp |
| Achieved Occupancy (%) | 85.16 % | 85.16 % | +0.00 pp | 85.16 % | +0.00 pp | 85.08 % | -0.08 pp | 82.46 % | -2.70 pp |
| Registers/Thread | 18.00 register/thread | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 18.00 register/thread | +0 | 16.00 register/thread | -2 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 1489.45 | +744.72 |
| No Eligible (%) | 82.66 % | 84.13 % | +1.47 pp | 82.06 % | -0.60 pp | 82.32 % | -0.34 pp | 87.19 % | +4.53 pp |
| IPC Active | 0.69 inst/cycle | 0.64 inst/cycle | -0.05 | 0.72 inst/cycle | +0.03 | 0.71 inst/cycle | +0.02 | 0.51 inst/cycle | -0.18 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)
Iter 2     2960000 ns  (1.020×)  — fuse_fp32_instructions_and_expand_autotune_configs

Final best: 2960000 ns  (1.0203×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,14 +3,22 @@

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
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=32, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=3),
     ],
     key=['n_elements'],
 )
@@ -26,16 +34,15 @@

     offsets = block_start + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
 
-    x = tl.load(in_ptr + offsets, mask=mask)
+    x = tl.load(in_ptr + offsets, mask=mask, eviction_policy='evict_first')
 
     # hardsigmoid: clamp(x + 3, 0, 6) / 6
-    x = x + 3.0
-    x = tl.where(x < 0.0, 0.0, x)
-    x = tl.where(x > 6.0, 6.0, x)
-    x = x * (1.0 / 6.0)
+    x = tl.math.fma(x, 1.0, 3.0)
+    x = tl.maximum(x, 0.0)
+    x = tl.minimum(x, 6.0)
+    x = tl.math.fma(x, 1.0 / 6.0, 0.0)
 
-    tl.store(out_ptr + offsets, x, mask=mask)
-
+    tl.store(out_ptr + offsets, x, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     x, = inputs
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
