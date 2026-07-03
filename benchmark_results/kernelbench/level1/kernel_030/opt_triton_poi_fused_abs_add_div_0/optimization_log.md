# Kernel Optimization Log

## Summary

- **Kernel**: `softsign_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3020000 ns — **Best latency**: 3100000 ns (**0.9742×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `softsign_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L1_gpu0_v2/kernel_030/synthesis/kernel_0_triton_poi_fused_abs_add_div_0.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (1572864,1,1), Waves/SM: 744.73
- **Registers/thread**: 26
- **Duration**: 3020000 ns
- **SM Throughput**: 19.46 % | **Memory**: 86.53 % | **DRAM**: 86.53 %
- **Achieved Occupancy**: 85.75 %
- **IPC Active**: 0.77 inst/cycle | **No Eligible**: 80.74 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_l2 | 0.977× | 0.993× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 1.002× | 1.000× | ✓ |  |
| C | add_software_pipelining_and_cache_modifier | 0.991× | 1.000× | ✓ |  |
| D | increase_block_size_and_add_num_stages_for_memory_latency_hiding | 1.002× | 0.974× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | eviction_policies... | Δ | vectorized_loads_... | Δ | add_software_pipe... | Δ | increase_block_si... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3020000 ns | +0 ns (+0.0%) | 3020000 ns | +0 ns (+0.0%) | 3100000 ns | +80000 ns (+2.6%) |
| SM Throughput (%) | 19.46 % | 22.38 % | +2.92 pp | 19.23 % | -0.23 pp | 19.44 % | -0.02 pp | 17.12 % | -2.34 pp |
| Memory Throughput (%) | 86.53 % | 85.95 % | -0.58 pp | 86.61 % | +0.08 pp | 86.65 % | +0.12 pp | 84.46 % | -2.07 pp |
| DRAM Throughput (%) | 86.53 % | 85.95 % | -0.58 pp | 86.61 % | +0.08 pp | 86.65 % | +0.12 pp | 84.46 % | -2.07 pp |
| Achieved Occupancy (%) | 85.75 % | 85.76 % | +0.01 pp | 85.64 % | -0.11 pp | 85.76 % | +0.01 pp | 37.26 % | -48.49 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 52.00 register/thread | +26 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 80.74 % | 77.83 % | -2.91 pp | 80.66 % | -0.08 pp | 80.50 % | -0.24 pp | 82.65 % | +1.91 pp |
| IPC Active | 0.77 inst/cycle | 0.88 inst/cycle | +0.11 | 0.77 inst/cycle | +0.00 | 0.77 inst/cycle | +0.00 | 0.69 inst/cycle | -0.08 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | l2_compression_compressible_output_allocation | 0.977× | 0.993× | ✓ |  |
| B | fma_fusion_and_compressible_output_allocation | 1.002× | 0.776× | ✓ |  |
| C | fused_fp32_instructions_and_larger_autotune_configs | 0.977× | 0.993× | ✓ |  |
| D | reduce_register_pressure_via_smaller_warps_and_larger_blocks_for_ilp | 0.977× | 0.965× | ✓ |  |


### Metric Comparison

| Metric | Baseline | l2_compression_co... | Δ | fma_fusion_and_co... | Δ | fused_fp32_instru... | Δ | reduce_register_p... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3890000 ns | +870000 ns (+28.8%) | 3040000 ns | +20000 ns (+0.7%) | 3130000 ns | +110000 ns (+3.6%) |
| SM Throughput (%) | 19.46 % | 17.70 % | -1.76 pp | 26.96 % | +7.50 pp | 24.28 % | +4.82 pp | 20.87 % | +1.41 pp |
| Memory Throughput (%) | 86.53 % | 85.96 % | -0.57 pp | 67.29 % | -19.24 pp | 86.06 % | -0.47 pp | 83.51 % | -3.02 pp |
| DRAM Throughput (%) | 86.53 % | 85.96 % | -0.57 pp | 67.29 % | -19.24 pp | 86.06 % | -0.47 pp | 83.51 % | -3.02 pp |
| Achieved Occupancy (%) | 85.75 % | 85.65 % | -0.10 pp | 52.85 % | -32.90 pp | 85.69 % | -0.06 pp | 37.61 % | -48.14 pp |
| Registers/Thread | 26.00 register/thread | 24.00 register/thread | -2 | 16.00 register/thread | -10 | 27.00 register/thread | +1 | 54.00 register/thread | +28 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 2978.91 | +2234.18 | 744.73 | +0.00 | 372.36 | -372.37 |
| No Eligible (%) | 80.74 % | 82.32 % | +1.58 pp | 73.02 % | -7.72 pp | 75.71 % | -5.03 pp | 78.83 % | -1.91 pp |
| IPC Active | 0.77 inst/cycle | 0.72 inst/cycle | -0.05 | 1.10 inst/cycle | +0.33 | 0.98 inst/cycle | +0.21 | 0.85 inst/cycle | +0.08 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 1

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_native_division_replacing_fast_dividef | 0.964× | 0.993× | ✓ |  |
| B | fma_fusion_via_direct_division_and_prune_bad_autotune_configs | 0.964× | 0.993× | ✓ |  |
| C | reduce_register_pressure_via_expression_fusion_for_occupancy | 0.964× | 0.962× | ✓ |  |
| D | purge_small_block_configs_to_restore_large_tile_selection | 0.990× | 1.000× | ✓ |  |
| E | add_triton_autotune | 0.973× | 0.391× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_via_na... | Δ | fma_fusion_via_di... | Δ | reduce_register_p... | Δ | purge_small_block... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3020000 ns | 3040000 ns | +20000 ns (+0.7%) | 3040000 ns | +20000 ns (+0.7%) | 3140000 ns | +120000 ns (+4.0%) | 3020000 ns | +0 ns (+0.0%) | 7730000 ns | +4710000 ns (+156.0%) |
| SM Throughput (%) | 19.46 % | 21.96 % | +2.50 pp | 21.87 % | +2.41 pp | 17.98 % | -1.48 pp | 21.62 % | +2.16 pp | 26.31 % | +6.85 pp |
| Memory Throughput (%) | 86.53 % | 86.03 % | -0.50 pp | 86.02 % | -0.51 pp | 83.34 % | -3.19 pp | 86.61 % | +0.08 pp | 33.87 % | -52.66 pp |
| DRAM Throughput (%) | 86.53 % | 86.03 % | -0.50 pp | 86.02 % | -0.51 pp | 83.34 % | -3.19 pp | 86.61 % | +0.08 pp | 33.87 % | -52.66 pp |
| Achieved Occupancy (%) | 85.75 % | 85.64 % | -0.11 pp | 85.74 % | -0.01 pp | 29.17 % | -56.58 pp | 85.64 % | -0.11 pp | 45.20 % | -40.55 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 85.00 register/thread | +59 | 27.00 register/thread | +1 | 16.00 register/thread | -10 |
| Waves/SM | 744.73 | 744.73 | +0.00 | 744.73 | +0.00 | 297.89 | -446.84 | 744.73 | +0.00 | 5957.82 | +5213.09 |
| No Eligible (%) | 80.74 % | 78.10 % | -2.64 pp | 78.07 % | -2.67 pp | 81.98 % | +1.24 pp | 78.15 % | -2.59 pp | 76.15 % | -4.59 pp |
| IPC Active | 0.77 inst/cycle | 0.88 inst/cycle | +0.11 | 0.88 inst/cycle | +0.11 | 0.72 inst/cycle | -0.05 | 0.88 inst/cycle | +0.11 | 0.95 inst/cycle | +0.18 |

---

## Cumulative Progress

```
Baseline:  3020000 ns  (1.000×)

Final best: 3100000 ns  (0.9742×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,19 @@

 from triton.language.extra import libdevice
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
     ],
     key=['n_elements'],
 )
@@ -24,14 +29,16 @@

 ):
     pid = tl.program_id(0)
     block_start = pid * BLOCK_SIZE
-    offsets = block_start + tl.arange(0, BLOCK_SIZE)
+    offsets = tl.max_contiguous(
+        tl.multiple_of(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
+        BLOCK_SIZE
+    )
     mask = offsets < n_elements
-    x = tl.load(x_ptr + offsets, mask=mask)
+    x = tl.load(x_ptr + offsets, mask=mask, eviction_policy='evict_first')
     abs_x = tl.abs(x)
     denom = abs_x + 1.0
-    result = x / denom
-    tl.store(out_ptr + offsets, result, mask=mask)
-
+    result = x * libdevice.fast_dividef(1.0, denom)
+    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
 
 def kernel_function(x):
     output = torch.empty_like(x)
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
