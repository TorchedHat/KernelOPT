# Kernel Optimization Log

## Summary

- **Kernel**: `bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 59680 ns — **Best latency**: 60960 ns (**0.9790×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `bias_relu_kernel`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_011/synthesis/kernel_16_triton_poi_fused_convolution_max_pool2d_.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 26
- **Duration**: 59680 ns
- **SM Throughput**: 35.30 % | **Memory**: 79.59 % | **DRAM**: 79.59 %
- **Achieved Occupancy**: 85.51 %
- **IPC Active**: 1.48 inst/cycle | **No Eligible**: 62.96 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | eviction_policies_streaming_loads_stores | — | fail | ✓ |  |
| B | vectorized_loads_via_contiguous_multiple_of_hints | 0.999× | 0.997× | ✓ |  |
| D | expand_autotune_block_sizes_and_cache_bias_in_l1 | 1.001× | 0.999× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | expand_autotune_b... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 59680 ns | 59870 ns | +190 ns (+0.3%) | 59740 ns | +60 ns (+0.1%) |
| SM Throughput (%) | 35.30 % | 46.17 % | +10.87 pp | 52.01 % | +16.71 pp |
| Memory Throughput (%) | 79.59 % | 79.69 % | +0.10 pp | 79.99 % | +0.40 pp |
| DRAM Throughput (%) | 79.59 % | 79.69 % | +0.10 pp | 79.99 % | +0.40 pp |
| Achieved Occupancy (%) | 85.51 % | 82.30 % | -3.21 pp | 82.45 % | -3.06 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 62.96 % | 51.73 % | -11.23 pp | 46.06 % | -16.90 pp |
| IPC Active | 1.48 inst/cycle | 1.93 inst/cycle | +0.45 | 2.16 inst/cycle | +0.68 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | software_pipeline_num_stages_to_hide_l1tex_latency | 1.002× | 0.997× | ✓ |  |
| B | software_pipelining_num_stages_to_hide_l1tex_latency | 0.999× | 0.981× | ✓ |  |
| C | bias_shared_memory_prefetch_to_eliminate_uncoalesced_gather | 1.004× | 0.989× | ✓ | **✓ best** |
| D | software_pipelining_via_num_stages_autotune | 1.002× | 1.002× | ✓ |  |


### Metric Comparison

| Metric | Baseline | software_pipeline... | Δ | software_pipelini... | Δ | bias_shared_memor... | Δ | software_pipelini... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 59680 ns | 59870 ns | +190 ns (+0.3%) | 60830 ns | +1150 ns (+1.9%) | 60320 ns | +640 ns (+1.1%) | 59550 ns | -130 ns (-0.2%) |
| SM Throughput (%) | 35.30 % | 52.66 % | +17.36 pp | 52.45 % | +17.15 pp | 55.58 % | +20.28 pp | 46.26 % | +10.96 pp |
| Memory Throughput (%) | 79.59 % | 79.80 % | +0.21 pp | 78.53 % | -1.06 pp | 78.91 % | -0.68 pp | 80.15 % | +0.56 pp |
| DRAM Throughput (%) | 79.59 % | 79.80 % | +0.21 pp | 78.53 % | -1.06 pp | 78.91 % | -0.68 pp | 80.15 % | +0.56 pp |
| Achieved Occupancy (%) | 85.51 % | 82.41 % | -3.10 pp | 82.23 % | -3.28 pp | 81.54 % | -3.97 pp | 82.25 % | -3.26 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 | 22.00 register/thread | -4 | 18.00 register/thread | -8 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 29.70 | +14.85 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 62.96 % | 46.29 % | -16.67 pp | 45.83 % | -17.13 pp | 40.34 % | -22.62 pp | 51.32 % | -11.64 pp |
| IPC Active | 1.48 inst/cycle | 2.13 inst/cycle | +0.65 | 2.19 inst/cycle | +0.71 | 2.37 inst/cycle | +0.89 | 1.95 inst/cycle | +0.47 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | evict_first_streaming_x_out_keep_bias_in_l2 | 1.004× | 0.987× | ✓ |  |
| B | fma_fusion_and_streaming_eviction_for_x_and_out | 0.999× | 0.971× | ✓ |  |
| D | evict_first_streaming_x_out_evict_last_bias_all_kernels | 1.004× | 0.964× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | evict_first_strea... | Δ | fma_fusion_and_st... | Δ | evict_first_strea... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 59680 ns | 60450 ns | +770 ns (+1.3%) | 61470 ns | +1790 ns (+3.0%) | 61890 ns | +2210 ns (+3.7%) |
| SM Throughput (%) | 35.30 % | 57.00 % | +21.70 pp | 53.98 % | +18.68 pp | 54.08 % | +18.78 pp |
| Memory Throughput (%) | 79.59 % | 78.64 % | -0.95 pp | 77.42 % | -2.17 pp | 76.91 % | -2.68 pp |
| DRAM Throughput (%) | 79.59 % | 78.64 % | -0.95 pp | 77.42 % | -2.17 pp | 76.91 % | -2.68 pp |
| Achieved Occupancy (%) | 85.51 % | 81.71 % | -3.80 pp | 82.37 % | -3.14 pp | 82.33 % | -3.18 pp |
| Registers/Thread | 26.00 register/thread | 22.00 register/thread | -4 | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 62.96 % | 40.82 % | -22.14 pp | 43.47 % | -19.49 pp | 42.98 % | -19.98 pp |
| IPC Active | 1.48 inst/cycle | 2.36 inst/cycle | +0.88 | 2.25 inst/cycle | +0.77 | 2.26 inst/cycle | +0.78 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_via_direct_bias_load_replacing_gather | — | fail | ✓ |  |
| B | fma_fusion_and_eliminate_integer_division_chan_idx | 1.003× | 0.979× | ✓ | **✓ best** |
| C | fma_fusion_via_integer_division_elimination | 1.003× | 0.970× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_el... | Δ | fma_fusion_via_in... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 59680 ns | 60960 ns | +1280 ns (+2.1%) | 61500 ns | +1820 ns (+3.0%) |
| SM Throughput (%) | 35.30 % | 53.40 % | +18.10 pp | 54.02 % | +18.72 pp |
| Memory Throughput (%) | 79.59 % | 78.05 % | -1.54 pp | 77.47 % | -2.12 pp |
| DRAM Throughput (%) | 79.59 % | 78.05 % | -1.54 pp | 77.47 % | -2.12 pp |
| Achieved Occupancy (%) | 85.51 % | 82.41 % | -3.10 pp | 82.70 % | -2.81 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 62.96 % | 42.96 % | -20.00 pp | 44.32 % | -18.64 pp |
| IPC Active | 1.48 inst/cycle | 2.22 inst/cycle | +0.74 | 2.22 inst/cycle | +0.74 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fma_fusion_and_num_warps_expansion_for_latency_hiding | 1.001× | 0.984× | ✓ |  |
| C | evict_policies_nhwc_linear_kernels_missing_annotations | 1.001× | 0.981× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fma_fusion_and_nu... | Δ | evict_policies_nh... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 59680 ns | 60640 ns | +960 ns (+1.6%) | 60830 ns | +1150 ns (+1.9%) |
| SM Throughput (%) | 35.30 % | 54.07 % | +18.77 pp | 53.04 % | +17.74 pp |
| Memory Throughput (%) | 79.59 % | 78.60 % | -0.99 pp | 78.33 % | -1.26 pp |
| DRAM Throughput (%) | 79.59 % | 78.60 % | -0.99 pp | 78.33 % | -1.26 pp |
| Achieved Occupancy (%) | 85.51 % | 82.55 % | -2.96 pp | 82.36 % | -3.15 pp |
| Registers/Thread | 26.00 register/thread | 18.00 register/thread | -8 | 18.00 register/thread | -8 |
| Waves/SM | 14.85 | 29.70 | +14.85 | 29.70 | +14.85 |
| No Eligible (%) | 62.96 % | 43.57 % | -19.39 pp | 43.79 % | -19.17 pp |
| IPC Active | 1.48 inst/cycle | 2.24 inst/cycle | +0.76 | 2.27 inst/cycle | +0.79 |

---

## Cumulative Progress

```
Baseline:  59680 ns  (1.000×)

Final best: 60960 ns  (0.9790×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,14 +4,29 @@

 import torch.nn.functional as F
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
     ],
-    key=['numel'],
+    key=['numel', 'spatial', 'channels'],
 )
 @triton.jit
 def bias_relu_kernel(
@@ -33,25 +48,40 @@

     # For standard NCHW: channel = (offset // spatial) % channels
     chan_idx = (offsets // spatial) % channels
     
-    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
-    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
+    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
+    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0, eviction_policy='evict_last')
     
     result = x + bias
     result = tl.maximum(result, 0.0)
     
-    tl.store(out_ptr + offsets, result, mask=mask)
+    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
 
 
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=1),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
+        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=4),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
     ],
     key=['numel'],
 )
 @triton.jit
+
 def bias_relu_nhwc_kernel(
     x_ptr,
     bias_ptr,
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `—`
