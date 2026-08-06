# Kernel Optimization Log

## Summary

- **Kernel**: `bias_relu_kernel` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 60510 ns — **Best latency**: 157150 ns (**0.3850×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `bias_relu_kernel`
- **File**: `/opt/kernelopt/kb_benchmark_results/beam_L3_rerun/kernel_012/synthesis/kernel_16_triton_poi_fused_convolution_max_pool2d_.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (31360,1,1), Waves/SM: 14.85
- **Registers/thread**: 26
- **Duration**: 60510 ns
- **SM Throughput**: 34.79 % | **Memory**: 78.41 % | **DRAM**: 78.41 %
- **Achieved Occupancy**: 85.34 %
- **IPC Active**: 1.47 inst/cycle | **No Eligible**: 62.93 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_eviction_policies_streaming_loads_stores | 0.998× | 0.995× | ✓ |  |
| B | vectorized_loads_with_alignment_hints | 0.999× | 1.004× | ✓ |  |
| C | memory_coalescing_bias_shared_memory_preload | 0.983× | 1.008× | ✓ |  |
| D | cache_bias_in_l1_and_expand_autotune_block_sizes | 0.999× | 0.927× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_eviction_poli... | Δ | vectorized_loads_... | Δ | memory_coalescing... | Δ | cache_bias_in_l1_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60510 ns | 60800 ns | +290 ns (+0.5%) | 60260 ns | -250 ns (-0.4%) | 60060 ns | -450 ns (-0.7%) | 65250 ns | +4740 ns (+7.8%) |
| SM Throughput (%) | 34.79 % | 37.75 % | +2.96 pp | 35.11 % | +0.32 pp | 35.95 % | +1.16 pp | 28.58 % | -6.21 pp |
| Memory Throughput (%) | 78.41 % | 77.92 % | -0.49 pp | 78.71 % | +0.30 pp | 78.98 % | +0.57 pp | 73.25 % | -5.16 pp |
| DRAM Throughput (%) | 78.41 % | 77.92 % | -0.49 pp | 78.71 % | +0.30 pp | 78.98 % | +0.57 pp | 73.25 % | -5.16 pp |
| Achieved Occupancy (%) | 85.34 % | 85.34 % | +0.00 pp | 85.14 % | -0.20 pp | 85.23 % | -0.11 pp | 59.07 % | -26.27 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 28.00 register/thread | +2 | 34.00 register/thread | +8 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 | 9.90 | -4.95 |
| No Eligible (%) | 62.93 % | 59.07 % | -3.86 pp | 62.89 % | -0.04 pp | 61.49 % | -1.44 pp | 70.04 % | +7.11 pp |
| IPC Active | 1.47 inst/cycle | 1.63 inst/cycle | +0.16 | 1.48 inst/cycle | +0.01 | 1.49 inst/cycle | +0.02 | 1.19 inst/cycle | -0.28 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | add_num_stages_pipelining_and_fix_nchw_bias_load | 0.984× | 1.014× | ✓ |  |
| C | add_num_stages_software_pipelining_to_hide_l1tex_latency | 1.002× | 0.978× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_num_stages_pi... | Δ | add_num_stages_so... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 60510 ns | 59650 ns | -860 ns (-1.4%) | 61860 ns | +1350 ns (+2.2%) |
| SM Throughput (%) | 34.79 % | 35.55 % | +0.76 pp | 37.96 % | +3.17 pp |
| Memory Throughput (%) | 78.41 % | 79.67 % | +1.26 pp | 76.62 % | -1.79 pp |
| DRAM Throughput (%) | 78.41 % | 79.67 % | +1.26 pp | 76.62 % | -1.79 pp |
| Achieved Occupancy (%) | 85.34 % | 85.43 % | +0.09 pp | 85.65 % | +0.31 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.93 % | 62.97 % | +0.04 pp | 59.55 % | -3.38 pp |
| IPC Active | 1.47 inst/cycle | 1.50 inst/cycle | +0.03 | 1.59 inst/cycle | +0.12 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | vectorized_loads_alignment_hints_fused_bias_broadcast | 1.001× | 1.005× | ✓ |  |
| C | 2d_grid_bias_broadcast_with_fma_fusion | 1.002× | 0.985× | ✓ | **✓ best** |
| D | vectorized_loads_with_bias_scalar_broadcast | 0.999× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | 2d_grid_bias_broa... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 60510 ns | 60190 ns | -320 ns (-0.5%) | 61410 ns | +900 ns (+1.5%) | 60480 ns | -30 ns (-0.0%) |
| SM Throughput (%) | 34.79 % | 35.72 % | +0.93 pp | 37.64 % | +2.85 pp | 40.00 % | +5.21 pp |
| Memory Throughput (%) | 78.41 % | 78.77 % | +0.36 pp | 77.11 % | -1.30 pp | 78.25 % | -0.16 pp |
| DRAM Throughput (%) | 78.41 % | 78.77 % | +0.36 pp | 77.11 % | -1.30 pp | 78.25 % | -0.16 pp |
| Achieved Occupancy (%) | 85.34 % | 85.28 % | -0.06 pp | 85.48 % | +0.14 pp | 84.83 % | -0.51 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 | 27.00 register/thread | +1 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.93 % | 62.84 % | -0.09 pp | 60.24 % | -2.69 pp | 58.50 % | -4.43 pp |
| IPC Active | 1.47 inst/cycle | 1.48 inst/cycle | +0.01 | 1.59 inst/cycle | +0.12 | 1.68 inst/cycle | +0.21 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fuse_fp32_add_relu_into_single_expression_for_fma_emission | 0.992× | 0.994× | ✓ |  |
| C | fma_fusion_single_expression_and_wider_warp_configs | 0.997× | 0.997× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fuse_fp32_add_rel... | Δ | fma_fusion_single... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 60510 ns | 60860 ns | +350 ns (+0.6%) | 60700 ns | +190 ns (+0.3%) |
| SM Throughput (%) | 34.79 % | 37.70 % | +2.91 pp | 38.43 % | +3.64 pp |
| Memory Throughput (%) | 78.41 % | 77.82 % | -0.59 pp | 78.01 % | -0.40 pp |
| DRAM Throughput (%) | 78.41 % | 77.82 % | -0.59 pp | 78.01 % | -0.40 pp |
| Achieved Occupancy (%) | 85.34 % | 85.54 % | +0.20 pp | 85.43 % | +0.09 pp |
| Registers/Thread | 26.00 register/thread | 26.00 register/thread | +0 | 26.00 register/thread | +0 |
| Waves/SM | 14.85 | 14.85 | +0.00 | 14.85 | +0.00 |
| No Eligible (%) | 62.93 % | 59.06 % | -3.87 pp | 59.15 % | -3.78 pp |
| IPC Active | 1.47 inst/cycle | 1.63 inst/cycle | +0.16 | 1.61 inst/cycle | +0.14 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | add_triton_autotune | 1.002× | 0.385× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 60510 ns | 157150 ns | +96640 ns (+159.7%) |
| SM Throughput (%) | 34.79 % | 64.51 % | +29.72 pp |
| Memory Throughput (%) | 78.41 % | 30.49 % | -47.92 pp |
| DRAM Throughput (%) | 78.41 % | 30.49 % | -47.92 pp |
| Achieved Occupancy (%) | 85.34 % | 61.97 % | -23.37 pp |
| Registers/Thread | 26.00 register/thread | 16.00 register/thread | -10 |
| Waves/SM | 14.85 | 118.79 | +103.94 |
| No Eligible (%) | 62.93 % | 34.32 % | -28.61 pp |
| IPC Active | 1.47 inst/cycle | 2.63 inst/cycle | +1.16 |


### Key Learnings

#### ❌ Plan D: `add_triton_autotune` (0.385×)

Expanding the autotune config space to include smaller BLOCK_SIZE values (128, 256) caused the autotuner to select a very small tile size, exploding the grid from 31,360 to 250,880 blocks and increasing total executed instructions by ~4.5x (17.9M → 80.3M). Although per-warp stall cycles improved (34.35 → 14.8 cycles/issued instruction) and DRAM throughput dropped (78% → 30%), the massive increase in kernel launch overhead and total instruction count caused wall-clock duration to increase from 60.7 µs to 157.15 µs (2.6x slower). When a memory-bound kernel is already well-occupancy-covered (85% achieved occupancy, 14.85 waves/SM), adding small BLOCK_SIZE configs to autotune risks selecting a configuration that over-decomposes the work, inflating grid size and total instruction count far beyond the benefit of reduced per-warp stalls.

> **Applicable when**: Never apply: this is a negative result. Do not add very small BLOCK_SIZE configs (128, 256) to autotune when the kernel is already memory-bandwidth-bound with high occupancy (>80% achieved) and adequate waves-per-SM (>10). The autotuner may select the smallest config, causing grid explosion.

> **Avoid when**: Do not expand autotune config space to include small BLOCK_SIZE values when: (1) the kernel is already achieving high occupancy (>80%), (2) waves_per_SM is already >10, (3) the kernel is memory-bandwidth-bound (DRAM throughput >70%). In these cases, smaller tiles increase grid overhead and total instruction count without proportional bandwidth benefit.

---

## Cumulative Progress

```
Baseline:  60510 ns  (1.000×)

Final best: 157150 ns  (0.3850×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -3,12 +3,29 @@

 import triton.language as tl
 
 
+
 @triton.autotune(
     configs=[
-        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 128},  num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=1),
+        triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
+        triton.Config({'BLOCK_SIZE': 128},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
     ],
     key=['n_elements'],
 )
@@ -25,26 +42,27 @@

     pid = tl.program_id(0)
     offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
     mask = offsets < n_elements
-    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
+    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
     # Determine channel index for bias
     # For NCHW layout: channel = (offset / channel_stride) % n_channels
     chan = (offsets // channel_stride) % n_channels
     bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
-    result = x + bias
-    result = tl.maximum(result, 0.0)
-    tl.store(out_ptr + offsets, result, mask=mask)
+    result = tl.maximum(x + bias, 0.0)
+    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
 
 
 @triton.autotune(
     configs=[
         triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
         triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
-        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
-        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
+        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
+        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
+        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
     ],
-    key=['n_elements'],
+    key=['spatial_size'],
 )
 @triton.jit
+
 def bias_relu_nhwc_kernel(
     x_ptr,
     bias_ptr,
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `—`
