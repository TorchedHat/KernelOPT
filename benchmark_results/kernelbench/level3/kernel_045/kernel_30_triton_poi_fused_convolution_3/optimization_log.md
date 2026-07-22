# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_30` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 19260 ns — **Best latency**: 19420 ns (**0.9918×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_30`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_30_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8192,1,1), Waves/SM: 3.88
- **Registers/thread**: 27
- **Duration**: 19260 ns
- **SM Throughput**: 17.88 % | **Memory**: 75.97 % | **DRAM**: 49.64 %
- **Achieved Occupancy**: 86.09 %
- **IPC Active**: 0.83 inst/cycle | **No Eligible**: 78.48 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | memory_coalescing_swap_tile_dimensions | — | 0.992× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | memory_coalescing... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 19260 ns | 19420 ns | +160 ns (+0.8%) |
| SM Throughput (%) | 17.88 % | 18.10 % | +0.22 pp |
| Memory Throughput (%) | 75.97 % | 75.20 % | -0.77 pp |
| DRAM Throughput (%) | 49.64 % | 49.39 % | -0.25 pp |
| Achieved Occupancy (%) | 86.09 % | 85.45 % | -0.64 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 |
| No Eligible (%) | 78.48 % | 78.50 % | +0.02 pp |
| IPC Active | 0.83 inst/cycle | 0.83 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_static_kernel | — | 0.988× | ✓ |  |
| B | shared_memory_tiled_transpose_for_coalesced_stores | — | 0.993× | ✓ |  |
| C | evict_first_streaming_store_with_fixed_tile_rebalance | — | 0.996× | ✓ |  |
| D | increase_yblock_eliminate_partial_wave_no_autotune | — | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | shared_memory_til... | Δ | evict_first_strea... | Δ | increase_yblock_e... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19490 ns | +230 ns (+1.2%) | 19390 ns | +130 ns (+0.7%) | 19330 ns | +70 ns (+0.4%) | 19490 ns | +230 ns (+1.2%) |
| SM Throughput (%) | 17.88 % | 18.24 % | +0.36 pp | 17.10 % | -0.78 pp | 18.68 % | +0.80 pp | 17.97 % | +0.09 pp |
| Memory Throughput (%) | 75.97 % | 75.36 % | -0.61 pp | 76.09 % | +0.12 pp | 76.39 % | +0.42 pp | 75.28 % | -0.69 pp |
| DRAM Throughput (%) | 49.64 % | 49.09 % | -0.55 pp | 49.22 % | -0.42 pp | 48.00 % | -1.64 pp | 49.11 % | -0.53 pp |
| Achieved Occupancy (%) | 86.09 % | 85.51 % | -0.58 pp | 85.22 % | -0.87 pp | 85.44 % | -0.65 pp | 85.47 % | -0.62 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 28.00 register/thread | +1 | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 78.48 % | 78.34 % | -0.14 pp | 80.05 % | +1.57 pp | 77.84 % | -0.64 pp | 78.53 % | +0.05 pp |
| IPC Active | 0.83 inst/cycle | 0.86 inst/cycle | +0.03 | 0.79 inst/cycle | -0.04 | 0.87 inst/cycle | +0.04 | 0.85 inst/cycle | +0.02 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | vectorized_loads_with_contiguity_hints_static_kernel_no_autotune | — | 0.998× | ✓ |  |
| B | vectorized_loads_static_contiguity_hints_no_autotune | — | 0.976× | ✓ |  |
| C | increase_num_warps_and_vectorize_store_hints_for_l1tex_stall_hiding | — | 0.987× | ✓ |  |
| D | vectorized_loads_with_contiguity_hints_static_kernel_no_autotune | — | 0.988× | ✓ |  |


### Metric Comparison

| Metric | Baseline | vectorized_loads_... | Δ | vectorized_loads_... | Δ | increase_num_warp... | Δ | vectorized_loads_... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 19260 ns | 19300 ns | +40 ns (+0.2%) | 19740 ns | +480 ns (+2.5%) | 19520 ns | +260 ns (+1.3%) | 19490 ns | +230 ns (+1.2%) |
| SM Throughput (%) | 17.88 % | 18.87 % | +0.99 pp | 16.49 % | -1.39 pp | 18.59 % | +0.71 pp | 18.18 % | +0.30 pp |
| Memory Throughput (%) | 75.97 % | 76.47 % | +0.50 pp | 74.72 % | -1.25 pp | 75.19 % | -0.78 pp | 75.32 % | -0.65 pp |
| DRAM Throughput (%) | 49.64 % | 47.93 % | -1.71 pp | 48.38 % | -1.26 pp | 48.91 % | -0.73 pp | 48.65 % | -0.99 pp |
| Achieved Occupancy (%) | 86.09 % | 85.65 % | -0.44 pp | 85.42 % | -0.67 pp | 85.38 % | -0.71 pp | 85.51 % | -0.58 pp |
| Registers/Thread | 27.00 register/thread | 27.00 register/thread | +0 | 28.00 register/thread | +1 | 27.00 register/thread | +0 | 27.00 register/thread | +0 |
| Waves/SM | 3.88 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 | 3.88 | +0.00 |
| No Eligible (%) | 78.48 % | 77.89 % | -0.59 pp | 79.81 % | +1.33 pp | 78.59 % | +0.11 pp | 78.70 % | +0.22 pp |
| IPC Active | 0.83 inst/cycle | 0.89 inst/cycle | +0.06 | 0.80 inst/cycle | -0.03 | 0.86 inst/cycle | +0.03 | 0.85 inst/cycle | +0.02 |

---

## Cumulative Progress

```
Baseline:  19260 ns  (1.000×)

Final best: 19420 ns  (0.9918×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -4,6 +4,7 @@

 from triton.language.extra import libdevice
 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
+
 
 @triton.jit
 def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,8 @@

     y3 = yindex
     y0 = (yindex % 512)
     y1 = yindex // 512
-    tmp0 = tl.load(in_ptr0 + (x2 + 2048*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 512*x2 + 1048576*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 2048*y3), mask=xmask & ymask, eviction_policy='evict_first')
+    tl.store(out_ptr0 + (y0 + 512*x2 + 1048576*y1), tmp0, mask=xmask & ymask, eviction_policy='evict_first')
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
