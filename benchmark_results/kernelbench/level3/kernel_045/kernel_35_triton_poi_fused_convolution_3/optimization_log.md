# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_35` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5250 ns — **Best latency**: 9330000 ns (**0.0006×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_35`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_35_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (4096,1,1), Waves/SM: 1.94
- **Registers/thread**: 28
- **Duration**: 5250 ns
- **SM Throughput**: 40.54 % | **Memory**: 39.73 % | **DRAM**: 4.60 %
- **Achieved Occupancy**: 69.70 %
- **IPC Active**: 2.77 inst/cycle | **No Eligible**: 31.44 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling_with_correct_index_arithmetic | — | 0.001× | ✓ | **✓ best** |
| B | fix_tile_sizes_reduce_partial_wave_masked_threads | — | 1.006× | ✓ |  |
| C | fix_xblock_yblock_tile_sizes_preserve_2d_indexing | — | 1.006× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_tile_sizes_re... | Δ | fix_xblock_yblock... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5250 ns | 9330000 ns | +9324750 ns (+177614.3%) | 5220 ns | -30 ns (-0.6%) | 5220 ns | -30 ns (-0.6%) |
| SM Throughput (%) | 40.54 % | 69.89 % | +29.35 pp | 40.15 % | -0.39 pp | 40.33 % | -0.21 pp |
| Memory Throughput (%) | 39.73 % | 88.31 % | +48.58 pp | 39.60 % | -0.13 pp | 39.79 % | +0.06 pp |
| DRAM Throughput (%) | 4.60 % | 0.01 % | -4.59 pp | 4.65 % | +0.05 pp | 4.65 % | +0.05 pp |
| Achieved Occupancy (%) | 69.70 % | 84.87 % | +15.17 pp | 70.32 % | +0.62 pp | 69.96 % | +0.26 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.44 % | 28.18 % | -3.26 pp | 29.64 % | -1.80 pp | 28.84 % | -2.60 pp |
| IPC Active | 2.77 inst/cycle | 2.87 inst/cycle | +0.10 | 2.78 inst/cycle | +0.01 | 2.76 inst/cycle | -0.01 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling_with_correct_index_arithmetic` (0.001×)

Converting a static 2D grid kernel to a persistent 1D grid-stride loop caused a ~1778x slowdown (5.25 µs → 9.33 ms). The executed instruction count exploded from 1.6M to 5.17B — a ~3157x increase — indicating the grid-stride loop iterated far more tiles than intended, likely because num_xtiles and num_ytiles were computed at runtime inside the kernel (using triton.cdiv which is a host-side function) rather than being passed as tl.constexpr, causing the loop bound to be incorrect or effectively unbounded. The persistent kernel pattern is structurally valid but requires that total_tiles be a compile-time constant or correctly computed host-side value; computing tile counts inside the kernel body using host-side utilities is an anti-pattern that produces wrong loop bounds in Triton.

> **Applicable when**: Never apply this specific pattern (computing triton.cdiv inside @triton.jit kernel body for loop bounds).

> **Avoid when**: Always avoid calling triton.cdiv() or any host-side utility inside a @triton.jit kernel body to compute loop bounds for grid-stride loops. The result is an incorrect or effectively unbounded loop that causes instruction count to explode by 3000x+. If implementing a persistent grid-stride loop in Triton, num_xtiles, num_ytiles, and total_tiles must be computed on the host and passed as tl.constexpr kernel arguments, not computed inside the kernel.

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | correct_tile_sizes_2d_grid_preserve_exact_indexing | — | 0.994× | ✓ |  |
| B | reduce_masked_thread_waste_via_static_tile_resize_xblock16_yblock64 | — | 1.000× | ✓ |  |
| C | revert_broken_persistent_loop_to_correct_static_2d_grid | — | 0.994× | ✓ |  |


### Metric Comparison

| Metric | Baseline | correct_tile_size... | Δ | reduce_masked_thr... | Δ | revert_broken_per... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5250 ns | 5280 ns | +30 ns (+0.6%) | 5250 ns | +0 ns (+0.0%) | 5280 ns | +30 ns (+0.6%) |
| SM Throughput (%) | 40.54 % | 38.96 % | -1.58 pp | 40.55 % | +0.01 pp | 40.41 % | -0.13 pp |
| Memory Throughput (%) | 39.73 % | 38.42 % | -1.31 pp | 40.01 % | +0.28 pp | 39.87 % | +0.14 pp |
| DRAM Throughput (%) | 4.60 % | 4.62 % | +0.02 pp | 4.63 % | +0.03 pp | 4.61 % | +0.01 pp |
| Achieved Occupancy (%) | 69.70 % | 64.45 % | -5.25 pp | 68.61 % | -1.09 pp | 71.71 % | +2.01 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.44 % | 31.48 % | +0.04 pp | 27.94 % | -3.50 pp | 37.51 % | +6.07 pp |
| IPC Active | 2.77 inst/cycle | 2.42 inst/cycle | -0.35 | 2.71 inst/cycle | -0.06 | 2.79 inst/cycle | +0.02 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 6

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | static_tile_resize_xblock16_yblock64_no_autotune | — | 1.006× | ✓ |  |
| B | fix_xblock_to_16_yblock_to_8_preserve_2d_grid_exact_indexing | — | 0.976× | ✓ |  |
| C | fix_launch_grid_only_preserve_exact_kernel_body | — | 0.960× | ✓ |  |


### Metric Comparison

| Metric | Baseline | static_tile_resiz... | Δ | fix_xblock_to_16_... | Δ | fix_launch_grid_o... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5250 ns | 5220 ns | -30 ns (-0.6%) | 5380 ns | +130 ns (+2.5%) | 5470 ns | +220 ns (+4.2%) |
| SM Throughput (%) | 40.54 % | 40.49 % | -0.05 pp | 40.15 % | -0.39 pp | 40.24 % | -0.30 pp |
| Memory Throughput (%) | 39.73 % | 39.94 % | +0.21 pp | 39.62 % | -0.11 pp | 39.70 % | -0.03 pp |
| DRAM Throughput (%) | 4.60 % | 4.62 % | +0.02 pp | 4.52 % | -0.08 pp | 4.44 % | -0.16 pp |
| Achieved Occupancy (%) | 69.70 % | 70.03 % | +0.33 pp | 71.81 % | +2.11 pp | 69.81 % | +0.11 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 28.00 register/thread | +0 | 28.00 register/thread | +0 |
| Waves/SM | 1.94 | 1.94 | +0.00 | 1.94 | +0.00 | 1.94 | +0.00 |
| No Eligible (%) | 31.44 % | 30.27 % | -1.17 pp | 29.95 % | -1.49 pp | 30.17 % | -1.27 pp |
| IPC Active | 2.77 inst/cycle | 2.76 inst/cycle | -0.01 | 2.63 inst/cycle | -0.14 | 2.75 inst/cycle | -0.02 |

---

## Cumulative Progress

```
Baseline:  5250 ns  (1.000×)

Final best: 9330000 ns  (0.0006×)
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
 def triton_poi_fused_convolution_35(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -19,10 +20,8 @@

     y3 = yindex
     y0 = (yindex % 256)
     y1 = yindex // 256
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), mask=xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, mask=xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
