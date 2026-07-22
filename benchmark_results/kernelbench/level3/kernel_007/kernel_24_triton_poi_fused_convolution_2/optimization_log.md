# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_24` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2620 ns — **Best latency**: 2530 ns (**1.0356×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`autotune_tile_sizes_for_occupancy`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_24`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_007/kernel_24_triton_poi_fused_convolution_2/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (384,1,1), Waves/SM: 0.18
- **Registers/thread**: 28
- **Duration**: 2620 ns
- **SM Throughput**: 6.99 % | **Memory**: 7.64 % | **DRAM**: 2.45 %
- **Achieved Occupancy**: 16.46 %
- **IPC Active**: 0.87 inst/cycle | **No Eligible**: 76.50 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_occupancy | 1.021× | 1.036× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2620 ns | 2530 ns | -90 ns (-3.4%) |
| SM Throughput (%) | 6.99 % | 8.02 % | +1.03 pp |
| Memory Throughput (%) | 7.64 % | 7.64 % | +0.00 pp |
| DRAM Throughput (%) | 2.45 % | 2.54 % | +0.09 pp |
| Achieved Occupancy (%) | 16.46 % | 15.29 % | -1.17 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 |
| No Eligible (%) | 76.50 % | 74.21 % | -2.29 pp |
| IPC Active | 0.87 inst/cycle | 0.94 inst/cycle | +0.07 |

---

## Iteration 2

**Best so far**: 1.0356× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.029× | 0.010× | ✓ |  |
| C | fix_grid_to_use_autotuned_tile_sizes | 0.963× | 0.900× | ✓ |  |
| D | add_triton_autotune | 0.964× | 0.942× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_grid_to_use_a... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 262270 ns | +259650 ns (+9910.3%) | 2910 ns | +290 ns (+11.1%) | 2780 ns | +160 ns (+6.1%) |
| SM Throughput (%) | 6.99 % | 34.47 % | +27.48 pp | 6.40 % | -0.59 pp | 7.52 % | +0.53 pp |
| Memory Throughput (%) | 7.64 % | 37.64 % | +30.00 pp | 7.41 % | -0.23 pp | 8.18 % | +0.54 pp |
| DRAM Throughput (%) | 2.45 % | 0.03 % | -2.42 pp | 2.22 % | -0.23 pp | 2.30 % | -0.15 pp |
| Achieved Occupancy (%) | 16.46 % | 18.18 % | +1.72 pp | 7.14 % | -9.32 pp | 15.80 % | -0.66 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 28.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.18 | +0.00 | 0.09 | -0.09 | 0.18 | +0.00 |
| No Eligible (%) | 76.50 % | 65.32 % | -11.18 pp | 72.98 % | -3.52 pp | 77.32 % | +0.82 pp |
| IPC Active | 0.87 inst/cycle | 1.42 inst/cycle | +0.55 | 0.81 inst/cycle | -0.06 | 0.80 inst/cycle | -0.07 |


### Key Learnings

#### ❌ Plan A: `persistent_tile_scheduling` (0.010×)

Converting a simple tiled kernel to a persistent while-loop that iterates over tiles caused executed instructions to explode from 172,608 to 70,267,392 (407x increase), and kernel duration to increase from 2.53 µs to 262.27 µs (103x regression). The root cause is a critical implementation flaw: the kernel_function grid was NOT updated to use the persistent 1D grid — it still used the old 2D grid formula (cdiv(xnumel,128), cdiv(ynumel,8)) = (1, 384), so each of the 384 CTAs ran the full persistent while-loop over all 96 tiles instead of distributing tiles across CTAs. This caused every tile to be processed 384 times instead of once, multiplying work by the grid size. When applying persistent kernel patterns, the launch grid MUST be changed to the 1D persistent grid simultaneously with the kernel body change.

> **Applicable when**: Never apply this specific pattern (persistent loop with unchanged 2D grid). The persistent kernel approach itself is valid when grid is too small (waves_per_sm < 0.5, achieved_occupancy < 30%), but ONLY when the grid launch is simultaneously converted to 1D.

> **Avoid when**: Do not apply a persistent while-loop kernel body without simultaneously updating the launch grid to a 1D persistent grid. If the grid formula is left unchanged (e.g. still uses the old 2D cdiv formula), every CTA will redundantly process all tiles, causing instruction count to multiply by grid_size and runtime to increase proportionally. This is an implementation anti-pattern that always causes severe regression regardless of kernel size.

---

## Iteration 3

**Best so far**: 1.0356× | **Memory items**: 7

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_lambda_to_respect_autotuned_tile_sizes | 0.963× | 0.910× | ✓ |  |
| C | flatten_2d_tile_loop_to_1d_and_fix_grid_launch | 0.016× | 0.007× | ✓ |  |
| D | add_triton_autotune | 1.006× | 0.953× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_lambda_t... | Δ | flatten_2d_tile_l... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2620 ns | 2880 ns | +260 ns (+9.9%) | 360540 ns | +357920 ns (+13661.1%) | 2750 ns | +130 ns (+5.0%) |
| SM Throughput (%) | 6.99 % | 6.51 % | -0.48 pp | 23.11 % | +16.12 pp | 7.06 % | +0.07 pp |
| Memory Throughput (%) | 7.64 % | 7.54 % | -0.10 pp | 31.25 % | +23.61 pp | 7.66 % | +0.02 pp |
| DRAM Throughput (%) | 2.45 % | 2.24 % | -0.21 pp | 0.02 % | -2.43 pp | 2.30 % | -0.15 pp |
| Achieved Occupancy (%) | 16.46 % | 7.09 % | -9.37 pp | 17.97 % | +1.51 pp | 15.65 % | -0.81 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 18.00 register/thread | -10 | 28.00 register/thread | +0 |
| Waves/SM | 0.18 | 0.09 | -0.09 | 0.18 | +0.00 | 0.18 | +0.00 |
| No Eligible (%) | 76.50 % | 74.41 % | -2.09 pp | 76.08 % | -0.42 pp | 81.33 % | +4.83 pp |
| IPC Active | 0.87 inst/cycle | 0.78 inst/cycle | -0.09 | 0.94 inst/cycle | +0.07 | 0.76 inst/cycle | -0.11 |

---

## Cumulative Progress

```
Baseline:  2620 ns  (1.000×)
Iter 1     2530 ns  (1.036×)  — autotune_tile_sizes_for_occupancy

Final best: 2530 ns  (1.0356×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,23 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4),
+        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8),
+        triton.Config({'XBLOCK': 1024, 'YBLOCK': 8}, num_warps=8),
+    ],
+    key=['xnumel'],
+)
 @triton.jit
 def triton_poi_fused_convolution_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 3072
     xnumel = 25
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +29,8 @@

     y3 = yindex
     y0 = (yindex % 32)
     y1 = yindex // 32
-    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 25*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 32*x2 + 800*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +38,7 @@

     ynumel = 3072
     xnumel = 25
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused_convolution_24[grid](arg_0, arg_1, 3072, 25, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused_convolution_24[grid](arg_0, arg_1, 3072, 25)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
