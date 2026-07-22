# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2820 ns — **Best latency**: 2820 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_015/kernel_8_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (512,1,1), Waves/SM: 0.24
- **Registers/thread**: 28
- **Duration**: 2820 ns
- **SM Throughput**: 9.04 % | **Memory**: 8.54 % | **DRAM**: 1.12 %
- **Achieved Occupancy**: 19.87 %
- **IPC Active**: 0.98 inst/cycle | **No Eligible**: 70.43 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | autotune_tile_sizes_for_occupancy | — | fail | ✓ |  |
| B | autotune_tile_sizes_for_small_grid_utilization | 1.046× | 0.910× | ✓ |  |
| C | autotune_tile_sizes_for_underutilized_transpose_kernel | 0.986× | 0.928× | ✓ |  |


### Metric Comparison

| Metric | Baseline | autotune_tile_siz... | Δ | autotune_tile_siz... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2820 ns | 3100 ns | +280 ns (+9.9%) | 3040 ns | +220 ns (+7.8%) |
| SM Throughput (%) | 9.04 % | 7.01 % | -2.03 pp | 5.52 % | -3.52 pp |
| Memory Throughput (%) | 8.54 % | 3.47 % | -5.07 pp | 5.09 % | -3.45 pp |
| DRAM Throughput (%) | 1.12 % | 1.00 % | -0.12 pp | 1.02 % | -0.10 pp |
| Achieved Occupancy (%) | 19.87 % | 18.88 % | -0.99 pp | 8.88 % | -10.99 pp |
| Registers/Thread | 28.00 register/thread | 28.00 register/thread | +0 | 30.00 register/thread | +2 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.12 | -0.12 |
| No Eligible (%) | 70.43 % | 58.18 % | -12.25 pp | 77.58 % | +7.15 pp |
| IPC Active | 0.98 inst/cycle | 1.42 inst/cycle | +0.44 | 0.76 inst/cycle | -0.22 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | persistent_tile_scheduling | 0.030× | 0.010× | ✓ |  |
| B | fix_grid_launch_and_autotune_configs_for_underutilized_transpose | 0.941× | 1.000× | ✓ |  |
| C | persistent_tile_scheduling | 0.027× | 0.010× | ✓ |  |


### Metric Comparison

| Metric | Baseline | persistent_tile_s... | Δ | fix_grid_launch_a... | Δ | persistent_tile_s... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2820 ns | 270880 ns | +268060 ns (+9505.7%) | 2820 ns | +0 ns (+0.0%) | 277980 ns | +275160 ns (+9757.4%) |
| SM Throughput (%) | 9.04 % | 41.00 % | +31.96 pp | 21.01 % | +11.97 pp | 41.37 % | +32.33 pp |
| Memory Throughput (%) | 8.54 % | 47.97 % | +39.43 pp | 17.63 % | +9.09 pp | 54.53 % | +45.99 pp |
| DRAM Throughput (%) | 1.12 % | 0.01 % | -1.11 pp | 1.12 % | +0.00 pp | 0.01 % | -1.11 pp |
| Achieved Occupancy (%) | 19.87 % | 24.24 % | +4.37 pp | 20.84 % | +0.97 pp | 24.24 % | +4.37 pp |
| Registers/Thread | 28.00 register/thread | 32.00 register/thread | +4 | 32.00 register/thread | +4 | 32.00 register/thread | +4 |
| Waves/SM | 0.24 | 0.24 | +0.00 | 0.24 | +0.00 | 0.24 | +0.00 |
| No Eligible (%) | 70.43 % | 58.55 % | -11.88 pp | 44.19 % | -26.24 pp | 57.97 % | -12.46 pp |
| IPC Active | 0.98 inst/cycle | 1.60 inst/cycle | +0.62 | 2.15 inst/cycle | +1.17 | 1.67 inst/cycle | +0.69 |


### Key Learnings

#### ⚠️ Plan A: `persistent_tile_scheduling` (0.010×)

Applying a persistent grid-stride while-loop to a sub-wave kernel caused a 99x regression. The implementation kept the original grid (512 CTAs) unchanged instead of reducing it to NUM_SMS (132) CTAs, so each of the 512 CTAs still ran the while-loop, iterating over all tiles and executing ~767x more instructions (91M vs 119K). The strategy is sound in principle but requires the grid to be set to exactly NUM_SMS CTAs; if the grid is not reduced, the persistent loop multiplies work instead of redistributing it.

> **Applicable when**: Kernel has waves_per_sm < 0.5, achieved occupancy < 25%, grid_size < 2*sm_count, and duration is dominated by launch overhead / instruction-cache cold-start (CPIStall rule triggered, kernel duration < 10us).

> **Avoid when**: The grid is not simultaneously reduced to NUM_SMS CTAs — if the original grid is kept while adding the while-loop, every CTA will iterate over all tiles, causing O(grid_size/NUM_SMS) instruction blowup. Also avoid when the kernel already has high occupancy or is memory-bandwidth bound.

#### ⚠️ Plan C: `persistent_tile_scheduling` (0.010×)

Applying a persistent grid-stride while-loop to a sub-wave kernel caused a 99x regression. The implementation kept the original grid (512 CTAs) unchanged instead of reducing it to NUM_SMS (132) CTAs, so each of the 512 CTAs still ran the while-loop, iterating over all tiles and executing ~767x more instructions (91M vs 119K). The strategy is sound in principle but requires the grid to be set to exactly NUM_SMS CTAs; if the grid is not reduced, the persistent loop multiplies work instead of redistributing it.

> **Applicable when**: Kernel has waves_per_sm < 0.5, achieved occupancy < 25%, grid_size < 2*sm_count, and duration is dominated by launch overhead / instruction-cache cold-start (CPIStall rule triggered, kernel duration < 10us).

> **Avoid when**: The grid is not simultaneously reduced to NUM_SMS CTAs — if the original grid is kept while adding the while-loop, every CTA will iterate over all tiles, causing O(grid_size/NUM_SMS) instruction blowup. Also avoid when the kernel already has high occupancy or is memory-bandwidth bound.

---

## Cumulative Progress

```
Baseline:  2820 ns  (1.000×)

Final best: 2820 ns  (1.0000×)
```


## Best kernel vs baseline (diff)

```diff
--- baseline_kernel

+++ best_kernel

@@ -5,13 +5,24 @@

 from torch._dynamo.testing import rand_strided
 from torch._inductor.runtime import triton_helpers
 
+
+@triton.autotune(
+    configs=[
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=8),
+        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
+    ],
+    key=['ynumel', 'xnumel'],
+)
 @triton.jit
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
     ynumel = 4096
     xnumel = 9
     yoffset = tl.program_id(1) * YBLOCK
     yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
-    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
+    ymask = yindex < ynumel
     xoffset = tl.program_id(0) * XBLOCK
     xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
     xmask = xindex < xnumel
@@ -19,10 +30,8 @@

     y3 = yindex
     y0 = (yindex % 128)
     y1 = yindex // 128
-    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask, eviction_policy='evict_last')
-    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask)
-
-
+    tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
+    tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, xmask & ymask)
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
@@ -30,7 +39,7 @@

     ynumel = 4096
     xnumel = 9
     grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
-    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8[grid](arg_0, arg_1, 4096, 9, YBLOCK=8, XBLOCK=128)
+    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8[grid](arg_0, arg_1, 4096, 9)
     return arg_1
 
 def get_inputs():
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
