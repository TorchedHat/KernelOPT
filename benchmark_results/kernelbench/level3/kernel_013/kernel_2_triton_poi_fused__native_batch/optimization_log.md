# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3300 ns — **Best latency**: 3330 ns (**0.9910×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_013/kernel_2_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.01
- **Registers/thread**: 34
- **Duration**: 3300 ns
- **SM Throughput**: 0.23 % | **Memory**: 0.42 % | **DRAM**: 0.09 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 0.76 inst/cycle | **No Eligible**: 79.61 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_preserve_tile_sizes | 0.988× | 1.031× | ✓ |  |
| B | fix_launch_grid_correct_2d_decomposition | 0.999× | 1.041× | ✓ |  |
| C | fix_3d_grid_launch_correct_yoffset_formula | 0.994× | 1.051× | ✓ |  |
| D | fix_grid_launch_with_correct_2d_indexing_and_vectorized_loads | 1.039× | 1.031× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_p... | Δ | fix_launch_grid_c... | Δ | fix_3d_grid_launc... | Δ | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3200 ns | -100 ns (-3.0%) | 3170 ns | -130 ns (-3.9%) | 3140 ns | -160 ns (-4.8%) | 3200 ns | -100 ns (-3.0%) |
| SM Throughput (%) | 0.23 % | 0.23 % | +0.00 pp | 0.22 % | -0.01 pp | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.39 % | -0.03 pp | 0.44 % | +0.02 pp | 0.49 % | +0.07 pp | 0.38 % | -0.04 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp | 6.20 % | +0.01 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 34.00 register/thread | 34.00 register/thread | +0 | 34.00 register/thread | +0 | 34.00 register/thread | +0 | 34.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 79.61 % | 80.49 % | +0.88 pp | 79.58 % | -0.03 pp | 88.03 % | +8.42 pp | 79.79 % | +0.18 pp |
| IPC Active | 0.76 inst/cycle | 0.70 inst/cycle | -0.06 | 0.78 inst/cycle | +0.02 | 0.60 inst/cycle | -0.16 | 0.77 inst/cycle | +0.01 |


### Key Learnings

#### ✅ Plan B: `fix_launch_grid_correct_2d_decomposition` (1.041×)

When a Triton kernel uses tl.program_id(1) and tl.program_id(2) to tile a large Y-dimension but is launched with a 1D grid tuple, program_id(1) and program_id(2) are always 0, silently skipping nearly all Y-elements. The correct fix is to launch a 2D grid: (cdiv(xnumel, XBLOCK), cdiv(ynumel, YBLOCK)), keeping XBLOCK and YBLOCK unchanged from the kernel body. In this case the grid changed from (8,) to (1, 262144), reducing elapsed cycles from 4920 to 4706 and warp cycles per issued instruction from 6.48 to 5.13, yielding a 1.04x speedup on the profiled kernel — though the NCU LaunchConfiguration rule still fires because the workload is inherently tiny relative to the 132-SM GPU.

> **Applicable when**: Triton kernel body uses tl.program_id(1) and/or tl.program_id(2) to index a Y-dimension, but the launch grid is a 1D tuple (only program_id(0) is non-zero). NCU LaunchConfiguration rule fires with very few blocks (e.g. 8) vs. large SM count. Achieved occupancy is far below theoretical. The kernel silently produces wrong results for all but the first YBLOCK rows.

> **Avoid when**: The kernel body only uses tl.program_id(0) (truly 1D kernel). The grid is already multi-dimensional and correctly sized. Changing XBLOCK or YBLOCK values — only the grid tuple should change, not the tile sizes the kernel body was written for.

#### ✅ Plan C: `fix_3d_grid_launch_correct_yoffset_formula` (1.051×)

When a Triton kernel uses tl.program_id(1) and tl.program_id(2) to tile a large Y-dimension via yoffset = (pid1 + pid2 * num_programs(1)) * YBLOCK, launching a 1D grid causes pid1=pid2=0 always, silently processing only the first YBLOCK rows. Switching to a 3D grid (grid_x, grid_y, 1) with grid_z=1 ensures pid2=0 always, making yoffset = pid1 * YBLOCK — a linear scan covering all ynumel rows exactly once. The observed improvement was modest (1.051x, duration 3.3→3.14us) because the grid size remained 8 blocks in the after profile, suggesting the fix was only partially applied or the kernel was still not covering the full output; however, Warp Cycles Per Issued Instruction improved from 6.48 to 4.99 and L2 Hit Rate improved from 38.91% to 49.2%, indicating better cache reuse.

> **Applicable when**: Triton kernel uses tl.program_id(1) and tl.program_id(2) to tile a large Y-dimension with formula yoffset = (pid1 + pid2 * num_programs(1)) * YBLOCK, but the launch grid is 1D (only grid_x). NCU LaunchConfiguration rule fires with very few blocks (e.g. 8) vs. many SMs (e.g. 132), waves_per_sm ~0.01, and achieved occupancy far below theoretical (e.g. 6% vs 75%). Memory throughput is near 0% despite the kernel being memory-bound.

> **Avoid when**: The kernel body does not use tl.program_id(1) or tl.program_id(2) for Y-dimension tiling, or the grid is already multi-dimensional and correctly sized. Also avoid if changing XBLOCK/YBLOCK values without verifying the kernel body's index arithmetic — prior attempts with incorrect tile sizes caused max_diff > 4.0 correctness failures.

---

## Iteration 3

**Best so far**: 1.0312× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_3d_grid_launch_xblock64_covers_full_x_dimension | 1.042× | 0.991× | ✓ | **✓ best** |
| B | kernel_fusion_complete_single_pass_launch | 1.007× | 1.041× | ✓ |  |
| C | fix_grid_launch_2d_exact_xblock128_yblock8 | 1.000× | 1.022× | ✓ |  |
| D | fix_launch_grid_2d_exact_tile_sizes_no_body_change | 0.997× | 0.974× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ | kernel_fusion_com... | Δ | fix_grid_launch_2... | Δ | fix_launch_grid_2... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3330 ns | +30 ns (+0.9%) | 3170 ns | -130 ns (-3.9%) | 3230 ns | -70 ns (-2.1%) | 3390 ns | +90 ns (+2.7%) |
| SM Throughput (%) | 0.23 % | 0.22 % | -0.01 pp | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.40 % | -0.02 pp | 0.47 % | +0.05 pp | 0.39 % | -0.03 pp | 0.40 % | -0.02 pp |
| DRAM Throughput (%) | 0.09 % | 0.08 % | -0.01 pp | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp | 0.08 % | -0.01 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 34.00 register/thread | 34.00 register/thread | +0 | 34.00 register/thread | +0 | 34.00 register/thread | +0 | 34.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 79.61 % | 79.66 % | +0.05 pp | 79.71 % | +0.10 pp | 79.78 % | +0.17 pp | 79.69 % | +0.08 pp |
| IPC Active | 0.76 inst/cycle | 0.76 inst/cycle | +0.00 | 0.75 inst/cycle | -0.01 | 0.77 inst/cycle | +0.01 | 0.77 inst/cycle | +0.01 |

---

## Iteration 4

**Best so far**: 1.0312× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_2d_grid_launch_large_yblock_maximize_sm_utilization | 1.034× | 1.012× | ✓ |  |
| C | fix_2d_grid_launch_exact_tile_sizes_xblock128_yblock8 | 1.040× | 1.041× | ✓ |  |
| D | fix_3d_grid_launch_full_ynumel_coverage | 1.042× | 1.041× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_2d_grid_launc... | Δ | fix_2d_grid_launc... | Δ | fix_3d_grid_launc... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3260 ns | -40 ns (-1.2%) | 3170 ns | -130 ns (-3.9%) | 3170 ns | -130 ns (-3.9%) |
| SM Throughput (%) | 0.23 % | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp | 0.23 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.37 % | -0.05 pp | 0.45 % | +0.03 pp | 0.49 % | +0.07 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 6.20 % | +0.01 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 34.00 register/thread | 34.00 register/thread | +0 | 34.00 register/thread | +0 | 34.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 79.61 % | 80.00 % | +0.39 pp | 80.49 % | +0.88 pp | 80.01 % | +0.40 pp |
| IPC Active | 0.76 inst/cycle | 0.76 inst/cycle | +0.00 | 0.61 inst/cycle | -0.15 | 0.70 inst/cycle | -0.06 |

---

## Iteration 5

**Best so far**: 1.0312× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_3d_grid_launch_correct_grid_y_with_xblock128_yblock8 | 0.999× | 0.991× | ✓ |  |
| C | fix_3d_grid_launch_xblock8_yblock32_full_ynumel_coverage | 1.040× | 1.041× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ | fix_3d_grid_launc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3330 ns | +30 ns (+0.9%) | 3170 ns | -130 ns (-3.9%) |
| SM Throughput (%) | 0.23 % | 0.22 % | -0.01 pp | 0.23 % | +0.00 pp |
| Memory Throughput (%) | 0.42 % | 0.39 % | -0.03 pp | 0.39 % | -0.03 pp |
| DRAM Throughput (%) | 0.09 % | 0.09 % | +0.00 pp | 0.09 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.21 % | +0.02 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 34.00 register/thread | 34.00 register/thread | +0 | 34.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 79.61 % | 79.96 % | +0.35 pp | 80.37 % | +0.76 pp |
| IPC Active | 0.76 inst/cycle | 0.57 inst/cycle | -0.19 | 0.71 inst/cycle | -0.05 |

---

## Cumulative Progress

```
Baseline:  3300 ns  (1.000×)
Iter 2     3200 ns  (1.031×)  — fix_grid_launch_with_correct_2d_indexing_and_vectorized_loads

Final best: 3330 ns  (0.9910×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_convolution_relu_2(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -31,8 +32,6 @@

     tmp8 = (tmp6 / tmp7)
     tl.store(out_ptr0 + (y5 + 16384*x3 + 1048576*y2), tmp8, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
