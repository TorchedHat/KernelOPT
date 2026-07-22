# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_31` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2400 ns — **Best latency**: 2240 ns (**1.0714×** vs baseline)
- **Iterations completed**: 3
- **Outcome**: Best configuration **`fix_grid_to_1d_x_plus_1d_y_eliminating_program_id2`** beats baseline latency.

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_31`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_045/kernel_31_triton_poi_fused_convolution_3/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 30
- **Duration**: 2400 ns
- **SM Throughput**: 0.04 % | **Memory**: 0.43 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.51 inst/cycle | **No Eligible**: 86.31 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_launch_and_increase_tile_sizes | — | 1.071× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2400 ns | 2240 ns | -160 ns (-6.7%) |
| SM Throughput (%) | 0.04 % | 0.05 % | +0.01 pp |
| Memory Throughput (%) | 0.43 % | 0.43 % | +0.00 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.10 % | 6.09 % | -0.01 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.31 % | 90.97 % | +4.66 pp |
| IPC Active | 0.51 inst/cycle | 0.51 inst/cycle | +0.00 |

---

## Iteration 2

**Best so far**: 1.0714× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_3d_grid_launch_for_2d_y_tiling_kernel | — | 0.938× | ✓ |  |
| C | fix_2d_grid_launch_with_correct_yblock_tiling | — | 1.026× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ | fix_2d_grid_launc... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2400 ns | 2560 ns | +160 ns (+6.7%) | 2340 ns | -60 ns (-2.5%) |
| SM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.05 % | +0.01 pp |
| Memory Throughput (%) | 0.43 % | 0.38 % | -0.05 pp | 0.42 % | -0.01 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.10 % | 6.12 % | +0.02 pp | 6.10 % | +0.00 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.31 % | 86.28 % | -0.03 pp | 86.37 % | +0.06 pp |
| IPC Active | 0.51 inst/cycle | 0.43 inst/cycle | -0.08 | 0.50 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0714× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| D | fix_grid_to_1d_x_plus_1d_y_eliminating_program_id2 | — | 1.071× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_to_1d_x_... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2400 ns | 2240 ns | -160 ns (-6.7%) |
| SM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Memory Throughput (%) | 0.43 % | 0.50 % | +0.07 pp |
| DRAM Throughput (%) | 0.04 % | 0.05 % | +0.01 pp |
| Achieved Occupancy (%) | 6.10 % | 6.12 % | +0.02 pp |
| Registers/Thread | 30.00 register/thread | 28.00 register/thread | -2 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 86.31 % | 86.42 % | +0.11 pp |
| IPC Active | 0.51 inst/cycle | 0.41 inst/cycle | -0.10 |

---

## Cumulative Progress

```
Baseline:  2400 ns  (1.000×)
Iter 1     2240 ns  (1.071×)  — fix_grid_launch_and_increase_tile_sizes

Final best: 2240 ns  (1.0714×)
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
 def triton_poi_fused_convolution_31(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -21,8 +22,6 @@

     y1 = yindex // 512
     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 512*x2 + 4608*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, TopK=4, ExpN=8
- t_pos=1.04, t_neg=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
