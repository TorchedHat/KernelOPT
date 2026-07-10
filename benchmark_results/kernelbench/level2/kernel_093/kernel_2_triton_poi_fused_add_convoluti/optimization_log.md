# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3330 ns — **Best latency**: 3330 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_093/kernel_2_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (16,1,1), Waves/SM: 0.01
- **Registers/thread**: 31
- **Duration**: 3330 ns
- **SM Throughput**: 0.89 % | **Memory**: 0.52 % | **DRAM**: 0.08 %
- **Achieved Occupancy**: 6.22 %
- **IPC Active**: 1.07 inst/cycle | **No Eligible**: 70.67 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_and_increase_tile_sizes_for_full_parallelism | 0.981× | 1.050× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_and_incr... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3330 ns | 3170 ns | -160 ns (-4.8%) |
| SM Throughput (%) | 0.89 % | 0.90 % | +0.01 pp |
| Memory Throughput (%) | 0.52 % | 0.63 % | +0.11 pp |
| DRAM Throughput (%) | 0.08 % | 0.09 % | +0.01 pp |
| Achieved Occupancy (%) | 6.22 % | 6.21 % | -0.01 pp |
| Registers/Thread | 31.00 register/thread | 31.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 70.67 % | 70.45 % | -0.22 pp |
| IPC Active | 1.07 inst/cycle | 1.13 inst/cycle | +0.06 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_3d_grid_launch_to_cover_full_y_dimension | 0.982× | 0.982× | ✓ |  |
| B | fix_grid_to_2d_preserve_xblock128_increase_yblock | 0.981× | 1.050× | ✓ |  |
| C | fix_1d_grid_to_2d_grid_preserving_kernel_body_exactly | 0.937× | 1.031× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ | fix_grid_to_2d_pr... | Δ | fix_1d_grid_to_2d... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3330 ns | 3390 ns | +60 ns (+1.8%) | 3170 ns | -160 ns (-4.8%) | 3230 ns | -100 ns (-3.0%) |
| SM Throughput (%) | 0.89 % | 0.92 % | +0.03 pp | 0.97 % | +0.08 pp | 0.96 % | +0.07 pp |
| Memory Throughput (%) | 0.52 % | 0.48 % | -0.04 pp | 0.47 % | -0.05 pp | 0.48 % | -0.04 pp |
| DRAM Throughput (%) | 0.08 % | 0.08 % | +0.00 pp | 0.09 % | +0.01 pp | 0.09 % | +0.01 pp |
| Achieved Occupancy (%) | 6.22 % | 6.22 % | +0.00 pp | 6.21 % | -0.01 pp | 6.22 % | +0.00 pp |
| Registers/Thread | 31.00 register/thread | 31.00 register/thread | +0 | 31.00 register/thread | +0 | 31.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 70.67 % | 70.08 % | -0.59 pp | 69.97 % | -0.70 pp | 70.61 % | -0.06 pp |
| IPC Active | 1.07 inst/cycle | 1.16 inst/cycle | +0.09 | 1.17 inst/cycle | +0.10 | 1.17 inst/cycle | +0.10 |

---

## Cumulative Progress

```
Baseline:  3330 ns  (1.000×)

Final best: 3330 ns  (1.0000×)
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
 def triton_poi_fused_add_convolution_gelu_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -37,8 +38,6 @@

     tmp15 = tmp13 * tmp14
     tl.store(out_ptr0 + (y0 + 16900*x2 + 2163200*y1), tmp15, xmask & ymask)
 
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
