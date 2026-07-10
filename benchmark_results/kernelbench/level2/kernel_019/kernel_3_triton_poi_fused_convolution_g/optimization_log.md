# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_gelu_native_group_norm_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3460 ns — **Best latency**: 3460 ns (**1.0000×** vs baseline)
- **Iterations completed**: 2
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_gelu_native_group_norm_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu3/kernel_019/kernel_3_triton_poi_fused_convolution_g/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.01
- **Registers/thread**: 40
- **Duration**: 3460 ns
- **SM Throughput**: 0.59 % | **Memory**: 0.49 % | **DRAM**: 0.08 %
- **Achieved Occupancy**: 6.21 %
- **IPC Active**: 0.97 inst/cycle | **No Eligible**: 75.37 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_launch_and_tile_sizes | 0.996× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3460 ns | 3490 ns | +30 ns (+0.9%) |
| SM Throughput (%) | 0.59 % | 0.57 % | -0.02 pp |
| Memory Throughput (%) | 0.49 % | 0.50 % | +0.01 pp |
| DRAM Throughput (%) | 0.08 % | 0.08 % | +0.00 pp |
| Achieved Occupancy (%) | 6.21 % | 6.21 % | +0.00 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 75.37 % | 75.15 % | -0.22 pp |
| IPC Active | 0.97 inst/cycle | 0.99 inst/cycle | +0.02 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 3

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | correct_3d_grid_launch_to_saturate_sms | 0.996× | 0.975× | ✓ |  |
| B | fix_grid_to_3d_with_fixed_xblock64 | 0.999× | 0.975× | ✓ |  |
| C | fix_2d_grid_launch_with_correct_index_decomposition | 0.999× | 0.925× | ✓ |  |
| D | correct_2d_grid_launch_with_vectorized_stores | 1.000× | 0.967× | ✓ |  |


### Metric Comparison

| Metric | Baseline | correct_3d_grid_l... | Δ | fix_grid_to_3d_wi... | Δ | fix_2d_grid_launc... | Δ | correct_2d_grid_l... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 3460 ns | 3550 ns | +90 ns (+2.6%) | 3550 ns | +90 ns (+2.6%) | 3740 ns | +280 ns (+8.1%) | 3580 ns | +120 ns (+3.5%) |
| SM Throughput (%) | 0.59 % | 0.60 % | +0.01 pp | 0.58 % | -0.01 pp | 0.58 % | -0.01 pp | 0.56 % | -0.03 pp |
| Memory Throughput (%) | 0.49 % | 0.45 % | -0.04 pp | 0.53 % | +0.04 pp | 0.40 % | -0.09 pp | 0.50 % | +0.01 pp |
| DRAM Throughput (%) | 0.08 % | 0.08 % | +0.00 pp | 0.08 % | +0.00 pp | 0.08 % | +0.00 pp | 0.08 % | +0.00 pp |
| Achieved Occupancy (%) | 6.21 % | 6.22 % | +0.01 pp | 6.21 % | +0.00 pp | 6.21 % | +0.00 pp | 6.21 % | +0.00 pp |
| Registers/Thread | 40.00 register/thread | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 | 40.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 75.37 % | 75.31 % | -0.06 pp | 78.14 % | +2.77 pp | 74.66 % | -0.71 pp | 74.33 % | -1.04 pp |
| IPC Active | 0.97 inst/cycle | 0.85 inst/cycle | -0.12 | 0.87 inst/cycle | -0.10 | 0.94 inst/cycle | -0.03 | 0.91 inst/cycle | -0.06 |

---

## Cumulative Progress

```
Baseline:  3460 ns  (1.000×)

Final best: 3460 ns  (1.0000×)
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
 def triton_poi_fused_convolution_gelu_native_group_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -45,8 +46,6 @@

     tmp23 = tmp21 + tmp22
     tl.store(out_ptr0 + (y0 + 66564*x2 + 4260096*y1), tmp23, xmask & ymask)
 
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
