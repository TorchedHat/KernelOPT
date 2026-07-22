# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 5180 ns — **Best latency**: 5310 ns (**0.9755×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_082/kernel_3_triton_poi_fused_add_convoluti/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.01
- **Registers/thread**: 54
- **Duration**: 5180 ns
- **SM Throughput**: 0.48 % | **Memory**: 0.36 % | **DRAM**: 0.18 %
- **Achieved Occupancy**: 6.23 %
- **IPC Active**: 1.02 inst/cycle | **No Eligible**: 70.28 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | increase_tile_sizes_and_grid_parallelism | 1.005× | 0.992× | ✓ | **✓ best** |
| D | fix_launch_grid_to_saturate_all_sms | 1.002× | 0.935× | ✓ |  |


### Metric Comparison

| Metric | Baseline | increase_tile_siz... | Δ | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 5220 ns | +40 ns (+0.8%) | 5540 ns | +360 ns (+6.9%) |
| SM Throughput (%) | 0.48 % | 0.46 % | -0.02 pp | 0.45 % | -0.03 pp |
| Memory Throughput (%) | 0.36 % | 0.35 % | -0.01 pp | 0.33 % | -0.03 pp |
| DRAM Throughput (%) | 0.18 % | 0.17 % | -0.01 pp | 0.16 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 | 54.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 70.28 % | 69.62 % | -0.66 pp | 71.02 % | +0.74 pp |
| IPC Active | 1.02 inst/cycle | 1.19 inst/cycle | +0.17 | 1.16 inst/cycle | +0.14 |

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_preserve_index_math | 1.007× | 1.012× | ✓ |  |
| B | fix_grid_launch_correct_2d_y_dimension | 1.024× | 0.976× | ✓ | **✓ best** |
| C | fix_launch_grid_correct_2d_y_dimension | 1.021× | 0.976× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_p... | Δ | fix_grid_launch_c... | Δ | fix_launch_grid_c... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 5120 ns | -60 ns (-1.2%) | 5310 ns | +130 ns (+2.5%) | 5310 ns | +130 ns (+2.5%) |
| SM Throughput (%) | 0.48 % | 0.48 % | +0.00 pp | 0.49 % | +0.01 pp | 0.48 % | +0.00 pp |
| Memory Throughput (%) | 0.36 % | 0.36 % | +0.00 pp | 0.42 % | +0.06 pp | 0.35 % | -0.01 pp |
| DRAM Throughput (%) | 0.18 % | 0.18 % | +0.00 pp | 0.17 % | -0.01 pp | 0.17 % | -0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 | 54.00 register/thread | +0 | 54.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 70.28 % | 70.69 % | +0.41 pp | 70.48 % | +0.20 pp | 73.52 % | +3.24 pp |
| IPC Active | 1.02 inst/cycle | 1.22 inst/cycle | +0.20 | 1.16 inst/cycle | +0.14 | 1.01 inst/cycle | -0.01 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_xblock64_single_x_tile_grid_y | 1.021× | 0.963× | ✓ |  |
| B | fix_grid_launch_xblock64_yblock_tiled_correct | 1.007× | 0.987× | ✓ |  |
| C | fix_grid_launch_flat_2d_correct_index_math | 1.022× | 0.915× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_x... | Δ | fix_grid_launch_x... | Δ | fix_grid_launch_f... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 5180 ns | 5380 ns | +200 ns (+3.9%) | 5250 ns | +70 ns (+1.4%) | 5660 ns | +480 ns (+9.3%) |
| SM Throughput (%) | 0.48 % | 0.46 % | -0.02 pp | 0.47 % | -0.01 pp | 0.48 % | +0.00 pp |
| Memory Throughput (%) | 0.36 % | 0.41 % | +0.05 pp | 0.38 % | +0.02 pp | 0.38 % | +0.02 pp |
| DRAM Throughput (%) | 0.18 % | 0.17 % | -0.01 pp | 0.17 % | -0.01 pp | 0.16 % | -0.02 pp |
| Achieved Occupancy (%) | 6.23 % | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp | 6.23 % | +0.00 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 | 54.00 register/thread | +0 | 54.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 | 0.01 | +0.00 | 0.01 | +0.00 |
| No Eligible (%) | 70.28 % | 70.29 % | +0.01 pp | 70.68 % | +0.40 pp | 70.52 % | +0.24 pp |
| IPC Active | 1.02 inst/cycle | 1.17 inst/cycle | +0.15 | 1.01 inst/cycle | -0.01 | 1.13 inst/cycle | +0.11 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 5

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| C | fix_grid_launch_xblock64_yblock512_2d_correct | 1.001× | 0.958× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_x... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 5180 ns | 5410 ns | +230 ns (+4.4%) |
| SM Throughput (%) | 0.48 % | 0.47 % | -0.01 pp |
| Memory Throughput (%) | 0.36 % | 0.34 % | -0.02 pp |
| DRAM Throughput (%) | 0.18 % | 0.17 % | -0.01 pp |
| Achieved Occupancy (%) | 6.23 % | 6.22 % | -0.01 pp |
| Registers/Thread | 54.00 register/thread | 54.00 register/thread | +0 |
| Waves/SM | 0.01 | 0.01 | +0.00 |
| No Eligible (%) | 70.28 % | 71.17 % | +0.89 pp |
| IPC Active | 1.02 inst/cycle | 1.19 inst/cycle | +0.17 |

---

## Cumulative Progress

```
Baseline:  5180 ns  (1.000×)

Final best: 5310 ns  (0.9755×)
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
 def triton_poi_fused_add_convolution_max_pool2d_with_indices_mul_tanh_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -53,8 +54,6 @@

     tmp30 = tl.maximum(tmp28, tmp29)
     tl.store(out_ptr0 + (y4 + 3969*x3 + 254016*y2), tmp30, xmask & ymask)
 
-
-
 def kernel_function(*inputs):
     arg_0 = inputs[0]
     arg_1 = inputs[1]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
