# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused_convolution_gelu_leaky_relu_mul_2` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 3300 ns — **Best latency**: 3300 ns (**1.0000×** vs baseline)
- **Iterations completed**: 5
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused_convolution_gelu_leaky_relu_mul_2`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L2_gpu4/kernel_054/kernel_2_triton_poi_fused_convolution_g/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (8,1,1), Waves/SM: 0.00
- **Registers/thread**: 32
- **Duration**: 3300 ns
- **SM Throughput**: 0.50 % | **Memory**: 0.51 % | **DRAM**: 0.07 %
- **Achieved Occupancy**: 6.19 %
- **IPC Active**: 1.14 inst/cycle | **No Eligible**: 70.39 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_3d_grid_launch_to_cover_full_ynumel_with_correct_decomposition | 1.003× | 0.965× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_3d_grid_launc... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3300 ns | 3420 ns | +120 ns (+3.6%) |
| SM Throughput (%) | 0.50 % | 0.46 % | -0.04 pp |
| Memory Throughput (%) | 0.51 % | 0.40 % | -0.11 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.20 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 70.39 % | 73.39 % | +3.00 pp |
| IPC Active | 1.14 inst/cycle | 1.04 inst/cycle | -0.10 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_to_3d_matching_kernel_program_id_usage | 1.030× | 0.991× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_to_3d_ma... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3300 ns | 3330 ns | +30 ns (+0.9%) |
| SM Throughput (%) | 0.50 % | 0.45 % | -0.05 pp |
| Memory Throughput (%) | 0.51 % | 0.51 % | +0.00 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.20 % | +0.01 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 70.39 % | 72.98 % | +2.59 pp |
| IPC Active | 1.14 inst/cycle | 1.00 inst/cycle | -0.14 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_launch_grid_to_cover_ynumel_without_modifying_kernel_body | 1.001× | 0.938× | ✓ |  |
| C | fix_launch_grid_to_3d_matching_kernel_program_id_usage_no_kernel_body_change | 1.035× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_launch_grid_t... | Δ | fix_launch_grid_t... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 3300 ns | 3520 ns | +220 ns (+6.7%) | 3300 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.50 % | 0.45 % | -0.05 pp | 0.48 % | -0.02 pp |
| Memory Throughput (%) | 0.51 % | 0.37 % | -0.14 pp | 0.42 % | -0.09 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp | 6.19 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 70.39 % | 71.84 % | +1.45 pp | 72.65 % | +2.26 pp |
| IPC Active | 1.14 inst/cycle | 1.08 inst/cycle | -0.06 | 1.13 inst/cycle | -0.01 |

---

## Iteration 5

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| E | add_triton_autotune | 0.642× | 0.991× | ✓ |  |


### Metric Comparison

| Metric | Baseline | add_triton_autotu... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 3300 ns | 3330 ns | +30 ns (+0.9%) |
| SM Throughput (%) | 0.50 % | 0.48 % | -0.02 pp |
| Memory Throughput (%) | 0.51 % | 0.41 % | -0.10 pp |
| DRAM Throughput (%) | 0.07 % | 0.07 % | +0.00 pp |
| Achieved Occupancy (%) | 6.19 % | 6.19 % | +0.00 pp |
| Registers/Thread | 32.00 register/thread | 32.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 70.39 % | 74.08 % | +3.69 pp |
| IPC Active | 1.14 inst/cycle | 1.07 inst/cycle | -0.07 |

---

## Cumulative Progress

```
Baseline:  3300 ns  (1.000×)

Final best: 3300 ns  (1.0000×)
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
 def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -39,8 +40,6 @@

     tmp17 = tmp11 * tmp16
     tl.store(out_ptr0 + (y0 + 64516*x2 + 4129024*y1), tmp17, xmask & ymask)
 
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
