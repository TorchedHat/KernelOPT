# Kernel Optimization Log

## Summary

- **Kernel**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14` on **H200 SXM5** (CC 9.0)
- **Baseline latency**: 2240 ns — **Best latency**: 2240 ns (**1.0000×** vs baseline)
- **Iterations completed**: 4
- **Outcome**: **No** candidate beat baseline latency (best equals or slower than starting point).

## Environment

- **Device**: H200 SXM5, 132 SMs, 139.8 GB HBM
- **Compute Capability**: 9.0
- **Peak HBM Bandwidth**: 4800 GB/s
- **CUDA**: 13.0 | **Torch**: 2.11.0+cu130 | **Triton**: 3.6.0

## Baseline Kernel

- **Name**: `triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14`
- **File**: `/mnt/podman_storage/ahpoddar/home/kernelopt/kernel-profiling-tool/kb_benchmark_results/beam_L3_rerun/kernel_009/kernel_14_triton_poi_fused__native_batch/standalone.py`
- **Framework**: triton
- **Launch**: Block (128,1,1), Grid (2,1,1), Waves/SM: 0.00
- **Registers/thread**: 30
- **Duration**: 2240 ns
- **SM Throughput**: 0.04 % | **Memory**: 0.44 % | **DRAM**: 0.04 %
- **Achieved Occupancy**: 6.10 %
- **IPC Active**: 0.49 inst/cycle | **No Eligible**: 88.07 %

## Iteration 1

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|

---

## Iteration 2

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_correctness_with_verified_2d_grid | 1.026× | 1.000× | ✓ |  |
| B | fix_grid_launch_with_correct_xblock_and_2d_y_dimension | 0.984× | 1.000× | ✓ |  |
| D | fix_grid_launch_and_vectorize_with_larger_yblock | 1.028× | 1.000× | ✓ | **✓ best** |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_c... | Δ | fix_grid_launch_w... | Δ | fix_grid_launch_a... | Δ |
|--------|----------|----------|-----|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2240 ns | +0 ns (+0.0%) | 2240 ns | +0 ns (+0.0%) | 2240 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.04 % | 0.05 % | +0.01 pp | 0.05 % | +0.01 pp | 0.05 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.43 % | -0.01 pp | 0.46 % | +0.02 pp | 0.45 % | +0.01 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.05 % | +0.01 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.10 % | 6.14 % | +0.04 pp | 6.10 % | +0.00 pp | 6.12 % | +0.02 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.07 % | 86.56 % | -1.51 pp | 86.96 % | -1.11 pp | 86.68 % | -1.39 pp |
| IPC Active | 0.49 inst/cycle | 0.35 inst/cycle | -0.14 | 0.50 inst/cycle | +0.01 | 0.42 inst/cycle | -0.07 |

---

## Iteration 3

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| A | fix_grid_launch_with_correct_3d_grid_matching_kernel_pid_usage | 1.026× | 1.014× | ✓ |  |
| D | add_triton_autotune | 1.025× | 1.000× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_w... | Δ | add_triton_autotu... | Δ |
|--------|----------|----------|-----|----------|-----|
| Duration (ns) | 2240 ns | 2210 ns | -30 ns (-1.3%) | 2240 ns | +0 ns (+0.0%) |
| SM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.05 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.44 % | +0.00 pp | 0.57 % | +0.13 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.10 % | 6.10 % | +0.00 pp | 6.12 % | +0.02 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.07 % | 86.81 % | -1.26 pp | 88.59 % | +0.52 pp |
| IPC Active | 0.49 inst/cycle | 0.50 inst/cycle | +0.01 | 0.43 inst/cycle | -0.06 |

---

## Iteration 4

**Best so far**: 1.0× | **Memory items**: 8

| Plan | Direction | Wall | NCU | Correctness | Kept? |
|------|-----------|------|-----|-------------|-------|
| B | fix_grid_launch_with_correct_3d_grid_and_verified_pid_mapping | 0.984× | 0.896× | ✓ |  |


### Metric Comparison

| Metric | Baseline | fix_grid_launch_w... | Δ |
|--------|----------|----------|-----|
| Duration (ns) | 2240 ns | 2500 ns | +260 ns (+11.6%) |
| SM Throughput (%) | 0.04 % | 0.05 % | +0.01 pp |
| Memory Throughput (%) | 0.44 % | 0.39 % | -0.05 pp |
| DRAM Throughput (%) | 0.04 % | 0.04 % | +0.00 pp |
| Achieved Occupancy (%) | 6.10 % | 6.14 % | +0.04 pp |
| Registers/Thread | 30.00 register/thread | 30.00 register/thread | +0 |
| Waves/SM | 0.00 | 0.00 | +0.00 |
| No Eligible (%) | 88.07 % | 88.55 % | +0.48 pp |
| IPC Active | 0.49 inst/cycle | 0.37 inst/cycle | -0.12 |

---

## Cumulative Progress

```
Baseline:  2240 ns  (1.000×)

Final best: 2240 ns  (1.0000×)
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
 def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
@@ -21,8 +22,6 @@

     y1 = yindex // 256
     tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
     tl.store(out_ptr0 + (y0 + 256*x2 + 2304*y1), tmp0, xmask & ymask)
-
-
 
 def kernel_function(*inputs):
     arg_0 = inputs[0]
```

## Config

- T=5, N=4, K=4, keep_n=4, q_cap=8
- spe_gate=1.04, slo_gate=1.15
- replay_mode=kernel
- kernel_args: `--profile-only`
